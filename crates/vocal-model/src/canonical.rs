use crate::{
    ModelError, RVC_TENSOR_SCHEMA, RvcModelManifest, RvcSynthConfig, Tensor, TensorDType, schema,
};
use half::{bf16, f16};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};

/// A tensor after conversion into the VFRM RVC Synthesizer ABI.
/// Source names are retained in metadata so the importer remains auditable.
#[derive(Clone, Debug)]
pub(crate) struct CanonicalTensor {
    pub tensor: Tensor,
    pub source_names: Vec<String>,
    pub transform: &'static str,
}

pub(crate) fn finalize(
    manifest: &RvcModelManifest,
    mut metadata: Value,
    mut entries: Vec<CanonicalTensor>,
) -> Result<(Value, Vec<Tensor>), ModelError> {
    let object = metadata
        .as_object_mut()
        .ok_or_else(|| ModelError::InvalidCache("model metadata must be a JSON object".into()))?;
    object.insert("canonical_schema".into(), RVC_TENSOR_SCHEMA.into());
    object.insert(
        "canonical_layout".into(),
        "contiguous-row-major-little-endian".into(),
    );
    object.insert("tensor_name_namespace".into(), "rvc.synth.v1".into());
    let graph = object
        .get("graph_schema")
        .cloned()
        .unwrap_or_else(|| json!(graph_schema(manifest)));
    object.insert("graph_schema".into(), graph);

    let mut seen = HashMap::<String, usize>::new();
    for entry in &entries {
        entry.tensor.validate()?;
        if !is_canonical_name(&entry.tensor.name) {
            return Err(ModelError::UnsupportedModel(format!(
                "non-canonical RVC tensor name {}",
                entry.tensor.name
            )));
        }
        if let Some(previous) = seen.insert(entry.tensor.name.clone(), entry.source_names.len()) {
            return Err(ModelError::UnsupportedModel(format!(
                "duplicate canonical tensor {} (source counts {previous} and {})",
                entry.tensor.name,
                entry.source_names.len()
            )));
        }
    }
    entries.sort_by(|a, b| a.tensor.name.cmp(&b.tensor.name));
    let mappings = entries
        .iter()
        .map(|entry| {
            json!({
                "canonical": entry.tensor.name,
                "source": entry.source_names,
                "transform": entry.transform,
                "dtype": format!("{:?}", entry.tensor.dtype).to_ascii_lowercase(),
                "shape": entry.tensor.shape,
            })
        })
        .collect::<Vec<_>>();
    object.insert("canonical_tensor_count".into(), json!(entries.len()));
    object.insert("tensor_map".into(), Value::Array(mappings));
    let phone_graph = object
        .get("graph_schema")
        .and_then(Value::as_str)
        .is_some_and(|schema| schema.starts_with("rvc-phone"));
    let inputs = if phone_graph {
        json!(["phone", "phone_lengths", "pitch?", "pitchf?", "ds"])
    } else {
        json!(["feats", "p_len", "pitch?", "pitchf?", "sid"])
    };
    object.insert(
        "canonical_contract".into(),
        json!({
            "inputs": inputs,
            "output": "audio",
            "weights": "all RVC Synthesizer parameters; weight normalization is folded",
            "layouts": {
                "linear": "[in,out]",
                "conv1d": "[out,in,k]",
                "conv_transpose1d": "[in,out,k]",
                "embedding": "[items,channels]",
                "norm": "[channels]",
            },
        }),
    );

    let tensors = entries
        .into_iter()
        .map(|entry| entry.tensor)
        .collect::<Vec<_>>();
    let config: RvcSynthConfig =
        serde_json::from_value(metadata["rvc_config"].clone()).map_err(|e| {
            ModelError::UnsupportedModel(format!("missing or invalid canonical RVC config: {e}"))
        })?;
    schema::validate_tensors(&config, manifest, &tensors)?;
    Ok((metadata, tensors))
}

pub(crate) fn normalize_pth(
    manifest: &RvcModelManifest,
    mut metadata: Value,
    tensors: Vec<Tensor>,
) -> Result<(Value, Vec<Tensor>), ModelError> {
    let mut source = BTreeMap::<String, Tensor>::new();
    for tensor in tensors {
        tensor.validate()?;
        if source.insert(tensor.name.clone(), tensor).is_some() {
            return Err(ModelError::UnsupportedModel(
                "duplicate checkpoint tensor".into(),
            ));
        }
    }
    let mut entries = Vec::new();
    let names = source.keys().cloned().collect::<Vec<_>>();
    let mut consumed = std::collections::HashSet::new();
    for name in &names {
        if consumed.contains(name) {
            continue;
        }
        if let Some(base) = name.strip_suffix(".weight_g") {
            let v_name = format!("{base}.weight_v");
            let g = source.get(name).ok_or_else(|| missing(name))?;
            let v = source
                .get(&v_name)
                .ok_or_else(|| ModelError::UnsupportedModel(format!("{name} has no {v_name}")))?;
            let folded = fold_weight_norm(base, g, v)?;
            consumed.insert(name.clone());
            consumed.insert(v_name.clone());
            entries.push(CanonicalTensor {
                tensor: folded,
                source_names: vec![name.clone(), v_name],
                transform: "weight_norm(g,v)->weight",
            });
            continue;
        }
        if name.ends_with(".weight_v") {
            let g_name = format!("{}.weight_g", name.trim_end_matches(".weight_v"));
            if !source.contains_key(&g_name) {
                return Err(ModelError::UnsupportedModel(format!(
                    "{name} has no {g_name}"
                )));
            }
            continue;
        }
        if name == "enc_p.emb_phone.weight" || name == "dec.m_source.l_linear.weight" {
            let tensor = source.get(name).unwrap();
            if tensor.shape.len() != 2 {
                return Err(ModelError::UnsupportedModel(format!(
                    "invalid Linear checkpoint shape for {name}: {:?}",
                    tensor.shape
                )));
            }
            let (tensor, transform) = if name == "enc_p.emb_phone.weight"
                && tensor.shape == vec![manifest.feature_dims() as u64, 192]
            {
                (tensor.clone(), "identity")
            } else if name == "enc_p.emb_phone.weight"
                && tensor.shape != vec![192, manifest.feature_dims() as u64]
            {
                return Err(ModelError::UnsupportedModel(format!(
                    "invalid phone Linear shape for {name}: {:?}",
                    tensor.shape
                )));
            } else {
                (transpose_2d(tensor)?, "transpose2d")
            };
            consumed.insert(name.clone());
            entries.push(CanonicalTensor {
                tensor,
                source_names: vec![name.clone()],
                transform,
            });
            continue;
        }
        if !is_model_parameter(name) {
            return Err(ModelError::UnsupportedModel(format!(
                "unsupported RVC checkpoint tensor {name}"
            )));
        }
        consumed.insert(name.clone());
        entries.push(CanonicalTensor {
            tensor: source.get(name).unwrap().clone(),
            source_names: vec![name.clone()],
            transform: "identity",
        });
    }
    if consumed.len() != source.len() {
        return Err(ModelError::UnsupportedModel(
            "some RVC checkpoint tensors could not be canonicalized".into(),
        ));
    }
    let tensors = entries.iter().map(|e| e.tensor.clone()).collect::<Vec<_>>();
    metadata["rvc_config"] =
        serde_json::to_value(RvcSynthConfig::from_pth(manifest, &metadata, &tensors)?)?;
    finalize(manifest, metadata, entries)
}

pub(crate) fn fold_weight_norm(base: &str, g: &Tensor, v: &Tensor) -> Result<Tensor, ModelError> {
    g.validate()?;
    v.validate()?;
    if g.dtype != v.dtype || v.shape.is_empty() || v.shape[0] == 0 {
        return Err(ModelError::UnsupportedModel(format!(
            "invalid weight normalization pair for {base}"
        )));
    }
    if g.shape.len() != v.shape.len()
        || g.shape.first() != Some(&v.shape[0])
        || g.shape.iter().skip(1).any(|&d| d != 1)
    {
        return Err(ModelError::UnsupportedModel(format!(
            "weight normalization shape mismatch for {base}: g={:?}, v={:?}",
            g.shape, v.shape
        )));
    }
    let values = to_f32(v)?;
    let scales = to_f32(g)?;
    let stride = values.len() / v.shape[0] as usize;
    let mut out = vec![0.0; values.len()];
    for channel in 0..v.shape[0] as usize {
        let norm = values[channel * stride..(channel + 1) * stride]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(ModelError::UnsupportedModel(format!(
                "weight normalization has zero or non-finite norm for {base}, channel {channel}"
            )));
        }
        let scale = scales[channel * scales.len() / v.shape[0] as usize];
        for i in 0..stride {
            out[channel * stride + i] = values[channel * stride + i] * scale / norm;
        }
    }
    Ok(Tensor {
        name: format!("{base}.weight"),
        dtype: v.dtype,
        shape: v.shape.clone(),
        data: from_f32(v.dtype, &out)?,
    })
}

pub(crate) fn transpose_2d(tensor: &Tensor) -> Result<Tensor, ModelError> {
    tensor.validate()?;
    if tensor.shape.len() != 2 {
        return Err(ModelError::UnsupportedModel(
            "Linear weight must be rank 2".into(),
        ));
    }
    let rows = tensor.shape[0] as usize;
    let cols = tensor.shape[1] as usize;
    let size = tensor.dtype.element_size();
    let mut transposed = vec![0; tensor.data.len()];
    for row in 0..rows {
        for col in 0..cols {
            let src = (row * cols + col) * size;
            let dst = (col * rows + row) * size;
            transposed[dst..dst + size].copy_from_slice(&tensor.data[src..src + size]);
        }
    }
    Ok(Tensor {
        name: tensor.name.clone(),
        dtype: tensor.dtype,
        shape: vec![tensor.shape[1], tensor.shape[0]],
        data: transposed,
    })
}

pub(crate) fn to_f32(tensor: &Tensor) -> Result<Vec<f32>, ModelError> {
    let size = tensor.dtype.element_size();
    if !tensor.data.len().is_multiple_of(size) {
        return Err(ModelError::UnsupportedModel(format!(
            "tensor {} has invalid byte length",
            tensor.name
        )));
    }
    Ok(match tensor.dtype {
        TensorDType::F32 => tensor
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        TensorDType::F16 => tensor
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16::from_bits(u16::from_le_bytes(*b)).to_f32())
            .collect(),
        TensorDType::BF16 => tensor
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_bits(u16::from_le_bytes(*b)).to_f32())
            .collect(),
        _ => {
            return Err(ModelError::UnsupportedModel(format!(
                "tensor {} is not a floating point tensor",
                tensor.name
            )));
        }
    })
}

pub(crate) fn from_f32(dtype: TensorDType, values: &[f32]) -> Result<Vec<u8>, ModelError> {
    let mut data = Vec::with_capacity(values.len() * dtype.element_size());
    match dtype {
        TensorDType::F32 => values.iter().for_each(|v| data.extend(v.to_le_bytes())),
        TensorDType::F16 => values
            .iter()
            .for_each(|v| data.extend(f16::from_f32(*v).to_bits().to_le_bytes())),
        TensorDType::BF16 => values
            .iter()
            .for_each(|v| data.extend(bf16::from_f32(*v).to_bits().to_le_bytes())),
        _ => {
            return Err(ModelError::UnsupportedModel(
                "weight normalization requires a floating point dtype".into(),
            ));
        }
    }
    Ok(data)
}

fn missing(name: &str) -> ModelError {
    ModelError::UnsupportedModel(format!("missing tensor {name}"))
}

pub(crate) fn graph_schema(manifest: &RvcModelManifest) -> &'static str {
    match (manifest.feature_dims(), manifest.uses_f0()) {
        (768, true) => "rvc-v2-f0-feats",
        (768, false) => "rvc-v2-nof0-feats",
        (256, true) => "rvc-v1-f0-feats",
        _ => "rvc-v1-nof0-feats",
    }
}

pub(crate) fn is_model_parameter(name: &str) -> bool {
    (name.starts_with("enc_p.")
        || name.starts_with("dec.")
        || name.starts_with("flow.")
        || name == "emb_g.weight")
        && !name.ends_with(".weight_g")
        && !name.ends_with(".weight_v")
}

pub(crate) fn is_canonical_name(name: &str) -> bool {
    is_model_parameter(name) && !name.contains('/')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_tensor(name: &str, shape: Vec<u64>, values: &[f32]) -> Tensor {
        Tensor {
            name: name.into(),
            dtype: TensorDType::F32,
            shape,
            data: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }
    }

    #[test]
    fn folds_weight_norm_in_output_channel_order() {
        let g = f32_tensor("layer.weight_g", vec![2, 1, 1], &[2.0, 3.0]);
        let v = f32_tensor("layer.weight_v", vec![2, 2, 1], &[3.0, 4.0, 0.0, 5.0]);
        let folded = fold_weight_norm("layer", &g, &v).unwrap();
        let values = to_f32(&folded).unwrap();
        assert_eq!(values, vec![1.2, 1.6, 0.0, 3.0]);
        assert_eq!(folded.shape, vec![2, 2, 1]);
    }

    #[test]
    fn transposes_linear_without_changing_dtype() {
        let input = f32_tensor(
            "enc_p.emb_phone.weight",
            vec![2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let output = transpose_2d(&input).unwrap();
        assert_eq!(output.shape, vec![3, 2]);
        assert_eq!(to_f32(&output).unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
