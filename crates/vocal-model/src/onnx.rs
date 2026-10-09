use crate::{ModelError, RvcModelManifest, RvcSynthConfig, Tensor, TensorDType, canonical};
use prost::Message;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};

#[derive(Message)]
struct ModelProto {
    #[prost(int64, tag = "1")]
    ir_version: i64,
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    opsets: Vec<OperatorSetIdProto>,
    #[prost(message, repeated, tag = "14")]
    metadata: Vec<StringStringEntryProto>,
}
#[derive(Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    nodes: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    name: String,
    #[prost(message, repeated, tag = "5")]
    initializers: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    inputs: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    outputs: Vec<ValueInfoProto>,
}
#[derive(Message)]
struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    inputs: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    outputs: Vec<String>,
    #[prost(string, tag = "3")]
    name: String,
    #[prost(string, tag = "4")]
    op_type: String,
    #[prost(message, repeated, tag = "5")]
    attributes: Vec<AttributeProto>,
}
#[derive(Message)]
struct AttributeProto {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(float, tag = "2")]
    float: f32,
    #[prost(int64, tag = "3")]
    int: i64,
    #[prost(int64, repeated, packed = "true", tag = "8")]
    ints: Vec<i64>,
}
#[derive(Message)]
struct TensorProto {
    #[prost(int64, repeated, packed = "true", tag = "1")]
    dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    data_type: i32,
    #[prost(float, repeated, packed = "true", tag = "4")]
    float_data: Vec<f32>,
    #[prost(int32, repeated, packed = "true", tag = "5")]
    int32_data: Vec<i32>,
    #[prost(int64, repeated, packed = "true", tag = "7")]
    int64_data: Vec<i64>,
    #[prost(string, tag = "8")]
    name: String,
    #[prost(bytes, tag = "9")]
    raw_data: Vec<u8>,
    #[prost(double, repeated, packed = "true", tag = "10")]
    double_data: Vec<f64>,
    #[prost(int32, tag = "14")]
    data_location: i32,
    #[prost(message, repeated, tag = "13")]
    external_data: Vec<StringStringEntryProto>,
}
#[derive(Message)]
struct ValueInfoProto {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(message, optional, tag = "2")]
    value_type: Option<TypeProto>,
}
#[derive(Message)]
struct TypeProto {
    #[prost(message, optional, tag = "1")]
    tensor_type: Option<TensorTypeProto>,
}
#[derive(Message)]
struct TensorTypeProto {
    #[prost(int32, tag = "1")]
    elem_type: i32,
    #[prost(message, optional, tag = "2")]
    shape: Option<TensorShapeProto>,
}
#[derive(Message)]
struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    dims: Vec<DimensionProto>,
}
#[derive(Message)]
struct DimensionProto {
    #[prost(int64, optional, tag = "1")]
    value: Option<i64>,
    #[prost(string, optional, tag = "2")]
    param: Option<String>,
}
#[derive(Message)]
struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    domain: String,
    #[prost(int64, tag = "2")]
    version: i64,
}
#[derive(Message)]
struct StringStringEntryProto {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(string, tag = "2")]
    value: String,
}

pub(crate) fn load_rvc_onnx(
    path: &Path,
    manifest: &RvcModelManifest,
) -> Result<(Value, Vec<Tensor>), ModelError> {
    let model = ModelProto::decode(fs::read(path)?.as_slice())
        .map_err(|e| ModelError::UnsupportedModel(format!("invalid ONNX protobuf: {e}")))?;
    if model.opsets.is_empty()
        || model
            .opsets
            .iter()
            .any(|x| !x.domain.is_empty() || !(13..=18).contains(&x.version))
    {
        return Err(ModelError::UnsupportedModel(
            "ONNX opset is outside the supported RVC export range 13..18".into(),
        ));
    }
    let graph = model
        .graph
        .ok_or_else(|| ModelError::UnsupportedModel("ONNX graph is missing".into()))?;
    let init_names: HashSet<_> = graph.initializers.iter().map(|x| x.name.as_str()).collect();
    let produced: HashSet<_> = graph
        .nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(String::as_str))
        .collect();
    let inputs: Vec<_> = graph
        .inputs
        .iter()
        .filter(|x| !produced.contains(x.name.as_str()) && !init_names.contains(x.name.as_str()))
        .collect();
    let input_names: HashSet<_> = inputs.iter().map(|x| x.name.as_str()).collect();
    validate_operator_set(&graph.nodes)?;
    let mut tensors = Vec::with_capacity(graph.initializers.len());
    for initializer in &graph.initializers {
        tensors.push(decode_initializer(initializer)?);
    }
    let tensor_names: HashSet<_> = tensors.iter().map(|t| t.name.as_str()).collect();
    validate_signature(
        manifest,
        &model.metadata,
        &inputs,
        &graph.outputs,
        &tensor_names,
    )?;
    let metadata_props: HashMap<_, _> = model
        .metadata
        .into_iter()
        .map(|p| (p.key, p.value))
        .collect();
    let declared = metadata_props
        .get("metadata")
        .and_then(|v| serde_json::from_str::<Value>(v).ok())
        .unwrap_or_else(|| json!({}));
    let schema = if input_names.contains("feats") {
        canonical::graph_schema(manifest)
    } else if input_names.contains("phone") {
        if input_names.contains("pitch") {
            "rvc-phone-f0"
        } else {
            "rvc-phone-nof0"
        }
    } else {
        return Err(ModelError::UnsupportedModel(
            "ONNX inputs are not a supported RVC interface".into(),
        ));
    };
    let ops = graph
        .nodes
        .iter()
        .map(|n| n.op_type.clone())
        .collect::<Vec<_>>();
    let metadata = json!({
        "source_format": "rvc-onnx",
        "graph_schema": schema,
        "onnx": { "ir_version": model.ir_version, "opsets": model.opsets.iter().map(|x| json!({"domain": x.domain, "version": x.version})).collect::<Vec<_>>(), "inputs": inputs.iter().map(|x| value_info_json(x)).collect::<Vec<_>>(), "outputs": graph.outputs.iter().map(value_info_json).collect::<Vec<_>>(), "node_count": graph.nodes.len(), "op_types": ops },
        "export_metadata": declared,
    });
    let entries = canonicalize_initializers(&graph, tensors)?;
    let canonical_tensors = entries
        .iter()
        .map(|entry| entry.tensor.clone())
        .collect::<Vec<_>>();
    let config = infer_onnx_config(manifest, &graph, &canonical_tensors)?;
    let mut metadata = metadata;
    metadata["rvc_config"] = serde_json::to_value(config)?;
    canonical::finalize(manifest, metadata, entries)
}

fn infer_onnx_config(
    manifest: &RvcModelManifest,
    graph: &GraphProto,
    tensors: &[Tensor],
) -> Result<RvcSynthConfig, ModelError> {
    let attr = |node: &NodeProto, name: &str| -> Option<Vec<u64>> {
        node.attributes
            .iter()
            .find(|a| a.name == name)
            .and_then(|a| {
                if !a.ints.is_empty() {
                    Some(
                        a.ints
                            .iter()
                            .map(|x| u64::try_from(*x).ok())
                            .collect::<Option<Vec<_>>>()?,
                    )
                } else {
                    u64::try_from(a.int).ok().map(|x| vec![x])
                }
            })
    };
    let mut ups = vec![0; 8];
    let mut kernels = vec![0; 8];
    for node in graph
        .nodes
        .iter()
        .filter(|n| n.op_type == "ConvTranspose" && n.name.starts_with("/dec/ups."))
    {
        let index = node
            .name
            .trim_start_matches("/dec/ups.")
            .split('/')
            .next()
            .and_then(|x| x.parse::<usize>().ok())
            .ok_or_else(|| {
                ModelError::UnsupportedModel(format!("bad RVC upsample node {}", node.name))
            })?;
        ups[index] = attr(node, "strides")
            .and_then(|x| x.first().copied())
            .unwrap_or(0);
        kernels[index] = attr(node, "kernel_shape")
            .and_then(|x| x.first().copied())
            .unwrap_or(0);
    }
    while ups.last() == Some(&0) {
        ups.pop();
        kernels.pop();
    }
    if ups.is_empty() || ups.contains(&0) {
        return Err(ModelError::UnsupportedModel(format!(
            "RVC ONNX has incomplete upsample attributes: {:?}",
            ups
        )));
    }
    let mut block_kernels = vec![0; 64];
    let mut dilations = vec![Vec::new(); 64];
    let prefix = if graph
        .nodes
        .iter()
        .any(|n| n.name.contains("/resblocks.0/convs1."))
    {
        "convs1"
    } else {
        "convs"
    };
    for node in graph
        .nodes
        .iter()
        .filter(|n| n.op_type == "Conv" && n.name.contains("/resblocks."))
    {
        let mut parts = node.name.trim_start_matches("/dec/resblocks.").split('/');
        let block = parts.next().and_then(|x| x.parse::<usize>().ok());
        let list = parts.next().unwrap_or_default();
        let index = list
            .rsplit_once('.')
            .and_then(|x| x.1.parse::<usize>().ok());
        if let (Some(block), Some(index)) = (block, index) {
            let kernel = attr(node, "kernel_shape")
                .and_then(|x| x.first().copied())
                .unwrap_or(0);
            if list.starts_with(prefix) && block < block_kernels.len() {
                block_kernels[block] = kernel;
                let dilation = attr(node, "dilations")
                    .and_then(|x| x.first().copied())
                    .unwrap_or(0);
                if dilations[block].len() <= index {
                    dilations[block].resize(index + 1, 0);
                }
                dilations[block][index] = dilation;
            }
        }
    }
    let mut dilations3 = Vec::new();
    for group in 0..3 {
        let block = group;
        let k = block_kernels.get(block).copied().unwrap_or(0);
        let d = dilations.get(block).cloned().unwrap_or_default();
        if k == 0 || d.contains(&0) {
            return Err(ModelError::UnsupportedModel(
                "RVC ONNX resblock attributes are incomplete".into(),
            ));
        }
        dilations3.push(d);
    }
    let resblock_type = if prefix == "convs1" { 1 } else { 2 };
    RvcSynthConfig::from_onnx_parts(manifest, tensors, ups, kernels, dilations3, resblock_type)
}

fn rvc_gemm_head(node: &NodeProto) -> Result<&str, ModelError> {
    let bias = node.inputs.get(2).map(String::as_str).unwrap_or_default();
    let head = bias
        .strip_suffix(".bias")
        .filter(|head| matches!(*head, "enc_p.emb_phone" | "dec.m_source.l_linear"))
        .ok_or_else(|| {
            ModelError::UnsupportedModel("Gemm is not a standard RVC phone/NSF linear".into())
        })?;
    let mut names = HashSet::new();
    for attribute in &node.attributes {
        let supported = match attribute.name.as_str() {
            "alpha" | "beta" => attribute.float == 1.,
            "transA" => attribute.int == 0,
            "transB" => matches!(attribute.int, 0 | 1),
            _ => false,
        };
        if !supported || !names.insert(attribute.name.as_str()) {
            return Err(ModelError::UnsupportedModel(format!(
                "unsupported RVC Gemm attribute {}",
                attribute.name
            )));
        }
    }
    Ok(head)
}

fn canonicalize_initializers(
    graph: &GraphProto,
    tensors: Vec<Tensor>,
) -> Result<Vec<canonical::CanonicalTensor>, ModelError> {
    let mut entries = Vec::new();
    for mut tensor in tensors {
        let source_name = tensor.name.clone();
        let mut canonical_name = None;
        let mut transform = "identity";
        // Simplifiers can fuse linear MatMul+Add into anonymous Gemm nodes.
        // The learned bias retains the semantic name and determines the head.
        if let Some(node) = graph
            .nodes
            .iter()
            .find(|n| n.op_type == "Gemm" && n.inputs.get(1) == Some(&source_name))
        {
            let head = rvc_gemm_head(node)?;
            if node
                .attributes
                .iter()
                .any(|a| a.name == "transB" && a.int == 1)
            {
                tensor = canonical::transpose_2d(&tensor)?;
            }
            canonical_name = Some(format!("{head}.weight"));
            transform = "onnx-gemm-weight->in-out";
        }
        if canonical_name.is_some() {
            // Already mapped above using the source-independent bias name.
        } else if canonical::is_model_parameter(&tensor.name) {
            canonical_name = Some(tensor.name.clone());
        } else if !tensor.name.starts_with('/') {
            if !is_graph_constant_name(&tensor.name) {
                return Err(ModelError::UnsupportedModel(format!(
                    "unknown named ONNX initializer {}",
                    tensor.name
                )));
            }
        } else if tensor.name.starts_with('/') {
            for node in &graph.nodes {
                let Some(input_index) = node.inputs.iter().position(|input| input == &tensor.name)
                else {
                    continue;
                };
                if input_index == 1 && matches!(node.op_type.as_str(), "Conv" | "ConvTranspose") {
                    let path = node
                        .name
                        .trim_start_matches('/')
                        .rsplit_once('/')
                        .map(|x| x.0);
                    if let Some(path) = path {
                        let candidate = path.replace('/', ".") + ".weight";
                        if candidate.starts_with("enc_p.")
                            || candidate.starts_with("dec.")
                            || candidate.starts_with("flow.")
                        {
                            canonical_name = Some(candidate);
                            transform = "onnx-folded-conv-weight";
                            break;
                        }
                        return Err(ModelError::UnsupportedModel(format!(
                            "cannot map folded ONNX convolution weight {} from node {}",
                            tensor.name, node.name
                        )));
                    }
                }
                if input_index == 1 && node.op_type == "MatMul" {
                    if node.name.starts_with("/enc_p/emb_phone/") {
                        canonical_name = Some("enc_p.emb_phone.weight".into());
                        transform = "onnx-linear-matmul-weight";
                        break;
                    }
                    if node.name.starts_with("/dec/m_source/l_linear/") {
                        canonical_name = Some("dec.m_source.l_linear.weight".into());
                        transform = "onnx-linear-matmul-weight";
                        break;
                    }
                }
            }
        }
        let Some(canonical_name) = canonical_name else {
            // Scalar shape constants and shape/index helpers are part of the ONNX
            // graph, not model parameters. They are intentionally reconstructed
            // by the fixed RVC graph and are not stored in VFRM.
            continue;
        };
        let mut normalized = tensor;
        normalized.name = canonical_name.clone();
        entries.push(canonical::CanonicalTensor {
            tensor: normalized,
            source_names: vec![source_name],
            transform,
        });
    }
    Ok(entries)
}

fn is_graph_constant_name(name: &str) -> bool {
    name.starts_with("onnx::")
        || name.strip_prefix("_v_").is_some_and(|suffix| {
            !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
        })
        || (!name.is_empty() && name.bytes().all(|byte| byte.is_ascii_digit()))
}

fn validate_operator_set(nodes: &[NodeProto]) -> Result<(), ModelError> {
    const ALLOWED: &[&str] = &[
        "Add",
        "Cast",
        "Clip",
        "Concat",
        "ConstantOfShape",
        "Conv",
        "ConvTranspose",
        "CumSum",
        "Div",
        "Equal",
        "Exp",
        "Expand",
        "Floor",
        "Gather",
        "Greater",
        "LayerNormalization",
        "LeakyRelu",
        "Less",
        "MatMul",
        "Gemm",
        "Mul",
        "Neg",
        "Pad",
        "Pow",
        "RandomNormalLike",
        "RandomUniformLike",
        "Range",
        "Relu",
        "Reshape",
        "Resize",
        "ScatterND",
        "Shape",
        "Sigmoid",
        "Sin",
        "Slice",
        "Softmax",
        "Split",
        "Sub",
        "Tanh",
        "Transpose",
        "Unsqueeze",
        "Where",
    ];
    for node in nodes {
        if !ALLOWED.contains(&node.op_type.as_str()) {
            return Err(ModelError::UnsupportedModel(format!(
                "ONNX operator {} is outside the RVC Synthesizer graph",
                node.op_type
            )));
        }
        if node.op_type == "Gemm" {
            rvc_gemm_head(node)?;
        }
    }
    for required in ["Conv", "ConvTranspose", "MatMul", "Tanh"] {
        if !nodes.iter().any(|node| node.op_type == required) {
            return Err(ModelError::UnsupportedModel(format!(
                "RVC ONNX graph is missing required operator {required}"
            )));
        }
    }
    Ok(())
}

fn validate_signature(
    manifest: &RvcModelManifest,
    props: &[StringStringEntryProto],
    inputs: &[&ValueInfoProto],
    outputs: &[ValueInfoProto],
    tensor_names: &HashSet<&str>,
) -> Result<(), ModelError> {
    let metadata = props
        .iter()
        .find(|x| x.key == "metadata")
        .and_then(|x| serde_json::from_str::<Value>(&x.value).ok());
    if let Some(meta) = metadata {
        let version = manifest.rvc_version()?;
        if let Some(layer) = meta.get("embOutputLayer") {
            if layer.as_u64() != Some(version.hubert_output_layer() as u64) {
                return Err(ModelError::UnsupportedModel(format!(
                    "ONNX embOutputLayer must be {} for standard RVC {}",
                    version.hubert_output_layer(),
                    version.as_str()
                )));
            }
        }
        if let Some(projected) = meta.get("useFinalProj") {
            if projected.as_bool() != Some(version == crate::RvcVersion::V1) {
                return Err(ModelError::UnsupportedModel(
                    "ONNX useFinalProj disagrees with RVC version".into(),
                ));
            }
        }
        if let Some(channels) = meta.get("embChannels").and_then(Value::as_u64)
            && channels != manifest.feature_dims() as u64
        {
            return Err(ModelError::UnsupportedModel(format!(
                "ONNX embChannels {channels} does not match RVC manifest {}",
                manifest.feature_dims()
            )));
        }
        if let Some(f0) = meta.get("f0").and_then(Value::as_bool) {
            let has_pitch = inputs.iter().any(|x| x.name == "pitch");
            if f0 != has_pitch {
                return Err(ModelError::UnsupportedModel(
                    "ONNX f0 metadata disagrees with inputs".into(),
                ));
            }
        }
        if let Some(sr) = meta.get("samplingRate").and_then(Value::as_u64)
            && sr != manifest.sample_rate as u64
        {
            return Err(ModelError::UnsupportedModel(format!(
                "ONNX sample rate {sr} does not match manifest {}",
                manifest.sample_rate
            )));
        }
    }
    let names: HashSet<_> = inputs.iter().map(|x| x.name.as_str()).collect();
    let feats = names.contains("feats");
    let phone = names.contains("phone");
    if feats == phone {
        return Err(ModelError::UnsupportedModel(
            "ONNX must use exactly one of feats or phone".into(),
        ));
    }
    let prefix = if feats { "feats" } else { "phone" };
    let length = if feats { "p_len" } else { "phone_lengths" };
    let speaker = if feats { "sid" } else { "ds" };
    require_input(inputs, prefix, 3, &[1, 10])?;
    require_input(inputs, length, 1, &[7])?;
    require_input(inputs, speaker, 1, &[7])?;
    let feature = inputs
        .iter()
        .find(|x| x.name == prefix)
        .ok_or_else(|| ModelError::UnsupportedModel(format!("ONNX input {prefix} is missing")))?;
    let shape = shape_of(feature)?;
    if shape.len() != 3
        || shape[0] != Some(1)
        || !matches!(shape[2], Some(256 | 768))
        || shape[2] != Some(manifest.feature_dims() as i64)
    {
        return Err(ModelError::UnsupportedModel(format!(
            "ONNX {prefix} shape is not [1, dynamic, {}]",
            manifest.feature_dims()
        )));
    }
    if names.contains("pitch") {
        require_input(inputs, "pitch", 2, &[7])?;
        require_input(inputs, "pitchf", 2, &[1, 10])?;
        let p =
            shape_of(inputs.iter().find(|x| x.name == "pitch").ok_or_else(|| {
                ModelError::UnsupportedModel("pitch missing with pitchf".into())
            })?)?;
        let pf =
            shape_of(inputs.iter().find(|x| x.name == "pitchf").ok_or_else(|| {
                ModelError::UnsupportedModel("pitchf missing with pitch".into())
            })?)?;
        if p.len() != 2 || pf.len() != 2 || p[0] != Some(1) || pf[0] != Some(1) {
            return Err(ModelError::UnsupportedModel(
                "RVC pitch inputs must be [1, dynamic]".into(),
            ));
        }
    }
    if names.contains("rnd") {
        require_input(inputs, "rnd", 3, &[1, 10])?;
        let rnd = shape_of(inputs.iter().find(|x| x.name == "rnd").ok_or_else(|| {
            ModelError::UnsupportedModel("rnd input disappeared during validation".into())
        })?)?;
        if rnd.len() != 3 || rnd[0] != Some(1) || rnd[1] != Some(192) {
            return Err(ModelError::UnsupportedModel(
                "RVC rnd input must be [1, 192, dynamic]".into(),
            ));
        }
    }
    let audio = outputs
        .iter()
        .find(|x| x.name == "audio")
        .ok_or_else(|| ModelError::UnsupportedModel("ONNX output audio is missing".into()))?;
    if ![1, 10].contains(&element_type(audio)?) {
        return Err(ModelError::UnsupportedModel(
            "ONNX audio output must be float or float16".into(),
        ));
    }
    let audio_shape = shape_of(audio)?;
    if !matches!(audio_shape.len(), 1 | 3) || (audio_shape.len() == 3 && audio_shape[1] != Some(1))
    {
        return Err(ModelError::UnsupportedModel(
            "ONNX audio output must be [T] or [T, 1, T]".into(),
        ));
    }
    if !tensor_names.iter().any(|n| *n == "emb_g.weight") {
        return Err(ModelError::UnsupportedModel(
            "ONNX RVC speaker embedding initializer is missing".into(),
        ));
    }
    Ok(())
}

fn require_input(
    inputs: &[&ValueInfoProto],
    name: &str,
    rank: usize,
    types: &[i32],
) -> Result<(), ModelError> {
    let value = inputs
        .iter()
        .find(|x| x.name == name)
        .ok_or_else(|| ModelError::UnsupportedModel(format!("ONNX RVC input {name} is missing")))?;
    if shape_of(value)?.len() != rank {
        return Err(ModelError::UnsupportedModel(format!(
            "ONNX RVC input {name} has wrong rank"
        )));
    }
    if !types.contains(&element_type(value)?) {
        return Err(ModelError::UnsupportedModel(format!(
            "ONNX RVC input {name} has unsupported element type"
        )));
    }
    Ok(())
}
fn element_type(value: &ValueInfoProto) -> Result<i32, ModelError> {
    value
        .value_type
        .as_ref()
        .and_then(|x| x.tensor_type.as_ref())
        .map(|x| x.elem_type)
        .ok_or_else(|| {
            ModelError::UnsupportedModel(format!(
                "ONNX input {} has no tensor element type",
                value.name
            ))
        })
}
fn shape_of(value: &ValueInfoProto) -> Result<Vec<Option<i64>>, ModelError> {
    let ty = value
        .value_type
        .as_ref()
        .and_then(|x| x.tensor_type.as_ref())
        .ok_or_else(|| {
            ModelError::UnsupportedModel(format!("ONNX input {} has no tensor type", value.name))
        })?;
    let shape = ty.shape.as_ref().ok_or_else(|| {
        ModelError::UnsupportedModel(format!("ONNX input {} has no shape", value.name))
    })?;
    Ok(shape.dims.iter().map(|d| d.value).collect())
}

#[cfg(test)]
mod version_tests {
    use super::*;
    fn port(name: &str, ty: i32, dims: &[Option<i64>]) -> ValueInfoProto {
        ValueInfoProto {
            name: name.into(),
            value_type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: ty,
                    shape: Some(TensorShapeProto {
                        dims: dims
                            .iter()
                            .map(|value| DimensionProto {
                                value: *value,
                                param: value.is_none().then(|| "frames".into()),
                            })
                            .collect(),
                    }),
                }),
            }),
        }
    }
    fn validate(width: i64, declared: Value) -> Result<(), ModelError> {
        let manifest: RvcModelManifest = serde_json::from_value(json!({
            "id":"v1", "display_name":"v1", "architecture":"rvc-v1", "version":"v1",
            "sample_rate":40000, "model_file":"v1.onnx", "index_file":null, "source":"test", "notes":null,
        })).unwrap();
        let inputs = vec![
            port("feats", 1, &[Some(1), None, Some(width)]),
            port("p_len", 7, &[Some(1)]),
            port("pitch", 7, &[Some(1), None]),
            port("pitchf", 1, &[Some(1), None]),
            port("sid", 7, &[Some(1)]),
        ];
        let props = [StringStringEntryProto {
            key: "metadata".into(),
            value: declared.to_string(),
        }];
        validate_signature(
            &manifest,
            &props,
            &inputs.iter().collect::<Vec<_>>(),
            &[port("audio", 1, &[None])],
            &HashSet::from(["emb_g.weight"]),
        )
    }
    #[test]
    fn voice_changer_application_version_is_independent_of_rvc_version() {
        assert!(validate(256, json!({"version":"2.1","embChannels":256,"embOutputLayer":9,"useFinalProj":true,"f0":true,"samplingRate":40000})).is_ok());
    }
    #[test]
    fn v1_rejects_v2_width_or_nonstandard_acoustic_contract() {
        assert!(validate(768, json!({})).is_err());
        assert!(validate(256, json!({"embOutputLayer":12})).is_err());
        assert!(validate(256, json!({"useFinalProj":false})).is_err());
        assert!(validate(256, json!({"embChannels":768})).is_err());
    }

    #[test]
    fn anonymous_gemm_uses_bias_name_and_normalizes_transposed_weights() {
        let mut graph = GraphProto::default();
        graph.nodes.push(NodeProto {
            name: "Gemm_1".into(),
            op_type: "Gemm".into(),
            inputs: vec![
                "feats".into(),
                "_v_123".into(),
                "enc_p.emb_phone.bias".into(),
            ],
            attributes: vec![AttributeProto {
                name: "transB".into(),
                int: 1,
                ..Default::default()
            }],
            ..Default::default()
        });
        let tensor = Tensor {
            name: "_v_123".into(),
            dtype: TensorDType::F32,
            shape: vec![2, 3],
            data: [1f32, 2., 3., 4., 5., 6.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        };
        let entries = canonicalize_initializers(&graph, vec![tensor.clone()]).unwrap();
        assert_eq!(entries[0].tensor.name, "enc_p.emb_phone.weight");
        assert_eq!(entries[0].tensor.shape, vec![3, 2]);
        assert_eq!(
            canonical::to_f32(&entries[0].tensor).unwrap(),
            vec![1., 4., 2., 5., 3., 6.]
        );
        graph.nodes[0].attributes.push(AttributeProto {
            name: "alpha".into(),
            float: 2.,
            ..Default::default()
        });
        assert!(
            validate_operator_set(&graph.nodes)
                .unwrap_err()
                .to_string()
                .contains("alpha")
        );
        assert!(canonicalize_initializers(&graph, vec![tensor]).is_err());
    }
}
fn value_info_json(value: &ValueInfoProto) -> Value {
    json!({"name": value.name, "shape": shape_of(value).unwrap_or_default()})
}

fn decode_initializer(tensor: &TensorProto) -> Result<Tensor, ModelError> {
    if tensor.name.is_empty() {
        return Err(ModelError::UnsupportedModel(
            "ONNX initializer has no name".into(),
        ));
    }
    if tensor.dims.iter().any(|&d| d < 0) {
        return Err(ModelError::UnsupportedModel(format!(
            "ONNX initializer {} has negative shape",
            tensor.name
        )));
    }
    let (dtype, size) = match tensor.data_type {
        1 => (TensorDType::F32, 4),
        2 => (TensorDType::U8, 1),
        6 => (TensorDType::I32, 4),
        7 => (TensorDType::I64, 8),
        10 => (TensorDType::F16, 2),
        16 => (TensorDType::BF16, 2),
        other => {
            return Err(ModelError::UnsupportedModel(format!(
                "ONNX initializer {} has unsupported data type {other}",
                tensor.name
            )));
        }
    };
    if tensor.data_location != 0 || !tensor.external_data.is_empty() {
        return Err(ModelError::UnsupportedModel(format!(
            "ONNX initializer {} uses external data",
            tensor.name
        )));
    }
    let count = tensor
        .dims
        .iter()
        .try_fold(1_u64, |a, &b| a.checked_mul(b as u64))
        .ok_or_else(|| ModelError::UnsupportedModel("ONNX initializer shape overflows".into()))?;
    let data = if !tensor.raw_data.is_empty() {
        tensor.raw_data.clone()
    } else if dtype == TensorDType::F32 {
        tensor
            .float_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    } else if dtype == TensorDType::I32 || dtype == TensorDType::U8 {
        tensor
            .int32_data
            .iter()
            .flat_map(|v| {
                if dtype == TensorDType::U8 {
                    vec![*v as u8]
                } else {
                    v.to_le_bytes().to_vec()
                }
            })
            .collect()
    } else if dtype == TensorDType::I64 {
        tensor
            .int64_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    } else if dtype == TensorDType::F16 || dtype == TensorDType::BF16 {
        return Err(ModelError::UnsupportedModel(format!(
            "ONNX initializer {} has no raw data for 16-bit type",
            tensor.name
        )));
    } else {
        vec![]
    };
    let expected = count
        .checked_mul(size)
        .ok_or_else(|| ModelError::UnsupportedModel("ONNX initializer size overflows".into()))?;
    if data.len() as u64 != expected {
        return Err(ModelError::UnsupportedModel(format!(
            "ONNX initializer {} has {} bytes, expected {expected}",
            tensor.name,
            data.len()
        )));
    }
    Ok(Tensor {
        name: tensor.name.clone(),
        dtype,
        shape: tensor.dims.iter().map(|&d| d as u64).collect(),
        data,
    })
}
