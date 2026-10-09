//! Infer checkpoint identity from checkpoint metadata, never from user guesses.
use crate::{ModelError, ModelPackage, RvcModelManifest};
use serde_json::Value;
use std::path::Path;

pub fn load_rvc_pth_auto(
    path: impl AsRef<Path>,
    display_name: &str,
    source: &str,
) -> Result<ModelPackage, ModelError> {
    let path = path.as_ref();
    crate::ensure_file_size(path, 2 * 1024 * 1024 * 1024)?;
    let (metadata, tensors) = crate::pickle::load_rvc_pth(path)?;
    let phone = tensors
        .iter()
        .find(|t| t.name == "enc_p.emb_phone.weight")
        .ok_or_else(|| ModelError::UnsupportedModel("RVC phone projection missing".into()))?;
    let version = match phone.shape.last().copied() {
        Some(768) => "v2",
        Some(256) => "v1",
        _ => {
            return Err(ModelError::UnsupportedModel(
                "unknown RVC feature width".into(),
            ));
        }
    };
    let rate_value = metadata.get("sr").or_else(|| {
        metadata
            .get("config")
            .and_then(Value::as_array)
            .and_then(|v| v.last())
    });
    let sample_rate = rate_value
        .and_then(|v| {
            v.as_u64().or_else(|| {
                v.as_str().and_then(|s| match s {
                    "32k" => Some(32000),
                    "40k" => Some(40000),
                    "48k" => Some(48000),
                    _ => s.parse().ok(),
                })
            })
        })
        .filter(|n| matches!(n, 32000 | 40000 | 48000))
        .ok_or_else(|| {
            ModelError::UnsupportedModel("checkpoint has no supported sample rate".into())
        })? as u32;
    let f0 = metadata
        .get("f0")
        .and_then(Value::as_i64)
        .map(|n| n != 0)
        .unwrap_or_else(|| tensors.iter().any(|t| t.name == "enc_p.emb_pitch.weight"));
    let name = display_name.trim();
    if name.is_empty() || name.len() > 512 {
        return Err(ModelError::InvalidConfig(
            "model name must contain 1–512 UTF-8 bytes".into(),
        ));
    }
    let manifest = RvcModelManifest {
        id: path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("imported-model")
            .into(),
        display_name: name.into(),
        architecture: format!("rvc-{version}{}", if f0 { "" } else { "-nono" }),
        version: version.into(),
        sample_rate,
        model_file: path.file_name().unwrap().to_string_lossy().into(),
        index_file: None,
        source: source.into(),
        notes: None,
    };
    crate::validate_rvc_checkpoint_metadata(&manifest, &metadata)?;
    let (metadata, tensors) = crate::canonical::normalize_pth(&manifest, metadata, tensors)?;
    crate::validate_rvc_tensors(&manifest, &tensors)?;
    Ok(ModelPackage {
        manifest,
        metadata,
        tensors,
    })
}
