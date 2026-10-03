//! Canonical metadata contract for the waveform feature extractors.
//!
//! Acoustic bundles intentionally keep the operator IR in metadata and the
//! initializer table in the same VFRM container.  The IR is consumed by the
//! native TensorRT builder; it is never an ONNX protobuf at runtime.

use crate::{ModelError, ModelPackage, TensorDType};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const ACOUSTIC_TENSOR_SCHEMA: &str = "acoustic-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AcousticKind {
    Hubert,
    Rmvpe,
}

impl AcousticKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hubert => "hubert",
            Self::Rmvpe => "rmvpe",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcousticTensorMap {
    pub canonical: String,
    pub source: String,
    pub sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcousticPort {
    pub name: String,
    pub shape: Vec<Value>,
    pub dtype: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcousticConfig {
    pub kind: AcousticKind,
    pub graph_schema: String,
    pub inputs: Vec<AcousticPort>,
    pub outputs: Vec<AcousticPort>,
    pub node_count: usize,
    pub tensor_map: Vec<AcousticTensorMap>,
}

impl ModelPackage {
    pub fn acoustic_config(&self) -> Result<Option<AcousticConfig>, ModelError> {
        if self
            .metadata
            .get("canonical_schema")
            .and_then(Value::as_str)
            != Some(ACOUSTIC_TENSOR_SCHEMA)
        {
            return Ok(None);
        }
        serde_json::from_value(self.metadata.clone())
            .map(Some)
            .map_err(ModelError::from)
    }

    pub fn validate_acoustic(
        &self,
        expected: Option<AcousticKind>,
    ) -> Result<AcousticConfig, ModelError> {
        let config = self.acoustic_config()?.ok_or_else(|| {
            ModelError::UnsupportedModel("expected acoustic-v1 VFRM package".into())
        })?;
        if expected.is_some_and(|kind| kind != config.kind) {
            return Err(ModelError::UnsupportedModel(format!(
                "acoustic package kind {} does not match requested kind",
                config.kind.as_str()
            )));
        }
        if config.node_count == 0 || config.node_count > 10_000 {
            return Err(ModelError::InvalidCache(
                "invalid acoustic node count".into(),
            ));
        }
        if config.inputs.is_empty() || config.outputs.is_empty() {
            return Err(ModelError::InvalidCache(
                "acoustic graph has no ports".into(),
            ));
        }
        let prefix = format!("{}.", config.kind.as_str());
        let mapped: std::collections::HashSet<_> = config
            .tensor_map
            .iter()
            .map(|entry| entry.canonical.as_str())
            .collect();
        for tensor in &self.tensors {
            if !tensor.name.starts_with(&prefix) {
                return Err(ModelError::InvalidCache(format!(
                    "acoustic tensor {} is outside {} namespace",
                    tensor.name, prefix
                )));
            }
            if !matches!(
                tensor.dtype,
                TensorDType::F16
                    | TensorDType::F32
                    | TensorDType::BF16
                    | TensorDType::I32
                    | TensorDType::I64
                    | TensorDType::U8
            ) {
                return Err(ModelError::InvalidCache(format!(
                    "unsupported acoustic dtype for {}",
                    tensor.name
                )));
            }
        }
        for name in mapped {
            if self.tensor(name).is_none() {
                return Err(ModelError::InvalidCache(format!(
                    "acoustic mapping misses tensor {name}"
                )));
            }
        }
        Ok(config)
    }
}
