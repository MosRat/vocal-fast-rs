use serde::{Deserialize, Serialize};
use serde_json::Value;
use vocal_types::ModelDescriptor;

use crate::schema::RVC_TENSOR_SCHEMA;
use crate::{ModelError, RvcSynthConfig, schema};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RvcModelManifest {
    pub id: String,
    pub display_name: String,
    pub architecture: String,
    pub version: String,
    pub sample_rate: u32,
    pub model_file: String,
    pub index_file: Option<String>,
    pub source: String,
    pub notes: Option<String>,
}

impl RvcModelManifest {
    pub fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            id: self.id.clone(),
            display_name: self.display_name.clone(),
            architecture: self.architecture.clone(),
            sample_rate: self.sample_rate,
            version: self.version.clone(),
        }
    }
    pub fn feature_dims(&self) -> usize {
        if self.version.to_ascii_lowercase().contains("v2")
            || self.architecture.to_ascii_lowercase().contains("v2")
        {
            768
        } else {
            256
        }
    }
    pub fn uses_f0(&self) -> bool {
        !self.architecture.to_ascii_lowercase().contains("nono")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TensorDType {
    U8 = 1,
    I32 = 2,
    I64 = 3,
    F16 = 4,
    F32 = 5,
    BF16 = 6,
}

impl TensorDType {
    pub(crate) fn from_byte(byte: u8) -> Result<Self, ModelError> {
        match byte {
            1 => Ok(Self::U8),
            2 => Ok(Self::I32),
            3 => Ok(Self::I64),
            4 => Ok(Self::F16),
            5 => Ok(Self::F32),
            6 => Ok(Self::BF16),
            _ => Err(ModelError::InvalidCache(format!(
                "unknown tensor dtype {byte}"
            ))),
        }
    }
    pub fn element_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 => 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub dtype: TensorDType,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn byte_tensor(name: impl Into<String>, data: Vec<u8>) -> Self {
        let len = data.len() as u64;
        Self {
            name: name.into(),
            dtype: TensorDType::U8,
            shape: vec![len],
            data,
        }
    }
    pub(crate) fn validate(&self) -> Result<(), ModelError> {
        if self.name.is_empty() || self.name.len() > u16::MAX as usize {
            return Err(ModelError::InvalidCache(
                "invalid tensor name length".into(),
            ));
        }
        if self.shape.len() > u8::MAX as usize {
            return Err(ModelError::InvalidCache("tensor rank exceeds u8".into()));
        }
        let elements = self.shape.iter().try_fold(1_u64, |a, &b| a.checked_mul(b));
        let expected = elements
            .and_then(|n| n.checked_mul(self.dtype.element_size() as u64))
            .ok_or_else(|| ModelError::InvalidCache("tensor shape overflows".into()))?;
        if expected != self.data.len() as u64 {
            return Err(ModelError::InvalidCache(format!(
                "tensor {} has {} bytes, expected {}",
                self.name,
                self.data.len(),
                expected
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ModelPackage {
    pub manifest: RvcModelManifest,
    pub metadata: Value,
    pub tensors: Vec<Tensor>,
}

impl ModelPackage {
    /// Check the complete canonical Synthesizer parameter directory before building a graph.
    pub fn validate_rvc_synth(&self) -> Result<RvcSynthConfig, ModelError> {
        if self
            .metadata
            .get("canonical_schema")
            .and_then(Value::as_str)
            != Some(RVC_TENSOR_SCHEMA)
        {
            return Err(ModelError::UnsupportedModel(
                "expected canonical rvc-synth-v1 package".into(),
            ));
        }
        let config = self.rvc_synth_config()?.ok_or_else(|| {
            ModelError::InvalidCache("canonical package has no rvc_config".into())
        })?;
        schema::validate_tensors(&config, &self.manifest, &self.tensors)?;
        Ok(config)
    }
    /// Find a tensor by its canonical RVC parameter name.
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    pub fn rvc_synth_config(&self) -> Result<Option<RvcSynthConfig>, ModelError> {
        self.metadata
            .get("rvc_config")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(ModelError::from)
    }
}
