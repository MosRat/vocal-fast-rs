use vocal_types::{ConversionConfig, ModelDescriptor, TypesError};

use crate::{EngineError, pipeline::RvcPipelineConfig};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendKind {
    Passthrough,
    RvcCpuPipeline,
    TensorRtRtx,
}

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub backend: BackendKind,
    pub model: ModelDescriptor,
}

impl EngineConfig {
    pub fn validate(&self) -> Result<(), EngineError> {
        self.model
            .validate()
            .map_err(|error| EngineError::InvalidConfig(error.to_string()))?;
        Ok(())
    }
}

impl RvcPipelineConfig {
    pub fn validate(&self) -> Result<(), EngineError> {
        if self.target_sample_rate == 0
            || self.output_sample_rate == 0
            || self.feature_dims == 0
            || self.feature_hop_samples == 0
        {
            return Err(EngineError::InvalidConfig(
                "sample rates, feature dimensions and hop size must be non-zero".into(),
            ));
        }
        if self.pad_samples > self.target_sample_rate as usize * 2 {
            return Err(EngineError::InvalidConfig(
                "pipeline padding is unreasonably large".into(),
            ));
        }
        Ok(())
    }
}

pub fn validate_conversion(config: ConversionConfig) -> Result<(), EngineError> {
    config
        .validate()
        .map_err(|error: TypesError| EngineError::InvalidInput(error.to_string()))
}
