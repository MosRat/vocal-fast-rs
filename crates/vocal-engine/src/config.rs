use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use vocal_types::{ConversionConfig, ModelDescriptor, TypesError};

use crate::{EngineError, pipeline::RvcPipelineConfig};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BackendKind {
    Passthrough,
    RvcCpuPipeline,
    TensorRtRtx,
    DirectMl,
}

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub backend: BackendKind,
    pub model: ModelDescriptor,
    /// Files needed by the native TensorRT backend.  Keeping this optional
    /// preserves the portable constructors while making the backend choice
    /// explicit for callers that have a compiled VFRM pipeline.
    pub tensorrt: Option<TensorRtEngineSpec>,
    pub directml: Option<DirectMlEngineSpec>,
    pub cpu: Option<CpuEngineSpec>,
}

/// Real CPU graphs require canonical Synth and acoustic VFRM sources.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuEngineSpec {
    pub synth: PathBuf,
    pub hubert: PathBuf,
    pub rmvpe: PathBuf,
    pub execution: crate::CpuExecutionConfig,
    pub window_samples: usize,
    pub speaker_id: usize,
}

/// Native DML graphs are compiled from VFRM when the idle engine is created.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectMlEngineSpec {
    pub sources_path: PathBuf,
    pub speaker_id: usize,
    pub adapter_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorRtEngineSpec {
    /// `pipeline.json` written by `vocal-cli model compile-trt-pipeline`.
    pub artifacts_path: PathBuf,
    /// Canonical `.vfrm`/`.vfr` package used to obtain the Synth schema.
    /// TensorRT plans own the device weights; the package is read once during
    /// engine construction and is not copied into the realtime worker.
    pub model_package_path: PathBuf,
    pub speaker_id: i32,
    /// Optional exact waveform length in the internal 16 kHz domain.  When
    /// omitted, the profile's optimized shape is selected.
    pub window_samples: Option<usize>,
}

/// Execution-only settings that can be applied to an already selected
/// TensorRT engine while it is idle. Build-time precision/source/profile
/// choices stay in the artifact manifest and are intentionally absent here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRtReloadConfig {
    pub artifacts_path: PathBuf,
    pub speaker_id: i32,
    pub window_samples: Option<usize>,
    pub graph_mode: String,
    pub cuda_graph: bool,
    pub native_cuda_graph: bool,
    pub graph_cache_size: usize,
    pub parallel_acoustic: bool,
    pub async_audio_d2h: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineReloadConfig {
    pub backend: BackendKind,
    pub tensorrt: Option<TensorRtReloadConfig>,
}

impl EngineConfig {
    pub fn portable(backend: BackendKind, model: ModelDescriptor) -> Self {
        Self {
            backend,
            model,
            tensorrt: None,
            directml: None,
            cpu: None,
        }
    }

    pub fn with_tensorrt(
        model: ModelDescriptor,
        artifacts_path: impl Into<PathBuf>,
        model_package_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            backend: BackendKind::TensorRtRtx,
            model,
            tensorrt: Some(TensorRtEngineSpec {
                artifacts_path: artifacts_path.into(),
                model_package_path: model_package_path.into(),
                speaker_id: 0,
                window_samples: None,
            }),
            directml: None,
            cpu: None,
        }
    }

    pub fn with_directml(model: ModelDescriptor, sources_path: impl Into<PathBuf>) -> Self {
        Self {
            backend: BackendKind::DirectMl,
            model,
            tensorrt: None,
            directml: Some(DirectMlEngineSpec {
                sources_path: sources_path.into(),
                speaker_id: 0,
                adapter_id: None,
            }),
            cpu: None,
        }
    }

    pub fn with_cpu_graph(model: ModelDescriptor, spec: CpuEngineSpec) -> Self {
        Self {
            backend: BackendKind::RvcCpuPipeline,
            model,
            tensorrt: None,
            directml: None,
            cpu: Some(spec),
        }
    }

    pub fn validate(&self) -> Result<(), EngineError> {
        self.model
            .validate()
            .map_err(|error| EngineError::InvalidConfig(error.to_string()))?;
        if self.backend == BackendKind::TensorRtRtx && self.tensorrt.is_none() {
            return Err(EngineError::InvalidConfig(
                "TensorRT backend requires pipeline artifacts and a VFRM model package".into(),
            ));
        }
        if self.backend == BackendKind::DirectMl && self.directml.is_none() {
            return Err(EngineError::InvalidConfig(
                "DirectML backend requires a VFRM sources manifest".into(),
            ));
        }
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
