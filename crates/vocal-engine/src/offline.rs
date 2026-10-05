//! Shared offline execution for every inference backend.
//!
//! The WAV boundary belongs to the CLI/application.  This module owns the
//! backend-neutral part: fixed-size windows, final-window padding, sequence
//! numbers, output trimming, and a small timing report.  A new backend only
//! needs to implement [`OfflinePipeline`]; it does not need to duplicate the
//! audio file and windowing code.

use std::time::Instant;

use serde::{Deserialize, Serialize};
use vocal_types::{AudioFormat, ConversionConfig, LatencyMetrics};

use crate::{BackendKind, EngineError, InferenceEngine};

/// The default internal RVC window.  TensorRT profiles use this shape, and it
/// is also a useful bounded block for the portable CPU and passthrough paths.
pub const DEFAULT_WINDOW_SAMPLES: usize = 11_200;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OfflineRunConfig {
    /// Number of mono 16 kHz samples submitted per backend call.
    pub window_samples: usize,
    /// Pad the last window with zeros before invoking the backend.
    pub pad_final_window: bool,
    pub conversion: ConversionConfig,
}

impl Default for OfflineRunConfig {
    fn default() -> Self {
        Self {
            window_samples: DEFAULT_WINDOW_SAMPLES,
            pad_final_window: true,
            conversion: ConversionConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OfflineRunReport {
    pub backend: BackendKind,
    pub input_samples: usize,
    pub output_samples: usize,
    pub windows: usize,
    pub elapsed_ms: f64,
    pub metrics: LatencyMetrics,
}

/// Backend boundary used by [`run_offline`].
pub trait OfflinePipeline: Send {
    fn backend_kind(&self) -> BackendKind;

    fn metrics(&self) -> LatencyMetrics {
        LatencyMetrics::default()
    }

    fn process_window(
        &mut self,
        samples: &[f32],
        sequence: u64,
        conversion: ConversionConfig,
    ) -> Result<Vec<f32>, EngineError>;
}

/// Adapt the normal streaming engine interface to the offline runner.  This
/// is intentionally generic over the concrete engine, so passthrough, CPU
/// RVC, TensorRT RTX, and future backends share exactly the same runner.
pub struct EngineOfflinePipeline {
    backend: BackendKind,
    format: AudioFormat,
    engine: Box<dyn InferenceEngine>,
}

impl EngineOfflinePipeline {
    pub fn new(
        backend: BackendKind,
        engine: Box<dyn InferenceEngine>,
        format: AudioFormat,
    ) -> Result<Self, EngineError> {
        format
            .validate()
            .map_err(|error| EngineError::InvalidConfig(error.to_string()))?;
        if format.channels != 1 {
            return Err(EngineError::InvalidConfig(
                "offline runner currently requires mono input".into(),
            ));
        }
        Ok(Self {
            backend,
            format,
            engine,
        })
    }

    pub fn engine(&self) -> &dyn InferenceEngine {
        self.engine.as_ref()
    }

    pub fn engine_mut(&mut self) -> &mut dyn InferenceEngine {
        self.engine.as_mut()
    }
}

impl OfflinePipeline for EngineOfflinePipeline {
    fn backend_kind(&self) -> BackendKind {
        self.backend
    }

    fn metrics(&self) -> LatencyMetrics {
        self.engine.metrics()
    }

    fn process_window(
        &mut self,
        samples: &[f32],
        sequence: u64,
        conversion: ConversionConfig,
    ) -> Result<Vec<f32>, EngineError> {
        Ok(self
            .engine
            .process_samples(self.format, samples, sequence, conversion)?
            .samples)
    }
}

pub fn run_offline<P: OfflinePipeline + ?Sized>(
    pipeline: &mut P,
    input: &[f32],
    config: &OfflineRunConfig,
) -> Result<(Vec<f32>, OfflineRunReport), EngineError> {
    if input.is_empty() || input.iter().any(|sample| !sample.is_finite()) {
        return Err(EngineError::InvalidInput(
            "offline input must be non-empty and finite".into(),
        ));
    }
    if config.window_samples == 0 {
        return Err(EngineError::InvalidConfig(
            "offline window size must be non-zero".into(),
        ));
    }

    let started = Instant::now();
    let mut output = Vec::with_capacity(input.len());
    let mut offset = 0usize;
    let mut windows = 0usize;
    while offset < input.len() {
        let valid = (input.len() - offset).min(config.window_samples);
        let needs_padding = valid < config.window_samples;
        let window = if needs_padding && config.pad_final_window {
            let mut padded = vec![0.0; config.window_samples];
            padded[..valid].copy_from_slice(&input[offset..offset + valid]);
            padded
        } else {
            input[offset..offset + valid].to_vec()
        };
        let result = pipeline.process_window(&window, windows as u64, config.conversion)?;
        let keep = if needs_padding {
            if config.pad_final_window {
                result.len().saturating_mul(valid) / window.len().max(1)
            } else {
                result.len()
            }
        } else {
            result.len()
        };
        output.extend_from_slice(&result[..keep.min(result.len())]);
        offset += valid;
        windows += 1;
    }

    let metrics = pipeline.metrics();
    Ok((
        output.clone(),
        OfflineRunReport {
            backend: pipeline.backend_kind(),
            input_samples: input.len(),
            output_samples: output.len(),
            windows,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            metrics,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EngineConfig, create_engine};
    use vocal_types::ModelDescriptor;

    fn descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "offline-fixture".into(),
            display_name: "Offline fixture".into(),
            architecture: "rvc-v2".into(),
            sample_rate: 16_000,
            version: "2".into(),
        }
    }

    #[test]
    fn all_portable_backends_use_the_same_window_runner() {
        for backend in [BackendKind::Passthrough, BackendKind::RvcCpuPipeline] {
            let engine = create_engine(EngineConfig::portable(backend, descriptor())).unwrap();
            let mut pipeline =
                EngineOfflinePipeline::new(backend, engine, AudioFormat::MONO_16K).unwrap();
            let input = vec![0.1; 12_345];
            let (output, report) =
                run_offline(&mut pipeline, &input, &OfflineRunConfig::default()).unwrap();
            assert_eq!(output.len(), input.len());
            assert_eq!(report.windows, 2);
            assert_eq!(report.backend, backend);
        }
    }
}
