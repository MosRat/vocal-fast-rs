//! Inference boundary. TensorRT RTX integration is intentionally isolated from
//! the portable baseline so the SDK can be linked only on RTX build machines.

use std::time::Instant;
use vocal_types::{AudioChunk, AudioFormat, ConversionConfig, LatencyMetrics, ModelDescriptor};

#[cfg_attr(not(feature = "tensorrt-rtx"), allow(dead_code))]
mod audio_adapter;
mod config;
mod error;
mod logging;
mod state;

#[cfg(feature = "tensorrt-rtx")]
mod tensorrt_backend;

pub mod offline;
pub mod pipeline;
pub mod tensorrt;

pub use config::{
    BackendKind, EngineConfig, EngineReloadConfig, TensorRtEngineSpec, TensorRtReloadConfig,
    validate_conversion,
};
pub use error::EngineError;
pub use state::{EngineLifecycle, EngineStatus};

#[cfg(feature = "tensorrt-rtx")]
pub use tensorrt_backend::{TensorRtBackendConfig, TensorRtEngine, create_tensorrt_engine};

pub trait InferenceEngine: Send {
    /// Apply execution-only changes while the engine is idle. Model/backend
    /// changes are owned by the runtime and must replace the engine instead.
    /// Backends that do not keep execution state can accept this as a no-op.
    fn reload_config(&mut self, _config: EngineReloadConfig) -> Result<(), EngineError> {
        Ok(())
    }

    fn process(
        &mut self,
        input: &AudioChunk,
        config: ConversionConfig,
    ) -> Result<AudioChunk, EngineError>;

    /// Process a ring slot without first allocating an `AudioChunk` for the
    /// borrowed input samples. Control-plane callers can keep using
    /// [`Self::process`]; real-time workers should prefer this method.
    fn process_samples(
        &mut self,
        format: AudioFormat,
        samples: &[f32],
        sequence: u64,
        config: ConversionConfig,
    ) -> Result<AudioChunk, EngineError> {
        self.process(
            &AudioChunk {
                format,
                samples: samples.to_vec(),
                sequence,
            },
            config,
        )
    }

    fn metrics(&self) -> LatencyMetrics;
    fn status(&self) -> EngineStatus {
        EngineStatus::default()
    }
}

pub fn create_engine(config: EngineConfig) -> Result<Box<dyn InferenceEngine>, EngineError> {
    config.validate()?;
    match config.backend {
        BackendKind::Passthrough => {
            logging::created("passthrough");
            Ok(Box::new(PassthroughEngine::new(config.model)))
        }
        BackendKind::RvcCpuPipeline => {
            logging::created("rvc-cpu");
            Ok(Box::new(PipelineEngine::new(config.model)?))
        }
        BackendKind::TensorRtRtx => {
            #[cfg(feature = "tensorrt-rtx")]
            {
                let spec = config.tensorrt.ok_or_else(|| {
                    EngineError::InvalidConfig(
                        "TensorRT backend requires pipeline artifacts and a VFRM model package"
                            .into(),
                    )
                })?;
                logging::created("tensorrt-rtx");
                return create_tensorrt_engine(config.model, spec);
            }
            #[cfg(not(feature = "tensorrt-rtx"))]
            {
                Err(EngineError::BackendUnavailable(
                    "TensorRT RTX backend requires the vocal-engine/tensorrt-rtx feature and an RTX SDK".into(),
                ))
            }
        }
    }
}

pub struct PipelineEngine {
    pipeline: pipeline::RvcPipeline,
    metrics: LatencyMetrics,
    history: Vec<f32>,
    last_timings: pipeline::PipelineTimings,
    status: EngineStatus,
}

impl PipelineEngine {
    fn new(model: ModelDescriptor) -> Result<Self, EngineError> {
        let config = pipeline::RvcPipelineConfig {
            target_sample_rate: if model.sample_rate == 0 {
                32_000
            } else {
                model.sample_rate
            },
            feature_dims: if model.version.to_ascii_lowercase().contains("v2")
                || model.architecture.to_ascii_lowercase().contains("v2")
            {
                768
            } else {
                256
            },
            ..Default::default()
        };
        Ok(Self {
            pipeline: pipeline::default_cpu_pipeline(config)?,
            metrics: LatencyMetrics::default(),
            history: Vec::new(),
            last_timings: pipeline::PipelineTimings::default(),
            status: EngineStatus {
                state: EngineLifecycle::Ready,
                ..Default::default()
            },
        })
    }

    pub fn from_pipeline(pipeline: pipeline::RvcPipeline) -> Self {
        Self {
            pipeline,
            metrics: LatencyMetrics::default(),
            history: Vec::new(),
            last_timings: pipeline::PipelineTimings::default(),
            status: EngineStatus {
                state: EngineLifecycle::Ready,
                ..Default::default()
            },
        }
    }

    pub fn reset(&mut self) {
        self.history.clear();
    }

    pub fn last_timings(&self) -> pipeline::PipelineTimings {
        self.last_timings
    }
}

impl InferenceEngine for PipelineEngine {
    fn process(
        &mut self,
        input: &AudioChunk,
        config: ConversionConfig,
    ) -> Result<AudioChunk, EngineError> {
        self.process_samples(input.format, &input.samples, input.sequence, config)
    }

    fn process_samples(
        &mut self,
        format: AudioFormat,
        samples: &[f32],
        sequence: u64,
        config: ConversionConfig,
    ) -> Result<AudioChunk, EngineError> {
        self.status
            .begin()
            .map_err(|error| EngineError::InvalidState(error.into()))?;
        if let Err(error) =
            validate_audio_samples(format, samples).and_then(|_| validate_conversion(config))
        {
            self.status.failure(error.to_string());
            logging::failed(&error);
            return Err(error);
        }
        let started = Instant::now();
        let channels = format.channels as usize;
        let mono = if channels == 1 {
            samples.to_vec()
        } else {
            samples
                .chunks_exact(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                .collect()
        };
        let context = pipeline::resampled_len(1_600, 16_000, format.sample_rate).max(160);
        if self.history.is_empty() {
            self.history.resize(context, 0.0);
        }
        self.history.extend_from_slice(&mono);
        let output = match self.pipeline.run(&self.history, format.sample_rate, config) {
            Ok(output) => output,
            Err(error) => {
                self.status.failure(error.to_string());
                logging::failed(&error);
                return Err(error);
            }
        };
        self.last_timings = output.timings;
        let output_mono =
            pipeline::resample_linear(&output.samples, output.sample_rate, format.sample_rate);
        let mut output_block = if output_mono.len() >= mono.len() {
            output_mono[output_mono.len() - mono.len()..].to_vec()
        } else {
            let mut block = vec![0.0; mono.len() - output_mono.len()];
            block.extend_from_slice(&output_mono);
            block
        };
        output_block.resize(mono.len(), 0.0);
        let keep_from = self.history.len().saturating_sub(context);
        // Keep the allocation owned by the streaming engine. Rebuilding a
        // Vec from a slice here used to allocate and copy the retained
        // context on every audio block, which showed up as periodic latency
        // spikes for the CPU backend. `drain` shifts the small retained tail
        // in place and reuses the existing capacity on the next block.
        if keep_from > 0 {
            self.history.drain(..keep_from);
        }
        let mut samples = Vec::with_capacity(output_block.len() * channels);
        for sample in output_block {
            samples.extend(std::iter::repeat_n(sample, channels));
        }
        let output_frames = samples.len() / channels;
        self.metrics.input_frames += (samples.len() / channels) as u64;
        self.metrics.output_frames += output_frames as u64;
        self.metrics.processing_micros += started.elapsed().as_micros() as u64;
        self.status.success();
        Ok(AudioChunk {
            format,
            samples,
            sequence,
        })
    }

    fn metrics(&self) -> LatencyMetrics {
        self.metrics
    }

    fn status(&self) -> EngineStatus {
        self.status.clone()
    }
}

pub fn create_pipeline_engine(pipeline: pipeline::RvcPipeline) -> Box<dyn InferenceEngine> {
    Box::new(PipelineEngine::from_pipeline(pipeline))
}

struct PassthroughEngine {
    metrics: LatencyMetrics,
    status: EngineStatus,
}

impl PassthroughEngine {
    fn new(_model: ModelDescriptor) -> Self {
        Self {
            metrics: LatencyMetrics::default(),
            status: EngineStatus {
                state: EngineLifecycle::Ready,
                ..Default::default()
            },
        }
    }
}

impl InferenceEngine for PassthroughEngine {
    fn process(
        &mut self,
        input: &AudioChunk,
        _config: ConversionConfig,
    ) -> Result<AudioChunk, EngineError> {
        self.process_samples(input.format, &input.samples, input.sequence, _config)
    }

    fn process_samples(
        &mut self,
        format: AudioFormat,
        samples: &[f32],
        sequence: u64,
        _config: ConversionConfig,
    ) -> Result<AudioChunk, EngineError> {
        self.status
            .begin()
            .map_err(|error| EngineError::InvalidState(error.into()))?;
        if let Err(error) =
            validate_audio_samples(format, samples).and_then(|_| validate_conversion(_config))
        {
            self.status.failure(error.to_string());
            logging::failed(&error);
            return Err(error);
        }
        let start = Instant::now();
        let output = AudioChunk {
            format,
            samples: samples.to_vec(),
            sequence,
        };
        self.metrics.input_frames += output.frames() as u64;
        self.metrics.output_frames += output.frames() as u64;
        self.metrics.processing_micros += start.elapsed().as_micros() as u64;
        self.status.success();
        Ok(output)
    }

    fn metrics(&self) -> LatencyMetrics {
        self.metrics
    }

    fn status(&self) -> EngineStatus {
        self.status.clone()
    }
}

fn validate_audio_samples(format: AudioFormat, samples: &[f32]) -> Result<(), EngineError> {
    format
        .validate()
        .map_err(|error| EngineError::InvalidInput(error.to_string()))?;
    if samples.is_empty() {
        return Err(EngineError::InvalidInput("audio chunk is empty".into()));
    }
    if !samples.len().is_multiple_of(format.channels as usize) {
        return Err(EngineError::InvalidInput(
            "interleaved samples do not match channel count".into(),
        ));
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(EngineError::InvalidInput(
            "audio samples must be finite".into(),
        ));
    }
    Ok(())
}

pub fn model_name(config: &EngineConfig) -> &str {
    &config.model.display_name
}

#[cfg(test)]
mod tests {
    use super::*;
    use vocal_types::{AudioChunk, AudioFormat};

    #[test]
    fn passthrough_preserves_audio() {
        let model = ModelDescriptor {
            id: "fixture".into(),
            display_name: "Fixture".into(),
            architecture: "rvc-v2".into(),
            sample_rate: 16_000,
            version: "1".into(),
        };
        let mut engine = create_engine(EngineConfig {
            backend: BackendKind::Passthrough,
            model,
            tensorrt: None,
        })
        .unwrap();
        let input = AudioChunk {
            format: AudioFormat::MONO_16K,
            samples: vec![0.1, -0.2],
            sequence: 7,
        };
        assert_eq!(
            engine
                .process(&input, ConversionConfig::default())
                .unwrap()
                .samples,
            input.samples
        );
    }

    #[test]
    fn cpu_pipeline_handles_interleaved_input() {
        let model = ModelDescriptor {
            id: "fixture".into(),
            display_name: "Fixture".into(),
            architecture: "rvc-v2".into(),
            sample_rate: 32_000,
            version: "1".into(),
        };
        let mut engine = create_engine(EngineConfig {
            backend: BackendKind::RvcCpuPipeline,
            model,
            tensorrt: None,
        })
        .unwrap();
        let input = AudioChunk {
            format: vocal_types::AudioFormat {
                sample_rate: 16_000,
                channels: 2,
            },
            samples: (0..320)
                .flat_map(|index| [0.1 * (index as f32).sin(), 0.2 * (index as f32).sin()])
                .collect(),
            sequence: 3,
        };
        let output = engine.process(&input, ConversionConfig::default()).unwrap();
        assert_eq!(output.format, input.format);
        assert_eq!(output.frames(), input.frames());
        assert_eq!(output.sequence, input.sequence);
        let second = engine
            .process(
                &AudioChunk {
                    sequence: 4,
                    ..input.clone()
                },
                ConversionConfig::default(),
            )
            .unwrap();
        assert_eq!(second.frames(), input.frames());
    }

    #[test]
    fn tensor_rt_backend_requires_pipeline_specification() {
        let model = ModelDescriptor {
            id: "fixture".into(),
            display_name: "Fixture".into(),
            architecture: "rvc-v2".into(),
            sample_rate: 32_000,
            version: "2".into(),
        };
        let result = create_engine(EngineConfig {
            backend: BackendKind::TensorRtRtx,
            model,
            tensorrt: None,
        });
        assert!(matches!(result, Err(EngineError::InvalidConfig(_))));
    }

    #[cfg(not(feature = "tensorrt-rtx"))]
    #[test]
    fn tensor_rt_backend_reports_disabled_feature() {
        let model = ModelDescriptor {
            id: "fixture".into(),
            display_name: "Fixture".into(),
            architecture: "rvc-v2".into(),
            sample_rate: 32_000,
            version: "2".into(),
        };
        let result = create_engine(EngineConfig {
            backend: BackendKind::TensorRtRtx,
            model,
            tensorrt: Some(TensorRtEngineSpec {
                artifacts_path: "missing-pipeline.json".into(),
                model_package_path: "missing.vfrm".into(),
                speaker_id: 0,
                window_samples: None,
            }),
        });
        assert!(matches!(result, Err(EngineError::BackendUnavailable(_))));
    }
}
