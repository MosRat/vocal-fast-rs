//! Inference boundary. TensorRT RTX integration is intentionally isolated from
//! the portable baseline so the SDK can be linked only on RTX build machines.

use std::time::Instant;
use vocal_types::{AudioChunk, AudioFormat, ConversionConfig, LatencyMetrics, ModelDescriptor};

#[cfg_attr(
    not(any(feature = "tensorrt-rtx", feature = "directml")),
    allow(dead_code)
)]
mod audio_adapter;
mod config;
mod cpu_config;
mod error;
mod logging;
mod state;
#[cfg_attr(
    not(any(feature = "cpu-graph", feature = "directml", feature = "tensorrt-rtx")),
    allow(dead_code)
)]
mod streaming_output;
#[cfg(any(feature = "cpu-graph", feature = "directml", feature = "tensorrt-rtx"))]
mod streaming_resampler;

#[cfg(feature = "cpu-graph")]
mod cpu_backend;
#[cfg(feature = "directml")]
mod directml_backend;
#[cfg(feature = "tensorrt-rtx")]
mod tensorrt_backend;

pub mod offline;
pub mod pipeline;
pub mod tensorrt;

pub use config::{
    BackendKind, CpuEngineSpec, DirectMlEngineSpec, EngineConfig, EngineReloadConfig,
    TensorRtEngineSpec, TensorRtReloadConfig, validate_conversion,
};
pub use cpu_config::CpuExecutionConfig;
pub use error::EngineError;
pub use state::{EngineLifecycle, EngineStatus};
#[cfg(feature = "cpu-graph")]
pub use vocal_cpu::operator_version as cpu_operator_version;
mod reload;
pub use reload::PreparedEngineReload;

#[cfg(feature = "cpu-graph")]
pub use cpu_backend::{CpuEngine, CpuStreamConfig};
#[cfg(feature = "directml")]
pub use directml_backend::{DirectMlEngine, DirectMlStreamConfig};
#[cfg(feature = "tensorrt-rtx")]
pub use tensorrt_backend::{TensorRtBackendConfig, TensorRtEngine, create_tensorrt_engine};

pub trait InferenceEngine: Send {
    /// Best-effort persistence of runtime compilation caches while idle.
    /// Controllers call this after the worker has returned engine ownership.
    fn flush_idle_caches(&mut self) {}

    /// Preferred processing batch in the caller's format. Zero means the
    /// backend can process each device block. Device periods remain independent.
    fn preferred_batch_frames(&self, _format: AudioFormat) -> usize {
        0
    }

    /// Discard all stream history on a new session or a capture discontinuity.
    /// The worker owns the engine and invokes this without a control-plane lock.
    fn reset_stream(&mut self) -> Result<(), EngineError> {
        Ok(())
    }

    /// Validate controls against this engine's capabilities before saving an
    /// idle update. This must not change stream state or poison engine status.
    fn validate_config(&self, config: ConversionConfig) -> Result<(), EngineError> {
        validate_conversion(config)
    }

    /// Prepare all fallible execution changes while idle. The old engine
    /// remains usable if preparation or the controller's persistence fails.
    fn prepare_reload_config(
        &mut self,
        _config: EngineReloadConfig,
    ) -> Result<Box<dyn PreparedEngineReload + '_>, EngineError> {
        Err(EngineError::InvalidConfig(
            "this engine does not support execution configuration reload".into(),
        ))
    }

    fn reload_config(&mut self, config: EngineReloadConfig) -> Result<(), EngineError> {
        self.prepare_reload_config(config)?.commit();
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
            if let Some(spec) = config.cpu {
                #[cfg(feature = "cpu-graph")]
                {
                    return Ok(Box::new(CpuEngine::from_spec(config.model, spec)?));
                }
                #[cfg(not(feature = "cpu-graph"))]
                {
                    let _ = spec;
                    return Err(EngineError::BackendUnavailable(
                        "CPU graphs require vocal-engine/cpu-graph".into(),
                    ));
                }
            }
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
        BackendKind::DirectMl => {
            #[cfg(feature = "directml")]
            {
                let spec = config
                    .directml
                    .ok_or_else(|| EngineError::InvalidConfig("missing DirectML sources".into()))?;
                logging::created("directml");
                Ok(Box::new(DirectMlEngine::from_spec(config.model, spec)?))
            }
            #[cfg(not(feature = "directml"))]
            {
                Err(EngineError::BackendUnavailable(
                    "DirectML backend requires vocal-engine/directml and Windows".into(),
                ))
            }
        }
    }
}

pub struct PipelineEngine {
    pipeline: pipeline::RvcPipeline,
    metrics: LatencyMetrics,
    history: Vec<f32>,
    format: Option<AudioFormat>,
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
            feature_dims: vocal_model::RvcVersion::from_labels(&model.version, &model.architecture)
                .map(|version| version.feature_width())
                .map_err(|error| EngineError::InvalidConfig(error.to_string()))?,
            ..Default::default()
        };
        Ok(Self {
            pipeline: pipeline::default_cpu_pipeline(config)?,
            metrics: LatencyMetrics::default(),
            history: Vec::new(),
            format: None,
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
            format: None,
            last_timings: pipeline::PipelineTimings::default(),
            status: EngineStatus {
                state: EngineLifecycle::Ready,
                ..Default::default()
            },
        }
    }

    pub fn reset(&mut self) {
        self.pipeline.reset_stream();
        self.history.clear();
        self.format = None;
        self.status.ready();
    }

    pub fn last_timings(&self) -> pipeline::PipelineTimings {
        self.last_timings
    }
}

impl InferenceEngine for PipelineEngine {
    fn prepare_reload_config(
        &mut self,
        config: EngineReloadConfig,
    ) -> Result<Box<dyn PreparedEngineReload + '_>, EngineError> {
        if config.backend != BackendKind::RvcCpuPipeline || config.tensorrt.is_some() {
            return Err(EngineError::InvalidConfig(
                "CPU backend changes require engine replacement".into(),
            ));
        }
        Ok(reload::prepared_reload(|| {}))
    }

    fn reset_stream(&mut self) -> Result<(), EngineError> {
        self.reset();
        Ok(())
    }

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
        if self.format != Some(format) {
            self.history.clear();
            self.format = Some(format);
        }
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
    fn reset_stream(&mut self) -> Result<(), EngineError> {
        self.status.ready();
        Ok(())
    }

    fn prepare_reload_config(
        &mut self,
        config: EngineReloadConfig,
    ) -> Result<Box<dyn PreparedEngineReload + '_>, EngineError> {
        if config.backend != BackendKind::Passthrough || config.tensorrt.is_some() {
            return Err(EngineError::InvalidConfig(
                "passthrough backend changes require engine replacement".into(),
            ));
        }
        Ok(reload::prepared_reload(|| {}))
    }

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
    fn portable_engines_recover_from_failed_controls_on_stream_reset() {
        for backend in [BackendKind::Passthrough, BackendKind::RvcCpuPipeline] {
            let mut engine = create_engine(EngineConfig::portable(
                backend,
                ModelDescriptor {
                    id: "fixture".into(),
                    display_name: "Fixture".into(),
                    architecture: "rvc-v2".into(),
                    sample_rate: 32000,
                    version: "2".into(),
                },
            ))
            .unwrap();
            let input = AudioChunk {
                format: AudioFormat::MONO_16K,
                samples: vec![0.1; 320],
                sequence: 0,
            };
            assert!(
                engine
                    .process(
                        &input,
                        ConversionConfig {
                            protect: f32::NAN,
                            ..Default::default()
                        }
                    )
                    .is_err()
            );
            assert_eq!(engine.status().state, EngineLifecycle::Failed);
            engine.reset_stream().unwrap();
            assert_eq!(engine.status().state, EngineLifecycle::Ready);
            assert!(engine.status().last_error.is_none());
            engine.process(&input, ConversionConfig::default()).unwrap();
            assert_eq!(engine.status().failures, 1);
        }
    }

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
            directml: None,
            cpu: None,
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
            architecture: "rvc-v1".into(),
            sample_rate: 32_000,
            version: "1".into(),
        };
        let mut engine = create_engine(EngineConfig {
            backend: BackendKind::RvcCpuPipeline,
            model,
            tensorrt: None,
            directml: None,
            cpu: None,
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
            directml: None,
            cpu: None,
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
            directml: None,
            cpu: None,
        });
        assert!(matches!(result, Err(EngineError::BackendUnavailable(_))));
    }
}
