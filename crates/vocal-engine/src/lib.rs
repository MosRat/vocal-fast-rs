//! Inference boundary. TensorRT RTX integration is intentionally isolated from
//! the portable baseline so the SDK can be linked only on RTX build machines.

use std::time::Instant;
use thiserror::Error;
use vocal_types::{AudioChunk, ConversionConfig, LatencyMetrics, ModelDescriptor};

pub mod pipeline;
pub mod tensorrt;

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

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("engine backend is unavailable: {0}")]
    BackendUnavailable(String),
    #[error("invalid audio chunk: {0}")]
    InvalidInput(String),
}

pub trait InferenceEngine: Send {
    fn process(
        &mut self,
        input: &AudioChunk,
        config: ConversionConfig,
    ) -> Result<AudioChunk, EngineError>;
    fn metrics(&self) -> LatencyMetrics;
}

pub fn create_engine(config: EngineConfig) -> Result<Box<dyn InferenceEngine>, EngineError> {
    match config.backend {
        BackendKind::Passthrough => Ok(Box::new(PassthroughEngine::new(config.model))),
        BackendKind::RvcCpuPipeline => Ok(Box::new(PipelineEngine::new(config.model)?)),
        BackendKind::TensorRtRtx => Err(EngineError::BackendUnavailable(
            "TensorRT RTX adapter is a build-time integration point; enable the SDK adapter on an RTX host".into(),
        )),
    }
}

pub struct PipelineEngine {
    pipeline: pipeline::RvcPipeline,
    metrics: LatencyMetrics,
    history: Vec<f32>,
    last_timings: pipeline::PipelineTimings,
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
        })
    }

    pub fn from_pipeline(pipeline: pipeline::RvcPipeline) -> Self {
        Self {
            pipeline,
            metrics: LatencyMetrics::default(),
            history: Vec::new(),
            last_timings: pipeline::PipelineTimings::default(),
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
        if input.format.channels == 0
            || !input
                .samples
                .len()
                .is_multiple_of(input.format.channels as usize)
        {
            return Err(EngineError::InvalidInput(
                "interleaved audio does not match channel count".into(),
            ));
        }
        let started = Instant::now();
        let channels = input.format.channels as usize;
        let mono = if channels == 1 {
            input.samples.clone()
        } else {
            input
                .samples
                .chunks_exact(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                .collect()
        };
        let context = pipeline::resampled_len(1_600, 16_000, input.format.sample_rate).max(160);
        if self.history.is_empty() {
            self.history.resize(context, 0.0);
        }
        self.history.extend_from_slice(&mono);
        let output = self
            .pipeline
            .run(&self.history, input.format.sample_rate, config)?;
        self.last_timings = output.timings;
        let output_mono = pipeline::resample_linear(
            &output.samples,
            output.sample_rate,
            input.format.sample_rate,
        );
        let mut output_block = if output_mono.len() >= mono.len() {
            output_mono[output_mono.len() - mono.len()..].to_vec()
        } else {
            let mut block = vec![0.0; mono.len() - output_mono.len()];
            block.extend_from_slice(&output_mono);
            block
        };
        output_block.resize(mono.len(), 0.0);
        let keep_from = self.history.len().saturating_sub(context);
        self.history = self.history[keep_from..].to_vec();
        let mut samples = Vec::with_capacity(output_block.len() * channels);
        for sample in output_block {
            samples.extend(std::iter::repeat_n(sample, channels));
        }
        let output_frames = samples.len() / channels;
        self.metrics.input_frames += input.frames() as u64;
        self.metrics.output_frames += output_frames as u64;
        self.metrics.processing_micros += started.elapsed().as_micros() as u64;
        Ok(AudioChunk {
            format: input.format,
            samples,
            sequence: input.sequence,
        })
    }

    fn metrics(&self) -> LatencyMetrics {
        self.metrics
    }
}

pub fn create_pipeline_engine(pipeline: pipeline::RvcPipeline) -> Box<dyn InferenceEngine> {
    Box::new(PipelineEngine::from_pipeline(pipeline))
}

struct PassthroughEngine {
    metrics: LatencyMetrics,
}

impl PassthroughEngine {
    fn new(_model: ModelDescriptor) -> Self {
        Self {
            metrics: LatencyMetrics::default(),
        }
    }
}

impl InferenceEngine for PassthroughEngine {
    fn process(
        &mut self,
        input: &AudioChunk,
        _config: ConversionConfig,
    ) -> Result<AudioChunk, EngineError> {
        if input.format.channels == 0 {
            return Err(EngineError::InvalidInput(
                "channel count must be non-zero".into(),
            ));
        }
        let start = Instant::now();
        let mut output = input.clone();
        output.sequence = input.sequence;
        self.metrics.input_frames += input.frames() as u64;
        self.metrics.output_frames += output.frames() as u64;
        self.metrics.processing_micros += start.elapsed().as_micros() as u64;
        Ok(output)
    }

    fn metrics(&self) -> LatencyMetrics {
        self.metrics
    }
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
}
