//! Concrete [`InferenceEngine`] adapter for the native TensorRT-RTX pipeline.
//!
//! `vocal-tensorrt` owns CUDA streams, device allocations, TensorRT contexts,
//! and graph replay.  This module owns only the application boundary: loading
//! the VFRM schema and pipeline manifest, converting interleaved audio to the
//! fixed internal window, and restoring the caller's format and duration.

use std::time::Instant;
use std::{fs, path::Path};

use vocal_model::{ModelPackage, RvcSynthConfig, load_cache_shared};
use vocal_types::{AudioChunk, AudioFormat, ConversionConfig, LatencyMetrics, ModelDescriptor};

use crate::{
    EngineError, EngineReloadConfig, InferenceEngine,
    audio_adapter::{
        StreamingWindow, apply_rms_mix, from_model_rate, interleave, to_internal_rate, to_mono,
    },
    config::{TensorRtEngineSpec, TensorRtReloadConfig},
    logging,
    state::{EngineLifecycle, EngineStatus},
};

use vocal_tensorrt::{
    DiagnosticsConfig, PipelineRuntimeConfig, RvcTrtPipeline, RvcTrtPipelineArtifacts,
    TensorRtConfig,
};

/// Runtime-owned TensorRT backend configuration.  Build options stay in the
/// serialized pipeline manifest; this object controls execution policy and
/// the selected profile window.
#[derive(Clone, Debug)]
pub struct TensorRtBackendConfig {
    pub artifacts_path: std::path::PathBuf,
    pub model_package_path: std::path::PathBuf,
    pub speaker_id: i32,
    pub window_samples: Option<usize>,
    pub runtime: PipelineRuntimeConfig,
    pub diagnostics: DiagnosticsConfig,
}

impl TensorRtBackendConfig {
    pub fn from_spec(spec: TensorRtEngineSpec) -> Result<Self, EngineError> {
        let config = TensorRtConfig::from_env().map_err(config_error)?;
        Ok(Self {
            artifacts_path: spec.artifacts_path,
            model_package_path: spec.model_package_path,
            speaker_id: spec.speaker_id,
            window_samples: spec.window_samples,
            runtime: config.runtime,
            diagnostics: config.diagnostics,
        })
    }

    pub fn with_runtime(
        spec: TensorRtEngineSpec,
        runtime: PipelineRuntimeConfig,
        diagnostics: DiagnosticsConfig,
    ) -> Self {
        Self {
            artifacts_path: spec.artifacts_path,
            model_package_path: spec.model_package_path,
            speaker_id: spec.speaker_id,
            window_samples: spec.window_samples,
            runtime,
            diagnostics,
        }
    }

    pub fn for_package(
        artifacts_path: impl Into<std::path::PathBuf>,
        speaker_id: i32,
        window_samples: Option<usize>,
        runtime: PipelineRuntimeConfig,
        diagnostics: DiagnosticsConfig,
    ) -> Self {
        Self {
            artifacts_path: artifacts_path.into(),
            model_package_path: std::path::PathBuf::new(),
            speaker_id,
            window_samples,
            runtime,
            diagnostics,
        }
    }
}

/// A streaming engine backed by HuBERT, RMVPE, the native frontend, and the
/// native Synth graph.  All CUDA state remains inside `RvcTrtPipeline`; this
/// type is `Send` and can be moved into the runtime's worker thread.
pub struct TensorRtEngine {
    pipeline: RvcTrtPipeline,
    window: StreamingWindow,
    synth_config: RvcSynthConfig,
    model_sample_rate: u32,
    speaker_id: i32,
    metrics: LatencyMetrics,
    status: EngineStatus,
    last_timings: Option<vocal_tensorrt::RvcTrtTimings>,
}

impl TensorRtEngine {
    pub fn from_spec(
        model: ModelDescriptor,
        spec: TensorRtEngineSpec,
    ) -> Result<Self, EngineError> {
        let config = TensorRtBackendConfig::from_spec(spec)?;
        Self::from_backend_config(model, config)
    }

    pub fn from_backend_config(
        model: ModelDescriptor,
        backend: TensorRtBackendConfig,
    ) -> Result<Self, EngineError> {
        model
            .validate()
            .map_err(|error| EngineError::InvalidConfig(error.to_string()))?;
        let artifacts = read_artifacts(&backend.artifacts_path)?;
        let package = load_cache_shared(&backend.model_package_path)
            .map_err(|error| backend_error("load VFRM package", error))?;
        Self::from_package_and_artifacts(model, &package, artifacts, backend)
    }

    /// Construct from a package already owned by the runtime model loader.
    /// This avoids parsing a second copy when a manifest directory was loaded
    /// into memory and also keeps the model cache and TensorRT plan handoff in
    /// one ownership path.
    pub fn from_package(
        model: ModelDescriptor,
        package: &ModelPackage,
        backend: TensorRtBackendConfig,
    ) -> Result<Self, EngineError> {
        let artifacts = read_artifacts(&backend.artifacts_path)?;
        Self::from_package_and_artifacts(model, package, artifacts, backend)
    }

    fn from_package_and_artifacts(
        model: ModelDescriptor,
        package: &ModelPackage,
        artifacts: RvcTrtPipelineArtifacts,
        backend: TensorRtBackendConfig,
    ) -> Result<Self, EngineError> {
        let synth_config = package
            .validate_rvc_synth()
            .map_err(|error| backend_error("validate VFRM Synth package", error))?;
        if package.manifest.id != model.id {
            return Err(EngineError::InvalidConfig(format!(
                "VFRM package id {} does not match selected model {}",
                package.manifest.id, model.id
            )));
        }
        if synth_config.sample_rate != model.sample_rate {
            return Err(EngineError::InvalidConfig(format!(
                "Synth sample rate {} does not match model descriptor {}",
                synth_config.sample_rate, model.sample_rate
            )));
        }
        let window_samples = backend
            .window_samples
            .unwrap_or_else(|| artifacts.profile.opt_samples.max(1) as usize);
        validate_window(&artifacts, window_samples)?;
        let pipeline = load_pipeline(&artifacts, &synth_config, &backend)?;
        let window = StreamingWindow::new(window_samples)?;
        Ok(Self {
            pipeline,
            window,
            synth_config,
            model_sample_rate: model.sample_rate,
            speaker_id: backend.speaker_id,
            metrics: LatencyMetrics::default(),
            status: EngineStatus {
                state: EngineLifecycle::Ready,
                ..Default::default()
            },
            last_timings: None,
        })
    }

    pub fn reset(&mut self) -> Result<(), EngineError> {
        self.window.reset();
        self.pipeline
            .reset_streaming_state()
            .map_err(|error| backend_error("reset TensorRT pipeline", error))?;
        self.last_timings = None;
        Ok(())
    }

    pub fn last_timings(&self) -> Option<vocal_tensorrt::RvcTrtTimings> {
        self.last_timings
    }

    pub fn cuda_graph_stats(&self) -> vocal_tensorrt::RvcCudaGraphStats {
        self.pipeline.cuda_graph_stats()
    }

    pub fn window_samples(&self) -> usize {
        self.window.target_samples()
    }

    /// Rebuild the internal TensorRT pipeline while keeping the owning engine
    /// object stable. The caller must ensure the worker is idle; construction
    /// happens entirely in locals so a failed reload leaves the old pipeline
    /// usable.
    fn reload_pipeline(&mut self, config: TensorRtReloadConfig) -> Result<(), EngineError> {
        let env = TensorRtConfig::from_env().map_err(config_error)?;
        let graph_mode =
            vocal_tensorrt::GraphReplayMode::parse(&config.graph_mode).ok_or_else(|| {
                EngineError::InvalidConfig(format!(
                    "invalid TensorRT graph mode: {}",
                    config.graph_mode
                ))
            })?;
        let mut runtime = env.runtime;
        runtime.graph_mode = graph_mode;
        runtime.cuda_graph = config.cuda_graph;
        runtime.native_cuda_graph = config.native_cuda_graph;
        runtime.graph_cache_size = config.graph_cache_size;
        runtime.parallel_acoustic = config.parallel_acoustic;
        runtime.async_audio_d2h = config.async_audio_d2h;
        let backend = TensorRtBackendConfig::with_runtime(
            TensorRtEngineSpec {
                artifacts_path: config.artifacts_path,
                model_package_path: std::path::PathBuf::new(),
                speaker_id: config.speaker_id,
                window_samples: config.window_samples,
            },
            runtime,
            env.diagnostics,
        );
        let artifacts = read_artifacts(&backend.artifacts_path)?;
        let window_samples = backend
            .window_samples
            .unwrap_or_else(|| artifacts.profile.opt_samples.max(1) as usize);
        validate_window(&artifacts, window_samples)?;
        let pipeline = load_pipeline(&artifacts, &self.synth_config, &backend)?;
        let window = StreamingWindow::new(window_samples)?;
        self.pipeline = pipeline;
        self.window = window;
        self.speaker_id = backend.speaker_id;
        Ok(())
    }

    fn process_block(
        &mut self,
        format: AudioFormat,
        samples: &[f32],
        sequence: u64,
        config: ConversionConfig,
    ) -> Result<AudioChunk, EngineError> {
        validate_native_conversion(config)?;
        let started = Instant::now();
        let mono = to_mono(format, samples);
        let internal = to_internal_rate(&mono, format.sample_rate);
        let advance_frames = internal.len().div_ceil(160).clamp(1, 1023);
        let window = self.window.push(&internal)?;
        let output = if self.pipeline.config.uses_f0 {
            self.pipeline
                .process_realtime(&window, self.speaker_id, advance_frames)
        } else {
            self.pipeline.process(&window, self.speaker_id)
        }
        .map_err(|error| backend_error("TensorRT inference", error))?;
        self.last_timings = Some(output.timings);
        let mut output_mono =
            from_model_rate(&output.audio, self.model_sample_rate, format, mono.len());
        if config.rms_mix_rate < 1.0 {
            apply_rms_mix(&mono, &mut output_mono, config.rms_mix_rate);
        }
        if output_mono.iter().any(|sample| !sample.is_finite()) {
            return Err(EngineError::BackendUnavailable(
                "TensorRT inference produced non-finite audio".into(),
            ));
        }
        let samples = interleave(&output_mono, format.channels);
        let frames = output_mono.len() as u64;
        self.metrics.input_frames = self.metrics.input_frames.saturating_add(mono.len() as u64);
        self.metrics.output_frames = self.metrics.output_frames.saturating_add(frames);
        self.metrics.processing_micros = self
            .metrics
            .processing_micros
            .saturating_add(started.elapsed().as_micros() as u64);
        Ok(AudioChunk {
            format,
            samples,
            sequence,
        })
    }
}

impl InferenceEngine for TensorRtEngine {
    fn reload_config(&mut self, config: EngineReloadConfig) -> Result<(), EngineError> {
        if config.backend != crate::BackendKind::TensorRtRtx {
            return Err(EngineError::InvalidConfig(
                "TensorRT engine cannot switch backend in place".into(),
            ));
        }
        let spec = config.tensorrt.ok_or_else(|| {
            EngineError::InvalidConfig("TensorRT reload requires execution settings".into())
        })?;
        self.reload_pipeline(spec)
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
        if let Err(error) = crate::validate_audio_samples(format, samples) {
            self.status.failure(error.to_string());
            logging::failed(&error);
            return Err(error);
        }
        match self.process_block(format, samples, sequence, config) {
            Ok(output) => {
                self.status.success();
                Ok(output)
            }
            Err(error) => {
                self.status.failure(error.to_string());
                logging::failed(&error);
                Err(error)
            }
        }
    }

    fn metrics(&self) -> LatencyMetrics {
        self.metrics
    }

    fn status(&self) -> EngineStatus {
        self.status.clone()
    }
}

pub fn create_tensorrt_engine(
    model: ModelDescriptor,
    spec: TensorRtEngineSpec,
) -> Result<Box<dyn InferenceEngine>, EngineError> {
    Ok(Box::new(TensorRtEngine::from_spec(model, spec)?))
}

fn read_artifacts(path: &Path) -> Result<RvcTrtPipelineArtifacts, EngineError> {
    let bytes = fs::read(path).map_err(|error| {
        EngineError::BackendUnavailable(format!(
            "read TensorRT artifacts {}: {error}",
            path.display()
        ))
    })?;
    serde_json::from_slice(&bytes).map_err(|error| {
        EngineError::BackendUnavailable(format!(
            "parse TensorRT artifact manifest {}: {error}",
            path.display()
        ))
    })
}

fn load_pipeline(
    artifacts: &RvcTrtPipelineArtifacts,
    synth_config: &RvcSynthConfig,
    backend: &TensorRtBackendConfig,
) -> Result<RvcTrtPipeline, EngineError> {
    RvcTrtPipeline::load_with_runtime_config(
        artifacts,
        synth_config.clone(),
        backend.runtime.clone(),
        backend.diagnostics.clone(),
    )
    .map_err(|error| backend_error("load TensorRT pipeline", error))
}

fn validate_window(
    artifacts: &RvcTrtPipelineArtifacts,
    window_samples: usize,
) -> Result<(), EngineError> {
    let profile = &artifacts.profile;
    if window_samples < profile.min_samples.max(0) as usize
        || window_samples > profile.max_samples.max(0) as usize
        || window_samples % 160 != 0
    {
        return Err(EngineError::InvalidConfig(format!(
            "TensorRT window {} is outside profile [{}..={}] or is not 160-sample aligned",
            window_samples, profile.min_samples, profile.max_samples
        )));
    }
    Ok(())
}

fn validate_native_conversion(config: ConversionConfig) -> Result<(), EngineError> {
    crate::validate_conversion(config)?;
    let default = ConversionConfig::default();
    if config.pitch_semitones != 0.0 {
        return Err(EngineError::InvalidConfig(
            "TensorRT backend currently requires pitch_semitones = 0; apply pitch in the native frontend profile before enabling it".into(),
        ));
    }
    if config.index_ratio != 0.0 {
        return Err(EngineError::InvalidConfig(
            "TensorRT backend has no loaded feature index; index_ratio must be 0".into(),
        ));
    }
    if (config.protect - default.protect).abs() > f32::EPSILON {
        return Err(EngineError::InvalidConfig(
            "TensorRT backend uses the native protection policy; protect must remain 0.33".into(),
        ));
    }
    Ok(())
}

fn config_error(error: impl std::fmt::Display) -> EngineError {
    EngineError::BackendUnavailable(format!("read TensorRT runtime configuration: {error}"))
}

fn backend_error(stage: &str, error: impl std::fmt::Display) -> EngineError {
    EngineError::BackendUnavailable(format!("{stage}: {error}"))
}
