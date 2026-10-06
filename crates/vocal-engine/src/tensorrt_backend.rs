//! Concrete [`InferenceEngine`] adapter for the native TensorRT-RTX pipeline.
//!
//! `vocal-tensorrt` owns CUDA streams, device allocations, TensorRT contexts,
//! and graph replay.  This module owns only the application boundary: loading
//! the VFRM schema and pipeline manifest, converting interleaved audio to the
//! fixed internal window, and restoring the caller's format and duration.

use std::time::Instant;
use std::{fs, path::Path};

use vocal_model::{ModelPackage, RetrievalIndex, RvcSynthConfig, load_cache_shared};
use vocal_types::{AudioChunk, AudioFormat, ConversionConfig, LatencyMetrics, ModelDescriptor};

use crate::{
    EngineError, EngineReloadConfig, InferenceEngine,
    audio_adapter::{
        RealtimeBufferState, StreamingOutputJoiner, StreamingWindow, apply_rms_mix, interleave,
        tail_with_length, to_internal_rate, to_mono,
    },
    config::{TensorRtEngineSpec, TensorRtReloadConfig},
    logging,
    state::{EngineLifecycle, EngineStatus},
};

use vocal_tensorrt::{
    DiagnosticsConfig, PipelineRuntimeConfig, RvcRealtimeConfig, RvcTrtPipeline,
    RvcTrtPipelineArtifacts, TensorRtConfig,
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
    retrieval: Option<RetrievalIndex>,
    window: StreamingWindow,
    synth_config: RvcSynthConfig,
    model_sample_rate: u32,
    speaker_id: i32,
    metrics: LatencyMetrics,
    status: EngineStatus,
    last_timings: Option<vocal_tensorrt::RvcTrtTimings>,
    output_joiner: StreamingOutputJoiner,
    realtime_config: RvcRealtimeConfig,
    stream_config: RvcRealtimeConfig,
    realtime_buffers: RealtimeBufferState,
    /// Rolling input level used by Applio's realtime output envelope. It
    /// suppresses decoder residue when the callback stream enters silence.
    input_level_buffer: Vec<f32>,
    /// Reference workers execute the model while the context window is being
    /// filled, but publish silence for these initial blocks. Keeping this
    /// counter outside the TensorRT pipeline also clears SOLA state exactly
    /// at a stream reset and avoids feeding zero-padded output into the joiner.
    warmup_remaining: usize,
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
        let retrieval = RetrievalIndex::from_package(package)
            .map_err(|error| backend_error("load FAISS retrieval index", error))?;
        let window_samples = backend
            .window_samples
            .unwrap_or_else(|| artifacts.profile.opt_samples.max(1) as usize);
        validate_window(&artifacts, window_samples)?;
        let pipeline = load_pipeline(&artifacts, &synth_config, &backend, retrieval.as_ref())?;
        let window = StreamingWindow::new(window_samples)?;
        let realtime_config = realtime_config_for_window(window_samples);
        let warmup_remaining = warmup_blocks_for_window(window_samples, realtime_config);
        let input_level_buffer = vec![0.0; realtime_level_window(realtime_config)];
        Ok(Self {
            pipeline,
            retrieval,
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
            output_joiner: StreamingOutputJoiner::new(
                16_000,
                realtime_config.crossfade_samples_16k,
                realtime_config.sola_search_samples_16k,
            ),
            realtime_config,
            stream_config: realtime_config,
            realtime_buffers: RealtimeBufferState::default(),
            input_level_buffer,
            warmup_remaining,
        })
    }

    pub fn reset(&mut self) -> Result<(), EngineError> {
        self.window.reset();
        self.output_joiner.reset();
        self.realtime_buffers.reset();
        self.input_level_buffer.fill(0.0);
        self.warmup_remaining =
            warmup_blocks_for_window(self.window.target_samples(), self.realtime_config);
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
        let pipeline = load_pipeline(
            &artifacts,
            &self.synth_config,
            &backend,
            self.retrieval.as_ref(),
        )?;
        let window = StreamingWindow::new(window_samples)?;
        let realtime_config = realtime_config_for_window(window_samples);
        self.pipeline = pipeline;
        self.window = window;
        self.realtime_config = realtime_config;
        self.stream_config = realtime_config;
        self.output_joiner = StreamingOutputJoiner::new(
            16_000,
            realtime_config.crossfade_samples_16k,
            realtime_config.sola_search_samples_16k,
        );
        self.realtime_buffers.reset();
        self.input_level_buffer = vec![0.0; realtime_level_window(realtime_config)];
        self.warmup_remaining = warmup_blocks_for_window(window_samples, realtime_config);
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
        validate_native_conversion(config, self.pipeline.has_retrieval_index())?;
        self.pipeline
            .set_retrieval_ratio(config.index_ratio)
            .map_err(|error| backend_error("set FAISS retrieval ratio", error))?;
        let started = Instant::now();
        let mono = to_mono(format, samples);
        let internal = to_internal_rate(&mono, format.sample_rate);
        let output_mono = if self.pipeline.config.uses_f0 {
            self.realtime_buffers.append_input(&internal, &mono);
            let target_internal = self.stream_config.chunk_samples_16k.max(160);
            let target_source =
                crate::pipeline::resampled_len(target_internal, 16_000, format.sample_rate);
            let mut chunk_count = 0usize;
            while let Some((chunk_internal, chunk_source)) = self
                .realtime_buffers
                .take_input_chunk(target_internal, target_source)
            {
                let converted = self.infer_realtime_chunk(
                    format,
                    &chunk_internal,
                    &chunk_source,
                    sequence.saturating_add(chunk_count as u64),
                    config,
                )?;
                self.realtime_buffers.append_output(&converted);
                chunk_count = chunk_count.saturating_add(1);
            }
            self.realtime_buffers.take_output(mono.len())
        } else {
            self.infer_non_realtime_chunk(format, &internal, &mono, sequence, config)?
        };
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

    fn infer_realtime_chunk(
        &mut self,
        format: AudioFormat,
        internal: &[f32],
        source: &[f32],
        sequence: u64,
        config: ConversionConfig,
    ) -> Result<Vec<f32>, EngineError> {
        let advance_frames = internal.len().div_ceil(160).clamp(1, 1023);
        let input_rms = self.update_input_level(internal);
        let window = self.window.push(internal)?;
        let output = self
            .pipeline
            .process_realtime(&window, self.speaker_id, advance_frames)
            .map_err(|error| backend_error("TensorRT inference", error))?;
        self.last_timings = Some(output.timings);
        let block_frames =
            crate::pipeline::resampled_len(internal.len(), 16_000, format.sample_rate);
        // New Synth plans crop after the encoder and already return the
        // publishable tail. Legacy plans return the complete decode and still
        // need the host-side skip_head crop. Length is the ABI-compatible
        // discriminator because both plans report the full RVC frame count in
        // `synth_frames`.
        let full_audio_samples = output.rvc_frames.saturating_mul(
            self.pipeline
                .config
                .upsample_rates
                .iter()
                .try_fold(1usize, |value, rate| value.checked_mul(*rate as usize))
                .ok_or_else(|| {
                    EngineError::BackendUnavailable(
                        "TensorRT Synth upsample product overflow".into(),
                    )
                })?,
        );
        let model_audio = if output.audio.len() >= full_audio_samples {
            crop_realtime_audio(
                &output.audio,
                output.rvc_frames,
                self.realtime_config,
                full_audio_samples / output.rvc_frames.max(1),
            )
        } else {
            output.audio
        };
        // Applio keeps the CUDA pipeline hot while the circular context is
        // filling, then returns zero blocks for `ceil(window/chunk)+1`
        // iterations. Do not pass these blocks through SOLA: its overlap
        // buffer must remain zero until the first publishable block.
        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            return Ok(vec![0.0; block_frames]);
        }
        let mut converted = crate::pipeline::resample_linear(
            &model_audio,
            self.model_sample_rate,
            format.sample_rate,
        );
        // Applio scales the realtime decoder output by the rolling callback
        // RMS before resampling. Besides preserving perceived loudness, this
        // prevents a neural decoder from sustaining a mechanical residue
        // after the input has gone quiet.
        for sample in &mut converted {
            *sample *= input_rms;
        }
        if let Some(directory) = std::env::var_os("VOCAL_TRT_DUMP_STREAM_BLOCKS") {
            let path = std::path::PathBuf::from(directory).join(format!("block-{sequence:04}.f32"));
            let bytes = converted
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect::<Vec<_>>();
            let _ = std::fs::write(path, bytes);
        }
        if self.output_joiner.sample_rate() != format.sample_rate {
            self.output_joiner = StreamingOutputJoiner::new(
                format.sample_rate,
                self.stream_config.crossfade_samples_16k,
                self.stream_config.sola_search_samples_16k,
            );
        }
        let needed = self.output_joiner.required_samples(block_frames);
        let extended = head_with_length(&converted, needed);
        let mut output = self.output_joiner.process(&extended, block_frames);
        if config.rms_mix_rate < 1.0 && !source.is_empty() {
            apply_rms_mix(source, &mut output, config.rms_mix_rate);
        }
        Ok(output)
    }

    fn infer_non_realtime_chunk(
        &mut self,
        format: AudioFormat,
        internal: &[f32],
        source: &[f32],
        _sequence: u64,
        config: ConversionConfig,
    ) -> Result<Vec<f32>, EngineError> {
        let window = self.window.push(internal)?;
        let output = self
            .pipeline
            .process(&window, self.speaker_id)
            .map_err(|error| backend_error("TensorRT inference", error))?;
        self.last_timings = Some(output.timings);
        let converted = crate::pipeline::resample_linear(
            &output.audio,
            self.model_sample_rate,
            format.sample_rate,
        );
        let mut result = tail_with_length(&converted, source.len());
        if config.rms_mix_rate < 1.0 {
            apply_rms_mix(source, &mut result, config.rms_mix_rate);
        }
        Ok(result)
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

fn realtime_config_for_window(window_samples: usize) -> RvcRealtimeConfig {
    match window_samples {
        11_200 => RvcRealtimeConfig::voice_changer(),
        35_520 => RvcRealtimeConfig::rvc_webui_legacy(),
        44_960 => RvcRealtimeConfig::applio(),
        // Custom profiles still use the current WebUI/Applio overlap policy.
        // The model-specific extra context is derived from the profile's
        // output frame count below, so this fallback remains safe for a
        // deployment with a non-standard fixed profile.
        _ => RvcRealtimeConfig::applio(),
    }
}

fn warmup_blocks_for_window(window_samples: usize, realtime: RvcRealtimeConfig) -> usize {
    let chunk = realtime.chunk_samples_16k.max(1);
    window_samples.div_ceil(chunk).saturating_add(1)
}

fn realtime_level_window(realtime: RvcRealtimeConfig) -> usize {
    realtime
        .chunk_samples_16k
        .saturating_add(realtime.crossfade_samples_16k)
        .max(1)
}

impl TensorRtEngine {
    fn update_input_level(&mut self, internal: &[f32]) -> f32 {
        let window = self.input_level_buffer.len();
        if internal.len() >= window {
            self.input_level_buffer
                .copy_from_slice(&internal[internal.len() - window..]);
        } else {
            self.input_level_buffer.copy_within(internal.len().., 0);
            let start = window - internal.len();
            self.input_level_buffer[start..].copy_from_slice(internal);
        }
        let mean_square = self
            .input_level_buffer
            .iter()
            .map(|sample| sample * sample)
            .sum::<f32>()
            / window as f32;
        mean_square.sqrt()
    }
}

fn crop_realtime_audio(
    audio: &[f32],
    rvc_frames: usize,
    realtime: RvcRealtimeConfig,
    upsample: usize,
) -> Vec<f32> {
    let skip_head = realtime.extra_samples_16k / 160;
    let return_frames = rvc_frames.saturating_sub(skip_head).max(1);
    let start = rvc_frames
        .saturating_sub(return_frames)
        .saturating_mul(upsample);
    let expected = return_frames.saturating_mul(upsample);
    let mut output = vec![0.0; expected];
    if start < audio.len() {
        let copied = (audio.len() - start).min(expected);
        output[..copied].copy_from_slice(&audio[start..start + copied]);
    }
    output
}

fn head_with_length(samples: &[f32], frames: usize) -> Vec<f32> {
    if samples.len() >= frames {
        return samples[..frames].to_vec();
    }
    let mut output = samples.to_vec();
    output.resize(frames, 0.0);
    output
}

fn load_pipeline(
    artifacts: &RvcTrtPipelineArtifacts,
    synth_config: &RvcSynthConfig,
    backend: &TensorRtBackendConfig,
    retrieval: Option<&RetrievalIndex>,
) -> Result<RvcTrtPipeline, EngineError> {
    RvcTrtPipeline::load_with_runtime_config_and_retrieval(
        artifacts,
        synth_config.clone(),
        backend.runtime.clone(),
        backend.diagnostics.clone(),
        retrieval,
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

fn validate_native_conversion(
    config: ConversionConfig,
    has_retrieval_index: bool,
) -> Result<(), EngineError> {
    crate::validate_conversion(config)?;
    let default = ConversionConfig::default();
    if config.pitch_semitones != 0.0 {
        return Err(EngineError::InvalidConfig(
            "TensorRT backend currently requires pitch_semitones = 0; apply pitch in the native frontend profile before enabling it".into(),
        ));
    }
    if config.index_ratio != 0.0 && !has_retrieval_index {
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

#[cfg(test)]
mod tests {
    use super::{crop_realtime_audio, realtime_config_for_window, warmup_blocks_for_window};

    #[test]
    fn realtime_warmup_matches_reference_window_chunks() {
        assert_eq!(
            warmup_blocks_for_window(11_200, realtime_config_for_window(11_200)),
            3
        );
        assert_eq!(
            warmup_blocks_for_window(44_960, realtime_config_for_window(44_960)),
            13
        );
    }

    #[test]
    fn host_crop_keeps_the_reference_tail_length() {
        let audio: Vec<_> = (0..70 * 400).map(|sample| sample as f32).collect();
        let cropped = crop_realtime_audio(&audio, 70, realtime_config_for_window(11_200), 400);
        assert_eq!(cropped.len(), 62 * 400);
        assert_eq!(cropped[0], 8.0 * 400.0);

        let audio: Vec<_> = (0..281 * 400).map(|sample| sample as f32).collect();
        let cropped = crop_realtime_audio(&audio, 281, realtime_config_for_window(44_960), 400);
        assert_eq!(cropped.len(), 31 * 400);
        assert_eq!(cropped[0], 250.0 * 400.0);
    }
}
