//! Actual CPU graph engine using the common resampling, chunk and SOLA boundary.
use crate::{
    CpuEngineSpec, EngineError, EngineLifecycle, EngineStatus, InferenceEngine,
    audio_adapter::{
        RealtimeBufferState, StreamingOutputJoiner, StreamingWindow, interleave, to_mono,
    },
};
use std::time::Instant;
use vocal_cpu::{CpuConfig, CpuF0Config, CpuPipeline, CpuQuantization, CpuSources};
use vocal_types::{AudioChunk, AudioFormat, ConversionConfig, LatencyMetrics, ModelDescriptor};

#[derive(Clone, Debug)]
pub struct CpuStreamConfig {
    pub chunk_samples_16k: usize,
    pub crossfade_samples_16k: usize,
    pub sola_search_samples_16k: usize,
    /// Diagnostic startup muting. Normal sessions publish zero-padded context
    /// tails immediately so speaking just after Start does not lose audio.
    pub warmup: bool,
}
impl Default for CpuStreamConfig {
    fn default() -> Self {
        Self {
            chunk_samples_16k: 4000,
            crossfade_samples_16k: 800,
            sola_search_samples_16k: 160,
            warmup: false,
        }
    }
}
impl CpuStreamConfig {
    pub fn for_window(samples: usize) -> Self {
        if samples == 11200 {
            Self {
                chunk_samples_16k: 8192,
                crossfade_samples_16k: 1365,
                sola_search_samples_16k: 192,
                warmup: false,
            }
        } else {
            Self::default()
        }
    }
}
pub struct CpuEngine {
    pipeline: CpuPipeline,
    stream: CpuStreamConfig,
    window: StreamingWindow,
    buffers: RealtimeBufferState,
    joiner: StreamingOutputJoiner,
    speaker: usize,
    warmup: usize,
    format: Option<AudioFormat>,
    resampler: Option<crate::streaming_resampler::ModelInputResampler>,
    status: EngineStatus,
    metrics: LatencyMetrics,
}
fn cpu_error(e: impl std::fmt::Display) -> EngineError {
    EngineError::BackendUnavailable(format!("CPU graph: {e}"))
}
impl CpuEngine {
    pub fn from_spec(model: ModelDescriptor, spec: CpuEngineSpec) -> Result<Self, EngineError> {
        spec.execution.validate()?;
        let window_samples = spec.window_samples;
        if !matches!(window_samples, 11200 | 44960) {
            return Err(EngineError::InvalidConfig(
                "CPU streaming window must be 11200 or 44960 samples".into(),
            ));
        }
        let config = CpuConfig {
            threads: spec.execution.resolved_threads(),
            spin_wait_us: spec.execution.spin_wait_us,
            pin_to_performance_cores: spec.execution.pin_to_performance_cores,
            profile: spec.execution.profile,
            window_samples,
            skip_head_frames: if window_samples == 44960 { 250 } else { 8 },
            f0: if window_samples == 44960 {
                CpuF0Config::rvc_realtime()
            } else {
                // The 8192-sample chunk is not an integral RMVPE hop advance.
                CpuF0Config::FullWindow
            },
            ..CpuConfig::default()
        };
        Self::from_sources(
            model,
            &CpuSources {
                synth: spec.synth,
                hubert: spec.hubert,
                rmvpe: spec.rmvpe,
            },
            config,
            CpuStreamConfig::for_window(window_samples),
            spec.speaker_id,
        )
    }
    pub fn from_sources(
        model: ModelDescriptor,
        sources: &CpuSources,
        config: CpuConfig,
        stream: CpuStreamConfig,
        speaker: usize,
    ) -> Result<Self, EngineError> {
        Self::from_sources_impl(model, sources, config, stream, speaker, None)
    }
    /// Explicit, calibrated precision recipe. Calibration must belong to these
    /// models and graph settings; ordinary backend dispatch stays FP32.
    pub fn from_sources_quantized(
        model: ModelDescriptor,
        sources: &CpuSources,
        config: CpuConfig,
        stream: CpuStreamConfig,
        speaker: usize,
        recipe: &CpuQuantization,
    ) -> Result<Self, EngineError> {
        Self::from_sources_impl(model, sources, config, stream, speaker, Some(recipe))
    }
    fn from_sources_impl(
        model: ModelDescriptor,
        sources: &CpuSources,
        config: CpuConfig,
        stream: CpuStreamConfig,
        speaker: usize,
        recipe: Option<&CpuQuantization>,
    ) -> Result<Self, EngineError> {
        if let CpuF0Config::Rolling {
            advance_samples, ..
        } = config.f0
        {
            if advance_samples != stream.chunk_samples_16k {
                return Err(EngineError::InvalidConfig(
                    "CPU rolling F0 advance must equal the playback chunk".into(),
                ));
            }
        }
        model
            .validate()
            .map_err(|e| EngineError::InvalidConfig(e.to_string()))?;
        let package = vocal_model::load_voice_cache_shared(&sources.synth).map_err(cpu_error)?;
        let c = package.validate_rvc_synth().map_err(cpu_error)?;
        let selected_version =
            vocal_model::RvcVersion::from_labels(&model.version, &model.architecture)
                .map_err(cpu_error)?;
        let package_version = package.manifest.rvc_version().map_err(cpu_error)?;
        if selected_version != package_version
            || c.feature_width as usize != selected_version.feature_width()
        {
            return Err(EngineError::InvalidConfig(
                "CPU Synth package and selected model disagree on RVC v1/v2 feature contract"
                    .into(),
            ));
        }
        if package.manifest.id != model.id
            || c.sample_rate != model.sample_rate
            || speaker >= c.speakers as usize
        {
            return Err(EngineError::InvalidConfig(
                "CPU graph sources or speaker do not match the selected model".into(),
            ));
        }
        drop(package);
        let pipeline = match recipe {
            Some(recipe) => CpuPipeline::from_sources_quantized(sources, config, recipe),
            None => CpuPipeline::from_sources(sources, config),
        }
        .map_err(cpu_error)?;
        let tail = (pipeline.frames - pipeline.config.skip_head_frames) * 160;
        let required = stream
            .chunk_samples_16k
            .checked_add(stream.crossfade_samples_16k)
            .and_then(|n| n.checked_add(stream.sola_search_samples_16k));
        if stream.chunk_samples_16k == 0
            || required.is_none_or(|n| n > tail)
            || !pipeline.config.append_hubert_frame
        {
            return Err(EngineError::InvalidConfig(
                "CPU realtime tail must cover chunk + crossfade + SOLA".into(),
            ));
        }
        let warmup = if stream.warmup {
            pipeline
                .config
                .window_samples
                .div_ceil(stream.chunk_samples_16k)
                + 1
        } else {
            0
        };
        Ok(Self {
            window: StreamingWindow::new(pipeline.config.window_samples)?,
            joiner: StreamingOutputJoiner::new(
                16000,
                stream.crossfade_samples_16k,
                stream.sola_search_samples_16k,
            ),
            pipeline,
            stream,
            speaker,
            warmup,
            buffers: RealtimeBufferState::default(),
            format: None,
            resampler: None,
            status: EngineStatus {
                state: EngineLifecycle::Ready,
                ..EngineStatus::default()
            },
            metrics: LatencyMetrics::default(),
        })
    }
    fn block(
        &mut self,
        format: AudioFormat,
        samples: &[f32],
        sequence: u64,
        conversion: ConversionConfig,
    ) -> Result<AudioChunk, EngineError> {
        crate::validate_audio_samples(format, samples)?;
        self.validate_config(conversion)?;
        let start = Instant::now();
        if self.format.is_some_and(|old| old != format) {
            self.reset_stream()?;
        }
        if self.format != Some(format) {
            self.resampler = Some(crate::streaming_resampler::ModelInputResampler::new(
                format.sample_rate,
            )?);
            self.joiner = StreamingOutputJoiner::new(
                format.sample_rate,
                self.stream.crossfade_samples_16k,
                self.stream.sola_search_samples_16k,
            );
            self.format = Some(format);
        }
        let mono = to_mono(format, samples);
        let internal = self.resampler.as_mut().unwrap().process(&mono)?;
        self.buffers.append_input(&internal, &mono);
        let source_chunk = crate::pipeline::resampled_len(
            self.stream.chunk_samples_16k,
            16000,
            format.sample_rate,
        );
        while let Some((internal, _source)) = self
            .buffers
            .take_input_chunk(self.stream.chunk_samples_16k, source_chunk)
        {
            let wave = self.window.push(&internal)?;
            let result = self
                .pipeline
                .process(wave, conversion, self.speaker, false)
                .map_err(cpu_error)?;
            tracing::debug!(
                hubert_ms = result.feature_ms,
                rmvpe_ms = result.pitch_ms,
                synth_ms = result.synth_ms,
                total_ms = result.total_ms,
                "CPU realtime stages"
            );
            if self.pipeline.config.profile {
                let mut timings = result.timings.clone();
                timings.sort_by(|a, b| b.ms.total_cmp(&a.ms));
                for node in timings.iter().take(10) {
                    tracing::info!(stage = %node.stage, operator = %node.operator, name = %node.name, ms = node.ms, "CPU slow operator");
                }
            }
            let output = if self.warmup > 0 {
                self.warmup -= 1;
                vec![0.; source_chunk]
            } else {
                let mut converted = crate::pipeline::resample_linear(
                    &result.audio,
                    self.pipeline.sample_rate,
                    format.sample_rate,
                );
                if converted.len() < self.joiner.required_samples(source_chunk) {
                    return Err(EngineError::InvalidState(
                        "CPU decoder output is shorter than streaming tail".into(),
                    ));
                }
                crate::streaming_output::apply_realtime_rms_mix(
                    crate::streaming_output::decoder_source_tail(
                        wave,
                        result.audio.len(),
                        self.pipeline.sample_rate,
                    ),
                    16000,
                    &mut converted,
                    format.sample_rate,
                    conversion.rms_mix_rate,
                );
                self.joiner.process(&converted, source_chunk)
            };
            self.buffers.append_output(&output);
        }
        let output = self.buffers.take_output(mono.len());
        self.metrics.input_frames += mono.len() as u64;
        self.metrics.output_frames += output.len() as u64;
        self.metrics.processing_micros += start.elapsed().as_micros() as u64;
        Ok(AudioChunk {
            format,
            samples: interleave(&output, format.channels),
            sequence,
        })
    }
}
impl InferenceEngine for CpuEngine {
    fn preferred_batch_frames(&self, format: AudioFormat) -> usize {
        crate::pipeline::resampled_len(self.stream.chunk_samples_16k, 16000, format.sample_rate)
    }
    fn reset_stream(&mut self) -> Result<(), EngineError> {
        self.pipeline.reset_stream();
        self.window.reset();
        self.buffers.reset();
        self.joiner.reset();
        self.resampler = None;
        self.format = None;
        self.warmup = if self.stream.warmup {
            self.pipeline
                .config
                .window_samples
                .div_ceil(self.stream.chunk_samples_16k)
                + 1
        } else {
            0
        };
        self.status.ready();
        Ok(())
    }
    fn validate_config(&self, conversion: ConversionConfig) -> Result<(), EngineError> {
        crate::validate_conversion(conversion)?;
        if conversion.index_ratio > 0. && !self.pipeline.has_retrieval_index() {
            return Err(EngineError::InvalidConfig(
                "CPU retrieval requested without an index".into(),
            ));
        }
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
            .map_err(|e| EngineError::InvalidState(e.into()))?;
        let result = self.block(format, samples, sequence, config);
        match &result {
            Ok(_) => self.status.success(),
            Err(e) => self.status.failure(e.to_string()),
        };
        result
    }
    fn metrics(&self) -> LatencyMetrics {
        self.metrics
    }
    fn status(&self) -> EngineStatus {
        self.status.clone()
    }
}
