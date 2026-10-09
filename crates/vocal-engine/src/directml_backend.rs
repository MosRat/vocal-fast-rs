//! Native DML execution with the shared streaming audio boundary.
use std::time::Instant;

use vocal_directml::{DirectMlControls, DirectMlPipeline, DirectMlSources, F0Config};
use vocal_types::{AudioChunk, AudioFormat, ConversionConfig, LatencyMetrics, ModelDescriptor};

use crate::{
    BackendKind, DirectMlEngineSpec, EngineError, EngineReloadConfig, InferenceEngine,
    audio_adapter::{
        RealtimeBufferState, StreamingOutputJoiner, StreamingWindow, interleave, to_mono,
    },
    logging,
    state::{EngineLifecycle, EngineStatus},
    streaming_output::{apply_realtime_rms_mix, decoder_source_tail},
};

#[derive(Clone, Debug)]
pub struct DirectMlStreamConfig {
    pub chunk_samples_16k: usize,
    pub crossfade_samples_16k: usize,
    pub sola_search_samples_16k: usize,
    /// Optional diagnostic muting; normal startup keeps the first utterance.
    pub warmup: bool,
}

impl Default for DirectMlStreamConfig {
    fn default() -> Self {
        Self {
            chunk_samples_16k: 4000,
            crossfade_samples_16k: 800,
            sola_search_samples_16k: 160,
            warmup: false,
        }
    }
}

impl DirectMlStreamConfig {
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

    fn validate(&self, pipeline: &DirectMlPipeline) -> Result<(), EngineError> {
        let required = self
            .chunk_samples_16k
            .checked_add(self.crossfade_samples_16k)
            .and_then(|n| n.checked_add(self.sola_search_samples_16k));
        let available = (pipeline.frames - pipeline.config().skip_head_frames) * 160;
        if let F0Config::Rolling {
            advance_samples, ..
        } = pipeline.config().f0
        {
            if advance_samples != self.chunk_samples_16k {
                return Err(EngineError::InvalidConfig(
                    "DML rolling F0 advance must equal the playback chunk".into(),
                ));
            }
        }
        if self.chunk_samples_16k == 0
            || self.chunk_samples_16k > pipeline.config().window_samples
            || required.is_none_or(|n| n > available)
        {
            return Err(EngineError::InvalidConfig(
                "DML decoder tail must cover chunk + crossfade + SOLA".into(),
            ));
        }
        Ok(())
    }
}

pub struct DirectMlEngine {
    pipeline: DirectMlPipeline,
    stream: DirectMlStreamConfig,
    window: StreamingWindow,
    buffers: RealtimeBufferState,
    joiner: StreamingOutputJoiner,
    warmup_remaining: usize,
    speaker: usize,
    format: Option<AudioFormat>,
    input_resampler: Option<crate::streaming_resampler::ModelInputResampler>,
    status: EngineStatus,
    metrics: LatencyMetrics,
    last_gpu_ms: Option<f64>,
}

impl DirectMlEngine {
    pub fn from_spec(
        model: ModelDescriptor,
        spec: DirectMlEngineSpec,
    ) -> Result<Self, EngineError> {
        let mut sources = DirectMlSources::load(&spec.sources_path).map_err(backend_error)?;
        if let Some(id) = spec.adapter_id {
            sources.config.adapter_id = Some(id);
        }
        let stream = DirectMlStreamConfig::for_window(sources.config.window_samples);
        Self::from_sources(model, &sources, stream, spec.speaker_id)
    }

    pub fn from_sources(
        model: ModelDescriptor,
        sources: &DirectMlSources,
        stream: DirectMlStreamConfig,
        speaker: usize,
    ) -> Result<Self, EngineError> {
        model
            .validate()
            .map_err(|e| EngineError::InvalidConfig(e.to_string()))?;
        let package =
            vocal_model::load_voice_cache_shared(&sources.synth).map_err(backend_error)?;
        let selected_version =
            vocal_model::RvcVersion::from_labels(&model.version, &model.architecture)
                .map_err(backend_error)?;
        let package_version = package.manifest.rvc_version().map_err(backend_error)?;
        if selected_version != package_version {
            return Err(EngineError::InvalidConfig(
                "DirectML Synth package and selected model disagree on RVC v1/v2 feature contract"
                    .into(),
            ));
        }
        if package.manifest.id != model.id || package.manifest.sample_rate != model.sample_rate {
            return Err(EngineError::InvalidConfig(
                "DML Synth package does not match selected model".into(),
            ));
        }
        let pipeline = DirectMlPipeline::from_sources(sources).map_err(backend_error)?;
        if speaker >= pipeline.speakers {
            return Err(EngineError::InvalidConfig(
                "DML speaker ID out of range".into(),
            ));
        }
        stream.validate(&pipeline)?;
        let window = StreamingWindow::new(pipeline.config().window_samples)?;
        let warmup_remaining = if stream.warmup {
            pipeline
                .config()
                .window_samples
                .div_ceil(stream.chunk_samples_16k)
                + 1
        } else {
            0
        };
        let joiner = StreamingOutputJoiner::new(
            16000,
            stream.crossfade_samples_16k,
            stream.sola_search_samples_16k,
        );
        Ok(Self {
            pipeline,
            stream,
            window,
            buffers: RealtimeBufferState::default(),
            joiner,
            warmup_remaining,
            speaker,
            format: None,
            input_resampler: None,
            metrics: LatencyMetrics::default(),
            status: EngineStatus {
                state: EngineLifecycle::Ready,
                ..Default::default()
            },
            last_gpu_ms: None,
        })
    }

    pub fn last_gpu_ms(&self) -> Option<f64> {
        self.last_gpu_ms
    }

    pub fn reset(&mut self) -> Result<(), EngineError> {
        self.pipeline.reset().map_err(backend_error)?;
        self.window.reset();
        self.buffers.reset();
        self.joiner.reset();
        self.warmup_remaining = if self.stream.warmup {
            self.window
                .target_samples()
                .div_ceil(self.stream.chunk_samples_16k)
                + 1
        } else {
            0
        };
        self.last_gpu_ms = None;
        self.format = None;
        self.input_resampler = None;
        self.status.ready();
        Ok(())
    }

    fn infer(
        &mut self,
        format: AudioFormat,
        internal: &[f32],
        config: ConversionConfig,
    ) -> Result<Vec<f32>, EngineError> {
        let wave = self.window.push(internal)?;
        let controls = DirectMlControls {
            speaker_id: self.speaker,
            pitch_semitones: config.pitch_semitones,
            auto_pitch: config.auto_pitch,
            index_ratio: config.index_ratio,
            protect: config.protect,
            valid_frames: self.pipeline.frames,
        };
        let result = self
            .pipeline
            .process(wave, &controls)
            .map_err(backend_error)?;
        self.last_gpu_ms = Some(result.gpu_ms);
        let frames = crate::pipeline::resampled_len(internal.len(), 16000, format.sample_rate);
        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            return Ok(vec![0.; frames]);
        }
        let mut converted = crate::pipeline::resample_linear(
            &result.audio,
            self.pipeline.sample_rate,
            format.sample_rate,
        );
        let needed = self.joiner.required_samples(frames);
        if converted.len() < needed {
            return Err(EngineError::BackendUnavailable(format!(
                "DML output {} is shorter than streaming tail {needed}",
                converted.len()
            )));
        }
        apply_realtime_rms_mix(
            decoder_source_tail(wave, result.audio.len(), self.pipeline.sample_rate),
            16000,
            &mut converted,
            format.sample_rate,
            config.rms_mix_rate,
        );
        Ok(self.joiner.process(&converted, frames))
    }

    fn process_block(
        &mut self,
        format: AudioFormat,
        samples: &[f32],
        sequence: u64,
        config: ConversionConfig,
    ) -> Result<AudioChunk, EngineError> {
        crate::validate_audio_samples(format, samples)?;
        self.validate_config(config)?;
        let started = Instant::now();
        if self.format.is_some_and(|old| old != format) {
            self.reset()?;
        }
        if self.format != Some(format) || self.joiner.sample_rate() != format.sample_rate {
            self.joiner = StreamingOutputJoiner::new(
                format.sample_rate,
                self.stream.crossfade_samples_16k,
                self.stream.sola_search_samples_16k,
            );
            self.format = Some(format);
            self.input_resampler = Some(crate::streaming_resampler::ModelInputResampler::new(
                format.sample_rate,
            )?);
        }
        let mono = to_mono(format, samples);
        let internal = self
            .input_resampler
            .as_mut()
            .ok_or_else(|| {
                EngineError::InvalidState("DML input converter was not prepared".into())
            })?
            .process(&mono)?;
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
            let output = self.infer(format, &internal, config)?;
            self.buffers.append_output(&output);
        }
        let output = self.buffers.take_output(mono.len());
        self.metrics.input_frames += mono.len() as u64;
        self.metrics.output_frames += output.len() as u64;
        self.metrics.processing_micros += started.elapsed().as_micros() as u64;
        Ok(AudioChunk {
            format,
            samples: interleave(&output, format.channels),
            sequence,
        })
    }
}

impl InferenceEngine for DirectMlEngine {
    fn preferred_batch_frames(&self, format: AudioFormat) -> usize {
        crate::pipeline::resampled_len(self.stream.chunk_samples_16k, 16_000, format.sample_rate)
    }

    fn reset_stream(&mut self) -> Result<(), EngineError> {
        self.reset()
    }

    fn validate_config(&self, config: ConversionConfig) -> Result<(), EngineError> {
        crate::validate_conversion(config)?;
        if config.index_ratio > 0. && !self.pipeline.has_retrieval_index() {
            return Err(EngineError::InvalidConfig(
                "DML sources must enable retrieval for a nonzero index ratio".into(),
            ));
        }
        Ok(())
    }

    fn prepare_reload_config(
        &mut self,
        config: EngineReloadConfig,
    ) -> Result<Box<dyn crate::PreparedEngineReload + '_>, EngineError> {
        if config.backend != BackendKind::DirectMl || config.tensorrt.is_some() {
            return Err(EngineError::InvalidConfig(
                "DML backend changes require engine replacement".into(),
            ));
        }
        Ok(crate::reload::prepared_reload(|| {}))
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

fn backend_error(error: impl std::fmt::Display) -> EngineError {
    EngineError::BackendUnavailable(format!("DirectML: {error}"))
}
