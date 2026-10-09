//! Native graph adapters for the common offline voice-conversion contract.
use super::{OfflineSegmentOutput, OfflineVoicePipeline};
use crate::{BackendKind, EngineError};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
use vocal_types::ConversionConfig;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OfflineModelSources {
    pub synth: PathBuf,
    pub hubert: PathBuf,
    pub rmvpe: PathBuf,
}
impl OfflineModelSources {
    pub fn from_bundle(
        model: impl AsRef<Path>,
        root: impl AsRef<Path>,
    ) -> Result<Self, EngineError> {
        let synth = vocal_model::resolve_model_path(model).map_err(backend_error)?;
        let [hubert, rmvpe] =
            vocal_model::materialize_acoustic_sources(&synth, root).map_err(backend_error)?;
        Ok(Self {
            synth,
            hubert,
            rmvpe,
        })
    }
    pub fn default_cache_root() -> PathBuf {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("vocal-fast")
            .join("cache")
            .join("offline")
    }
}
fn backend_error(e: impl std::fmt::Display) -> EngineError {
    EngineError::BackendUnavailable(format!("offline inference: {e}"))
}

fn trace_base(
    root: &Option<PathBuf>,
    sequence: usize,
    waveform: &[f32],
    metadata: serde_json::Value,
) -> Result<Option<PathBuf>, EngineError> {
    let Some(root) = root else { return Ok(None) };
    let path = root.join(format!("{sequence:04}"));
    std::fs::create_dir_all(&path).map_err(backend_error)?;
    trace_f32(&path, "waveform", waveform)?;
    std::fs::write(
        path.join("metadata.json"),
        serde_json::to_vec_pretty(&metadata).map_err(backend_error)?,
    )
    .map_err(backend_error)?;
    Ok(Some(path))
}
fn trace_f32(path: &Path, name: &str, values: &[f32]) -> Result<(), EngineError> {
    let bytes: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path.join(format!("{name}.bin")), bytes).map_err(backend_error)
}

#[cfg(feature = "cpu-graph")]
pub struct CpuOfflinePipeline {
    sources: OfflineModelSources,
    pub config: vocal_cpu::CpuConfig,
    pub trace_directory: Option<PathBuf>,
    pipeline: Option<vocal_cpu::CpuPipeline>,
    sample_rate: u32,
}
#[cfg(feature = "cpu-graph")]
impl CpuOfflinePipeline {
    pub fn new(
        sources: OfflineModelSources,
        config: vocal_cpu::CpuConfig,
    ) -> Result<Self, EngineError> {
        config.validate().map_err(backend_error)?;
        let package =
            vocal_model::load_voice_cache_shared(&sources.synth).map_err(backend_error)?;
        Ok(Self {
            sample_rate: package.manifest.sample_rate,
            sources,
            config,
            trace_directory: None,
            pipeline: None,
        })
    }
}
#[cfg(feature = "cpu-graph")]
impl OfflineVoicePipeline for CpuOfflinePipeline {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::RvcCpuPipeline
    }
    fn output_sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn infer_segment(
        &mut self,
        waveform: &[f32],
        sequence: usize,
        conversion: ConversionConfig,
    ) -> Result<OfflineSegmentOutput, EngineError> {
        let start = Instant::now();
        if self
            .pipeline
            .as_ref()
            .is_none_or(|p| p.config.window_samples != waveform.len())
        {
            self.pipeline = None;
            let mut config = self.config.clone();
            config.window_samples = waveform.len();
            config.skip_head_frames = 0;
            config.append_hubert_frame = false;
            self.pipeline = Some(
                vocal_cpu::CpuPipeline::from_sources(
                    &vocal_cpu::CpuSources {
                        synth: self.sources.synth.clone(),
                        hubert: self.sources.hubert.clone(),
                        rmvpe: self.sources.rmvpe.clone(),
                    },
                    config,
                )
                .map_err(backend_error)?,
            );
        }
        let preparation_ms = start.elapsed().as_secs_f64() * 1000.;
        let p = self.pipeline.as_mut().unwrap();
        let result = p
            .process(waveform, conversion, 0, self.trace_directory.is_some())
            .map_err(backend_error)?;
        if let Some(path) = trace_base(
            &self.trace_directory,
            sequence,
            waveform,
            serde_json::json!({
                "backend":"cpu","sources":self.sources,"frames":p.frames,"hubert_frames":p.hubert_frames,
                "latent_channels":p.latent_channels,"sample_rate":p.sample_rate,"feature_width":p.feature_width,
                "controls":{"pitch_semitones":conversion.pitch_semitones,"auto_pitch":conversion.auto_pitch,"index_ratio":conversion.index_ratio,"protect":conversion.protect},
                "skip_head":0,"offline":true,"timings":{"hubert_ms":result.feature_ms,"rmvpe_ms":result.pitch_ms,"retrieval_ms":result.retrieval_ms,"synth_ms":result.synth_ms}
            }),
        )? {
            for (name, data) in &result.debug {
                std::fs::write(path.join(format!("{name}.bin")), data).map_err(backend_error)?;
            }
            std::fs::write(
                path.join("profile.json"),
                serde_json::to_vec_pretty(&result.timings).map_err(backend_error)?,
            )
            .map_err(backend_error)?;
        }
        tracing::info!(
            hubert_ms = result.feature_ms,
            rmvpe_ms = result.pitch_ms,
            retrieval_ms = result.retrieval_ms,
            synth_ms = result.synth_ms,
            "CPU graph stages"
        );
        Ok(OfflineSegmentOutput {
            audio: result.audio,
            preparation_ms,
            inference_ms: result.total_ms,
        })
    }
}

#[cfg(feature = "directml")]
pub struct DirectMlOfflinePipeline {
    sources: OfflineModelSources,
    pub config: vocal_directml::DirectMlConfig,
    pub trace_directory: Option<PathBuf>,
    pipeline: Option<vocal_directml::DirectMlOfflineGraph>,
    sample_rate: u32,
}
#[cfg(feature = "directml")]
impl DirectMlOfflinePipeline {
    pub fn new(
        sources: OfflineModelSources,
        config: vocal_directml::DirectMlConfig,
    ) -> Result<Self, EngineError> {
        let package =
            vocal_model::load_voice_cache_shared(&sources.synth).map_err(backend_error)?;
        Ok(Self {
            sample_rate: package.manifest.sample_rate,
            sources,
            config,
            trace_directory: None,
            pipeline: None,
        })
    }
}
#[cfg(feature = "directml")]
impl OfflineVoicePipeline for DirectMlOfflinePipeline {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::DirectMl
    }
    fn output_sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn infer_segment(
        &mut self,
        waveform: &[f32],
        sequence: usize,
        conversion: ConversionConfig,
    ) -> Result<OfflineSegmentOutput, EngineError> {
        let prepare = Instant::now();
        if self
            .pipeline
            .as_ref()
            .is_none_or(|p| p.config().window_samples != waveform.len())
        {
            self.pipeline = None;
            let mut config = self.config.clone();
            config.window_samples = waveform.len();
            config.skip_head_frames = 0;
            config.append_hubert_frame = false;
            config.f0 = vocal_directml::F0Config::FullWindow;
            config.debug_outputs = self.trace_directory.is_some();
            let package =
                vocal_model::load_voice_cache_shared(&self.sources.synth).map_err(backend_error)?;
            let has_index = vocal_model::RetrievalIndex::from_package(&package)
                .map_err(backend_error)?
                .is_some();
            let sources = vocal_directml::DirectMlSources {
                synth: self.sources.synth.clone(),
                hubert: self.sources.hubert.clone(),
                rmvpe: self.sources.rmvpe.clone(),
                config,
                enable_retrieval: has_index,
            };
            self.pipeline = Some(
                vocal_directml::DirectMlOfflineGraph::from_sources(&sources)
                    .map_err(backend_error)?,
            );
        }
        let mut preparation_ms = prepare.elapsed().as_secs_f64() * 1000.;
        let p = self.pipeline.as_mut().expect("prepared DML graph");
        let controls = vocal_directml::DirectMlControls {
            speaker_id: 0,
            pitch_semitones: conversion.pitch_semitones,
            auto_pitch: conversion.auto_pitch,
            index_ratio: conversion.index_ratio,
            protect: conversion.protect,
            valid_frames: p.frames,
        };
        let result = p.process(waveform, &controls).map_err(backend_error)?;
        preparation_ms += result.preparation_ms;
        let inference_ms = result.inference_ms;
        if let Some(path) = trace_base(
            &self.trace_directory,
            sequence,
            waveform,
            serde_json::json!({
                "backend": "directml", "sources": self.sources, "frames": p.frames, "hubert_frames": p.hubert_frames,
                "latent_channels": p.latent_channels, "sample_rate": p.sample_rate, "feature_width": p.feature_width,
                "controls": controls, "skip_head": 0, "offline": true,
            }),
        )? {
            for (name, bytes) in &result.debug.tensors {
                if [
                    "synth_random",
                    "synth_noise",
                    "feats",
                    "pitchf",
                    "pitch",
                    "hubert_raw",
                    "hubert_features",
                    "rmvpe_pitchf",
                    "audio",
                    "auto_pitch_offset",
                ]
                .contains(&name.as_str())
                {
                    std::fs::write(path.join(format!("{name}.bin")), bytes)
                        .map_err(backend_error)?;
                }
            }
        }
        Ok(OfflineSegmentOutput {
            audio: result.audio,
            preparation_ms,
            inference_ms,
        })
    }
}

#[cfg(feature = "tensorrt-rtx")]
pub struct TensorRtOfflinePipeline {
    sources: OfflineModelSources,
    root: PathBuf,
    pub config: vocal_tensorrt::TensorRtConfig,
    pub trace_directory: Option<PathBuf>,
    pipeline: Option<vocal_tensorrt::RvcTrtPipeline>,
    sample_rate: u32,
    fingerprint: u64,
}
#[cfg(feature = "tensorrt-rtx")]
impl TensorRtOfflinePipeline {
    /// The embedding process must start with NVIDIA_TF32_OVERRIDE=0 before
    /// initializing CUDA. RTX 1.6 otherwise produces measurable FP32 RMVPE
    /// drift despite the builder's disabled TF32 flag. The CLI owns a child
    /// process for this policy; a library never changes its host environment.
    pub fn new(
        sources: OfflineModelSources,
        root: PathBuf,
        mut config: vocal_tensorrt::TensorRtConfig,
    ) -> Result<Self, EngineError> {
        use std::hash::{Hash, Hasher};
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            return Err(EngineError::InvalidConfig(
                "aligned TensorRT offline inference requires a process started with NVIDIA_TF32_OVERRIDE=0 before CUDA initialization".into(),
            ));
        }
        config.build.dispatch_tune = vocal_tensorrt::DispatchTune::Auto;
        config.build.debug_outputs |= config.diagnostics.debug_graph_outputs;
        config.build.rmvpe_debug_outputs |= config.diagnostics.rmvpe_debug_outputs;
        let package =
            vocal_model::load_voice_cache_shared(&sources.synth).map_err(backend_error)?;
        config.build.rvc_version = package.manifest.rvc_version().map_err(backend_error)?;
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        "vocal-offline-full-v1".hash(&mut hash);
        // RTX embeds this process-wide NVIDIA setting in serialized plans and
        // refuses contexts built under a different value.
        std::env::var_os("NVIDIA_TF32_OVERRIDE").hash(&mut hash);
        for source in [&sources.synth, &sources.hubert, &sources.rmvpe] {
            let path = std::fs::canonicalize(source).map_err(backend_error)?;
            let stat = std::fs::metadata(&path).map_err(backend_error)?;
            path.hash(&mut hash);
            stat.len().hash(&mut hash);
            stat.modified().map_err(backend_error)?.hash(&mut hash);
        }
        serde_json::to_vec(&config.build)
            .map_err(backend_error)?
            .hash(&mut hash);
        Ok(Self {
            sample_rate: package.manifest.sample_rate,
            sources,
            root,
            config,
            trace_directory: None,
            pipeline: None,
            fingerprint: hash.finish(),
        })
    }
    fn prepare(&mut self, samples: usize) -> Result<(), EngineError> {
        use vocal_tensorrt::{RvcTrtPipeline, RvcTrtPipelineArtifacts, RvcTrtProfile};
        if self
            .pipeline
            .as_ref()
            .is_some_and(|p| p.profile.max_samples as usize == samples)
        {
            return Ok(());
        }
        self.pipeline = None;
        let dir = self
            .root
            .join("tensorrt")
            .join(format!("{:016x}-{samples}", self.fingerprint));
        let manifest = dir.join("pipeline.json");
        let package =
            vocal_model::load_voice_cache_shared(&self.sources.synth).map_err(backend_error)?;
        let cached = std::fs::read(&manifest)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<RvcTrtPipelineArtifacts>(&bytes).ok())
            .filter(|a| {
                a.profile.offline
                    && a.profile.max_samples as usize == samples
                    && package
                        .manifest
                        .rvc_version()
                        .is_ok_and(|v| a.rvc_version == v)
                    && [
                        &a.hubert_plan,
                        &a.rmvpe_plan,
                        &a.frontend_plan,
                        &a.synth_plan,
                    ]
                    .iter()
                    .all(|p| p.is_file())
            });
        let artifacts = if let Some(cached) = cached {
            cached
        } else {
            let profile = RvcTrtProfile::for_offline(samples).map_err(backend_error)?;
            tracing::info!(samples, path = %dir.display(), "building offline TensorRT plans");
            let mut config = self.config.clone();
            config.profile = profile;
            let artifacts = RvcTrtPipelineArtifacts::compile_with_config_and_progress(
                &config,
                &package,
                &self.sources.hubert,
                &self.sources.rmvpe,
                &dir,
                |p| {
                    tracing::info!(
                        stage = p.name,
                        finished = p.finished,
                        completed = p.completed,
                        total = p.total,
                        "offline plan preparation"
                    )
                },
            )
            .map_err(backend_error)?;
            let temp = manifest.with_extension("json.tmp");
            std::fs::write(
                &temp,
                serde_json::to_vec_pretty(&artifacts).map_err(backend_error)?,
            )
            .map_err(backend_error)?;
            std::fs::rename(&temp, &manifest).map_err(backend_error)?;
            artifacts
        };
        let index = vocal_model::RetrievalIndex::from_package(&package).map_err(backend_error)?;
        let mut diagnostics = self.config.diagnostics.clone();
        diagnostics.debug_intermediates = self.trace_directory.is_some();
        let pipeline = RvcTrtPipeline::load_with_runtime_config_and_retrieval(
            &artifacts,
            package.validate_rvc_synth().map_err(backend_error)?,
            self.config.runtime.clone(),
            diagnostics,
            index.as_ref(),
        )
        .map_err(backend_error)?;
        self.pipeline = Some(pipeline);
        Ok(())
    }
}
#[cfg(feature = "tensorrt-rtx")]
impl OfflineVoicePipeline for TensorRtOfflinePipeline {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::TensorRtRtx
    }
    fn output_sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn infer_segment(
        &mut self,
        waveform: &[f32],
        sequence: usize,
        conversion: ConversionConfig,
    ) -> Result<OfflineSegmentOutput, EngineError> {
        crate::validate_conversion(conversion)?;
        let prepare = Instant::now();
        self.prepare(waveform.len())?;
        let preparation_ms = prepare.elapsed().as_secs_f64() * 1000.;
        let p = self.pipeline.as_mut().expect("prepared TensorRT graph");
        p.set_retrieval_ratio(conversion.index_ratio)
            .map_err(backend_error)?;
        p.set_protect(conversion.protect).map_err(backend_error)?;
        p.set_pitch_controls(conversion.pitch_semitones, conversion.auto_pitch)
            .map_err(backend_error)?;
        let inference = Instant::now();
        let result = p.process(waveform, 0).map_err(backend_error)?;
        let inference_ms = inference.elapsed().as_secs_f64() * 1000.;
        if let Some(path) = trace_base(
            &self.trace_directory,
            sequence,
            waveform,
            serde_json::json!({
                "backend": "tensorrt", "sources": self.sources, "frames": result.synth_frames, "hubert_frames": result.hubert_frames,
                "latent_channels": p.config.latent_channels, "sample_rate": self.sample_rate, "feature_width": p.config.feature_width,
                "controls": conversion, "skip_head": result.synth_skip_head, "offline": true,
            }),
        )? {
            trace_f32(&path, "audio", &result.audio)?;
            for (name, values) in [
                ("synth_random", &result.synth_random),
                ("synth_noise", &result.synth_noise),
                ("feats", &result.rvc_features),
                ("pitchf", &result.aligned_pitchf),
                ("rmvpe_pitchf", &result.pitchf),
                ("hubert_features", &result.hubert_features),
                ("rmvpe_mel", &result.rmvpe_mel),
                ("rmvpe_bottleneck", &result.rmvpe_bottleneck),
                ("rmvpe_unet", &result.rmvpe_unet),
                ("rmvpe_gru", &result.rmvpe_gru),
                ("rmvpe_logits", &result.rmvpe_logits),
            ] {
                if let Some(values) = values {
                    trace_f32(&path, name, values)?;
                }
            }
            if let Some(pitch) = &result.pitch {
                std::fs::write(
                    path.join("pitch.bin"),
                    pitch
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .map_err(backend_error)?;
            }
        }
        Ok(OfflineSegmentOutput {
            audio: result.audio,
            preparation_ms,
            inference_ms,
        })
    }
}
