use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use serde::{Deserialize, Serialize};
use vocal_engine::BackendKind;
use vocal_types::{AudioFormat, AudioProcessingConfig, ConversionConfig, TypesError};

use crate::error::{RuntimeError, RuntimeResult, io_at};

pub const RUNTIME_CONFIG_SCHEMA_VERSION: u32 = 1;
static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(1);

/// GUI-owned TensorRT pipeline policy. Paths are kept as strings at the
/// runtime boundary so the same JSON can be consumed by the Tauri frontend;
/// the engine converts them to `PathBuf` only when it constructs an engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(default)]
pub struct TensorRtPipelineConfig {
    pub artifacts_path: Option<String>,
    pub hubert_source: Option<String>,
    pub rmvpe_source: Option<String>,
    pub cache_directory: Option<String>,
    pub profile: String,
    pub window_samples: Option<usize>,
    pub precision: String,
    pub hubert_precision: String,
    pub rmvpe_precision: String,
    pub synth_precision: String,
    pub graph_mode: String,
    pub cuda_graph: bool,
    pub native_cuda_graph: bool,
    pub graph_cache_size: usize,
    pub parallel_acoustic: bool,
    pub async_audio_d2h: bool,
    pub rmvpe_fft: String,
    pub synth_full_fp16: bool,
}

impl Default for TensorRtPipelineConfig {
    fn default() -> Self {
        Self {
            artifacts_path: None,
            hubert_source: None,
            rmvpe_source: None,
            cache_directory: None,
            profile: "reference".into(),
            window_samples: None,
            precision: "fp32".into(),
            hubert_precision: "fp16".into(),
            rmvpe_precision: "fp32".into(),
            synth_precision: "fp16".into(),
            graph_mode: "native_subgraphs".into(),
            cuda_graph: true,
            native_cuda_graph: true,
            graph_cache_size: 2,
            parallel_acoustic: false,
            async_audio_d2h: true,
            rmvpe_fft: "cufft".into(),
            synth_full_fp16: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RuntimeBackend {
    #[default]
    Passthrough,
    RvcCpu,
    TensorRtRtx,
}

/// Result of applying a configuration edit to an idle runtime.
///
/// `HotReload` changes values consumed on the next block without touching the
/// selected engine. `ReloadTensorRtPipeline` recreates TensorRT's internal
/// contexts/plans from an already published artifact manifest while keeping
/// the owning engine object stable. `PlanBuildRequired` means the JSON config
/// was accepted but the active TensorRT plans still need an explicit build.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ConfigUpdateKind {
    #[default]
    Unchanged,
    HotReload,
    RebuildEngine,
    ReloadTensorRtPipeline,
    PlanBuildRequired,
    LoadModelRequired,
}

impl RuntimeBackend {
    pub const fn engine_kind(self) -> BackendKind {
        match self {
            Self::Passthrough => BackendKind::Passthrough,
            Self::RvcCpu => BackendKind::RvcCpuPipeline,
            Self::TensorRtRtx => BackendKind::TensorRtRtx,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(default)]
pub struct RuntimeConfig {
    #[serde(default = "current_schema_version")]
    pub schema_version: u32,
    pub input_format: AudioFormat,
    pub output_format: AudioFormat,
    pub conversion: ConversionConfig,
    pub backend: RuntimeBackend,
    pub model_path: Option<String>,
    /// Optional library root used by controllers to discover model artifacts.
    /// It is intentionally independent from `model_path`, which remains the
    /// last selected artifact and is restored directly at startup.
    pub model_directory: Option<String>,
    /// TensorRT pipeline manifest produced by `compile-trt-pipeline`.
    pub tensorrt_artifacts_path: Option<String>,
    pub tensorrt_window_samples: Option<usize>,
    /// Typed TensorRT settings used by the GUI and persisted with the rest of
    /// the runtime configuration. The two legacy fields above remain accepted
    /// for config files written before this group was introduced.
    #[serde(default)]
    pub tensorrt: TensorRtPipelineConfig,
    pub speaker_id: i32,
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub block_frames: usize,
    pub queue_blocks: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            schema_version: RUNTIME_CONFIG_SCHEMA_VERSION,
            input_format: AudioFormat::MONO_48K,
            output_format: AudioFormat::MONO_48K,
            conversion: ConversionConfig::default(),
            backend: RuntimeBackend::Passthrough,
            model_path: None,
            model_directory: None,
            tensorrt_artifacts_path: None,
            tensorrt_window_samples: None,
            tensorrt: TensorRtPipelineConfig::default(),
            speaker_id: 0,
            input_device: None,
            output_device: None,
            block_frames: 480,
            queue_blocks: 3,
        }
    }
}

impl RuntimeConfig {
    pub fn validate(&self) -> RuntimeResult<()> {
        if self.schema_version > RUNTIME_CONFIG_SCHEMA_VERSION {
            return Err(RuntimeError::Config(format!(
                "configuration schema {} is newer than supported schema {}",
                self.schema_version, RUNTIME_CONFIG_SCHEMA_VERSION
            )));
        }
        AudioProcessingConfig {
            input: self.input_format,
            output: self.output_format,
            conversion: self.conversion,
        }
        .validate()
        .map_err(|error: TypesError| RuntimeError::Config(error.to_string()))?;
        if self.block_frames == 0 || self.queue_blocks == 0 {
            return Err(RuntimeError::Config(
                "block_frames and queue_blocks must be non-zero".into(),
            ));
        }
        if self.block_frames > 8192 || self.queue_blocks > 64 {
            return Err(RuntimeError::Config(
                "block_frames must be <= 8192 and queue_blocks must be <= 64".into(),
            ));
        }
        if self.input_format != self.output_format {
            return Err(RuntimeError::Config(
                "input and output formats must match for duplex streaming".into(),
            ));
        }
        let artifacts_path = self
            .tensorrt
            .artifacts_path
            .as_deref()
            .filter(|path| !path.is_empty())
            .or_else(|| {
                self.tensorrt_artifacts_path
                    .as_deref()
                    .filter(|path| !path.is_empty())
            });
        if self.backend == RuntimeBackend::TensorRtRtx && artifacts_path.is_none() {
            return Err(RuntimeError::Config(
                "TensorRT backend requires tensorrt_artifacts_path".into(),
            ));
        }
        if self
            .tensorrt
            .window_samples
            .or(self.tensorrt_window_samples)
            == Some(0)
        {
            return Err(RuntimeError::Config(
                "tensorrt_window_samples must be non-zero when provided".into(),
            ));
        }
        Ok(())
    }

    pub fn tensorrt_artifacts(&self) -> Option<&str> {
        self.tensorrt
            .artifacts_path
            .as_deref()
            .filter(|path| !path.is_empty())
            .or_else(|| {
                self.tensorrt_artifacts_path
                    .as_deref()
                    .filter(|path| !path.is_empty())
            })
    }

    pub fn tensorrt_window(&self) -> Option<usize> {
        self.tensorrt
            .window_samples
            .or(self.tensorrt_window_samples)
    }
}

/// Classify a validated configuration edit before mutating runtime state.
/// This is deliberately independent of the current engine implementation so
/// controllers can explain whether Save is a cheap idle update, an engine
/// replacement, or requires a fresh TensorRT artifact build.
pub fn classify_config_change(
    old: &RuntimeConfig,
    next: &RuntimeConfig,
    _model_loaded: bool,
) -> ConfigUpdateKind {
    if old.backend != next.backend {
        return ConfigUpdateKind::RebuildEngine;
    }
    if old.model_path != next.model_path {
        return ConfigUpdateKind::LoadModelRequired;
    }
    if old.backend == RuntimeBackend::TensorRtRtx {
        let artifacts_changed = old.tensorrt_artifacts() != next.tensorrt_artifacts();
        let runtime_changed = old.speaker_id != next.speaker_id
            || old.tensorrt_window() != next.tensorrt_window()
            || old.tensorrt.graph_mode != next.tensorrt.graph_mode
            || old.tensorrt.cuda_graph != next.tensorrt.cuda_graph
            || old.tensorrt.native_cuda_graph != next.tensorrt.native_cuda_graph
            || old.tensorrt.graph_cache_size != next.tensorrt.graph_cache_size
            || old.tensorrt.parallel_acoustic != next.tensorrt.parallel_acoustic
            || old.tensorrt.async_audio_d2h != next.tensorrt.async_audio_d2h;
        if artifacts_changed || runtime_changed {
            return ConfigUpdateKind::ReloadTensorRtPipeline;
        }
        if trt_plan_config_changed(old, next) {
            return ConfigUpdateKind::PlanBuildRequired;
        }
    }
    if old.input_format != next.input_format
        || old.output_format != next.output_format
        || old.conversion.pitch_semitones != next.conversion.pitch_semitones
        || old.conversion.index_ratio != next.conversion.index_ratio
        || old.conversion.protect != next.conversion.protect
        || old.conversion.rms_mix_rate != next.conversion.rms_mix_rate
        || old.input_device != next.input_device
        || old.output_device != next.output_device
        || old.block_frames != next.block_frames
        || old.queue_blocks != next.queue_blocks
        || old.model_directory != next.model_directory
    {
        return ConfigUpdateKind::HotReload;
    }
    ConfigUpdateKind::Unchanged
}

fn trt_plan_config_changed(old: &RuntimeConfig, next: &RuntimeConfig) -> bool {
    old.tensorrt.hubert_source != next.tensorrt.hubert_source
        || old.tensorrt.rmvpe_source != next.tensorrt.rmvpe_source
        || old.tensorrt.profile != next.tensorrt.profile
        || old.tensorrt.precision != next.tensorrt.precision
        || old.tensorrt.hubert_precision != next.tensorrt.hubert_precision
        || old.tensorrt.rmvpe_precision != next.tensorrt.rmvpe_precision
        || old.tensorrt.synth_precision != next.tensorrt.synth_precision
        || old.tensorrt.rmvpe_fft != next.tensorrt.rmvpe_fft
        || old.tensorrt.synth_full_fp16 != next.tensorrt.synth_full_fp16
}

#[derive(Clone, Debug)]
pub struct ConfigStore {
    path: PathBuf,
}

impl ConfigStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load(&self) -> RuntimeResult<Option<RuntimeConfig>> {
        if !self.path.is_file() {
            return Ok(None);
        }
        let bytes = fs::read(&self.path).map_err(|error| io_at(&self.path, error))?;
        let mut config: RuntimeConfig = serde_json::from_slice(&bytes)?;
        if config.schema_version == 0 {
            config.schema_version = RUNTIME_CONFIG_SCHEMA_VERSION;
        }
        config.validate()?;
        Ok(Some(config))
    }

    pub fn save(&self, config: &RuntimeConfig) -> RuntimeResult<()> {
        config.validate()?;
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|error| io_at(parent, error))?;
        }
        let bytes = serde_json::to_vec_pretty(config)?;
        let temp = self.temp_path();
        let result = (|| {
            let mut file = fs::File::create(&temp).map_err(|error| io_at(&temp, error))?;
            file.write_all(&bytes)
                .map_err(|error| io_at(&temp, error))?;
            file.sync_all().map_err(|error| io_at(&temp, error))?;
            replace_file(&temp, &self.path).map_err(|error| io_at(&self.path, error))?;
            sync_parent(self.path.parent())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result
    }

    fn temp_path(&self) -> PathBuf {
        let name = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("config.json");
        self.path.with_file_name(format!(
            ".{name}.{}.{}.tmp",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }
}

fn current_schema_version() -> u32 {
    RUNTIME_CONFIG_SCHEMA_VERSION
}

#[cfg(not(windows))]
fn replace_file(temp: &Path, target: &Path) -> std::io::Result<()> {
    fs::rename(temp, target)
}

#[cfg(windows)]
fn replace_file(temp: &Path, target: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };
    let source: Vec<u16> = temp.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = target.as_os_str().encode_wide().chain(Some(0)).collect();
    let flags = MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH;
    let status = unsafe { MoveFileExW(source.as_ptr(), destination.as_ptr(), flags) };
    (status != 0)
        .then_some(())
        .ok_or_else(std::io::Error::last_os_error)
}

fn sync_parent(parent: Option<&Path>) -> RuntimeResult<()> {
    #[cfg(unix)]
    if let Some(parent) = parent {
        fs::File::open(parent)
            .and_then(|file| file.sync_all())
            .map_err(|error| io_at(parent, error))?;
    }
    #[cfg(not(unix))]
    let _ = parent;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_idle_updates_without_rebuilding_engine() {
        let old = RuntimeConfig::default();
        let mut next = old.clone();
        next.conversion.pitch_semitones = 2.0;
        assert_eq!(
            classify_config_change(&old, &next, true),
            ConfigUpdateKind::HotReload
        );
    }

    #[test]
    fn classifies_model_and_backend_changes_separately() {
        let old = RuntimeConfig::default();
        let mut model = old.clone();
        model.model_path = Some("voice.vfrm".into());
        assert_eq!(
            classify_config_change(&old, &model, false),
            ConfigUpdateKind::LoadModelRequired
        );
        let mut backend = old.clone();
        backend.backend = RuntimeBackend::RvcCpu;
        assert_eq!(
            classify_config_change(&old, &backend, true),
            ConfigUpdateKind::RebuildEngine
        );
    }

    #[test]
    fn classifies_tensorrt_plan_and_runtime_changes() {
        let mut old = RuntimeConfig::default();
        old.backend = RuntimeBackend::TensorRtRtx;
        old.tensorrt.artifacts_path = Some("old/pipeline.json".into());
        let mut plan = old.clone();
        plan.tensorrt.profile = "realtime_44960".into();
        assert_eq!(
            classify_config_change(&old, &plan, true),
            ConfigUpdateKind::PlanBuildRequired
        );
        let mut runtime = old.clone();
        runtime.tensorrt.graph_mode = "whole_pipeline".into();
        assert_eq!(
            classify_config_change(&old, &runtime, true),
            ConfigUpdateKind::ReloadTensorRtPipeline
        );
    }

    #[test]
    fn config_store_round_trips_and_migrates_unversioned_data() {
        let path = std::env::temp_dir().join(format!(
            "vocal-runtime-config-{}-{}.json",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let store = ConfigStore::new(&path);
        let config = RuntimeConfig::default();
        store.save(&config).expect("save config");
        assert_eq!(
            store.load().expect("load config").unwrap().schema_version,
            1
        );
        let mut legacy = serde_json::to_value(config).expect("serialize config");
        legacy.as_object_mut().unwrap().remove("schemaVersion");
        fs::write(&path, serde_json::to_vec(&legacy).unwrap()).expect("write legacy config");
        assert_eq!(store.load().unwrap().unwrap().schema_version, 1);
        fs::remove_file(path).expect("remove config");
    }
}
