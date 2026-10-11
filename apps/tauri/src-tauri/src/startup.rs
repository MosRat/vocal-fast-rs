//! Shell bootstrap reads bounded metadata and probes devices, never tensor
//! payloads or engine caches. Resources remain readable while the actor loads.

use std::{
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::Instant,
};

use serde::Serialize;

use crate::{
    audio::AudioDeviceCatalog,
    config::ConfigStore,
    error::{GuiError, GuiResult},
    snapshot::{BootPhase, ModelSummary, UiBootStatus},
    state::AppState,
};

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StartupModel {
    pub path: String,
    pub summary: ModelSummary,
    pub components: vocal_model::ModelComponents,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StartupResources {
    pub audio_devices: Option<AudioDeviceCatalog>,
    pub processing_backends: vocal_runtime::ProcessingBackendCatalog,
    pub saved_model: Option<StartupModel>,
    pub notices: Vec<String>,
}

pub struct StartupTracker {
    boot: Mutex<BootState>,
    resources: OnceLock<StartupResources>,
}

struct BootState {
    revision: u64,
    phase: BootPhase,
    message: Option<String>,
    started_at: Instant,
    finished_ms: Option<u64>,
}

impl Default for StartupTracker {
    fn default() -> Self {
        Self {
            boot: Mutex::new(BootState {
                revision: 0,
                phase: BootPhase::Starting,
                message: None,
                started_at: Instant::now(),
                finished_ms: None,
            }),
            resources: OnceLock::new(),
        }
    }
}

impl StartupTracker {
    pub fn set(&self, phase: BootPhase, message: Option<String>) -> GuiResult<UiBootStatus> {
        let mut boot = self
            .boot
            .lock()
            .map_err(|_| GuiError::Internal("startup state lock poisoned".into()))?;
        if boot.phase == BootPhase::ShuttingDown
            || phase_order(phase) < phase_order(boot.phase)
            || (boot.phase == BootPhase::Failed && phase != BootPhase::ShuttingDown)
        {
            return Err(GuiError::InvalidState {
                state: format!("{:?}", boot.phase),
                action: format!("boot_{phase:?}"),
            });
        }
        boot.revision = boot.revision.saturating_add(1);
        boot.phase = phase;
        boot.message = message;
        if matches!(phase, BootPhase::Ready | BootPhase::Failed) && boot.finished_ms.is_none() {
            boot.finished_ms = Some(boot.elapsed_ms());
        }
        Ok(boot.snapshot())
    }

    pub fn status(&self) -> GuiResult<UiBootStatus> {
        Ok(self
            .boot
            .lock()
            .map_err(|_| GuiError::Internal("startup state lock poisoned".into()))?
            .snapshot())
    }

    pub fn resources(&self) -> Option<StartupResources> {
        self.resources.get().cloned()
    }

    pub fn finish(&self, resources: StartupResources) -> GuiResult<()> {
        self.resources
            .set(resources)
            .map_err(|_| GuiError::InvalidState {
                state: "startup_already_initialized".into(),
                action: "initialize_startup".into(),
            })
    }
}

impl BootState {
    fn elapsed_ms(&self) -> u64 {
        self.started_at
            .elapsed()
            .as_millis()
            .min(u128::from(u64::MAX)) as u64
    }

    fn snapshot(&self) -> UiBootStatus {
        UiBootStatus {
            revision: self.revision,
            phase: self.phase,
            message: self.message.clone(),
            elapsed_ms: self.finished_ms.unwrap_or_else(|| self.elapsed_ms()),
            recoverable: self.phase != BootPhase::ShuttingDown,
        }
    }
}

fn phase_order(phase: BootPhase) -> u8 {
    match phase {
        BootPhase::Starting => 0,
        BootPhase::RestoringConfig => 1,
        BootPhase::InitializingIo => 2,
        BootPhase::DetectingDevices => 3,
        BootPhase::Ready | BootPhase::Failed => 4,
        BootPhase::ShuttingDown => 5,
    }
}

/// Supervise the blocking bootstrap so panics and task failures still reveal a
/// recoverable UI. Window operations are dispatched on Tauri's main thread.
pub fn launch(
    app: tauri::AppHandle,
    state: AppState,
    config_path: PathBuf,
    cache_path: Option<PathBuf>,
) {
    tauri::async_runtime::spawn(async move {
        let worker = state.clone();
        let result = tauri::async_runtime::spawn_blocking(move || {
            initialize(&worker, config_path, cache_path)
        })
        .await
        .map_err(|error| GuiError::Internal(format!("startup worker failed: {error}")))
        .and_then(|result| result);
        if state
            .boot_status()
            .is_ok_and(|status| status.phase == BootPhase::ShuttingDown)
        {
            return;
        }
        if let Err(error) = result {
            state.fail_boot(&error);
        }
        let handle = app.clone();
        if let Err(error) = app.run_on_main_thread(move || crate::reveal_main(&handle)) {
            state.fail_boot(&GuiError::Internal(format!(
                "failed to reveal the main window: {error}"
            )));
        }
    });
}

/// Call on a blocking task. Model inspection reads only headers/directories;
/// the main window explicitly claims any pending restore after its first paint.
pub fn initialize(
    state: &AppState,
    config_path: PathBuf,
    cache_path: Option<PathBuf>,
) -> GuiResult<()> {
    state.set_boot_phase(
        BootPhase::RestoringConfig,
        Some("Restoring application settings…".into()),
    )?;
    state.set_config_store(ConfigStore::new(config_path))?;

    state.set_boot_phase(
        BootPhase::InitializingIo,
        Some("Preparing folders and model metadata…".into()),
    )?;
    let mut notices = Vec::new();
    if let Some(path) = cache_path {
        state.set_default_tensorrt_cache_directory(path)?;
    }
    let config = state.snapshot()?.config;
    if let Some(path) = &config.tensorrt.cache_directory {
        if let Err(error) = std::fs::create_dir_all(path) {
            notices.push(format!("Cannot prepare the cache folder {path}: {error}"));
        }
    }
    if let Some(path) = &config.model_directory {
        if !std::path::Path::new(path).is_dir() {
            notices.push(format!("The saved model library is unavailable: {path}"));
        }
    }
    let saved_model = config.model_path.as_ref().and_then(|path| {
        match vocal_model::inspect_model_artifact(path) {
            Ok(inspection) => {
                let request = vocal_runtime::model_library::ModelDependencyRequest {
                    model_path: PathBuf::from(path),
                    library_directory: config.model_directory.clone().map(PathBuf::from),
                    cache_directory: config.tensorrt.cache_directory.clone().map(PathBuf::from),
                    allow_download: false,
                    hubert_source: None,
                    rmvpe_source: None,
                };
                match vocal_runtime::model_library::inspect_dependencies(&request) {
                    Ok(status) if status.phase == vocal_runtime::model_library::DependencyPhase::Ready => {
                        let resolved = status
                            .resolved_model_path
                            .map(|value| value.display().to_string())
                            .unwrap_or_else(|| path.clone());
                        let resolved_inspection = vocal_model::inspect_model_artifact(&resolved)
                            .unwrap_or(inspection);
                        Some(StartupModel {
                            path: resolved,
                            summary: ModelSummary::from(&resolved_inspection.manifest),
                            components: resolved_inspection.components,
                        })
                    }
                    Ok(_) => {
                        notices.push("The saved voice needs HuBERT and RMVPE setup before it can be restored.".into());
                        None
                    }
                    Err(error) => {
                        notices.push(format!("The saved voice dependencies could not be inspected: {error}"));
                        None
                    }
                }
            }
            Err(error) => {
                notices.push(format!(
                    "The saved model is unavailable: {error}. Choose a model to continue."
                ));
                None
            }
        }
    });

    state.set_boot_phase(
        BootPhase::DetectingDevices,
        Some("Discovering audio devices and processing backends…".into()),
    )?;
    // Independent OS probes have their own thread/COM lifetimes. No AppState
    // lock or runtime actor is held while driver discovery executes.
    let (processing_backends, audio_devices) = std::thread::scope(|scope| -> GuiResult<_> {
        let backends = std::thread::Builder::new()
            .name("vocal-startup-backends".into())
            .spawn_scoped(scope, vocal_runtime::detect_processing_backends)
            .map_err(|error| GuiError::Internal(format!("cannot spawn backend probe: {error}")))?;
        let audio = std::thread::Builder::new()
            .name("vocal-startup-audio".into())
            .spawn_scoped(scope, || state.list_audio_devices())
            .map_err(|error| GuiError::Internal(format!("cannot spawn audio probe: {error}")))?;
        let processing_backends = backends
            .join()
            .map_err(|_| GuiError::Internal("processing backend probe panicked".into()))?;
        let audio_devices = audio
            .join()
            .map_err(|_| GuiError::Internal("audio device probe panicked".into()))?;
        Ok((processing_backends, audio_devices))
    })?;
    if let Some(notice) = state.initialize_processing_backend(processing_backends.clone())? {
        notices.push(notice);
    }
    let audio_devices = match audio_devices {
        Ok(devices) => Some(devices),
        Err(error) => {
            notices.push(error.to_string());
            None
        }
    };
    state.finish_startup(StartupResources {
        audio_devices,
        processing_backends,
        saved_model,
        notices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ready_elapsed_time_is_frozen() {
        let tracker = StartupTracker::default();
        let ready = tracker.set(BootPhase::Ready, None).unwrap();
        tracker.boot.lock().unwrap().started_at -= std::time::Duration::from_secs(10);
        assert_eq!(tracker.status().unwrap().elapsed_ms, ready.elapsed_ms);
    }

    #[test]
    fn late_worker_cannot_regress_a_ready_or_closing_boot() {
        let tracker = StartupTracker::default();
        let ready = tracker.set(BootPhase::Ready, None).unwrap();
        assert!(tracker.set(BootPhase::RestoringConfig, None).is_err());
        assert_eq!(tracker.status().unwrap().revision, ready.revision);
        let closing = tracker.set(BootPhase::ShuttingDown, None).unwrap();
        assert!(tracker.set(BootPhase::Ready, None).is_err());
        assert_eq!(tracker.status().unwrap().revision, closing.revision);
    }
}
