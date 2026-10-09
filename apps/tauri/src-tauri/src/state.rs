use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use vocal_runtime::{RuntimeBackend, RuntimeContext};
use vocal_types::{AudioChunk, AudioFormat, LifecycleState};

use crate::{
    audio::{AudioDeviceCatalog, AudioSession, AudioTelemetry},
    config::{ConfigStore, GuiConfig},
    error::{GuiError, GuiResult},
    events::{BackendEvent, ErrorReport, EventSink},
    preparation::{PreparationKind, PreparationOperation, PreparationPhase, PreparationTracker},
    snapshot::{
        BootPhase, TensorRtPipelinePhase, TensorRtPipelineStatus, UiBootStatus, UiEngineStatus,
        UiStatus,
    },
    startup::{StartupResources, StartupTracker},
};

use crate::tensorrt::PrepareTensorRtRequest;

pub use crate::snapshot::UiSnapshot;
pub use vocal_runtime::ModelCatalog;

struct RuntimeState {
    runtime: RuntimeContext,
    audio: Option<AudioSession>,
    last_error: Option<ErrorReport>,
    startup: Arc<StartupTracker>,
    pending_restore: Option<PendingModelRestore>,
    tensorrt_pipeline: TensorRtPipelineStatus,
    preparation: Arc<PreparationTracker>,
}

struct PendingModelRestore {
    path: String,
    revision: u64,
}

#[derive(Clone)]
pub struct AppState {
    inner: Arc<Mutex<RuntimeState>>,
    command_lock: Arc<Mutex<()>>,
    pub events: Arc<EventSink>,
    preparation: Arc<PreparationTracker>,
    startup: Arc<StartupTracker>,
    pub model_library: vocal_runtime::model_library::ModelLibrary,
}

impl AppState {
    pub fn new(config: GuiConfig) -> Self {
        let runtime =
            RuntimeContext::new(config).expect("default runtime configuration must validate");
        let events = Arc::new(EventSink::default());
        let preparation = Arc::new(PreparationTracker::new(events.clone()));
        let startup = Arc::new(StartupTracker::default());
        let library_events = events.clone();
        let model_library = vocal_runtime::model_library::ModelLibrary::new(move |status| {
            library_events.publish(BackendEvent::model_import(status))
        });
        Self {
            inner: Arc::new(Mutex::new(RuntimeState {
                runtime,
                audio: None,
                last_error: None,
                startup: startup.clone(),
                pending_restore: None,
                tensorrt_pipeline: TensorRtPipelineStatus::default(),
                preparation: preparation.clone(),
            })),
            command_lock: Arc::new(Mutex::new(())),
            events,
            preparation,
            startup,
            model_library,
        }
    }

    pub fn set_config_store(&self, store: ConfigStore) -> GuiResult<()> {
        let runtime = self.lock()?.runtime.clone();
        runtime.set_config_store(store.runtime_store())?;
        Ok(())
    }

    pub fn initialize_processing_backend(
        &self,
        catalog: vocal_runtime::ProcessingBackendCatalog,
    ) -> GuiResult<Option<String>> {
        let runtime = self.lock()?.runtime.clone();
        let notice = runtime.initialize_processing_backend_with_catalog(catalog)?;
        if let Some(message) = &notice {
            tracing::warn!(target: "vocal.gui", %message, "processing backend fallback");
        }
        Ok(notice)
    }

    pub fn set_default_tensorrt_cache_directory(&self, path: std::path::PathBuf) -> GuiResult<()> {
        let runtime = self.lock()?.runtime.clone();
        let snapshot = runtime.snapshot()?;
        if snapshot.config.tensorrt.cache_directory.is_some() {
            return Ok(());
        }
        let mut config = snapshot.config;
        config.tensorrt.cache_directory = Some(path.display().to_string());
        runtime.update_config(config, snapshot.config_revision)?;
        Ok(())
    }

    pub fn set_boot_phase(
        &self,
        phase: BootPhase,
        message: Option<String>,
    ) -> GuiResult<UiBootStatus> {
        let status = self.startup.set(phase, message)?;
        self.events.publish(BackendEvent::boot(status.clone()));
        Ok(status)
    }

    pub fn startup_resources(&self) -> Option<StartupResources> {
        self.startup.resources()
    }

    pub fn boot_status(&self) -> GuiResult<UiBootStatus> {
        self.startup.status()
    }

    pub fn finish_startup(&self, resources: StartupResources) -> GuiResult<()> {
        let _operation = self.operation()?;
        if self.boot_status()?.phase == BootPhase::ShuttingDown {
            return Err(GuiError::InvalidState {
                state: "shutting_down".into(),
                action: "finish_startup".into(),
            });
        }
        let revision = self.lock()?.runtime.snapshot()?.config_revision;
        let pending = resources
            .saved_model
            .as_ref()
            .map(|model| PendingModelRestore {
                path: model.path.clone(),
                revision,
            });
        let notice = (!resources.notices.is_empty()).then(|| resources.notices.join("\n"));
        self.startup.finish(resources)?;
        self.lock()?.pending_restore = pending;
        self.set_boot_phase(BootPhase::Ready, notice)?;
        self.events
            .publish(BackendEvent::snapshot(self.snapshot()?));
        Ok(())
    }

    pub fn fail_boot(&self, error: &GuiError) {
        if self
            .set_boot_phase(BootPhase::Failed, Some(error.to_string()))
            .is_err()
        {
            return;
        }
        if let Ok(mut state) = self.inner.lock() {
            state.last_error = Some(ErrorReport::from_error(error));
        }
        self.events
            .publish(BackendEvent::error(ErrorReport::from_error(error)));
        if let Ok(snapshot) = self.snapshot() {
            self.events.publish(BackendEvent::snapshot(snapshot));
        }
    }

    pub fn snapshot(&self) -> GuiResult<UiSnapshot> {
        let mut state = self.lock()?;
        state.snapshot()
    }

    pub fn list_audio_devices(&self) -> GuiResult<AudioDeviceCatalog> {
        AudioSession::enumerate().map_err(GuiError::AudioDiscovery)
    }

    pub fn scan_model_directory(&self, path: String) -> GuiResult<ModelCatalog> {
        let runtime = self.lock()?.runtime.clone();
        runtime.scan_model_directory(path).map_err(Into::into)
    }

    pub fn edit_model_metadata(
        &self,
        path: String,
        revision: String,
        edit: vocal_model::ModelMetadataEdit,
    ) -> GuiResult<vocal_model::ModelMetadataView> {
        let _operation = self.operation()?;
        let runtime = {
            let state = self.lock()?;
            if state.audio.is_some() {
                return Err(GuiError::InvalidState {
                    state: "audio_session_active".into(),
                    action: "edit_model_metadata".into(),
                });
            }
            state.runtime.clone()
        };
        let result = runtime.edit_model_metadata(path, revision, edit)?;
        self.events
            .publish(BackendEvent::snapshot(self.snapshot()?));
        Ok(result)
    }

    pub fn attach_channel(
        &self,
        label: String,
        channel: tauri::ipc::Channel<BackendEvent>,
    ) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        self.events.attach(label, channel)?;
        let snapshot = self.snapshot()?;
        self.events
            .publish(BackendEvent::snapshot(snapshot.clone()));
        Ok(snapshot)
    }

    pub fn attach_boot_channel(
        &self,
        label: String,
        channel: tauri::ipc::Channel<BackendEvent>,
    ) -> GuiResult<UiBootStatus> {
        self.events.attach(label, channel)?;
        self.startup.status()
    }

    /// New windows can observe preparation even while the runtime actor is loading.
    pub fn attach_preparation_channel(
        &self,
        label: String,
        channel: tauri::ipc::Channel<BackendEvent>,
    ) -> GuiResult<crate::preparation::PreparationStatus> {
        self.events.attach(label, channel)?;
        Ok(self.preparation.snapshot())
    }

    pub fn update_config(
        &self,
        config: GuiConfig,
        expected_revision: u64,
    ) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        {
            let state = self.lock()?;
            if state.audio.is_some() {
                return Err(GuiError::InvalidState {
                    state: "audio_session_active".into(),
                    action: "update_config".into(),
                });
            }
        }
        let runtime = self.lock()?.runtime.clone();
        let progress = self.preparation.begin(PreparationKind::UpdateConfig);
        progress.report(
            PreparationPhase::LoadingEngine,
            "Applying configuration and updating the idle engine",
        );
        if let Err(error) = runtime.update_config(config, expected_revision) {
            progress.report(PreparationPhase::Failed, error.to_string());
            // A failed candidate leaves the active engine/configuration intact;
            // reporting it must not mark that engine as a runtime failure.
            return Err(GuiError::idle_update(error));
        }
        progress.report(PreparationPhase::Ready, "Configuration saved");
        {
            let mut state = self.lock()?;
            state.pending_restore = None;
            state.last_error = None;
        }
        let snapshot = self.snapshot()?;
        self.events
            .publish(BackendEvent::snapshot(snapshot.clone()));
        Ok(snapshot)
    }

    pub fn prepare_tensorrt_pipeline(
        &self,
        request: PrepareTensorRtRequest,
    ) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        let (runtime, mut config, revision) = {
            let state = self.lock()?;
            if state.audio.is_some() {
                return Err(GuiError::InvalidState {
                    state: "audio_session_active".into(),
                    action: "prepare_tensorrt_pipeline".into(),
                });
            }
            let snapshot = state.runtime.snapshot()?;
            (
                state.runtime.clone(),
                snapshot.config,
                snapshot.config_revision,
            )
        };
        if revision != request.expected_revision {
            return Err(GuiError::ConfigConflict {
                expected: request.expected_revision,
                actual: revision,
            });
        }

        vocal_runtime::detect_processing_backends()
            .ensure_available(RuntimeBackend::TensorRtRtx)?;

        self.lock()?.pending_restore = None;
        let started = Instant::now();
        let progress = self.preparation.begin(PreparationKind::PrepareTensorRt);
        let prepared = self.prepare_pipeline(&config, request, &progress)?;
        config.backend = RuntimeBackend::TensorRtRtx;
        config.tensorrt = prepared.config.clone();
        config.tensorrt.artifacts_path = Some(prepared.artifacts_path.display().to_string());
        config.tensorrt_artifacts_path = config.tensorrt.artifacts_path.clone();
        config.tensorrt_window_samples = config.tensorrt.window_samples;
        config.model_path = Some(prepared.model_path.display().to_string());
        progress.report(
            PreparationPhase::LoadingEngine,
            "Applying the prepared pipeline to the idle engine",
        );
        if let Err(error) = runtime.apply_prepared_tensorrt_pipeline(config, revision) {
            let error = GuiError::idle_update(error);
            self.pipeline_failed(&error, started);
            progress.report(PreparationPhase::Failed, error.to_string());
            return Err(error);
        }
        progress.report(PreparationPhase::Ready, "TensorRT plans ready");
        self.pipeline_ready(&prepared, started)?;
        self.snapshot()
    }

    // Caller already owns the operation guard. Never reacquire it during Load.
    fn prepare_pipeline(
        &self,
        config: &GuiConfig,
        request: PrepareTensorRtRequest,
        progress: &PreparationOperation,
    ) -> GuiResult<crate::tensorrt::PreparedTensorRt> {
        self.set_tensorrt_pipeline(TensorRtPipelineStatus {
            phase: TensorRtPipelinePhase::CheckingCache,
            message: Some("Checking the TensorRT pipeline cache".into()),
            artifacts_path: None,
            cache_hit: false,
            elapsed_ms: 0,
        })?;
        let started = Instant::now();
        crate::tensorrt::prepare(config, request, progress).map_err(|error| {
            let error = GuiError::idle_candidate(error);
            self.pipeline_failed(&error, started);
            progress.report(PreparationPhase::Failed, error.to_string());
            error
        })
    }

    fn pipeline_failed(&self, error: &GuiError, started: Instant) {
        let _ = self.set_tensorrt_pipeline(TensorRtPipelineStatus {
            phase: TensorRtPipelinePhase::Failed,
            message: Some(error.to_string()),
            artifacts_path: None,
            cache_hit: false,
            elapsed_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        });
    }

    fn pipeline_ready(
        &self,
        prepared: &crate::tensorrt::PreparedTensorRt,
        started: Instant,
    ) -> GuiResult<()> {
        let elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        self.set_tensorrt_pipeline(TensorRtPipelineStatus {
            phase: TensorRtPipelinePhase::Ready,
            message: Some(if prepared.cache_hit {
                "TensorRT pipeline ready (cache hit)".into()
            } else {
                "TensorRT pipeline ready".into()
            }),
            artifacts_path: Some(prepared.artifacts_path.display().to_string()),
            cache_hit: prepared.cache_hit,
            elapsed_ms,
        })
    }

    pub fn load_model(&self, path: String) -> GuiResult<UiSnapshot> {
        self.load_model_with_backend(path, None)
    }

    pub fn load_model_with_backend(
        &self,
        path: String,
        selection: Option<vocal_runtime::ModelBackendSelection>,
    ) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        self.lock()?.pending_restore = None;
        self.load_model_locked(path, selection)
    }

    /// Only the visible main window invokes this after subscribing and painting.
    /// Taking the request under the command lock prevents reloads and manual
    /// model actions from restoring an obsolete selection a second time.
    pub fn restore_saved_model(&self, expected_revision: u64) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        if self.startup.status()?.phase != BootPhase::Ready {
            return Err(GuiError::InvalidState {
                state: "startup_not_ready".into(),
                action: "restore_saved_model".into(),
            });
        }
        let pending = self.lock()?.pending_restore.take();
        let Some(pending) = pending else {
            return self.snapshot();
        };
        let actual = self.lock()?.runtime.snapshot()?.config_revision;
        if expected_revision != actual || pending.revision != actual {
            return Err(GuiError::ConfigConflict {
                expected: expected_revision,
                actual,
            });
        }
        self.load_model_locked(pending.path, None)
    }

    fn load_model_locked(
        &self,
        path: String,
        selection: Option<vocal_runtime::ModelBackendSelection>,
    ) -> GuiResult<UiSnapshot> {
        let runtime = {
            let state = self.lock()?;
            if state.audio.is_some() {
                return Err(GuiError::InvalidState {
                    state: "audio_session_active".into(),
                    action: "load_model".into(),
                });
            }
            state.runtime.clone()
        };
        let started = Instant::now();
        let progress = self.preparation.begin(PreparationKind::LoadModel);
        let result = (|| -> GuiResult<()> {
            let current = runtime.snapshot()?;
            let mut config = current.config;
            if let Some(selection) = selection.as_ref() {
                if current.config_revision != selection.expected_revision {
                    return Err(vocal_runtime::RuntimeError::ConfigConflict {
                        expected: selection.expected_revision,
                        actual: current.config_revision,
                    }
                    .into());
                }
                let catalog = vocal_runtime::detect_processing_backends();
                config.backend = if selection.automatic {
                    catalog.recommended
                } else {
                    selection.backend
                };
                catalog.ensure_available(config.backend)?;
                config.backend_auto = selection.automatic;
                config.tensorrt.artifacts_path = selection.artifacts_path.clone();
                config.tensorrt_artifacts_path = selection.artifacts_path.clone();
                if config.backend == RuntimeBackend::DirectMl {
                    config.directml_sources_path = selection.directml_sources_path.clone();
                    config.directml_device_id = selection.directml_device_id.clone();
                }
            }
            config.validate()?;
            let inspection = vocal_model::inspect_model_artifact(&path)?;
            if config.backend == RuntimeBackend::TensorRtRtx
                && inspection.components.automatic_tensor_rt
            {
                vocal_runtime::detect_processing_backends().ensure_available(config.backend)?;
                let prepared = self.prepare_pipeline(
                    &config,
                    PrepareTensorRtRequest {
                        expected_revision: current.config_revision,
                        model_path: Some(path.clone()),
                        hubert_source: None,
                        rmvpe_source: None,
                        cache_directory: None,
                    },
                    &progress,
                )?;
                config.tensorrt = prepared.config.clone();
                config.tensorrt_artifacts_path = config.tensorrt.artifacts_path.clone();
                config.tensorrt_window_samples = config.tensorrt.window_samples;
                progress.report(
                    PreparationPhase::LoadingEngine,
                    "Loading TensorRT plans, allocating GPU buffers and warming the engine",
                );
                if let Err(error) =
                    runtime.load_model_with_config(path, config, current.config_revision)
                {
                    let error = GuiError::from(error);
                    self.pipeline_failed(&error, started);
                    return Err(error);
                }
                self.pipeline_ready(&prepared, started)?;
            } else {
                progress.report(
                    PreparationPhase::LoadingEngine,
                    if config.backend == RuntimeBackend::RvcCpu {
                        "Preparing acoustic weights, compiling oneDNN FP32 operators and allocating CPU streaming buffers"
                    } else {
                        "Loading model weights and initializing the processing engine"
                    },
                );
                match selection {
                    Some(selection) => runtime.load_model_with_backend(path, selection)?,
                    None => runtime.load_model(path)?,
                };
            }
            progress.report(
                PreparationPhase::Ready,
                if self.preparation.snapshot().cache_hit {
                    "Model ready · reused cached plans"
                } else {
                    "Model ready"
                },
            );
            Ok(())
        })();
        match result {
            Ok(_) => {
                let snapshot = {
                    let mut state = self.lock()?;
                    state.last_error = None;
                    state.snapshot()?
                };
                self.events
                    .publish(BackendEvent::snapshot(snapshot.clone()));
                Ok(snapshot)
            }
            Err(error) => {
                progress.report(PreparationPhase::Failed, error.to_string());
                self.fail(error.clone());
                Err(error)
            }
        }
    }

    pub fn unload_model(&self) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        self.lock()?.pending_restore = None;
        let runtime = {
            let state = self.lock()?;
            if state.audio.is_some() {
                return Err(GuiError::InvalidState {
                    state: "audio_session_active".into(),
                    action: "unload_model".into(),
                });
            }
            state.runtime.clone()
        };
        runtime.unload_model()?;
        let snapshot = self.snapshot()?;
        self.events
            .publish(BackendEvent::snapshot(snapshot.clone()));
        Ok(snapshot)
    }

    pub fn start(&self) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        let runtime = {
            let state = self.lock()?;
            if state.audio.is_some() {
                return Err(GuiError::InvalidState {
                    state: "audio_session_active".into(),
                    action: "start".into(),
                });
            }
            state.runtime.clone()
        };
        let config = runtime.config()?;
        runtime.start_processing()?;
        let reporter: Arc<dyn Fn(String) + Send + Sync> = Arc::new({
            let state = self.clone();
            move |message| state.report(&GuiError::Audio(message))
        });
        let session = match AudioSession::start(&config, runtime.clone(), reporter) {
            Ok(session) => session,
            Err(error) => {
                let _ = runtime.abort_processing_start();
                return Err(GuiError::AudioStartup(error));
            }
        };
        let snapshot = {
            let mut state = self.lock()?;
            state.audio = Some(session);
            state.last_error = None;
            state.snapshot()?
        };
        self.events
            .publish(BackendEvent::snapshot(snapshot.clone()));
        Ok(snapshot)
    }

    pub fn stop(&self) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        let session = {
            let mut state = self.lock()?;
            state.audio.take()
        };
        drop(session);
        let runtime = self.lock()?.runtime.clone();
        if runtime.snapshot()?.lifecycle == LifecycleState::Running {
            runtime.stop_processing()?;
        }
        let snapshot = self.snapshot()?;
        self.events
            .publish(BackendEvent::snapshot(snapshot.clone()));
        Ok(snapshot)
    }

    /// Stop the audio adapter and mark the application as shutting down. This
    /// is shared by the tray Quit action and native close handling so every
    /// exit path releases the audio worker before the process terminates.
    pub fn shutdown(&self) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        self.set_boot_phase(BootPhase::ShuttingDown, None)?;
        let session = {
            let mut state = self.lock()?;
            state.pending_restore = None;
            state.audio.take()
        };
        drop(session);
        let runtime = self.lock()?.runtime.clone();
        if runtime.snapshot()?.lifecycle == LifecycleState::Running {
            runtime.stop_processing()?;
        }
        let snapshot = self.snapshot()?;
        self.events
            .publish(BackendEvent::snapshot(snapshot.clone()));
        Ok(snapshot)
    }

    pub fn process(
        &self,
        samples: Vec<f32>,
        format: AudioFormat,
        sequence: u64,
    ) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        let runtime = {
            let state = self.lock()?;
            if state.audio.is_some() {
                return Err(GuiError::InvalidState {
                    state: "audio_session_active".into(),
                    action: "process_block".into(),
                });
            }
            state.runtime.clone()
        };
        runtime.process(&AudioChunk {
            format,
            samples,
            sequence,
        })?;
        let snapshot = self.snapshot()?;
        self.events
            .publish(BackendEvent::snapshot(snapshot.clone()));
        Ok(snapshot)
    }

    pub fn clear_error(&self) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        let session = {
            let mut state = self.lock()?;
            state.audio.take()
        };
        drop(session);
        let runtime = self.lock()?.runtime.clone();
        let snapshot = {
            runtime.clear_error()?;
            let mut state = self.lock()?;
            state.last_error = None;
            state.snapshot()?
        };
        self.events
            .publish(BackendEvent::snapshot(snapshot.clone()));
        Ok(snapshot)
    }

    pub fn report(&self, error: &GuiError) {
        let runtime = self.lock().ok().map(|mut state| {
            state.last_error = Some(ErrorReport::from_error(error));
            state.runtime.clone()
        });
        if let Some(runtime) = runtime.as_ref()
            && matches!(
                error,
                GuiError::Engine(_) | GuiError::Audio(_) | GuiError::Internal(_)
            )
        {
            let _ = runtime.report_failure(error.code(), error.to_string());
        }
        let snapshot = self.snapshot().ok();
        self.events
            .publish(BackendEvent::error(ErrorReport::from_error(error)));
        if let Some(snapshot) = snapshot {
            self.events.publish(BackendEvent::snapshot(snapshot));
        }
    }

    fn fail(&self, error: GuiError) {
        self.report(&error);
    }

    fn set_tensorrt_pipeline(&self, status: TensorRtPipelineStatus) -> GuiResult<()> {
        let snapshot = {
            let mut state = self.lock()?;
            state.tensorrt_pipeline = status;
            state.snapshot()?
        };
        self.events.publish(BackendEvent::snapshot(snapshot));
        Ok(())
    }

    fn operation(&self) -> GuiResult<std::sync::MutexGuard<'_, ()>> {
        self.command_lock
            .lock()
            .map_err(|_| GuiError::Internal("application command lock poisoned".into()))
    }

    fn lock(&self) -> GuiResult<std::sync::MutexGuard<'_, RuntimeState>> {
        self.inner
            .lock()
            .map_err(|_| GuiError::Internal("application state lock poisoned".into()))
    }
}

impl RuntimeState {
    fn snapshot(&mut self) -> GuiResult<UiSnapshot> {
        let audio_failure = self
            .audio
            .as_mut()
            .and_then(AudioSession::reap_if_unhealthy);
        if let Some(message) = audio_failure {
            // Drop outside the audio callback and before changing the runtime
            // state. AudioSession::Drop joins the worker after its engine lease
            // has been returned, so a subsequent start can safely acquire a
            // fresh lease instead of seeing a stale session handle.
            self.audio.take();
            let runtime = self.runtime.snapshot()?;
            if runtime.lifecycle != LifecycleState::Failed {
                self.runtime
                    .report_failure("audio.session", message.clone())?;
                self.last_error = Some(ErrorReport {
                    code: "audio_session".into(),
                    message,
                    recoverable: true,
                    details: None,
                });
            }
        }
        let runtime = self.runtime.snapshot()?;
        let audio = self
            .audio
            .as_mut()
            .map(AudioSession::snapshot)
            .unwrap_or_else(AudioTelemetry::stopped);
        let runtime_error = runtime.last_error.map(|error| ErrorReport {
            code: error.code,
            message: error.message,
            recoverable: true,
            details: None,
        });
        let engine = runtime.engine.map(UiEngineStatus::from);
        Ok(UiSnapshot {
            boot: self.startup.status()?,
            lifecycle: runtime.lifecycle,
            status: UiStatus {
                session: runtime.lifecycle,
                model: if runtime.model.is_some() {
                    LifecycleState::Ready
                } else {
                    LifecycleState::Created
                },
                engine: engine
                    .as_ref()
                    .map(|engine| engine.state)
                    .unwrap_or(LifecycleState::Created),
                audio: audio.state,
            },
            config: runtime.config,
            config_revision: runtime.config_revision,
            last_config_update: runtime.last_config_update,
            model: runtime.model,
            engine,
            last_error: self.last_error.clone().or(runtime_error),
            audio,
            tensorrt_pipeline: self.tensorrt_pipeline.clone(),
            preparation: self.preparation.snapshot(),
            model_restore_pending: self.pending_restore.is_some(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_idle_pipeline_candidate_reports_without_poisoning_active_engine() {
        let state = AppState::new(GuiConfig::default());
        state
            .lock()
            .unwrap()
            .runtime
            .load_model_descriptor(vocal_types::ModelDescriptor {
                id: "fixture".into(),
                display_name: "Fixture".into(),
                architecture: "rvc-v2".into(),
                sample_rate: 40000,
                version: "2".into(),
            })
            .unwrap();
        let error = GuiError::idle_update(vocal_runtime::RuntimeError::Engine(
            vocal_engine::EngineError::BackendUnavailable("candidate GPU pipeline failed".into()),
        ));
        state.report(&error);
        let failed = state.snapshot().unwrap();
        assert_eq!(failed.lifecycle, LifecycleState::Ready);
        assert_eq!(failed.engine.unwrap().state, LifecycleState::Ready);
        assert_eq!(failed.last_error.unwrap().code, "config_update_failed");
        assert_eq!(
            state
                .update_config(failed.config, failed.config_revision)
                .unwrap()
                .lifecycle,
            LifecycleState::Ready
        );
        assert!(state.snapshot().unwrap().last_error.is_none());
    }

    fn startup_resources() -> StartupResources {
        StartupResources {
            audio_devices: None,
            processing_backends: vocal_runtime::ProcessingBackendCatalog {
                cpu: Default::default(),
                recommended: RuntimeBackend::Passthrough,
                backends: Vec::new(),
            },
            saved_model: Some(crate::startup::StartupModel {
                path: "nonexistent-startup-model.vfrm".into(),
                summary: vocal_runtime::ModelSummary {
                    id: "saved".into(),
                    display_name: "Saved voice".into(),
                    architecture: "rvc".into(),
                    version: "2".into(),
                    sample_rate: 48000,
                    feature_dims: 768,
                    uses_f0: true,
                },
                components: vocal_model::ModelComponents::default(),
            }),
            notices: Vec::new(),
        }
    }

    #[test]
    fn shell_ready_does_not_load_weights_or_create_an_engine() {
        let state = AppState::new(GuiConfig::default());
        state.finish_startup(startup_resources()).unwrap();
        let snapshot = state.snapshot().unwrap();
        assert_eq!(snapshot.boot.phase, BootPhase::Ready);
        assert!(snapshot.model.is_none());
        assert!(snapshot.engine.is_none());
        assert!(snapshot.model_restore_pending);
        assert_eq!(snapshot.preparation.phase, PreparationPhase::Idle);
    }

    #[test]
    fn failed_restore_is_claimed_once_and_never_changes_shell_readiness() {
        let state = AppState::new(GuiConfig::default());
        state.finish_startup(startup_resources()).unwrap();
        state
            .restore_saved_model(0)
            .expect_err("missing saved model");
        let failed = state.snapshot().unwrap();
        assert_eq!(failed.boot.phase, BootPhase::Ready);
        assert_eq!(failed.preparation.phase, PreparationPhase::Failed);
        assert!(!failed.model_restore_pending);
        let repeated = state.restore_saved_model(0).unwrap();
        assert_eq!(
            repeated.preparation.operation_id,
            failed.preparation.operation_id
        );
        assert_eq!(state.clear_error().unwrap().boot.phase, BootPhase::Ready);
    }

    #[test]
    fn manual_model_action_or_configuration_change_cancels_saved_restore() {
        let state = AppState::new(GuiConfig::default());
        state.finish_startup(startup_resources()).unwrap();
        let saved = state.update_config(GuiConfig::default(), 0).unwrap();
        assert!(!saved.model_restore_pending);
        assert_eq!(
            state
                .restore_saved_model(1)
                .unwrap()
                .preparation
                .operation_id,
            saved.preparation.operation_id
        );

        let state = AppState::new(GuiConfig::default());
        state.finish_startup(startup_resources()).unwrap();
        state.unload_model().unwrap();
        assert!(!state.snapshot().unwrap().model_restore_pending);

        let state = AppState::new(GuiConfig::default());
        state.finish_startup(startup_resources()).unwrap();
        state
            .load_model("manual-nonexistent-model.vfrm".into())
            .unwrap_err();
        let failed = state.snapshot().unwrap();
        assert!(!failed.model_restore_pending);
        assert_eq!(
            state
                .restore_saved_model(0)
                .unwrap()
                .preparation
                .operation_id,
            failed.preparation.operation_id
        );
    }

    #[test]
    fn stale_restore_revision_does_not_start_preparation() {
        let state = AppState::new(GuiConfig::default());
        state.finish_startup(startup_resources()).unwrap();
        assert!(matches!(
            state.restore_saved_model(99),
            Err(GuiError::ConfigConflict { actual: 0, .. })
        ));
        assert_eq!(
            state.snapshot().unwrap().preparation.phase,
            PreparationPhase::Idle
        );
    }

    #[test]
    fn startup_catalog_and_status_do_not_acquire_the_runtime_lock() {
        let state = AppState::new(GuiConfig::default());
        state.finish_startup(startup_resources()).unwrap();
        let _runtime_lock = state.lock().unwrap();
        assert!(state.startup_resources().unwrap().saved_model.is_some());
        assert_eq!(state.startup.status().unwrap().phase, BootPhase::Ready);
    }

    #[test]
    fn progress_does_not_acquire_the_runtime_state_lock() {
        let state = AppState::new(GuiConfig::default());
        let _runtime_lock = state.lock().unwrap();
        let progress = state.preparation.begin(PreparationKind::LoadModel);
        progress.report(PreparationPhase::LoadingEngine, "Loading engine");
        assert_eq!(
            state.preparation.snapshot().phase,
            PreparationPhase::LoadingEngine
        );
        progress.report(PreparationPhase::Ready, "Ready");
    }

    #[test]
    fn failed_model_load_reports_the_cause_and_allows_a_new_operation() {
        let state = AppState::new(GuiConfig::default());
        state
            .load_model("nonexistent-model.vfrm".into())
            .expect_err("missing model");
        let failed = state.snapshot().unwrap().preparation;
        assert_eq!(failed.phase, PreparationPhase::Failed);
        assert!(failed.message.contains("nonexistent-model"));
        let ready = state
            .update_config(GuiConfig::default(), 0)
            .unwrap()
            .preparation;
        assert_eq!(ready.phase, PreparationPhase::Ready);
        assert!(ready.operation_id > failed.operation_id);
    }

    #[test]
    fn configuration_revision_is_optimistic_and_monotonic() {
        let state = AppState::new(GuiConfig::default());
        let mut config = GuiConfig::default();
        config.backend = vocal_runtime::RuntimeBackend::RvcCpu;
        let snapshot = state.update_config(config, 0).expect("update config");
        assert_eq!(snapshot.config_revision, 1);
        let error = state
            .update_config(GuiConfig::default(), 0)
            .expect_err("stale revision must fail");
        assert!(matches!(error, GuiError::ConfigConflict { actual: 1, .. }));
    }

    #[test]
    fn start_requires_a_loaded_engine() {
        let state = AppState::new(GuiConfig::default());
        let error = state.start().expect_err("missing model must fail");
        assert!(matches!(error, GuiError::InvalidState { .. }));
    }

    #[test]
    fn unload_clears_model_state_and_advances_revision() {
        let state = AppState::new(GuiConfig::default());
        let snapshot = state.unload_model().expect("unload without model");
        assert_eq!(snapshot.lifecycle, LifecycleState::Created);
        assert_eq!(snapshot.config_revision, 1);
        assert!(snapshot.model.is_none());
        assert!(snapshot.config.model_path.is_none());
    }

    #[test]
    fn snapshot_exposes_independent_subsystem_states() {
        let state = AppState::new(GuiConfig::default());
        let snapshot = state.snapshot().expect("snapshot");
        assert_eq!(snapshot.boot.phase, BootPhase::Starting);
        assert_eq!(snapshot.status.session, LifecycleState::Created);
        assert_eq!(snapshot.status.model, LifecycleState::Created);
        assert_eq!(snapshot.status.engine, LifecycleState::Created);
        assert_eq!(snapshot.status.audio, LifecycleState::Stopped);
    }

    #[test]
    fn stop_is_idempotent_when_no_audio_session_exists() {
        let state = AppState::new(GuiConfig::default());
        let snapshot = state.stop().expect("stop without session");
        assert_eq!(snapshot.lifecycle, LifecycleState::Created);
        assert_eq!(snapshot.boot.phase, BootPhase::Starting);
    }
}
