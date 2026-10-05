use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use vocal_runtime::{RuntimeContext, RuntimeBackend};
use vocal_types::{AudioChunk, AudioFormat, LifecycleState};

use crate::{
    audio::{AudioDeviceCatalog, AudioSession, AudioTelemetry},
    config::{ConfigStore, GuiConfig},
    error::{GuiError, GuiResult},
    events::{BackendEvent, ErrorReport, EventSink},
    snapshot::{
        BootPhase, TensorRtPipelinePhase, TensorRtPipelineStatus, UiBootStatus, UiEngineStatus,
        UiStatus,
    },
};

use crate::tensorrt::PrepareTensorRtRequest;

pub use crate::snapshot::{UiMetrics, UiSnapshot};
pub use vocal_runtime::ModelCatalog;

struct RuntimeState {
    runtime: RuntimeContext,
    audio: Option<AudioSession>,
    last_error: Option<ErrorReport>,
    boot: BootState,
    tensorrt_pipeline: TensorRtPipelineStatus,
}

struct BootState {
    phase: BootPhase,
    message: Option<String>,
    recoverable: bool,
    started_at: Instant,
}

impl BootState {
    fn new() -> Self {
        Self {
            phase: BootPhase::Starting,
            message: None,
            recoverable: true,
            started_at: Instant::now(),
        }
    }

    fn set(&mut self, phase: BootPhase, message: Option<String>, recoverable: bool) {
        self.phase = phase;
        self.message = message;
        self.recoverable = recoverable;
    }

    fn snapshot(&self) -> UiBootStatus {
        UiBootStatus {
            phase: self.phase,
            message: self.message.clone(),
            elapsed_ms: self
                .started_at
                .elapsed()
                .as_millis()
                .min(u128::from(u64::MAX)) as u64,
            recoverable: self.recoverable,
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    inner: Arc<Mutex<RuntimeState>>,
    command_lock: Arc<Mutex<()>>,
    pub events: Arc<EventSink>,
}

impl AppState {
    pub fn new(config: GuiConfig) -> Self {
        let runtime =
            RuntimeContext::new(config).expect("default runtime configuration must validate");
        Self {
            inner: Arc::new(Mutex::new(RuntimeState {
                runtime,
                audio: None,
                last_error: None,
                boot: BootState::new(),
                tensorrt_pipeline: TensorRtPipelineStatus::default(),
            })),
            command_lock: Arc::new(Mutex::new(())),
            events: Arc::new(EventSink::default()),
        }
    }

    pub fn set_config_store(&self, store: ConfigStore) -> GuiResult<()> {
        let runtime = self.lock()?.runtime.clone();
        runtime.set_config_store(store.runtime_store())?;
        Ok(())
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
    ) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        let snapshot = {
            let mut state = self.lock()?;
            state.boot.set(phase, message, true);
            state.snapshot()?
        };
        self.events
            .publish(BackendEvent::snapshot(snapshot.clone()));
        Ok(snapshot)
    }

    pub fn mark_boot_ready(&self) {
        if let Ok(mut state) = self.inner.lock() {
            state.boot.set(BootPhase::Ready, None, true);
            if state
                .last_error
                .as_ref()
                .is_some_and(|error| error.code == "startup_timeout")
            {
                state.last_error = None;
            }
        }
        if let Ok(snapshot) = self.snapshot() {
            self.events.publish(BackendEvent::snapshot(snapshot));
        }
    }

    pub fn fail_boot(&self, error: &GuiError) {
        if let Ok(mut state) = self.inner.lock() {
            state
                .boot
                .set(BootPhase::Failed, Some(error.to_string()), true);
            state.last_error = Some(ErrorReport::from_error(error));
        }
        self.events
            .publish(BackendEvent::error(ErrorReport::from_error(error)));
        if let Ok(snapshot) = self.snapshot() {
            self.events.publish(BackendEvent::snapshot(snapshot));
        }
    }

    /// Reveal the main UI after a long restore without cancelling the worker.
    /// The worker can still complete later and transition the boot state to
    /// `Ready`; this keeps a slow disk or antivirus scan from looking frozen.
    pub fn mark_boot_timeout(&self) {
        let error = ErrorReport {
            code: "startup_timeout".into(),
            message: "Startup is taking longer than expected; the model is still loading.".into(),
            recoverable: true,
        };
        if let Ok(mut state) = self.inner.lock() {
            state
                .boot
                .set(BootPhase::Failed, Some(error.message.clone()), true);
            state.last_error = Some(error.clone());
        }
        self.events.publish(BackendEvent::error(error));
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

    pub fn persisted_model_path(&self) -> GuiResult<Option<String>> {
        Ok(self.lock()?.runtime.persisted_model_path()?)
    }

    pub fn scan_model_directory(&self, path: String) -> GuiResult<ModelCatalog> {
        let runtime = self.lock()?.runtime.clone();
        runtime.scan_model_directory(path).map_err(Into::into)
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
        runtime.update_config(config, expected_revision)?;
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
            (state.runtime.clone(), snapshot.config, snapshot.config_revision)
        };
        if revision != request.expected_revision {
            return Err(GuiError::ConfigConflict {
                expected: request.expected_revision,
                actual: revision,
            });
        }

        self.set_tensorrt_pipeline(TensorRtPipelineStatus {
            phase: TensorRtPipelinePhase::CheckingCache,
            message: Some("Checking the TensorRT pipeline cache".into()),
            artifacts_path: None,
            cache_hit: false,
            elapsed_ms: 0,
        })?;
        self.set_tensorrt_pipeline(TensorRtPipelineStatus {
            phase: TensorRtPipelinePhase::Building,
            message: Some("Preparing the TensorRT pipeline".into()),
            artifacts_path: None,
            cache_hit: false,
            elapsed_ms: 0,
        })?;
        let started = Instant::now();
        let prepared = crate::tensorrt::prepare(&config, request).map_err(|error| {
            let _ = self.set_tensorrt_pipeline(TensorRtPipelineStatus {
                phase: TensorRtPipelinePhase::Failed,
                message: Some(error.to_string()),
                artifacts_path: None,
                cache_hit: false,
                elapsed_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
            });
            error
        })?;
        config.backend = RuntimeBackend::TensorRtRtx;
        config.tensorrt = prepared.config;
        config.tensorrt.artifacts_path = Some(prepared.artifacts_path.display().to_string());
        config.tensorrt_artifacts_path = config.tensorrt.artifacts_path.clone();
        config.tensorrt_window_samples = config.tensorrt.window_samples;
        config.model_path = Some(prepared.model_path.display().to_string());
        if let Err(error) = runtime.update_config(config, revision) {
            let error = GuiError::from(error);
            let _ = self.set_tensorrt_pipeline(TensorRtPipelineStatus {
                phase: TensorRtPipelinePhase::Failed,
                message: Some(error.to_string()),
                artifacts_path: Some(prepared.artifacts_path.display().to_string()),
                cache_hit: prepared.cache_hit,
                elapsed_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
            });
            return Err(error);
        }
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
        })?;
        self.snapshot()
    }

    pub fn load_model(&self, path: String) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
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
        let result = runtime.load_model(path);
        match result {
            Ok(_) => {
                let snapshot = {
                    let mut state = self.lock()?;
                    state.last_error = None;
                    if state.boot.phase == BootPhase::Failed {
                        state.boot.set(BootPhase::Ready, None, true);
                    }
                    state.snapshot()?
                };
                self.events
                    .publish(BackendEvent::snapshot(snapshot.clone()));
                Ok(snapshot)
            }
            Err(error) => {
                let error = GuiError::from(error);
                self.fail(error.clone());
                Err(error)
            }
        }
    }

    pub fn unload_model(&self) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
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
                return Err(GuiError::Audio(error));
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
        let session = {
            let mut state = self.lock()?;
            state.boot.set(BootPhase::ShuttingDown, None, false);
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
        self.events.publish(BackendEvent::metrics(snapshot.metrics));
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
            let runtime_snapshot = runtime.clear_error()?;
            let mut state = self.lock()?;
            state.last_error = None;
            if state.boot.phase == BootPhase::Failed
                && runtime_snapshot.lifecycle != LifecycleState::Loading
            {
                state.boot.set(BootPhase::Ready, None, true);
            }
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
                });
            }
        }
        let runtime = self.runtime.snapshot()?;
        let audio = self
            .audio
            .as_ref()
            .map(AudioSession::snapshot)
            .unwrap_or_else(AudioTelemetry::stopped);
        let metrics = if audio.active {
            UiMetrics {
                input_frames: audio.input_frames,
                processing_micros: audio.processing_micros,
                output_frames: audio.output_frames,
            }
        } else {
            runtime.metrics.into()
        };
        let runtime_error = runtime.last_error.map(|error| ErrorReport {
            code: error.code,
            message: error.message,
            recoverable: true,
        });
        let engine = runtime.engine.map(UiEngineStatus::from);
        Ok(UiSnapshot {
            boot: self.boot.snapshot(),
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
            model: runtime.model,
            metrics,
            engine,
            last_error: self.last_error.clone().or(runtime_error),
            audio,
            tensorrt_pipeline: self.tensorrt_pipeline.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
