use std::sync::{Arc, Mutex};

use vocal_runtime::RuntimeContext;
use vocal_types::{AudioChunk, AudioFormat, LifecycleState};

use crate::{
    audio::{AudioDeviceCatalog, AudioSession, AudioTelemetry},
    config::{ConfigStore, GuiConfig},
    error::{GuiError, GuiResult},
    events::{BackendEvent, ErrorReport, EventSink},
    snapshot::{UiEngineStatus, UiStatus},
};

pub use crate::snapshot::{UiMetrics, UiSnapshot};

struct RuntimeState {
    runtime: RuntimeContext,
    audio: Option<AudioSession>,
    last_error: Option<ErrorReport>,
}

#[derive(Clone)]
pub struct AppState {
    inner: Arc<Mutex<RuntimeState>>,
    command_lock: Arc<Mutex<()>>,
    pub events: Arc<EventSink>,
}

impl AppState {
    pub fn new(config: GuiConfig) -> Self {
        let runtime = RuntimeContext::new(config).expect("default runtime configuration must validate");
        Self {
            inner: Arc::new(Mutex::new(RuntimeState {
                runtime,
                audio: None,
                last_error: None,
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

    pub fn attach_channel(
        &self,
        label: String,
        channel: tauri::ipc::Channel<BackendEvent>,
    ) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        self.events.attach(label, channel)?;
        let snapshot = self.snapshot()?;
        self.events.publish(BackendEvent::snapshot(snapshot.clone()));
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
        self.events.publish(BackendEvent::snapshot(snapshot.clone()));
        Ok(snapshot)
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
                let snapshot = self.snapshot()?;
                self.events.publish(BackendEvent::snapshot(snapshot.clone()));
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
        self.events.publish(BackendEvent::snapshot(snapshot.clone()));
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
        self.events.publish(BackendEvent::snapshot(snapshot.clone()));
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
        runtime.stop_processing()?;
        let snapshot = self.snapshot()?;
        self.events.publish(BackendEvent::snapshot(snapshot.clone()));
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
        self.events.publish(BackendEvent::snapshot(snapshot.clone()));
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
        self.events.publish(BackendEvent::snapshot(snapshot.clone()));
        Ok(snapshot)
    }

    pub fn report(&self, error: &GuiError) {
        let runtime = self.lock().ok().map(|mut state| {
            state.last_error = Some(ErrorReport::from_error(error));
            state.runtime.clone()
        });
        if let Some(runtime) = runtime.as_ref()
            && matches!(error, GuiError::Engine(_) | GuiError::Audio(_) | GuiError::Internal(_))
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
        if let Some(audio) = self.audio.as_mut() {
            audio.reap_if_dead();
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
        assert_eq!(snapshot.status.session, LifecycleState::Created);
        assert_eq!(snapshot.status.model, LifecycleState::Created);
        assert_eq!(snapshot.status.engine, LifecycleState::Created);
        assert_eq!(snapshot.status.audio, LifecycleState::Stopped);
    }
}
