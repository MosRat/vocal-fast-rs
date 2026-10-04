use std::sync::{Arc, Mutex};

use serde::Serialize;
use vocal_engine::{BackendKind, EngineConfig, EngineStatus, InferenceEngine, create_engine};
use vocal_model::{ModelLoader, RvcModelManifest};
use vocal_types::{AudioChunk, AudioFormat, LatencyMetrics, LifecycleState, ModelDescriptor};

use crate::{
    config::{ConfigStore, GuiBackend, GuiConfig},
    error::{GuiError, GuiResult},
    events::{BackendEvent, ErrorReport, EventSink},
};

const MAX_CONTROL_BLOCK_SAMPLES: usize = 1_048_576;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSummary {
    pub id: String,
    pub display_name: String,
    pub architecture: String,
    pub version: String,
    pub sample_rate: u32,
    pub feature_dims: usize,
    pub uses_f0: bool,
}

impl From<&RvcModelManifest> for ModelSummary {
    fn from(manifest: &RvcModelManifest) -> Self {
        Self {
            id: manifest.id.clone(),
            display_name: manifest.display_name.clone(),
            architecture: manifest.architecture.clone(),
            version: manifest.version.clone(),
            sample_rate: manifest.sample_rate,
            feature_dims: manifest.feature_dims(),
            uses_f0: manifest.uses_f0(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UiSnapshot {
    pub lifecycle: LifecycleState,
    pub config: GuiConfig,
    pub config_revision: u64,
    pub model: Option<ModelSummary>,
    pub metrics: UiMetrics,
    pub engine: Option<UiEngineStatus>,
    pub last_error: Option<ErrorReport>,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UiMetrics {
    pub input_frames: u64,
    pub processing_micros: u64,
    pub output_frames: u64,
}

impl From<LatencyMetrics> for UiMetrics {
    fn from(metrics: LatencyMetrics) -> Self {
        Self {
            input_frames: metrics.input_frames,
            processing_micros: metrics.processing_micros,
            output_frames: metrics.output_frames,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UiEngineStatus {
    pub state: LifecycleState,
    pub processed_chunks: u64,
    pub failures: u64,
    pub last_error: Option<String>,
}

impl From<EngineStatus> for UiEngineStatus {
    fn from(status: EngineStatus) -> Self {
        Self {
            state: status.state,
            processed_chunks: status.processed_chunks,
            failures: status.failures,
            last_error: status.last_error,
        }
    }
}

struct RuntimeState {
    lifecycle: LifecycleState,
    config: GuiConfig,
    config_revision: u64,
    model: Option<ModelSummary>,
    descriptor: Option<ModelDescriptor>,
    engine: Option<Box<dyn InferenceEngine>>,
    metrics: LatencyMetrics,
    last_error: Option<ErrorReport>,
    config_store: Option<ConfigStore>,
}

#[derive(Clone)]
pub struct AppState {
    inner: Arc<Mutex<RuntimeState>>,
    command_lock: Arc<Mutex<()>>,
    pub events: Arc<EventSink>,
}

impl AppState {
    pub fn new(config: GuiConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RuntimeState {
                lifecycle: LifecycleState::Created,
                config,
                config_revision: 0,
                model: None,
                descriptor: None,
                engine: None,
                metrics: LatencyMetrics::default(),
                last_error: None,
                config_store: None,
            })),
            command_lock: Arc::new(Mutex::new(())),
            events: Arc::new(EventSink::default()),
        }
    }

    pub fn set_config_store(&self, store: ConfigStore) -> GuiResult<()> {
        let loaded = store.load()?;
        let mut state = self.lock()?;
        state.config_store = Some(store);
        if let Some(config) = loaded {
            state.config = config;
            state.config_revision = state.config_revision.saturating_add(1);
        }
        Ok(())
    }

    pub fn snapshot(&self) -> GuiResult<UiSnapshot> {
        Ok(self.lock()?.snapshot())
    }

    pub fn attach_channel(
        &self,
        channel: tauri::ipc::Channel<BackendEvent>,
    ) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        self.events.attach(channel)?;
        let snapshot = self.snapshot()?;
        self.events.send(BackendEvent::Snapshot(snapshot.clone()))?;
        Ok(snapshot)
    }

    pub fn update_config(
        &self,
        config: GuiConfig,
        expected_revision: u64,
    ) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        config.validate()?;
        let (descriptor, store) = {
            let state = self.lock()?;
            if state.config_revision != expected_revision {
                return Err(GuiError::ConfigConflict {
                    expected: expected_revision,
                    actual: state.config_revision,
                });
            }
            if matches!(
                state.lifecycle,
                LifecycleState::Running | LifecycleState::Stopping
            ) {
                return Err(GuiError::InvalidState {
                    state: format!("{:?}", state.lifecycle),
                    action: "update_config".into(),
                });
            }
            (descriptor_clone(&state), state.config_store.clone())
        };
        let replacement = descriptor
            .as_ref()
            .map(|descriptor| Self::create_engine(&config, descriptor))
            .transpose()?;
        if let Some(store) = &store {
            store.save(&config)?;
        }
        let snapshot = {
            let mut state = self.lock()?;
            if state.config_revision != expected_revision {
                return Err(GuiError::ConfigConflict {
                    expected: expected_revision,
                    actual: state.config_revision,
                });
            }
            state.config = config;
            state.config_revision = state.config_revision.saturating_add(1);
            if replacement.is_some() {
                state.engine = replacement;
            }
            state.snapshot()
        };
        self.events.send(BackendEvent::Snapshot(snapshot.clone()))?;
        Ok(snapshot)
    }

    pub fn load_model(&self, path: String) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        self.transition(LifecycleState::Loading, "load_model")?;
        let result = self.load_model_inner(path);
        match result {
            Ok(snapshot) => {
                self.events.send(BackendEvent::Snapshot(snapshot.clone()))?;
                Ok(snapshot)
            }
            Err(error) => {
                self.fail(error.clone());
                Err(error)
            }
        }
    }

    fn load_model_inner(&self, path: String) -> GuiResult<UiSnapshot> {
        let input = std::path::PathBuf::from(&path);
        if !input.is_file() {
            return Err(GuiError::Model(format!(
                "model path does not exist: {path}"
            )));
        }
        let mut loader = ModelLoader::new(Default::default())?;
        let manifest = if input
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("vfr"))
        {
            loader.load_cache(&input)?.manifest
        } else {
            let manifest = loader.load_manifest(&input)?;
            let root = input.parent().unwrap_or_else(|| std::path::Path::new("."));
            loader.load_rvc_directory(root, manifest.clone())?;
            manifest
        };
        let descriptor = manifest.descriptor();
        let (mut config, store) = {
            let state = self.lock()?;
            (state.config.clone(), state.config_store.clone())
        };
        config.model_path = Some(path.clone());
        let engine = Self::create_engine(&config, &descriptor)?;
        if let Some(store) = &store {
            store.save(&config)?;
        }
        let summary = ModelSummary::from(&manifest);
        let snapshot = {
            let mut state = self.lock()?;
            state.config = config;
            state.config_revision = state.config_revision.saturating_add(1);
            state.model = Some(summary);
            state.descriptor = Some(descriptor);
            state.engine = Some(engine);
            state.lifecycle = LifecycleState::Ready;
            state.last_error = None;
            state.snapshot()
        };
        Ok(snapshot)
    }

    pub fn start(&self) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        let snapshot = {
            let mut state = self.lock()?;
            if state.engine.is_none() {
                return Err(GuiError::InvalidState {
                    state: format!("{:?}", state.lifecycle),
                    action: "start".into(),
                });
            }
            if state.lifecycle == LifecycleState::Stopped {
                state.lifecycle = LifecycleState::Ready;
            }
            if !state.lifecycle.can_transition_to(LifecycleState::Running) {
                return Err(GuiError::InvalidState {
                    state: format!("{:?}", state.lifecycle),
                    action: "start".into(),
                });
            }
            state.lifecycle = LifecycleState::Running;
            state.last_error = None;
            state.snapshot()
        };
        self.events.send(BackendEvent::Snapshot(snapshot.clone()))?;
        Ok(snapshot)
    }

    pub fn stop(&self) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        let (stopping, snapshot) = {
            let mut state = self.lock()?;
            if !matches!(
                state.lifecycle,
                LifecycleState::Running | LifecycleState::Ready
            ) {
                return Err(GuiError::InvalidState {
                    state: format!("{:?}", state.lifecycle),
                    action: "stop".into(),
                });
            }
            if !state.lifecycle.can_transition_to(LifecycleState::Stopping) {
                return Err(GuiError::InvalidState {
                    state: format!("{:?}", state.lifecycle),
                    action: "stop".into(),
                });
            }
            state.lifecycle = LifecycleState::Stopping;
            let stopping = state.snapshot();
            state.lifecycle = LifecycleState::Stopped;
            (stopping, state.snapshot())
        };
        self.events.send(BackendEvent::Snapshot(stopping))?;
        self.events.send(BackendEvent::Snapshot(snapshot.clone()))?;
        Ok(snapshot)
    }

    pub fn process(
        &self,
        samples: Vec<f32>,
        format: AudioFormat,
        sequence: u64,
    ) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        if samples.len() > MAX_CONTROL_BLOCK_SAMPLES {
            return Err(GuiError::InvalidAudio(format!(
                "control block exceeds {MAX_CONTROL_BLOCK_SAMPLES} samples"
            )));
        }
        let input = AudioChunk {
            format,
            samples,
            sequence,
        };
        input
            .validate()
            .map_err(|error| GuiError::InvalidAudio(error.to_string()))?;
        let (mut engine, config) = {
            let mut state = self.lock()?;
            if state.lifecycle != LifecycleState::Running {
                return Err(GuiError::InvalidState {
                    state: format!("{:?}", state.lifecycle),
                    action: "process".into(),
                });
            }
            let engine = state.engine.take().ok_or_else(|| GuiError::InvalidState {
                state: "no_engine".into(),
                action: "process".into(),
            })?;
            (engine, state.config.conversion)
        };
        let result = engine.process(&input, config);
        let mut state = self.lock()?;
        state.engine = Some(engine);
        let _output = result.map_err(GuiError::from)?;
        state.metrics = state
            .engine
            .as_ref()
            .map(|engine| engine.metrics())
            .unwrap_or_default();
        let snapshot = state.snapshot();
        drop(state);
        self.events.send(BackendEvent::Metrics(snapshot.metrics))?;
        self.events.send(BackendEvent::Snapshot(snapshot.clone()))?;
        Ok(snapshot)
    }

    pub fn clear_error(&self) -> GuiResult<UiSnapshot> {
        let _operation = self.operation()?;
        let replacement = {
            let state = self.lock()?;
            if state.lifecycle == LifecycleState::Failed {
                state
                    .descriptor
                    .as_ref()
                    .map(|descriptor| Self::create_engine(&state.config, descriptor))
                    .transpose()?
            } else {
                None
            }
        };
        let snapshot = {
            let mut state = self.lock()?;
            if let Some(engine) = replacement {
                state.engine = Some(engine);
            }
            state.last_error = None;
            if state.lifecycle == LifecycleState::Failed {
                state.lifecycle = if state.engine.is_some() {
                    LifecycleState::Ready
                } else {
                    LifecycleState::Created
                };
            }
            state.snapshot()
        };
        self.events.send(BackendEvent::Snapshot(snapshot.clone()))?;
        Ok(snapshot)
    }

    fn transition(&self, next: LifecycleState, action: &str) -> GuiResult<()> {
        let snapshot = {
            let mut state = self.lock()?;
            if !state.lifecycle.can_transition_to(next) {
                return Err(GuiError::InvalidState {
                    state: format!("{:?}", state.lifecycle),
                    action: action.into(),
                });
            }
            state.lifecycle = next;
            state.snapshot()
        };
        self.events.send(BackendEvent::Snapshot(snapshot))?;
        Ok(())
    }

    fn fail(&self, error: GuiError) {
        let snapshot = if let Ok(mut state) = self.lock() {
            state.lifecycle = LifecycleState::Failed;
            state.last_error = Some(ErrorReport::from_error(&error));
            Some(state.snapshot())
        } else {
            None
        };
        let _ = self
            .events
            .send(BackendEvent::Error(ErrorReport::from_error(&error)));
        if let Some(snapshot) = snapshot {
            let _ = self.events.send(BackendEvent::Snapshot(snapshot));
        }
    }

    pub fn report(&self, error: &GuiError) {
        let snapshot = if let Ok(mut state) = self.lock() {
            state.last_error = Some(ErrorReport::from_error(error));
            if matches!(error, GuiError::Engine(_) | GuiError::Internal(_)) {
                state.lifecycle = LifecycleState::Failed;
            }
            Some(state.snapshot())
        } else {
            None
        };
        let _ = self
            .events
            .send(BackendEvent::Error(ErrorReport::from_error(error)));
        if let Some(snapshot) = snapshot {
            let _ = self.events.send(BackendEvent::Snapshot(snapshot));
        }
    }

    fn operation(&self) -> GuiResult<std::sync::MutexGuard<'_, ()>> {
        self.command_lock
            .lock()
            .map_err(|_| GuiError::Internal("command lock poisoned".into()))
    }

    fn lock(&self) -> GuiResult<std::sync::MutexGuard<'_, RuntimeState>> {
        self.inner
            .lock()
            .map_err(|_| GuiError::Internal("application state lock poisoned".into()))
    }

    fn create_engine(
        config: &GuiConfig,
        descriptor: &ModelDescriptor,
    ) -> GuiResult<Box<dyn InferenceEngine>> {
        let backend = match config.backend {
            GuiBackend::Passthrough => BackendKind::Passthrough,
            GuiBackend::RvcCpu => BackendKind::RvcCpuPipeline,
        };
        Ok(create_engine(EngineConfig {
            backend,
            model: descriptor.clone(),
        })?)
    }
}

fn descriptor_clone(state: &RuntimeState) -> Option<ModelDescriptor> {
    state.descriptor.clone()
}

impl RuntimeState {
    fn snapshot(&self) -> UiSnapshot {
        UiSnapshot {
            lifecycle: self.lifecycle,
            config: self.config.clone(),
            config_revision: self.config_revision,
            model: self.model.clone(),
            metrics: self.metrics.into(),
            engine: self.engine.as_ref().map(|engine| {
                let mut status = engine.status();
                status.state = self.lifecycle;
                status.into()
            }),
            last_error: self.last_error.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_revision_is_optimistic_and_monotonic() {
        let state = AppState::new(GuiConfig::default());
        let mut config = GuiConfig::default();
        config.backend = GuiBackend::RvcCpu;
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
}
