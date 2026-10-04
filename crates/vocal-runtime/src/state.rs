use std::sync::Arc;

use serde::Serialize;
use vocal_engine::{EngineConfig, EngineStatus, InferenceEngine, create_engine};
use vocal_model::{ModelLoader, ModelPackage, RvcModelManifest};
use vocal_types::{AudioChunk, LatencyMetrics, LifecycleState, ModelDescriptor};

use crate::{
    RuntimeConfig,
    error::{RuntimeError, RuntimeResult},
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

impl ModelSummary {
    fn from_descriptor(descriptor: &ModelDescriptor) -> Self {
        Self {
            id: descriptor.id.clone(),
            display_name: descriptor.display_name.clone(),
            architecture: descriptor.architecture.clone(),
            version: descriptor.version.clone(),
            sample_rate: descriptor.sample_rate,
            feature_dims: if descriptor.version.to_ascii_lowercase().contains("v2")
                || descriptor.architecture.to_ascii_lowercase().contains("v2")
            {
                768
            } else {
                256
            },
            uses_f0: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeMetrics {
    pub input_frames: u64,
    pub processing_micros: u64,
    pub output_frames: u64,
}

impl From<LatencyMetrics> for RuntimeMetrics {
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
pub struct RuntimeEngineStatus {
    pub state: LifecycleState,
    pub processed_chunks: u64,
    pub failures: u64,
    pub last_error: Option<String>,
}

impl From<EngineStatus> for RuntimeEngineStatus {
    fn from(status: EngineStatus) -> Self {
        Self {
            state: status.state,
            processed_chunks: status.processed_chunks,
            failures: status.failures,
            last_error: status.last_error,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeSnapshot {
    pub lifecycle: LifecycleState,
    pub config: RuntimeConfig,
    pub config_revision: u64,
    pub model: Option<ModelSummary>,
    pub metrics: RuntimeMetrics,
    pub engine: Option<RuntimeEngineStatus>,
    pub last_error: Option<RuntimeErrorInfo>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeErrorInfo {
    pub code: String,
    pub message: String,
}

struct RuntimeInner {
    lifecycle: LifecycleState,
    config: RuntimeConfig,
    config_revision: u64,
    model: Option<ModelSummary>,
    descriptor: Option<ModelDescriptor>,
    /// Keep the parsed package alive while its engine is selected. The model
    /// cache is weak by design, so this strong reference lets controllers
    /// share parsed packages without re-reading the file.
    model_package: Option<Arc<ModelPackage>>,
    engine: Option<Box<dyn InferenceEngine>>,
    metrics: LatencyMetrics,
    last_error: Option<RuntimeErrorInfo>,
    config_store: Option<crate::ConfigStore>,
}

#[derive(Clone)]
pub struct RuntimeContext {
    inner: std::sync::Arc<std::sync::Mutex<RuntimeInner>>,
    operation: std::sync::Arc<std::sync::Mutex<()>>,
}

impl RuntimeContext {
    pub fn new(config: RuntimeConfig) -> RuntimeResult<Self> {
        config.validate()?;
        Ok(Self {
            inner: std::sync::Arc::new(std::sync::Mutex::new(RuntimeInner {
                lifecycle: LifecycleState::Created,
                config,
                config_revision: 0,
                model: None,
                descriptor: None,
                model_package: None,
                engine: None,
                metrics: LatencyMetrics::default(),
                last_error: None,
                config_store: None,
            })),
            operation: std::sync::Arc::new(std::sync::Mutex::new(())),
        })
    }

    pub fn set_config_store(&self, store: crate::ConfigStore) -> RuntimeResult<()> {
        let _operation = self.operation()?;
        let loaded = store.load()?;
        let mut state = self.lock()?;
        state.config_store = Some(store);
        if let Some(config) = loaded {
            state.config = config;
            state.config_revision = state.config_revision.saturating_add(1);
        }
        Ok(())
    }

    pub fn config(&self) -> RuntimeResult<RuntimeConfig> {
        Ok(self.lock()?.config.clone())
    }

    pub fn snapshot(&self) -> RuntimeResult<RuntimeSnapshot> {
        Ok(self.lock()?.snapshot())
    }

    pub fn persisted_model_path(&self) -> RuntimeResult<Option<String>> {
        Ok(self.lock()?.config.model_path.clone())
    }

    pub fn update_config(
        &self,
        config: RuntimeConfig,
        expected_revision: u64,
    ) -> RuntimeResult<RuntimeSnapshot> {
        let _operation = self.operation()?;
        config.validate()?;
        let (descriptor, store) = {
            let state = self.lock()?;
            self.check_revision(&state, expected_revision)?;
            if matches!(
                state.lifecycle,
                LifecycleState::Running | LifecycleState::Stopping
            ) {
                return Err(RuntimeError::State {
                    state: format!("{:?}", state.lifecycle),
                    action: "update_config".into(),
                });
            }
            (state.descriptor.clone(), state.config_store.clone())
        };
        let replacement = descriptor
            .as_ref()
            .map(|descriptor| create_runtime_engine(&config, descriptor))
            .transpose()?;
        if let Some(store) = &store {
            store.save(&config)?;
        }
        let mut state = self.lock()?;
        self.check_revision(&state, expected_revision)?;
        state.config = config;
        state.config_revision = state.config_revision.saturating_add(1);
        if replacement.is_some() {
            state.engine = replacement;
        }
        Ok(state.snapshot())
    }

    pub fn load_model(&self, path: impl Into<String>) -> RuntimeResult<RuntimeSnapshot> {
        let _operation = self.operation()?;
        self.transition(LifecycleState::Loading, "load_model")?;
        match self.load_model_inner(path.into()) {
            Ok(snapshot) => Ok(snapshot),
            Err(error) => {
                self.fail(&error);
                Err(error)
            }
        }
    }

    pub fn unload_model(&self) -> RuntimeResult<RuntimeSnapshot> {
        let _operation = self.operation()?;
        let mut state = self.lock()?;
        if matches!(
            state.lifecycle,
            LifecycleState::Loading | LifecycleState::Running | LifecycleState::Stopping
        ) {
            return Err(RuntimeError::State {
                state: format!("{:?}", state.lifecycle),
                action: "unload_model".into(),
            });
        }
        state.engine = None;
        state.model = None;
        state.descriptor = None;
        state.model_package = None;
        state.config.model_path = None;
        state.metrics = LatencyMetrics::default();
        state.config_revision = state.config_revision.saturating_add(1);
        if let Some(store) = &state.config_store {
            store.save(&state.config)?;
        }
        state.lifecycle = LifecycleState::Created;
        state.last_error = None;
        Ok(state.snapshot())
    }

    pub fn process(&self, input: &AudioChunk) -> RuntimeResult<AudioChunk> {
        let _operation = self.operation()?;
        if input.samples.len() > MAX_CONTROL_BLOCK_SAMPLES {
            return Err(RuntimeError::Config(format!(
                "control block exceeds {MAX_CONTROL_BLOCK_SAMPLES} samples"
            )));
        }
        input.validate()?;
        let mut state = self.lock()?;
        if state.lifecycle != LifecycleState::Running {
            return Err(RuntimeError::State {
                state: format!("{:?}", state.lifecycle),
                action: "process".into(),
            });
        }
        let config = state.config.conversion;
        let engine = state.engine.as_mut().ok_or_else(|| RuntimeError::State {
            state: "no_engine".into(),
            action: "process".into(),
        })?;
        let result = engine.process(input, config);
        match result {
            Ok(output) => {
                state.metrics = engine.metrics();
                Ok(output)
            }
            Err(error) => {
                state.last_error = Some(RuntimeErrorInfo {
                    code: error.code().into(),
                    message: error.to_string(),
                });
                state.lifecycle = LifecycleState::Failed;
                Err(error.into())
            }
        }
    }

    /// Install an already-resolved model descriptor and construct the
    /// configured engine. This keeps descriptor based adapters and file based
    /// model loading on the same lifecycle and processing API.
    pub fn load_model_descriptor(
        &self,
        descriptor: ModelDescriptor,
    ) -> RuntimeResult<RuntimeSnapshot> {
        let _operation = self.operation()?;
        self.transition(LifecycleState::Loading, "load_model_descriptor")?;
        let result = (|| {
            descriptor.validate()?;
            let config = self.lock()?.config.clone();
            let engine = create_runtime_engine(&config, &descriptor)?;
            let store = self.lock()?.config_store.clone();
            if let Some(store) = &store {
                let mut persisted = config.clone();
                persisted.model_path = None;
                store.save(&persisted)?;
            }
            let mut state = self.lock()?;
            state.config.model_path = None;
            state.config_revision = state.config_revision.saturating_add(1);
            state.model = Some(ModelSummary::from_descriptor(&descriptor));
            state.descriptor = Some(descriptor);
            state.model_package = None;
            state.engine = Some(engine);
            state.lifecycle = LifecycleState::Ready;
            state.last_error = None;
            Ok(state.snapshot())
        })();
        match result {
            Ok(snapshot) => Ok(snapshot),
            Err(error) => {
                self.fail(&error);
                Err(error)
            }
        }
    }

    pub fn start_processing(&self) -> RuntimeResult<RuntimeSnapshot> {
        let _operation = self.operation()?;
        let mut state = self.lock()?;
        if state.descriptor.is_none() || state.engine.is_none() {
            return Err(RuntimeError::State {
                state: format!("{:?}", state.lifecycle),
                action: "start_without_model".into(),
            });
        }
        if !state.lifecycle.can_transition_to(LifecycleState::Running) {
            return Err(RuntimeError::State {
                state: format!("{:?}", state.lifecycle),
                action: "start".into(),
            });
        }
        state.lifecycle = LifecycleState::Running;
        state.last_error = None;
        Ok(state.snapshot())
    }

    pub fn stop_processing(&self) -> RuntimeResult<RuntimeSnapshot> {
        let _operation = self.operation()?;
        let mut state = self.lock()?;
        if !matches!(
            state.lifecycle,
            LifecycleState::Running | LifecycleState::Ready
        ) {
            return Err(RuntimeError::State {
                state: format!("{:?}", state.lifecycle),
                action: "stop".into(),
            });
        }
        state.lifecycle = LifecycleState::Stopped;
        Ok(state.snapshot())
    }

    /// Roll back a start attempt when the device adapter could not open. The
    /// engine remains loaded and ready for another device attempt.
    pub fn abort_processing_start(&self) -> RuntimeResult<RuntimeSnapshot> {
        let _operation = self.operation()?;
        let mut state = self.lock()?;
        if state.engine.is_some() {
            state.lifecycle = LifecycleState::Ready;
        } else {
            state.lifecycle = LifecycleState::Created;
        }
        Ok(state.snapshot())
    }

    /// Publish a fatal adapter failure into the shared runtime state. The
    /// controller decides how to transport the resulting snapshot.
    pub fn report_failure(
        &self,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> RuntimeResult<RuntimeSnapshot> {
        let _operation = self.operation()?;
        let mut state = self.lock()?;
        state.lifecycle = LifecycleState::Failed;
        state.last_error = Some(RuntimeErrorInfo {
            code: code.into(),
            message: message.into(),
        });
        Ok(state.snapshot())
    }

    pub fn clear_error(&self) -> RuntimeResult<RuntimeSnapshot> {
        let _operation = self.operation()?;
        let mut state = self.lock()?;
        state.last_error = None;
        if state.lifecycle == LifecycleState::Failed {
            state.lifecycle = if state.engine.is_some() {
                LifecycleState::Ready
            } else {
                LifecycleState::Created
            };
        }
        Ok(state.snapshot())
    }

    fn load_model_inner(&self, path: String) -> RuntimeResult<RuntimeSnapshot> {
        let input = std::path::PathBuf::from(&path);
        if !input.is_file() {
            return Err(RuntimeError::Config(format!(
                "model path does not exist: {path}"
            )));
        }
        let mut loader = ModelLoader::new(Default::default())?;
        let (package, manifest) = if input
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("vfr"))
        {
            let package = loader.load_cache_shared(&input)?;
            (package.clone(), package.manifest.clone())
        } else {
            let manifest = loader.load_manifest(&input)?;
            let root = input.parent().unwrap_or_else(|| std::path::Path::new("."));
            let package = Arc::new(loader.load_rvc_directory(root, manifest.clone())?);
            (package, manifest)
        };
        let descriptor = manifest.descriptor();
        let mut config = self.lock()?.config.clone();
        config.model_path = Some(path);
        let engine = create_runtime_engine(&config, &descriptor)?;
        let store = self.lock()?.config_store.clone();
        if let Some(store) = &store {
            store.save(&config)?;
        }
        let mut state = self.lock()?;
        state.config = config;
        state.config_revision = state.config_revision.saturating_add(1);
        state.model = Some(ModelSummary::from(&manifest));
        state.descriptor = Some(descriptor);
        state.model_package = Some(package);
        state.engine = Some(engine);
        state.lifecycle = LifecycleState::Ready;
        state.last_error = None;
        Ok(state.snapshot())
    }

    fn transition(&self, next: LifecycleState, action: &str) -> RuntimeResult<()> {
        let mut state = self.lock()?;
        if !state.lifecycle.can_transition_to(next) {
            return Err(RuntimeError::State {
                state: format!("{:?}", state.lifecycle),
                action: action.into(),
            });
        }
        state.lifecycle = next;
        Ok(())
    }

    fn fail(&self, error: &RuntimeError) {
        if let Ok(mut state) = self.inner.lock() {
            state.lifecycle = LifecycleState::Failed;
            state.last_error = Some(RuntimeErrorInfo {
                code: error.code().into(),
                message: error.to_string(),
            });
        }
    }

    fn check_revision(&self, state: &RuntimeInner, expected: u64) -> RuntimeResult<()> {
        if state.config_revision != expected {
            return Err(RuntimeError::ConfigConflict {
                expected,
                actual: state.config_revision,
            });
        }
        Ok(())
    }

    fn lock(&self) -> RuntimeResult<std::sync::MutexGuard<'_, RuntimeInner>> {
        self.inner.lock().map_err(|_| RuntimeError::Lock)
    }

    fn operation(&self) -> RuntimeResult<std::sync::MutexGuard<'_, ()>> {
        self.operation.lock().map_err(|_| RuntimeError::Lock)
    }
}

fn create_runtime_engine(
    config: &RuntimeConfig,
    descriptor: &ModelDescriptor,
) -> RuntimeResult<Box<dyn InferenceEngine>> {
    Ok(create_engine(EngineConfig {
        backend: config.backend.engine_kind(),
        model: descriptor.clone(),
    })?)
}

impl RuntimeInner {
    fn snapshot(&self) -> RuntimeSnapshot {
        RuntimeSnapshot {
            lifecycle: self.lifecycle,
            config: self.config.clone(),
            config_revision: self.config_revision,
            model: self.model.clone(),
            metrics: self.metrics.into(),
            engine: self.engine.as_ref().map(|engine| {
                let mut status: RuntimeEngineStatus = engine.status().into();
                status.state = self.lifecycle;
                status
            }),
            last_error: self.last_error.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "fixture".into(),
            display_name: "Fixture".into(),
            architecture: "passthrough".into(),
            sample_rate: 16_000,
            version: "0.1".into(),
        }
    }

    #[test]
    fn context_starts_without_model_and_rejects_processing() {
        let context = RuntimeContext::new(RuntimeConfig::default()).expect("context");
        let error = context.start_processing().expect_err("missing model");
        assert!(matches!(error, RuntimeError::State { .. }));
    }

    #[test]
    fn descriptor_load_uses_shared_lifecycle_and_supports_restart() {
        let format = vocal_types::AudioFormat::MONO_16K;
        let context = RuntimeContext::new(RuntimeConfig {
            input_format: format,
            output_format: format,
            ..RuntimeConfig::default()
        })
        .expect("context");
        let snapshot = context
            .load_model_descriptor(descriptor())
            .expect("load descriptor");
        assert_eq!(snapshot.lifecycle, LifecycleState::Ready);
        context.start_processing().expect("start");
        let output = context
            .process(&AudioChunk {
                format,
                samples: vec![0.0; 160],
                sequence: 4,
            })
            .expect("process");
        assert_eq!(output.sequence, 4);
        assert_eq!(output.samples.len(), 160);
        context.stop_processing().expect("stop");
        context.start_processing().expect("restart loaded engine");
    }

    #[test]
    fn stale_config_revision_is_rejected_without_mutation() {
        let context = RuntimeContext::new(RuntimeConfig::default()).expect("context");
        let mut config = RuntimeConfig::default();
        config.queue_blocks = 2;
        let snapshot = context.update_config(config, 0).expect("update");
        assert_eq!(snapshot.config_revision, 1);
        let error = context
            .update_config(RuntimeConfig::default(), 0)
            .expect_err("stale update");
        assert!(matches!(error, RuntimeError::ConfigConflict { .. }));
        assert_eq!(context.config().unwrap().queue_blocks, 2);
    }

    #[test]
    fn clear_error_restores_loaded_engine_to_ready() {
        let context = RuntimeContext::new(RuntimeConfig::default()).expect("context");
        context.load_model_descriptor(descriptor()).expect("load");
        context
            .report_failure("test.failure", "test failure")
            .expect("report");
        let snapshot = context.clear_error().expect("clear");
        assert_eq!(snapshot.lifecycle, LifecycleState::Ready);
        assert!(snapshot.last_error.is_none());
    }
}
