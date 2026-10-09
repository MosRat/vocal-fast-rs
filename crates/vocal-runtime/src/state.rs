use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::UNIX_EPOCH,
};

use serde::Serialize;
use vocal_engine::{
    EngineConfig, EngineReloadConfig, EngineStatus, InferenceEngine, TensorRtReloadConfig,
    create_engine,
};
use vocal_model::{
    ModelArtifactKind, ModelLoadConfig, ModelLoader, ModelPackage, RvcModelManifest,
    inspect_model_artifact_with_config, model_artifact_kind, resolve_model_path,
};
use vocal_types::{AudioChunk, LatencyMetrics, LifecycleState, ModelDescriptor};

use crate::{
    ConfigUpdateKind, RuntimeBackend, RuntimeConfig, classify_config_change,
    error::{RuntimeError, RuntimeResult},
};

const MAX_CONTROL_BLOCK_SAMPLES: usize = 1_048_576;

#[cfg(test)]
#[path = "idle_reload_tests.rs"]
mod idle_reload_tests;

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
            feature_dims: vocal_model::RvcVersion::from_labels(
                &descriptor.version,
                &descriptor.architecture,
            )
            .map(|version| version.feature_width())
            .unwrap_or(0),
            uses_f0: true,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCatalogEntry {
    pub path: String,
    pub kind: ModelArtifactKind,
    pub size_bytes: u64,
    pub modified_at_ms: u64,
    pub components: vocal_model::ModelComponents,
    #[serde(flatten)]
    pub model: ModelSummary,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCatalogIssue {
    pub path: String,
    pub message: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCatalog {
    pub root: String,
    pub entries: Vec<ModelCatalogEntry>,
    pub issues: Vec<ModelCatalogIssue>,
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
    pub last_config_update: ConfigUpdateKind,
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
    last_config_update: ConfigUpdateKind,
    model: Option<ModelSummary>,
    descriptor: Option<ModelDescriptor>,
    /// Canonical path of the package currently resident in `model_package`.
    /// `RuntimeConfig::model_path` is the persisted selection and may point
    /// at a pending replacement, so it must not be used to decide whether an
    /// existing TensorRT engine is compatible with newly published plans.
    loaded_model_path: Option<String>,
    /// Keep the parsed package alive while its engine is selected. The model
    /// cache is weak by design, so this strong reference lets controllers
    /// share parsed packages without re-reading the file.
    model_package: Option<Arc<ModelPackage>>,
    /// The control actor owns this engine while no audio model worker is
    /// running. During an active stream ownership moves to that worker.
    engine: Option<Box<dyn InferenceEngine>>,
    /// Configuration actually applied to the resident engine. Saved plan/model
    /// selections may be pending and must not become the reload baseline.
    engine_config: Option<RuntimeConfig>,
    worker_status: Option<EngineStatus>,
    worker_generation: u64,
    worker_active: bool,
    metrics: LatencyMetrics,
    last_error: Option<RuntimeErrorInfo>,
    config_store: Option<crate::ConfigStore>,
    /// Heavy operations may free gigabytes on this long-lived actor. Collect
    /// only after the command and its candidate temporaries have been dropped.
    collect_memory: bool,
}

type RuntimeCommand = Box<dyn FnOnce(&mut RuntimeInner) + Send + 'static>;

/// Thread-safe handle for the runtime control actor.
///
/// Calls enqueue bounded commands and wait for a typed reply. The actor is the
/// only owner of mutable lifecycle state; no controller shares an engine
/// reference across threads.
#[derive(Clone)]
pub struct RuntimeContext {
    control: std::sync::mpsc::SyncSender<RuntimeCommand>,
}

impl RuntimeContext {
    const CONTROL_CAPACITY: usize = 64;

    pub fn new(config: RuntimeConfig) -> RuntimeResult<Self> {
        config.validate()?;
        let (control, commands) = std::sync::mpsc::sync_channel(Self::CONTROL_CAPACITY);
        let mut state = RuntimeInner {
            lifecycle: LifecycleState::Created,
            config,
            config_revision: 0,
            last_config_update: ConfigUpdateKind::Unchanged,
            model: None,
            descriptor: None,
            loaded_model_path: None,
            model_package: None,
            engine: None,
            engine_config: None,
            worker_status: None,
            worker_generation: 0,
            worker_active: false,
            metrics: LatencyMetrics::default(),
            last_error: None,
            config_store: None,
            collect_memory: false,
        };
        std::thread::Builder::new()
            .name("vocal-runtime-control".into())
            .spawn(move || runtime_actor(commands, &mut state))
            .map_err(|error| RuntimeError::ControlClosed(error.to_string()))?;
        Ok(Self { control })
    }

    fn request<T, F>(&self, command: F) -> RuntimeResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut RuntimeInner) -> RuntimeResult<T> + Send + 'static,
    {
        let (reply, result) = std::sync::mpsc::sync_channel(1);
        self.control
            .send(Box::new(move |state| {
                let outcome = command(state);
                // The command has dropped temporary/candidate resources. A
                // load/unload reply promises this reclamation is complete,
                // rather than racing the GUI's completed-state notification.
                if std::mem::take(&mut state.collect_memory) {
                    crate::memory::collect_idle();
                }
                let _ = reply.send(outcome);
            }))
            .map_err(|_| RuntimeError::ControlClosed("runtime control actor stopped".into()))?;
        result
            .recv()
            .map_err(|_| RuntimeError::ControlClosed("runtime control response dropped".into()))?
    }

    pub fn set_config_store(&self, store: crate::ConfigStore) -> RuntimeResult<()> {
        self.request(move |state| state.set_config_store(store))
    }

    /// Resolve a fresh/automatic backend, or recover a saved unavailable one,
    /// before model restoration. Runtime configuration remains authoritative.
    pub fn initialize_processing_backend(&self) -> RuntimeResult<Option<String>> {
        self.initialize_processing_backend_with_catalog(crate::detect_processing_backends())
    }

    /// Reuse a controller's startup probe without probing device drivers twice.
    pub fn initialize_processing_backend_with_catalog(
        &self,
        catalog: crate::ProcessingBackendCatalog,
    ) -> RuntimeResult<Option<String>> {
        self.request(move |state| state.initialize_processing_backend(catalog))
    }

    pub fn config(&self) -> RuntimeResult<RuntimeConfig> {
        self.request(|state| Ok(state.config.clone()))
    }

    pub fn snapshot(&self) -> RuntimeResult<RuntimeSnapshot> {
        self.request(|state| Ok(state.snapshot()))
    }

    pub fn persisted_model_path(&self) -> RuntimeResult<Option<String>> {
        self.request(|state| Ok(state.config.model_path.clone()))
    }

    pub fn scan_model_directory(&self, root: impl Into<String>) -> RuntimeResult<ModelCatalog> {
        // Directory inspection is read-only and can touch slow or
        // antivirus-scanned files. Keep it off the control actor so a catalog
        // refresh cannot delay lifecycle commands or worker ownership handoff.
        scan_model_directory_impl(root.into())
    }

    pub fn update_config(
        &self,
        config: RuntimeConfig,
        expected_revision: u64,
    ) -> RuntimeResult<RuntimeSnapshot> {
        self.request(move |state| state.update_config(config, expected_revision))
    }

    /// Update display metadata while preserving the resident engine and tensor ownership.
    pub fn edit_model_metadata(
        &self,
        path: String,
        revision: String,
        edit: vocal_model::ModelMetadataEdit,
    ) -> RuntimeResult<vocal_model::ModelMetadataView> {
        self.request(move |state| {
            if state.worker_status.is_some()
                || state.worker_active
                || matches!(
                    state.lifecycle,
                    LifecycleState::Running | LifecycleState::Loading | LifecycleState::Stopping
                )
            {
                return Err(RuntimeError::State {
                    state: "audio_session_active".into(),
                    action: "edit_model_metadata".into(),
                });
            }
            let view = vocal_model::edit_model_metadata(path, &revision, edit)?;
            if state.loaded_model_path.as_deref() == Some(view.inspection.path.as_str()) {
                state.model = Some(ModelSummary::from(&view.inspection.manifest));
                state.descriptor = Some(view.inspection.manifest.descriptor());
            }
            Ok(view)
        })
    }

    /// Commit a successfully built/published TensorRT manifest, including a
    /// rebuild at the same path. The idle engine prepares replacement internal
    /// contexts before persistence; changing models still requires Model Load.
    pub fn apply_prepared_tensorrt_pipeline(
        &self,
        config: RuntimeConfig,
        expected_revision: u64,
    ) -> RuntimeResult<RuntimeSnapshot> {
        self.request(move |state| state.update_config_inner(config, expected_revision, true))
    }

    pub fn load_model(&self, path: impl Into<String>) -> RuntimeResult<RuntimeSnapshot> {
        let path = path.into();
        self.request(move |state| state.load_model(path, None))
    }

    pub fn load_model_with_backend(
        &self,
        path: impl Into<String>,
        selection: crate::ModelBackendSelection,
    ) -> RuntimeResult<RuntimeSnapshot> {
        let path = path.into();
        self.request(move |state| state.load_model(path, Some(selection)))
    }

    /// Commit a prepared model and its derived configuration in one actor command.
    /// Engine construction and persistence succeed before any prior state is replaced.
    pub fn load_model_with_config(
        &self,
        path: impl Into<String>,
        config: RuntimeConfig,
        expected_revision: u64,
    ) -> RuntimeResult<RuntimeSnapshot> {
        let path = path.into();
        self.request(move |state| {
            state.check_revision(expected_revision)?;
            if state.worker_status.is_some() {
                return Err(RuntimeError::State {
                    state: "model_worker_owns_engine".into(),
                    action: "load_model".into(),
                });
            }
            config.validate()?;
            crate::detect_processing_backends().ensure_available(config.backend)?;
            state.load_model_config(path, config)
        })
    }

    pub fn unload_model(&self) -> RuntimeResult<RuntimeSnapshot> {
        self.request(RuntimeInner::unload_model)
    }

    /// Process a synchronous, non-audio block through the control actor. The
    /// live audio path uses take_engine_for_worker instead, so it never sends
    /// samples through this command channel.
    pub fn process(&self, input: &AudioChunk) -> RuntimeResult<AudioChunk> {
        let input = input.clone();
        self.request(move |state| state.process(&input))
    }

    pub fn load_model_descriptor(
        &self,
        descriptor: ModelDescriptor,
    ) -> RuntimeResult<RuntimeSnapshot> {
        self.request(move |state| state.load_model_descriptor(descriptor))
    }

    pub fn start_processing(&self) -> RuntimeResult<RuntimeSnapshot> {
        self.request(RuntimeInner::start_processing)
    }

    pub fn stop_processing(&self) -> RuntimeResult<RuntimeSnapshot> {
        self.request(RuntimeInner::stop_processing)
    }

    pub fn abort_processing_start(&self) -> RuntimeResult<RuntimeSnapshot> {
        self.request(RuntimeInner::abort_processing_start)
    }

    pub fn report_failure(
        &self,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> RuntimeResult<RuntimeSnapshot> {
        let code = code.into();
        let message = message.into();
        self.request(move |state| state.report_failure(code, message))
    }

    pub fn clear_error(&self) -> RuntimeResult<RuntimeSnapshot> {
        self.request(RuntimeInner::clear_error)
    }

    /// Transfer engine ownership to the dedicated audio/model worker after
    /// the lifecycle has entered Running.
    pub fn take_engine_for_worker(&self) -> RuntimeResult<(u64, Box<dyn InferenceEngine>)> {
        self.request(RuntimeInner::take_engine_for_worker)
    }

    /// Query idle engine timing before opening the device graph. No engine
    /// reference leaves the actor; active workers own their timing and engine.
    pub fn processing_batch_frames(&self) -> RuntimeResult<usize> {
        self.request(|state| {
            let engine = state.engine.as_ref().ok_or_else(|| RuntimeError::State {
                state: "engine_unavailable".into(),
                action: "processing_batch_frames".into(),
            })?;
            Ok(engine.preferred_batch_frames(state.config.input_format))
        })
    }

    /// Mark a reserved worker lease as started. A separate activation step
    /// lets a failed thread spawn roll back without racing a stale worker.
    pub fn activate_worker(&self, generation: u64) -> RuntimeResult<()> {
        self.request(move |state| state.activate_worker(generation))
    }

    /// Return the engine after the model worker exits. This is a control
    /// message, so the worker never touches shared runtime state directly.
    pub fn complete_worker(
        &self,
        generation: u64,
        engine: Box<dyn InferenceEngine>,
        failure: Option<(String, String)>,
    ) -> RuntimeResult<RuntimeSnapshot> {
        self.request(move |state| state.complete_worker(generation, engine, failure))
    }
}

fn runtime_actor(commands: std::sync::mpsc::Receiver<RuntimeCommand>, state: &mut RuntimeInner) {
    while let Ok(command) = commands.recv() {
        command(state);
    }
}

impl RuntimeInner {
    fn initialize_processing_backend(
        &mut self,
        catalog: crate::ProcessingBackendCatalog,
    ) -> RuntimeResult<Option<String>> {
        if self.descriptor.is_some() || self.worker_status.is_some() {
            return Err(RuntimeError::State {
                state: "model_already_loaded".into(),
                action: "initialize_processing_backend".into(),
            });
        }
        let unavailable = catalog.ensure_available(self.config.backend).err();
        if self.config.backend_auto || unavailable.is_some() {
            let mut config = self.config.clone();
            config.backend = catalog.recommended;
            if let Some(store) = &self.config_store {
                store.save(&config)?;
            }
            self.config = config;
            self.config_revision = self.config_revision.saturating_add(1);
        }
        Ok(unavailable.map(|error| {
            format!(
                "The saved processing backend is unavailable ({error}). Using {:?}.",
                self.config.backend
            )
        }))
    }

    fn set_config_store(&mut self, store: crate::ConfigStore) -> RuntimeResult<()> {
        if self.descriptor.is_some() || self.worker_status.is_some() {
            return Err(RuntimeError::State {
                state: "model_already_loaded".into(),
                action: "set_config_store".into(),
            });
        }
        let loaded = store.load()?;
        self.config_store = Some(store);
        if let Some(config) = loaded {
            self.config = config;
            self.config_revision = self.config_revision.saturating_add(1);
        }
        Ok(())
    }

    fn update_config(
        &mut self,
        config: RuntimeConfig,
        expected_revision: u64,
    ) -> RuntimeResult<RuntimeSnapshot> {
        self.update_config_inner(config, expected_revision, false)
    }

    fn update_config_inner(
        &mut self,
        config: RuntimeConfig,
        expected_revision: u64,
        pipeline_prepared: bool,
    ) -> RuntimeResult<RuntimeSnapshot> {
        config.validate()?;
        self.check_revision(expected_revision)?;
        if pipeline_prepared
            && (config.backend != RuntimeBackend::TensorRtRtx
                || config.tensorrt_artifacts().is_none())
        {
            return Err(RuntimeError::Config(
                "prepared pipeline requires TensorRT and a published artifacts manifest".into(),
            ));
        }
        if self.worker_status.is_some()
            || matches!(
                self.lifecycle,
                LifecycleState::Running | LifecycleState::Stopping
            )
        {
            return Err(RuntimeError::State {
                state: format!("{:?}", self.lifecycle),
                action: "update_config".into(),
            });
        }
        let model_needs_reload = self.descriptor.is_some()
            && config.model_path.as_deref() != self.loaded_model_path.as_deref();
        if config.backend != self.config.backend {
            crate::detect_processing_backends().ensure_available(config.backend)?;
        }
        let pending_plans =
            config.backend == RuntimeBackend::TensorRtRtx && config.tensorrt_artifacts().is_none();
        let mut change = if model_needs_reload {
            ConfigUpdateKind::LoadModelRequired
        } else {
            classify_config_change(
                self.engine_config.as_ref().unwrap_or(&self.config),
                &config,
                self.descriptor.is_some(),
            )
        };
        if pipeline_prepared && !model_needs_reload && self.descriptor.is_some() {
            change = if self.engine.is_some()
                && self
                    .engine_config
                    .as_ref()
                    .is_some_and(|applied| applied.backend == RuntimeBackend::TensorRtRtx)
            {
                ConfigUpdateKind::ReloadTensorRtPipeline
            } else {
                ConfigUpdateKind::RebuildEngine
            };
        }
        if !model_needs_reload && self.descriptor.is_some() && self.engine.is_none() {
            change = ConfigUpdateKind::RebuildEngine;
        }
        if pending_plans && !model_needs_reload {
            change = ConfigUpdateKind::PlanBuildRequired;
        }
        if matches!(
            change,
            ConfigUpdateKind::RebuildEngine | ConfigUpdateKind::ReloadTensorRtPipeline
        ) || (pending_plans
            && self.engine.is_some()
            && self
                .engine_config
                .as_ref()
                .is_none_or(|applied| applied.backend != config.backend))
        {
            // Mark before preparation so failed candidates are reclaimed too.
            self.collect_memory = true;
        }
        let replacement = if change == ConfigUpdateKind::RebuildEngine {
            self.descriptor
                .as_ref()
                .map(|descriptor| {
                    create_runtime_engine(&config, descriptor, self.model_package.as_deref())
                })
                .transpose()?
        } else {
            None
        };
        if !matches!(
            change,
            ConfigUpdateKind::LoadModelRequired | ConfigUpdateKind::PlanBuildRequired
        ) && let Some(engine) = replacement.as_ref().or(self.engine.as_ref())
        {
            engine.validate_config(config.conversion)?;
        }
        // A prepared reload holds the old engine unchanged until persistence
        // succeeds. Dropping it on save failure also drops all candidate GPU state.
        {
            let prepared = if change == ConfigUpdateKind::ReloadTensorRtPipeline {
                let reload = engine_reload_config(&config)?;
                self.engine
                    .as_mut()
                    .map(|engine| engine.prepare_reload_config(reload))
                    .transpose()?
            } else {
                None
            };
            if let Some(store) = &self.config_store {
                store.save(&config)?;
            }
            if let Some(prepared) = prepared {
                prepared.commit();
            }
        }
        if !matches!(
            change,
            ConfigUpdateKind::LoadModelRequired | ConfigUpdateKind::PlanBuildRequired
        ) && (replacement.is_some() || self.engine.is_some())
        {
            self.engine_config = Some(config.clone());
        }
        self.config = config;
        self.config_revision = self.config_revision.saturating_add(1);
        self.last_config_update = change;
        if replacement.is_some() {
            self.engine = replacement;
            self.lifecycle = LifecycleState::Ready;
            self.last_error = None;
        } else if change == ConfigUpdateKind::ReloadTensorRtPipeline && self.engine.is_some() {
            self.lifecycle = LifecycleState::Ready;
            self.last_error = None;
        } else if pending_plans
            && self
                .engine_config
                .as_ref()
                .is_none_or(|applied| applied.backend != self.config.backend)
        {
            // Keep the parsed model resident but never run the old backend
            // under the newly selected backend's name.
            self.engine = None;
            self.engine_config = None;
            self.lifecycle = LifecycleState::Created;
        }
        Ok(self.snapshot())
    }

    fn load_model(
        &mut self,
        path: String,
        selection: Option<crate::ModelBackendSelection>,
    ) -> RuntimeResult<RuntimeSnapshot> {
        if self.worker_status.is_some() {
            return Err(RuntimeError::State {
                state: "model_worker_owns_engine".into(),
                action: "load_model".into(),
            });
        }
        let mut config = self.config.clone();
        if let Some(selection) = selection {
            self.check_revision(selection.expected_revision)?;
            let catalog = crate::detect_processing_backends();
            config.backend = if selection.automatic {
                catalog.recommended
            } else {
                selection.backend
            };
            catalog.ensure_available(config.backend)?;
            config.backend_auto = selection.automatic;
            config.tensorrt.artifacts_path = selection.artifacts_path.clone();
            config.tensorrt_artifacts_path = selection.artifacts_path;
            if config.backend == RuntimeBackend::DirectMl {
                config.directml_sources_path = selection.directml_sources_path;
                config.directml_device_id = selection.directml_device_id;
            }
            config.validate()?;
        }
        self.load_model_config(path, config)
    }

    fn load_model_config(
        &mut self,
        path: String,
        config: RuntimeConfig,
    ) -> RuntimeResult<RuntimeSnapshot> {
        if self.worker_status.is_some() {
            return Err(RuntimeError::State {
                state: "model_worker_owns_engine".into(),
                action: "load_model".into(),
            });
        }
        self.transition(LifecycleState::Loading, "load_model")?;
        self.collect_memory = true;
        match self.load_model_inner(path, config) {
            Ok(snapshot) => Ok(snapshot),
            Err(error) => {
                self.fail(&error);
                Err(error)
            }
        }
    }

    fn unload_model(&mut self) -> RuntimeResult<RuntimeSnapshot> {
        if self.worker_status.is_some()
            || matches!(
                self.lifecycle,
                LifecycleState::Loading | LifecycleState::Running | LifecycleState::Stopping
            )
        {
            return Err(RuntimeError::State {
                state: format!("{:?}", self.lifecycle),
                action: "unload_model".into(),
            });
        }
        let mut config = self.config.clone();
        config.model_path = None;
        if let Some(store) = &self.config_store {
            store.save(&config)?;
        }
        self.collect_memory = true;
        self.engine = None;
        self.engine_config = None;
        self.worker_status = None;
        self.worker_active = false;
        self.worker_generation = self.worker_generation.wrapping_add(1);
        self.model = None;
        self.descriptor = None;
        self.loaded_model_path = None;
        self.model_package = None;
        self.config = config;
        self.metrics = LatencyMetrics::default();
        self.config_revision = self.config_revision.saturating_add(1);
        self.last_config_update = ConfigUpdateKind::Unchanged;
        self.lifecycle = LifecycleState::Created;
        self.last_error = None;
        Ok(self.snapshot())
    }

    fn process(&mut self, input: &AudioChunk) -> RuntimeResult<AudioChunk> {
        if input.samples.len() > MAX_CONTROL_BLOCK_SAMPLES {
            return Err(RuntimeError::Config(format!(
                "control block exceeds {MAX_CONTROL_BLOCK_SAMPLES} samples"
            )));
        }
        input.validate()?;
        if self.lifecycle != LifecycleState::Running {
            return Err(RuntimeError::State {
                state: format!("{:?}", self.lifecycle),
                action: "process".into(),
            });
        }
        let config = self.config.conversion;
        let engine = self.engine.as_mut().ok_or_else(|| RuntimeError::State {
            state: "model_worker_owns_engine".into(),
            action: "process".into(),
        })?;
        match engine.process(input, config) {
            Ok(output) => {
                self.metrics = engine.metrics();
                Ok(output)
            }
            Err(error) => {
                self.last_error = Some(RuntimeErrorInfo {
                    code: error.code().into(),
                    message: error.to_string(),
                });
                self.lifecycle = LifecycleState::Failed;
                Err(error.into())
            }
        }
    }

    fn load_model_descriptor(
        &mut self,
        descriptor: ModelDescriptor,
    ) -> RuntimeResult<RuntimeSnapshot> {
        self.transition(LifecycleState::Loading, "load_model_descriptor")?;
        self.collect_memory = true;
        let result = (|| {
            descriptor.validate()?;
            let config = self.config.clone();
            let engine = create_runtime_engine(&config, &descriptor, None)?;
            if let Some(store) = &self.config_store {
                let mut persisted = config.clone();
                persisted.model_path = None;
                store.save(&persisted)?;
            }
            self.config.model_path = None;
            self.config_revision = self.config_revision.saturating_add(1);
            self.model = Some(ModelSummary::from_descriptor(&descriptor));
            self.descriptor = Some(descriptor);
            self.loaded_model_path = None;
            self.model_package = None;
            self.engine = Some(engine);
            self.engine_config = Some(self.config.clone());
            self.worker_status = None;
            self.worker_active = false;
            self.lifecycle = LifecycleState::Ready;
            self.last_error = None;
            Ok(self.snapshot())
        })();
        match result {
            Ok(snapshot) => Ok(snapshot),
            Err(error) => {
                self.fail(&error);
                Err(error)
            }
        }
    }

    fn start_processing(&mut self) -> RuntimeResult<RuntimeSnapshot> {
        if self.worker_status.is_some()
            || matches!(
                self.lifecycle,
                LifecycleState::Loading | LifecycleState::Running | LifecycleState::Stopping
            )
        {
            return Err(RuntimeError::State {
                state: format!("{:?}", self.lifecycle),
                action: "start_while_active".into(),
            });
        }
        if self.config.backend == RuntimeBackend::TensorRtRtx {
            crate::detect_processing_backends().ensure_available(self.config.backend)?;
        }
        if matches!(
            self.last_config_update,
            ConfigUpdateKind::PlanBuildRequired | ConfigUpdateKind::LoadModelRequired
        ) {
            return Err(RuntimeError::State {
                state: format!("{:?}", self.last_config_update),
                action: "start_before_engine_preparation".into(),
            });
        }
        if self.descriptor.is_none() || self.engine.is_none() {
            return Err(RuntimeError::State {
                state: format!("{:?}", self.lifecycle),
                action: "start_without_model".into(),
            });
        }
        if !self.lifecycle.can_transition_to(LifecycleState::Running) {
            return Err(RuntimeError::State {
                state: format!("{:?}", self.lifecycle),
                action: "start".into(),
            });
        }
        let engine = self.engine.as_mut().expect("engine checked above");
        engine.validate_config(self.config.conversion)?;
        engine.reset_stream()?;
        self.lifecycle = LifecycleState::Running;
        self.last_error = None;
        Ok(self.snapshot())
    }

    fn stop_processing(&mut self) -> RuntimeResult<RuntimeSnapshot> {
        if !matches!(
            self.lifecycle,
            LifecycleState::Running | LifecycleState::Ready
        ) {
            return Err(RuntimeError::State {
                state: format!("{:?}", self.lifecycle),
                action: "stop".into(),
            });
        }
        self.lifecycle = if self.worker_status.is_some() {
            LifecycleState::Stopping
        } else {
            LifecycleState::Stopped
        };
        Ok(self.snapshot())
    }

    fn abort_processing_start(&mut self) -> RuntimeResult<RuntimeSnapshot> {
        if self.worker_status.is_some() {
            if self.worker_active {
                return Err(RuntimeError::State {
                    state: format!("{:?}", self.lifecycle),
                    action: "abort_processing_start_with_active_worker".into(),
                });
            }
            self.worker_status = None;
            self.worker_generation = self.worker_generation.wrapping_add(1);
        }
        if self.engine.is_none()
            && let Some(descriptor) = self.descriptor.as_ref()
        {
            self.engine = Some(create_runtime_engine(
                &self.config,
                descriptor,
                self.model_package.as_deref(),
            )?);
            self.engine_config = Some(self.config.clone());
        }
        self.lifecycle = if self.engine.is_some() {
            LifecycleState::Ready
        } else {
            LifecycleState::Created
        };
        Ok(self.snapshot())
    }

    fn report_failure(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> RuntimeResult<RuntimeSnapshot> {
        self.lifecycle = LifecycleState::Failed;
        self.last_error = Some(RuntimeErrorInfo {
            code: code.into(),
            message: message.into(),
        });
        Ok(self.snapshot())
    }

    fn clear_error(&mut self) -> RuntimeResult<RuntimeSnapshot> {
        if self.worker_status.is_some()
            || matches!(
                self.lifecycle,
                LifecycleState::Loading | LifecycleState::Running | LifecycleState::Stopping
            )
        {
            return Err(RuntimeError::State {
                state: format!("{:?}", self.lifecycle),
                action: "clear_error_while_worker_active".into(),
            });
        }
        if let Some(engine) = self.engine.as_mut() {
            engine.reset_stream()?;
        }
        self.last_error = None;
        if self.lifecycle == LifecycleState::Failed {
            self.lifecycle = if self.engine.is_some() {
                LifecycleState::Ready
            } else {
                LifecycleState::Created
            };
        }
        Ok(self.snapshot())
    }

    fn load_model_inner(
        &mut self,
        path: String,
        mut config: RuntimeConfig,
    ) -> RuntimeResult<RuntimeSnapshot> {
        let input = resolve_model_path(Path::new(&path))?;
        if !input.is_file() {
            return Err(RuntimeError::Config(format!(
                "model path does not exist: {path}"
            )));
        }
        let mut loader = ModelLoader::new(Default::default())?;
        let (package, manifest) = if input
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| {
                value.eq_ignore_ascii_case("vfr") || value.eq_ignore_ascii_case("vfrm")
            }) {
            let inspection = vocal_model::inspect_model_artifact(&input)?;
            if !inspection.components.synth.present {
                return Err(RuntimeError::Config(
                    "select a voice model; this file contains only acoustic weights".into(),
                ));
            }
            let package = loader.load_voice_cache_shared(&input)?;
            (package.clone(), package.manifest.clone())
        } else {
            let manifest = loader.load_manifest(&input)?;
            let root = input.parent().unwrap_or_else(|| Path::new("."));
            let package = Arc::new(loader.load_rvc_directory(root, manifest.clone())?);
            (package, manifest)
        };
        let descriptor = manifest.descriptor();
        config.model_path = Some(input.display().to_string());
        let pending_plans =
            config.backend == RuntimeBackend::TensorRtRtx && config.tensorrt_artifacts().is_none();
        let engine = if pending_plans {
            crate::detect_processing_backends().ensure_available(config.backend)?;
            None
        } else {
            Some(create_runtime_engine(
                &config,
                &descriptor,
                Some(package.as_ref()),
            )?)
        };
        if let Some(store) = &self.config_store {
            store.save(&config)?;
        }
        self.config = config;
        self.config_revision = self.config_revision.saturating_add(1);
        self.model = Some(ModelSummary::from(&manifest));
        self.descriptor = Some(descriptor);
        self.loaded_model_path = Some(input.display().to_string());
        self.model_package = Some(package);
        self.engine = engine;
        self.engine_config = self.engine.as_ref().map(|_| self.config.clone());
        self.worker_status = None;
        self.worker_active = false;
        self.lifecycle = if pending_plans {
            LifecycleState::Created
        } else {
            LifecycleState::Ready
        };
        self.last_config_update = if pending_plans {
            ConfigUpdateKind::PlanBuildRequired
        } else {
            ConfigUpdateKind::Unchanged
        };
        self.last_error = None;
        Ok(self.snapshot())
    }

    fn take_engine_for_worker(&mut self) -> RuntimeResult<(u64, Box<dyn InferenceEngine>)> {
        if self.lifecycle != LifecycleState::Running {
            return Err(RuntimeError::State {
                state: format!("{:?}", self.lifecycle),
                action: "take_engine_for_worker".into(),
            });
        }
        let engine = self.engine.take().ok_or_else(|| RuntimeError::State {
            state: "model_worker_owns_engine".into(),
            action: "take_engine_for_worker".into(),
        })?;
        self.worker_generation = self.worker_generation.wrapping_add(1);
        self.worker_active = false;
        self.worker_status = Some(engine.status());
        Ok((self.worker_generation, engine))
    }

    fn activate_worker(&mut self, generation: u64) -> RuntimeResult<()> {
        if self.worker_generation != generation || self.worker_status.is_none() {
            return Err(RuntimeError::State {
                state: format!("{:?}", self.lifecycle),
                action: "activate_stale_worker".into(),
            });
        }
        self.worker_active = true;
        Ok(())
    }

    fn complete_worker(
        &mut self,
        generation: u64,
        mut engine: Box<dyn InferenceEngine>,
        failure: Option<(String, String)>,
    ) -> RuntimeResult<RuntimeSnapshot> {
        if self.engine.is_some()
            || self.worker_status.is_none()
            || self.worker_generation != generation
        {
            return Err(RuntimeError::State {
                state: format!("{:?}", self.lifecycle),
                action: "complete_worker_without_active_worker".into(),
            });
        }
        engine.flush_idle_caches();
        self.metrics = engine.metrics();
        self.worker_status = None;
        self.worker_active = false;
        self.engine = Some(engine);
        if let Some((code, message)) = failure {
            self.lifecycle = LifecycleState::Failed;
            self.last_error = Some(RuntimeErrorInfo { code, message });
        } else if self.lifecycle == LifecycleState::Stopping {
            self.lifecycle = LifecycleState::Stopped;
        }
        Ok(self.snapshot())
    }

    fn transition(&mut self, next: LifecycleState, action: &str) -> RuntimeResult<()> {
        if !self.lifecycle.can_transition_to(next) {
            return Err(RuntimeError::State {
                state: format!("{:?}", self.lifecycle),
                action: action.into(),
            });
        }
        self.lifecycle = next;
        Ok(())
    }

    fn fail(&mut self, error: &RuntimeError) {
        self.lifecycle = LifecycleState::Failed;
        self.last_error = Some(RuntimeErrorInfo {
            code: error.code().into(),
            message: error.to_string(),
        });
    }

    fn check_revision(&self, expected: u64) -> RuntimeResult<()> {
        if self.config_revision != expected {
            return Err(RuntimeError::ConfigConflict {
                expected,
                actual: self.config_revision,
            });
        }
        Ok(())
    }
}

fn scan_model_directory_impl(root_input: String) -> RuntimeResult<ModelCatalog> {
    // Catalog inspection only reads filesystem metadata and never mutates
    // runtime state. Do not take the control operation mutex here: a slow
    // antivirus/network directory scan must not stall audio processing.
    let root = PathBuf::from(root_input);
    let root = fs::canonicalize(&root).map_err(|error| RuntimeError::Io {
        path: root.clone(),
        message: error.to_string(),
    })?;
    if !root.is_dir() {
        return Err(RuntimeError::Config(format!(
            "model library is not a directory: {}",
            root.display()
        )));
    }

    const MAX_DEPTH: usize = 3;
    const MAX_ENTRIES: usize = 512;
    const MAX_MODEL_FILES: usize = 4096;
    const MAX_ISSUES: usize = 512;
    const MAX_SCAN_METADATA_BYTES: u64 = 4 * 1024 * 1024;
    let load_config = ModelLoadConfig {
        max_metadata_bytes: MAX_SCAN_METADATA_BYTES,
        ..ModelLoadConfig::default()
    };
    let mut directories = vec![(root.clone(), 0_usize)];
    let mut visited = HashSet::new();
    let mut entries = Vec::new();
    let mut issues = Vec::new();
    let mut model_files_seen = 0_usize;
    while let Some((directory, depth)) = directories.pop() {
        if model_files_seen >= MAX_MODEL_FILES {
            break;
        }
        if !visited.insert(directory.clone()) {
            continue;
        }
        let read_dir = match fs::read_dir(&directory) {
            Ok(read_dir) => read_dir,
            Err(error) => {
                push_catalog_issue(
                    &mut issues,
                    MAX_ISSUES,
                    directory.display().to_string(),
                    error.to_string(),
                );
                continue;
            }
        };
        for item in read_dir {
            if entries.len() >= MAX_ENTRIES {
                push_catalog_issue(
                    &mut issues,
                    MAX_ISSUES,
                    root.display().to_string(),
                    format!("model library limit reached ({MAX_ENTRIES} entries)"),
                );
                break;
            }
            let item = match item {
                Ok(item) => item,
                Err(error) => {
                    push_catalog_issue(
                        &mut issues,
                        MAX_ISSUES,
                        directory.display().to_string(),
                        error.to_string(),
                    );
                    continue;
                }
            };
            let path = item.path();
            // Import/cache/editor staging is never a published library artifact.
            if item.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            let metadata = match item.metadata() {
                Ok(metadata) => metadata,
                Err(error) => {
                    push_catalog_issue(
                        &mut issues,
                        MAX_ISSUES,
                        path.display().to_string(),
                        error.to_string(),
                    );
                    continue;
                }
            };
            if metadata.is_dir() {
                let canonical_path = match canonical_library_path(&path, &root) {
                    Ok(path) => path,
                    Err(error) => {
                        push_catalog_issue(
                            &mut issues,
                            MAX_ISSUES,
                            path.display().to_string(),
                            error,
                        );
                        continue;
                    }
                };
                if depth < MAX_DEPTH {
                    directories.push((canonical_path, depth + 1));
                }
                continue;
            }
            if !metadata.is_file() || !is_model_catalog_file(&path) {
                continue;
            }
            model_files_seen = model_files_seen.saturating_add(1);
            if model_files_seen > MAX_MODEL_FILES {
                push_catalog_issue(
                    &mut issues,
                    MAX_ISSUES,
                    root.display().to_string(),
                    format!("model file scan limit reached ({MAX_MODEL_FILES} files)"),
                );
                break;
            }
            let canonical_path = match canonical_library_path(&path, &root) {
                Ok(path) => path,
                Err(error) => {
                    push_catalog_issue(&mut issues, MAX_ISSUES, path.display().to_string(), error);
                    continue;
                }
            };
            let inspection = match inspect_model_artifact_with_config(&path, &load_config) {
                Ok(inspection) => inspection,
                Err(error) => {
                    let is_named_manifest = path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| {
                            name.eq_ignore_ascii_case("manifest.json")
                                || name.eq_ignore_ascii_case("model.json")
                                || name.eq_ignore_ascii_case("vocal-fast.json")
                        });
                    if is_named_manifest || is_vfrm_path(&path) {
                        push_catalog_issue(
                            &mut issues,
                            MAX_ISSUES,
                            path.display().to_string(),
                            error.to_string(),
                        );
                    }
                    continue;
                }
            };
            let manifest = inspection.manifest;
            let path_string = canonical_path.display().to_string();
            let artifact_bytes = metadata.len();
            let kind = model_artifact_kind(&canonical_path).unwrap_or(ModelArtifactKind::Manifest);
            let referenced_bytes = if kind == ModelArtifactKind::Manifest {
                referenced_model_bytes(&canonical_path, &manifest)
            } else {
                0
            };
            entries.push(ModelCatalogEntry {
                path: path_string,
                kind,
                size_bytes: artifact_bytes.saturating_add(referenced_bytes),
                modified_at_ms: metadata
                    .modified()
                    .ok()
                    .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                    .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
                    .unwrap_or_default(),
                components: inspection.components,
                model: ModelSummary::from(&manifest),
            });
        }
    }
    entries.sort_by(|left, right| {
        left.model
            .display_name
            .to_ascii_lowercase()
            .cmp(&right.model.display_name.to_ascii_lowercase())
            .then_with(|| {
                right
                    .components
                    .automatic_tensor_rt
                    .cmp(&left.components.automatic_tensor_rt)
            })
            .then_with(|| {
                right
                    .components
                    .retrieval
                    .present
                    .cmp(&left.components.retrieval.present)
            })
            .then_with(|| artifact_priority(left.kind).cmp(&artifact_priority(right.kind)))
            .then_with(|| left.path.cmp(&right.path))
    });
    let mut identities = HashSet::new();
    entries.retain(|entry| identities.insert(catalog_identity(entry)));
    Ok(ModelCatalog {
        root: root.display().to_string(),
        entries,
        issues,
    })
}

fn is_vfrm_path(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| {
            value.eq_ignore_ascii_case("vfr") || value.eq_ignore_ascii_case("vfrm")
        })
}

fn is_model_catalog_file(path: &Path) -> bool {
    is_vfrm_path(path)
        || path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("json"))
}

fn canonical_library_path(path: &Path, root: &Path) -> Result<PathBuf, String> {
    let canonical = fs::canonicalize(path).map_err(|error| error.to_string())?;
    if canonical.starts_with(root) {
        Ok(canonical)
    } else {
        Err("skipped path outside model library".into())
    }
}

fn artifact_priority(kind: ModelArtifactKind) -> u8 {
    match kind {
        ModelArtifactKind::Vfrm => 0,
        ModelArtifactKind::Manifest => 1,
    }
}

fn catalog_identity(entry: &ModelCatalogEntry) -> String {
    // Imports are independent records even when IDs match. Prefer a cache over
    // its manifest within the same folder.
    let model = &entry.model;
    if model.id.trim().is_empty() {
        return format!("path:{}", entry.path);
    }
    format!(
        "{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
        model.id,
        model.version,
        model.architecture,
        model.sample_rate,
        entry.components.automatic_tensor_rt,
        entry.components.retrieval.present,
        entry.components.synth.present,
        entry.components.hubert.present,
        entry.components.rmvpe.present,
        Path::new(&entry.path)
            .parent()
            .unwrap_or(Path::new(""))
            .display()
    )
}

fn push_catalog_issue(
    issues: &mut Vec<ModelCatalogIssue>,
    limit: usize,
    path: String,
    message: String,
) {
    if issues.len() < limit {
        issues.push(ModelCatalogIssue { path, message });
    }
}

fn referenced_model_bytes(manifest_path: &Path, manifest: &RvcModelManifest) -> u64 {
    let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let size = |relative: &str| {
        let candidate = root.join(relative);
        let canonical = fs::canonicalize(candidate).ok()?;
        if !canonical.starts_with(root) {
            return None;
        }
        fs::metadata(canonical).ok().map(|metadata| metadata.len())
    };
    size(&manifest.model_file)
        .unwrap_or_default()
        .saturating_add(
            manifest
                .index_file
                .as_deref()
                .and_then(size)
                .unwrap_or_default(),
        )
}

fn create_runtime_engine(
    config: &RuntimeConfig,
    descriptor: &ModelDescriptor,
    package: Option<&ModelPackage>,
) -> RuntimeResult<Box<dyn InferenceEngine>> {
    let engine = build_runtime_engine(config, descriptor, package)?;
    engine.validate_config(config.conversion)?;
    Ok(engine)
}

fn engine_reload_config(config: &RuntimeConfig) -> RuntimeResult<EngineReloadConfig> {
    let artifacts_path = config
        .tensorrt_artifacts()
        .ok_or_else(|| RuntimeError::Config("TensorRT artifacts path is not configured".into()))?;
    Ok(EngineReloadConfig {
        backend: config.backend.engine_kind(),
        tensorrt: Some(TensorRtReloadConfig {
            artifacts_path: PathBuf::from(artifacts_path),
            speaker_id: config.speaker_id,
            window_samples: config.tensorrt_window(),
            graph_mode: config.tensorrt.graph_mode.clone(),
            cuda_graph: config.tensorrt.cuda_graph,
            native_cuda_graph: config.tensorrt.native_cuda_graph,
            graph_cache_size: config.tensorrt.graph_cache_size,
            parallel_acoustic: config.tensorrt.parallel_acoustic,
            async_audio_d2h: config.tensorrt.async_audio_d2h,
        }),
    })
}

fn build_runtime_engine(
    config: &RuntimeConfig,
    descriptor: &ModelDescriptor,
    #[allow(unused_variables)] package: Option<&ModelPackage>,
) -> RuntimeResult<Box<dyn InferenceEngine>> {
    #[cfg(not(feature = "cpu-graph"))]
    if config.backend == RuntimeBackend::RvcCpu
        && package.is_some_and(|p| {
            p.metadata["canonical_schema"].as_str() == Some(vocal_model::RVC_TENSOR_SCHEMA)
        })
    {
        return Err(RuntimeError::Config(
            "CPU model inference requires the cpu-graph feature".into(),
        ));
    }
    #[cfg(feature = "cpu-graph")]
    if config.backend == RuntimeBackend::RvcCpu
        && package.is_some_and(|p| {
            p.metadata["canonical_schema"].as_str() == Some(vocal_model::RVC_TENSOR_SCHEMA)
        })
    {
        return crate::cpu::create(config, descriptor);
    }
    if config.backend == RuntimeBackend::TensorRtRtx {
        crate::detect_processing_backends().ensure_available(config.backend)?;
        #[cfg(feature = "tensorrt-rtx")]
        {
            let package = package.ok_or_else(|| {
                RuntimeError::Config("TensorRT backend requires a loaded VFRM model package".into())
            })?;
            let artifacts_path = config.tensorrt_artifacts().ok_or_else(|| {
                RuntimeError::Config("TensorRT artifacts path is not configured".into())
            })?;
            let mut trt_config = vocal_tensorrt::TensorRtConfig::from_env()
                .map_err(|error| RuntimeError::Config(error.to_string()))?;
            let gui = &config.tensorrt;
            trt_config.runtime.cuda_graph = gui.cuda_graph;
            trt_config.runtime.native_cuda_graph = gui.native_cuda_graph;
            trt_config.runtime.graph_cache_size = gui.graph_cache_size;
            trt_config.runtime.parallel_acoustic = gui.parallel_acoustic;
            trt_config.runtime.async_audio_d2h = gui.async_audio_d2h;
            if let Some(mode) = vocal_tensorrt::GraphReplayMode::parse(&gui.graph_mode) {
                trt_config.runtime.graph_mode = mode;
            } else {
                return Err(RuntimeError::Config(format!(
                    "invalid TensorRT graph mode: {}",
                    gui.graph_mode
                )));
            }
            let trt = vocal_engine::TensorRtBackendConfig::for_package(
                PathBuf::from(artifacts_path),
                config.speaker_id,
                config.tensorrt_window(),
                trt_config.runtime,
                trt_config.diagnostics,
            );
            return Ok(Box::new(vocal_engine::TensorRtEngine::from_package(
                descriptor.clone(),
                package,
                trt,
            )?));
        }
        #[cfg(not(feature = "tensorrt-rtx"))]
        {
            return Err(vocal_engine::EngineError::BackendUnavailable(
                "TensorRT backend requires the vocal-runtime/tensorrt-rtx feature".into(),
            )
            .into());
        }
    }
    if config.backend == RuntimeBackend::DirectMl {
        crate::detect_processing_backends().ensure_available(config.backend)?;
        #[cfg(feature = "directml")]
        return crate::directml::create(config, descriptor);
        #[cfg(not(feature = "directml"))]
        return Err(RuntimeError::Config(
            "DirectML requires the directml feature".into(),
        ));
    }
    Ok(create_engine(EngineConfig {
        backend: config.backend.engine_kind(),
        model: descriptor.clone(),
        tensorrt: None,
        directml: None,
        cpu: None,
    })?)
}

impl RuntimeInner {
    fn snapshot(&self) -> RuntimeSnapshot {
        RuntimeSnapshot {
            lifecycle: self.lifecycle,
            config: self.config.clone(),
            config_revision: self.config_revision,
            last_config_update: self.last_config_update,
            model: self.model.clone(),
            metrics: self.metrics.into(),
            engine: self
                .engine
                .as_ref()
                .map(|engine| engine.status())
                .or_else(|| self.worker_status.clone())
                .map(|status| {
                    let mut status: RuntimeEngineStatus = status.into();
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
            architecture: "rvc-v1".into(),
            sample_rate: 16_000,
            version: "v1".into(),
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
    fn model_worker_owns_engine_between_start_and_completion() {
        let format = vocal_types::AudioFormat::MONO_16K;
        let context = RuntimeContext::new(RuntimeConfig {
            input_format: format,
            output_format: format,
            ..RuntimeConfig::default()
        })
        .expect("context");
        context.load_model_descriptor(descriptor()).expect("load");
        context.start_processing().expect("start");
        let (generation, engine) = context.take_engine_for_worker().expect("transfer engine");
        context
            .activate_worker(generation)
            .expect("activate worker");
        let snapshot = context.snapshot().expect("snapshot");
        assert!(snapshot.engine.is_some());
        let error = context
            .process(&AudioChunk {
                format,
                samples: vec![0.0; 160],
                sequence: 1,
            })
            .expect_err("worker owns engine");
        assert!(matches!(error, RuntimeError::State { .. }));
        context
            .complete_worker(generation, engine, None)
            .expect("return engine");
        context.stop_processing().expect("stop");
    }

    #[test]
    fn stop_waits_for_worker_return_and_rejects_duplicate_completion() {
        let format = vocal_types::AudioFormat::MONO_16K;
        let context = RuntimeContext::new(RuntimeConfig {
            input_format: format,
            output_format: format,
            ..RuntimeConfig::default()
        })
        .expect("context");
        context.load_model_descriptor(descriptor()).expect("load");
        context.start_processing().expect("start");
        let (generation, engine) = context.take_engine_for_worker().expect("transfer engine");
        context
            .activate_worker(generation)
            .expect("activate worker");
        let stopping = context.stop_processing().expect("stop request");
        assert_eq!(stopping.lifecycle, LifecycleState::Stopping);
        let completed = context
            .complete_worker(generation, engine, None)
            .expect("return engine");
        assert_eq!(completed.lifecycle, LifecycleState::Stopped);

        assert!(matches!(
            context.take_engine_for_worker(),
            Err(RuntimeError::State { .. })
        ));
    }

    #[test]
    fn clear_error_waits_until_worker_returns_engine() {
        let format = vocal_types::AudioFormat::MONO_16K;
        let context = RuntimeContext::new(RuntimeConfig {
            input_format: format,
            output_format: format,
            ..RuntimeConfig::default()
        })
        .expect("context");
        context.load_model_descriptor(descriptor()).expect("load");
        context.start_processing().expect("start");
        let (generation, engine) = context.take_engine_for_worker().expect("transfer");
        context
            .activate_worker(generation)
            .expect("activate worker");
        context
            .report_failure("audio.worker", "worker stopped")
            .expect("report failure");
        assert!(matches!(
            context.load_model("replacement.vfr"),
            Err(RuntimeError::State { .. })
        ));
        assert!(matches!(
            context.clear_error(),
            Err(RuntimeError::State { .. })
        ));
        context
            .complete_worker(generation, engine, None)
            .expect("return engine");
        assert_eq!(
            context.clear_error().expect("clear error").lifecycle,
            LifecycleState::Ready
        );
    }

    #[test]
    fn aborted_worker_lease_cannot_replace_a_rebuilt_engine() {
        let format = vocal_types::AudioFormat::MONO_16K;
        let context = RuntimeContext::new(RuntimeConfig {
            input_format: format,
            output_format: format,
            ..RuntimeConfig::default()
        })
        .expect("context");
        context.load_model_descriptor(descriptor()).expect("load");
        context.start_processing().expect("start");
        let (generation, engine) = context.take_engine_for_worker().expect("transfer");

        let rollback = context.abort_processing_start().expect("rollback");
        assert_eq!(rollback.lifecycle, LifecycleState::Ready);
        assert!(rollback.engine.is_some());
        assert!(matches!(
            context.complete_worker(generation, engine, None),
            Err(RuntimeError::State { .. })
        ));
        assert!(context.snapshot().expect("snapshot").engine.is_some());
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
    fn stale_backend_selection_leaves_model_and_configuration_untouched() {
        let context = RuntimeContext::new(RuntimeConfig::default()).expect("context");
        context.load_model_descriptor(descriptor()).expect("load");
        let selection = crate::ModelBackendSelection {
            expected_revision: 0,
            backend: RuntimeBackend::RvcCpu,
            automatic: false,
            artifacts_path: None,
            directml_sources_path: None,
            directml_device_id: None,
        };
        assert!(matches!(
            context.load_model_with_backend("missing.vfr", selection),
            Err(RuntimeError::ConfigConflict { .. })
        ));
        let snapshot = context.snapshot().unwrap();
        assert_eq!(snapshot.lifecycle, LifecycleState::Ready);
        assert_eq!(snapshot.config.backend, RuntimeBackend::Passthrough);
        assert_eq!(snapshot.config_revision, 1);
        assert!(snapshot.engine.is_some());
    }

    #[test]
    fn automatic_startup_and_missing_backend_recovery_use_catalog_policy() {
        let unavailable = crate::ProcessingBackendCatalog {
            cpu: Default::default(),
            recommended: RuntimeBackend::Passthrough,
            backends: vec![crate::BackendAvailability {
                backend: RuntimeBackend::Passthrough,
                available: true,
                reason: None,
                device: Some("CPU".into()),
                description: "Monitoring".into(),
                devices: Vec::new(),
                recommended_device_id: None,
            }],
        };
        let context = RuntimeContext::new(RuntimeConfig {
            backend: RuntimeBackend::TensorRtRtx,
            backend_auto: false,
            ..RuntimeConfig::default()
        })
        .unwrap();
        let notice = context
            .request(move |state| state.initialize_processing_backend(unavailable))
            .unwrap();
        assert!(notice.is_some());
        assert_eq!(
            context.config().unwrap().backend,
            RuntimeBackend::Passthrough
        );
        assert!(!context.config().unwrap().backend_auto);

        let available = crate::ProcessingBackendCatalog {
            cpu: Default::default(),
            recommended: RuntimeBackend::TensorRtRtx,
            backends: vec![crate::BackendAvailability {
                backend: RuntimeBackend::TensorRtRtx,
                available: true,
                reason: None,
                device: Some("GPU".into()),
                description: "GPU conversion".into(),
                devices: Vec::new(),
                recommended_device_id: None,
            }],
        };
        let automatic = RuntimeContext::new(RuntimeConfig::default()).unwrap();
        automatic
            .request(move |state| state.initialize_processing_backend(available))
            .unwrap();
        assert_eq!(
            automatic.config().unwrap().backend,
            RuntimeBackend::TensorRtRtx
        );
        assert!(automatic.snapshot().unwrap().engine.is_none());
    }

    #[cfg(not(feature = "tensorrt-rtx"))]
    #[test]
    fn unavailable_backend_is_rejected_without_replacing_the_idle_engine() {
        let context = RuntimeContext::new(RuntimeConfig::default()).unwrap();
        context.load_model_descriptor(descriptor()).unwrap();
        let mut config = context.config().unwrap();
        config.backend = RuntimeBackend::TensorRtRtx;
        assert!(matches!(
            context.update_config(config, 1),
            Err(RuntimeError::Engine(
                vocal_engine::EngineError::BackendUnavailable(_)
            ))
        ));
        assert_eq!(
            context.config().unwrap().backend,
            RuntimeBackend::Passthrough
        );
        assert!(context.snapshot().unwrap().engine.is_some());
        context
            .start_processing()
            .expect("old engine is still usable");
    }

    #[test]
    fn pending_model_or_plan_changes_cannot_start_an_obsolete_engine() {
        let context = RuntimeContext::new(RuntimeConfig::default()).unwrap();
        context.load_model_descriptor(descriptor()).unwrap();
        let mut config = context.config().unwrap();
        config.model_path = Some("replacement.vfr".into());
        context.update_config(config, 1).unwrap();
        assert!(context.start_processing().is_err());
    }

    #[test]
    fn idle_hot_reload_keeps_engine_state() {
        let context = RuntimeContext::new(RuntimeConfig::default()).expect("context");
        context.load_model_descriptor(descriptor()).expect("load");
        context.start_processing().expect("start");
        context
            .process(&AudioChunk {
                format: vocal_types::AudioFormat::MONO_48K,
                samples: vec![0.0; 480],
                sequence: 1,
            })
            .expect("first block");
        context.stop_processing().expect("stop");

        let mut config = context.config().expect("config");
        config.conversion.pitch_semitones = 1.0;
        let snapshot = context.update_config(config, 1).expect("hot reload");
        assert_eq!(snapshot.last_config_update, ConfigUpdateKind::HotReload);

        context.start_processing().expect("restart");
        context
            .process(&AudioChunk {
                format: vocal_types::AudioFormat::MONO_48K,
                samples: vec![0.0; 480],
                sequence: 2,
            })
            .expect("second block");
        assert_eq!(
            context.snapshot().unwrap().engine.unwrap().processed_chunks,
            2
        );
    }

    #[test]
    fn pending_model_path_prevents_follow_up_engine_reload() {
        // Descriptor-only fixture exercises deferred replacement without
        // advertising an uncompiled CPU backend in capability discovery.
        let context = RuntimeContext::new(RuntimeConfig {
            backend: RuntimeBackend::RvcCpu,
            ..Default::default()
        })
        .expect("context");
        context.load_model_descriptor(descriptor()).expect("load");

        let mut config = context.config().expect("config");
        config.model_path = Some("pending.vfrm".into());
        let snapshot = context
            .update_config(config.clone(), 1)
            .expect("select model");
        assert_eq!(
            snapshot.last_config_update,
            ConfigUpdateKind::LoadModelRequired
        );

        config.backend = RuntimeBackend::Passthrough;
        let snapshot = context
            .update_config(config, snapshot.config_revision)
            .expect("defer backend change");
        assert_eq!(
            snapshot.last_config_update,
            ConfigUpdateKind::LoadModelRequired
        );
        assert_eq!(snapshot.model.unwrap().id, "fixture");
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

    #[test]
    fn prepared_model_config_commit_is_atomic_and_revision_checked() {
        let context = RuntimeContext::new(RuntimeConfig::default()).unwrap();
        let original = context.load_model_descriptor(descriptor()).unwrap();
        let mut candidate = original.config.clone();
        candidate.queue_blocks = 2;
        assert!(
            context
                .load_model_with_config(
                    "missing-voice.vfrm",
                    candidate.clone(),
                    original.config_revision - 1
                )
                .is_err()
        );
        assert!(
            context
                .load_model_with_config(
                    "missing-voice.vfrm",
                    candidate.clone(),
                    original.config_revision
                )
                .is_err()
        );
        let failed = context.snapshot().unwrap();
        assert_eq!(failed.config_revision, original.config_revision);
        assert_eq!(failed.config.queue_blocks, original.config.queue_blocks);
        assert_eq!(failed.model.unwrap().id, original.model.unwrap().id);
        assert!(failed.engine.is_some());
        context.clear_error().unwrap();

        let path = std::env::temp_dir().join(format!(
            "prepared-model-{}-{}.vfrm",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        vocal_model::write_cache(
            &path,
            &vocal_model::ModelPackage {
                manifest: vocal_model::RvcModelManifest {
                    id: "prepared".into(),
                    display_name: "Prepared voice".into(),
                    architecture: "rvc-v2".into(),
                    version: "v2".into(),
                    sample_rate: 48_000,
                    model_file: "voice.pth".into(),
                    index_file: None,
                    source: "test".into(),
                    notes: None,
                },
                metadata: serde_json::json!({}),
                tensors: vec![vocal_model::Tensor::byte_tensor("rvc.pth", vec![1, 2, 3])],
            },
        )
        .unwrap();
        let loaded = context
            .load_model_with_config(
                path.display().to_string(),
                candidate.clone(),
                original.config_revision,
            )
            .unwrap();
        assert_eq!(loaded.config_revision, original.config_revision + 1);
        assert_eq!(loaded.config.queue_blocks, 2);
        assert_eq!(loaded.model.unwrap().id, "prepared");
        assert_eq!(loaded.lifecycle, LifecycleState::Ready);
        context.start_processing().unwrap();
        let (generation, engine) = context.take_engine_for_worker().unwrap();
        assert!(
            context
                .load_model_with_config(
                    path.display().to_string(),
                    candidate,
                    loaded.config_revision
                )
                .is_err()
        );
        assert_eq!(
            context.snapshot().unwrap().config_revision,
            loaded.config_revision
        );
        context.stop_processing().unwrap();
        context.complete_worker(generation, engine, None).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn model_catalog_scans_metadata_without_loading_payload_and_prefers_cache() {
        let root = std::env::temp_dir().join(format!(
            "vocal-runtime-models-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("create model root");
        let manifest = serde_json::json!({
            "id": "fixture",
            "display_name": "Fixture",
            "architecture": "rvc-v2",
            "version": "1",
            "sample_rate": 32000,
            "model_file": "model.pth",
            "index_file": null,
            "source": "test",
            "notes": null
        });
        std::fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .expect("write manifest");
        std::fs::write(root.join("model.pth"), [0_u8; 4]).expect("write model");

        let cache = vocal_model::ModelPackage {
            manifest: serde_json::from_value(manifest).unwrap(),
            metadata: serde_json::json!({}),
            tensors: vec![vocal_model::Tensor::byte_tensor("rvc.pth", vec![1, 2, 3])],
        };
        vocal_model::write_cache(root.join("fixture.vfrm"), &cache).expect("write cache");

        let context = RuntimeContext::new(RuntimeConfig::default()).expect("context");
        let catalog = context
            .scan_model_directory(root.to_string_lossy().into_owned())
            .unwrap();
        assert_eq!(catalog.entries.len(), 1);
        assert_eq!(catalog.entries[0].kind, ModelArtifactKind::Vfrm);
        assert_eq!(catalog.entries[0].model.id, "fixture");
        assert!(catalog.entries[0].components.synth.present);
        assert!(catalog.issues.is_empty());
        let imported = root.join("imported");
        std::fs::create_dir(&imported).unwrap();
        vocal_model::write_cache(imported.join("model.vfrm"), &cache).unwrap();
        let hidden = root.join(".import-staging");
        std::fs::create_dir(&hidden).unwrap();
        vocal_model::write_cache(hidden.join("model.vfrm"), &cache).unwrap();
        let refreshed = context
            .scan_model_directory(root.to_string_lossy().into_owned())
            .unwrap();
        assert_eq!(
            refreshed.entries.len(),
            2,
            "imported records stay visible, staging stays hidden"
        );
        std::fs::remove_dir_all(root).expect("remove model root");
    }
}
