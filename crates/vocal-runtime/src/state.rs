use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::UNIX_EPOCH,
};

use serde::Serialize;
use vocal_engine::{EngineConfig, EngineStatus, InferenceEngine, create_engine};
use vocal_model::{
    ModelArtifactKind, ModelLoadConfig, ModelLoader, ModelPackage, RvcModelManifest,
    inspect_model_manifest_with_config, model_artifact_kind, resolve_model_path,
};
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

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCatalogEntry {
    pub path: String,
    pub kind: ModelArtifactKind,
    pub size_bytes: u64,
    pub modified_at_ms: u64,
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

    pub fn scan_model_directory(&self, root: impl Into<String>) -> RuntimeResult<ModelCatalog> {
        // Catalog inspection only reads filesystem metadata and never mutates
        // runtime state. Do not take the control operation mutex here: a slow
        // antivirus/network directory scan must not stall audio processing.
        let root = PathBuf::from(root.into());
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
                        push_catalog_issue(
                            &mut issues,
                            MAX_ISSUES,
                            path.display().to_string(),
                            error,
                        );
                        continue;
                    }
                };
                let manifest = match inspect_model_manifest_with_config(&path, &load_config) {
                    Ok(manifest) => manifest,
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
                let path_string = canonical_path.display().to_string();
                let artifact_bytes = metadata.len();
                let kind =
                    model_artifact_kind(&canonical_path).unwrap_or(ModelArtifactKind::Manifest);
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
                    model: ModelSummary::from(&manifest),
                });
            }
        }
        entries.sort_by(|left, right| {
            left.model
                .display_name
                .to_ascii_lowercase()
                .cmp(&right.model.display_name.to_ascii_lowercase())
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
        // The engine is exclusively mutable behind `inner`; lifecycle control
        // methods also take that mutex and reject changes while Running. The
        // separate control-operation mutex is intentionally omitted here so a
        // slow control-plane command cannot delay an audio worker block.
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
        config.model_path = Some(input.display().to_string());
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
    let model = &entry.model;
    if model.id.trim().is_empty() {
        return format!("path:{}", entry.path);
    }
    format!(
        "{}\u{1f}{}\u{1f}{}\u{1f}{}",
        model.id, model.version, model.architecture, model.sample_rate
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
            tensors: vec![vocal_model::Tensor::byte_tensor("payload", vec![1, 2, 3])],
        };
        vocal_model::write_cache(root.join("fixture.vfrm"), &cache).expect("write cache");

        let context = RuntimeContext::new(RuntimeConfig::default()).expect("context");
        let catalog = context
            .scan_model_directory(root.to_string_lossy().into_owned())
            .unwrap();
        assert_eq!(catalog.entries.len(), 1);
        assert_eq!(catalog.entries[0].kind, ModelArtifactKind::Vfrm);
        assert_eq!(catalog.entries[0].model.id, "fixture");
        assert!(catalog.issues.is_empty());
        std::fs::remove_dir_all(root).expect("remove model root");
    }
}
