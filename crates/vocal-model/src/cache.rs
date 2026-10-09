//! Shared in-process cache for VFRM packages.
//!
//! Package parsing is relatively expensive because it validates the compact
//! directory and owns a copy of every tensor. The cache keeps a weak reference
//! keyed by the file identity, so repeated model selection in a Tauri worker or
//! pipeline rebuild can reuse the parsed package without keeping unused voice
//! models alive forever.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock, Weak},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{ModelError, ModelLoadConfig, ModelPackage, load_cache_selected};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ModelCacheKey {
    canonical_path: PathBuf,
    file_size: u64,
    modified: SystemTime,
    max_metadata_bytes: u64,
    max_tensors: u64,
    max_tensor_bytes: u64,
    validate_canonical: bool,
    voice_only: bool,
}

impl ModelCacheKey {
    fn from_path(
        path: impl AsRef<Path>,
        config: &ModelLoadConfig,
        voice_only: bool,
    ) -> Result<Self, ModelError> {
        let canonical_path = fs::canonicalize(path)?;
        let metadata = fs::metadata(&canonical_path)?;
        Ok(Self {
            canonical_path,
            file_size: metadata.len(),
            modified: metadata.modified().unwrap_or(UNIX_EPOCH),
            max_metadata_bytes: config.max_metadata_bytes,
            max_tensors: config.max_tensors,
            max_tensor_bytes: config.max_tensor_bytes,
            validate_canonical: config.validate_canonical,
            voice_only,
        })
    }
}

/// Weakly-held parsed model packages indexed by canonical path and file
/// signature. A caller owns the returned `Arc`; once all callers release it,
/// the next lookup will parse the file again if needed.
#[derive(Debug, Default)]
pub struct ModelPackageCache {
    entries: Mutex<HashMap<ModelCacheKey, Arc<Mutex<Weak<ModelPackage>>>>>,
}

fn live(entry: &Arc<Mutex<Weak<ModelPackage>>>) -> bool {
    Arc::strong_count(entry) > 1
        || entry
            .try_lock()
            .map_or(true, |package| package.strong_count() > 0)
}

impl ModelPackageCache {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub fn load(
        &self,
        path: impl AsRef<Path>,
        config: &ModelLoadConfig,
    ) -> Result<Arc<ModelPackage>, ModelError> {
        self.load_selected(path.as_ref(), config, false)
    }

    pub fn load_voice(
        &self,
        path: impl AsRef<Path>,
        config: &ModelLoadConfig,
    ) -> Result<Arc<ModelPackage>, ModelError> {
        self.load_selected(path.as_ref(), config, true)
    }

    fn load_selected(
        &self,
        path: &Path,
        config: &ModelLoadConfig,
        voice_only: bool,
    ) -> Result<Arc<ModelPackage>, ModelError> {
        config.validate()?;
        let key = ModelCacheKey::from_path(path, config, voice_only)?;
        let entry = {
            let mut entries = self
                .entries
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            entries.retain(|_, entry| live(entry));
            Arc::clone(entries.entry(key.clone()).or_default())
        };

        // Only identical requests wait together. Unrelated paths keep loading
        // concurrently, without duplicate multi-hundred-MiB package allocations.
        let mut current = entry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(package) = current.upgrade() {
            return Ok(package);
        }
        let package = Arc::new(load_cache_selected(
            &key.canonical_path,
            config,
            voice_only,
        )?);
        *current = Arc::downgrade(&package);
        Ok(package)
    }

    pub fn clear(&self) {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    pub fn len(&self) -> usize {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        entries.retain(|_, entry| live(entry));
        entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

static SHARED_MODEL_CACHE: OnceLock<ModelPackageCache> = OnceLock::new();

pub fn shared_model_cache() -> &'static ModelPackageCache {
    SHARED_MODEL_CACHE.get_or_init(ModelPackageCache::new)
}
