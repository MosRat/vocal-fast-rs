//! VFRM source manifests, complete bundles and their storage lifecycle.
use crate::{DirectMlConfig, DirectMlError};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectMlSources {
    pub synth: PathBuf,
    pub hubert: PathBuf,
    pub rmvpe: PathBuf,
    #[serde(default)]
    pub config: DirectMlConfig,
    #[serde(default)]
    pub enable_retrieval: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_manifest_publication_reuses_and_repairs_cache() {
        let root =
            std::env::temp_dir().join(format!("vocal-dml-source-test-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let root = std::fs::canonicalize(root).unwrap();
        let path = root.join("sources.json");
        let sources = DirectMlSources {
            synth: root.join("synth.vfrm"),
            hubert: root.join("hubert.vfrm"),
            rmvpe: root.join("rmvpe.vfrm"),
            config: DirectMlConfig::realtime_44960(),
            enable_retrieval: false,
        };
        sources.save(&path).unwrap();
        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
        sources.save(&path).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            modified
        );
        assert_eq!(DirectMlSources::load(&path).unwrap(), sources);
        std::fs::write(&path, b"incomplete").unwrap();
        sources.save(&path).unwrap();
        assert_eq!(DirectMlSources::load(&path).unwrap(), sources);
        let mut invalid = sources.clone();
        invalid.config.window_samples = 0;
        assert!(invalid.save(&path).is_err());
        assert_eq!(DirectMlSources::load(&path).unwrap(), sources);
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
}
impl DirectMlSources {
    /// Default per-user cache, independent of the working directory and TRT.
    pub fn default_cache_root() -> PathBuf {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("vocal-fast")
            .join("cache")
            .join("directml")
    }
    /// Save a validated manifest using an atomic rename; preserve an identical
    /// cached manifest and its modification time on subsequent loads.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), DirectMlError> {
        use std::{
            io::Write,
            sync::atomic::{AtomicU64, Ordering},
        };
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        self.config.validate()?;
        let path = path.as_ref();
        if Self::load(path).is_ok_and(|old| old == *self) {
            return Ok(());
        }
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent)?;
        let tmp = path.with_extension(format!(
            "tmp-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        let result = (|| {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)?;
            file.write_all(&serde_json::to_vec_pretty(self)?)?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&tmp, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(tmp);
        }
        result
    }
    /// Publish a reproducible source manifest beside the extracted acoustics.
    pub fn prepare_bundle(
        model: impl AsRef<Path>,
        cache_root: impl AsRef<Path>,
        config: DirectMlConfig,
    ) -> Result<(Self, PathBuf), DirectMlError> {
        let sources = Self::from_bundle(model, cache_root, config)?;
        let path = sources
            .hubert
            .parent()
            .ok_or_else(|| DirectMlError::Config("missing acoustic cache directory".into()))?
            .join("directml-sources.json");
        sources.save(&path)?;
        Ok((sources, path))
    }
    /// Materialize a complete VFRM bundle into a caller-owned cache directory.
    /// Graph compilation remains the responsibility of `from_sources`.
    pub fn from_bundle(
        model: impl AsRef<Path>,
        cache_root: impl AsRef<Path>,
        config: DirectMlConfig,
    ) -> Result<Self, DirectMlError> {
        config.validate()?;
        let synth = vocal_model::resolve_model_path(model)?;
        let inspection = vocal_model::inspect_model_artifact(&synth)?;
        let [hubert, rmvpe] = vocal_model::materialize_acoustic_sources(&synth, cache_root)?;
        Ok(Self {
            synth,
            hubert,
            rmvpe,
            config,
            enable_retrieval: inspection.components.retrieval.present,
        })
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self, DirectMlError> {
        let path = path.as_ref();
        let mut value: Self = serde_json::from_slice(&std::fs::read(path)?)?;
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        for source in [&mut value.synth, &mut value.hubert, &mut value.rmvpe] {
            if source.is_relative() {
                *source = root.join(&*source);
            }
        }
        value.config.validate()?;
        Ok(value)
    }
}
