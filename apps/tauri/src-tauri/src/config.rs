//! Tauri's configuration adapter.
//!
//! The schema and atomic persistence implementation live in `vocal-runtime`.
//! This wrapper only translates runtime errors into the GUI error protocol.

pub use vocal_runtime::RuntimeConfig as GuiConfig;

use std::path::PathBuf;

use crate::error::{GuiError, GuiResult};

#[derive(Clone, Debug)]
pub struct ConfigStore {
    inner: vocal_runtime::ConfigStore,
}

impl ConfigStore {
    pub fn new(path: PathBuf) -> Self {
        Self {
            inner: vocal_runtime::ConfigStore::new(path),
        }
    }

    #[allow(dead_code)]
    pub fn load(&self) -> GuiResult<Option<GuiConfig>> {
        self.inner.load().map_err(GuiError::from)
    }

    #[allow(dead_code)]
    pub fn save(&self, config: &GuiConfig) -> GuiResult<()> {
        self.inner.save(config).map_err(GuiError::from)
    }

    pub(crate) fn runtime_store(&self) -> vocal_runtime::ConfigStore {
        self.inner.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_runtime_config_round_trips() {
        let path = std::env::temp_dir().join(format!(
            "vocal-tauri-config-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let store = ConfigStore::new(path.clone());
        let config = GuiConfig::default();
        store.save(&config).expect("save");
        assert_eq!(store.load().expect("load").unwrap().schema_version, 1);
        std::fs::remove_file(path).expect("remove");
    }
}
