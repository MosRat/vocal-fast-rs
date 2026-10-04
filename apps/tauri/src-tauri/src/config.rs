use std::{fs, path::PathBuf};

use serde::{Deserialize, Serialize};
use vocal_types::{AudioFormat, AudioProcessingConfig, ConversionConfig, TypesError};

use crate::error::{GuiError, GuiResult, io_at};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GuiBackend {
    #[default]
    Passthrough,
    RvcCpu,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GuiConfig {
    pub input_format: AudioFormat,
    pub output_format: AudioFormat,
    pub conversion: ConversionConfig,
    pub backend: GuiBackend,
    pub model_path: Option<String>,
}

impl Default for GuiConfig {
    fn default() -> Self {
        Self {
            input_format: AudioFormat::MONO_16K,
            output_format: AudioFormat::MONO_16K,
            conversion: ConversionConfig::default(),
            backend: GuiBackend::Passthrough,
            model_path: None,
        }
    }
}

impl GuiConfig {
    pub fn validate(&self) -> GuiResult<()> {
        AudioProcessingConfig {
            input: self.input_format,
            output: self.output_format,
            conversion: self.conversion,
        }
        .validate()
        .map_err(|error: TypesError| GuiError::InvalidConfig(error.to_string()))
    }
}

#[derive(Clone, Debug)]
pub struct ConfigStore {
    path: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_store_round_trips_and_rejects_invalid_values() {
        let path = std::env::temp_dir().join(format!(
            "vocal-fast-config-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let store = ConfigStore::new(path.clone());
        let config = GuiConfig::default();
        store.save(&config).expect("config save");
        assert!(store.load().expect("config load").is_some());
        let mut invalid = config;
        invalid.input_format.sample_rate = 0;
        assert!(matches!(
            store.save(&invalid),
            Err(GuiError::InvalidConfig(_))
        ));
        std::fs::remove_file(path).expect("remove test config");
    }
}

impl ConfigStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn load(&self) -> GuiResult<Option<GuiConfig>> {
        if !self.path.is_file() {
            return Ok(None);
        }
        let bytes = fs::read(&self.path).map_err(|error| io_at(&self.path, error))?;
        let config: GuiConfig = serde_json::from_slice(&bytes)
            .map_err(|error| GuiError::InvalidConfig(format!("cannot parse config: {error}")))?;
        config.validate()?;
        Ok(Some(config))
    }

    pub fn save(&self, config: &GuiConfig) -> GuiResult<()> {
        config.validate()?;
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|error| io_at(parent, error))?;
        }
        let bytes = serde_json::to_vec_pretty(config)?;
        let temp = self.path.with_extension("json.tmp");
        fs::write(&temp, bytes).map_err(|error| io_at(&temp, error))?;
        if let Err(error) = fs::rename(&temp, &self.path) {
            if error.kind() != std::io::ErrorKind::AlreadyExists {
                return Err(io_at(&self.path, error));
            }
            fs::remove_file(&self.path).map_err(|error| io_at(&self.path, error))?;
            fs::rename(&temp, &self.path).map_err(|error| io_at(&self.path, error))?;
        }
        Ok(())
    }
}
