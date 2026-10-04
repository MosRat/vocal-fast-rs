use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use serde::{Deserialize, Serialize};
use vocal_types::{AudioFormat, AudioProcessingConfig, ConversionConfig, TypesError};

use crate::error::{GuiError, GuiResult, io_at};

pub const GUI_CONFIG_SCHEMA_VERSION: u32 = 1;
static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GuiBackend {
    #[default]
    Passthrough,
    RvcCpu,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(default)]
pub struct GuiConfig {
    #[serde(default = "current_schema_version")]
    pub schema_version: u32,
    pub input_format: AudioFormat,
    pub output_format: AudioFormat,
    pub conversion: ConversionConfig,
    pub backend: GuiBackend,
    pub model_path: Option<String>,
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub block_frames: usize,
    pub queue_blocks: usize,
}

impl Default for GuiConfig {
    fn default() -> Self {
        Self {
            schema_version: GUI_CONFIG_SCHEMA_VERSION,
            input_format: AudioFormat::MONO_48K,
            output_format: AudioFormat::MONO_48K,
            conversion: ConversionConfig::default(),
            backend: GuiBackend::Passthrough,
            model_path: None,
            input_device: None,
            output_device: None,
            block_frames: 480,
            queue_blocks: 3,
        }
    }
}

impl GuiConfig {
    pub fn validate(&self) -> GuiResult<()> {
        if self.schema_version > GUI_CONFIG_SCHEMA_VERSION {
            return Err(GuiError::InvalidConfig(format!(
                "configuration schema {} is newer than supported schema {}",
                self.schema_version, GUI_CONFIG_SCHEMA_VERSION
            )));
        }
        AudioProcessingConfig {
            input: self.input_format,
            output: self.output_format,
            conversion: self.conversion,
        }
        .validate()
        .map_err(|error: TypesError| GuiError::InvalidConfig(error.to_string()))?;
        if self.block_frames == 0 || self.queue_blocks == 0 {
            return Err(GuiError::InvalidConfig(
                "block_frames and queue_blocks must be non-zero".into(),
            ));
        }
        if self.block_frames > 8192 || self.queue_blocks > 64 {
            return Err(GuiError::InvalidConfig(
                "block_frames must be <= 8192 and queue_blocks must be <= 64".into(),
            ));
        }
        if self.input_format != self.output_format {
            return Err(GuiError::InvalidConfig(
                "input and output formats must match for duplex streaming".into(),
            ));
        }
        Ok(())
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

    #[test]
    fn unversioned_config_is_migrated_in_memory() {
        let path = std::env::temp_dir().join(format!(
            "vocal-fast-config-migrate-{}-{}.json",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let mut value = serde_json::to_value(GuiConfig::default()).expect("serialize config");
        value
            .as_object_mut()
            .expect("config object")
            .remove("schemaVersion");
        fs::write(&path, serde_json::to_vec(&value).expect("serialize legacy config"))
            .expect("write legacy config");
        let config = ConfigStore::new(path.clone())
            .load()
            .expect("load legacy config")
            .expect("legacy config exists");
        assert_eq!(config.schema_version, GUI_CONFIG_SCHEMA_VERSION);
        fs::remove_file(path).expect("remove test config");
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
        let mut config: GuiConfig = serde_json::from_slice(&bytes)
            .map_err(|error| GuiError::InvalidConfig(format!("cannot parse config: {error}")))?;
        // Schema 0 was the unversioned format shipped before the control
        // plane grew migrations. It is structurally identical to schema 1.
        if config.schema_version == 0 {
            config.schema_version = GUI_CONFIG_SCHEMA_VERSION;
        }
        config.validate()?;
        Ok(Some(config))
    }

    pub fn save(&self, config: &GuiConfig) -> GuiResult<()> {
        config.validate()?;
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|error| io_at(parent, error))?;
        }
        let bytes = serde_json::to_vec_pretty(config)?;
        let temp = self.temp_path();
        let result = (|| {
            let mut file = fs::File::create(&temp).map_err(|error| io_at(&temp, error))?;
            file.write_all(&bytes).map_err(|error| io_at(&temp, error))?;
            // Ensure a successful command means the bytes reached the OS
            // before replacing the previous revision.
            file.sync_all().map_err(|error| io_at(&temp, error))?;
            replace_file(&temp, &self.path).map_err(|error| io_at(&self.path, error))?;
            sync_parent(self.path.parent())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result
    }

    fn temp_path(&self) -> PathBuf {
        let name = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("config.json");
        self.path.with_file_name(format!(
            ".{name}.{}.{}.tmp",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }
}

fn current_schema_version() -> u32 {
    GUI_CONFIG_SCHEMA_VERSION
}

#[cfg(not(windows))]
fn replace_file(temp: &Path, target: &Path) -> std::io::Result<()> {
    fs::rename(temp, target)
}

#[cfg(windows)]
fn replace_file(temp: &Path, target: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };
    let source: Vec<u16> = temp.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = target.as_os_str().encode_wide().chain(Some(0)).collect();
    let flags = MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH;
    let status = unsafe { MoveFileExW(source.as_ptr(), destination.as_ptr(), flags) };
    (status != 0).then_some(()).ok_or_else(std::io::Error::last_os_error)
}

fn sync_parent(parent: Option<&Path>) -> GuiResult<()> {
    #[cfg(unix)]
    if let Some(parent) = parent {
        fs::File::open(parent)
            .and_then(|file| file.sync_all())
            .map_err(|error| io_at(parent, error))?;
    }
    #[cfg(not(unix))]
    let _ = parent;
    Ok(())
}
