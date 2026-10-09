//! Model import is independent of engine ownership and the runtime control actor.
mod convert;
mod download;
mod jobs;
mod provider;

pub use jobs::ModelLibrary;
pub use provider::{
    ModelProvider, RemoteFile, RemoteModelCatalog, RemoteRepository, list_remote_models,
};
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
};

pub type LibraryResult<T> = Result<T, LibraryError>;
#[derive(Clone, Debug, thiserror::Error, Serialize)]
#[error("{message}")]
#[serde(rename_all = "camelCase")]
pub struct LibraryError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}
impl LibraryError {
    pub(crate) fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            retryable: matches!(code, "network" | "http" | "busy" | "disk"),
        }
    }
}
impl From<std::io::Error> for LibraryError {
    fn from(e: std::io::Error) -> Self {
        Self::new("disk", e.to_string())
    }
}
impl From<vocal_model::ModelError> for LibraryError {
    fn from(e: vocal_model::ModelError) -> Self {
        Self::new("invalidModel", e.to_string())
    }
}
impl From<reqwest::Error> for LibraryError {
    fn from(e: reqwest::Error) -> Self {
        Self::new("network", e.without_url().to_string())
    }
}

pub(crate) fn check_cancel(cancel: &AtomicBool) -> LibraryResult<()> {
    if cancel.load(Ordering::Acquire) {
        Err(LibraryError::new(
            "cancelled",
            "Import cancelled; no model was published",
        ))
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum ImportSource {
    Remote {
        repository: RemoteRepository,
        model_file: String,
        index_file: Option<String>,
    },
    Local {
        model_path: PathBuf,
        index_path: Option<PathBuf>,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelImportRequest {
    pub source: ImportSource,
    pub display_name: String,
    pub library_directory: Option<PathBuf>,
    pub acoustic_bundle: Option<PathBuf>,
    pub hubert_source: Option<PathBuf>,
    pub rmvpe_source: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ImportPhase {
    #[default]
    Idle,
    Queued,
    Listing,
    Downloading,
    Verifying,
    Converting,
    Bundling,
    Publishing,
    Ready,
    Cancelling,
    Cancelled,
    Failed,
}
impl ImportPhase {
    pub fn is_active(self) -> bool {
        !matches!(
            self,
            Self::Idle | Self::Ready | Self::Cancelled | Self::Failed
        )
    }
}
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelImportStatus {
    pub job_id: u64,
    pub revision: u64,
    pub phase: ImportPhase,
    pub message: String,
    pub file: Option<String>,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub elapsed_ms: u64,
    pub output_path: Option<PathBuf>,
    pub library_directory: Option<PathBuf>,
    pub complete: bool,
    pub warnings: Vec<String>,
    pub error: Option<LibraryError>,
}

pub fn default_library_directory() -> PathBuf {
    std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|v| PathBuf::from(v).join(".local/share")))
        .unwrap_or_else(std::env::temp_dir)
        .join("vocal-fast/model-library")
}

pub(crate) fn validate_relative(path: &str) -> LibraryResult<()> {
    if path.is_empty()
        || path.len() > 2048
        || path.contains(['\\', ':', '\0'])
        || path.starts_with('/')
        || path.split('/').any(|s| {
            s.is_empty()
                || s == "."
                || s == ".."
                || s.ends_with(['.', ' '])
                || s.chars().any(|c| c < ' ' || "<>\"|?*".contains(c))
                || matches!(
                    s.split('.')
                        .next()
                        .unwrap_or_default()
                        .to_ascii_uppercase()
                        .as_str(),
                    "CON"
                        | "PRN"
                        | "AUX"
                        | "NUL"
                        | "COM1"
                        | "COM2"
                        | "COM3"
                        | "COM4"
                        | "COM5"
                        | "COM6"
                        | "COM7"
                        | "COM8"
                        | "COM9"
                        | "LPT1"
                        | "LPT2"
                        | "LPT3"
                        | "LPT4"
                        | "LPT5"
                        | "LPT6"
                        | "LPT7"
                        | "LPT8"
                        | "LPT9"
                )
        })
    {
        return Err(LibraryError::new(
            "path",
            "Repository file has an unsafe path",
        ));
    }
    Ok(())
}

pub(crate) fn client() -> LibraryResult<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .user_agent("vocal-fast/0.1 model-library")
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(60))
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() >= 10 {
                attempt.error("too many redirects")
            } else if attempt.url().scheme() != "https" {
                attempt.error("model download requires HTTPS")
            } else {
                attempt.follow()
            }
        }))
        .build()?)
}
