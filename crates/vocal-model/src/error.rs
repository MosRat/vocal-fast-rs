use std::{io, path::PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("manifest or metadata JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid model cache: {0}")]
    InvalidCache(String),
    #[error("missing model file: {0}")]
    MissingFile(PathBuf),
    #[error("unsupported RVC model: {0}")]
    UnsupportedModel(String),
    #[error("invalid model loader configuration: {0}")]
    InvalidConfig(String),
}
