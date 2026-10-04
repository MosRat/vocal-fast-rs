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

impl ModelError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Io(_) => "model.io",
            Self::Json(_) => "model.json",
            Self::InvalidCache(_) => "model.invalid_cache",
            Self::MissingFile(_) => "model.missing_file",
            Self::UnsupportedModel(_) => "model.unsupported",
            Self::InvalidConfig(_) => "model.invalid_config",
        }
    }

    pub const fn is_recoverable(&self) -> bool {
        matches!(self, Self::MissingFile(_) | Self::InvalidConfig(_))
    }
}
