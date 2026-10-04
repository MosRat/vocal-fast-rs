//! Error types shared by the safe TensorRT and CUDA wrappers.

use std::path::PathBuf;

use thiserror::Error;

/// A malformed environment value or incompatible semantic configuration.
///
/// Environment parsing is intentionally kept separate from native API errors
/// so callers can report a useful variable name and expected value before any
/// CUDA/TensorRT handle is created.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ConfigError {
    #[error("invalid value for {name}={value:?}; expected {expected}")]
    InvalidValue {
        name: String,
        value: String,
        expected: String,
    },
    #[error("invalid TensorRT configuration: {0}")]
    InvalidCombination(String),
    #[error("invalid TensorRT profile {0:?}")]
    InvalidProfile(String),
}

/// Errors crossing the Rust/native TensorRT boundary.
#[derive(Debug, Error)]
pub enum NativeBuildError {
    #[error("configuration error: {0}")]
    Config(#[from] ConfigError),
    #[error("TensorRT stage {stage} failed: {source}")]
    Stage {
        stage: &'static str,
        #[source]
        source: Box<NativeBuildError>,
    },
    #[error("TensorRT-RTX SDK file is missing: {0}")]
    Missing(PathBuf),
    #[error("TensorRT-RTX API failed: {0}")]
    Api(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid graph JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid argument: {0}")]
    Argument(String),
}

impl NativeBuildError {
    /// Add the graph stage that was active when a native error crossed the
    /// FFI boundary. This keeps the original error available as its source.
    pub fn with_stage(self, stage: &'static str) -> Self {
        Self::Stage {
            stage,
            source: Box::new(self),
        }
    }
}
