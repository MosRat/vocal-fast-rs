//! Error types shared by the safe TensorRT and CUDA wrappers.

use std::path::PathBuf;

use thiserror::Error;

/// Errors crossing the Rust/native TensorRT boundary.
#[derive(Debug, Error)]
pub enum NativeBuildError {
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
