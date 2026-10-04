use std::{io, path::PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("runtime configuration error: {0}")]
    Config(String),
    #[error("runtime state error for {action}: {state}")]
    State { state: String, action: String },
    #[error("configuration revision conflict: expected {expected}, actual {actual}")]
    ConfigConflict { expected: u64, actual: u64 },
    #[error("runtime lock was poisoned")]
    Lock,
    #[error("model error: {0}")]
    Model(#[from] vocal_model::ModelError),
    #[error("engine error: {0}")]
    Engine(#[from] vocal_engine::EngineError),
    #[error("audio type error: {0}")]
    Types(#[from] vocal_types::TypesError),
    #[error("I/O error at {path}: {message}")]
    Io { path: PathBuf, message: String },
    #[error("configuration JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl RuntimeError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Config(_) => "runtime.invalid_config",
            Self::State { .. } => "runtime.invalid_state",
            Self::ConfigConflict { .. } => "runtime.config_conflict",
            Self::Lock => "runtime.lock",
            Self::Model(error) => error.code(),
            Self::Engine(error) => error.code(),
            Self::Types(_) => "runtime.invalid_types",
            Self::Io { .. } => "runtime.io",
            Self::Json(_) => "runtime.json",
        }
    }
}

pub(crate) fn io_at(path: impl Into<PathBuf>, error: io::Error) -> RuntimeError {
    RuntimeError::Io {
        path: path.into(),
        message: error.to_string(),
    }
}

pub type RuntimeResult<T> = Result<T, RuntimeError>;
