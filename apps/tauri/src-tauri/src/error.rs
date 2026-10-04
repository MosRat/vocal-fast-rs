use std::{array::TryFromSliceError, io, num::TryFromIntError, path::PathBuf};

use serde::Serialize;
use thiserror::Error;

#[derive(Clone, Debug, Error, Serialize)]
#[serde(tag = "kind", content = "details")]
#[serde(rename_all = "snake_case")]
pub enum GuiError {
    #[error("invalid GUI configuration: {0}")]
    InvalidConfig(String),
    #[error("configuration revision conflict: expected {expected}, actual {actual}")]
    ConfigConflict { expected: u64, actual: u64 },
    #[error("invalid application state for {action}: {state}")]
    InvalidState { state: String, action: String },
    #[error("model error: {0}")]
    Model(String),
    #[error("engine error: {0}")]
    Engine(String),
    #[error("invalid audio input: {0}")]
    InvalidAudio(String),
    #[error("audio device error: {0}")]
    Audio(String),
    #[error("audio device discovery error: {0}")]
    AudioDiscovery(String),
    #[error("I/O error at {path}: {message}")]
    Io { path: String, message: String },
    #[error("channel error: {0}")]
    Channel(String),
    #[error("invalid binary payload: {0}")]
    Binary(String),
    #[error("numeric value is out of range: {0}")]
    IntegerRange(String),
    #[error("internal GUI error: {0}")]
    Internal(String),
}

impl GuiError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidConfig(_) => "invalid_config",
            Self::ConfigConflict { .. } => "config_conflict",
            Self::InvalidState { .. } => "invalid_state",
            Self::Model(_) => "model_error",
            Self::Engine(_) => "engine_error",
            Self::InvalidAudio(_) => "invalid_audio",
            Self::Audio(_) => "audio_error",
            Self::AudioDiscovery(_) => "audio_discovery_error",
            Self::Io { .. } => "io_error",
            Self::Channel(_) => "channel_error",
            Self::Binary(_) => "binary_error",
            Self::IntegerRange(_) => "integer_range",
            Self::Internal(_) => "internal_error",
        }
    }
}

impl From<vocal_model::ModelError> for GuiError {
    fn from(error: vocal_model::ModelError) -> Self {
        match error {
            vocal_model::ModelError::Io(error) => Self::Io {
                path: String::new(),
                message: error.to_string(),
            },
            other => Self::Model(other.to_string()),
        }
    }
}

impl From<vocal_engine::EngineError> for GuiError {
    fn from(error: vocal_engine::EngineError) -> Self {
        match error {
            vocal_engine::EngineError::BackendUnavailable(message) => Self::Engine(message),
            vocal_engine::EngineError::InvalidConfig(message) => Self::InvalidConfig(message),
            vocal_engine::EngineError::InvalidInput(message) => Self::InvalidAudio(message),
            vocal_engine::EngineError::InvalidState(message) => Self::InvalidState {
                state: message,
                action: "process".into(),
            },
        }
    }
}

impl From<io::Error> for GuiError {
    fn from(error: io::Error) -> Self {
        Self::Io {
            path: String::new(),
            message: error.to_string(),
        }
    }
}

impl From<serde_json::Error> for GuiError {
    fn from(error: serde_json::Error) -> Self {
        Self::Internal(format!("JSON: {error}"))
    }
}

impl From<TryFromSliceError> for GuiError {
    fn from(error: TryFromSliceError) -> Self {
        Self::Binary(error.to_string())
    }
}

impl From<TryFromIntError> for GuiError {
    fn from(error: TryFromIntError) -> Self {
        Self::IntegerRange(error.to_string())
    }
}

impl From<tauri::Error> for GuiError {
    fn from(error: tauri::Error) -> Self {
        Self::Internal(error.to_string())
    }
}

pub fn io_at(path: impl Into<PathBuf>, error: io::Error) -> GuiError {
    GuiError::Io {
        path: path.into().display().to_string(),
        message: error.to_string(),
    }
}

pub type GuiResult<T> = Result<T, GuiError>;
