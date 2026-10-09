use std::{
    array::TryFromSliceError,
    io,
    num::{ParseIntError, TryFromIntError},
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CliError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("model error: {0}")]
    Model(#[from] vocal_model::ModelError),
    #[error("engine error: {0}")]
    Engine(#[from] vocal_engine::EngineError),
    #[error("runtime error: {0}")]
    Runtime(#[from] vocal_runtime::RuntimeError),
    #[error("TensorRT graph error: {0}")]
    Graph(#[from] vocal_engine::tensorrt::GraphError),
    #[error("audio error: {0}")]
    Audio(#[from] vocal_audio::AudioError),
    #[error("WAV error: {0}")]
    Wav(#[from] vocal_audio::WavError),
    #[cfg(feature = "directml")]
    #[error("DirectML error: {0}")]
    DirectMl(#[from] vocal_directml::DirectMlError),
    #[error("invalid integer: {0}")]
    ParseInt(#[from] ParseIntError),
    #[error("invalid binary input: {0}")]
    Binary(#[from] TryFromSliceError),
    #[error("numeric value does not fit host size: {0}")]
    IntegerRange(#[from] TryFromIntError),
    #[error("{0}")]
    Message(String),
    #[error("{0}")]
    External(String),
}

impl From<Box<dyn std::error::Error>> for CliError {
    fn from(error: Box<dyn std::error::Error>) -> Self {
        Self::External(error.to_string())
    }
}

impl From<&'static str> for CliError {
    fn from(error: &'static str) -> Self {
        Self::Message(error.into())
    }
}

impl From<String> for CliError {
    fn from(error: String) -> Self {
        Self::Message(error)
    }
}

#[cfg(feature = "tensorrt-rtx")]
impl From<vocal_tensorrt::NativeBuildError> for CliError {
    fn from(error: vocal_tensorrt::NativeBuildError) -> Self {
        Self::External(error.to_string())
    }
}

#[cfg(feature = "tensorrt-rtx")]
impl From<vocal_tensorrt::ConfigError> for CliError {
    fn from(error: vocal_tensorrt::ConfigError) -> Self {
        Self::External(error.to_string())
    }
}

pub type CliResult<T> = Result<T, CliError>;
