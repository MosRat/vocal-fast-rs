use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("engine backend is unavailable: {0}")]
    BackendUnavailable(String),
    #[error("invalid engine configuration: {0}")]
    InvalidConfig(String),
    #[error("invalid audio input: {0}")]
    InvalidInput(String),
    #[error("invalid engine state: {0}")]
    InvalidState(String),
}
