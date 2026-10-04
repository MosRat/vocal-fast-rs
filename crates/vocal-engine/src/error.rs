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

impl EngineError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::BackendUnavailable(_) => "engine.backend_unavailable",
            Self::InvalidConfig(_) => "engine.invalid_config",
            Self::InvalidInput(_) => "engine.invalid_input",
            Self::InvalidState(_) => "engine.invalid_state",
        }
    }

    pub const fn is_recoverable(&self) -> bool {
        matches!(self, Self::InvalidConfig(_) | Self::InvalidInput(_))
    }
}
