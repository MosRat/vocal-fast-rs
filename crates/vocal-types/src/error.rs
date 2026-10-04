use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum TypesError {
    #[error("invalid audio format {sample_rate} Hz/{channels} channels: {reason}")]
    InvalidAudioFormat {
        sample_rate: u32,
        channels: u16,
        reason: &'static str,
    },
    #[error("invalid audio chunk: {0}")]
    InvalidChunk(&'static str),
    #[error("invalid conversion configuration: {0}")]
    InvalidConversion(&'static str),
    #[error("invalid model descriptor: {0}")]
    InvalidDescriptor(&'static str),
}
