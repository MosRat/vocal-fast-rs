//! Public error and control-plane fault types.

use thiserror::Error;

/// Errors returned directly by the allocation-free audio core.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AudioError {
    #[error("audio queue is full")]
    QueueFull,
    #[error("audio queue is empty")]
    QueueEmpty,
    #[error("invalid audio configuration: {0}")]
    InvalidConfig(String),
    #[error("audio processing failed: {0}")]
    Processing(String),
    #[error("audio block has the wrong number of samples: expected {expected}, got {actual}")]
    BlockSizeMismatch { expected: usize, actual: usize },
    #[error("audio callback data is not a whole number of frames")]
    PartialFrame,
    #[error("audio buffer allocation failed: {0}")]
    AllocationFailed(String),
    #[error("audio stream operation timed out")]
    Timeout,
}

impl AudioError {
    /// Stable machine-readable identifier for control-plane adapters.
    pub const fn code(&self) -> &'static str {
        match self {
            Self::QueueFull => "audio.queue_full",
            Self::QueueEmpty => "audio.queue_empty",
            Self::InvalidConfig(_) => "audio.invalid_config",
            Self::Processing(_) => "audio.processing",
            Self::BlockSizeMismatch { .. } => "audio.block_size_mismatch",
            Self::PartialFrame => "audio.partial_frame",
            Self::AllocationFailed(_) => "audio.allocation_failed",
            Self::Timeout => "audio.timeout",
        }
    }

    /// Queue pressure and scheduler timeouts can be retried without tearing
    /// down the device. Configuration and memory errors require intervention.
    pub const fn is_transient(&self) -> bool {
        matches!(self, Self::QueueFull | Self::QueueEmpty | Self::Timeout)
    }
}

/// Coarse class attached to a backend fault. The audio callback records only
/// compact atomics; formatting and logging happen on the control plane.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AudioFaultKind {
    #[default]
    Unknown,
    Startup,
    Capture,
    Render,
    Worker,
    Shutdown,
}

/// Direction associated with a backend fault.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AudioFaultDirection {
    Capture,
    Render,
    Worker,
}

/// Structured control-plane fault returned by a backend after a fatal device
/// condition. `occurrences` includes repeated reports of the same first
/// fault, so callers can distinguish a single failure from a noisy driver.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioFault {
    pub kind: AudioFaultKind,
    pub direction: Option<AudioFaultDirection>,
    pub message: String,
    pub occurrences: u64,
}

impl AudioFault {
    pub fn new(
        kind: AudioFaultKind,
        direction: Option<AudioFaultDirection>,
        message: String,
    ) -> Self {
        Self {
            kind,
            direction,
            message,
            occurrences: 1,
        }
    }

    pub const fn code(&self) -> &'static str {
        match self.kind {
            AudioFaultKind::Unknown => "audio.backend_unknown",
            AudioFaultKind::Startup => "audio.backend_startup",
            AudioFaultKind::Capture => "audio.backend_capture",
            AudioFaultKind::Render => "audio.backend_render",
            AudioFaultKind::Worker => "audio.backend_worker",
            AudioFaultKind::Shutdown => "audio.backend_shutdown",
        }
    }
}
