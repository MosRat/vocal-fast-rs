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
    #[error("audio block has the wrong number of samples: expected {expected}, got {actual}")]
    BlockSizeMismatch { expected: usize, actual: usize },
    #[error("audio callback data is not a whole number of frames")]
    PartialFrame,
    #[error("audio buffer allocation failed: {0}")]
    AllocationFailed(String),
    #[error("audio stream operation timed out")]
    Timeout,
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
}
