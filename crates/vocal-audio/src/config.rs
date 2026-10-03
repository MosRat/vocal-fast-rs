use std::time::Duration;

use thiserror::Error;
use vocal_types::AudioFormat;

/// Fixed-size block configuration shared by the device and worker sides.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamConfig {
    pub format: AudioFormat,
    /// Number of interleaved frames in one inference block.
    pub block_frames: usize,
    /// Number of pre-allocated blocks in each queue.
    pub queue_blocks: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            format: AudioFormat::MONO_48K,
            // 10 ms at 48 kHz. A real-time model can choose a larger block
            // when its receptive field requires it.
            block_frames: 480,
            queue_blocks: 8,
        }
    }
}

impl StreamConfig {
    pub fn validate(self) -> Result<Self, AudioError> {
        if self.format.sample_rate == 0
            || self.format.channels == 0
            || self.block_frames == 0
            || self.queue_blocks == 0
        {
            return Err(AudioError::InvalidConfig(
                "sample rate, channels, block_frames and queue_blocks must be non-zero".into(),
            ));
        }
        let block_samples = self
            .block_frames
            .checked_mul(self.format.channels as usize)
            .ok_or_else(|| AudioError::InvalidConfig("block size overflows usize".into()))?;
        if block_samples == 0 {
            return Err(AudioError::InvalidConfig(
                "block contains no samples".into(),
            ));
        }
        block_samples
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| AudioError::InvalidConfig("block byte size overflows usize".into()))?;
        self.queue_blocks
            .checked_mul(block_samples)
            .and_then(|samples| samples.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                AudioError::InvalidConfig("queue allocation size overflows usize".into())
            })?;
        if self.queue_blocks > isize::MAX as usize {
            return Err(AudioError::InvalidConfig(
                "queue_blocks is too large for a bounded ring".into(),
            ));
        }
        Ok(self)
    }

    pub const fn block_samples(self) -> usize {
        self.block_frames * self.format.channels as usize
    }

    /// Duration represented by one inference block, rounded up to a whole
    /// microsecond for telemetry and latency budgeting.
    pub fn block_duration(self) -> Duration {
        let micros = (self.block_frames as u128 * 1_000_000)
            .div_ceil(self.format.sample_rate as u128)
            .min(u64::MAX as u128) as u64;
        Duration::from_micros(micros)
    }
}

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
}

pub(crate) fn allocate_samples(samples: usize) -> Result<Vec<f32>, AudioError> {
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(samples)
        .map_err(|error| AudioError::AllocationFailed(error.to_string()))?;
    buffer.resize(samples, 0.0);
    Ok(buffer)
}
