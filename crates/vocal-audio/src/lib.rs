//! Real-time audio boundary. Device code stays behind a feature so the core
//! model and engine crates can be built on CI machines without audio drivers.

use std::collections::VecDeque;
use thiserror::Error;
use vocal_types::{AudioChunk, AudioFormat};

#[derive(Clone, Copy, Debug)]
pub struct StreamConfig {
    pub format: AudioFormat,
    pub block_frames: usize,
    pub queue_blocks: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            format: AudioFormat::MONO_48K,
            block_frames: 480,
            queue_blocks: 8,
        }
    }
}

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("audio queue is full")]
    QueueFull,
    #[error("audio queue is empty")]
    QueueEmpty,
    #[error("invalid audio configuration: {0}")]
    InvalidConfig(String),
}

/// Bounded queue used between the device callback and the conversion worker.
/// Device adapters should move an already-owned chunk with `push_owned` so the
/// callback does not copy or resize audio data.
pub struct AudioRing {
    config: StreamConfig,
    queue: VecDeque<AudioChunk>,
    next_sequence: u64,
}

impl AudioRing {
    pub fn new(config: StreamConfig) -> Result<Self, AudioError> {
        if config.format.channels == 0 || config.block_frames == 0 || config.queue_blocks == 0 {
            return Err(AudioError::InvalidConfig(
                "channels, block_frames and queue_blocks must be non-zero".into(),
            ));
        }
        Ok(Self {
            config,
            queue: VecDeque::with_capacity(config.queue_blocks),
            next_sequence: 0,
        })
    }

    pub fn config(&self) -> StreamConfig {
        self.config
    }

    pub fn push(&mut self, samples: &[f32]) -> Result<(), AudioError> {
        let expected = self.config.block_frames * self.config.format.channels as usize;
        if samples.len() != expected {
            return Err(AudioError::InvalidConfig(format!(
                "expected {expected} samples, got {}",
                samples.len()
            )));
        }
        if self.queue.len() >= self.config.queue_blocks {
            return Err(AudioError::QueueFull);
        }
        self.queue.push_back(AudioChunk {
            format: self.config.format,
            samples: samples.to_vec(),
            sequence: self.next_sequence,
        });
        self.next_sequence = self.next_sequence.wrapping_add(1);
        Ok(())
    }

    pub fn push_owned(&mut self, mut chunk: AudioChunk) -> Result<(), AudioError> {
        let expected = self.config.block_frames * self.config.format.channels as usize;
        if chunk.format != self.config.format || chunk.samples.len() != expected {
            return Err(AudioError::InvalidConfig(
                "audio chunk does not match stream configuration".into(),
            ));
        }
        if self.queue.len() >= self.config.queue_blocks {
            return Err(AudioError::QueueFull);
        }
        chunk.sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.queue.push_back(chunk);
        Ok(())
    }

    pub fn pop(&mut self) -> Result<AudioChunk, AudioError> {
        self.queue.pop_front().ok_or(AudioError::QueueEmpty)
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

#[cfg(feature = "cpal-backend")]
pub mod cpal_backend {
    //! Thin adapter placeholder. Device selection and callback wiring belong in
    //! the application layer, while `AudioRing` remains deterministic and testable.
    pub use cpal;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_preserves_sequence_and_bounds() {
        let mut ring = AudioRing::new(StreamConfig {
            format: AudioFormat::MONO_16K,
            block_frames: 2,
            queue_blocks: 1,
        })
        .unwrap();
        ring.push(&[0.0, 0.1]).unwrap();
        assert_eq!(
            ring.push(&[0.2, 0.3]).unwrap_err().to_string(),
            "audio queue is full"
        );
        assert_eq!(ring.pop().unwrap().sequence, 0);
        assert!(ring.is_empty());
    }
}
