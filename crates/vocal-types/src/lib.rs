//! Shared, dependency-light data types used by every runtime layer.

use serde::{Deserialize, Serialize};

mod config;
mod error;
mod state;

pub use config::{AudioProcessingConfig, ConversionConfig};
pub use error::TypesError;
pub use state::{LifecycleState, ProcessingStatus};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioFormat {
    pub const MONO_16K: Self = Self {
        sample_rate: 16_000,
        channels: 1,
    };
    pub const MONO_32K: Self = Self {
        sample_rate: 32_000,
        channels: 1,
    };
    pub const MONO_48K: Self = Self {
        sample_rate: 48_000,
        channels: 1,
    };

    pub const fn frame_samples(self) -> usize {
        self.channels as usize
    }

    pub fn validate(self) -> Result<(), TypesError> {
        if self.sample_rate == 0 {
            return Err(TypesError::InvalidAudioFormat {
                sample_rate: self.sample_rate,
                channels: self.channels,
                reason: "sample rate must be non-zero",
            });
        }
        if self.channels == 0 {
            return Err(TypesError::InvalidAudioFormat {
                sample_rate: self.sample_rate,
                channels: self.channels,
                reason: "channel count must be non-zero",
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioChunk {
    pub format: AudioFormat,
    /// Interleaved f32 PCM normalized to [-1, 1].
    pub samples: Vec<f32>,
    pub sequence: u64,
}

impl AudioChunk {
    pub fn frames(&self) -> usize {
        self.samples.len() / self.format.channels.max(1) as usize
    }

    pub fn validate(&self) -> Result<(), TypesError> {
        self.format.validate()?;
        if self.samples.is_empty() {
            return Err(TypesError::InvalidChunk("audio chunk is empty"));
        }
        if !self
            .samples
            .len()
            .is_multiple_of(self.format.channels as usize)
        {
            return Err(TypesError::InvalidChunk(
                "interleaved samples do not match channel count",
            ));
        }
        if self.samples.iter().any(|sample| !sample.is_finite()) {
            return Err(TypesError::InvalidChunk("audio samples must be finite"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelDescriptor {
    pub id: String,
    pub display_name: String,
    pub architecture: String,
    pub sample_rate: u32,
    pub version: String,
}

impl ModelDescriptor {
    pub fn validate(&self) -> Result<(), TypesError> {
        if self.id.trim().is_empty() {
            return Err(TypesError::InvalidDescriptor("model id is empty"));
        }
        if self.display_name.trim().is_empty() {
            return Err(TypesError::InvalidDescriptor("model display name is empty"));
        }
        if self.sample_rate == 0 {
            return Err(TypesError::InvalidDescriptor("model sample rate is zero"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct LatencyMetrics {
    pub input_frames: u64,
    pub processing_micros: u64,
    pub output_frames: u64,
}
