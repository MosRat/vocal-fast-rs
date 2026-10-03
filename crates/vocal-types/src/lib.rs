//! Shared, dependency-light data types used by every runtime layer.

use serde::{Deserialize, Serialize};

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
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ConversionConfig {
    /// Pitch shift applied before coarse/continuous F0 generation.
    pub pitch_semitones: f32,
    /// Blend ratio for the optional added index, in [0, 1].
    pub index_ratio: f32,
    /// Unvoiced feature protection ratio, in [0, 1].
    pub protect: f32,
    /// Output RMS mix ratio; 1 keeps synthesizer loudness, 0 follows input.
    pub rms_mix_rate: f32,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            pitch_semitones: 0.0,
            index_ratio: 0.0,
            protect: 0.33,
            rms_mix_rate: 1.0,
        }
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

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct LatencyMetrics {
    pub input_frames: u64,
    pub processing_micros: u64,
    pub output_frames: u64,
}
