use serde::{Deserialize, Serialize};

use crate::{AudioFormat, TypesError};

/// Match the input register before synthesis. Manual pitch remains additive.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AutoPitchConfig {
    pub enabled: bool,
    pub target_hz: f32,
}

impl Default for AutoPitchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_hz: 255.0,
        }
    }
}

impl AutoPitchConfig {
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            target_hz: 255.0,
        }
    }

    pub fn validate(self) -> Result<(), TypesError> {
        if !self.target_hz.is_finite() || !(50.0..=1200.0).contains(&self.target_hz) {
            return Err(TypesError::InvalidConversion(
                "automatic pitch target must be finite and within [50, 1200] Hz",
            ));
        }
        Ok(())
    }
}

/// Conversion controls shared by the audio, engine and CLI layers.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ConversionConfig {
    /// Pitch shift applied before coarse/continuous F0 generation.
    pub pitch_semitones: f32,
    pub auto_pitch: AutoPitchConfig,
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
            auto_pitch: AutoPitchConfig::default(),
            index_ratio: 0.0,
            protect: 0.33,
            rms_mix_rate: 1.0,
        }
    }
}

impl ConversionConfig {
    pub fn validate(self) -> Result<(), TypesError> {
        self.auto_pitch.validate()?;
        if !self.pitch_semitones.is_finite() || !(-96.0..=96.0).contains(&self.pitch_semitones) {
            return Err(TypesError::InvalidConversion(
                "pitch shift must be finite and within [-96, 96] semitones",
            ));
        }
        for (name, value) in [
            ("index ratio", self.index_ratio),
            ("protect ratio", self.protect),
            ("RMS mix rate", self.rms_mix_rate),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(TypesError::InvalidConversion(match name {
                    "index ratio" => "index ratio must be finite and within [0, 1]",
                    "protect ratio" => "protect ratio must be finite and within [0, 1]",
                    _ => "RMS mix rate must be finite and within [0, 1]",
                }));
            }
        }
        Ok(())
    }
}

/// Application-level audio format and conversion defaults.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioProcessingConfig {
    pub input: AudioFormat,
    pub output: AudioFormat,
    pub conversion: ConversionConfig,
}

impl Default for AudioProcessingConfig {
    fn default() -> Self {
        Self {
            input: AudioFormat::MONO_16K,
            output: AudioFormat::MONO_16K,
            conversion: ConversionConfig::default(),
        }
    }
}

impl AudioProcessingConfig {
    pub fn validate(self) -> Result<(), TypesError> {
        self.input.validate()?;
        self.output.validate()?;
        self.conversion.validate()
    }
}
