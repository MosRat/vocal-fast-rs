//! Small dependency-free WAV reader/writer for offline inference tools.
//!
//! Device streaming stays in the platform backends. This module only handles
//! deterministic PCM/IEEE-float file conversion at the application boundary.

use std::{fs, path::Path};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WavError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid WAV: {0}")]
    Invalid(String),
    #[error("unsupported WAV format: {0}")]
    Unsupported(String),
}

#[derive(Clone, Debug)]
pub struct WavData {
    pub sample_rate: u32,
    pub channels: u16,
    /// Interleaved samples normalized to approximately [-1, 1].
    pub samples: Vec<f32>,
}

impl WavData {
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels.max(1) as usize
    }

    pub fn mono(&self) -> Vec<f32> {
        let channels = self.channels.max(1) as usize;
        if channels == 1 {
            return self.samples.clone();
        }
        self.samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
            .collect()
    }

    /// Convert to mono 16 kHz with linear interpolation. The model boundary
    /// uses 16 kHz waveform samples, while the output writer preserves the
    /// Synth model's native sample rate.
    pub fn mono_16k(&self) -> Vec<f32> {
        resample_linear(&self.mono(), self.sample_rate, 16_000)
    }
}

pub fn read_wav(path: impl AsRef<Path>) -> Result<WavData, WavError> {
    read_wav_bytes(&fs::read(path)?)
}

pub fn read_wav_bytes(bytes: &[u8]) -> Result<WavData, WavError> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(WavError::Invalid("expected RIFF/WAVE header".into()));
    }
    let mut cursor = 12usize;
    let mut format = None;
    let mut data = None;
    while cursor.checked_add(8).is_some_and(|end| end <= bytes.len()) {
        let id = &bytes[cursor..cursor + 4];
        let size = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        let start = cursor + 8;
        let end = start
            .checked_add(size)
            .ok_or_else(|| WavError::Invalid("chunk size overflow".into()))?;
        if end > bytes.len() {
            return Err(WavError::Invalid("chunk extends past file".into()));
        }
        match id {
            b"fmt " => format = Some(parse_format(&bytes[start..end])?),
            b"data" => data = Some(&bytes[start..end]),
            _ => {}
        }
        cursor = end + (size & 1);
    }
    let format = format.ok_or_else(|| WavError::Invalid("fmt chunk missing".into()))?;
    let data = data.ok_or_else(|| WavError::Invalid("data chunk missing".into()))?;
    decode_samples(format, data)
}

#[derive(Clone, Copy)]
struct WavFormat {
    code: u16,
    channels: u16,
    sample_rate: u32,
    bits: u16,
    block_align: u16,
}

fn parse_format(bytes: &[u8]) -> Result<WavFormat, WavError> {
    if bytes.len() < 16 {
        return Err(WavError::Invalid("fmt chunk is truncated".into()));
    }
    let format = WavFormat {
        code: u16::from_le_bytes(bytes[0..2].try_into().unwrap()),
        channels: u16::from_le_bytes(bytes[2..4].try_into().unwrap()),
        sample_rate: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        bits: u16::from_le_bytes(bytes[14..16].try_into().unwrap()),
        block_align: u16::from_le_bytes(bytes[12..14].try_into().unwrap()),
    };
    if format.channels == 0 || format.sample_rate == 0 || format.block_align == 0 {
        return Err(WavError::Invalid("fmt chunk contains zero geometry".into()));
    }
    if !matches!(format.code, 1 | 3) {
        return Err(WavError::Unsupported(format!(
            "audio format code {} (only PCM=1 and float=3 are supported)",
            format.code
        )));
    }
    if format.code == 3 && format.bits != 32 {
        return Err(WavError::Unsupported(
            "IEEE float WAV must be 32-bit".into(),
        ));
    }
    if format.code == 1 && !matches!(format.bits, 8 | 16 | 24 | 32) {
        return Err(WavError::Unsupported(format!(
            "PCM {}-bit samples",
            format.bits
        )));
    }
    Ok(format)
}

fn decode_samples(format: WavFormat, data: &[u8]) -> Result<WavData, WavError> {
    if data.len() % format.block_align as usize != 0 {
        return Err(WavError::Invalid(
            "data chunk has an incomplete frame".into(),
        ));
    }
    let bytes_per_sample = (format.bits / 8) as usize;
    if bytes_per_sample * format.channels as usize != format.block_align as usize {
        return Err(WavError::Invalid(
            "fmt block alignment does not match sample width".into(),
        ));
    }
    let mut samples = Vec::with_capacity(data.len() / bytes_per_sample);
    for sample in data.chunks_exact(bytes_per_sample) {
        let value = if format.code == 3 {
            f32::from_le_bytes(sample.try_into().unwrap())
        } else {
            match format.bits {
                8 => (sample[0] as f32 - 128.0) / 128.0,
                16 => i16::from_le_bytes(sample.try_into().unwrap()) as f32 / 32768.0,
                24 => {
                    let raw = i32::from_le_bytes([
                        sample[0],
                        sample[1],
                        sample[2],
                        if sample[2] & 0x80 != 0 { 0xff } else { 0 },
                    ]);
                    raw as f32 / 8_388_608.0
                }
                32 => i32::from_le_bytes(sample.try_into().unwrap()) as f32 / 2_147_483_648.0,
                _ => unreachable!(),
            }
        };
        samples.push(if value.is_finite() { value } else { 0.0 });
    }
    Ok(WavData {
        sample_rate: format.sample_rate,
        channels: format.channels,
        samples,
    })
}

pub fn resample_linear(input: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if input.is_empty() || source_rate == 0 || target_rate == 0 {
        return Vec::new();
    }
    if source_rate == target_rate {
        return input.to_vec();
    }
    let output_len = ((input.len() as u128 * target_rate as u128) / source_rate as u128)
        .max(1)
        .min(usize::MAX as u128) as usize;
    let scale = source_rate as f64 / target_rate as f64;
    (0..output_len)
        .map(|index| {
            let position = index as f64 * scale;
            let left = position.floor() as usize;
            let right = (left + 1).min(input.len() - 1);
            let fraction = (position - left as f64) as f32;
            input[left.min(input.len() - 1)] * (1.0 - fraction) + input[right] * fraction
        })
        .collect()
}

pub fn write_wav_pcm16(
    path: impl AsRef<Path>,
    sample_rate: u32,
    samples: &[f32],
) -> Result<(), WavError> {
    if sample_rate == 0 {
        return Err(WavError::Invalid("output sample rate is zero".into()));
    }
    let data_len = samples
        .len()
        .checked_mul(2)
        .ok_or_else(|| WavError::Invalid("output is too large".into()))?;
    let riff_len = 36usize
        .checked_add(data_len)
        .ok_or_else(|| WavError::Invalid("output RIFF size overflow".into()))?;
    if riff_len > u32::MAX as usize {
        return Err(WavError::Invalid("output exceeds RIFF size limit".into()));
    }
    let byte_rate = sample_rate
        .checked_mul(2)
        .ok_or_else(|| WavError::Invalid("output byte rate overflow".into()))?;
    let mut bytes = Vec::with_capacity(44 + data_len);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(riff_len as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(data_len as u32).to_le_bytes());
    for &sample in samples {
        let sample = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm16_round_trip_and_downmix() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&40u32.to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16_000u32.to_le_bytes());
        bytes.extend_from_slice(&64_000u32.to_le_bytes());
        bytes.extend_from_slice(&4u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&16_384i16.to_le_bytes());
        bytes.extend_from_slice(&(-16_384i16).to_le_bytes());
        let wav = read_wav_bytes(&bytes).expect("decode");
        assert_eq!(wav.frames(), 1);
        assert!(wav.mono()[0].abs() < 1e-6);
    }
}
