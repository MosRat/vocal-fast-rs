//! Audio format and streaming-window adapters shared by engine backends.
//!
//! The neural backends consume mono 16 kHz windows, while the public engine
//! contract uses interleaved PCM in the caller's format.  Keeping these
//! conversions here prevents each backend from growing its own subtly
//! different crop, padding, and channel handling rules.

use vocal_types::AudioFormat;

use crate::{EngineError, pipeline};

/// Convert an interleaved block to mono by averaging its channels.
pub(crate) fn to_mono(format: AudioFormat, samples: &[f32]) -> Vec<f32> {
    let channels = format.channels as usize;
    if channels == 1 {
        return samples.to_vec();
    }
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Restore a mono block to an interleaved output with the requested channel
/// count.  Every channel receives the same converted voice signal.
pub(crate) fn interleave(mono: &[f32], channels: u16) -> Vec<f32> {
    let channels = channels.max(1) as usize;
    let mut output = Vec::with_capacity(mono.len().saturating_mul(channels));
    for &sample in mono {
        output.extend(std::iter::repeat_n(sample, channels));
    }
    output
}

/// Crop or left-pad a converted waveform to exactly `frames` samples.
pub(crate) fn tail_with_length(samples: &[f32], frames: usize) -> Vec<f32> {
    if samples.len() >= frames {
        return samples[samples.len() - frames..].to_vec();
    }
    let mut output = vec![0.0; frames - samples.len()];
    output.extend_from_slice(samples);
    output
}

/// Apply the RVC RMS mix policy to a block after it has been converted back to
/// the public output sample rate.
pub(crate) fn apply_rms_mix(source: &[f32], output: &mut [f32], rate: f32) {
    if source.is_empty() || output.is_empty() || rate >= 1.0 {
        return;
    }
    let rate = rate.clamp(0.0, 1.0);
    let source_rms = rms(source).max(1e-6);
    let output_rms = rms(output).max(1e-6);
    let gain = (source_rms / output_rms).powf(1.0 - rate);
    for sample in output {
        *sample *= gain;
    }
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt()
}

/// A reusable fixed-size internal-rate window for TensorRT profiles.
///
/// The history contains only real input samples.  Until a complete window is
/// available, the left side is zero padded; once full, the oldest samples are
/// discarded in place.  The returned `Vec` is owned by the adapter and is
/// reused on the next call, so no device buffer is involved in this boundary.
pub(crate) struct StreamingWindow {
    target_samples: usize,
    history: Vec<f32>,
    scratch: Vec<f32>,
}

impl StreamingWindow {
    pub(crate) fn new(target_samples: usize) -> Result<Self, EngineError> {
        if target_samples == 0 {
            return Err(EngineError::InvalidConfig(
                "TensorRT window length must be non-zero".into(),
            ));
        }
        Ok(Self {
            target_samples,
            history: Vec::with_capacity(target_samples),
            scratch: Vec::with_capacity(target_samples),
        })
    }

    pub(crate) fn target_samples(&self) -> usize {
        self.target_samples
    }

    pub(crate) fn reset(&mut self) {
        self.history.clear();
        self.scratch.clear();
    }

    pub(crate) fn push(&mut self, samples: &[f32]) -> Result<&[f32], EngineError> {
        if samples.is_empty() {
            return Err(EngineError::InvalidInput(
                "resampled audio block is empty".into(),
            ));
        }
        if samples.len() > self.target_samples {
            return Err(EngineError::InvalidInput(format!(
                "audio block of {} internal samples exceeds TensorRT window {}",
                samples.len(),
                self.target_samples
            )));
        }
        self.history.extend_from_slice(samples);
        if self.history.len() > self.target_samples {
            let keep_from = self.history.len() - self.target_samples;
            self.history.drain(..keep_from);
        }
        self.scratch.clear();
        self.scratch
            .resize(self.target_samples.saturating_sub(self.history.len()), 0.0);
        self.scratch.extend_from_slice(&self.history);
        Ok(&self.scratch)
    }
}

/// Convert a mono public-rate block to the internal waveform domain.
pub(crate) fn to_internal_rate(mono: &[f32], sample_rate: u32) -> Vec<f32> {
    pipeline::resample_linear(mono, sample_rate, pipeline::HUBERT_SAMPLE_RATE)
}

/// Convert model-rate audio back to a public-rate block with exact duration.
pub(crate) fn from_model_rate(
    model_audio: &[f32],
    model_sample_rate: u32,
    output_format: AudioFormat,
    frames: usize,
) -> Vec<f32> {
    let converted =
        pipeline::resample_linear(model_audio, model_sample_rate, output_format.sample_rate);
    tail_with_length(&converted, frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_and_interleave_preserve_frame_count() {
        let format = AudioFormat {
            sample_rate: 48_000,
            channels: 2,
        };
        let mono = to_mono(format, &[1.0, 3.0, -1.0, 1.0]);
        assert_eq!(mono, vec![2.0, 0.0]);
        assert_eq!(interleave(&mono, 2), vec![2.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn streaming_window_zero_pads_then_keeps_latest_context() {
        let mut window = StreamingWindow::new(4).unwrap();
        assert_eq!(window.push(&[1.0, 2.0]).unwrap(), &[0.0, 0.0, 1.0, 2.0]);
        assert_eq!(window.push(&[3.0, 4.0]).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(window.push(&[5.0, 6.0]).unwrap(), &[3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn tail_crop_and_padding_are_deterministic() {
        assert_eq!(tail_with_length(&[1.0, 2.0, 3.0], 2), vec![2.0, 3.0]);
        assert_eq!(tail_with_length(&[1.0], 3), vec![0.0, 0.0, 1.0]);
    }
}
