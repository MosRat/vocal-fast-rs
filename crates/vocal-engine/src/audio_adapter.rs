//! Audio format and streaming-window adapters shared by engine backends.
//!
//! The neural backends consume mono 16 kHz windows, while the public engine
//! contract uses interleaved PCM in the caller's format.  Keeping these
//! conversions here prevents each backend from growing its own subtly
//! different crop, padding, and channel handling rules.

use std::collections::VecDeque;

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

/// Buffers the public callback stream until the fixed realtime inference
/// block is available, and retains generated samples until the next callback
/// can consume them.  Keeping this state independent of TensorRT makes the
/// boundary easy to exercise without a CUDA device and gives future streaming
/// backends the same callback contract.
#[derive(Default)]
pub(crate) struct RealtimeBufferState {
    pending_internal: Vec<f32>,
    pending_source: Vec<f32>,
    pending_output: VecDeque<f32>,
}

impl RealtimeBufferState {
    pub(crate) fn append_input(&mut self, internal: &[f32], source: &[f32]) {
        self.pending_internal.extend_from_slice(internal);
        self.pending_source.extend_from_slice(source);
    }

    /// Remove one complete internal-rate inference block.  The source-rate
    /// companion may be shorter by one rounded sample for uncommon rates; in
    /// that case the available source samples are returned and inference still
    /// advances on the exact internal block.
    pub(crate) fn take_input_chunk(
        &mut self,
        target_internal: usize,
        target_source: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        if target_internal == 0 || self.pending_internal.len() < target_internal {
            return None;
        }
        let internal = self.pending_internal.drain(..target_internal).collect();
        let source_len = target_source.min(self.pending_source.len());
        let source = self.pending_source.drain(..source_len).collect();
        Some((internal, source))
    }

    pub(crate) fn append_output(&mut self, output: &[f32]) {
        self.pending_output.extend(output.iter().copied());
    }

    /// Return exactly `frames` samples, zero-filling while the first inference
    /// block is still warming up or when a final partial input has not yet
    /// reached the fixed inference size.
    pub(crate) fn take_output(&mut self, frames: usize) -> Vec<f32> {
        let mut output = Vec::with_capacity(frames);
        for _ in 0..frames {
            output.push(self.pending_output.pop_front().unwrap_or(0.0));
        }
        output
    }

    #[cfg(test)]
    fn pending_internal_len(&self) -> usize {
        self.pending_internal.len()
    }

    #[cfg(test)]
    fn pending_source_len(&self) -> usize {
        self.pending_source.len()
    }

    #[cfg(test)]
    fn pending_output_len(&self) -> usize {
        self.pending_output.len()
    }

    pub(crate) fn reset(&mut self) {
        self.pending_internal.clear();
        self.pending_source.clear();
        self.pending_output.clear();
    }
}

/// Convert a mono public-rate block to the internal waveform domain.
pub(crate) fn to_internal_rate(mono: &[f32], sample_rate: u32) -> Vec<f32> {
    pipeline::resample_linear(mono, sample_rate, pipeline::HUBERT_SAMPLE_RATE)
}

/// Join the extended output returned by a realtime voice-conversion window.
///
/// The model window contains the current playback block plus a short
/// crossfade/SOLA lookahead. The joiner keeps the previous crossfade tail,
/// selects the best aligned candidate in the next block, and emits exactly
/// `block_frames` samples. This is the same state boundary used by the
/// Applio/WebUI realtime workers, expressed as a backend-neutral adapter.
pub(crate) struct StreamingOutputJoiner {
    sample_rate: u32,
    crossfade_samples: usize,
    sola_search_samples: usize,
    sola_buffer: Vec<f32>,
    initialized: bool,
}

impl StreamingOutputJoiner {
    pub(crate) fn new(
        sample_rate: u32,
        crossfade_samples_16k: usize,
        sola_search_samples_16k: usize,
    ) -> Self {
        let crossfade_samples = pipeline::resampled_len(crossfade_samples_16k, 16_000, sample_rate);
        let sola_search_samples =
            pipeline::resampled_len(sola_search_samples_16k, 16_000, sample_rate);
        Self {
            sample_rate,
            crossfade_samples,
            sola_search_samples,
            sola_buffer: vec![0.0; crossfade_samples],
            initialized: false,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.sola_buffer.fill(0.0);
        self.initialized = false;
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub(crate) fn required_samples(&self, block_frames: usize) -> usize {
        block_frames
            .saturating_add(self.crossfade_samples)
            .saturating_add(self.sola_search_samples)
    }

    pub(crate) fn process(&mut self, audio: &[f32], block_frames: usize) -> Vec<f32> {
        if block_frames == 0 {
            return Vec::new();
        }
        let required = self.required_samples(block_frames);
        let mut input = audio.to_vec();
        if input.len() < required {
            input.resize(required, 0.0);
        }

        let search_end = self
            .crossfade_samples
            .saturating_add(self.sola_search_samples)
            .min(input.len());
        let offset = if self.initialized {
            find_sola_offset(
                &self.sola_buffer,
                &input[..search_end],
                self.crossfade_samples,
                self.sola_search_samples,
            )
        } else {
            0
        };
        self.initialized = true;

        let required_after_offset = offset
            .saturating_add(block_frames)
            .saturating_add(self.crossfade_samples);
        if input.len() < required_after_offset {
            input.resize(required_after_offset, 0.0);
        }

        let current = &input[offset..];
        let overlap = self.crossfade_samples.min(block_frames);
        let mut output = current[..block_frames].to_vec();
        let fade_in = |index: usize| {
            let phase = if overlap <= 1 {
                1.0
            } else {
                index as f32 / (overlap - 1) as f32
            };
            (0.5 * std::f32::consts::PI * phase).sin().powi(2)
        };
        if overlap > 0 {
            for index in 0..overlap {
                let fade_in = fade_in(index);
                output[index] = self.sola_buffer[index] * (1.0 - fade_in) + output[index] * fade_in;
            }
        }
        self.sola_buffer
            .copy_from_slice(&current[block_frames..block_frames + self.crossfade_samples]);
        output
    }
}

fn find_sola_offset(
    previous: &[f32],
    candidates: &[f32],
    overlap: usize,
    max_offset: usize,
) -> usize {
    if overlap == 0 || candidates.len() < overlap {
        return 0;
    }
    let previous_energy = previous
        .iter()
        .take(overlap)
        .map(|sample| f64::from(*sample) * f64::from(*sample))
        .sum::<f64>();
    if previous_energy <= 1e-12 {
        return 0;
    }
    let max_offset = max_offset.min(candidates.len().saturating_sub(overlap));
    let mut best_offset = 0;
    let mut best_score = f64::NEG_INFINITY;
    for offset in 0..=max_offset {
        let candidate = &candidates[offset..offset + overlap];
        let mut numerator = 0.0;
        let mut candidate_energy = 0.0;
        for (&left, &right) in previous.iter().take(overlap).zip(candidate) {
            numerator += f64::from(left) * f64::from(right);
            candidate_energy += f64::from(right) * f64::from(right);
        }
        let score = numerator / (previous_energy * candidate_energy + 1e-8).sqrt();
        if score > best_score {
            best_score = score;
            best_offset = offset;
        }
    }
    best_offset
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
    fn realtime_buffers_wait_for_a_complete_fixed_chunk() {
        let mut buffers = RealtimeBufferState::default();
        for _ in 0..24 {
            buffers.append_input(&[1.0; 160], &[2.0; 480]);
            assert!(buffers.take_input_chunk(4_000, 12_000).is_none());
        }
        buffers.append_input(&[1.0; 160], &[2.0; 480]);
        let (internal, source) = buffers
            .take_input_chunk(4_000, 12_000)
            .expect("25 callbacks form one 4000-sample internal block");
        assert_eq!(internal.len(), 4_000);
        assert_eq!(source.len(), 12_000);
        assert_eq!(buffers.pending_internal_len(), 0);
        assert_eq!(buffers.pending_source_len(), 0);
    }

    #[test]
    fn realtime_buffers_keep_a_partial_tail_until_the_next_chunk() {
        let mut buffers = RealtimeBufferState::default();
        buffers.append_input(&[1.0; 3_200], &[2.0; 9_600]);
        assert!(buffers.take_input_chunk(4_000, 12_000).is_none());
        assert_eq!(buffers.pending_internal_len(), 3_200);
        buffers.append_input(&[1.0; 800], &[2.0; 2_400]);
        assert!(buffers.take_input_chunk(4_000, 12_000).is_some());
        assert_eq!(buffers.pending_internal_len(), 0);
    }

    #[test]
    fn realtime_buffers_zero_fill_callbacks_and_reset_all_state() {
        let mut buffers = RealtimeBufferState::default();
        buffers.append_output(&[1.0, 2.0]);
        assert_eq!(buffers.take_output(4), vec![1.0, 2.0, 0.0, 0.0]);
        assert_eq!(buffers.pending_output_len(), 0);
        buffers.append_input(&[1.0], &[2.0]);
        buffers.append_output(&[3.0]);
        buffers.reset();
        assert_eq!(buffers.pending_internal_len(), 0);
        assert_eq!(buffers.pending_source_len(), 0);
        assert_eq!(buffers.pending_output_len(), 0);
    }

    #[test]
    fn tail_crop_and_padding_are_deterministic() {
        assert_eq!(tail_with_length(&[1.0, 2.0, 3.0], 2), vec![2.0, 3.0]);
        assert_eq!(tail_with_length(&[1.0], 3), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn streaming_output_joiner_preserves_block_length() {
        let mut joiner = StreamingOutputJoiner::new(16_000, 800, 160);
        let block = 1_600usize;
        let required = joiner.required_samples(block);
        let first = vec![0.0; required];
        let second = vec![1.0; required];
        assert_eq!(joiner.process(&first, block).len(), block);
        assert_eq!(joiner.process(&second, block).len(), block);
    }

    #[test]
    fn streaming_output_joiner_allows_crossfade_to_be_disabled() {
        let mut joiner = StreamingOutputJoiner::new(16_000, 0, 0);
        assert_eq!(joiner.required_samples(4), 4);
        assert_eq!(
            joiner.process(&[1.0, 2.0, 3.0, 4.0], 4),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            joiner.process(&[5.0, 6.0, 7.0, 8.0], 4),
            vec![5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn sola_search_selects_the_normalized_correlation_peak() {
        let previous = [1.0, 0.0, 0.0, 0.0];
        let candidates = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        assert_eq!(find_sola_offset(&previous, &candidates, 4, 3), 3);
    }
}
