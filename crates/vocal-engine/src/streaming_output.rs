//! Time-aligned envelope matching and speech-safe overlap joining.
//!
//! A decoder tail starts before the newest input chunk. Envelope matching must
//! use that same source interval, before SOLA moves/crossfades either waveform.

use crate::pipeline;

/// Select the source interval represented by a cropped decoder waveform.
pub(crate) fn decoder_source_tail(
    window: &[f32],
    decoded_samples: usize,
    decoded_rate: u32,
) -> &[f32] {
    let frames = pipeline::resampled_len(decoded_samples, decoded_rate, 16_000);
    &window[window.len().saturating_sub(frames)..]
}

/// RVC/WebUI-style centered 40 ms RMS, sampled every 10 ms and interpolated.
/// Unlike a callback-wide gain, this preserves attacks/releases inside a block.
pub(crate) fn apply_realtime_rms_mix(
    source: &[f32],
    source_rate: u32,
    output: &mut [f32],
    output_rate: u32,
    rate: f32,
) {
    if rate >= 1.0 || source.is_empty() || output.is_empty() {
        return;
    }
    let source_rms = rms_envelope(source, source_rate);
    let output_rms = rms_envelope(output, output_rate);
    let hop = (output_rate / 100).max(1) as usize;
    let exponent = 1.0 - rate.clamp(0.0, 1.0);
    for (index, sample) in output.iter_mut().enumerate() {
        let frame = index / hop;
        let phase = (index % hop) as f32 / hop as f32;
        let interpolate = |envelope: &[f32]| {
            let left = envelope[frame.min(envelope.len() - 1)];
            let right = envelope[(frame + 1).min(envelope.len() - 1)];
            left + (right - left) * phase
        };
        // WebUI floors decoder RMS at 1e-3 to avoid amplifying near-silence.
        let gain = (interpolate(&source_rms) / interpolate(&output_rms).max(1e-3)).powf(exponent);
        *sample *= gain;
    }
}

fn rms_envelope(samples: &[f32], rate: u32) -> Vec<f32> {
    let hop = (rate / 100).max(1) as usize;
    let half = (rate / 50).max(1) as usize;
    let mut prefix = Vec::with_capacity(samples.len() + 1);
    prefix.push(0.0_f64);
    for &sample in samples {
        prefix.push(prefix.last().copied().unwrap_or_default() + f64::from(sample).powi(2));
    }
    (0..=samples.len().div_ceil(hop))
        .map(|frame| {
            let center = (frame * hop).min(samples.len());
            let start = center.saturating_sub(half);
            let end = center.saturating_add(half).min(samples.len());
            ((prefix[end] - prefix[start]) / (2 * half) as f64).sqrt() as f32
        })
        .collect()
}

pub(crate) struct StreamingOutputJoiner {
    sample_rate: u32,
    crossfade_samples: usize,
    sola_search_samples: usize,
    sola_buffer: Vec<f32>,
    initialized: bool,
    offset: usize,
}

impl StreamingOutputJoiner {
    pub(crate) fn new(sample_rate: u32, crossfade_16k: usize, search_16k: usize) -> Self {
        let crossfade_samples = pipeline::resampled_len(crossfade_16k, 16_000, sample_rate);
        Self {
            sample_rate,
            crossfade_samples,
            sola_search_samples: pipeline::resampled_len(search_16k, 16_000, sample_rate),
            sola_buffer: vec![0.0; crossfade_samples],
            initialized: false,
            offset: 0,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.sola_buffer.fill(0.0);
        self.initialized = false;
        self.offset = 0;
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub(crate) fn required_samples(&self, block: usize) -> usize {
        block
            .saturating_add(self.crossfade_samples)
            .saturating_add(self.sola_search_samples)
    }

    pub(crate) fn process(&mut self, audio: &[f32], block: usize) -> Vec<f32> {
        if block == 0 {
            return Vec::new();
        }
        let required = self.required_samples(block);
        let padded;
        let input = if audio.len() < required {
            padded = {
                let mut samples = audio.to_vec();
                samples.resize(required, 0.0);
                samples
            };
            padded.as_slice()
        } else {
            audio
        };
        let overlap = self.crossfade_samples.min(block);
        let search = &input[..self.crossfade_samples + self.sola_search_samples];
        // Weak/noisy matches have no useful phase information. Preserve the
        // last position instead of randomly skipping up to 10 ms of consonants.
        if self.initialized {
            let alignment =
                find_sola_alignment(&self.sola_buffer, search, self.sola_search_samples);
            if alignment.correlation >= 0.35 {
                self.offset = alignment.offset;
            }
        }
        let current = &input[self.offset..];
        let mut output = current[..block].to_vec();
        let previous_energy = energy(&self.sola_buffer[..overlap]);
        let current_energy = energy(&current[..overlap]);
        let ramp = (self.sample_rate / 200).max(1) as usize; // 5 ms anti-click handover
        let ramp = ramp.min(overlap);
        let onset = !self.initialized || previous_energy * 16.0 < current_energy;
        let release = !onset && current_energy * 16.0 < previous_energy;
        let correlation =
            correlation(&self.sola_buffer[..overlap], &current[..overlap]).clamp(0.0, 1.0) as f32;
        for index in 0..overlap {
            let (start, length) = if onset {
                (0, ramp)
            } else if release {
                (overlap - ramp, ramp)
            } else {
                (0, overlap)
            };
            let phase = if index < start {
                0.0
            } else if length <= 1 {
                1.0
            } else {
                ((index - start) as f32 / (length - 1) as f32).min(1.0)
            };
            let new_weight = (0.5 * std::f32::consts::PI * phase).sin().powi(2);
            let old_weight = 1.0 - new_weight;
            // Correlated voiced overlaps keep unity gain. Uncorrelated breath
            // and consonant noise use equal power, avoiding the usual level dip.
            let power = old_weight * old_weight
                + new_weight * new_weight
                + 2.0 * correlation * old_weight * new_weight;
            output[index] = (self.sola_buffer[index] * old_weight + current[index] * new_weight)
                / power.sqrt().max(std::f32::consts::FRAC_1_SQRT_2);
        }
        self.sola_buffer
            .copy_from_slice(&current[block..block + self.crossfade_samples]);
        self.initialized = true;
        output
    }
}

fn energy(samples: &[f32]) -> f64 {
    samples
        .iter()
        .map(|&sample| f64::from(sample).powi(2))
        .sum()
}

fn correlation(left: &[f32], right: &[f32]) -> f64 {
    let denominator = (energy(left) * energy(right)).sqrt();
    if denominator <= 1e-12 {
        return 0.0;
    }
    left.iter()
        .zip(right)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum::<f64>()
        / denominator
}

struct Alignment {
    offset: usize,
    correlation: f64,
}

fn find_sola_alignment(previous: &[f32], candidates: &[f32], search: usize) -> Alignment {
    let overlap = previous.len();
    let mut result = Alignment {
        offset: 0,
        correlation: 0.0,
    };
    // Below -80 dBFS, decoder residue cannot establish a reliable alignment.
    let previous_energy = energy(previous);
    if overlap == 0 || previous_energy < overlap as f64 * 1e-8 || candidates.len() < overlap {
        return result;
    }
    for offset in 0..=search.min(candidates.len() - overlap) {
        let candidate = &candidates[offset..offset + overlap];
        let mut candidate_energy = 0.0;
        let mut numerator = 0.0;
        for (&a, &b) in previous.iter().zip(candidate) {
            candidate_energy += f64::from(b).powi(2);
            numerator += f64::from(a) * f64::from(b);
        }
        if candidate_energy < overlap as f64 * 1e-8 {
            continue;
        }
        let score = numerator / (previous_energy * candidate_energy).sqrt();
        if score > result.correlation {
            result = Alignment {
                offset,
                correlation: score,
            };
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onset_after_silence_keeps_weak_consonant() {
        let mut joiner = StreamingOutputJoiner::new(16_000, 800, 160);
        joiner.process(&vec![0.0; 4960], 4000);
        let mut audio = vec![0.0; 4960];
        audio[160..320].fill(0.01); // a short 10 ms consonant after silence
        let output = joiner.process(&audio, 4000);
        assert_eq!(&output[160..320], &audio[160..320]);
    }

    #[test]
    fn silent_decoder_revision_preserves_previous_release() {
        let mut joiner = StreamingOutputJoiner::new(16_000, 800, 160);
        let mut audio = vec![0.0; 4960];
        audio[4000..4400].fill(0.1);
        joiner.process(&audio, 4000);
        let output = joiner.process(&vec![0.0; 4960], 4000);
        assert_eq!(&output[..400], &audio[4000..4400]);
        assert!(output[800..].iter().all(|&sample| sample == 0.0));
    }

    #[test]
    fn correlated_overlap_keeps_unity_gain_and_length() {
        let mut joiner = StreamingOutputJoiner::new(16_000, 800, 160);
        joiner.process(&vec![0.2; 4960], 4000);
        let output = joiner.process(&vec![0.2; 4960], 4000);
        assert_eq!(output.len(), 4000);
        assert!(output.iter().all(|&sample| (sample - 0.2).abs() < 1e-6));
    }

    #[test]
    fn uncorrelated_overlap_preserves_noise_power() {
        let mut joiner = StreamingOutputJoiner::new(16_000, 800, 0);
        let mut previous = vec![0.0; 4800];
        for (i, sample) in previous[4000..].iter_mut().enumerate() {
            *sample = if i % 2 == 0 { 0.1 } else { -0.1 };
        }
        joiner.process(&previous, 4000);
        let current: Vec<_> = (0..4800)
            .map(|i| if i % 4 < 2 { 0.1 } else { -0.1 })
            .collect();
        let output = joiner.process(&current, 4000);
        let rms = (energy(&output[..800]) / 800.0).sqrt();
        assert!((rms - 0.1).abs() < 0.001, "noise RMS: {rms}");
    }

    #[test]
    fn reset_clears_alignment_and_overlap() {
        let mut joiner = StreamingOutputJoiner::new(16_000, 800, 160);
        joiner.process(&vec![0.2; 4960], 4000);
        joiner.reset();
        assert!(
            joiner
                .process(&vec![0.0; 4960], 4000)
                .iter()
                .all(|&s| s == 0.0)
        );
        assert_eq!(joiner.offset, 0);
    }

    #[test]
    fn an_unreliable_match_keeps_the_existing_time_position() {
        let mut joiner = StreamingOutputJoiner::new(16_000, 800, 160);
        joiner.initialized = true;
        joiner.offset = 73;
        joiner.sola_buffer.fill(1e-6);
        let mut input = vec![0.0; 4960];
        input[73 + 160..73 + 320].fill(0.01);
        let output = joiner.process(&input, 4000);
        assert_eq!(joiner.offset, 73);
        assert_eq!(&output[160..320], &input[233..393]);
    }

    #[test]
    fn onset_and_release_work_at_public_sample_rates() {
        for rate in [16_000, 44_100, 48_000, 96_000] {
            let mut joiner = StreamingOutputJoiner::new(rate, 800, 160);
            let block = pipeline::resampled_len(4000, 16000, rate);
            let start = (rate / 100) as usize;
            let len = joiner.required_samples(block);
            let mut input = vec![0.0; len];
            input[start..2 * start].fill(0.01);
            input[block + start..block + 2 * start].fill(0.01);
            let onset = joiner.process(&input, block);
            assert_eq!(onset.len(), block);
            assert_eq!(&onset[start..2 * start], &input[start..2 * start]);
            let release = joiner.process(&vec![0.0; len], block);
            assert_eq!(
                &release[start..2 * start],
                &input[block + start..block + 2 * start]
            );
        }
    }

    #[test]
    fn disabled_overlap_and_empty_block() {
        let mut joiner = StreamingOutputJoiner::new(16_000, 0, 0);
        assert_eq!(joiner.process(&[1., 2., 3., 4.], 4), [1., 2., 3., 4.]);
        assert!(joiner.process(&[], 0).is_empty());
    }

    #[test]
    fn correlation_search_requires_a_real_signal() {
        assert_eq!(
            find_sola_alignment(&[1., 0., 0., 0.], &[0., 0., 0., 1., 0., 0., 0.], 3).offset,
            3
        );
        assert_eq!(
            find_sola_alignment(&[1e-6; 4], &[1e-6; 7], 3).correlation,
            0.0
        );
    }

    #[test]
    fn envelope_uses_decoder_interval_and_retains_tail_inside_silent_new_chunk() {
        let mut window = vec![0.0; 44960];
        window[40000..40960].fill(0.1); // voice ends before the newest chunk
        let source = decoder_source_tail(&window, 12400, 40000);
        assert_eq!(source.len(), 4960);
        let mut output = vec![0.2; 14880];
        apply_realtime_rms_mix(source, 16000, &mut output, 48000, 0.0);
        assert!(
            output[1000] > 0.05,
            "old voiced tail must survive a new silent chunk"
        );
        assert!(output[6000..].iter().all(|&s| s == 0.0));
        let jump = output
            .windows(2)
            .map(|s| (s[1] - s[0]).abs())
            .fold(0., f32::max);
        assert!(
            jump < 0.001,
            "envelope gain must interpolate continuously: {jump}"
        );
    }

    #[test]
    fn synth_volume_policy_is_a_noop() {
        let mut output = vec![0.2; 100];
        apply_realtime_rms_mix(&[0.; 100], 16000, &mut output, 16000, 1.0);
        assert_eq!(output, vec![0.2; 100]);
    }

    #[test]
    #[ignore = "requires VOCAL_DSP_TRACE_DIR from the DirectML streaming validator"]
    fn replay_real_decoder_tails() {
        let dir = std::path::PathBuf::from(std::env::var_os("VOCAL_DSP_TRACE_DIR").unwrap());
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("trace.json")).unwrap()).unwrap();
        let rate = metadata["sample_rate"].as_u64().unwrap() as u32;
        let advance = metadata["advance_samples"].as_u64().unwrap() as usize;
        let chunks = metadata["chunks"].as_u64().unwrap() as usize;
        let block = pipeline::resampled_len(advance, 16000, rate);
        let mut joiner = StreamingOutputJoiner::new(rate, 800, 160);
        let mut output = Vec::new();
        for i in 0..chunks {
            let bytes = std::fs::read(dir.join(format!("{i:04}/audio.bin"))).unwrap();
            let audio: Vec<_> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            if i < 13 {
                output.extend(std::iter::repeat_n(0.0, block));
            } else {
                output.extend(joiner.process(&audio, block));
            }
        }
        assert!(output.iter().all(|s| s.is_finite()));
        vocal_audio::write_wav_pcm16(dir.join("../trace-after.wav"), rate, &output).unwrap();
    }
}
