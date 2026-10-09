//! Applio offline preprocessing, segment selection and native-rate output.
use std::time::Instant;

use serde::{Deserialize, Serialize};
use vocal_types::ConversionConfig;

use crate::{BackendKind, EngineError};

const RATE: usize = 16_000;
const HOP: usize = 160;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AlignedOfflineConfig {
    pub pad_seconds: usize,
    pub query_seconds: usize,
    pub center_seconds: usize,
    pub max_seconds: usize,
    pub conversion: ConversionConfig,
}
impl Default for AlignedOfflineConfig {
    fn default() -> Self {
        Self {
            pad_seconds: 1,
            query_seconds: 6,
            center_seconds: 38,
            max_seconds: 41,
            conversion: ConversionConfig {
                protect: 0.5,
                ..Default::default()
            },
        }
    }
}

pub struct OfflineSegmentOutput {
    /// Full decoder output at the model rate, including reflected context.
    pub audio: Vec<f32>,
    pub preparation_ms: f64,
    pub inference_ms: f64,
}

/// Offline inference accepts 16 kHz model input and returns native-rate audio.
/// It has no callback history, startup mute, crossfade or realtime RMS gain.
pub trait OfflineVoicePipeline {
    fn backend_kind(&self) -> BackendKind;
    fn output_sample_rate(&self) -> u32;
    /// Segmented Applio conversion needs F0 extracted over the entire file
    /// before slicing. Backends must opt in only when they preserve that state.
    fn supports_global_f0_segments(&self) -> bool {
        false
    }
    fn infer_segment(
        &mut self,
        waveform: &[f32],
        sequence: usize,
        conversion: ConversionConfig,
    ) -> Result<OfflineSegmentOutput, EngineError>;
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AlignedOfflineReport {
    pub backend: BackendKind,
    pub input_samples_16k: usize,
    pub output_samples: usize,
    pub output_sample_rate: u32,
    pub segment_samples_16k: Vec<usize>,
    pub preparation_ms: f64,
    pub inference_ms: f64,
    /// DSP plus inference; graph preparation is reported separately.
    pub processing_ms: f64,
    pub elapsed_ms_including_preparation: f64,
}

pub fn run_aligned_offline<P: OfflineVoicePipeline + ?Sized>(
    pipeline: &mut P,
    input: &[f32],
    config: &AlignedOfflineConfig,
) -> Result<(Vec<f32>, AlignedOfflineReport), EngineError> {
    config
        .conversion
        .validate()
        .map_err(|e| EngineError::InvalidConfig(e.to_string()))?;
    if input.len() < 2 || input.iter().any(|v| !v.is_finite()) {
        return Err(EngineError::InvalidInput(
            "offline input must contain finite audio samples".into(),
        ));
    }
    if config.pad_seconds == 0
        || config.pad_seconds > 5
        || config.query_seconds == 0
        || config.query_seconds >= config.center_seconds
        || config.center_seconds >= config.max_seconds
        || config.max_seconds > 45
        || config.center_seconds + config.query_seconds + 2 * config.pad_seconds > 49
        || pipeline.output_sample_rate() == 0
    {
        return Err(EngineError::InvalidConfig(
            "invalid offline segment geometry or output rate".into(),
        ));
    }
    if input.len() + HOP > config.max_seconds * RATE && !pipeline.supports_global_f0_segments() {
        return Err(EngineError::InvalidConfig(
            "segmented offline conversion requires global file F0; this backend currently supports one Applio segment (less than 41 seconds at the default settings)".into(),
        ));
    }
    let started = Instant::now();
    // The normal Applio file entry limits input peaks before its pipeline.
    let peak = input.iter().fold(0f32, |a, x| a.max(x.abs()));
    let scale = if peak > 0.95 { 0.95 / peak } else { 1. };
    let normalized: Vec<_> = input.iter().map(|x| x * scale).collect();
    let filtered = highpass(&normalized);
    let cuts = segment_cuts(&filtered, config);
    let pad = config.pad_seconds * RATE;
    let reflected = reflect_pad(&filtered, pad);
    let output_rate = pipeline.output_sample_rate();
    let trim = config.pad_seconds * output_rate as usize;
    let mut output = Vec::new();
    let mut segment_samples = Vec::new();
    let mut preparation_ms = 0.;
    let mut inference_ms = 0.;
    let mut begin = 0;
    for (sequence, cut) in cuts
        .into_iter()
        .map(Some)
        .chain(std::iter::once(None))
        .enumerate()
    {
        let end = cut
            .map_or(reflected.len(), |t| t + 2 * pad + HOP)
            .min(reflected.len());
        let wave: Vec<_> = reflected[begin..end].iter().map(|x| *x as f32).collect();
        tracing::info!(sequence, samples = wave.len(), "converting offline segment");
        let result = pipeline.infer_segment(&wave, sequence, config.conversion)?;
        if result.audio.len() <= 2 * trim || result.audio.iter().any(|v| !v.is_finite()) {
            return Err(EngineError::BackendUnavailable(
                "offline decoder output is invalid or shorter than reflected context".into(),
            ));
        }
        segment_samples.push(wave.len());
        preparation_ms += result.preparation_ms;
        inference_ms += result.inference_ms;
        output.extend_from_slice(&result.audio[trim..result.audio.len() - trim]);
        if let Some(cut) = cut {
            begin = cut;
        }
    }
    if config.conversion.rms_mix_rate < 1. {
        mix_envelope(
            &filtered,
            &mut output,
            output_rate as usize,
            config.conversion.rms_mix_rate,
        );
    }
    let peak = output.iter().fold(0f32, |a, x| a.max(x.abs()));
    if peak > 0.99 {
        for sample in &mut output {
            *sample *= 0.99 / peak;
        }
    }
    let elapsed = started.elapsed().as_secs_f64() * 1000.;
    let report = AlignedOfflineReport {
        backend: pipeline.backend_kind(),
        input_samples_16k: input.len(),
        output_samples: output.len(),
        output_sample_rate: output_rate,
        segment_samples_16k: segment_samples,
        preparation_ms,
        inference_ms,
        processing_ms: (elapsed - preparation_ms).max(0.),
        elapsed_ms_including_preparation: elapsed,
    };
    Ok((output, report))
}

fn reflect_pad(input: &[f64], padding: usize) -> Vec<f64> {
    let period = 2 * (input.len() - 1) as isize;
    (-(padding as isize)..input.len() as isize + padding as isize)
        .map(|i| {
            let i = i.rem_euclid(period) as usize;
            input[if i < input.len() {
                i
            } else {
                period as usize - i
            }]
        })
        .collect()
}

fn segment_cuts(audio: &[f64], config: &AlignedOfflineConfig) -> Vec<usize> {
    if audio.len() + HOP <= config.max_seconds * RATE {
        return Vec::new();
    }
    let padded = reflect_pad(audio, HOP / 2);
    let query = config.query_seconds * RATE;
    (config.center_seconds * RATE..audio.len())
        .step_by(config.center_seconds * RATE)
        .map(|center| {
            let mut best = center - query;
            let mut minimum = f64::INFINITY;
            for i in center - query..(center + query).min(audio.len()) {
                let sum = padded[i..i + HOP].iter().sum::<f64>().abs();
                if sum < minimum {
                    minimum = sum;
                    best = i;
                }
            }
            best / HOP * HOP
        })
        .collect()
}

// scipy.signal.butter(5, 48, btype="high", fs=16000), direct-form-II
// transposed with scipy lfilter_zi and filtfilt's odd extension (18 samples).
// Keep preprocessing in f64, casting only the model input to f32.
fn highpass(input: &[f32]) -> Vec<f64> {
    const B: [f64; 6] = [
        0.9699606451838447,
        -4.849803225919223,
        9.699606451838447,
        -9.699606451838447,
        4.849803225919223,
        -0.9699606451838447,
    ];
    const A: [f64; 6] = [
        1.,
        -4.939001819168364,
        9.757863526739543,
        -9.639544849413458,
        4.761506797356209,
        -0.9408236532054606,
    ];
    const ZI: [f64; 5] = [
        -0.9699607413707367,
        3.879842959615721,
        -5.81976443080129,
        3.879842948235015,
        -0.9699607356787477,
    ];
    fn filter(audio: &mut [f64], initial: f64) {
        let mut state = ZI.map(|z| z * initial);
        for x in audio {
            let source = *x;
            let y = B[0] * source + state[0];
            for i in 0..4 {
                state[i] = B[i + 1] * source + state[i + 1] - A[i + 1] * y;
            }
            state[4] = B[5] * source - A[5] * y;
            *x = y;
        }
    }
    let edge = 18.min(input.len() - 1);
    let mut extended = Vec::with_capacity(input.len() + 2 * edge);
    extended.extend(
        (1..=edge)
            .rev()
            .map(|i| 2. * input[0] as f64 - input[i] as f64),
    );
    extended.extend(input.iter().map(|x| *x as f64));
    extended.extend(
        (1..=edge).map(|i| 2. * input[input.len() - 1] as f64 - input[input.len() - 1 - i] as f64),
    );
    let initial = extended[0];
    filter(&mut extended, initial);
    extended.reverse();
    let initial = extended[0];
    filter(&mut extended, initial);
    extended.reverse();
    extended[edge..edge + input.len()].to_vec()
}

fn mix_envelope(source: &[f64], output: &mut [f32], rate: usize, mix: f32) {
    fn envelope(audio: &[f64], rate: usize) -> Vec<f32> {
        (0..=audio.len() / (rate / 2))
            .map(|i| {
                let center = i * (rate / 2);
                let start = center.saturating_sub(rate / 2);
                let end = (center + rate / 2).min(audio.len());
                (audio[start..end].iter().map(|x| x * x).sum::<f64>() / rate as f64).sqrt() as f32
            })
            .collect()
    }
    fn at(values: &[f32], i: usize, size: usize) -> f32 {
        let position = ((i as f64 + 0.5) * values.len() as f64 / size as f64 - 0.5).max(0.);
        let left = (position as usize).min(values.len() - 1);
        let right = (left + 1).min(values.len() - 1);
        values[left] + (values[right] - values[left]) * (position - left as f64) as f32
    }
    let src = envelope(source, RATE);
    let dst = envelope(&output.iter().map(|x| *x as f64).collect::<Vec<_>>(), rate);
    let len = output.len();
    for (i, x) in output.iter_mut().enumerate() {
        *x *= at(&src, i, len).powf(1. - mix) * at(&dst, i, len).max(1e-6).powf(mix - 1.);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct NativeRate {
        windows: Vec<usize>,
    }
    impl OfflineVoicePipeline for NativeRate {
        fn backend_kind(&self) -> BackendKind {
            BackendKind::DirectMl
        }
        fn output_sample_rate(&self) -> u32 {
            40_000
        }
        fn supports_global_f0_segments(&self) -> bool {
            true
        }
        fn infer_segment(
            &mut self,
            wave: &[f32],
            _: usize,
            _: ConversionConfig,
        ) -> Result<OfflineSegmentOutput, EngineError> {
            self.windows.push(wave.len());
            let frames = (wave.len() / 160).min(((wave.len() - 400) / 320 + 1) * 2);
            Ok(OfflineSegmentOutput {
                audio: vec![0.25; frames * 400],
                preparation_ms: 0.,
                inference_ms: 0.,
            })
        }
    }
    #[test]
    fn full_segment_preserves_native_rate_and_synth_loudness_without_startup_mute() {
        let mut pipeline = NativeRate { windows: vec![] };
        let (output, report) =
            run_aligned_offline(&mut pipeline, &vec![0.1; 433_600], &Default::default()).unwrap();
        assert_eq!(pipeline.windows, [465_600]);
        assert_eq!(output.len(), 1_083_200);
        assert!(output.iter().all(|x| *x == 0.25));
        assert_eq!(report.output_sample_rate, 40_000);
        assert_eq!(report.segment_samples_16k, [465_600]);
    }
    #[test]
    fn reflected_context_handles_short_sources_and_highpass_rejects_dc() {
        assert_eq!(
            reflect_pad(&[1., 2., 3.], 4),
            [1., 2., 3., 2., 1., 2., 3., 2., 1., 2., 3.]
        );
        assert!(highpass(&vec![0.2; 16000]).iter().all(|x| x.abs() < 1e-4));
    }
    #[test]
    fn long_files_split_at_low_energy_and_keep_hop_aligned_context() {
        let cuts = segment_cuts(&vec![0.; 50 * RATE], &Default::default());
        assert_eq!(cuts, [32 * RATE]);
        let mut pipeline = NativeRate { windows: vec![] };
        let (_, report) =
            run_aligned_offline(&mut pipeline, &vec![0.; 50 * RATE], &Default::default()).unwrap();
        assert_eq!(report.segment_samples_16k, [34 * RATE + HOP, 20 * RATE]);
    }
}
