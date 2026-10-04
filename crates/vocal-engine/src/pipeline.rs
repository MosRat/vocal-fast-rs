//! Rust implementation of the data flow used by the reference RVC pipeline.
//!
//! The expensive neural stages are traits.  The CPU fixture implements those
//! traits deterministically, while a TensorRT RTX adapter can feed the same
//! `FeatureMatrix`, `PitchTrack`, and `SynthesisInput` tensors to an engine.

use std::f32::consts::PI;
use std::time::Instant;

use crate::{EngineError, config::validate_conversion};
use vocal_types::ConversionConfig;

pub const HUBERT_SAMPLE_RATE: u32 = 16_000;
pub const PITCH_HOP_SAMPLES: usize = 160;
pub const HUBERT_HOP_SAMPLES: usize = 320;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum F0Method {
    ZeroCrossing,
    Rmvpe,
    Fcpe,
    Disabled,
}

#[derive(Clone, Debug)]
pub struct RvcPipelineConfig {
    pub target_sample_rate: u32,
    pub output_sample_rate: u32,
    pub feature_dims: usize,
    pub feature_hop_samples: usize,
    pub pad_samples: usize,
    pub model_uses_f0: bool,
    pub f0_method: F0Method,
    pub speaker_id: u32,
}

impl Default for RvcPipelineConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 32_000,
            output_sample_rate: 16_000,
            feature_dims: 1,
            feature_hop_samples: HUBERT_HOP_SAMPLES,
            pad_samples: 160,
            model_uses_f0: true,
            f0_method: F0Method::ZeroCrossing,
            speaker_id: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FeatureMatrix {
    pub frames: usize,
    pub dims: usize,
    pub data: Vec<f32>,
}

impl FeatureMatrix {
    pub fn new(frames: usize, dims: usize, data: Vec<f32>) -> Result<Self, EngineError> {
        if dims == 0 || data.len() != frames.saturating_mul(dims) {
            return Err(EngineError::InvalidInput(format!(
                "feature matrix shape ({frames}, {dims}) does not match {} values",
                data.len()
            )));
        }
        Ok(Self { frames, dims, data })
    }

    pub fn row(&self, frame: usize) -> &[f32] {
        let start = frame * self.dims;
        &self.data[start..start + self.dims]
    }

    pub fn interpolate_2x(&self) -> Self {
        if self.frames == 0 {
            return self.clone();
        }
        let frames = self.frames * 2;
        let mut data = vec![0.0; frames * self.dims];
        for frame in 0..frames {
            let source = frame as f32 / 2.0;
            let left = source.floor() as usize;
            let right = (left + 1).min(self.frames - 1);
            let weight = source - left as f32;
            for dim in 0..self.dims {
                data[frame * self.dims + dim] = self.data[left * self.dims + dim] * (1.0 - weight)
                    + self.data[right * self.dims + dim] * weight;
            }
        }
        Self {
            frames,
            dims: self.dims,
            data,
        }
    }

    fn blend_rows(&mut self, other: &Self, ratio: f32) -> Result<(), EngineError> {
        if self.frames != other.frames || self.dims != other.dims {
            return Err(EngineError::InvalidInput(
                "feature blend shape mismatch".into(),
            ));
        }
        let ratio = ratio.clamp(0.0, 1.0);
        for (left, right) in self.data.iter_mut().zip(&other.data) {
            *left = *left * (1.0 - ratio) + *right * ratio;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct PitchTrack {
    pub coarse: Vec<i32>,
    pub hz: Vec<f32>,
}

pub trait FeatureEncoder: Send {
    fn encode(&mut self, audio: &[f32], sample_rate: u32) -> Result<FeatureMatrix, EngineError>;
}

pub trait PitchExtractor: Send {
    fn extract(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        frame_count: usize,
        pitch_semitones: f32,
        method: F0Method,
    ) -> Result<PitchTrack, EngineError>;
}

pub trait FeatureIndex: Send {
    fn blend(&mut self, features: &mut FeatureMatrix, index_ratio: f32) -> Result<(), EngineError>;
}

pub struct SynthesisInput<'a> {
    pub features: &'a FeatureMatrix,
    pub pitch: Option<&'a PitchTrack>,
    pub source_audio: &'a [f32],
    pub source_sample_rate: u32,
    pub target_sample_rate: u32,
    pub output_samples: usize,
    pub speaker_id: u32,
}

pub trait Synthesizer: Send {
    fn synthesize(&mut self, input: SynthesisInput<'_>) -> Result<Vec<f32>, EngineError>;
}

/// Tensor contract for the native TensorRT RTX binding.
///
/// `features` is the HuBERT/ContentVec sequence after index fusion and 2x
/// interpolation. `pitch_coarse` and `pitch_hz` are omitted for no-F0 models.
/// The executor owns CUDA buffers and should enqueue one inference without
/// synchronizing the audio callback thread.
pub struct TensorRtRtxInput<'a> {
    pub features: &'a FeatureMatrix,
    pub pitch: Option<&'a PitchTrack>,
    pub p_len: usize,
    pub speaker_id: u32,
    pub output_samples: usize,
}

pub trait TensorRtRtxExecutor: Send {
    fn enqueue(&mut self, input: TensorRtRtxInput<'_>) -> Result<Vec<f32>, EngineError>;
}

pub struct TensorRtRtxSynthesizer<E> {
    executor: E,
}

impl<E> TensorRtRtxSynthesizer<E> {
    pub fn new(executor: E) -> Self {
        Self { executor }
    }
}

impl<E: TensorRtRtxExecutor> Synthesizer for TensorRtRtxSynthesizer<E> {
    fn synthesize(&mut self, input: SynthesisInput<'_>) -> Result<Vec<f32>, EngineError> {
        self.executor.enqueue(TensorRtRtxInput {
            features: input.features,
            pitch: input.pitch,
            p_len: input.features.frames,
            speaker_id: input.speaker_id,
            output_samples: input.output_samples,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PipelineTimings {
    pub preprocessing_micros: u64,
    pub feature_micros: u64,
    pub index_micros: u64,
    pub pitch_micros: u64,
    pub synthesis_micros: u64,
    pub postprocessing_micros: u64,
}

#[derive(Clone, Debug)]
pub struct PipelineOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub feature_frames: usize,
    pub pitch_frames: usize,
    pub timings: PipelineTimings,
}

pub struct RvcPipeline {
    config: RvcPipelineConfig,
    encoder: Box<dyn FeatureEncoder>,
    pitch: Box<dyn PitchExtractor>,
    index: Option<Box<dyn FeatureIndex>>,
    synthesizer: Box<dyn Synthesizer>,
}

impl RvcPipeline {
    pub fn new(
        config: RvcPipelineConfig,
        encoder: Box<dyn FeatureEncoder>,
        pitch: Box<dyn PitchExtractor>,
        index: Option<Box<dyn FeatureIndex>>,
        synthesizer: Box<dyn Synthesizer>,
    ) -> Result<Self, EngineError> {
        config.validate()?;
        Ok(Self {
            config,
            encoder,
            pitch,
            index,
            synthesizer,
        })
    }

    pub fn config(&self) -> &RvcPipelineConfig {
        &self.config
    }

    pub fn set_index(&mut self, index: Option<Box<dyn FeatureIndex>>) {
        self.index = index;
    }

    pub fn run(
        &mut self,
        input: &[f32],
        input_sample_rate: u32,
        conversion: ConversionConfig,
    ) -> Result<PipelineOutput, EngineError> {
        self.config.validate()?;
        validate_conversion(conversion)?;
        if input.is_empty()
            || input_sample_rate == 0
            || input.iter().any(|sample| !sample.is_finite())
        {
            return Err(EngineError::InvalidInput(
                "input audio must be non-empty, finite and have a sample rate".into(),
            ));
        }
        let mut timings = PipelineTimings::default();
        let started = Instant::now();
        let source = resample_linear(input, input_sample_rate, HUBERT_SAMPLE_RATE);
        let source = high_pass_filter(&source, HUBERT_SAMPLE_RATE);
        let padded = reflect_pad(&source, self.config.pad_samples);
        timings.preprocessing_micros = started.elapsed().as_micros() as u64;

        let started = Instant::now();
        let mut features = self.encoder.encode(&padded, HUBERT_SAMPLE_RATE)?;
        timings.feature_micros = started.elapsed().as_micros() as u64;

        let started = Instant::now();
        if conversion.index_ratio > 0.0
            && let Some(index) = self.index.as_mut()
        {
            index.blend(&mut features, conversion.index_ratio)?;
        }
        timings.index_micros = started.elapsed().as_micros() as u64;

        let started = Instant::now();
        let mut pitch = if self.config.model_uses_f0 && self.config.f0_method != F0Method::Disabled
        {
            Some(self.pitch.extract(
                &padded,
                HUBERT_SAMPLE_RATE,
                padded.len().div_ceil(PITCH_HOP_SAMPLES),
                conversion.pitch_semitones,
                self.config.f0_method,
            )?)
        } else {
            None
        };
        timings.pitch_micros = started.elapsed().as_micros() as u64;

        let features_before_interpolation = if pitch.is_some() && conversion.protect < 0.5 {
            Some(features.clone())
        } else {
            None
        };
        let mut features = features.interpolate_2x();
        if let Some(track) = pitch.as_mut() {
            track.coarse.truncate(features.frames);
            track.hz.truncate(features.frames);
        }
        if let (Some(before), Some(pitch)) = (features_before_interpolation, pitch.as_ref()) {
            let before = before.interpolate_2x();
            let limit = features.frames.min(pitch.hz.len());
            let protect = conversion.protect.clamp(0.0, 1.0);
            for frame in 0..limit {
                let voiced = if pitch.hz[frame] > 0.0 { 1.0 } else { protect };
                let start = frame * features.dims;
                for dim in 0..features.dims {
                    features.data[start + dim] = features.data[start + dim] * voiced
                        + before.data[start + dim] * (1.0 - voiced);
                }
            }
        }

        let started = Instant::now();
        let output_samples = resampled_len(
            padded.len(),
            HUBERT_SAMPLE_RATE,
            self.config.target_sample_rate,
        );
        let mut synthesized = self.synthesizer.synthesize(SynthesisInput {
            features: &features,
            pitch: pitch.as_ref(),
            source_audio: &padded,
            source_sample_rate: HUBERT_SAMPLE_RATE,
            target_sample_rate: self.config.target_sample_rate,
            output_samples,
            speaker_id: self.config.speaker_id,
        })?;
        timings.synthesis_micros = started.elapsed().as_micros() as u64;

        let started = Instant::now();
        synthesized.resize(output_samples, 0.0);
        let target_pad = resampled_len(
            self.config.pad_samples,
            HUBERT_SAMPLE_RATE,
            self.config.target_sample_rate,
        );
        if synthesized.len() > target_pad.saturating_mul(2) {
            synthesized = synthesized[target_pad..synthesized.len() - target_pad].to_vec();
        }
        if conversion.rms_mix_rate < 1.0 {
            let source_at_target =
                resample_linear(&source, HUBERT_SAMPLE_RATE, self.config.target_sample_rate);
            change_rms(&source_at_target, &mut synthesized, conversion.rms_mix_rate);
        }
        let output = resample_linear(
            &synthesized,
            self.config.target_sample_rate,
            self.config.output_sample_rate,
        );
        timings.postprocessing_micros = started.elapsed().as_micros() as u64;
        Ok(PipelineOutput {
            samples: output,
            sample_rate: self.config.output_sample_rate,
            feature_frames: features.frames,
            pitch_frames: pitch.as_ref().map_or(0, |track| track.hz.len()),
            timings,
        })
    }
}

#[derive(Default)]
pub struct FrameEnergyEncoder {
    pub feature_dims: usize,
    pub hop_samples: usize,
}

impl FrameEnergyEncoder {
    pub fn new(feature_dims: usize, hop_samples: usize) -> Self {
        Self {
            feature_dims,
            hop_samples,
        }
    }
}

impl FeatureEncoder for FrameEnergyEncoder {
    fn encode(&mut self, audio: &[f32], _sample_rate: u32) -> Result<FeatureMatrix, EngineError> {
        if self.feature_dims == 0 || self.hop_samples == 0 {
            return Err(EngineError::InvalidInput(
                "feature encoder has invalid dimensions".into(),
            ));
        }
        let frames = audio.len().div_ceil(self.hop_samples).max(1);
        let mut data = vec![0.0; frames * self.feature_dims];
        for frame in 0..frames {
            let start = frame * self.hop_samples;
            let end = (start + self.hop_samples).min(audio.len());
            let energy = if start < end {
                (audio[start..end]
                    .iter()
                    .map(|sample| sample * sample)
                    .sum::<f32>()
                    / (end - start) as f32)
                    .sqrt()
            } else {
                0.0
            };
            for dim in 0..self.feature_dims {
                data[frame * self.feature_dims + dim] =
                    energy * (1.0 + dim as f32 / self.feature_dims as f32);
            }
        }
        FeatureMatrix::new(frames, self.feature_dims, data)
    }
}

#[derive(Default)]
pub struct ZeroCrossingPitchExtractor;

impl PitchExtractor for ZeroCrossingPitchExtractor {
    fn extract(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        frame_count: usize,
        pitch_semitones: f32,
        method: F0Method,
    ) -> Result<PitchTrack, EngineError> {
        if method == F0Method::Disabled {
            return Ok(PitchTrack {
                coarse: vec![1; frame_count],
                hz: vec![0.0; frame_count],
            });
        }
        if method != F0Method::ZeroCrossing {
            return Err(EngineError::BackendUnavailable(format!(
                "{method:?} F0 extraction requires a TensorRT/ONNX pitch adapter"
            )));
        }
        let mut hz = Vec::with_capacity(frame_count);
        for frame in 0..frame_count {
            let start = frame * PITCH_HOP_SAMPLES;
            let end = (start + PITCH_HOP_SAMPLES * 2).min(audio.len());
            if end.saturating_sub(start) < 2 {
                hz.push(0.0);
                continue;
            }
            let slice = &audio[start..end];
            let rms = (slice.iter().map(|sample| sample * sample).sum::<f32>()
                / slice.len() as f32)
                .sqrt();
            if rms < 0.005 {
                hz.push(0.0);
                continue;
            }
            let crossings = slice
                .windows(2)
                .filter(|pair| (pair[0] >= 0.0) != (pair[1] >= 0.0))
                .count();
            let estimate = crossings as f32 * sample_rate as f32 / (2.0 * slice.len() as f32);
            let shifted = (estimate * 2.0_f32.powf(pitch_semitones / 12.0)).clamp(50.0, 1100.0);
            hz.push(if estimate > 0.0 { shifted } else { 0.0 });
        }
        let coarse = hz.iter().map(|value| f0_to_coarse(*value)).collect();
        Ok(PitchTrack { coarse, hz })
    }
}

pub struct InMemoryFeatureIndex {
    vectors: FeatureMatrix,
    neighbours: usize,
}

impl InMemoryFeatureIndex {
    pub fn new(vectors: FeatureMatrix, neighbours: usize) -> Self {
        Self {
            vectors,
            neighbours: neighbours.max(1),
        }
    }
}

impl FeatureIndex for InMemoryFeatureIndex {
    fn blend(&mut self, features: &mut FeatureMatrix, index_ratio: f32) -> Result<(), EngineError> {
        if self.vectors.dims != features.dims {
            return Err(EngineError::InvalidInput(
                "index and feature dimensions differ".into(),
            ));
        }
        let ratio = index_ratio.clamp(0.0, 1.0);
        if ratio == 0.0 || self.vectors.frames == 0 {
            return Ok(());
        }
        let k = self.neighbours.min(self.vectors.frames);
        let mut replacement = vec![0.0; features.data.len()];
        for frame in 0..features.frames {
            let query = features.row(frame);
            let mut nearest: Vec<(f32, usize)> = (0..self.vectors.frames)
                .map(|candidate| {
                    let distance = query
                        .iter()
                        .zip(self.vectors.row(candidate))
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>();
                    (distance, candidate)
                })
                .collect();
            nearest.sort_by(|left, right| left.0.total_cmp(&right.0));
            let nearest = &nearest[..k];
            let mut weight_sum = 0.0;
            for (distance, candidate) in nearest {
                let weight = 1.0 / (distance.max(1e-6));
                weight_sum += weight;
                for dim in 0..features.dims {
                    replacement[frame * features.dims + dim] +=
                        self.vectors.data[candidate * features.dims + dim] * weight;
                }
            }
            for dim in 0..features.dims {
                replacement[frame * features.dims + dim] /= weight_sum.max(1e-6);
            }
        }
        let replacement = FeatureMatrix::new(features.frames, features.dims, replacement)?;
        features.blend_rows(&replacement, ratio)
    }
}

#[derive(Default)]
pub struct PassthroughSynthesizer;

impl Synthesizer for PassthroughSynthesizer {
    fn synthesize(&mut self, input: SynthesisInput<'_>) -> Result<Vec<f32>, EngineError> {
        let _ = (input.features.frames, input.pitch, input.speaker_id);
        Ok(resample_linear(
            input.source_audio,
            input.source_sample_rate,
            input.target_sample_rate,
        ))
    }
}

pub fn default_cpu_pipeline(config: RvcPipelineConfig) -> Result<RvcPipeline, EngineError> {
    RvcPipeline::new(
        config.clone(),
        Box::new(FrameEnergyEncoder::new(
            config.feature_dims,
            config.feature_hop_samples,
        )),
        Box::new(ZeroCrossingPitchExtractor),
        None,
        Box::new(PassthroughSynthesizer),
    )
}

pub fn resampled_len(input_len: usize, input_rate: u32, output_rate: u32) -> usize {
    ((input_len as u64 * output_rate as u64 + input_rate as u64 / 2) / input_rate as u64) as usize
}

pub fn resample_linear(input: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if input.is_empty() || input_rate == output_rate {
        return input.to_vec();
    }
    let output_len = resampled_len(input.len(), input_rate, output_rate).max(1);
    let scale = input_rate as f32 / output_rate as f32;
    (0..output_len)
        .map(|index| {
            let position = index as f32 * scale;
            let left = position.floor() as usize;
            let right = (left + 1).min(input.len() - 1);
            let weight = position - left as f32;
            input[left.min(input.len() - 1)] * (1.0 - weight) + input[right] * weight
        })
        .collect()
}

fn reflect_pad(input: &[f32], padding: usize) -> Vec<f32> {
    if padding == 0 || input.len() < 2 {
        return input.to_vec();
    }
    let mut output = Vec::with_capacity(input.len() + padding * 2);
    for offset in 0..padding {
        output.push(input[(padding - offset).min(input.len() - 1)]);
    }
    output.extend_from_slice(input);
    for offset in 0..padding {
        output.push(input[input.len() - 2 - offset.min(input.len() - 2)]);
    }
    output
}

fn high_pass_filter(input: &[f32], sample_rate: u32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let cutoff = 48.0;
    let rc = 1.0 / (2.0 * PI * cutoff);
    let dt = 1.0 / sample_rate as f32;
    let alpha = rc / (rc + dt);
    let mut previous_input = input[0];
    let mut previous_output = 0.0;
    let mut output = Vec::with_capacity(input.len());
    for &sample in input {
        let filtered = alpha * (previous_output + sample - previous_input);
        output.push(filtered);
        previous_input = sample;
        previous_output = filtered;
    }
    output
}

fn f0_to_coarse(f0: f32) -> i32 {
    if f0 <= 0.0 {
        return 1;
    }
    let min_mel = 1127.0_f32 * (1.0_f32 + 50.0_f32 / 700.0_f32).ln();
    let max_mel = 1127.0_f32 * (1.0_f32 + 1100.0_f32 / 700.0_f32).ln();
    let mel = 1127.0_f32 * (1.0_f32 + f0 / 700.0_f32).ln();
    (((mel - min_mel) * 254.0 / (max_mel - min_mel) + 1.0).round()).clamp(1.0, 255.0) as i32
}

fn change_rms(source: &[f32], output: &mut [f32], rate: f32) {
    if source.is_empty() || output.is_empty() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_preserves_duration_with_fixture_backend() {
        let mut pipeline = default_cpu_pipeline(RvcPipelineConfig {
            target_sample_rate: 32_000,
            output_sample_rate: 16_000,
            ..Default::default()
        })
        .unwrap();
        let input: Vec<f32> = (0..1600)
            .map(|index| (index as f32 * 0.05).sin() * 0.2)
            .collect();
        let output = pipeline
            .run(&input, 16_000, ConversionConfig::default())
            .unwrap();
        assert_eq!(output.sample_rate, 16_000);
        assert_eq!(output.samples.len(), input.len());
        assert!(output.feature_frames > 0);
        assert!(output.samples.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn index_blend_moves_features_toward_nearest_vector() {
        let mut features = FeatureMatrix::new(1, 2, vec![0.0, 0.0]).unwrap();
        let vectors = FeatureMatrix::new(2, 2, vec![1.0, 1.0, 3.0, 3.0]).unwrap();
        let mut index = InMemoryFeatureIndex::new(vectors, 1);
        index.blend(&mut features, 1.0).unwrap();
        assert_eq!(features.data, vec![1.0, 1.0]);
    }

    #[test]
    fn pitch_estimator_detects_voiced_signal() {
        let audio: Vec<f32> = (0..640)
            .map(|index| (2.0 * PI * 440.0 * index as f32 / 16_000.0).sin())
            .collect();
        let mut extractor = ZeroCrossingPitchExtractor;
        let pitch = extractor
            .extract(&audio, 16_000, 4, 0.0, F0Method::ZeroCrossing)
            .unwrap();
        assert!(pitch.hz.iter().any(|value| *value > 300.0));
    }

    struct MockTensorRt;

    impl TensorRtRtxExecutor for MockTensorRt {
        fn enqueue(&mut self, input: TensorRtRtxInput<'_>) -> Result<Vec<f32>, EngineError> {
            assert_eq!(input.p_len, input.features.frames);
            Ok(vec![0.25; input.output_samples])
        }
    }

    #[test]
    fn tensor_rt_executor_contract_receives_aligned_inputs() {
        let mut pipeline = RvcPipeline::new(
            RvcPipelineConfig {
                model_uses_f0: false,
                ..Default::default()
            },
            Box::new(FrameEnergyEncoder::new(768, HUBERT_HOP_SAMPLES)),
            Box::new(ZeroCrossingPitchExtractor),
            None,
            Box::new(TensorRtRtxSynthesizer::new(MockTensorRt)),
        )
        .unwrap();
        let output = pipeline
            .run(&vec![0.1; 1600], 16_000, ConversionConfig::default())
            .unwrap();
        assert!(
            output
                .samples
                .iter()
                .all(|sample| (*sample - 0.25).abs() < 1e-6)
        );
    }
}
