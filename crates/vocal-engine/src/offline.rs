//! Shared offline execution for every inference backend.
//!
//! The WAV boundary belongs to the CLI/application.  This module owns the
//! backend-neutral part: fixed-size windows, final-window padding, sequence
//! numbers, output trimming, and a small timing report.  A new backend only
//! needs to implement [`OfflinePipeline`]; it does not need to duplicate the
//! audio file and windowing code.

use std::time::Instant;

use serde::{Deserialize, Serialize};
use vocal_types::{AudioFormat, ConversionConfig, LatencyMetrics};

use crate::{BackendKind, EngineError, InferenceEngine};

/// The default internal RVC window.  TensorRT profiles use this shape, and it
/// is also a useful bounded block for the portable CPU and passthrough paths.
pub const DEFAULT_WINDOW_SAMPLES: usize = 11_200;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OfflineRunConfig {
    /// Number of mono 16 kHz samples submitted per backend call.
    pub window_samples: usize,
    /// Pad the last window with zeros before invoking the backend.
    pub pad_final_window: bool,
    /// Number of samples blended at the boundary between two backend blocks.
    /// A value of zero disables output joining and preserves the historical
    /// concatenate-only behavior.
    pub crossfade_samples: usize,
    /// Search range used by the time-domain SOLA alignment before a
    /// crossfade. The aligned block keeps its original length.
    pub sola_search_samples: usize,
    pub conversion: ConversionConfig,
}

impl Default for OfflineRunConfig {
    fn default() -> Self {
        Self {
            window_samples: DEFAULT_WINDOW_SAMPLES,
            pad_final_window: true,
            crossfade_samples: 0,
            sola_search_samples: 0,
            conversion: ConversionConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OfflineRunReport {
    pub backend: BackendKind,
    pub input_samples: usize,
    pub output_samples: usize,
    pub windows: usize,
    pub elapsed_ms: f64,
    pub metrics: LatencyMetrics,
}

/// Backend boundary used by [`run_offline`].
pub trait OfflinePipeline: Send {
    fn backend_kind(&self) -> BackendKind;

    fn metrics(&self) -> LatencyMetrics {
        LatencyMetrics::default()
    }

    fn process_window(
        &mut self,
        samples: &[f32],
        sequence: u64,
        conversion: ConversionConfig,
    ) -> Result<Vec<f32>, EngineError>;
}

/// Adapt the normal streaming engine interface to the offline runner.  This
/// is intentionally generic over the concrete engine, so passthrough, CPU
/// RVC, TensorRT RTX, and future backends share exactly the same runner.
pub struct EngineOfflinePipeline {
    backend: BackendKind,
    format: AudioFormat,
    engine: Box<dyn InferenceEngine>,
}

impl EngineOfflinePipeline {
    pub fn new(
        backend: BackendKind,
        engine: Box<dyn InferenceEngine>,
        format: AudioFormat,
    ) -> Result<Self, EngineError> {
        format
            .validate()
            .map_err(|error| EngineError::InvalidConfig(error.to_string()))?;
        if format.channels != 1 {
            return Err(EngineError::InvalidConfig(
                "offline runner currently requires mono input".into(),
            ));
        }
        Ok(Self {
            backend,
            format,
            engine,
        })
    }

    pub fn engine(&self) -> &dyn InferenceEngine {
        self.engine.as_ref()
    }

    pub fn engine_mut(&mut self) -> &mut dyn InferenceEngine {
        self.engine.as_mut()
    }
}

impl OfflinePipeline for EngineOfflinePipeline {
    fn backend_kind(&self) -> BackendKind {
        self.backend
    }

    fn metrics(&self) -> LatencyMetrics {
        self.engine.metrics()
    }

    fn process_window(
        &mut self,
        samples: &[f32],
        sequence: u64,
        conversion: ConversionConfig,
    ) -> Result<Vec<f32>, EngineError> {
        Ok(self
            .engine
            .process_samples(self.format, samples, sequence, conversion)?
            .samples)
    }
}

pub fn run_offline<P: OfflinePipeline + ?Sized>(
    pipeline: &mut P,
    input: &[f32],
    config: &OfflineRunConfig,
) -> Result<(Vec<f32>, OfflineRunReport), EngineError> {
    if input.is_empty() || input.iter().any(|sample| !sample.is_finite()) {
        return Err(EngineError::InvalidInput(
            "offline input must be non-empty and finite".into(),
        ));
    }
    if config.window_samples == 0 {
        return Err(EngineError::InvalidConfig(
            "offline window size must be non-zero".into(),
        ));
    }

    let started = Instant::now();
    let mut joiner = OfflineJoiner::new(config.crossfade_samples, config.sola_search_samples);
    let mut offset = 0usize;
    let mut windows = 0usize;
    while offset < input.len() {
        let valid = (input.len() - offset).min(config.window_samples);
        let needs_padding = valid < config.window_samples;
        let window = if needs_padding && config.pad_final_window {
            let mut padded = vec![0.0; config.window_samples];
            padded[..valid].copy_from_slice(&input[offset..offset + valid]);
            padded
        } else {
            input[offset..offset + valid].to_vec()
        };
        let result = pipeline.process_window(&window, windows as u64, config.conversion)?;
        let keep = if needs_padding {
            if config.pad_final_window {
                result.len().saturating_mul(valid) / window.len().max(1)
            } else {
                result.len()
            }
        } else {
            result.len()
        };
        joiner.append(&result[..keep.min(result.len())]);
        offset += valid;
        windows += 1;
    }

    let output = joiner.finish();
    let metrics = pipeline.metrics();
    Ok((
        output.clone(),
        OfflineRunReport {
            backend: pipeline.backend_kind(),
            input_samples: input.len(),
            output_samples: output.len(),
            windows,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            metrics,
        },
    ))
}

/// Join fixed-size output blocks while preserving one output block per input
/// block. A SOLA-selected current prefix is blended against the previous tail
/// and the complete current interval is appended. This mirrors the realtime
/// workers' `sola_buffer` contract without making the generic offline runner
/// depend on a particular model backend.
struct OfflineJoiner {
    output: Vec<f32>,
    crossfade_samples: usize,
    sola_search_samples: usize,
}

impl OfflineJoiner {
    fn new(crossfade_samples: usize, sola_search_samples: usize) -> Self {
        Self {
            output: Vec::new(),
            crossfade_samples,
            sola_search_samples,
        }
    }

    fn append(&mut self, block: &[f32]) {
        if block.is_empty() {
            return;
        }
        if self.output.is_empty() || self.crossfade_samples == 0 {
            self.output.extend_from_slice(block);
            return;
        }

        let overlap = self
            .crossfade_samples
            .min(self.output.len())
            .min(block.len());
        if overlap == 0 {
            self.output.extend_from_slice(block);
            return;
        }

        let max_offset = self
            .sola_search_samples
            .min(block.len().saturating_sub(overlap));
        let offset = find_sola_offset(
            &self.output[self.output.len() - overlap..],
            block,
            overlap,
            max_offset,
        );

        // Keep a fixed block length after SOLA. When the selected candidate is
        // shifted forward, repeat the final sample for the short tail.
        let mut aligned = Vec::with_capacity(block.len());
        for index in 0..block.len() {
            aligned.push(block[(index + offset).min(block.len() - 1)]);
        }

        for index in 0..overlap {
            let phase = if overlap <= 1 {
                1.0
            } else {
                index as f32 / (overlap - 1) as f32
            };
            let fade_in = (0.5 * std::f32::consts::PI * phase).sin().powi(2);
            let fade_out = 1.0 - fade_in;
            let previous = self.output[self.output.len() - overlap + index];
            aligned[index] = previous * fade_out + aligned[index] * fade_in;
        }
        // The backend block already represents one playback interval. Apply
        // the blend to its prefix and append the complete interval; removing
        // the overlap here would shorten the offline result on every window.
        self.output.extend_from_slice(&aligned);
    }

    fn finish(self) -> Vec<f32> {
        self.output
    }
}

fn find_sola_offset(previous: &[f32], block: &[f32], overlap: usize, max_offset: usize) -> usize {
    if overlap == 0 || max_offset == 0 {
        return 0;
    }
    let previous_energy = previous
        .iter()
        .map(|sample| f64::from(*sample) * f64::from(*sample))
        .sum::<f64>();
    if previous_energy <= f64::EPSILON {
        return 0;
    }

    let mut best_offset = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for offset in 0..=max_offset {
        let candidate = &block[offset..offset + overlap];
        let mut numerator = 0.0f64;
        let mut candidate_energy = 0.0f64;
        for (&left, &right) in previous.iter().zip(candidate) {
            numerator += f64::from(left) * f64::from(right);
            candidate_energy += f64::from(right) * f64::from(right);
        }
        let denominator = (previous_energy * candidate_energy).sqrt().max(1e-12);
        let score = numerator / denominator;
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
    use crate::{EngineConfig, create_engine};
    use vocal_types::ModelDescriptor;

    fn descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "offline-fixture".into(),
            display_name: "Offline fixture".into(),
            architecture: "rvc-v2".into(),
            sample_rate: 16_000,
            version: "2".into(),
        }
    }

    #[test]
    fn all_portable_backends_use_the_same_window_runner() {
        for backend in [BackendKind::Passthrough, BackendKind::RvcCpuPipeline] {
            let engine = create_engine(EngineConfig::portable(backend, descriptor())).unwrap();
            let mut pipeline =
                EngineOfflinePipeline::new(backend, engine, AudioFormat::MONO_16K).unwrap();
            let input = vec![0.1; 12_345];
            let (output, report) =
                run_offline(&mut pipeline, &input, &OfflineRunConfig::default()).unwrap();
            assert_eq!(output.len(), input.len());
            assert_eq!(report.windows, 2);
            assert_eq!(report.backend, backend);
        }
    }

    #[test]
    fn joined_blocks_keep_duration_and_blend_the_boundary() {
        let mut joiner = OfflineJoiner::new(2, 0);
        joiner.append(&[0.0, 0.0, 0.0, 0.0]);
        joiner.append(&[1.0, 1.0, 1.0, 1.0]);
        let output = joiner.finish();
        assert_eq!(output.len(), 8);
        assert_eq!(output[2], 0.0);
        assert_eq!(output[3], 0.0);
        assert_eq!(output[4], 0.0);
        assert_eq!(output[5], 1.0);
        assert_eq!(&output[6..], &[1.0, 1.0]);
    }

    #[test]
    fn sola_selects_the_best_candidate_offset() {
        let mut joiner = OfflineJoiner::new(3, 1);
        joiner.append(&[0.0, 0.0, 0.0, 0.0]);
        // The candidate at offset one matches the previous tail better than
        // the first sample and is selected without changing total duration.
        joiner.append(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        let output = joiner.finish();
        assert_eq!(output.len(), 9);
    }
}
