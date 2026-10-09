use crate::DirectMlError;
use serde::{Deserialize, Serialize};
pub use vocal_graph::{Precision, PrecisionConfig, RetrievalSearch};

/// GPU binding storage. State history is never part of a normal host upload.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IoBuffers {
    #[default]
    Dedicated,
    Pooled,
}

/// Execution choices independent of model geometry and mixed precision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExecutionConfig {
    /// Share a healthy device/queue while another graph still owns it.
    pub reuse_device: bool,
    pub io_buffers: IoBuffers,
    /// One transient GPU arena for initializer-owned weights.
    pub pooled_weights: bool,
    /// Bounded, reusable host staging allocation during initialization.
    pub weight_staging_mib: u32,
    /// Reinterpret dense reshapes without inserting an Identity dispatch.
    pub contiguous_views: bool,
    /// GEMM's C operand can change FP16 rounding; enable after model parity.
    pub fuse_gemm_bias: bool,
}
impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            reuse_device: true,
            // Pooled bindings reduce allocations/copies but did not improve
            // dispatch latency on the measured NVIDIA adapter.
            io_buffers: IoBuffers::Dedicated,
            pooled_weights: true,
            weight_staging_mib: 16,
            // Paired real-model measurements showed no stable benefit from
            // dense-view elision; retain it as an explicit experiment.
            contiguous_views: false,
            fuse_gemm_bias: false,
        }
    }
}
impl ExecutionConfig {
    pub fn validate(&self) -> Result<(), DirectMlError> {
        if !(1..=64).contains(&self.weight_staging_mib) {
            return Err(DirectMlError::Config(
                "DirectML weight staging must be 1..=64 MiB".into(),
            ));
        }
        Ok(())
    }
    /// Reproduce the original bindings and native graph lowering.
    pub fn legacy() -> Self {
        Self {
            reuse_device: false,
            io_buffers: IoBuffers::Dedicated,
            pooled_weights: false,
            weight_staging_mib: 16,
            contiguous_views: false,
            fuse_gemm_bias: false,
        }
    }
}

/// Fixed streaming geometry; history is retained on the GPU between calls.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum F0Config {
    #[default]
    FullWindow,
    Rolling {
        advance_samples: usize,
        rmvpe_samples: usize,
    },
}
impl F0Config {
    pub fn rvc_realtime() -> Self {
        Self::Rolling {
            advance_samples: 4000,
            rmvpe_samples: 4960,
        }
    }
    pub(crate) fn validate(&self, window: usize) -> Result<(), DirectMlError> {
        if let Self::Rolling {
            advance_samples,
            rmvpe_samples,
        } = *self
        {
            // RMVPE emits n/160+1 frames, of which [3:-1] is cached.
            // Requiring an integral hop advance prevents cumulative F0 drift.
            if advance_samples == 0
                || advance_samples % 160 != 0
                || rmvpe_samples < 4960
                || rmvpe_samples % 160 != 0
                || rmvpe_samples > window
                || window / 160 > 1024
                || advance_samples / 160 > rmvpe_samples / 160 - 3
            {
                return Err(DirectMlError::Config(
                    "invalid rolling F0 window or hop advance".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DirectMlConfig {
    pub adapter_index: u32,
    /// Prefer a display adapter interface ID over an enumeration index.
    pub adapter_id: Option<String>,
    pub window_samples: usize,
    pub skip_head_frames: usize,
    /// Realtime repeats an appended final HuBERT frame; offline uses real frames.
    pub append_hubert_frame: bool,
    pub precision: PrecisionConfig,
    pub execution: ExecutionConfig,
    pub f0: F0Config,
    pub retrieval_search: RetrievalSearch,
    pub debug_outputs: bool,
    pub debug_layer: bool,
}
impl Default for DirectMlConfig {
    fn default() -> Self {
        Self {
            adapter_index: 0,
            adapter_id: None,
            window_samples: 44960,
            skip_head_frames: 250,
            append_hubert_frame: true,
            precision: PrecisionConfig::default(),
            execution: ExecutionConfig::default(),
            f0: F0Config::default(),
            retrieval_search: RetrievalSearch::default(),
            debug_outputs: false,
            debug_layer: false,
        }
    }
}
impl DirectMlConfig {
    pub fn offline(window_samples: usize) -> Self {
        Self {
            window_samples,
            skip_head_frames: 0,
            append_hubert_frame: false,
            f0: F0Config::FullWindow,
            ..Self::default()
        }
    }
    pub fn realtime_44960() -> Self {
        Self {
            // Short rolling windows produced voiced/unvoiced and octave
            // differences on continuous speech. Keep the full acoustic context.
            f0: F0Config::FullWindow,
            ..Self::default()
        }
    }
    pub fn realtime_11200() -> Self {
        Self {
            window_samples: 11200,
            skip_head_frames: 8,
            ..Self::default()
        }
    }
    pub fn validate(&self) -> Result<(), DirectMlError> {
        self.execution.validate()?;
        if self.window_samples < 4960
            || self.window_samples > 800_000
            || (self.append_hubert_frame && self.window_samples % 160 != 0)
            || self.skip_head_frames >= self.window_samples / 160
            || !self.precision.synth_input_scale.is_finite()
            || self.precision.synth_input_scale <= 0.
        {
            return Err(DirectMlError::Config(
                "invalid fixed window, tail crop or Synth scale".into(),
            ));
        }
        self.f0.validate(self.window_samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn offline_accepts_long_unaligned_waveforms_without_realtime_history() {
        let config = DirectMlConfig::offline(465_603);
        config.validate().unwrap();
        assert!(!config.append_hubert_frame);
        assert_eq!(config.f0, F0Config::FullWindow);
        assert_eq!(config.skip_head_frames, 0);
        let old: DirectMlConfig = serde_json::from_str("{}").unwrap();
        assert!(old.append_hubert_frame);
    }
    #[test]
    fn rejects_unaligned_windows_crops_and_nonfinite_scaling() {
        for config in [
            DirectMlConfig {
                window_samples: 44961,
                ..Default::default()
            },
            DirectMlConfig {
                skip_head_frames: 281,
                ..Default::default()
            },
            DirectMlConfig {
                precision: PrecisionConfig {
                    synth_input_scale: f32::NAN,
                    ..Default::default()
                },
                ..Default::default()
            },
        ] {
            assert!(config.validate().is_err());
        }
        assert!(DirectMlConfig::realtime_11200().validate().is_ok());
    }
    #[test]
    fn old_manifests_receive_the_precision_policy_defaults() {
        let config: DirectMlConfig =
            serde_json::from_str("{\"window_samples\":11200,\"skip_head_frames\":8}").unwrap();
        assert_eq!(config.precision, PrecisionConfig::default());
        assert_eq!(config.f0, F0Config::FullWindow);
        assert_eq!(config.execution, ExecutionConfig::default());
        assert_eq!(
            serde_json::from_str::<PrecisionConfig>(
                &serde_json::to_string(&PrecisionConfig::fp32()).unwrap()
            )
            .unwrap(),
            PrecisionConfig::fp32()
        );
    }

    #[test]
    fn execution_policy_roundtrips_and_rejects_misspelled_options() {
        for execution in [ExecutionConfig::legacy(), ExecutionConfig::default()] {
            let config = DirectMlConfig {
                execution,
                ..Default::default()
            };
            assert_eq!(
                serde_json::from_str::<DirectMlConfig>(&serde_json::to_string(&config).unwrap())
                    .unwrap(),
                config
            );
        }
        assert!(
            serde_json::from_str::<DirectMlConfig>(r#"{"execution":{"pool_io":true}}"#).is_err()
        );
        for weight_staging_mib in [0, 65] {
            assert!(
                ExecutionConfig {
                    weight_staging_mib,
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
    }

    #[test]
    fn rolling_f0_rejects_drift_and_uncovered_cache_gaps() {
        assert!(DirectMlConfig::realtime_44960().validate().is_ok());
        assert_eq!(DirectMlConfig::realtime_44960().f0, F0Config::FullWindow);
        for (advance_samples, rmvpe_samples) in [(8192, 10080), (4960, 4960), (4000, 44961)] {
            assert!(
                DirectMlConfig {
                    f0: F0Config::Rolling {
                        advance_samples,
                        rmvpe_samples
                    },
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
    }
}
