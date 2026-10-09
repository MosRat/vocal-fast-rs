use crate::CpuError;
use serde::{Deserialize, Serialize};

/// RMVPE context policy. Rolling mode requires one integral-hop playback block
/// per call; history is kept in unshifted Hz so control changes apply immediately.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum CpuF0Config {
    #[default]
    FullWindow,
    Rolling {
        advance_samples: usize,
        rmvpe_samples: usize,
    },
}
impl CpuF0Config {
    pub fn rvc_realtime() -> Self {
        Self::Rolling {
            advance_samples: 4000,
            rmvpe_samples: 4960,
        }
    }
    pub(crate) fn samples(&self, window: usize) -> usize {
        match *self {
            Self::FullWindow => window,
            Self::Rolling { rmvpe_samples, .. } => rmvpe_samples,
        }
    }
    fn validate(&self, window: usize, realtime: bool) -> Result<(), CpuError> {
        if let Self::Rolling {
            advance_samples,
            rmvpe_samples,
        } = *self
        {
            if !realtime
                || advance_samples == 0
                || advance_samples % 160 != 0
                || rmvpe_samples < 4960
                || rmvpe_samples % 160 != 0
                || rmvpe_samples > window
                || window / 160 > crate::pitch::HISTORY_FRAMES
                || advance_samples / 160 > rmvpe_samples / 160 - 3
            {
                return Err(CpuError::Config(
                    "rolling F0 requires realtime geometry, integral 160-sample advances and compatible RMVPE context/history".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CpuConfig {
    /// Persistent intra-op pool, shared by graphs with identical thread policies.
    pub threads: usize,
    /// Bounded worker spin between operators, then sleep. Zero saves power.
    pub spin_wait_us: u32,
    /// Windows: assign physical performance cores before SMT siblings. Caller
    /// affinity is scoped to compile/run; unavailable topology uses the OS policy.
    pub pin_to_performance_cores: bool,
    pub fuse: bool,
    /// Fuse exclusively consumed Linear/MatMul -> exact GELU into oneDNN.
    pub fuse_linear_gelu: bool,
    /// Fuse per-channel temporal mean/variance and affine normalization. The
    /// native NHWC kernel reduces contiguous channel tiles with FP64 sums.
    pub fuse_channel_norm: bool,
    /// Prepare layout materialization as oneDNN reorders; use checked row-wise
    /// FP32/I32 GatherElements when the selected axis is contiguous.
    pub fast_tensor_io: bool,
    /// Enable local AVX2 kernels. oneDNN and RustFFT dispatch their own ISA.
    pub avx2: bool,
    /// Keep convolution activations in NHWC layout; weights are packed once.
    pub channels_last: bool,
    pub convolution: crate::CpuConvolutionConfig,
    pub linear_mode: crate::CpuLinearMode,
    pub attention: crate::CpuAttentionConfig,
    /// Per-node timing is diagnostic and adds clock reads to the execution path.
    pub profile: bool,
    pub window_samples: usize,
    pub skip_head_frames: usize,
    pub append_hubert_frame: bool,
    pub f0: CpuF0Config,
}
impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            threads: std::thread::available_parallelism().map_or(1, |n| n.get().min(6)),
            spin_wait_us: 0,
            pin_to_performance_cores: false,
            fuse: true,
            fuse_linear_gelu: false,
            fuse_channel_norm: true,
            fast_tensor_io: true,
            avx2: true,
            channels_last: true,
            convolution: crate::CpuConvolutionConfig::default(),
            linear_mode: crate::CpuLinearMode::default(),
            attention: crate::CpuAttentionConfig::default(),
            profile: false,
            window_samples: 11200,
            skip_head_frames: 8,
            append_hubert_frame: true,
            f0: CpuF0Config::FullWindow,
        }
    }
}
impl CpuConfig {
    pub fn realtime_44960() -> Self {
        Self {
            window_samples: 44960,
            skip_head_frames: 250,
            ..Self::default()
        }
    }
    pub fn offline(samples: usize) -> Self {
        Self {
            window_samples: samples,
            skip_head_frames: 0,
            append_hubert_frame: false,
            ..Self::default()
        }
    }
    pub fn validate(&self) -> Result<(), CpuError> {
        self.attention.validate()?;
        self.f0
            .validate(self.window_samples, self.append_hubert_frame)?;
        if self.threads == 0
            || self.spin_wait_us > 1000
            || self.threads > 256
            || !(4960..=800000).contains(&self.window_samples)
            || self.skip_head_frames >= self.window_samples / 160
            || (self.append_hubert_frame && self.window_samples % 160 != 0)
        {
            return Err(CpuError::Config(
                "invalid CPU threads, window or tail crop".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_f0_rejects_fractional_hops_and_offline_use() {
        let mut config = CpuConfig::realtime_44960();
        config.f0 = CpuF0Config::rvc_realtime();
        config.validate().unwrap();
        config.f0 = CpuF0Config::Rolling {
            advance_samples: 8192,
            rmvpe_samples: 10080,
        };
        assert!(config.validate().is_err());
        config.f0 = CpuF0Config::Rolling {
            advance_samples: 4960,
            rmvpe_samples: 4960,
        };
        assert!(config.validate().is_err());
        config.f0 = CpuF0Config::rvc_realtime();
        config.append_hubert_frame = false;
        assert!(config.validate().is_err());
        config.append_hubert_frame = true;
        config.window_samples = 160 * 1025;
        assert!(config.validate().is_err());
    }
}
