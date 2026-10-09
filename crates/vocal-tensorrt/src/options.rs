//! User-facing builder and TensorRT profile options.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const DEFAULT_TRT_ROOT: &str = "C:/Program Files/NVIDIA Corporation/TensorRT-RTX/1.6.1.120";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrecisionMode {
    Fp32,
    Fp16,
    Bf16,
}

impl Default for PrecisionMode {
    fn default() -> Self {
        Self::Fp32
    }
}

impl PrecisionMode {
    pub fn code(self) -> i32 {
        match self {
            Self::Fp32 => 0,
            Self::Fp16 => 1,
            Self::Bf16 => 2,
        }
    }

    pub fn is_native_type_available(self) -> bool {
        // RTX 1.6 exposes kHALF/kBF16 tensor types. Native acoustic graphs
        // carry those types through their internal layers explicitly.
        true
    }

    pub fn from_name(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "fp16" | "half" => Self::Fp16,
            "bf16" | "bfloat16" => Self::Bf16,
            _ => Self::Fp32,
        }
    }
}

/// Dispatch policy is deliberately a stable ABI knob rather than a hard-coded
/// tactic choice.  TensorRT-RTX still owns tactic selection; the policy lets a
/// future builder plug in shape-specific tactic sources or timing-cache keys
/// without changing the pipeline manifest format.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatchTune {
    Auto,
    Realtime11200,
    Realtime44960,
}

impl Default for DispatchTune {
    fn default() -> Self {
        Self::Auto
    }
}

impl DispatchTune {
    pub fn code(self) -> i32 {
        match self {
            Self::Auto => 0,
            Self::Realtime11200 => 1,
            Self::Realtime44960 => 2,
        }
    }

    pub fn from_name(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "11200" | "realtime_11200" | "voice_changer" => Self::Realtime11200,
            "44960" | "realtime_44960" | "rvc_webui" | "applio" => Self::Realtime44960,
            _ => Self::Auto,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeBuildOptions {
    /// Model feature contract. Full pipeline builds derive this from Synth metadata.
    #[serde(default)]
    pub rvc_version: vocal_model::RvcVersion,
    pub sdk_root: PathBuf,
    pub output: PathBuf,
    /// Optional TensorRT timing-cache file used by native VFRM builds.
    #[serde(default)]
    pub timing_cache: Option<PathBuf>,
    pub workspace_bytes: u64,
    pub fp16: bool,
    #[serde(default)]
    pub precision: PrecisionMode,
    pub min_frames: i32,
    pub opt_frames: i32,
    pub max_frames: i32,
    /// Dynamic waveform/sample profile used by HuBERT and RMVPE ONNX plans.
    /// The legacy frame fields remain the defaults for VFRM Synth plans.
    pub min_samples: i32,
    pub opt_samples: i32,
    pub max_samples: i32,
    /// Use TensorRT's fused IAttention operator for relative-free graphs.
    /// RVC checkpoints with relative key/value embeddings keep the native
    /// exact path because IAttention cannot express that correction.
    #[serde(default)]
    pub fused_attention: bool,
    /// Emit selected RVC Synth stage tensors as additional plan outputs.
    /// This is a diagnostic-only build mode; production plans keep a single
    /// public `audio` output.
    pub debug_outputs: bool,
    /// Emit RMVPE mel/U-Net/GRU/logit tensors for stage parity diagnostics.
    /// This is independent of the Synth debug outputs above.
    #[serde(default)]
    pub rmvpe_debug_outputs: bool,
    /// Use the experimental radix-2 FFT frontend. The compact DFT
    /// convolution remains the production default until the larger graph's
    /// build and replay cost is characterized on each GPU.
    #[serde(default)]
    pub rmvpe_fft_radix2: bool,
    /// Use the external cuFFT STFT frontend. TensorRT receives an FP32
    /// magnitude tensor, so this mode avoids plugins while keeping mel basis
    /// and learned RMVPE layers in the native graph.
    #[serde(default)]
    pub rmvpe_fft_cufft: bool,
    /// Explicit INT8 per-output-channel weight quantization. This is an
    /// experiment: activations stay floating point and sensitive affine
    /// parameters are kept in their original precision.
    #[serde(default)]
    pub quantize_weights: bool,
    /// Fuse Q/K/V projections into one GEMM/Conv1d. The original path remains
    /// available through `VOCAL_TRT_FUSED_QKV=0` for A/B measurements.
    #[serde(default = "default_fused_qkv")]
    pub fused_qkv: bool,
    /// Fuse only the RVC encoder Q/K/V Conv1d projections. This is separate
    /// from HuBERT fusion because long Synth sequences accumulate a larger
    /// numerical difference when the three convolutions share one tactic.
    #[serde(default)]
    pub fused_rvc_qkv: bool,
    /// Per-engine precision overrides. `precision` remains the compatibility
    /// fallback; RMVPE additionally exposes region-level precision fields,
    /// while a non-FP32 Synth build uses mixed precision: FP32 encoder/flow
    /// and a low-precision NSF/HiFi-GAN decoder.
    #[serde(default)]
    pub hubert_precision: PrecisionMode,
    /// HuBERT convolutional feature extractor precision. The default follows
    /// `hubert_precision`; override it to keep the sensitive frontend in
    /// FP32 while the projection and Transformer tail use FP16/BF16.
    #[serde(default)]
    pub hubert_feature_precision: PrecisionMode,
    #[serde(default)]
    pub rmvpe_precision: PrecisionMode,
    /// RMVPE U-Net encoder/decoder precision. The default is FP32 for F0
    /// parity; FP16 is an explicit Tensor Core throughput experiment.
    #[serde(default)]
    pub rmvpe_encoder_precision: PrecisionMode,
    #[serde(default)]
    pub rmvpe_decoder_precision: PrecisionMode,
    #[serde(default)]
    pub rmvpe_stft_precision: PrecisionMode,
    #[serde(default)]
    pub rmvpe_gru_precision: PrecisionMode,
    #[serde(default)]
    pub rmvpe_logits_precision: PrecisionMode,
    #[serde(default)]
    pub synth_precision: PrecisionMode,
    /// Precision used by Synth HiFi-GAN residual blocks. The default FP16
    /// value applies only to the first three decoder stages through
    /// `synth_resblock_stage_mask`; the residual sum remains FP32.
    #[serde(default)]
    pub synth_resblock_precision: PrecisionMode,
    /// Bit mask of upsample stages whose residual blocks use the selected
    /// low precision. Bit 0 is the first decoder stage; the last bit is the
    /// highest temporal-resolution stage after all preceding upsampling.
    /// The default `0x7` selects stages 0..2. The mask is ignored for an FP32
    /// Synth graph.
    #[serde(default)]
    pub synth_resblock_stage_mask: u32,
    /// Optional per-operation precision override for residual `convs1`.
    /// A zero stage mask inherits `synth_resblock_precision` and
    /// `synth_resblock_stage_mask`, preserving the original behavior.
    #[serde(default)]
    pub synth_resblock_conv1_precision: PrecisionMode,
    #[serde(default)]
    pub synth_resblock_conv1_stage_mask: u32,
    /// Optional per-operation precision override for residual `convs2`.
    #[serde(default)]
    pub synth_resblock_conv2_precision: PrecisionMode,
    #[serde(default)]
    pub synth_resblock_conv2_stage_mask: u32,
    /// Selective W8A16 mask for Synth `convs2`: bit
    /// `(resblock_index * 3 + conv2_suffix)`. This is intended for measured
    /// decoder hotspots and requires `synth_precision = Fp16`.
    #[serde(default)]
    pub synth_w8a16_conv2_mask: u64,
    /// Experimental calibrated W8A8 masks for Synth residual Conv1/Conv2
    /// nodes. The native path currently uses one per-tensor activation scale,
    /// so validation limits the masks to one selected node.
    #[serde(default)]
    pub synth_w8a8_conv1_mask: u64,
    #[serde(default)]
    pub synth_w8a8_conv2_mask: u64,
    #[serde(default = "default_w8a8_activation_scale")]
    pub synth_w8a8_activation_scale: f32,
    /// Fold the residual-block average into the following decoder weights.
    /// This removes four elementwise scale kernels while keeping the same
    /// FP32 operation order. It is opt-in because low-precision constants are
    /// rounded at build time.
    #[serde(default)]
    pub synth_fuse_resblock_scale: bool,
    /// Replace ConvTranspose1d with an equivalent phase-split Conv1d + shuffle
    /// graph. Phase-dependent asymmetric padding handles both divisible and
    /// non-divisible RVC kernel/stride pairs; unsupported output-length
    /// contracts are rejected at build time.
    #[serde(default)]
    pub synth_polyphase_upsample: bool,
    /// Move all learned Synth decoder convolutions (pre/upsample/noise,
    /// residual blocks and post) to the decoder precision. Residual branch
    /// accumulation remains FP32 unless `synth_fp16_residual_accum` is set.
    /// The NSF phase/source stays FP32 unless `synth_fp16_source` is set.
    #[serde(default)]
    pub synth_full_fp16: bool,
    /// Positive scale applied immediately before the first FP32 -> FP16
    /// decoder boundary. The first upsample kernel is folded by `1/scale`,
    /// preserving the affine result while reducing FP16 quantization of the
    /// large `decoder_pre` activation. `1.0` disables the transform.
    #[serde(default = "default_synth_fp16_input_scale")]
    pub synth_fp16_input_scale: f32,
    /// Allow residual branch sums to remain in FP16 in the full-decoder
    /// experiment. This saves cast kernels but is more numerically sensitive.
    #[serde(default)]
    pub synth_fp16_residual_accum: bool,
    /// Run the NSF source arithmetic in the decoder precision. This is kept
    /// separate because phase accumulation is more sensitive than Conv1d.
    #[serde(default)]
    pub synth_fp16_source: bool,
    /// Keep the learned NSF noise convolutions and their adds in the decoder
    /// precision while leaving the phase/source generation in FP32. This
    /// removes the per-stage FP32 round trip around `noise_convs`; it is an
    /// opt-in experiment because the final-stage noise path is voice
    /// dependent.
    #[serde(default)]
    pub synth_fp16_noise: bool,
    /// Optional per-upsampler mask for the learned NSF noise convolutions.
    /// Bit 0 is the first decoder stage. A nonzero mask enables only those
    /// stages in the decoder FP16 domain; the boolean switch above remains a
    /// compatibility shorthand for all stages.
    #[serde(default)]
    pub synth_fp16_noise_stage_mask: u32,
    /// Keep a complete decoder stage in FP16: upsample/deconvolution, NSF
    /// noise convolution, residual convolutions, and residual accumulation.
    /// This is the coherent low-precision mode used to remove intermediate
    /// FP32 round trips. The NSF phase/source and final tanh remain governed
    /// by their dedicated switches.
    #[serde(default)]
    pub synth_fp16_coherent_stage_mask: u32,
    /// Stages that must remain entirely FP32 when Synth uses FP16. The
    /// default `0x8` keeps the final waveform-rate stage FP32 while stages 0..2
    /// use the mixed FP16 recipe.
    #[serde(default = "default_synth_fp16_fp32_stage_mask")]
    pub synth_fp16_fp32_stage_mask: u32,
    /// Run only the final learned `conv_post` projection in FP16, then cast
    /// its result back to FP32 for tanh. The projection is small, so this is
    /// kept as a diagnostic knob rather than a default optimization.
    #[serde(default)]
    pub synth_fp16_post: bool,
    /// Emit one structured build summary through `tracing`.
    #[serde(default)]
    pub log_build_options: bool,
    #[serde(default)]
    pub dispatch_tune: DispatchTune,
    /// Bitset of TensorRT tactic sources. Bit 0 enables edge-mask
    /// convolutions and bit 1 enables JIT convolutions. Keeping this in the
    /// manifest makes fixed-profile engine builds reproducible when one of
    /// the RTX tactics is numerically unstable for a voice model.
    #[serde(default = "default_tactic_sources")]
    pub tactic_sources: u32,
}

fn default_fused_qkv() -> bool {
    true
}

pub(crate) fn default_w8a8_activation_scale() -> f32 {
    1.0
}

pub(crate) fn default_synth_fp16_input_scale() -> f32 {
    1.0
}

pub(crate) const fn default_synth_fp16_fp32_stage_mask() -> u32 {
    0x8
}

pub(crate) fn default_tactic_sources() -> u32 {
    0b11
}

impl Default for NativeBuildOptions {
    fn default() -> Self {
        Self {
            rvc_version: vocal_model::RvcVersion::V2,
            sdk_root: PathBuf::from(DEFAULT_TRT_ROOT),
            output: "rvc.plan".into(),
            timing_cache: None,
            workspace_bytes: 1 << 30,
            fp16: false,
            precision: PrecisionMode::Fp32,
            min_frames: 8,
            opt_frames: 192,
            max_frames: 2048,
            // Conservative all-reference profile points at 16 kHz.  The
            // pipeline compiler replaces these with the selected
            // RvcTrtProfile; these defaults are for direct ONNX builds.
            min_samples: 11200,
            opt_samples: 35520,
            max_samples: 44960,
            fused_attention: false,
            debug_outputs: false,
            rmvpe_debug_outputs: false,
            rmvpe_fft_radix2: false,
            rmvpe_fft_cufft: false,
            quantize_weights: false,
            fused_qkv: true,
            fused_rvc_qkv: false,
            hubert_precision: PrecisionMode::Fp32,
            hubert_feature_precision: PrecisionMode::Fp32,
            rmvpe_precision: PrecisionMode::Fp32,
            rmvpe_encoder_precision: PrecisionMode::Fp32,
            rmvpe_decoder_precision: PrecisionMode::Fp32,
            // These regions are intentionally fixed at FP32. Their fields are
            // serialized for an explicit ABI contract, but are not user
            // tunables because F0 decode and recurrent state are sensitive.
            rmvpe_stft_precision: PrecisionMode::Fp32,
            rmvpe_gru_precision: PrecisionMode::Fp32,
            rmvpe_logits_precision: PrecisionMode::Fp32,
            synth_precision: PrecisionMode::Fp16,
            synth_resblock_precision: PrecisionMode::Fp16,
            synth_resblock_stage_mask: 0x7,
            synth_resblock_conv1_precision: PrecisionMode::Fp32,
            synth_resblock_conv1_stage_mask: 0,
            synth_resblock_conv2_precision: PrecisionMode::Fp32,
            synth_resblock_conv2_stage_mask: 0,
            synth_w8a16_conv2_mask: 0,
            synth_w8a8_conv1_mask: 0,
            synth_w8a8_conv2_mask: 0,
            synth_w8a8_activation_scale: default_w8a8_activation_scale(),
            synth_fuse_resblock_scale: false,
            synth_polyphase_upsample: false,
            synth_full_fp16: false,
            synth_fp16_input_scale: default_synth_fp16_input_scale(),
            synth_fp16_residual_accum: false,
            synth_fp16_source: false,
            // The validated mixed012 recipe keeps the learned NSF noise
            // convolutions in the same FP16 region as stages 0..2. Stage 3
            // is still forced back to FP32 by synth_fp16_fp32_stage_mask.
            synth_fp16_noise: true,
            synth_fp16_noise_stage_mask: 0,
            synth_fp16_coherent_stage_mask: 0,
            synth_fp16_fp32_stage_mask: default_synth_fp16_fp32_stage_mask(),
            synth_fp16_post: false,
            log_build_options: false,
            dispatch_tune: DispatchTune::Auto,
            tactic_sources: default_tactic_sources(),
        }
    }
}

impl NativeBuildOptions {
    /// Strict environment adapter retained for callers that only need a
    /// builder. Applications that also run a pipeline should prefer the full
    /// [`crate::TensorRtConfig`] so all groups are parsed together.
    pub fn from_env() -> Result<Self, crate::ConfigError> {
        Ok(crate::TensorRtConfig::from_env()?.build)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OnnxProfileKind {
    Frames,
    Samples,
}
