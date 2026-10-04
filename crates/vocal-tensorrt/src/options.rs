//! User-facing builder and TensorRT profile options.

use serde::{Deserialize, Serialize};
use std::{env, path::PathBuf};

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

    pub fn from_env() -> Self {
        Self::from_name(&env::var("VOCAL_TRT_PRECISION").unwrap_or_else(|_| "fp32".into()))
    }

    pub fn from_name(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "fp16" | "half" => Self::Fp16,
            "bf16" | "bfloat16" => Self::Bf16,
            _ => Self::Fp32,
        }
    }

    fn from_env_or(name: &str, fallback: Self) -> Self {
        env::var(name)
            .map(|value| Self::from_name(&value))
            .unwrap_or(fallback)
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

    pub fn from_env() -> Self {
        match env::var("VOCAL_TRT_DISPATCH_TUNE")
            .unwrap_or_else(|_| "auto".into())
            .to_ascii_lowercase()
            .as_str()
        {
            "11200" | "realtime_11200" | "voice_changer" => Self::Realtime11200,
            "44960" | "realtime_44960" | "rvc_webui" | "applio" => Self::Realtime44960,
            _ => Self::Auto,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeBuildOptions {
    pub sdk_root: PathBuf,
    pub output: PathBuf,
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
    /// Precision used by Synth HiFi-GAN residual blocks.  A value different
    /// from Fp32 removes the per-stage FP32 round trip and lets TensorRT use
    /// low-precision convolution tactics.  The default remains Fp32 because
    /// residual accumulation is the most sensitive decoder region.
    #[serde(default)]
    pub synth_resblock_precision: PrecisionMode,
    /// Bit mask of upsample stages whose residual blocks use the selected
    /// low precision. Bit 0 is the first (highest temporal resolution) stage.
    /// Zero keeps all residual stages in FP32; the mask is ignored for an
    /// FP32 Synth graph.
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

pub(crate) fn default_tactic_sources() -> u32 {
    0b11
}

impl Default for NativeBuildOptions {
    fn default() -> Self {
        let precision = PrecisionMode::from_env();
        // Keep RMVPE conservative by default.  Its U-Net can be lowered to
        // FP16 independently, but the native full-chain F0 gate must pass
        // before that is selected for a deployment.  An explicit RMVPE
        // precision request remains unambiguous and selects both U-Net
        // regions unless a region override is provided.
        let rmvpe_precision_env = env::var("VOCAL_TRT_RMVPE_PRECISION").ok();
        let rmvpe_precision = rmvpe_precision_env
            .as_deref()
            .map(PrecisionMode::from_name)
            .unwrap_or(PrecisionMode::Fp32);
        let rmvpe_unet_default = rmvpe_precision_env
            .as_deref()
            .map(PrecisionMode::from_name)
            .unwrap_or(PrecisionMode::Fp32);
        let hubert_precision = PrecisionMode::from_env_or("VOCAL_TRT_HUBERT_PRECISION", precision);
        let hubert_feature_precision =
            PrecisionMode::from_env_or("VOCAL_TRT_HUBERT_FEATURE_PRECISION", hubert_precision);
        let fft_mode = env::var("VOCAL_TRT_RMVPE_FFT")
            .unwrap_or_default()
            .to_ascii_lowercase();
        Self {
            sdk_root: env::var_os("TENSORRT_RTX_ROOT")
                .or_else(|| env::var_os("TRT_RTX_ROOT"))
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(DEFAULT_TRT_ROOT)),
            output: "rvc.plan".into(),
            workspace_bytes: 1 << 30,
            fp16: false,
            precision,
            min_frames: env_i32("VOCAL_TRT_MIN_FRAMES", 8),
            opt_frames: env_i32("VOCAL_TRT_OPT_FRAMES", 192),
            max_frames: env_i32("VOCAL_TRT_MAX_FRAMES", 2048),
            // Conservative all-reference profile points at 16 kHz.  The
            // pipeline compiler replaces these with the selected
            // RvcTrtProfile; these defaults are for direct ONNX builds.
            min_samples: 11200,
            opt_samples: 35520,
            max_samples: 44960,
            fused_attention: env::var("VOCAL_TRT_FUSED_ATTENTION")
                .ok()
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
            debug_outputs: env::var("VOCAL_TRT_DEBUG_GRAPH_OUTPUTS")
                .ok()
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
            rmvpe_debug_outputs: env::var("VOCAL_TRT_RMVPE_DEBUG_OUTPUTS")
                .ok()
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
            rmvpe_fft_radix2: matches!(fft_mode.as_str(), "1" | "true" | "radix2"),
            rmvpe_fft_cufft: matches!(fft_mode.as_str(), "cufft" | "cuda" | "cu_fft"),
            quantize_weights: env::var("VOCAL_TRT_WEIGHT_INT8")
                .ok()
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
            fused_qkv: env::var("VOCAL_TRT_FUSED_QKV")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(true),
            fused_rvc_qkv: env::var("VOCAL_TRT_FUSED_RVC_QKV")
                .ok()
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
            hubert_precision,
            hubert_feature_precision,
            rmvpe_precision,
            rmvpe_encoder_precision: PrecisionMode::from_env_or(
                "VOCAL_TRT_RMVPE_ENCODER_PRECISION",
                rmvpe_unet_default,
            ),
            rmvpe_decoder_precision: PrecisionMode::from_env_or(
                "VOCAL_TRT_RMVPE_DECODER_PRECISION",
                rmvpe_unet_default,
            ),
            // These regions are intentionally fixed at FP32. Their fields are
            // serialized for an explicit ABI contract, but are not user
            // tunables because F0 decode and recurrent state are sensitive.
            rmvpe_stft_precision: PrecisionMode::Fp32,
            rmvpe_gru_precision: PrecisionMode::Fp32,
            rmvpe_logits_precision: PrecisionMode::Fp32,
            synth_precision: PrecisionMode::from_env_or(
                "VOCAL_TRT_SYNTH_PRECISION",
                PrecisionMode::Fp32,
            ),
            synth_resblock_precision: PrecisionMode::from_env_or(
                "VOCAL_TRT_SYNTH_RESBLOCK_PRECISION",
                PrecisionMode::Fp32,
            ),
            synth_resblock_stage_mask: env::var("VOCAL_TRT_SYNTH_RESBLOCK_STAGE_MASK")
                .ok()
                .and_then(|v| {
                    let value = v.trim();
                    if let Some(hex) = value
                        .strip_prefix("0x")
                        .or_else(|| value.strip_prefix("0X"))
                    {
                        u32::from_str_radix(hex, 16).ok()
                    } else {
                        value.parse::<u32>().ok()
                    }
                })
                .unwrap_or(0),
            synth_resblock_conv1_precision: PrecisionMode::from_env_or(
                "VOCAL_TRT_SYNTH_RESBLOCK_CONV1_PRECISION",
                PrecisionMode::Fp32,
            ),
            synth_resblock_conv1_stage_mask: stage_mask_from_env(
                "VOCAL_TRT_SYNTH_RESBLOCK_CONV1_STAGE_MASK",
            ),
            synth_resblock_conv2_precision: PrecisionMode::from_env_or(
                "VOCAL_TRT_SYNTH_RESBLOCK_CONV2_PRECISION",
                PrecisionMode::Fp32,
            ),
            synth_resblock_conv2_stage_mask: stage_mask_from_env(
                "VOCAL_TRT_SYNTH_RESBLOCK_CONV2_STAGE_MASK",
            ),
            synth_fuse_resblock_scale: env::var("VOCAL_TRT_SYNTH_FUSE_RESBLOCK_SCALE")
                .ok()
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
            synth_polyphase_upsample: env::var("VOCAL_TRT_SYNTH_UPSAMPLE_IMPL")
                .ok()
                .is_some_and(|v| v.eq_ignore_ascii_case("polyphase")),
            dispatch_tune: DispatchTune::from_env(),
            tactic_sources: env::var("VOCAL_TRT_TACTIC_SOURCES")
                .ok()
                .and_then(|value| {
                    let value = value.trim();
                    if let Some(hex) = value
                        .strip_prefix("0x")
                        .or_else(|| value.strip_prefix("0X"))
                    {
                        u32::from_str_radix(hex, 16).ok()
                    } else {
                        value.parse::<u32>().ok()
                    }
                })
                .unwrap_or_else(default_tactic_sources),
        }
    }
}

fn env_i32(name: &str, fallback: i32) -> i32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(fallback)
}

fn stage_mask_from_env(name: &str) -> u32 {
    env::var(name)
        .ok()
        .and_then(|value| {
            let value = value.trim();
            if let Some(hex) = value
                .strip_prefix("0x")
                .or_else(|| value.strip_prefix("0X"))
            {
                u32::from_str_radix(hex, 16).ok()
            } else {
                value.parse::<u32>().ok()
            }
        })
        .unwrap_or(0)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OnnxProfileKind {
    Frames,
    Samples,
}
