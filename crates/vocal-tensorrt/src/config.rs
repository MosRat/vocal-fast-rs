//! Semantic TensorRT configuration.
//!
//! Environment variables are an integration boundary only. The rest of the
//! crate consumes these typed values, which makes embedding the pipeline in a
//! Tauri worker or another Rust service independent of process environment.

use std::{env, path::PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    error::ConfigError,
    options::{DispatchTune, NativeBuildOptions, PrecisionMode, default_tactic_sources},
    pipeline::RvcTrtProfile,
};

/// CUDA Graph ownership strategy for the realtime pipeline.
///
/// `NativeSubgraphs` delegates capture/replay to TensorRT-RTX for each
/// execution context. `WholePipeline` uses one Rust/CUDA graph for the serial
/// device-resident pipeline. `HybridSubgraphs` captures an acoustic graph and
/// a decode graph separately, preserving a smaller failure and shape scope.
/// `Disabled` is the explicit eager baseline used for comparisons.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GraphReplayMode {
    NativeSubgraphs,
    WholePipeline,
    HybridSubgraphs,
    Disabled,
}

impl Default for GraphReplayMode {
    fn default() -> Self {
        Self::NativeSubgraphs
    }
}

impl GraphReplayMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NativeSubgraphs => "native_subgraphs",
            Self::WholePipeline => "whole_pipeline",
            Self::HybridSubgraphs => "hybrid_subgraphs",
            Self::Disabled => "disabled",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "native" | "native_subgraphs" | "per_stage" | "subgraphs" => {
                Some(Self::NativeSubgraphs)
            }
            "whole" | "whole_pipeline" | "pipeline" | "external" => Some(Self::WholePipeline),
            "hybrid" | "hybrid_subgraphs" | "split" => Some(Self::HybridSubgraphs),
            "disabled" | "eager" | "off" | "none" => Some(Self::Disabled),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipelineRuntimeConfig {
    #[serde(default)]
    pub graph_mode: GraphReplayMode,
    pub cuda_graph: bool,
    pub native_cuda_graph: bool,
    /// Number of application-level CUDA Graph executables retained for
    /// different dynamic shape keys. Native TensorRT graph replay has its own
    /// SDK-managed cache; this value only applies to the Rust fallback path.
    #[serde(default = "default_graph_cache_size")]
    pub graph_cache_size: usize,
    pub parallel_acoustic: bool,
    pub async_audio_d2h: bool,
}

const fn default_graph_cache_size() -> usize {
    2
}

impl Default for PipelineRuntimeConfig {
    fn default() -> Self {
        Self {
            graph_mode: GraphReplayMode::NativeSubgraphs,
            cuda_graph: true,
            native_cuda_graph: true,
            graph_cache_size: default_graph_cache_size(),
            parallel_acoustic: false,
            async_audio_d2h: true,
        }
    }
}

impl PipelineRuntimeConfig {
    /// Resolve the explicit mode while preserving the two boolean variables
    /// used by older callers. A non-native legacy configuration maps to the
    /// whole-pipeline external graph; disabling graph replay always selects
    /// the eager baseline.
    pub fn resolved_graph_mode(&self) -> GraphReplayMode {
        // Non-default enum values are explicit selections and therefore do
        // not inherit either legacy boolean. The default NativeSubgraphs value
        // remains compatible with callers that only set the old flags.
        if self.graph_mode != GraphReplayMode::NativeSubgraphs {
            return self.graph_mode;
        }
        if !self.cuda_graph {
            return GraphReplayMode::Disabled;
        }
        if !self.native_cuda_graph {
            GraphReplayMode::WholePipeline
        } else {
            GraphReplayMode::NativeSubgraphs
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiagnosticsConfig {
    pub trace: bool,
    pub debug_graph_outputs: bool,
    pub rmvpe_debug_outputs: bool,
    pub debug_intermediates: bool,
    pub layer_profile: bool,
    pub log_build_options: bool,
    pub allow_nonfinite: bool,
    pub dump_hubert: Option<PathBuf>,
    pub dump_audio: Option<PathBuf>,
}

impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self {
            trace: false,
            debug_graph_outputs: false,
            rmvpe_debug_outputs: false,
            debug_intermediates: false,
            layer_profile: false,
            log_build_options: false,
            allow_nonfinite: false,
            dump_hubert: None,
            dump_audio: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipelineRunConfig {
    pub warmup: usize,
    pub iterations: usize,
    pub realtime_f0: bool,
    pub realtime_advance_frames: Option<usize>,
    pub test_frames: usize,
}

impl Default for PipelineRunConfig {
    fn default() -> Self {
        Self {
            warmup: 0,
            iterations: 1,
            realtime_f0: false,
            realtime_advance_frames: None,
            test_frames: 8,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorRtConfig {
    pub build: NativeBuildOptions,
    pub profile: RvcTrtProfile,
    pub runtime: PipelineRuntimeConfig,
    pub diagnostics: DiagnosticsConfig,
    pub run: PipelineRunConfig,
}

impl Default for TensorRtConfig {
    fn default() -> Self {
        Self {
            build: NativeBuildOptions::default(),
            profile: RvcTrtProfile::reference(),
            runtime: PipelineRuntimeConfig::default(),
            diagnostics: DiagnosticsConfig::default(),
            run: PipelineRunConfig::default(),
        }
    }
}

impl TensorRtConfig {
    /// Parse all supported process variables once and validate the resulting
    /// combination. No runtime code should call `std::env::var` directly.
    pub fn from_env() -> Result<Self, ConfigError> {
        let mut config = Self::default();
        let build = &mut config.build;

        if let Some(value) = first_env(&[
            "TENSORRT_RTX_ROOT",
            "TRT_RTX_ROOT",
            "TENSORRT_ROOT",
            "TRT_ROOT",
        ]) {
            build.sdk_root = PathBuf::from(value);
        }
        build.timing_cache = env::var_os("VOCAL_TRT_TIMING_CACHE").map(PathBuf::from);
        build.precision = precision("VOCAL_TRT_PRECISION", build.precision)?;
        build.fp16 = bool_value("VOCAL_TRT_FP16", build.fp16)?;
        build.min_frames = positive_i32("VOCAL_TRT_MIN_FRAMES", build.min_frames)?;
        build.opt_frames = positive_i32("VOCAL_TRT_OPT_FRAMES", build.opt_frames)?;
        build.max_frames = positive_i32("VOCAL_TRT_MAX_FRAMES", build.max_frames)?;
        build.min_samples = positive_i32("VOCAL_TRT_MIN_SAMPLES", build.min_samples)?;
        build.opt_samples = positive_i32("VOCAL_TRT_OPT_SAMPLES", build.opt_samples)?;
        build.max_samples = positive_i32("VOCAL_TRT_MAX_SAMPLES", build.max_samples)?;
        build.fused_attention = bool_value("VOCAL_TRT_FUSED_ATTENTION", false)?;
        build.debug_outputs = bool_value("VOCAL_TRT_DEBUG_GRAPH_OUTPUTS", false)?;
        build.rmvpe_debug_outputs = bool_value("VOCAL_TRT_RMVPE_DEBUG_OUTPUTS", false)?;
        build.quantize_weights = bool_value("VOCAL_TRT_WEIGHT_INT8", false)?;
        build.fused_qkv = bool_value("VOCAL_TRT_FUSED_QKV", true)?;
        build.fused_rvc_qkv = bool_value("VOCAL_TRT_FUSED_RVC_QKV", false)?;
        build.hubert_precision = precision("VOCAL_TRT_HUBERT_PRECISION", build.precision)?;
        build.hubert_feature_precision =
            precision("VOCAL_TRT_HUBERT_FEATURE_PRECISION", build.hubert_precision)?;
        build.rmvpe_precision = precision("VOCAL_TRT_RMVPE_PRECISION", PrecisionMode::Fp32)?;
        build.rmvpe_encoder_precision =
            precision("VOCAL_TRT_RMVPE_ENCODER_PRECISION", build.rmvpe_precision)?;
        build.rmvpe_decoder_precision =
            precision("VOCAL_TRT_RMVPE_DECODER_PRECISION", build.rmvpe_precision)?;
        build.synth_precision = precision("VOCAL_TRT_SYNTH_PRECISION", PrecisionMode::Fp32)?;
        build.synth_resblock_precision =
            precision("VOCAL_TRT_SYNTH_RESBLOCK_PRECISION", PrecisionMode::Fp32)?;
        build.synth_resblock_stage_mask = mask(
            "VOCAL_TRT_SYNTH_RESBLOCK_STAGE_MASK",
            build.synth_resblock_stage_mask,
        )?;
        build.synth_resblock_conv1_precision = precision(
            "VOCAL_TRT_SYNTH_RESBLOCK_CONV1_PRECISION",
            PrecisionMode::Fp32,
        )?;
        build.synth_resblock_conv1_stage_mask =
            mask("VOCAL_TRT_SYNTH_RESBLOCK_CONV1_STAGE_MASK", 0)?;
        build.synth_resblock_conv2_precision = precision(
            "VOCAL_TRT_SYNTH_RESBLOCK_CONV2_PRECISION",
            PrecisionMode::Fp32,
        )?;
        build.synth_resblock_conv2_stage_mask =
            mask("VOCAL_TRT_SYNTH_RESBLOCK_CONV2_STAGE_MASK", 0)?;
        build.synth_w8a16_conv2_mask = u64_mask(
            "VOCAL_TRT_SYNTH_W8A16_CONV2_MASK",
            build.synth_w8a16_conv2_mask,
        )?;
        build.synth_w8a8_conv1_mask = u64_mask(
            "VOCAL_TRT_SYNTH_W8A8_CONV1_MASK",
            build.synth_w8a8_conv1_mask,
        )?;
        build.synth_w8a8_conv2_mask = u64_mask(
            "VOCAL_TRT_SYNTH_W8A8_CONV2_MASK",
            build.synth_w8a8_conv2_mask,
        )?;
        build.synth_w8a8_activation_scale = positive_float(
            "VOCAL_TRT_SYNTH_W8A8_ACTIVATION_SCALE",
            build.synth_w8a8_activation_scale,
        )?;
        build.synth_fuse_resblock_scale = bool_value("VOCAL_TRT_SYNTH_FUSE_RESBLOCK_SCALE", false)?;
        build.synth_polyphase_upsample = match env::var("VOCAL_TRT_SYNTH_UPSAMPLE_IMPL") {
            Ok(value) if value.eq_ignore_ascii_case("polyphase") => true,
            Ok(value) if value.eq_ignore_ascii_case("convtranspose") => false,
            Ok(value) => {
                return Err(ConfigError::InvalidValue {
                    name: "VOCAL_TRT_SYNTH_UPSAMPLE_IMPL".into(),
                    value,
                    expected: "polyphase or convtranspose".into(),
                });
            }
            Err(_) => false,
        };
        build.synth_full_fp16 = bool_value("VOCAL_TRT_SYNTH_FULL_FP16", false)?;
        build.synth_fp16_input_scale = positive_float(
            "VOCAL_TRT_SYNTH_FP16_INPUT_SCALE",
            build.synth_fp16_input_scale,
        )?;
        build.synth_fp16_residual_accum = bool_value("VOCAL_TRT_SYNTH_FP16_RESIDUAL_ACCUM", false)?;
        build.synth_fp16_source = bool_value("VOCAL_TRT_SYNTH_FP16_SOURCE", false)?;
        build.synth_fp16_noise = bool_value("VOCAL_TRT_SYNTH_FP16_NOISE", false)?;
        build.synth_fp16_noise_stage_mask = mask(
            "VOCAL_TRT_SYNTH_FP16_NOISE_STAGE_MASK",
            build.synth_fp16_noise_stage_mask,
        )?;
        build.synth_fp16_coherent_stage_mask = mask(
            "VOCAL_TRT_SYNTH_FP16_COHERENT_STAGE_MASK",
            build.synth_fp16_coherent_stage_mask,
        )?;
        build.synth_fp16_post = bool_value("VOCAL_TRT_SYNTH_FP16_POST", false)?;
        build.log_build_options = bool_value("VOCAL_TRT_LOG_BUILD_OPTIONS", false)?;
        build.dispatch_tune = dispatch("VOCAL_TRT_DISPATCH_TUNE", DispatchTune::Auto)?;
        build.tactic_sources = mask("VOCAL_TRT_TACTIC_SOURCES", default_tactic_sources())?;
        let fft = env::var("VOCAL_TRT_RMVPE_FFT").unwrap_or_default();
        match fft.to_ascii_lowercase().as_str() {
            "" | "0" | "false" | "none" => {}
            "1" | "true" | "radix2" => build.rmvpe_fft_radix2 = true,
            "cufft" | "cuda" | "cu_fft" => build.rmvpe_fft_cufft = true,
            _ => {
                return Err(ConfigError::InvalidValue {
                    name: "VOCAL_TRT_RMVPE_FFT".into(),
                    value: fft,
                    expected: "none, radix2, or cufft".into(),
                });
            }
        }

        config.profile = match env::var("VOCAL_TRT_PROFILE") {
            Ok(value) => RvcTrtProfile::from_name(&value)?,
            Err(_) => RvcTrtProfile::reference(),
        };
        config.runtime.cuda_graph = bool_value("VOCAL_TRT_CUDA_GRAPH", true)?;
        config.runtime.native_cuda_graph = bool_value("VOCAL_TRT_NATIVE_CUDA_GRAPH", true)?;
        if let Some(value) = env::var("VOCAL_TRT_GRAPH_MODE").ok() {
            let graph_mode =
                GraphReplayMode::parse(&value).ok_or_else(|| ConfigError::InvalidValue {
                    name: "VOCAL_TRT_GRAPH_MODE".into(),
                    value,
                    expected: "native_subgraphs, whole_pipeline, hybrid_subgraphs, or disabled"
                        .into(),
                })?;
            // An explicit mode is the authoritative setting. Align the legacy
            // flags so old callers cannot remap the selected mode.
            config.runtime.cuda_graph = graph_mode != GraphReplayMode::Disabled;
            config.runtime.native_cuda_graph = graph_mode == GraphReplayMode::NativeSubgraphs;
            config.runtime.graph_mode = graph_mode;
        }
        config.runtime.graph_cache_size = positive_usize(
            "VOCAL_TRT_GRAPH_CACHE_SIZE",
            config.runtime.graph_cache_size,
        )?;
        config.runtime.parallel_acoustic = bool_value("VOCAL_TRT_PARALLEL_ACOUSTIC", false)?;
        config.runtime.async_audio_d2h = bool_value("VOCAL_TRT_ASYNC_AUDIO_D2H", true)?;

        config.diagnostics.trace = bool_value("VOCAL_TRT_TRACE", false)?;
        config.diagnostics.debug_graph_outputs = build.debug_outputs;
        config.diagnostics.rmvpe_debug_outputs = build.rmvpe_debug_outputs;
        config.diagnostics.debug_intermediates =
            bool_value("VOCAL_TRT_DEBUG_INTERMEDIATES", false)?;
        config.diagnostics.layer_profile = bool_value("VOCAL_TRT_LAYER_PROFILE", false)?;
        config.diagnostics.log_build_options = build.log_build_options;
        config.diagnostics.allow_nonfinite = bool_value("VOCAL_TRT_ALLOW_NONFINITE", false)?;
        config.diagnostics.dump_hubert = env::var_os("VOCAL_TRT_DUMP_HUBERT").map(PathBuf::from);
        config.diagnostics.dump_audio = env::var_os("VOCAL_TRT_DUMP_AUDIO").map(PathBuf::from);

        config.run.warmup = positive_usize("VOCAL_TRT_WARMUP", config.run.warmup)?;
        config.run.iterations =
            positive_usize("VOCAL_TRT_PIPELINE_ITERATIONS", config.run.iterations)?;
        config.run.realtime_f0 = bool_value("VOCAL_TRT_REALTIME_F0", false)?;
        config.run.realtime_advance_frames =
            optional_positive_usize("VOCAL_TRT_REALTIME_ADVANCE_FRAMES")?;
        config.run.test_frames = positive_usize("VOCAL_TRT_TEST_FRAMES", config.run.test_frames)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.build.min_frames <= 0
            || self.build.min_frames > self.build.opt_frames
            || self.build.opt_frames > self.build.max_frames
            || self.build.min_samples <= 0
            || self.build.min_samples > self.build.opt_samples
            || self.build.opt_samples > self.build.max_samples
            || self.build.tactic_sources > 3
        {
            return Err(ConfigError::InvalidCombination(
                "frame/sample profiles must be positive and min <= opt <= max; tactic sources must be 0..=3".into(),
            ));
        }
        if self.runtime.graph_cache_size == 0 || self.runtime.graph_cache_size > 64 {
            return Err(ConfigError::InvalidCombination(
                "CUDA graph cache size must be in 1..=64".into(),
            ));
        }
        if self.build.rmvpe_fft_radix2 && self.build.rmvpe_fft_cufft {
            return Err(ConfigError::InvalidCombination(
                "RMVPE FFT cannot select both radix2 and cuFFT".into(),
            ));
        }
        if self.build.synth_w8a16_conv2_mask != 0
            && self.build.synth_precision != PrecisionMode::Fp16
        {
            return Err(ConfigError::InvalidCombination(
                "Synth W8A16 Conv2 mask requires VOCAL_TRT_SYNTH_PRECISION=fp16".into(),
            ));
        }
        if self.build.synth_w8a16_conv2_mask != 0 && self.build.quantize_weights {
            return Err(ConfigError::InvalidCombination(
                "Synth W8A16 Conv2 mask cannot be combined with VOCAL_TRT_WEIGHT_INT8".into(),
            ));
        }
        let w8a8_mask = self.build.synth_w8a8_conv1_mask | self.build.synth_w8a8_conv2_mask;
        if w8a8_mask != 0 {
            if self.build.synth_precision != PrecisionMode::Fp16 {
                return Err(ConfigError::InvalidCombination(
                    "Synth W8A8 masks require VOCAL_TRT_SYNTH_PRECISION=fp16".into(),
                ));
            }
            if self.build.quantize_weights {
                return Err(ConfigError::InvalidCombination(
                    "Synth W8A8 masks cannot be combined with VOCAL_TRT_WEIGHT_INT8".into(),
                ));
            }
            if w8a8_mask.count_ones() != 1 {
                return Err(ConfigError::InvalidCombination(
                    "Synth W8A8 currently supports exactly one selected Conv1/Conv2 node".into(),
                ));
            }
            if self.build.synth_w8a16_conv2_mask & self.build.synth_w8a8_conv2_mask != 0 {
                return Err(ConfigError::InvalidCombination(
                    "Synth W8A8 and W8A16 masks cannot select the same Conv2 node".into(),
                ));
            }
        }
        if (self.build.synth_full_fp16
            || self.build.synth_fp16_residual_accum
            || self.build.synth_fp16_source
            || self.build.synth_fp16_noise
            || self.build.synth_fp16_noise_stage_mask != 0
            || self.build.synth_fp16_coherent_stage_mask != 0
            || self.build.synth_fp16_post)
            && self.build.synth_precision != PrecisionMode::Fp16
        {
            return Err(ConfigError::InvalidCombination(
                "Synth FP16 decoder options require VOCAL_TRT_SYNTH_PRECISION=fp16".into(),
            ));
        }
        if self.build.synth_fp16_residual_accum && !self.build.synth_full_fp16 {
            return Err(ConfigError::InvalidCombination(
                "VOCAL_TRT_SYNTH_FP16_RESIDUAL_ACCUM requires VOCAL_TRT_SYNTH_FULL_FP16=1".into(),
            ));
        }
        if self.build.synth_fp16_noise_stage_mask & !0x0f != 0 {
            return Err(ConfigError::InvalidCombination(
                "Synth FP16 noise stage mask must select only upsample stages 0..=3".into(),
            ));
        }
        if self.build.synth_fp16_coherent_stage_mask & !0x0f != 0 {
            return Err(ConfigError::InvalidCombination(
                "Synth coherent FP16 stage mask must select only upsample stages 0..=3".into(),
            ));
        }
        if !self.build.synth_fp16_input_scale.is_finite()
            || self.build.synth_fp16_input_scale <= 0.0
        {
            return Err(ConfigError::InvalidCombination(
                "Synth FP16 input scale must be finite and positive".into(),
            ));
        }
        self.profile
            .validate()
            .map_err(|error| ConfigError::InvalidCombination(error.to_string()))
    }

    /// Resolve the graph mode from the complete configuration.
    pub fn resolved_graph_mode(&self) -> GraphReplayMode {
        self.runtime.resolved_graph_mode()
    }
}

fn first_env(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| env::var(name).ok())
}

fn invalid(name: &str, value: impl Into<String>, expected: &str) -> ConfigError {
    ConfigError::InvalidValue {
        name: name.into(),
        value: value.into(),
        expected: expected.into(),
    }
}

fn bool_value(name: &str, default: bool) -> Result<bool, ConfigError> {
    let Some(value) = env::var(name).ok() else {
        return Ok(default);
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(invalid(name, value, "true/false, 1/0, yes/no, or on/off")),
    }
}

fn positive_i32(name: &str, default: i32) -> Result<i32, ConfigError> {
    let Some(value) = env::var(name).ok() else {
        return Ok(default);
    };
    value
        .parse()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(name, value, "a positive 32-bit integer"))
}

fn positive_usize(name: &str, default: usize) -> Result<usize, ConfigError> {
    let Some(value) = env::var(name).ok() else {
        return Ok(default);
    };
    value
        .parse()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(name, value, "a positive integer"))
}

fn optional_positive_usize(name: &str) -> Result<Option<usize>, ConfigError> {
    let Some(value) = env::var(name).ok() else {
        return Ok(None);
    };
    value
        .parse()
        .ok()
        .filter(|value| *value > 0)
        .map(Some)
        .ok_or_else(|| invalid(name, value, "a positive integer"))
}

fn mask(name: &str, default: u32) -> Result<u32, ConfigError> {
    let Some(value) = env::var(name).ok() else {
        return Ok(default);
    };
    let trimmed = value.trim();
    let digits = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
        .unwrap_or(trimmed);
    u32::from_str_radix(
        digits,
        if trimmed.starts_with("0x") || trimmed.starts_with("0X") {
            16
        } else {
            10
        },
    )
    .map_err(|_| invalid(name, value, "a decimal or hexadecimal 32-bit mask"))
}

fn u64_mask(name: &str, default: u64) -> Result<u64, ConfigError> {
    let Some(value) = env::var(name).ok() else {
        return Ok(default);
    };
    let trimmed = value.trim();
    let digits = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
        .unwrap_or(trimmed);
    u64::from_str_radix(
        digits,
        if trimmed.starts_with("0x") || trimmed.starts_with("0X") {
            16
        } else {
            10
        },
    )
    .map_err(|_| invalid(name, value, "a decimal or hexadecimal 64-bit mask"))
}

fn positive_float(name: &str, default: f32) -> Result<f32, ConfigError> {
    let Some(value) = env::var(name).ok() else {
        return Ok(default);
    };
    value
        .trim()
        .parse::<f32>()
        .ok()
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| invalid(name, value, "a positive finite float"))
}

fn precision(name: &str, default: PrecisionMode) -> Result<PrecisionMode, ConfigError> {
    let Some(value) = env::var(name).ok() else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "fp32" | "float" | "float32" => Ok(PrecisionMode::Fp32),
        "fp16" | "half" | "float16" => Ok(PrecisionMode::Fp16),
        "bf16" | "bfloat16" => Ok(PrecisionMode::Bf16),
        _ => Err(invalid(name, value, "fp32, fp16, or bf16")),
    }
}

fn dispatch(name: &str, default: DispatchTune) -> Result<DispatchTune, ConfigError> {
    let Some(value) = env::var(name).ok() else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "auto" => Ok(DispatchTune::Auto),
        "11200" | "realtime_11200" | "voice_changer" => Ok(DispatchTune::Realtime11200),
        "44960" | "realtime_44960" | "rvc_webui" | "applio" => Ok(DispatchTune::Realtime44960),
        _ => Err(ConfigError::InvalidValue {
            name: name.into(),
            value,
            expected: "auto, 11200, or 44960".into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn clear() {
        for name in [
            "VOCAL_TRT_PROFILE",
            "VOCAL_TRT_GRAPH_CACHE_SIZE",
            "VOCAL_TRT_GRAPH_MODE",
            "VOCAL_TRT_CUDA_GRAPH",
            "VOCAL_TRT_NATIVE_CUDA_GRAPH",
            "VOCAL_TRT_PRECISION",
            "VOCAL_TRT_TACTIC_SOURCES",
            "VOCAL_TRT_SYNTH_RESBLOCK_STAGE_MASK",
            "VOCAL_TRT_SYNTH_W8A16_CONV2_MASK",
            "VOCAL_TRT_SYNTH_PRECISION",
            "VOCAL_TRT_WEIGHT_INT8",
            "VOCAL_TRT_SYNTH_W8A8_CONV1_MASK",
            "VOCAL_TRT_SYNTH_W8A8_CONV2_MASK",
            "VOCAL_TRT_SYNTH_W8A8_ACTIVATION_SCALE",
            "VOCAL_TRT_SYNTH_FULL_FP16",
            "VOCAL_TRT_SYNTH_FP16_INPUT_SCALE",
            "VOCAL_TRT_SYNTH_FP16_RESIDUAL_ACCUM",
            "VOCAL_TRT_SYNTH_FP16_SOURCE",
            "VOCAL_TRT_SYNTH_FP16_NOISE",
            "VOCAL_TRT_SYNTH_FP16_NOISE_STAGE_MASK",
            "VOCAL_TRT_SYNTH_FP16_COHERENT_STAGE_MASK",
            "VOCAL_TRT_SYNTH_FP16_POST",
        ] {
            unsafe { env::remove_var(name) };
        }
    }

    #[test]
    fn defaults_are_stable_without_environment() {
        let _guard = env_lock();
        clear();
        let config = TensorRtConfig::from_env().expect("default config");
        assert_eq!(config.profile.max_samples, 44_960);
        assert_eq!(config.build.tactic_sources, 3);
        assert!(config.runtime.cuda_graph);
        assert_eq!(config.runtime.graph_cache_size, 2);
        assert_eq!(
            config.runtime.resolved_graph_mode(),
            GraphReplayMode::NativeSubgraphs
        );
    }

    #[test]
    fn parses_hex_masks_and_aliases() {
        let _guard = env_lock();
        clear();
        unsafe {
            env::set_var("VOCAL_TRT_PROFILE", "44960");
            env::set_var("VOCAL_TRT_PRECISION", "half");
            env::set_var("VOCAL_TRT_TACTIC_SOURCES", "0x2");
            env::set_var("VOCAL_TRT_SYNTH_RESBLOCK_STAGE_MASK", "0xc");
            env::set_var("VOCAL_TRT_SYNTH_PRECISION", "fp16");
            env::set_var("VOCAL_TRT_SYNTH_W8A16_CONV2_MASK", "0x30000");
            env::set_var("VOCAL_TRT_SYNTH_W8A8_CONV1_MASK", "0x1000");
            env::set_var("VOCAL_TRT_SYNTH_W8A8_ACTIVATION_SCALE", "0.0079044188");
            env::set_var("VOCAL_TRT_SYNTH_FULL_FP16", "1");
            env::set_var("VOCAL_TRT_SYNTH_FP16_INPUT_SCALE", "0.0078125");
            env::set_var("VOCAL_TRT_SYNTH_FP16_SOURCE", "1");
            env::set_var("VOCAL_TRT_SYNTH_FP16_NOISE", "1");
            env::set_var("VOCAL_TRT_SYNTH_FP16_NOISE_STAGE_MASK", "0x4");
            env::set_var("VOCAL_TRT_SYNTH_FP16_COHERENT_STAGE_MASK", "0x2");
            env::set_var("VOCAL_TRT_SYNTH_FP16_POST", "1");
            env::set_var("VOCAL_TRT_GRAPH_CACHE_SIZE", "4");
            env::set_var("VOCAL_TRT_GRAPH_MODE", "hybrid");
        }
        let config = TensorRtConfig::from_env().expect("aliased config");
        assert_eq!(config.build.precision, PrecisionMode::Fp16);
        assert_eq!(config.build.tactic_sources, 2);
        assert_eq!(config.build.synth_resblock_stage_mask, 12);
        assert_eq!(config.build.synth_w8a16_conv2_mask, 0x30000);
        assert_eq!(config.build.synth_w8a8_conv1_mask, 0x1000);
        assert!((config.build.synth_w8a8_activation_scale - 0.0079044).abs() < 1e-6);
        assert!(config.build.synth_full_fp16);
        assert!((config.build.synth_fp16_input_scale - 0.0078125).abs() < 1e-7);
        assert!(config.build.synth_fp16_source);
        assert!(config.build.synth_fp16_noise);
        assert_eq!(config.build.synth_fp16_noise_stage_mask, 4);
        assert_eq!(config.build.synth_fp16_coherent_stage_mask, 2);
        assert!(config.build.synth_fp16_post);
        assert_eq!(config.profile.min_samples, 44_960);
        assert_eq!(config.runtime.graph_cache_size, 4);
        assert_eq!(
            config.runtime.resolved_graph_mode(),
            GraphReplayMode::HybridSubgraphs
        );
        clear();
    }

    #[test]
    fn rejects_w8a16_mask_for_non_fp16_synth() {
        let _guard = env_lock();
        clear();
        unsafe {
            env::set_var("VOCAL_TRT_SYNTH_PRECISION", "fp32");
            env::set_var("VOCAL_TRT_SYNTH_W8A16_CONV2_MASK", "0x30000");
        }
        assert!(matches!(
            TensorRtConfig::from_env(),
            Err(ConfigError::InvalidCombination(_))
        ));
        clear();
    }

    #[test]
    fn rejects_w8a16_mask_with_global_weight_int8() {
        let _guard = env_lock();
        clear();
        unsafe {
            env::set_var("VOCAL_TRT_SYNTH_PRECISION", "fp16");
            env::set_var("VOCAL_TRT_SYNTH_W8A16_CONV2_MASK", "0x30000");
            env::set_var("VOCAL_TRT_WEIGHT_INT8", "1");
        }
        assert!(matches!(
            TensorRtConfig::from_env(),
            Err(ConfigError::InvalidCombination(_))
        ));
        clear();
    }

    #[test]
    fn rejects_multiple_w8a8_nodes() {
        let _guard = env_lock();
        clear();
        unsafe {
            env::set_var("VOCAL_TRT_SYNTH_PRECISION", "fp16");
            env::set_var("VOCAL_TRT_SYNTH_W8A8_CONV1_MASK", "0x3");
            env::set_var("VOCAL_TRT_SYNTH_W8A8_ACTIVATION_SCALE", "0.01");
        }
        assert!(matches!(
            TensorRtConfig::from_env(),
            Err(ConfigError::InvalidCombination(_))
        ));
        clear();
    }

    #[test]
    fn rejects_noise_mask_outside_decoder_stages() {
        let _guard = env_lock();
        clear();
        unsafe {
            env::set_var("VOCAL_TRT_SYNTH_PRECISION", "fp16");
            env::set_var("VOCAL_TRT_SYNTH_FP16_NOISE_STAGE_MASK", "0x10");
        }
        assert!(matches!(
            TensorRtConfig::from_env(),
            Err(ConfigError::InvalidCombination(_))
        ));
        clear();
    }

    #[test]
    fn rejects_invalid_values() {
        let _guard = env_lock();
        clear();
        unsafe { env::set_var("VOCAL_TRT_PRECISION", "fp8") };
        assert!(matches!(
            TensorRtConfig::from_env(),
            Err(ConfigError::InvalidValue { .. })
        ));
        clear();
    }

    #[test]
    fn legacy_graph_flags_resolve_to_whole_pipeline() {
        let mut runtime = PipelineRuntimeConfig::default();
        runtime.native_cuda_graph = false;
        assert_eq!(
            runtime.resolved_graph_mode(),
            GraphReplayMode::WholePipeline
        );
        runtime.cuda_graph = false;
        assert_eq!(runtime.resolved_graph_mode(), GraphReplayMode::Disabled);
    }

    #[test]
    fn explicit_graph_mode_wins_over_legacy_native_flag() {
        let mut runtime = PipelineRuntimeConfig::default();
        runtime.graph_mode = GraphReplayMode::HybridSubgraphs;
        assert_eq!(
            runtime.resolved_graph_mode(),
            GraphReplayMode::HybridSubgraphs
        );

        runtime.graph_mode = GraphReplayMode::WholePipeline;
        runtime.native_cuda_graph = true;
        assert_eq!(
            runtime.resolved_graph_mode(),
            GraphReplayMode::WholePipeline
        );
        runtime.cuda_graph = false;
        assert_eq!(
            runtime.resolved_graph_mode(),
            GraphReplayMode::WholePipeline
        );

        let _guard = env_lock();
        clear();
        unsafe {
            env::set_var("VOCAL_TRT_GRAPH_MODE", "native_subgraphs");
            env::set_var("VOCAL_TRT_CUDA_GRAPH", "0");
            env::set_var("VOCAL_TRT_NATIVE_CUDA_GRAPH", "0");
        }
        let config = TensorRtConfig::from_env().expect("explicit native mode");
        assert_eq!(
            config.runtime.resolved_graph_mode(),
            GraphReplayMode::NativeSubgraphs
        );
        clear();
    }

    #[test]
    fn graph_mode_parser_accepts_stable_names_and_aliases() {
        for (value, expected) in [
            ("native_subgraphs", GraphReplayMode::NativeSubgraphs),
            ("PER_STAGE", GraphReplayMode::NativeSubgraphs),
            ("whole_pipeline", GraphReplayMode::WholePipeline),
            ("external", GraphReplayMode::WholePipeline),
            ("hybrid", GraphReplayMode::HybridSubgraphs),
            ("split", GraphReplayMode::HybridSubgraphs),
            ("disabled", GraphReplayMode::Disabled),
            ("eager", GraphReplayMode::Disabled),
        ] {
            assert_eq!(GraphReplayMode::parse(value), Some(expected));
            assert_eq!(
                GraphReplayMode::parse(value).unwrap().as_str(),
                expected.as_str()
            );
        }
        assert_eq!(GraphReplayMode::parse("unknown"), None);
    }
}
