//! End-to-end CUDA pipeline for the waveform HuBERT/RMVPE exports and the
//! source-independent VFRM Synth graph.
//!
//! The four engines execute on one stream.  Activations stay device resident;
//! only the waveform upload and final audio download cross the host boundary.

use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use serde::{Deserialize, Serialize};
use vocal_model::{ModelPackage, RvcSynthConfig};

use crate::timing::CudaEvent;
use crate::{
    CudaBuffer, CudaGraphExec, CudaStream, OwnedContext, PinnedHostRing, Runtime,
    builder::NativeTensorRtBuilder,
    config::{DiagnosticsConfig, GraphReplayMode, PipelineRuntimeConfig, TensorRtConfig},
    cufft::{CufftFrontend, unique_shapes},
    error::ConfigError,
    error::NativeBuildError,
    options::{
        DispatchTune, NativeBuildOptions, OnnxProfileKind, PrecisionMode, default_tactic_sources,
    },
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RvcTrtProfile {
    pub min_samples: i32,
    pub opt_samples: i32,
    pub max_samples: i32,
    pub min_hubert_frames: i32,
    pub opt_hubert_frames: i32,
    pub max_hubert_frames: i32,
    pub min_pitch_frames: i32,
    pub opt_pitch_frames: i32,
    pub max_pitch_frames: i32,
    pub min_rvc_frames: i32,
    pub opt_rvc_frames: i32,
    pub max_rvc_frames: i32,
}

/// The host-side realtime window used by the reference applications, expressed
/// at the RVC/HuBERT internal rate (16 kHz).  `chunk_samples_16k` is the audio
/// block that will be returned to the user; the other fields are context used
/// by the model and by the overlap joiner.  TensorRT receives their sum after
/// it has been rounded up to the 160-sample RVC hop.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RvcRealtimeConfig {
    pub chunk_samples_16k: usize,
    pub extra_samples_16k: usize,
    pub crossfade_samples_16k: usize,
    pub sola_search_samples_16k: usize,
}

impl RvcRealtimeConfig {
    pub const fn new(
        chunk_samples_16k: usize,
        extra_samples_16k: usize,
        crossfade_samples_16k: usize,
        sola_search_samples_16k: usize,
    ) -> Self {
        Self {
            chunk_samples_16k,
            extra_samples_16k,
            crossfade_samples_16k,
            sola_search_samples_16k,
        }
    }

    /// Convert a source-rate window to the internal 16 kHz domain.  The
    /// reference implementations use integer truncation for each component
    /// before they round the complete conversion window to a model hop.
    pub const fn from_source_samples(
        source_rate: usize,
        chunk_samples: usize,
        extra_samples: usize,
        crossfade_samples: usize,
        sola_search_samples: usize,
    ) -> Self {
        assert!(source_rate > 0);
        Self::new(
            chunk_samples * 16_000 / source_rate,
            extra_samples * 16_000 / source_rate,
            crossfade_samples * 16_000 / source_rate,
            sola_search_samples * 16_000 / source_rate,
        )
    }

    /// RVC requires an integral 10 ms (160 sample) feature hop.  This is the
    /// actual waveform length sent to HuBERT and RMVPE by the reference code.
    pub const fn inference_samples(self) -> usize {
        let total = self.chunk_samples_16k
            + self.extra_samples_16k
            + self.crossfade_samples_16k
            + self.sola_search_samples_16k;
        round_up_samples(total, RVC_HOP_SAMPLES)
    }

    /// RVC WebUI's current realtime GUI defaults (250 ms block, 50 ms
    /// crossfade, 2.5 s context and a 10 ms SOLA search at 16 kHz).
    pub const fn rvc_webui() -> Self {
        Self::new(4_000, 40_000, 800, 160)
    }

    /// The older `configs/config.json`/VST defaults (130/80/2000/10 ms).
    pub const fn rvc_webui_legacy() -> Self {
        Self::new(2_080, 32_000, 1_280, 160)
    }

    /// voice-changer's default 48 kHz callback (192 * 128 samples), 4096
    /// sample overlap/context and 12 ms SOLA search, after its 16 kHz RVC
    /// resample.  The per-component truncation mirrors `RVCr2.py`.
    pub const fn voice_changer() -> Self {
        Self::from_source_samples(48_000, 24_576, 4_096, 4_096, 576)
    }

    /// Applio's realtime UI defaults (250 ms, 50 ms, 2.5 s, 10 ms).  Its
    /// Python worker runs at 48 kHz before converting the complete window to
    /// 16 kHz, but the resulting internal values are the same as WebUI.
    pub const fn applio() -> Self {
        Self::rvc_webui()
    }
}

const RVC_HOP_SAMPLES: usize = 160;
/// Realtime RMVPE window used by RVC WebUI/Applio. Their `block+800` input is
/// rounded to the 5120-sample extractor window and trimmed by one hop.
pub const RMVPE_REALTIME_SAMPLES: usize = 4_960;
pub const RVC_F0_CACHE_FRAMES: usize = 1_024;
const RVC_F0_LEFT_TRIM: usize = 3;
const RVC_F0_RIGHT_TRIM: usize = 1;

const fn round_up_samples(samples: usize, hop: usize) -> usize {
    if samples == 0 {
        0
    } else {
        ((samples + hop - 1) / hop) * hop
    }
}

impl Default for RvcTrtProfile {
    fn default() -> Self {
        Self::reference()
    }
}

impl RvcTrtProfile {
    /// Build an exact TensorRT profile for one application window.  Fixed
    /// profiles let TensorRT choose tactics and CUDA graphs for the shape that
    /// the host actually submits, while still allowing callers to construct a
    /// wider profile explicitly with [`Self::reference`].
    pub fn for_realtime(config: RvcRealtimeConfig) -> Self {
        Self::from_sample_points([
            config.inference_samples(),
            config.inference_samples(),
            config.inference_samples(),
        ])
    }

    /// Profile covering the three reference applications.  The points are
    /// intentionally the measured application windows, not powers of two:
    /// voice-changer 11200, the legacy WebUI 35520, and current WebUI/Applio
    /// 44960 internal samples.  The middle point is kept as TensorRT's `opt`
    /// shape so all three measured workloads are represented; use an exact
    /// preset for a fixed deployment when one shape should receive the tactic
    /// and CUDA-graph priority.
    pub fn reference() -> Self {
        Self::from_sample_points([
            RvcRealtimeConfig::voice_changer().inference_samples(),
            RvcRealtimeConfig::rvc_webui_legacy().inference_samples(),
            RvcRealtimeConfig::rvc_webui().inference_samples(),
        ])
    }

    pub fn voice_changer() -> Self {
        Self::for_realtime(RvcRealtimeConfig::voice_changer())
    }

    pub fn rvc_webui_legacy() -> Self {
        Self::for_realtime(RvcRealtimeConfig::rvc_webui_legacy())
    }

    pub fn rvc_webui() -> Self {
        Self::for_realtime(RvcRealtimeConfig::rvc_webui())
    }

    pub fn applio() -> Self {
        Self::for_realtime(RvcRealtimeConfig::applio())
    }

    /// Fixed 44960-sample profile used by the current RVC WebUI and Applio
    /// realtime windows. This explicit name is useful for deployment scripts
    /// that do not want to depend on an application label.
    pub fn realtime_44960() -> Self {
        Self::applio()
    }

    fn from_sample_points(samples: [usize; 3]) -> Self {
        let a = frame_shape(samples[0]);
        let o = frame_shape(samples[1]);
        let z = frame_shape(samples[2]);
        Self {
            min_samples: samples[0] as i32,
            opt_samples: samples[1] as i32,
            max_samples: samples[2] as i32,
            min_hubert_frames: a.hubert as i32,
            opt_hubert_frames: o.hubert as i32,
            max_hubert_frames: z.hubert as i32,
            min_pitch_frames: a.pitch as i32,
            opt_pitch_frames: o.pitch as i32,
            max_pitch_frames: z.pitch as i32,
            min_rvc_frames: a.rvc as i32,
            opt_rvc_frames: o.rvc as i32,
            max_rvc_frames: z.rvc as i32,
        }
    }

    /// Previous broad profile retained for compatibility with older serialized
    /// pipeline manifests.  It is not derived from a reference application's
    /// realtime window and should only be selected explicitly.
    pub fn legacy() -> Self {
        Self {
            min_samples: 3200,
            opt_samples: 6400,
            max_samples: 12800,
            min_hubert_frames: 8,
            opt_hubert_frames: 24,
            max_hubert_frames: 64,
            min_pitch_frames: 16,
            opt_pitch_frames: 41,
            max_pitch_frames: 81,
            min_rvc_frames: 8,
            opt_rvc_frames: 48,
            max_rvc_frames: 128,
        }
    }

    /// Select a profile with `VOCAL_TRT_PROFILE`.  Supported values are
    /// `voice_changer`, `rvc_webui_legacy`, `rvc_webui`, `applio`,
    /// `reference` (the default), and `legacy` (the old synthetic range).
    pub fn from_env() -> Self {
        crate::config::TensorRtConfig::from_env()
            .map(|config| config.profile)
            .unwrap_or_else(|_| Self::reference())
    }

    pub fn from_name(value: &str) -> Result<Self, ConfigError> {
        if value.eq_ignore_ascii_case("reference") {
            Ok(Self::reference())
        } else if value.eq_ignore_ascii_case("legacy") {
            Ok(Self::legacy())
        } else if value.eq_ignore_ascii_case("voice_changer")
            || value.eq_ignore_ascii_case("voice-changer")
            || value.eq_ignore_ascii_case("vc")
            || value.eq_ignore_ascii_case("11200")
            || value.eq_ignore_ascii_case("realtime_11200")
            || value.eq_ignore_ascii_case("fixed_11200")
        {
            Ok(Self::voice_changer())
        } else if value.eq_ignore_ascii_case("rvc_webui_legacy") {
            Ok(Self::rvc_webui_legacy())
        } else if value.eq_ignore_ascii_case("rvc_webui")
            || value.eq_ignore_ascii_case("rvc_webui_current")
            || value.eq_ignore_ascii_case("webui")
            || value.eq_ignore_ascii_case("rvc")
            || value.eq_ignore_ascii_case("44960")
            || value.eq_ignore_ascii_case("realtime_44960")
            || value.eq_ignore_ascii_case("fixed_44960")
        {
            Ok(Self::realtime_44960())
        } else if value.eq_ignore_ascii_case("applio") {
            Ok(Self::applio())
        } else {
            Err(ConfigError::InvalidProfile(value.to_owned()))
        }
    }

    pub fn validate(&self) -> Result<(), NativeBuildError> {
        for (name, min, opt, max) in [
            (
                "samples",
                self.min_samples,
                self.opt_samples,
                self.max_samples,
            ),
            (
                "HuBERT frames",
                self.min_hubert_frames,
                self.opt_hubert_frames,
                self.max_hubert_frames,
            ),
            (
                "pitch frames",
                self.min_pitch_frames,
                self.opt_pitch_frames,
                self.max_pitch_frames,
            ),
            (
                "RVC frames",
                self.min_rvc_frames,
                self.opt_rvc_frames,
                self.max_rvc_frames,
            ),
        ] {
            if min <= 0 || min > opt || opt > max {
                return Err(NativeBuildError::Argument(format!(
                    "invalid {name} profile"
                )));
            }
        }
        for (name, samples) in [
            ("minimum", self.min_samples),
            ("optimum", self.opt_samples),
            ("maximum", self.max_samples),
        ] {
            if samples % RVC_HOP_SAMPLES as i32 != 0 {
                return Err(NativeBuildError::Argument(format!(
                    "{name} waveform sample profile {samples} is not aligned to the 160-sample RVC hop"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RvcTrtPipelineArtifacts {
    pub hubert_plan: PathBuf,
    /// Compatibility alias for the full-window RMVPE plan. New manifests also
    /// carry the two fixed-shape plans below.
    pub rmvpe_plan: PathBuf,
    #[serde(default)]
    pub rmvpe_realtime_plan: Option<PathBuf>,
    #[serde(default)]
    pub rmvpe_full_plan: Option<PathBuf>,
    pub frontend_plan: PathBuf,
    pub synth_plan: PathBuf,
    pub profile: RvcTrtProfile,
    /// Typed precision used by the native HuBERT/RMVPE acoustic plans. Their
    /// public waveform/features/pitch bindings remain FP32 so the four-stage
    /// pipeline can reuse one Rust buffer ABI. A non-FP32 Synth plan uses
    /// mixed precision: FP32 encoder/flow and low-precision decoder.
    #[serde(default)]
    pub precision: PrecisionMode,
    #[serde(default)]
    pub hubert_precision: PrecisionMode,
    #[serde(default)]
    pub hubert_feature_precision: PrecisionMode,
    #[serde(default)]
    pub rmvpe_fft_mode: u8,
    /// Whether RMVPE plans consume the external cuFFT magnitude tensor.
    #[serde(default)]
    pub rmvpe_external_stft: bool,
    /// Whether the RMVPE plan exposes parity-only mel/U-Net/GRU/logit
    /// outputs. Runtime binding uses the manifest so a debug plan cannot be
    /// accidentally executed as a production plan (or vice versa).
    #[serde(default)]
    pub rmvpe_debug_outputs: bool,
    #[serde(default)]
    pub rmvpe_precision: PrecisionMode,
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
    #[serde(default)]
    pub synth_resblock_precision: PrecisionMode,
    #[serde(default)]
    pub synth_resblock_stage_mask: u32,
    #[serde(default)]
    pub synth_resblock_conv1_precision: PrecisionMode,
    #[serde(default)]
    pub synth_resblock_conv1_stage_mask: u32,
    #[serde(default)]
    pub synth_resblock_conv2_precision: PrecisionMode,
    #[serde(default)]
    pub synth_resblock_conv2_stage_mask: u32,
    #[serde(default)]
    pub synth_w8a16_conv2_mask: u64,
    #[serde(default)]
    pub synth_w8a8_conv1_mask: u64,
    #[serde(default)]
    pub synth_w8a8_conv2_mask: u64,
    #[serde(default = "crate::options::default_w8a8_activation_scale")]
    pub synth_w8a8_activation_scale: f32,
    #[serde(default)]
    pub synth_fuse_resblock_scale: bool,
    #[serde(default)]
    pub synth_polyphase_upsample: bool,
    #[serde(default)]
    pub synth_full_fp16: bool,
    #[serde(default = "crate::options::default_synth_fp16_input_scale")]
    pub synth_fp16_input_scale: f32,
    #[serde(default)]
    pub synth_fp16_residual_accum: bool,
    #[serde(default)]
    pub synth_fp16_source: bool,
    #[serde(default)]
    pub synth_fp16_noise: bool,
    #[serde(default)]
    pub synth_fp16_noise_stage_mask: u32,
    #[serde(default)]
    pub synth_fp16_coherent_stage_mask: u32,
    #[serde(default = "crate::options::default_synth_fp16_fp32_stage_mask")]
    pub synth_fp16_fp32_stage_mask: u32,
    #[serde(default)]
    pub synth_fp16_post: bool,
    #[serde(default)]
    pub dispatch_tune: DispatchTune,
    #[serde(default = "default_tactic_sources")]
    pub tactic_sources: u32,
    /// Whether the Synth graph was built with experimental static INT8
    /// weight Q/DQ nodes. This is recorded for reproducible benchmarking;
    /// runtime execution does not consult an environment variable.
    #[serde(default)]
    pub weight_int8: bool,
    /// Whether native acoustic and Synth plans used the fused Q/K/V path.
    #[serde(default)]
    pub fused_qkv: bool,
    /// Whether the RVC encoder used the optional fused Q/K/V Conv1d path.
    /// HuBERT fusion is recorded separately in `fused_qkv`.
    #[serde(default)]
    pub fused_rvc_qkv: bool,
}

impl RvcTrtPipelineArtifacts {
    /// Compile all four plans using one semantic configuration object.
    pub fn compile_with_config(
        config: &TensorRtConfig,
        package: &ModelPackage,
        hubert_onnx: impl AsRef<Path>,
        rmvpe_onnx: impl AsRef<Path>,
        output_dir: impl AsRef<Path>,
    ) -> Result<Self, NativeBuildError> {
        let mut build = config.build.clone();
        // Diagnostics are a separate public group, but these three switches
        // are materialized in the native plan and therefore need to reach the
        // builder when a Rust caller sets them directly.
        build.debug_outputs |= config.diagnostics.debug_graph_outputs;
        build.rmvpe_debug_outputs |= config.diagnostics.rmvpe_debug_outputs;
        build.log_build_options |= config.diagnostics.log_build_options;
        Self::compile(
            build,
            package,
            hubert_onnx,
            rmvpe_onnx,
            output_dir,
            config.profile.clone(),
        )
    }

    pub fn compile(
        builder_options: NativeBuildOptions,
        package: &ModelPackage,
        hubert_onnx: impl AsRef<Path>,
        rmvpe_onnx: impl AsRef<Path>,
        output_dir: impl AsRef<Path>,
        profile: RvcTrtProfile,
    ) -> Result<Self, NativeBuildError> {
        profile.validate()?;
        fs::create_dir_all(output_dir.as_ref())?;
        let external_stft = builder_options.rmvpe_fft_cufft;
        let is_vfrm = |path: &Path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    extension.eq_ignore_ascii_case("vfr") || extension.eq_ignore_ascii_case("vfrm")
                })
        };
        let compile = |name: &str,
                       options: NativeBuildOptions|
         -> Result<PathBuf, NativeBuildError> {
            let builder = NativeTensorRtBuilder::new(options);
            let result = match name {
                "hubert" if is_vfrm(hubert_onnx.as_ref()) => {
                    let package = vocal_model::load_cache_shared(hubert_onnx.as_ref())
                        .map_err(|e| NativeBuildError::Api(e.to_string()))?;
                    builder.compile_hubert_vfrm(&package)
                }
                "hubert" => builder
                    .compile_onnx_file_with_profile(hubert_onnx.as_ref(), OnnxProfileKind::Samples),
                "rmvpe" if is_vfrm(rmvpe_onnx.as_ref()) => {
                    let package = vocal_model::load_cache_shared(rmvpe_onnx.as_ref())
                        .map_err(|e| NativeBuildError::Api(e.to_string()))?;
                    builder.compile_rmvpe_vfrm(&package)
                }
                "rmvpe" => {
                    if external_stft {
                        return Err(NativeBuildError::Argument(
                            "cuFFT RMVPE frontend requires an RMVPE VFRM package".into(),
                        ));
                    }
                    builder.compile_onnx_file_with_profile(
                        rmvpe_onnx.as_ref(),
                        OnnxProfileKind::Samples,
                    )
                }
                "frontend" => builder.compile_frontend(),
                "synth" => builder.compile_vfrm(package),
                _ => Err(NativeBuildError::Argument(
                    "unknown RVC TensorRT graph".into(),
                )),
            };
            let stage = match name {
                "hubert" => "build.hubert",
                "rmvpe" => "build.rmvpe",
                "frontend" => "build.frontend",
                "synth" => "build.synth",
                _ => "build.unknown",
            };
            result.map_err(|error| error.with_stage(stage))
        };
        let precision = builder_options.precision;
        let hubert_precision = builder_options.hubert_precision;
        let hubert_feature_precision = builder_options.hubert_feature_precision;
        let rmvpe_fft_mode = if builder_options.rmvpe_fft_radix2 {
            1
        } else if builder_options.rmvpe_fft_cufft {
            2
        } else {
            0
        };
        if builder_options.rmvpe_fft_radix2 && builder_options.rmvpe_fft_cufft {
            return Err(NativeBuildError::Argument(
                "RMVPE FFT mode cannot select both radix2 and cuFFT".into(),
            ));
        }
        let rmvpe_debug_outputs = builder_options.rmvpe_debug_outputs;
        let rmvpe_precision = builder_options.rmvpe_precision;
        let rmvpe_encoder_precision = builder_options.rmvpe_encoder_precision;
        let rmvpe_decoder_precision = builder_options.rmvpe_decoder_precision;
        let rmvpe_stft_precision = builder_options.rmvpe_stft_precision;
        let rmvpe_gru_precision = builder_options.rmvpe_gru_precision;
        let rmvpe_logits_precision = builder_options.rmvpe_logits_precision;
        let synth_precision = builder_options.synth_precision;
        let synth_resblock_precision = builder_options.synth_resblock_precision;
        let synth_resblock_stage_mask = builder_options.synth_resblock_stage_mask;
        let synth_resblock_conv1_precision = builder_options.synth_resblock_conv1_precision;
        let synth_resblock_conv1_stage_mask = builder_options.synth_resblock_conv1_stage_mask;
        let synth_resblock_conv2_precision = builder_options.synth_resblock_conv2_precision;
        let synth_resblock_conv2_stage_mask = builder_options.synth_resblock_conv2_stage_mask;
        let synth_w8a16_conv2_mask = builder_options.synth_w8a16_conv2_mask;
        let synth_w8a8_conv1_mask = builder_options.synth_w8a8_conv1_mask;
        let synth_w8a8_conv2_mask = builder_options.synth_w8a8_conv2_mask;
        let synth_w8a8_activation_scale = builder_options.synth_w8a8_activation_scale;
        let synth_fuse_resblock_scale = builder_options.synth_fuse_resblock_scale;
        let synth_polyphase_upsample = builder_options.synth_polyphase_upsample;
        let synth_full_fp16 = builder_options.synth_full_fp16;
        let synth_fp16_input_scale = builder_options.synth_fp16_input_scale;
        let synth_fp16_residual_accum = builder_options.synth_fp16_residual_accum;
        let synth_fp16_source = builder_options.synth_fp16_source;
        let synth_fp16_noise = builder_options.synth_fp16_noise;
        let synth_fp16_noise_stage_mask = builder_options.synth_fp16_noise_stage_mask;
        let synth_fp16_coherent_stage_mask = builder_options.synth_fp16_coherent_stage_mask;
        let synth_fp16_fp32_stage_mask = builder_options.synth_fp16_fp32_stage_mask;
        let synth_fp16_post = builder_options.synth_fp16_post;
        let dispatch_tune = if builder_options.dispatch_tune == DispatchTune::Auto {
            if profile.min_samples == 11_200 && profile.max_samples == 11_200 {
                DispatchTune::Realtime11200
            } else if profile.min_samples == 44_960 && profile.max_samples == 44_960 {
                DispatchTune::Realtime44960
            } else {
                DispatchTune::Auto
            }
        } else {
            builder_options.dispatch_tune
        };
        let weight_int8 = builder_options.quantize_weights;
        let fused_qkv = builder_options.fused_qkv;
        let fused_rvc_qkv = builder_options.fused_rvc_qkv;
        let mut options = builder_options.clone();
        options.output = output_dir.as_ref().join("hubert.plan");
        options.min_samples = profile.min_samples;
        options.opt_samples = profile.opt_samples;
        options.max_samples = profile.max_samples;
        options.precision = hubert_precision;
        options.hubert_feature_precision = hubert_feature_precision;
        options.dispatch_tune = dispatch_tune;
        let hubert_plan = compile("hubert", options)?;
        // RMVPE has two application shapes. Build each as a fixed profile so
        // TensorRT can select one tactic set and its native CUDA graph without
        // paying a dynamic-shape transition on every callback.
        let mut options = builder_options.clone();
        options.output = output_dir.as_ref().join("rmvpe-realtime.plan");
        options.min_samples = RMVPE_REALTIME_SAMPLES as i32;
        options.opt_samples = RMVPE_REALTIME_SAMPLES as i32;
        options.max_samples = RMVPE_REALTIME_SAMPLES as i32;
        options.precision = rmvpe_precision;
        options.dispatch_tune = dispatch_tune;
        let rmvpe_realtime_plan = compile("rmvpe", options)?;
        let mut options = builder_options.clone();
        options.output = output_dir.as_ref().join("rmvpe-full.plan");
        // Exact application presets use one fixed full shape. The legacy
        // `reference` profile still spans three windows, so retain its broad
        // range for compatibility while the 4960 realtime plan stays fixed.
        options.min_samples = if profile.min_samples == profile.max_samples {
            profile.max_samples
        } else {
            profile.min_samples
        };
        options.opt_samples = if profile.min_samples == profile.max_samples {
            profile.max_samples
        } else {
            profile.opt_samples
        };
        options.max_samples = profile.max_samples;
        options.precision = rmvpe_precision;
        options.dispatch_tune = dispatch_tune;
        let rmvpe_full_plan = compile("rmvpe", options)?;
        let rmvpe_plan = rmvpe_full_plan.clone();
        let mut options = builder_options.clone();
        options.output = output_dir.as_ref().join("frontend.plan");
        options.min_frames = profile.min_hubert_frames.min(profile.min_pitch_frames);
        options.opt_frames = profile.opt_hubert_frames.max(profile.opt_pitch_frames);
        options.max_frames = profile
            .max_hubert_frames
            .max(profile.max_pitch_frames)
            .max(profile.max_rvc_frames);
        options.precision = PrecisionMode::Fp32;
        options.dispatch_tune = dispatch_tune;
        let frontend_plan = compile("frontend", options)?;
        let mut options = builder_options.clone();
        options.output = output_dir.as_ref().join("synth.plan");
        options.min_frames = profile.min_rvc_frames;
        options.opt_frames = profile.opt_rvc_frames;
        options.max_frames = profile.max_rvc_frames;
        // Keep the acoustic encoder/flow precision independent from the
        // decoder. The native ABI carries `synth_precision` separately, so a
        // Synth FP16 plan must not silently lower the RVC front half.
        options.precision = precision;
        options.dispatch_tune = dispatch_tune;
        let synth_plan = compile("synth", options)?;
        Ok(Self {
            hubert_plan,
            rmvpe_plan,
            rmvpe_realtime_plan: Some(rmvpe_realtime_plan),
            rmvpe_full_plan: Some(rmvpe_full_plan),
            frontend_plan,
            synth_plan,
            profile,
            precision,
            hubert_precision,
            hubert_feature_precision,
            rmvpe_fft_mode,
            rmvpe_external_stft: external_stft,
            rmvpe_debug_outputs,
            rmvpe_precision,
            rmvpe_encoder_precision,
            rmvpe_decoder_precision,
            rmvpe_stft_precision,
            rmvpe_gru_precision,
            rmvpe_logits_precision,
            synth_precision,
            synth_resblock_precision,
            synth_resblock_stage_mask,
            synth_resblock_conv1_precision,
            synth_resblock_conv1_stage_mask,
            synth_resblock_conv2_precision,
            synth_resblock_conv2_stage_mask,
            synth_w8a16_conv2_mask,
            synth_w8a8_conv1_mask,
            synth_w8a8_conv2_mask,
            synth_w8a8_activation_scale,
            synth_fuse_resblock_scale,
            synth_polyphase_upsample,
            synth_full_fp16,
            synth_fp16_input_scale,
            synth_fp16_residual_accum,
            synth_fp16_source,
            synth_fp16_noise,
            synth_fp16_noise_stage_mask,
            synth_fp16_coherent_stage_mask,
            synth_fp16_fp32_stage_mask,
            synth_fp16_post,
            dispatch_tune,
            tactic_sources: builder_options.tactic_sources,
            weight_int8,
            fused_qkv,
            fused_rvc_qkv,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RvcTrtTimings {
    pub upload_ms: f64,
    pub hubert_enqueue_ms: f64,
    pub rmvpe_enqueue_ms: f64,
    pub frontend_enqueue_ms: f64,
    pub synth_enqueue_ms: f64,
    pub total_ms: f64,
}

pub struct RvcTrtOutput {
    pub audio: Vec<f32>,
    /// Optional debug snapshots used for parity tests. They are copied after
    /// the final stream synchronization and can be ignored by a live caller.
    pub hubert_features: Option<Vec<f32>>,
    /// Features after the native HuBERT append/interpolation/crop frontend.
    /// This is populated only in debug mode so parity tools can compare the
    /// exact Synth input rather than reconstructing it on the host.
    pub rvc_features: Option<Vec<f32>>,
    pub pitchf: Option<Vec<f32>>,
    /// F0 after the native frontend crop, also only for parity diagnostics.
    pub aligned_pitchf: Option<Vec<f32>>,
    pub pitch: Option<Vec<i32>>,
    /// RMVPE diagnostic tensors emitted by a debug build. These remain on
    /// device in the production graph and are absent unless the plan was
    /// compiled with `VOCAL_TRT_RMVPE_DEBUG_OUTPUTS=1`.
    pub rmvpe_mel: Option<Vec<f32>>,
    pub rmvpe_bottleneck: Option<Vec<f32>>,
    pub rmvpe_unet: Option<Vec<f32>>,
    pub rmvpe_gru: Option<Vec<f32>>,
    pub rmvpe_logits: Option<Vec<f32>>,
    pub hubert_frames: usize,
    pub pitch_frames: usize,
    pub rvc_frames: usize,
    pub timings: RvcTrtTimings,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PipelineShapeKey {
    samples: usize,
    rmvpe_samples: usize,
    hubert_frames: usize,
    pitch_frames: usize,
    frontend_pitch_frames: usize,
    rvc_frames: usize,
    f0_advance_frames: usize,
    streaming_f0: bool,
}

struct StageEvents {
    hubert_start: CudaEvent,
    hubert_end: CudaEvent,
    rmvpe_start: CudaEvent,
    rmvpe_end: CudaEvent,
    frontend_start: CudaEvent,
    frontend_end: CudaEvent,
    synth_start: CudaEvent,
    synth_end: CudaEvent,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct RvcCudaGraphStats {
    #[serde(default)]
    pub mode: GraphReplayMode,
    pub enabled: bool,
    /// Opt-in two-stream HuBERT/RMVPE execution. It is reported so benchmark
    /// JSON cannot be mistaken for the serial graph path.
    #[serde(default)]
    pub parallel_acoustic: bool,
    /// TensorRT-RTX owns capture/replay through IRuntimeConfig. This is the
    /// preferred path on RTX 1.6 and does not increment the application-level
    /// capture/replay counters below.
    #[serde(default)]
    pub native_strategy: bool,
    /// Whether the Rust/CUDA wrapper is using an externally instantiated graph
    /// (the fallback for engines that opt out of TensorRT's native strategy).
    #[serde(default)]
    pub external_strategy: bool,
    /// Shape-specialization transitions observed by the Rust manager while
    /// TensorRT performs its own capture/replay. These are lifecycle counters
    /// (the SDK does not expose its internal graph cache counters).
    #[serde(default)]
    pub native_captures: u64,
    #[serde(default)]
    pub native_replays: u64,
    #[serde(default)]
    pub native_invalidations: u64,
    pub captures: u64,
    pub replays: u64,
    pub invalidations: u64,
    /// Number of executable whole-pipeline graphs currently retained.
    #[serde(default)]
    pub graph_cache_entries: usize,
    /// Configured upper bound for the application-level graph cache.
    #[serde(default)]
    pub graph_cache_capacity: usize,
    /// Cache lookup counters for the application-level graph path.
    #[serde(default)]
    pub graph_cache_hits: u64,
    #[serde(default)]
    pub graph_cache_misses: u64,
    #[serde(default)]
    pub graph_evictions: u64,
}

/// A captured whole-pipeline graph is tied to the dynamic shape key and the
/// fixed device buffer addresses installed during capture. Keeping the key
/// beside the executable lets the fallback path retain both the 11200 and
/// 44960 realtime graphs instead of throwing one away on every transition.
struct ExternalGraphEntry {
    key: PipelineShapeKey,
    exec: CudaGraphExec,
    last_used: u64,
}

struct HybridGraphEntry {
    key: PipelineShapeKey,
    acoustic: Option<CudaGraphExec>,
    decode: Option<CudaGraphExec>,
    last_used: u64,
}

pub struct RvcTrtPipeline {
    pub config: RvcSynthConfig,
    pub profile: RvcTrtProfile,
    hubert: OwnedContext,
    rmvpe_realtime: OwnedContext,
    rmvpe_full: OwnedContext,
    frontend: OwnedContext,
    synth: OwnedContext,
    stream: CudaStream,
    acoustic_hubert_stream: CudaStream,
    acoustic_rmvpe_stream: CudaStream,
    parallel_acoustic: bool,
    runtime_config: PipelineRuntimeConfig,
    diagnostics: DiagnosticsConfig,
    rmvpe_external_stft: bool,
    cufft_frontends: Vec<(usize, CufftFrontend)>,
    stage_events: StageEvents,
    waveform: CudaBuffer,
    waveform_host_ring: PinnedHostRing,
    control_host_ring: PinnedHostRing,
    audio_host_ring: PinnedHostRing,
    threshold: CudaBuffer,
    hubert_features: CudaBuffer,
    pitchf_raw: CudaBuffer,
    rmvpe_magnitude: CudaBuffer,
    p_len: CudaBuffer,
    sid: CudaBuffer,
    feats: CudaBuffer,
    pitch: CudaBuffer,
    pitchf: CudaBuffer,
    rmvpe_debug_mel: CudaBuffer,
    rmvpe_debug_bottleneck: CudaBuffer,
    rmvpe_debug_unet: CudaBuffer,
    rmvpe_debug_gru: CudaBuffer,
    rmvpe_debug_logits: CudaBuffer,
    pitch_cache: CudaBuffer,
    pitch_cache_scratch: CudaBuffer,
    rnd: CudaBuffer,
    nsf_noise: CudaBuffer,
    audio: CudaBuffer,
    // Reused host staging for the deterministic baseline latent/source
    // fixtures. Keeping these vectors across calls removes allocator and
    // per-element byte-conversion work from the realtime hot path.
    host_rnd: Vec<f32>,
    host_noise: Vec<f32>,
    last_sid: Option<i32>,
    last_p_len: Option<i32>,
    debug_intermediates: bool,
    rmvpe_debug_outputs: bool,
    // Resolved output lengths are deterministic for a given profile shape.
    // Keep them on the host so steady callbacks do not query TensorRT's shape
    // machine through FFI on every invocation.
    hubert_frame_shape: Option<(usize, usize)>,
    pitch_frame_shape: Option<(usize, bool, usize)>,
    shape_key: Option<PipelineShapeKey>,
    native_cuda_graphs: bool,
    graph_mode: GraphReplayMode,
    native_graph_shape: Option<PipelineShapeKey>,
    native_graph_captures: u64,
    native_graph_replays: u64,
    native_graph_invalidations: u64,
    cuda_graphs: bool,
    external_graph_capability_checked: bool,
    external_graph_cache: Vec<ExternalGraphEntry>,
    external_graph_cache_capacity: usize,
    external_graph_tick: u64,
    graph_captures: u64,
    graph_replays: u64,
    graph_invalidations: u64,
    graph_cache_hits: u64,
    graph_cache_misses: u64,
    graph_evictions: u64,
    hybrid_cuda_graphs: bool,
    hybrid_graph_capability_checked: bool,
    hybrid_acoustic_capturable: bool,
    hybrid_decode_capturable: bool,
    hybrid_graph_cache: Vec<HybridGraphEntry>,
}

impl RvcTrtPipeline {
    pub fn enable_layer_profiling(&mut self, enabled: bool) {
        self.hubert.enable_profiling(enabled);
        self.rmvpe_realtime.enable_profiling(enabled);
        self.rmvpe_full.enable_profiling(enabled);
        self.frontend.enable_profiling(enabled);
        self.synth.enable_profiling(enabled);
    }

    pub fn layer_timings(
        &mut self,
    ) -> std::collections::BTreeMap<&'static str, Vec<(String, f64)>> {
        std::collections::BTreeMap::from([
            ("hubert", self.hubert.layer_timings()),
            ("rmvpe_realtime", self.rmvpe_realtime.layer_timings()),
            ("rmvpe_full", self.rmvpe_full.layer_timings()),
            ("frontend", self.frontend.layer_timings()),
            ("synth", self.synth.layer_timings()),
        ])
    }

    /// Load using the complete semantic configuration. This is the preferred
    /// embedding API because build, runtime, and diagnostics choices stay
    /// together at the call site.
    pub fn load_with_config(
        artifacts: &RvcTrtPipelineArtifacts,
        model_config: RvcSynthConfig,
        config: &TensorRtConfig,
    ) -> Result<Self, NativeBuildError> {
        Self::load_with_runtime_config(
            artifacts,
            model_config,
            config.runtime.clone(),
            config.diagnostics.clone(),
        )
    }

    /// Compatibility environment adapter for applications that have not yet
    /// migrated to [`TensorRtConfig`].
    pub fn load_from_env(
        artifacts: &RvcTrtPipelineArtifacts,
        model_config: RvcSynthConfig,
    ) -> Result<Self, NativeBuildError> {
        let config = TensorRtConfig::from_env()?;
        Self::load_with_config(artifacts, model_config, &config)
    }

    pub fn load(
        artifacts: &RvcTrtPipelineArtifacts,
        config: RvcSynthConfig,
    ) -> Result<Self, NativeBuildError> {
        Self::load_with_runtime_config(
            artifacts,
            config,
            PipelineRuntimeConfig::default(),
            DiagnosticsConfig::default(),
        )
    }

    /// Load a pipeline with typed runtime and diagnostic policy. Build-time
    /// options remain recorded in the artifact manifest; these values control
    /// only execution and host-side observability.
    pub fn load_with_runtime_config(
        artifacts: &RvcTrtPipelineArtifacts,
        config: RvcSynthConfig,
        runtime_config: PipelineRuntimeConfig,
        diagnostics: DiagnosticsConfig,
    ) -> Result<Self, NativeBuildError> {
        artifacts.profile.validate()?;
        if runtime_config.graph_cache_size == 0 || runtime_config.graph_cache_size > 64 {
            return Err(NativeBuildError::Argument(
                "graph cache size must be in 1..=64".into(),
            ));
        }
        if config.feature_width != 768 {
            return Err(NativeBuildError::Argument(
                "waveform HuBERT TensorRT pipeline requires a 768-wide RVC v2 model".into(),
            ));
        }
        let hubert_runtime = Runtime::shared(&artifacts.hubert_plan)
            .map_err(|error| error.with_stage("load.hubert"))?;
        let rmvpe_realtime_path = artifacts
            .rmvpe_realtime_plan
            .as_ref()
            .unwrap_or(&artifacts.rmvpe_plan);
        let rmvpe_full_path = artifacts
            .rmvpe_full_plan
            .as_ref()
            .unwrap_or(&artifacts.rmvpe_plan);
        let rmvpe_realtime_runtime = Runtime::shared(rmvpe_realtime_path)
            .map_err(|error| error.with_stage("load.rmvpe_realtime"))?;
        let rmvpe_full_runtime = Runtime::shared(rmvpe_full_path)
            .map_err(|error| error.with_stage("load.rmvpe_full"))?;
        let frontend_runtime = Runtime::shared(&artifacts.frontend_plan)
            .map_err(|error| error.with_stage("load.frontend"))?;
        let synth_runtime = Runtime::shared(&artifacts.synth_plan)
            .map_err(|error| error.with_stage("load.synth"))?;
        let p = &artifacts.profile;
        let graph_mode = runtime_config.resolved_graph_mode();
        // TensorRT-RTX's whole-graph strategy performs capture after JIT
        // specialization and owns replay inside enqueueV3. It must be chosen
        // while creating each execution context; changing it after the first
        // enqueue is not supported by the SDK.
        let native_cuda_graphs = graph_mode == GraphReplayMode::NativeSubgraphs;
        let hubert = OwnedContext::create_with_cuda_graphs(hubert_runtime, native_cuda_graphs)?;
        let mut rmvpe_realtime =
            OwnedContext::create_with_cuda_graphs(rmvpe_realtime_runtime, native_cuda_graphs)?;
        let mut rmvpe_full =
            OwnedContext::create_with_cuda_graphs(rmvpe_full_runtime, native_cuda_graphs)?;
        let frontend = OwnedContext::create_with_cuda_graphs(frontend_runtime, native_cuda_graphs)?;
        let synth = OwnedContext::create_with_cuda_graphs(synth_runtime, native_cuda_graphs)?;
        let stream = CudaStream::new()?;
        let acoustic_hubert_stream = CudaStream::new()?;
        let acoustic_rmvpe_stream = CudaStream::new()?;
        // The external path remains requested until the first concrete shape
        // and all tensor bindings have been installed.  The parallel path is
        // deliberately restricted to TensorRT-RTX's native graph strategy;
        // a host-side CUDA stream capture cannot contain the synchronization
        // between the two acoustic streams.
        let external_cuda_graphs = graph_mode == GraphReplayMode::WholePipeline;
        let hybrid_cuda_graphs = graph_mode == GraphReplayMode::HybridSubgraphs;
        let parallel_acoustic =
            native_cuda_graphs && !external_cuda_graphs && runtime_config.parallel_acoustic;
        let rmvpe_external_stft = artifacts.rmvpe_external_stft;
        let graph_cache_capacity = runtime_config.graph_cache_size.max(1);
        let mut cufft_frontends = Vec::new();
        if rmvpe_external_stft {
            let shapes = unique_shapes([
                RMVPE_REALTIME_SAMPLES,
                p.min_samples as usize,
                p.opt_samples as usize,
                p.max_samples as usize,
            ]);
            for samples in shapes {
                let cufft_stream = if parallel_acoustic {
                    &acoustic_rmvpe_stream
                } else {
                    &stream
                };
                cufft_frontends.push((samples, CufftFrontend::new(samples, cufft_stream)?));
            }
        }
        // `isStreamCapturable` is queried lazily in `process_inner` because
        // dynamic TensorRT shapes are not concrete at context creation.
        let stage_events = StageEvents {
            hubert_start: CudaEvent::new()?,
            hubert_end: CudaEvent::new()?,
            rmvpe_start: CudaEvent::new()?,
            rmvpe_end: CudaEvent::new()?,
            frontend_start: CudaEvent::new()?,
            frontend_end: CudaEvent::new()?,
            synth_start: CudaEvent::new()?,
            synth_end: CudaEvent::new()?,
        };
        let latent = config.latent_channels as usize;
        let up = config
            .upsample_rates
            .iter()
            .try_fold(1usize, |v, r| v.checked_mul(*r as usize))
            .ok_or_else(|| NativeBuildError::Argument("upsample product overflow".into()))?;
        let alloc = |bytes: usize| CudaBuffer::new(bytes.max(4));
        // These capacities cover the largest shipped realtime profile. The
        // RMVPE U-Net reflects mel time to a multiple of 32 (44960 -> 288),
        // so its debug bottleneck/GRU/logit outputs use the padded length even
        // though public pitch is cropped to 282 frames. Buffers are allocated
        // once so parity diagnostics never introduce an undersized output
        // binding or a per-call device allocation.
        // Older pipeline manifests did not persist the debug-output flag.
        // Infer it from the loaded execution context so such plans still
        // receive an address for every declared output. New manifests keep
        // the explicit flag, while production plans without these tensors
        // simply return an API error from the probe and remain unchanged.
        let plan_has_rmvpe_debug_outputs = rmvpe_full.tensor_shape("rmvpe_debug_mel").is_ok()
            || rmvpe_realtime.tensor_shape("rmvpe_debug_mel").is_ok();
        let rmvpe_debug_outputs = artifacts.rmvpe_debug_outputs || plan_has_rmvpe_debug_outputs;
        let debug_intermediates = diagnostics.debug_intermediates;
        let rmvpe_max_padded_frames = ((p.max_pitch_frames as usize + 31) / 32) * 32;
        let pitch_cache = alloc(RVC_F0_CACHE_FRAMES * 4)?;
        let pitch_cache_scratch = alloc(RVC_F0_CACHE_FRAMES * 4)?;
        pitch_cache.zero()?;
        pitch_cache_scratch.zero()?;
        let mut pipeline = Self {
            config,
            profile: p.clone(),
            hubert,
            rmvpe_realtime,
            rmvpe_full,
            frontend,
            synth,
            stream,
            acoustic_hubert_stream,
            acoustic_rmvpe_stream,
            parallel_acoustic,
            runtime_config,
            diagnostics,
            rmvpe_external_stft,
            cufft_frontends,
            stage_events,
            waveform: alloc(p.max_samples as usize * 4)?,
            waveform_host_ring: PinnedHostRing::new(3, p.max_samples as usize * 4)?,
            control_host_ring: PinnedHostRing::new(3, std::mem::size_of::<i32>())?,
            audio_host_ring: PinnedHostRing::new(3, p.max_rvc_frames as usize * up * 4)?,
            threshold: alloc(4)?,
            hubert_features: alloc(p.max_hubert_frames as usize * 768 * 4)?,
            pitchf_raw: alloc(p.max_pitch_frames as usize * 4)?,
            rmvpe_magnitude: alloc(p.max_pitch_frames as usize * 513 * 4)?,
            p_len: alloc(4)?,
            sid: alloc(4)?,
            feats: alloc(p.max_rvc_frames as usize * 768 * 4)?,
            pitch: alloc(p.max_rvc_frames as usize * 4)?,
            pitchf: alloc(p.max_rvc_frames as usize * 4)?,
            rmvpe_debug_mel: alloc(p.max_pitch_frames as usize * 128 * 4)?,
            rmvpe_debug_bottleneck: alloc(512 * (rmvpe_max_padded_frames / 32) * 4 * 4)?,
            rmvpe_debug_unet: alloc(rmvpe_max_padded_frames * 16 * 128 * 4)?,
            rmvpe_debug_gru: alloc(rmvpe_max_padded_frames * 512 * 4)?,
            rmvpe_debug_logits: alloc(rmvpe_max_padded_frames * 360 * 4)?,
            pitch_cache,
            pitch_cache_scratch,
            rnd: alloc(latent * p.max_rvc_frames as usize * 4)?,
            nsf_noise: alloc(p.max_rvc_frames as usize * up * 4)?,
            audio: alloc(p.max_rvc_frames as usize * up * 4)?,
            host_rnd: Vec::with_capacity(latent * p.max_rvc_frames as usize),
            host_noise: Vec::with_capacity(p.max_rvc_frames as usize * up),
            last_sid: None,
            last_p_len: None,
            debug_intermediates,
            rmvpe_debug_outputs,
            hubert_frame_shape: None,
            pitch_frame_shape: None,
            shape_key: None,
            native_cuda_graphs,
            graph_mode,
            native_graph_shape: None,
            native_graph_captures: 0,
            native_graph_replays: 0,
            native_graph_invalidations: 0,
            cuda_graphs: external_cuda_graphs,
            external_graph_capability_checked: false,
            external_graph_cache: Vec::with_capacity(graph_cache_capacity),
            external_graph_cache_capacity: graph_cache_capacity,
            external_graph_tick: 0,
            graph_captures: 0,
            graph_replays: 0,
            graph_invalidations: 0,
            graph_cache_hits: 0,
            graph_cache_misses: 0,
            graph_evictions: 0,
            hybrid_cuda_graphs,
            hybrid_graph_capability_checked: false,
            hybrid_acoustic_capturable: false,
            hybrid_decode_capturable: false,
            hybrid_graph_cache: Vec::with_capacity(graph_cache_capacity),
        };
        pipeline.seed_static_inputs(latent, up)?;
        pipeline.threshold.upload_f32(&[0.03_f32])?;
        Ok(pipeline)
    }

    /// Seed the deterministic baseline latent/source buffers once. Every
    /// runtime shape reads a prefix of these max-profile allocations, so no
    /// per-block host generation or upload is needed during graph replay.
    fn seed_static_inputs(&mut self, latent: usize, up: usize) -> Result<(), NativeBuildError> {
        self.host_rnd.clear();
        self.host_rnd
            .reserve(latent * self.profile.max_rvc_frames as usize);
        for i in 0..(latent * self.profile.max_rvc_frames as usize) {
            self.host_rnd.push(0.01_f32 * ((i as f32) * 0.017).sin());
        }
        self.host_noise.clear();
        self.host_noise
            .reserve(self.profile.max_rvc_frames as usize * up);
        for i in 0..(self.profile.max_rvc_frames as usize * up) {
            self.host_noise.push(0.001_f32 * ((i as f32) * 0.013).sin());
        }
        self.rnd.upload_f32(&self.host_rnd)?;
        self.nsf_noise.upload_f32(&self.host_noise)?;
        Ok(())
    }

    /// Enable intermediate downloads for parity/debug reports. Keep this off
    /// in a live converter so only the final waveform crosses D2H.
    pub fn enable_debug_intermediates(&mut self, enabled: bool) {
        self.debug_intermediates = enabled;
    }

    /// Enable or disable application-level CUDA Graph replay. Changing this
    /// setting invalidates the current executable and leaves the TensorRT
    /// eager path available for the next call.
    pub fn enable_cuda_graphs(&mut self, enabled: bool) {
        let external_enabled = enabled && self.graph_mode == GraphReplayMode::WholePipeline;
        let hybrid_enabled = enabled && self.graph_mode == GraphReplayMode::HybridSubgraphs;
        if self.cuda_graphs != external_enabled || self.hybrid_cuda_graphs != hybrid_enabled {
            self.external_graph_cache.clear();
            self.hybrid_graph_cache.clear();
            self.graph_invalidations = self.graph_invalidations.saturating_add(1);
            self.external_graph_capability_checked = false;
            self.hybrid_graph_capability_checked = false;
            self.hybrid_acoustic_capturable = false;
            self.hybrid_decode_capturable = false;
        }
        // Native TensorRT graph mode is selected at context creation and is
        // intentionally immutable. This setter controls only application-level
        // whole/hybrid replay; callers choose the mode before loading through
        // `PipelineRuntimeConfig::graph_mode`.
        self.cuda_graphs = external_enabled;
        self.hybrid_cuda_graphs = hybrid_enabled;
    }

    pub fn cuda_graph_stats(&self) -> RvcCudaGraphStats {
        RvcCudaGraphStats {
            mode: self.graph_mode,
            enabled: self.native_cuda_graphs || self.cuda_graphs || self.hybrid_cuda_graphs,
            parallel_acoustic: self.parallel_acoustic,
            native_strategy: self.native_cuda_graphs,
            external_strategy: self.cuda_graphs || self.hybrid_cuda_graphs,
            native_captures: self.native_graph_captures,
            native_replays: self.native_graph_replays,
            native_invalidations: self.native_graph_invalidations,
            captures: self.graph_captures,
            replays: self.graph_replays,
            invalidations: self.graph_invalidations,
            graph_cache_entries: self.external_graph_cache.len() + self.hybrid_graph_cache.len(),
            graph_cache_capacity: self.external_graph_cache_capacity,
            graph_cache_hits: self.graph_cache_hits,
            graph_cache_misses: self.graph_cache_misses,
            graph_evictions: self.graph_evictions,
        }
    }

    /// Reset the streaming F0 cache and the graph executable. This mirrors the
    /// reset performed by the reference realtime workers after a device/model
    /// change and prevents stale pitch context crossing sessions.
    pub fn reset_streaming_state(&mut self) -> Result<(), NativeBuildError> {
        self.pitch_cache.zero()?;
        self.pitch_cache_scratch.zero()?;
        self.last_sid = None;
        self.last_p_len = None;
        self.shape_key = None;
        self.native_graph_shape = None;
        self.native_graph_invalidations = self.native_graph_invalidations.saturating_add(1);
        self.external_graph_cache.clear();
        self.hybrid_graph_cache.clear();
        self.external_graph_capability_checked = false;
        self.hybrid_graph_capability_checked = false;
        self.hybrid_acoustic_capturable = false;
        self.hybrid_decode_capturable = false;
        self.graph_invalidations = self.graph_invalidations.saturating_add(1);
        Ok(())
    }

    /// Run one complete waveform window through HuBERT, RMVPE, frontend, and
    /// Synth. `waveform` is the entire inference window in internal 16 kHz
    /// samples; this method currently has no chunk crop, overlap crossfade,
    /// or SOLA alignment stage.
    pub fn process(
        &mut self,
        waveform: &[f32],
        sid: i32,
    ) -> Result<RvcTrtOutput, NativeBuildError> {
        self.process_inner(waveform, sid, None)
    }

    /// Realtime entry point matching RVC WebUI/Applio's stateful worker. The
    /// full HuBERT window is retained, while RMVPE consumes the fixed recent
    /// context and its 3:-1 output is appended to a 1024-frame device cache.
    /// `advance_frames` is the number of 10 ms RVC hops represented by the new
    /// callback block.
    pub fn process_realtime(
        &mut self,
        waveform: &[f32],
        sid: i32,
        advance_frames: usize,
    ) -> Result<RvcTrtOutput, NativeBuildError> {
        if advance_frames == 0 || advance_frames > RVC_F0_CACHE_FRAMES {
            return Err(NativeBuildError::Argument(
                "realtime F0 advance must be in 1..=1024 frames".into(),
            ));
        }
        self.process_inner(waveform, sid, Some(advance_frames))
    }

    fn enqueue_stage_sequence(
        &mut self,
        streaming_f0: bool,
        pitch_frames: usize,
        advance_frames: usize,
        rmvpe_samples: usize,
        rmvpe_waveform: *const std::ffi::c_void,
    ) -> Result<(), NativeBuildError> {
        if self.parallel_acoustic {
            return self.enqueue_parallel_acoustic(
                streaming_f0,
                pitch_frames,
                advance_frames,
                rmvpe_samples,
                rmvpe_waveform,
            );
        }
        self.enqueue_acoustic_sequence(
            streaming_f0,
            pitch_frames,
            advance_frames,
            rmvpe_samples,
            rmvpe_waveform,
        )?;
        self.enqueue_decode_sequence()
    }

    fn enqueue_acoustic_sequence(
        &mut self,
        streaming_f0: bool,
        pitch_frames: usize,
        advance_frames: usize,
        rmvpe_samples: usize,
        rmvpe_waveform: *const std::ffi::c_void,
    ) -> Result<(), NativeBuildError> {
        let trace = self.diagnostics.trace;
        self.stage_events.hubert_start.record(&self.stream)?;
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: enqueue hubert");
        }
        self.hubert
            .enqueue_named(&self.stream, Some("rvc.hubert"))?;
        self.stage_events.hubert_end.record(&self.stream)?;
        self.stage_events.rmvpe_start.record(&self.stream)?;
        if self.rmvpe_external_stft {
            let frontend = self
                .cufft_frontends
                .iter_mut()
                .find(|(samples, _)| *samples == rmvpe_samples)
                .map(|(_, frontend)| frontend)
                .ok_or_else(|| {
                    NativeBuildError::Argument(format!(
                        "missing cuFFT frontend plan for {rmvpe_samples} samples"
                    ))
                })?;
            frontend.enqueue(
                rmvpe_waveform,
                rmvpe_samples * std::mem::size_of::<f32>(),
                &self.rmvpe_magnitude,
                &self.stream,
            )?;
        }
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: enqueue rmvpe streaming={streaming_f0}");
        }
        if streaming_f0 {
            self.rmvpe_realtime
                .enqueue_named(&self.stream, Some("rvc.rmvpe.realtime"))?;
        } else {
            self.rmvpe_full
                .enqueue_named(&self.stream, Some("rvc.rmvpe.full"))?;
        }
        self.stage_events.rmvpe_end.record(&self.stream)?;
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: rmvpe enqueued");
        }
        if streaming_f0 {
            let append_frames = pitch_frames
                .saturating_sub(RVC_F0_LEFT_TRIM + RVC_F0_RIGHT_TRIM)
                .min(RVC_F0_CACHE_FRAMES);
            if append_frames == 0 || advance_frames >= RVC_F0_CACHE_FRAMES {
                return Err(NativeBuildError::Argument(
                    "RMVPE streaming output/cache geometry is invalid".into(),
                ));
            }
            let retained = RVC_F0_CACHE_FRAMES - advance_frames;
            self.pitch_cache_scratch.device_copy_from(
                &self.pitch_cache,
                0,
                advance_frames * std::mem::size_of::<f32>(),
                retained * std::mem::size_of::<f32>(),
                &self.stream,
            )?;
            self.pitch_cache_scratch.device_copy_from(
                &self.pitchf_raw,
                (RVC_F0_CACHE_FRAMES - append_frames) * std::mem::size_of::<f32>(),
                RVC_F0_LEFT_TRIM * std::mem::size_of::<f32>(),
                append_frames * std::mem::size_of::<f32>(),
                &self.stream,
            )?;
            self.pitch_cache.device_copy_from(
                &self.pitch_cache_scratch,
                0,
                0,
                RVC_F0_CACHE_FRAMES * std::mem::size_of::<f32>(),
                &self.stream,
            )?;
        }
        Ok(())
    }

    fn enqueue_decode_sequence(&mut self) -> Result<(), NativeBuildError> {
        let trace = self.diagnostics.trace;
        self.stage_events.frontend_start.record(&self.stream)?;
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: enqueue frontend");
        }
        self.frontend
            .enqueue_named(&self.stream, Some("rvc.frontend"))?;
        self.stage_events.frontend_end.record(&self.stream)?;
        self.stage_events.synth_start.record(&self.stream)?;
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: enqueue synth");
        }
        self.synth.enqueue_named(&self.stream, Some("rvc.synth"))?;
        self.stage_events.synth_end.record(&self.stream)?;
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: all enqueued");
        }
        Ok(())
    }

    /// Run HuBERT and RMVPE on independent streams after the waveform upload
    /// has completed.  Their outputs are consumed only after both streams are
    /// synchronized, so frontend/Synth retain the same ordering and bindings
    /// as the serial path.  This is opt-in because TensorRT-RTX tactic and GPU
    /// clock behavior can make the overlap workload dependent.
    fn enqueue_parallel_acoustic(
        &mut self,
        streaming_f0: bool,
        pitch_frames: usize,
        advance_frames: usize,
        rmvpe_samples: usize,
        rmvpe_waveform: *const std::ffi::c_void,
    ) -> Result<(), NativeBuildError> {
        let trace = self.diagnostics.trace;
        // The waveform upload is queued on the owner stream.  A host-side
        // synchronization is enough here: it occurs once per callback and
        // avoids introducing a cross-stream event into the native graph path.
        self.stream.synchronize()?;
        self.stage_events
            .hubert_start
            .record(&self.acoustic_hubert_stream)?;
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: enqueue hubert parallel");
        }
        self.hubert
            .enqueue_named(&self.acoustic_hubert_stream, Some("rvc.hubert"))?;
        self.stage_events
            .hubert_end
            .record(&self.acoustic_hubert_stream)?;

        self.stage_events
            .rmvpe_start
            .record(&self.acoustic_rmvpe_stream)?;
        if self.rmvpe_external_stft {
            let frontend = self
                .cufft_frontends
                .iter_mut()
                .find(|(samples, _)| *samples == rmvpe_samples)
                .map(|(_, frontend)| frontend)
                .ok_or_else(|| {
                    NativeBuildError::Argument(format!(
                        "missing cuFFT frontend plan for {rmvpe_samples} samples"
                    ))
                })?;
            frontend.enqueue(
                rmvpe_waveform,
                rmvpe_samples * std::mem::size_of::<f32>(),
                &self.rmvpe_magnitude,
                &self.acoustic_rmvpe_stream,
            )?;
        }
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: enqueue rmvpe parallel streaming={streaming_f0}");
        }
        if streaming_f0 {
            self.rmvpe_realtime
                .enqueue_named(&self.acoustic_rmvpe_stream, Some("rvc.rmvpe.realtime"))?;
        } else {
            self.rmvpe_full
                .enqueue_named(&self.acoustic_rmvpe_stream, Some("rvc.rmvpe.full"))?;
        }
        if streaming_f0 {
            let append_frames = pitch_frames
                .saturating_sub(RVC_F0_LEFT_TRIM + RVC_F0_RIGHT_TRIM)
                .min(RVC_F0_CACHE_FRAMES);
            if append_frames == 0 || advance_frames >= RVC_F0_CACHE_FRAMES {
                return Err(NativeBuildError::Argument(
                    "RMVPE streaming output/cache geometry is invalid".into(),
                ));
            }
            let retained = RVC_F0_CACHE_FRAMES - advance_frames;
            self.pitch_cache_scratch.device_copy_from(
                &self.pitch_cache,
                0,
                advance_frames * std::mem::size_of::<f32>(),
                retained * std::mem::size_of::<f32>(),
                &self.acoustic_rmvpe_stream,
            )?;
            self.pitch_cache_scratch.device_copy_from(
                &self.pitchf_raw,
                (RVC_F0_CACHE_FRAMES - append_frames) * std::mem::size_of::<f32>(),
                RVC_F0_LEFT_TRIM * std::mem::size_of::<f32>(),
                append_frames * std::mem::size_of::<f32>(),
                &self.acoustic_rmvpe_stream,
            )?;
            self.pitch_cache.device_copy_from(
                &self.pitch_cache_scratch,
                0,
                0,
                RVC_F0_CACHE_FRAMES * std::mem::size_of::<f32>(),
                &self.acoustic_rmvpe_stream,
            )?;
        }
        self.stage_events
            .rmvpe_end
            .record(&self.acoustic_rmvpe_stream)?;
        self.acoustic_hubert_stream.synchronize()?;
        self.acoustic_rmvpe_stream.synchronize()?;

        self.stage_events.frontend_start.record(&self.stream)?;
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: enqueue frontend after parallel acoustic");
        }
        self.frontend
            .enqueue_named(&self.stream, Some("rvc.frontend"))?;
        self.stage_events.frontend_end.record(&self.stream)?;
        self.stage_events.synth_start.record(&self.stream)?;
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: enqueue synth after parallel acoustic");
        }
        self.synth.enqueue_named(&self.stream, Some("rvc.synth"))?;
        self.stage_events.synth_end.record(&self.stream)?;
        Ok(())
    }

    /// Capture one serial sequence and return its executable when supported.
    /// Capture itself does not execute device work. The RAII guard aborts an
    /// unfinished capture, leaving the stream safe for an eager fallback.
    fn capture_sequence<F>(&mut self, enqueue: F) -> Result<Option<CudaGraphExec>, NativeBuildError>
    where
        F: FnOnce(&mut Self) -> Result<(), NativeBuildError>,
    {
        let capture = match self.stream.begin_capture() {
            Ok(capture) => capture,
            Err(_) => return Ok(None),
        };
        match enqueue(self) {
            Ok(()) => match capture.finish() {
                Ok(graph) => Ok(Some(graph)),
                Err(_) => Ok(None),
            },
            Err(_) => {
                // The RAII guard aborts the active capture. No device work has
                // executed, so the caller can safely enqueue the same sequence
                // eagerly or capture the other hybrid subgraph.
                drop(capture);
                Ok(None)
            }
        }
    }

    /// Capture and launch the two serial hybrid subgraphs. A failed optional
    /// capture leaves the other subgraph available, so HybridSubgraphs can
    /// still replay one graph and enqueue the other eagerly.
    fn capture_hybrid_graph(
        &mut self,
        graph_key: PipelineShapeKey,
        acoustic_capturable: bool,
        decode_capturable: bool,
        streaming_f0: bool,
        pitch_frames: usize,
        advance_frames: usize,
        rmvpe_samples: usize,
        rmvpe_waveform: *const std::ffi::c_void,
    ) -> Result<bool, NativeBuildError> {
        let acoustic = if acoustic_capturable {
            self.capture_sequence(|pipeline| {
                pipeline.enqueue_acoustic_sequence(
                    streaming_f0,
                    pitch_frames,
                    advance_frames,
                    rmvpe_samples,
                    rmvpe_waveform,
                )
            })?
        } else {
            None
        };
        let decode = if decode_capturable {
            self.capture_sequence(|pipeline| pipeline.enqueue_decode_sequence())?
        } else {
            None
        };
        if acoustic.is_none() && decode.is_none() {
            return Ok(false);
        }
        let captured_count = u64::from(acoustic.is_some()) + u64::from(decode.is_some());

        if self.hybrid_graph_cache.len() >= self.external_graph_cache_capacity {
            let oldest = self
                .hybrid_graph_cache
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(index, _)| index)
                .expect("non-empty hybrid graph cache at capacity");
            self.hybrid_graph_cache.remove(oldest);
            self.graph_evictions = self.graph_evictions.saturating_add(1);
        }
        self.hybrid_graph_cache.push(HybridGraphEntry {
            key: graph_key,
            acoustic,
            decode,
            last_used: self.external_graph_tick,
        });
        self.graph_captures = self.graph_captures.saturating_add(captured_count);
        let acoustic_cached = self
            .hybrid_graph_cache
            .last()
            .and_then(|entry| entry.acoustic.as_ref())
            .is_some();
        if acoustic_cached {
            self.hybrid_graph_cache
                .last()
                .and_then(|entry| entry.acoustic.as_ref())
                .expect("hybrid acoustic graph is present")
                .launch(&self.stream)?;
        } else {
            self.enqueue_acoustic_sequence(
                streaming_f0,
                pitch_frames,
                advance_frames,
                rmvpe_samples,
                rmvpe_waveform,
            )?;
        }
        let decode_cached = self
            .hybrid_graph_cache
            .last()
            .and_then(|entry| entry.decode.as_ref())
            .is_some();
        if decode_cached {
            self.hybrid_graph_cache
                .last()
                .and_then(|entry| entry.decode.as_ref())
                .expect("hybrid decode graph is present")
                .launch(&self.stream)?;
        } else {
            self.enqueue_decode_sequence()?;
        }
        self.graph_replays = self
            .graph_replays
            .saturating_add(u64::from(acoustic_cached) + u64::from(decode_cached));
        Ok(true)
    }

    fn process_inner(
        &mut self,
        waveform: &[f32],
        sid: i32,
        realtime_advance_frames: Option<usize>,
    ) -> Result<RvcTrtOutput, NativeBuildError> {
        let trace = self.diagnostics.trace;
        let start = Instant::now();
        let p = &self.profile;
        if waveform.len() < p.min_samples as usize || waveform.len() > p.max_samples as usize {
            return Err(NativeBuildError::Argument(format!(
                "waveform length {} outside [{}..={}]",
                waveform.len(),
                p.min_samples,
                p.max_samples
            )));
        }
        self.waveform_host_ring
            .upload_f32(&self.waveform, waveform, &self.stream)?;
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: waveform uploaded samples={}", waveform.len());
        }
        let upload_ms = start.elapsed().as_secs_f64() * 1000.0;
        let streaming_f0 = realtime_advance_frames.is_some() && self.config.uses_f0;
        let rmvpe_samples = if streaming_f0 {
            if waveform.len() < RMVPE_REALTIME_SAMPLES {
                return Err(NativeBuildError::Argument(format!(
                    "realtime waveform length {} is shorter than RMVPE context {}",
                    waveform.len(),
                    RMVPE_REALTIME_SAMPLES
                )));
            }
            RMVPE_REALTIME_SAMPLES
        } else {
            waveform.len()
        };
        let rmvpe_waveform = self
            .waveform
            .device_ptr_offset((waveform.len() - rmvpe_samples) * std::mem::size_of::<f32>());
        let advance_frames = realtime_advance_frames.unwrap_or(0);
        let same_waveform = self
            .shape_key
            .is_some_and(|key| key.samples == waveform.len());
        let same_rmvpe = self.shape_key.is_some_and(|key| {
            key.rmvpe_samples == rmvpe_samples && key.streaming_f0 == streaming_f0
        });
        if !same_waveform {
            self.hubert
                .set_input_shape("waveform", &[1, waveform.len() as i64])?;
        }
        self.hubert
            .set_tensor_address_cached("waveform", self.waveform.device_ptr())?;
        self.hubert
            .set_tensor_address_cached("features", self.hubert_features.device_ptr())?;
        let hubert_frames = if let Some((samples, frames)) = self.hubert_frame_shape {
            if samples == waveform.len() {
                frames
            } else {
                let hubert_shape = self.hubert.tensor_shape("features")?;
                let frames =
                    concrete_dim(&hubert_shape, 1, hubert_frames_for_samples(waveform.len()))?;
                self.hubert_frame_shape = Some((waveform.len(), frames));
                frames
            }
        } else {
            let hubert_shape = self.hubert.tensor_shape("features")?;
            let frames = concrete_dim(&hubert_shape, 1, hubert_frames_for_samples(waveform.len()))?;
            self.hubert_frame_shape = Some((waveform.len(), frames));
            frames
        };
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: hubert frames={hubert_frames}");
        }

        let rmvpe = if streaming_f0 {
            &mut self.rmvpe_realtime
        } else {
            &mut self.rmvpe_full
        };
        if !same_rmvpe {
            if self.rmvpe_external_stft {
                rmvpe.set_input_shape(
                    "stft_magnitude",
                    &[1, 513, (rmvpe_samples / 160 + 1) as i64],
                )?;
            } else {
                rmvpe.set_input_shape("waveform", &[1, rmvpe_samples as i64])?;
            }
        }
        if self.rmvpe_external_stft {
            rmvpe.set_tensor_address_cached("stft_magnitude", self.rmvpe_magnitude.device_ptr())?;
        } else {
            rmvpe.set_tensor_address_cached("waveform", rmvpe_waveform)?;
        }
        rmvpe.set_tensor_address_cached("threshold", self.threshold.device_ptr())?;
        rmvpe.set_tensor_address_cached("pitchf", self.pitchf_raw.device_ptr())?;
        if self.rmvpe_debug_outputs {
            rmvpe
                .set_tensor_address_cached("rmvpe_debug_mel", self.rmvpe_debug_mel.device_ptr())?;
            rmvpe.set_tensor_address_cached(
                "rmvpe_debug_bottleneck",
                self.rmvpe_debug_bottleneck.device_ptr(),
            )?;
            rmvpe.set_tensor_address_cached(
                "rmvpe_debug_unet",
                self.rmvpe_debug_unet.device_ptr(),
            )?;
            rmvpe
                .set_tensor_address_cached("rmvpe_debug_gru", self.rmvpe_debug_gru.device_ptr())?;
            rmvpe.set_tensor_address_cached(
                "rmvpe_debug_logits",
                self.rmvpe_debug_logits.device_ptr(),
            )?;
            if trace {
                tracing::trace!(target: "vocal_tensorrt",
                    "trt trace: rmvpe debug shapes mel={:?} bottleneck={:?} unet={:?} gru={:?} logits={:?}",
                    rmvpe.tensor_shape("rmvpe_debug_mel")?,
                    rmvpe.tensor_shape("rmvpe_debug_bottleneck")?,
                    rmvpe.tensor_shape("rmvpe_debug_unet")?,
                    rmvpe.tensor_shape("rmvpe_debug_gru")?,
                    rmvpe.tensor_shape("rmvpe_debug_logits")?,
                );
            }
        }
        let pitch_frames = if let Some((samples, streaming, frames)) = self.pitch_frame_shape {
            if samples == rmvpe_samples && streaming == streaming_f0 {
                frames
            } else {
                let pitch_shape = rmvpe.tensor_shape("pitchf")?;
                let frames = concrete_dim(&pitch_shape, 1, rmvpe_samples / 160 + 1)?;
                self.pitch_frame_shape = Some((rmvpe_samples, streaming_f0, frames));
                frames
            }
        } else {
            let pitch_shape = rmvpe.tensor_shape("pitchf")?;
            let frames = concrete_dim(&pitch_shape, 1, rmvpe_samples / 160 + 1)?;
            self.pitch_frame_shape = Some((rmvpe_samples, streaming_f0, frames));
            frames
        };
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: rmvpe frames={pitch_frames}");
        }

        let p_len_frames = waveform.len() / RVC_HOP_SAMPLES;
        let rvc_frames = if streaming_f0 {
            p_len_frames.min((hubert_frames + 1) * 2)
        } else {
            p_len_frames.min((hubert_frames + 1) * 2).min(pitch_frames)
        };
        if rvc_frames < p.min_rvc_frames as usize || rvc_frames > p.max_rvc_frames as usize {
            return Err(NativeBuildError::Argument(format!(
                "aligned RVC frame count {rvc_frames} outside profile"
            )));
        }
        // Bind the RVC-valid prefix of F0 to the frontend in both modes.
        // RMVPE produces one boundary frame beyond p_len for the full window
        // (282 vs 281 at 44960); using the prefix makes the crop explicit in
        // the execution shape and avoids an extra dynamic shape input.
        let frontend_pitch_frames = rvc_frames;
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: frontend rvc_frames={rvc_frames}");
        }
        let same_shape = self.shape_key.is_some_and(|key| {
            key.samples == waveform.len()
                && key.rmvpe_samples == rmvpe_samples
                && key.hubert_frames == hubert_frames
                && key.pitch_frames == pitch_frames
                && key.frontend_pitch_frames == frontend_pitch_frames
                && key.rvc_frames == rvc_frames
                && key.f0_advance_frames == advance_frames
                && key.streaming_f0 == streaming_f0
        });
        if !same_shape {
            self.frontend
                .set_input_shape("hubert_features", &[1, hubert_frames as i64, 768])?;
            self.frontend
                .set_input_shape("pitchf_in", &[1, frontend_pitch_frames as i64])?;
        }
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: frontend input shapes set");
        }
        for (name, buffer) in [
            ("hubert_features", &self.hubert_features),
            ("pitchf_in", &self.pitchf_raw),
            ("feats", &self.feats),
            ("pitch", &self.pitch),
            ("pitchf", &self.pitchf),
        ] {
            self.frontend
                .set_tensor_address_cached(name, buffer.device_ptr())?;
        }
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: frontend addresses set");
        }
        let pitchf_input = if streaming_f0 {
            self.pitch_cache
                .device_ptr_offset((RVC_F0_CACHE_FRAMES - rvc_frames) * 4)
        } else {
            self.pitchf_raw.device_ptr()
        };
        self.frontend
            .set_tensor_address_cached("pitchf_in", pitchf_input)?;

        let up = self
            .config
            .upsample_rates
            .iter()
            .try_fold(1usize, |v, r| v.checked_mul(*r as usize))
            .ok_or_else(|| NativeBuildError::Argument("upsample product overflow".into()))?;
        if !same_shape {
            self.synth.set_input_shape(
                "feats",
                &[1, rvc_frames as i64, self.config.feature_width as i64],
            )?;
            self.synth.set_input_shape("p_len", &[1])?;
            self.synth.set_input_shape("sid", &[1])?;
            self.synth.set_input_shape(
                "rnd",
                &[1, self.config.latent_channels as i64, rvc_frames as i64],
            )?;
            if self.config.uses_f0 {
                self.synth
                    .set_input_shape("pitch", &[1, rvc_frames as i64])?;
                self.synth
                    .set_input_shape("pitchf", &[1, rvc_frames as i64])?;
                self.synth
                    .set_input_shape("nsf_noise", &[1, 1, (rvc_frames * up) as i64])?;
            }
        }
        if trace {
            tracing::trace!(target: "vocal_tensorrt", "trt trace: synth shapes set");
        }
        let p_len_value = i32::try_from(p_len_frames)
            .map_err(|_| NativeBuildError::Argument("RVC frame length exceeds i32".into()))?;
        if self.last_p_len != Some(p_len_value) {
            self.control_host_ring.upload_bytes(
                &self.p_len,
                &p_len_value.to_le_bytes(),
                &self.stream,
            )?;
            self.last_p_len = Some(p_len_value);
        }
        if self.last_sid != Some(sid) {
            self.control_host_ring
                .upload_bytes(&self.sid, &sid.to_le_bytes(), &self.stream)?;
            self.last_sid = Some(sid);
        }
        self.synth
            .set_tensor_address_cached("feats", self.feats.device_ptr())?;
        self.synth
            .set_tensor_address_cached("p_len", self.p_len.device_ptr())?;
        self.synth
            .set_tensor_address_cached("sid", self.sid.device_ptr())?;
        self.synth
            .set_tensor_address_cached("rnd", self.rnd.device_ptr())?;
        self.synth
            .set_tensor_address_cached("audio", self.audio.device_ptr())?;
        if self.config.uses_f0 {
            self.synth
                .set_tensor_address_cached("pitch", self.pitch.device_ptr())?;
            self.synth
                .set_tensor_address_cached("pitchf", self.pitchf.device_ptr())?;
            self.synth
                .set_tensor_address_cached("nsf_noise", self.nsf_noise.device_ptr())?;
        }
        let graph_key = PipelineShapeKey {
            samples: waveform.len(),
            rmvpe_samples,
            hubert_frames,
            pitch_frames,
            frontend_pitch_frames,
            rvc_frames,
            f0_advance_frames: advance_frames,
            streaming_f0,
        };
        if (self.cuda_graphs && !self.external_graph_capability_checked)
            || (self.hybrid_cuda_graphs && !self.hybrid_graph_capability_checked)
        {
            // TensorRT may reject external capture for Myelin kernels,
            // blocking allocation, or data-dependent control flow. Query
            // after dynamic shapes are concrete. Hybrid mode keeps acoustic
            // and decode capability independent so one unsupported subgraph
            // can remain eager while the other still replays.
            if self.cuda_graphs {
                let capability = self
                    .hubert
                    .is_stream_capturable(&self.stream)
                    .and_then(|hubert| {
                        self.rmvpe_realtime
                            .is_stream_capturable(&self.stream)
                            .and_then(|realtime| {
                                self.rmvpe_full
                                    .is_stream_capturable(&self.stream)
                                    .map(|full| hubert && realtime && full)
                            })
                    })
                    .and_then(|both| {
                        self.frontend
                            .is_stream_capturable(&self.stream)
                            .map(|frontend| both && frontend)
                    })
                    .and_then(|both| {
                        self.synth
                            .is_stream_capturable(&self.stream)
                            .map(|synth| both && synth)
                    })
                    .unwrap_or(false);
                self.external_graph_capability_checked = true;
                if !capability {
                    self.cuda_graphs = false;
                    self.graph_invalidations = self.graph_invalidations.saturating_add(1);
                }
            }
            if self.hybrid_cuda_graphs {
                let hubert = self
                    .hubert
                    .is_stream_capturable(&self.stream)
                    .unwrap_or(false);
                let rmvpe = if streaming_f0 {
                    self.rmvpe_realtime
                        .is_stream_capturable(&self.stream)
                        .unwrap_or(false)
                } else {
                    self.rmvpe_full
                        .is_stream_capturable(&self.stream)
                        .unwrap_or(false)
                };
                let frontend = self
                    .frontend
                    .is_stream_capturable(&self.stream)
                    .unwrap_or(false);
                let synth = self
                    .synth
                    .is_stream_capturable(&self.stream)
                    .unwrap_or(false);
                self.hybrid_graph_capability_checked = true;
                self.hybrid_acoustic_capturable = hubert && rmvpe;
                self.hybrid_decode_capturable = frontend && synth;
                if !self.hybrid_acoustic_capturable && !self.hybrid_decode_capturable {
                    self.hybrid_cuda_graphs = false;
                    self.graph_invalidations = self.graph_invalidations.saturating_add(1);
                }
            }
        }
        if self.native_cuda_graphs {
            if self.native_graph_shape == Some(graph_key) {
                self.native_graph_replays = self.native_graph_replays.saturating_add(1);
            } else {
                if self.native_graph_shape.is_some() {
                    self.native_graph_invalidations =
                        self.native_graph_invalidations.saturating_add(1);
                }
                self.native_graph_shape = Some(graph_key);
                self.native_graph_captures = self.native_graph_captures.saturating_add(1);
            }
        }
        if trace {
            tracing::trace!(target: "vocal_tensorrt",
                "trt trace: enqueue sequence mode={} native={} whole={} hybrid={}",
                self.graph_mode.as_str(), self.native_cuda_graphs, self.cuda_graphs,
                self.hybrid_cuda_graphs
            );
        }
        // The external graph path uses a bounded LRU. A graph contains only
        // device operations and fixed binding addresses; H2D/D2H and scalar
        // shape bookkeeping remain outside the capture. Retaining multiple
        // entries avoids recapture when a worker alternates its normal 11200
        // and 44960 windows.
        let graph_replay = if self.cuda_graphs {
            self.external_graph_tick = self.external_graph_tick.wrapping_add(1);
            if let Some(index) = self
                .external_graph_cache
                .iter()
                .position(|entry| entry.key == graph_key)
            {
                self.external_graph_cache[index].last_used = self.external_graph_tick;
                self.external_graph_cache[index].exec.launch(&self.stream)?;
                self.graph_cache_hits = self.graph_cache_hits.saturating_add(1);
                self.graph_replays = self.graph_replays.saturating_add(1);
                true
            } else {
                self.graph_cache_misses = self.graph_cache_misses.saturating_add(1);
                false
            }
        } else if self.hybrid_cuda_graphs {
            self.external_graph_tick = self.external_graph_tick.wrapping_add(1);
            if let Some(index) = self
                .hybrid_graph_cache
                .iter()
                .position(|entry| entry.key == graph_key)
            {
                let acoustic_cached = self.hybrid_graph_cache[index].acoustic.is_some();
                if acoustic_cached {
                    self.hybrid_graph_cache[index]
                        .acoustic
                        .as_ref()
                        .expect("hybrid acoustic graph is present")
                        .launch(&self.stream)?;
                } else {
                    self.enqueue_acoustic_sequence(
                        streaming_f0,
                        pitch_frames,
                        advance_frames,
                        rmvpe_samples,
                        rmvpe_waveform,
                    )?;
                }
                let decode_cached = self.hybrid_graph_cache[index].decode.is_some();
                if decode_cached {
                    self.hybrid_graph_cache[index]
                        .decode
                        .as_ref()
                        .expect("hybrid decode graph is present")
                        .launch(&self.stream)?;
                } else {
                    self.enqueue_decode_sequence()?;
                }
                self.hybrid_graph_cache[index].last_used = self.external_graph_tick;
                self.graph_cache_hits = self.graph_cache_hits.saturating_add(1);
                self.graph_replays = self
                    .graph_replays
                    .saturating_add(u64::from(acoustic_cached) + u64::from(decode_cached));
                true
            } else {
                self.graph_cache_misses = self.graph_cache_misses.saturating_add(1);
                false
            }
        } else {
            false
        };
        if !graph_replay {
            let capture = if self.cuda_graphs {
                match self.stream.begin_capture() {
                    Ok(capture) => Some(capture),
                    Err(_) => {
                        self.cuda_graphs = false;
                        self.graph_invalidations = self.graph_invalidations.saturating_add(1);
                        None
                    }
                }
            } else {
                None
            };
            if let Some(capture) = capture {
                match self.enqueue_stage_sequence(
                    streaming_f0,
                    pitch_frames,
                    advance_frames,
                    rmvpe_samples,
                    rmvpe_waveform,
                ) {
                    Ok(()) => match capture.finish() {
                        Ok(exec) => {
                            if self.external_graph_cache.len() >= self.external_graph_cache_capacity
                            {
                                let oldest = self
                                    .external_graph_cache
                                    .iter()
                                    .enumerate()
                                    .min_by_key(|(_, entry)| entry.last_used)
                                    .map(|(index, _)| index)
                                    .expect("non-empty graph cache at capacity");
                                self.external_graph_cache.remove(oldest);
                                self.graph_evictions = self.graph_evictions.saturating_add(1);
                            }
                            self.external_graph_cache.push(ExternalGraphEntry {
                                key: graph_key,
                                exec,
                                last_used: self.external_graph_tick,
                            });
                            self.graph_captures = self.graph_captures.saturating_add(1);
                            self.external_graph_cache
                                .last()
                                .expect("graph cache insertion succeeded")
                                .exec
                                .launch(&self.stream)?;
                            self.graph_replays = self.graph_replays.saturating_add(1);
                        }
                        Err(_) => {
                            self.cuda_graphs = false;
                            self.graph_invalidations = self.graph_invalidations.saturating_add(1);
                            self.enqueue_stage_sequence(
                                streaming_f0,
                                pitch_frames,
                                advance_frames,
                                rmvpe_samples,
                                rmvpe_waveform,
                            )?;
                        }
                    },
                    Err(_) => {
                        // TensorRT-RTX engines can reject stream capture (the
                        // Myelin runner reports this at enqueue time). The
                        // capture guard aborts the stream and we replay the
                        // same fixed bindings eagerly once, then disable
                        // capture for this pipeline instance.
                        drop(capture);
                        self.cuda_graphs = false;
                        self.graph_invalidations = self.graph_invalidations.saturating_add(1);
                        self.enqueue_stage_sequence(
                            streaming_f0,
                            pitch_frames,
                            advance_frames,
                            rmvpe_samples,
                            rmvpe_waveform,
                        )?;
                    }
                }
            } else if self.hybrid_cuda_graphs {
                let acoustic_capturable = self.hybrid_acoustic_capturable;
                let decode_capturable = self.hybrid_decode_capturable;
                let captured = self.capture_hybrid_graph(
                    graph_key,
                    acoustic_capturable,
                    decode_capturable,
                    streaming_f0,
                    pitch_frames,
                    advance_frames,
                    rmvpe_samples,
                    rmvpe_waveform,
                )?;
                if !captured {
                    self.hybrid_cuda_graphs = false;
                    self.graph_invalidations = self.graph_invalidations.saturating_add(1);
                    self.enqueue_stage_sequence(
                        streaming_f0,
                        pitch_frames,
                        advance_frames,
                        rmvpe_samples,
                        rmvpe_waveform,
                    )?;
                }
            } else {
                self.enqueue_stage_sequence(
                    streaming_f0,
                    pitch_frames,
                    advance_frames,
                    rmvpe_samples,
                    rmvpe_waveform,
                )?;
            }
        }
        // Queue the final audio copy on the same stream before synchronizing.
        // Pageable cudaMemcpy after synchronization adds a second host-side
        // synchronization point on every callback.  A pinned ring keeps the
        // copy ordered with the graph while retaining an owned Vec API.
        let audio_samples = rvc_frames
            .checked_mul(up)
            .ok_or_else(|| NativeBuildError::Argument("audio sample length overflow".into()))?;
        let audio_bytes_len = audio_samples
            .checked_mul(4)
            .ok_or_else(|| NativeBuildError::Argument("audio byte length overflow".into()))?;
        let async_audio_d2h = self.runtime_config.async_audio_d2h;
        let audio_host_slot = if self.debug_intermediates || !async_audio_d2h {
            None
        } else {
            Some(self.audio_host_ring.download_bytes_async(
                &self.audio,
                audio_bytes_len,
                &self.stream,
            )?)
        };
        self.stream.synchronize()?;
        self.shape_key = Some(graph_key);
        let hubert_enqueue_ms = self
            .stage_events
            .hubert_start
            .elapsed(&self.stage_events.hubert_end)?;
        let rmvpe_enqueue_ms = self
            .stage_events
            .rmvpe_start
            .elapsed(&self.stage_events.rmvpe_end)?;
        let frontend_enqueue_ms = self
            .stage_events
            .frontend_start
            .elapsed(&self.stage_events.frontend_end)?;
        let synth_enqueue_ms = self
            .stage_events
            .synth_start
            .elapsed(&self.stage_events.synth_end)?;
        let mut audio = vec![0_f32; audio_samples];
        if let Some(slot) = audio_host_slot {
            self.audio_host_ring.copy_f32(slot, &mut audio)?;
        } else {
            self.audio.download_f32(&mut audio)?;
        }
        let (
            hubert_features,
            rvc_features,
            pitchf,
            aligned_pitchf,
            pitch,
            rmvpe_mel,
            rmvpe_bottleneck,
            rmvpe_unet,
            rmvpe_gru,
            rmvpe_logits,
        ) = if self.debug_intermediates {
            let mut feature_bytes = vec![0_u8; hubert_frames * 768 * 4];
            self.hubert_features.download_bytes(&mut feature_bytes)?;
            let features = feature_bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect::<Vec<_>>();
            let mut rvc_feature_bytes = vec![0_u8; rvc_frames * 768 * 4];
            self.feats.download_bytes(&mut rvc_feature_bytes)?;
            let rvc_features = rvc_feature_bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect::<Vec<_>>();
            let mut raw_pitch_bytes = vec![0_u8; pitch_frames * 4];
            self.pitchf_raw.download_bytes(&mut raw_pitch_bytes)?;
            let raw_pitch = raw_pitch_bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .take(rvc_frames)
                .collect::<Vec<_>>();
            let mut aligned_pitch_bytes = vec![0_u8; rvc_frames * 4];
            self.pitchf.download_bytes(&mut aligned_pitch_bytes)?;
            let aligned_pitchf = aligned_pitch_bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect::<Vec<_>>();
            let mut pitch_bytes = vec![0_u8; rvc_frames * 4];
            self.pitch.download_bytes(&mut pitch_bytes)?;
            let coarse = pitch_bytes
                .chunks_exact(4)
                .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
                .collect::<Vec<_>>();
            let (rmvpe_mel, rmvpe_bottleneck, rmvpe_unet, rmvpe_gru, rmvpe_logits) =
                if self.rmvpe_debug_outputs {
                    let mut mel_bytes = vec![0_u8; pitch_frames * 128 * 4];
                    self.rmvpe_debug_mel.download_bytes(&mut mel_bytes)?;
                    let mel = mel_bytes
                        .chunks_exact(4)
                        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                        .take(128 * pitch_frames)
                        .collect::<Vec<_>>();
                    let padded_pitch_frames = ((pitch_frames + 31) / 32) * 32;
                    let bottleneck_frames = padded_pitch_frames / 32;
                    let mut bottleneck_bytes = vec![0_u8; 512 * bottleneck_frames * 4 * 4];
                    self.rmvpe_debug_bottleneck
                        .download_bytes(&mut bottleneck_bytes)?;
                    let bottleneck = bottleneck_bytes
                        .chunks_exact(4)
                        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                        .take(512 * bottleneck_frames * 4)
                        .collect::<Vec<_>>();
                    let mut unet_bytes = vec![0_u8; padded_pitch_frames * 16 * 128 * 4];
                    self.rmvpe_debug_unet.download_bytes(&mut unet_bytes)?;
                    let unet = unet_bytes
                        .chunks_exact(4)
                        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                        .take(padded_pitch_frames * 16 * 128)
                        .collect::<Vec<_>>();
                    let mut gru_bytes = vec![0_u8; padded_pitch_frames * 512 * 4];
                    self.rmvpe_debug_gru.download_bytes(&mut gru_bytes)?;
                    let gru = gru_bytes
                        .chunks_exact(4)
                        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                        .take(512 * padded_pitch_frames)
                        .collect::<Vec<_>>();
                    let mut logits_bytes = vec![0_u8; padded_pitch_frames * 360 * 4];
                    self.rmvpe_debug_logits.download_bytes(&mut logits_bytes)?;
                    let logits = logits_bytes
                        .chunks_exact(4)
                        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                        .take(360 * padded_pitch_frames)
                        .collect::<Vec<_>>();
                    (
                        Some(mel),
                        Some(bottleneck),
                        Some(unet),
                        Some(gru),
                        Some(logits),
                    )
                } else {
                    (None, None, None, None, None)
                };
            (
                Some(features),
                Some(rvc_features),
                Some(raw_pitch),
                Some(aligned_pitchf),
                Some(coarse),
                rmvpe_mel,
                rmvpe_bottleneck,
                rmvpe_unet,
                rmvpe_gru,
                rmvpe_logits,
            )
        } else {
            (None, None, None, None, None, None, None, None, None, None)
        };
        let non_finite = audio.iter().any(|v| !v.is_finite())
            || hubert_features
                .as_ref()
                .is_some_and(|v| v.iter().any(|x| !x.is_finite()))
            || pitchf
                .as_ref()
                .is_some_and(|v| v.iter().any(|x| !x.is_finite()));
        if non_finite && !self.diagnostics.allow_nonfinite {
            return Err(NativeBuildError::Api(
                "TensorRT pipeline produced non-finite output".into(),
            ));
        }
        Ok(RvcTrtOutput {
            audio,
            hubert_features,
            rvc_features,
            pitchf,
            aligned_pitchf,
            pitch,
            rmvpe_mel,
            rmvpe_bottleneck,
            rmvpe_unet,
            rmvpe_gru,
            rmvpe_logits,
            hubert_frames,
            pitch_frames,
            rvc_frames,
            timings: RvcTrtTimings {
                upload_ms,
                hubert_enqueue_ms,
                rmvpe_enqueue_ms,
                frontend_enqueue_ms,
                synth_enqueue_ms,
                total_ms: start.elapsed().as_secs_f64() * 1000.0,
            },
        })
    }
}

fn concrete_dim(shape: &[i64], axis: usize, fallback: usize) -> Result<usize, NativeBuildError> {
    let value = shape.get(axis).copied().unwrap_or(-1);
    if value > 0 {
        usize::try_from(value)
            .map_err(|_| NativeBuildError::Argument("tensor dimension overflow".into()))
    } else {
        Ok(fallback)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FrameShape {
    hubert: usize,
    pitch: usize,
    rvc: usize,
}

fn frame_shape(samples: usize) -> FrameShape {
    let hubert = hubert_frames_for_samples(samples);
    let pitch = pitch_frames_for_samples(samples);
    FrameShape {
        hubert,
        pitch,
        // WebUI/Applio append the last HuBERT frame, interpolate by two, and
        // crop to waveform_len/160. Keep the profile at the actual p_len so
        // the native cache and Synth see the same sequence length.
        rvc: (samples / RVC_HOP_SAMPLES).min((hubert + 1) * 2).min(pitch),
    }
}

fn hubert_frames_for_samples(samples: usize) -> usize {
    ((((samples / 80).saturating_sub(3) / 2).saturating_sub(1) / 2) + 1).max(1)
}

fn pitch_frames_for_samples(samples: usize) -> usize {
    samples / RVC_HOP_SAMPLES + 1
}

#[cfg(test)]
mod tests {
    use super::{
        GraphReplayMode, RvcCudaGraphStats, RvcRealtimeConfig, RvcTrtProfile, frame_shape,
    };

    #[test]
    fn reference_profile_matches_real_application_windows() {
        let profile = RvcTrtProfile::reference();
        assert_eq!(
            (
                profile.min_samples,
                profile.opt_samples,
                profile.max_samples
            ),
            (11_200, 35_520, 44_960)
        );
        assert_eq!(
            (
                profile.min_hubert_frames,
                profile.opt_hubert_frames,
                profile.max_hubert_frames
            ),
            (34, 110, 140)
        );
        assert_eq!(
            (
                profile.min_pitch_frames,
                profile.opt_pitch_frames,
                profile.max_pitch_frames
            ),
            (71, 223, 282)
        );
        assert_eq!(
            (
                profile.min_rvc_frames,
                profile.opt_rvc_frames,
                profile.max_rvc_frames
            ),
            (70, 222, 281)
        );
        profile.validate().expect("reference profile is ordered");
    }

    #[test]
    fn exact_presets_are_derived_from_context_windows() {
        let vc = RvcRealtimeConfig::voice_changer();
        assert_eq!(
            (
                vc.chunk_samples_16k,
                vc.extra_samples_16k,
                vc.crossfade_samples_16k,
                vc.sola_search_samples_16k,
                vc.inference_samples()
            ),
            (8192, 1365, 1365, 192, 11_200)
        );
        let profile = RvcTrtProfile::voice_changer();
        assert_eq!(
            (
                profile.min_samples,
                profile.opt_samples,
                profile.max_samples
            ),
            (11_200, 11_200, 11_200)
        );
        assert_eq!(
            (
                profile.min_hubert_frames,
                profile.min_pitch_frames,
                profile.min_rvc_frames
            ),
            (34, 71, 70)
        );
        assert_eq!(RvcRealtimeConfig::rvc_webui().inference_samples(), 44_960);
        assert_eq!(
            RvcRealtimeConfig::rvc_webui_legacy().inference_samples(),
            35_520
        );
        assert_eq!(RvcRealtimeConfig::applio(), RvcRealtimeConfig::rvc_webui());
    }

    #[test]
    fn graph_stats_keep_mode_in_json_reports() {
        let stats = RvcCudaGraphStats {
            mode: GraphReplayMode::HybridSubgraphs,
            enabled: true,
            captures: 2,
            replays: 4,
            ..Default::default()
        };
        let encoded = serde_json::to_value(stats).expect("stats serialize");
        assert_eq!(encoded["mode"], "hybrid_subgraphs");
        assert_eq!(encoded["captures"], 2);
        let decoded: RvcCudaGraphStats = serde_json::from_value(encoded).expect("stats decode");
        assert_eq!(decoded.mode, GraphReplayMode::HybridSubgraphs);
        assert_eq!(decoded.replays, 4);
    }

    #[test]
    fn env_profile_names_are_explicit() {
        assert_eq!(
            RvcTrtProfile::from_name("applio").unwrap().min_samples,
            44_960
        );
        assert_eq!(
            RvcTrtProfile::from_name("realtime_44960")
                .unwrap()
                .max_samples,
            44_960
        );
        assert_eq!(
            RvcTrtProfile::from_name("voice_changer")
                .unwrap()
                .min_samples,
            11_200
        );
        assert_eq!(
            RvcTrtProfile::from_name("reference").unwrap().max_samples,
            44_960
        );
    }

    #[test]
    fn old_voice_changer_points_are_not_the_default_any_more() {
        let profile = RvcTrtProfile::voice_changer();
        assert_eq!(
            (
                profile.min_samples,
                profile.opt_samples,
                profile.max_samples
            ),
            (11_200, 11_200, 11_200)
        );
        profile
            .validate()
            .expect("voice-changer profile is ordered");
        assert_eq!(RvcTrtProfile::default().min_samples, 11_200);
        assert_ne!(RvcTrtProfile::default().max_samples, 12_288);
    }

    #[test]
    fn sample_profile_must_follow_rvc_hop() {
        let mut profile = RvcTrtProfile::voice_changer();
        profile.opt_samples += 1;
        assert!(profile.validate().is_err());
    }

    #[test]
    fn full_window_drops_only_the_rmvpe_boundary_frame() {
        let full = frame_shape(44_960);
        assert_eq!((full.hubert, full.pitch, full.rvc), (140, 282, 281));
        assert_eq!(full.pitch, full.rvc + 1);

        let realtime = frame_shape(11_200);
        assert_eq!(
            (realtime.hubert, realtime.pitch, realtime.rvc),
            (34, 71, 70)
        );
        assert_eq!(realtime.pitch, realtime.rvc + 1);
    }

    #[test]
    fn legacy_profile_remains_available_for_old_manifests() {
        let profile = RvcTrtProfile::legacy();
        assert_eq!(
            (
                profile.min_samples,
                profile.opt_samples,
                profile.max_samples
            ),
            (3200, 6400, 12800)
        );
        profile.validate().expect("legacy profile is ordered");
    }
}
