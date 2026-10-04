//! Versioned, pointer-free data conversion for the C++ ABI.
//!
//! The native builder only receives this module's `repr(C)` values.  Serde
//! structures and VFRM storage never cross the FFI boundary directly.

use half::{bf16, f16};
use std::ffi::CString;
use vocal_model::{RvcSynthConfig, Tensor, TensorDType};

use crate::error::NativeBuildError;

#[repr(C)]
pub(crate) struct NativeConfig {
    pub abi_version: i32,
    pub feature_width: i32,
    pub uses_f0: i32,
    pub sample_rate: i32,
    pub hidden: i32,
    pub latent: i32,
    pub filter: i32,
    pub heads: i32,
    pub encoder_layers: i32,
    pub encoder_kernel: i32,
    pub window: i32,
    pub relative_heads: i32,
    pub speakers: i32,
    pub speaker_channels: i32,
    pub decoder_channels: i32,
    pub resblock_type: i32,
    pub upsample_count: i32,
    pub resblock_count: i32,
    pub flow_blocks: i32,
    pub flow_layers: i32,
    pub flow_kernel: i32,
    pub norm_epsilon: f32,
    pub rates: [i32; 8],
    pub kernels: [i32; 8],
    pub res_kernels: [i32; 8],
    pub res_dilations: [i32; 24],
    pub flow_dilations: [i32; 16],
}

#[repr(C)]
pub(crate) struct NativeWeight {
    pub name: *const std::os::raw::c_char,
    pub data: *const f32,
    pub count: i64,
    pub rank: i32,
    pub dims: [i64; 4],
}

#[repr(C)]
pub(crate) struct NativeOptions {
    pub workspace: u64,
    pub min_frames: i32,
    pub opt_frames: i32,
    pub max_frames: i32,
    pub debug_outputs: i32,
    pub fp16: i32,
    pub fused_attention: i32,
    pub precision: i32,
    pub quantize_weights: i32,
    pub fused_qkv: i32,
    pub dispatch_tune: i32,
    pub synth_resblock_precision: i32,
    pub synth_resblock_stage_mask: i32,
    pub synth_resblock_conv1_precision: i32,
    pub synth_resblock_conv1_stage_mask: i32,
    pub synth_resblock_conv2_precision: i32,
    pub synth_resblock_conv2_stage_mask: i32,
    pub synth_fuse_resblock_scale: i32,
    pub synth_polyphase_upsample: i32,
    pub rmvpe_encoder_precision: i32,
    pub rmvpe_decoder_precision: i32,
    pub rmvpe_stft_precision: i32,
    pub rmvpe_gru_precision: i32,
    pub rmvpe_logits_precision: i32,
    pub rmvpe_debug_outputs: i32,
    pub rmvpe_fft_mode: i32,
    pub fused_rvc_qkv: i32,
    pub hubert_feature_precision: i32,
    pub rmvpe_external_stft: i32,
    pub tactic_sources: i32,
    pub timing_cache_path: *const std::os::raw::c_char,
}

pub(crate) fn native_config(c: &RvcSynthConfig) -> Result<NativeConfig, NativeBuildError> {
    if c.upsample_rates.len() > 8 || c.resblock_kernels.len() > 8 || c.flow_dilations.len() > 16 {
        return Err(NativeBuildError::Argument(
            "canonical configuration exceeds native ABI array bounds".into(),
        ));
    }
    let mut rates = [0; 8];
    let mut kernels = [0; 8];
    let mut res_kernels = [0; 8];
    let mut res_dilations = [0; 24];
    let mut flow_dilations = [0; 16];
    for (i, value) in c.upsample_rates.iter().copied().enumerate() {
        rates[i] = i32v(value, "upsample rate")?;
        kernels[i] = i32v(c.upsample_kernels[i], "upsample kernel")?;
    }
    for (i, value) in c.resblock_kernels.iter().copied().enumerate() {
        res_kernels[i] = i32v(value, "resblock kernel")?;
        for (j, dilation) in c.resblock_dilations[i].iter().copied().enumerate() {
            let index = i * 3 + j;
            if index >= res_dilations.len() {
                return Err(NativeBuildError::Argument(
                    "resblock dilation array is too large".into(),
                ));
            }
            res_dilations[index] = i32v(dilation, "resblock dilation")?;
        }
    }
    for (i, value) in c.flow_dilations.iter().copied().enumerate() {
        flow_dilations[i] = i32v(value, "flow dilation")?;
    }
    Ok(NativeConfig {
        abi_version: 2,
        feature_width: i32v(c.feature_width, "feature width")?,
        uses_f0: c.uses_f0 as i32,
        sample_rate: i32::try_from(c.sample_rate)
            .map_err(|_| NativeBuildError::Argument("sample rate exceeds i32 range".into()))?,
        hidden: i32v(c.hidden_channels, "hidden channels")?,
        latent: i32v(c.latent_channels, "latent channels")?,
        filter: i32v(c.filter_channels, "filter channels")?,
        heads: i32v(c.attention_heads, "attention heads")?,
        encoder_layers: i32::try_from(c.encoder_layers).map_err(|_| {
            NativeBuildError::Argument("encoder layer count exceeds i32 range".into())
        })?,
        encoder_kernel: i32v(c.encoder_kernel, "encoder kernel")?,
        window: i32v(c.relative_window, "relative window")?,
        relative_heads: i32v(c.relative_heads, "relative heads")?,
        speakers: i32v(c.speakers, "speaker count")?,
        speaker_channels: i32v(c.speaker_channels, "speaker channels")?,
        decoder_channels: i32v(c.decoder_channels, "decoder channels")?,
        resblock_type: i32::from(c.resblock_type),
        upsample_count: i32::try_from(c.upsample_rates.len())
            .map_err(|_| NativeBuildError::Argument("upsample count exceeds i32 range".into()))?,
        resblock_count: i32::try_from(c.resblock_kernels.len())
            .map_err(|_| NativeBuildError::Argument("resblock count exceeds i32 range".into()))?,
        flow_blocks: i32::try_from(c.flow_blocks)
            .map_err(|_| NativeBuildError::Argument("flow block count exceeds i32 range".into()))?,
        flow_layers: i32::try_from(c.flow_layers)
            .map_err(|_| NativeBuildError::Argument("flow layer count exceeds i32 range".into()))?,
        flow_kernel: i32v(c.flow_kernel, "flow kernel")?,
        norm_epsilon: c.norm_epsilon,
        rates,
        kernels,
        res_kernels,
        res_dilations,
        flow_dilations,
    })
}

fn i32v(value: u64, field: &str) -> Result<i32, NativeBuildError> {
    i32::try_from(value)
        .map_err(|_| NativeBuildError::Argument(format!("{field} exceeds i32 range")))
}

pub(crate) fn tensor_f32(tensor: &Tensor) -> Result<Vec<f32>, NativeBuildError> {
    let count = tensor
        .shape
        .iter()
        .try_fold(1usize, |acc, &d| {
            usize::try_from(d).ok().and_then(|d| acc.checked_mul(d))
        })
        .ok_or_else(|| {
            NativeBuildError::Argument(format!("tensor {} shape overflows", tensor.name))
        })?;
    let mut values = Vec::with_capacity(count);
    match tensor.dtype {
        TensorDType::F32 => {
            let (chunks, remainder) = tensor.data.as_chunks::<4>();
            values.extend(chunks.iter().map(|bytes| f32::from_le_bytes(*bytes)));
            if !remainder.is_empty() {
                return Err(NativeBuildError::Argument(format!(
                    "tensor {} byte length is not aligned to F32",
                    tensor.name
                )));
            }
        }
        TensorDType::F16 => {
            let (chunks, remainder) = tensor.data.as_chunks::<2>();
            values.extend(
                chunks
                    .iter()
                    .map(|bytes| f16::from_bits(u16::from_le_bytes(*bytes)).to_f32()),
            );
            if !remainder.is_empty() {
                return Err(NativeBuildError::Argument(format!(
                    "tensor {} byte length is not aligned to F16",
                    tensor.name
                )));
            }
        }
        TensorDType::BF16 => {
            let (chunks, remainder) = tensor.data.as_chunks::<2>();
            values.extend(
                chunks
                    .iter()
                    .map(|bytes| bf16::from_bits(u16::from_le_bytes(*bytes)).to_f32()),
            );
            if !remainder.is_empty() {
                return Err(NativeBuildError::Argument(format!(
                    "tensor {} byte length is not aligned to BF16",
                    tensor.name
                )));
            }
        }
        _ => {
            return Err(NativeBuildError::Argument(format!(
                "tensor {} is not floating point",
                tensor.name
            )));
        }
    }
    if values.len() != count {
        return Err(NativeBuildError::Argument(format!(
            "tensor {} element count mismatch",
            tensor.name
        )));
    }
    Ok(values)
}

pub(crate) fn c_string(value: impl AsRef<std::path::Path>) -> Result<CString, NativeBuildError> {
    CString::new(value.as_ref().to_string_lossy().as_bytes())
        .map_err(|_| NativeBuildError::Argument("path contains NUL".into()))
}
