//! TensorRT engine builders.
//!
//! ONNX import and the source-independent VFRM graph share the same output
//! serialization path but have separate entry points. This keeps future graph
//! backends from growing another FFI implementation in the public API.

use std::ffi::CString;
use std::{
    fs,
    path::{Path, PathBuf},
    ptr, slice,
};

use crate::options::OnnxProfileKind;
use crate::{
    NativeBuildOptions,
    abi::{self, NativeOptions, NativeWeight},
    error::NativeBuildError,
    ffi,
    graph::NativeGraphPlan,
};
use serde_json::Value;
use vocal_model::{AcousticKind, ModelPackage, TensorDType};

#[derive(Clone, Debug)]
pub struct NativeTensorRtBuilder {
    pub options: NativeBuildOptions,
}

impl NativeTensorRtBuilder {
    pub fn new(options: NativeBuildOptions) -> Self {
        ffi::install_log_bridge();
        Self { options }
    }

    /// Convert the public typed build policy into the stable native ABI once.
    /// All graph entry points use the same option layout; only their profile
    /// dimensions and graph-specific weight directories differ.
    fn native_options(
        &self,
        timing_cache: Option<&CString>,
        rmvpe_external_stft: i32,
    ) -> NativeOptions {
        NativeOptions {
            workspace: self.options.workspace_bytes,
            min_frames: self.options.min_frames,
            opt_frames: self.options.opt_frames,
            max_frames: self.options.max_frames,
            debug_outputs: self.options.debug_outputs as i32,
            fp16: self.options.fp16 as i32,
            fused_attention: self.options.fused_attention as i32,
            precision: self.options.precision.code(),
            synth_precision: self.options.synth_precision.code(),
            quantize_weights: self.options.quantize_weights as i32,
            fused_qkv: self.options.fused_qkv as i32,
            dispatch_tune: self.options.dispatch_tune.code(),
            synth_resblock_precision: self.options.synth_resblock_precision.code(),
            synth_resblock_stage_mask: self.options.synth_resblock_stage_mask as i32,
            synth_resblock_conv1_precision: self.options.synth_resblock_conv1_precision.code(),
            synth_resblock_conv1_stage_mask: self.options.synth_resblock_conv1_stage_mask as i32,
            synth_resblock_conv2_precision: self.options.synth_resblock_conv2_precision.code(),
            synth_resblock_conv2_stage_mask: self.options.synth_resblock_conv2_stage_mask as i32,
            synth_w8a16_conv2_mask: self.options.synth_w8a16_conv2_mask,
            synth_w8a8_conv1_mask: self.options.synth_w8a8_conv1_mask,
            synth_w8a8_conv2_mask: self.options.synth_w8a8_conv2_mask,
            synth_w8a8_activation_scale: self.options.synth_w8a8_activation_scale,
            synth_fuse_resblock_scale: self.options.synth_fuse_resblock_scale as i32,
            synth_polyphase_upsample: self.options.synth_polyphase_upsample as i32,
            synth_full_fp16: self.options.synth_full_fp16 as i32,
            synth_fp16_input_scale: self.options.synth_fp16_input_scale,
            synth_fp16_residual_accum: self.options.synth_fp16_residual_accum as i32,
            synth_fp16_source: self.options.synth_fp16_source as i32,
            synth_fp16_noise: self.options.synth_fp16_noise as i32,
            synth_fp16_noise_stage_mask: self.options.synth_fp16_noise_stage_mask as i32,
            synth_fp16_coherent_stage_mask: self.options.synth_fp16_coherent_stage_mask as i32,
            synth_fp16_fp32_stage_mask: self.options.synth_fp16_fp32_stage_mask as i32,
            synth_fp16_post: self.options.synth_fp16_post as i32,
            rmvpe_encoder_precision: self.options.rmvpe_encoder_precision.code(),
            rmvpe_decoder_precision: self.options.rmvpe_decoder_precision.code(),
            rmvpe_stft_precision: self.options.rmvpe_stft_precision.code(),
            rmvpe_gru_precision: self.options.rmvpe_gru_precision.code(),
            rmvpe_logits_precision: self.options.rmvpe_logits_precision.code(),
            rmvpe_debug_outputs: self.options.rmvpe_debug_outputs as i32,
            rmvpe_fft_mode: self.options.rmvpe_fft_radix2 as i32,
            fused_rvc_qkv: self.options.fused_rvc_qkv as i32,
            hubert_feature_precision: self.options.hubert_feature_precision.code(),
            rmvpe_external_stft,
            tactic_sources: self.options.tactic_sources as i32,
            timing_cache_path: timing_cache.map_or(ptr::null(), |path| path.as_ptr()),
        }
    }

    pub fn compile_onnx_file(&self, onnx: impl AsRef<Path>) -> Result<PathBuf, NativeBuildError> {
        self.compile_onnx_file_with_profile(onnx, OnnxProfileKind::Frames)
    }

    pub fn compile_onnx_file_with_profile(
        &self,
        onnx: impl AsRef<Path>,
        profile_kind: OnnxProfileKind,
    ) -> Result<PathBuf, NativeBuildError> {
        self.require_onnx_sdk()?;
        let path = abi::c_string(onnx)?;
        let (mut handle, mut error) = (ptr::null_mut(), ptr::null_mut());
        let code = unsafe {
            ffi::vocal_trt_build_onnx_profile(
                path.as_ptr(),
                self.options.workspace_bytes,
                self.options.fp16 as i32,
                self.options.min_frames,
                self.options.opt_frames,
                self.options.max_frames,
                self.options.min_samples,
                self.options.opt_samples,
                self.options.max_samples,
                match profile_kind {
                    OnnxProfileKind::Frames => 0,
                    OnnxProfileKind::Samples => 1,
                },
                &mut handle,
                &mut error,
            )
        };
        if code != 0 {
            return Err(NativeBuildError::Api(ffi::native_message(
                error,
                "TensorRT ONNX build failed",
            )));
        }
        self.write_build(handle)
    }

    pub fn compile_frontend(&self) -> Result<PathBuf, NativeBuildError> {
        let (mut handle, mut error) = (ptr::null_mut(), ptr::null_mut());
        let code = unsafe {
            ffi::vocal_trt_build_rvc_frontend(
                self.options.workspace_bytes,
                self.options.min_frames,
                self.options.opt_frames,
                self.options.max_frames,
                &mut handle,
                &mut error,
            )
        };
        if code != 0 {
            return Err(NativeBuildError::Api(ffi::native_message(
                error,
                "TensorRT RVC frontend build failed",
            )));
        }
        self.write_build(handle)
    }

    pub fn compile_vfrm(&self, package: &ModelPackage) -> Result<PathBuf, NativeBuildError> {
        if self.options.log_build_options {
            tracing::info!(
                target: "vocal_tensorrt::builder",
                output = %self.options.output.display(),
                min_frames = self.options.min_frames,
                opt_frames = self.options.opt_frames,
                max_frames = self.options.max_frames,
                precision = ?self.options.precision,
                fp16 = self.options.fp16,
                debug = self.options.debug_outputs,
                fused_hubert_qkv = self.options.fused_qkv,
                fused_rvc_qkv = self.options.fused_rvc_qkv,
                dispatch = self.options.dispatch_tune.code(),
                tactic_sources = format_args!("0x{:x}", self.options.tactic_sources),
                resblock_precision = ?self.options.synth_resblock_precision,
                resblock_mask = format_args!("0x{:x}", self.options.synth_resblock_stage_mask),
                w8a16_conv2_mask = format_args!("0x{:x}", self.options.synth_w8a16_conv2_mask),
                w8a8_conv1_mask = format_args!("0x{:x}", self.options.synth_w8a8_conv1_mask),
                w8a8_conv2_mask = format_args!("0x{:x}", self.options.synth_w8a8_conv2_mask),
                synth_full_fp16 = self.options.synth_full_fp16,
                synth_fp16_input_scale = self.options.synth_fp16_input_scale,
                synth_fp16_residual_accum = self.options.synth_fp16_residual_accum,
                synth_fp16_source = self.options.synth_fp16_source,
                synth_fp16_noise = self.options.synth_fp16_noise,
                synth_fp16_noise_stage_mask = format_args!("0x{:x}", self.options.synth_fp16_noise_stage_mask),
                synth_fp16_coherent_stage_mask = format_args!("0x{:x}", self.options.synth_fp16_coherent_stage_mask),
                synth_fp16_fp32_stage_mask = format_args!("0x{:x}", self.options.synth_fp16_fp32_stage_mask),
                synth_fp16_post = self.options.synth_fp16_post,
                "building native RVC Synth plan"
            );
        }
        let plan = NativeGraphPlan::from_package(package)?;
        plan.validate_weights(package)?;
        let config = package
            .rvc_synth_config()
            .map_err(|e| NativeBuildError::Api(e.to_string()))?
            .ok_or_else(|| NativeBuildError::Argument("missing rvc_config".into()))?;
        let native_config = abi::native_config(&config)?;
        let timing_cache = self.timing_cache_cstring()?;
        let options = self.native_options(timing_cache.as_ref(), 0);

        let canonical = package
            .tensors
            .iter()
            .filter(|tensor| !tensor.name.starts_with("aux."))
            .collect::<Vec<_>>();
        let names = canonical
            .iter()
            .map(|tensor| {
                CString::new(tensor.name.as_bytes())
                    .map_err(|_| NativeBuildError::Argument("tensor name contains NUL".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let data = canonical
            .iter()
            .map(|tensor| abi::tensor_f32(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        let data16 = canonical
            .iter()
            .map(|tensor| abi::tensor_u16(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        let native_weights = canonical
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                let mut dims = [1_i64; 4];
                if tensor.shape.len() > dims.len() {
                    return Err(NativeBuildError::Argument(format!(
                        "tensor {} rank {} exceeds native ABI limit",
                        tensor.name,
                        tensor.shape.len()
                    )));
                }
                for (dimension, value) in tensor.shape.iter().copied().enumerate() {
                    dims[dimension] = i64::try_from(value).map_err(|_| {
                        NativeBuildError::Argument(format!(
                            "tensor {} dimension exceeds i64",
                            tensor.name
                        ))
                    })?;
                }
                Ok(NativeWeight {
                    name: names[index].as_ptr(),
                    data: data[index].as_ptr(),
                    data16: if data16[index].is_empty() {
                        ptr::null()
                    } else {
                        data16[index].as_ptr()
                    },
                    dtype: tensor.dtype as i32,
                    count: i64::try_from(data[index].len()).map_err(|_| {
                        NativeBuildError::Argument(format!("tensor {} is too large", tensor.name))
                    })?,
                    rank: i32::try_from(tensor.shape.len()).map_err(|_| {
                        NativeBuildError::Argument(format!(
                            "tensor {} rank exceeds i32",
                            tensor.name
                        ))
                    })?,
                    dims,
                })
            })
            .collect::<Result<Vec<_>, NativeBuildError>>()?;

        let (mut handle, mut error) = (ptr::null_mut(), ptr::null_mut());
        let code = unsafe {
            ffi::vocal_trt_build_rvc(
                &native_config,
                native_weights.as_ptr(),
                native_weights.len(),
                &options,
                &mut handle,
                &mut error,
            )
        };
        if code != 0 {
            return Err(NativeBuildError::Api(ffi::native_message(
                error,
                "VFRM TensorRT graph build failed",
            )));
        }
        self.write_build(handle)
    }

    /// Build the hand-expanded HuBERT graph directly from an acoustic VFRM
    /// package.  The integer shape constants and ONNX operator IR stay in the
    /// package metadata; only floating point parameters cross the C ABI.
    pub fn compile_hubert_vfrm(&self, package: &ModelPackage) -> Result<PathBuf, NativeBuildError> {
        if !self.options.precision.is_native_type_available() {
            return Err(NativeBuildError::Argument(format!(
                "native HuBERT VFRM graph does not support {:?} on this TensorRT-RTX runtime",
                self.options.precision
            )));
        }
        let _plan =
            crate::acoustic::AcousticGraphPlan::from_package(package, AcousticKind::Hubert)?;
        let timing_cache = self.timing_cache_cstring()?;
        let mut options = self.native_options(timing_cache.as_ref(), 0);
        options.min_frames = self.options.min_samples;
        options.opt_frames = self.options.opt_samples;
        options.max_frames = self.options.max_samples;
        let canonical = package
            .tensors
            .iter()
            .filter(|tensor| {
                matches!(
                    tensor.dtype,
                    TensorDType::F16 | TensorDType::F32 | TensorDType::BF16
                ) && tensor.name.starts_with("hubert.")
            })
            .collect::<Vec<_>>();
        if canonical.is_empty() {
            return Err(NativeBuildError::Argument(
                "HuBERT VFRM has no floating point parameters".into(),
            ));
        }
        let names = canonical
            .iter()
            .map(|tensor| {
                CString::new(tensor.name.as_bytes())
                    .map_err(|_| NativeBuildError::Argument("tensor name contains NUL".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let data = canonical
            .iter()
            .map(|tensor| abi::tensor_f32(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        let data16 = canonical
            .iter()
            .map(|tensor| abi::tensor_u16(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        let native_weights = canonical
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                let mut dims = [1_i64; 4];
                if tensor.shape.len() > dims.len() {
                    return Err(NativeBuildError::Argument(format!(
                        "tensor {} rank exceeds native ABI",
                        tensor.name
                    )));
                }
                for (axis, dim) in tensor.shape.iter().copied().enumerate() {
                    dims[axis] = i64::try_from(dim).map_err(|_| {
                        NativeBuildError::Argument("HuBERT tensor dimension overflow".into())
                    })?;
                }
                Ok(abi::NativeWeight {
                    name: names[index].as_ptr(),
                    data: data[index].as_ptr(),
                    data16: if data16[index].is_empty() {
                        ptr::null()
                    } else {
                        data16[index].as_ptr()
                    },
                    dtype: tensor.dtype as i32,
                    count: i64::try_from(data[index].len()).map_err(|_| {
                        NativeBuildError::Argument("HuBERT tensor too large".into())
                    })?,
                    rank: i32::try_from(tensor.shape.len())
                        .map_err(|_| NativeBuildError::Argument("HuBERT rank overflow".into()))?,
                    dims,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (mut handle, mut error) = (ptr::null_mut(), ptr::null_mut());
        let code = unsafe {
            ffi::vocal_trt_build_hubert(
                native_weights.as_ptr(),
                native_weights.len(),
                &options,
                &mut handle,
                &mut error,
            )
        };
        if code != 0 {
            return Err(NativeBuildError::Api(ffi::native_message(
                error,
                "HuBERT native TensorRT build failed",
            )));
        }
        self.write_build(handle)
    }

    /// Build the RMVPE acoustic-v1 package without opening its source ONNX.
    ///
    /// The converter keeps the ONNX operator ids in metadata.  We use those
    /// ids only to assign stable native ABI names (`rmvpe.node.<id>.*`); the
    /// C++ side expands the known RMVPE E2E topology into TensorRT layers.
    /// This makes the native graph independent of exporter tensor hashes while
    /// still rejecting a package whose expected parameters are absent.
    pub fn compile_rmvpe_vfrm(&self, package: &ModelPackage) -> Result<PathBuf, NativeBuildError> {
        if !self.options.precision.is_native_type_available() {
            return Err(NativeBuildError::Argument(format!(
                "native RMVPE VFRM graph does not support {:?} on this TensorRT-RTX runtime",
                self.options.precision
            )));
        }
        let plan = crate::acoustic::AcousticGraphPlan::from_package(package, AcousticKind::Rmvpe)?;
        if plan.kernel_count() != 2 {
            return Err(NativeBuildError::Argument(format!(
                "RMVPE acoustic-v1 requires STFT and GRU kernel stages; found {}",
                plan.kernel_count()
            )));
        }
        let nodes = package
            .metadata
            .get("nodes")
            .and_then(Value::as_array)
            .ok_or_else(|| NativeBuildError::Argument("RMVPE package has no operator IR".into()))?;
        let mut selected: Vec<(String, &vocal_model::Tensor)> = Vec::new();
        let mut add = |name: String, source: &str| -> Result<(), NativeBuildError> {
            let tensor = package.tensor(source).ok_or_else(|| {
                NativeBuildError::Argument(format!("RMVPE native mapping misses {source}"))
            })?;
            if !matches!(
                tensor.dtype,
                TensorDType::F16 | TensorDType::F32 | TensorDType::BF16
            ) {
                // Shape/index constants are part of the acoustic IR but are
                // rebuilt by the handwritten graph and never cross this ABI.
                return Ok(());
            }
            if !selected.iter().any(|(n, _)| n == &name) {
                selected.push((name, tensor));
            }
            Ok(())
        };
        for node in nodes {
            let id = node
                .get("id")
                .and_then(Value::as_u64)
                .ok_or_else(|| NativeBuildError::Argument("RMVPE node has no id".into()))?;
            let op = node.get("op").and_then(Value::as_str).unwrap_or_default();
            let inputs = node
                .get("inputs")
                .and_then(Value::as_array)
                .ok_or_else(|| NativeBuildError::Argument("RMVPE node has no inputs".into()))?;
            let suffixes: &[&str] = match op {
                "Conv" | "ConvTranspose" => &["weight", "bias"],
                "BatchNormalization" => &["weight", "bias", "mean", "var"],
                "GRU" => &["W", "R", "B"],
                "MatMul" => &["weight"],
                "Add" => &["bias"],
                _ => &[],
            };
            for (slot, suffix) in suffixes.iter().enumerate() {
                let input_index = if op == "Add" && id == 361 {
                    0
                } else {
                    slot + 1
                };
                let Some(source) = inputs.get(input_index).and_then(Value::as_str) else {
                    continue;
                };
                if package.tensor(source).is_some() {
                    add(format!("rmvpe.node.{id}.{suffix}"), source)?;
                }
            }
        }
        // The mel basis is the only parameter consumed by the handwritten
        // Fourier frontend that is not a Conv/MatMul node initializer slot.
        add("rmvpe.mel_basis".into(), "rmvpe.mel_extractor.mel_basis")?;
        if selected.is_empty() {
            return Err(NativeBuildError::Argument(
                "RMVPE package has no native parameters".into(),
            ));
        }
        let names = selected
            .iter()
            .map(|(name, _)| {
                CString::new(name.as_bytes()).map_err(|_| {
                    NativeBuildError::Argument("RMVPE native name contains NUL".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let data = selected
            .iter()
            .map(|(_, tensor)| abi::tensor_f32(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        let data16 = selected
            .iter()
            .map(|(_, tensor)| abi::tensor_u16(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        let native_weights = selected
            .iter()
            .enumerate()
            .map(|(index, (_, tensor))| {
                let mut dims = [1_i64; 4];
                if tensor.shape.len() > dims.len() {
                    return Err(NativeBuildError::Argument(format!(
                        "RMVPE tensor {} rank exceeds native ABI",
                        tensor.name
                    )));
                }
                for (axis, dim) in tensor.shape.iter().copied().enumerate() {
                    dims[axis] = i64::try_from(dim).map_err(|_| {
                        NativeBuildError::Argument("RMVPE tensor dimension overflow".into())
                    })?;
                }
                Ok(NativeWeight {
                    name: names[index].as_ptr(),
                    data: data[index].as_ptr(),
                    data16: if data16[index].is_empty() {
                        ptr::null()
                    } else {
                        data16[index].as_ptr()
                    },
                    dtype: tensor.dtype as i32,
                    count: i64::try_from(data[index].len())
                        .map_err(|_| NativeBuildError::Argument("RMVPE tensor too large".into()))?,
                    rank: i32::try_from(tensor.shape.len()).map_err(|_| {
                        NativeBuildError::Argument("RMVPE tensor rank overflow".into())
                    })?,
                    dims,
                })
            })
            .collect::<Result<Vec<_>, NativeBuildError>>()?;
        let timing_cache = self.timing_cache_cstring()?;
        let mut options =
            self.native_options(timing_cache.as_ref(), self.options.rmvpe_fft_cufft as i32);
        // RMVPE's waveform binding is profiled in complete 160-sample hops.
        // Keep the measured application sample points exact; the graph itself
        // exposes the additional STFT boundary frame at runtime (`T/160+1`)
        // and does not need that extra frame folded into the input profile.
        options.min_frames = (self.options.min_samples / 160).max(1);
        options.opt_frames = (self.options.opt_samples / 160).max(options.min_frames);
        options.max_frames = (self.options.max_samples / 160).max(options.opt_frames);
        let (mut handle, mut error) = (ptr::null_mut(), ptr::null_mut());
        let code = unsafe {
            ffi::vocal_trt_build_rmvpe(
                native_weights.as_ptr(),
                native_weights.len(),
                &options,
                &mut handle,
                &mut error,
            )
        };
        if code != 0 {
            return Err(NativeBuildError::Api(ffi::native_message(
                error,
                "RMVPE native TensorRT build failed",
            )));
        }
        self.write_build(handle)
    }

    /// Compatibility alias retained for callers that used the experimental
    /// native builder before the backend was split into modules.
    #[deprecated(note = "use compile_vfrm")]
    pub fn compile_vfrm2(&self, package: &ModelPackage) -> Result<PathBuf, NativeBuildError> {
        self.compile_vfrm(package)
    }

    fn require_onnx_sdk(&self) -> Result<(), NativeBuildError> {
        for path in [
            self.options.sdk_root.join("include/NvInfer.h"),
            self.options.sdk_root.join("bin/tensorrt_rtx_1_6.dll"),
            self.options
                .sdk_root
                .join("bin/tensorrt_onnxparser_rtx_1_6.dll"),
        ] {
            if !path.is_file() {
                return Err(NativeBuildError::Missing(path));
            }
        }
        Ok(())
    }

    fn timing_cache_cstring(&self) -> Result<Option<CString>, NativeBuildError> {
        self.options
            .timing_cache
            .as_ref()
            .map(|path| {
                CString::new(path.to_string_lossy().as_bytes()).map_err(|_| {
                    NativeBuildError::Argument("timing cache path contains NUL".into())
                })
            })
            .transpose()
    }

    fn write_build(&self, handle: *mut ffi::Build) -> Result<PathBuf, NativeBuildError> {
        if handle.is_null() {
            return Err(NativeBuildError::Api(
                "TensorRT builder returned a null handle".into(),
            ));
        }
        let mut bytes_ptr = ptr::null_mut();
        let mut bytes_len = 0;
        let code = unsafe { ffi::vocal_trt_build_bytes(handle, &mut bytes_ptr, &mut bytes_len) };
        if code != 0 || bytes_ptr.is_null() {
            unsafe { ffi::vocal_trt_build_free(handle) };
            return Err(NativeBuildError::Api(
                "serialized TensorRT engine unavailable".into(),
            ));
        }
        let write_result = unsafe {
            fs::write(
                &self.options.output,
                slice::from_raw_parts(bytes_ptr, bytes_len),
            )
        };
        unsafe {
            ffi::vocal_trt_free_bytes(bytes_ptr, bytes_len);
            ffi::vocal_trt_build_free(handle);
        }
        write_result
            .map(|_| self.options.output.clone())
            .map_err(NativeBuildError::Io)
    }
}
