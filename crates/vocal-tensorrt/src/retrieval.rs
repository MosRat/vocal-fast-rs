//! CUDA resident FAISS-compatible retrieval for the RVC feature path.
//!
//! The model crate owns the bounded FAISS parser. This module only transfers
//! its normalized IVF-Flat arrays to device memory and exposes the one hot
//! operation needed by the pipeline. No FAISS runtime or cuBLAS dependency is
//! required at inference time.

use std::ptr;

use vocal_model::RetrievalIndex;

use crate::{CudaBuffer, CudaStream, NativeBuildError, ffi};

pub(crate) mod protection;

#[cfg(test)]
mod tests;

pub struct GpuRetrievalIndex {
    raw: *mut ffi::RetrievalGpuIndex,
    dimensions: usize,
    nlist: usize,
    nprobe: usize,
    vector_count: usize,
}

unsafe impl Send for GpuRetrievalIndex {}

impl GpuRetrievalIndex {
    pub fn new(index: &RetrievalIndex) -> Result<Self, NativeBuildError> {
        if index.is_empty() {
            return Err(NativeBuildError::Argument(
                "retrieval index has no vectors".into(),
            ));
        }
        let dimensions = index.dimensions();
        let nlist = index.nlist();
        let nprobe = index.nprobe();
        if dimensions == 0 || nlist == 0 || nprobe == 0 || nprobe > nlist {
            return Err(NativeBuildError::Argument(
                "retrieval index has invalid dimensions or IVF configuration".into(),
            ));
        }
        let (mut raw, mut error) = (ptr::null_mut(), ptr::null_mut());
        let code = unsafe {
            ffi::vocal_faiss_gpu_create(
                i32::try_from(dimensions).map_err(|_| {
                    NativeBuildError::Argument("retrieval width exceeds i32".into())
                })?,
                i32::try_from(nlist).map_err(|_| {
                    NativeBuildError::Argument("retrieval nlist exceeds i32".into())
                })?,
                i32::try_from(nprobe).map_err(|_| {
                    NativeBuildError::Argument("retrieval nprobe exceeds i32".into())
                })?,
                index.len(),
                index.centroids().as_ptr(),
                index.centroids().len(),
                index.offsets().as_ptr(),
                index.offsets().len(),
                index.vectors().as_ptr(),
                index.vectors().len(),
                index.ids().as_ptr(),
                index.ids().len(),
                &mut raw,
                &mut error,
            )
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "create CUDA retrieval index failed",
            )))
        } else if raw.is_null() {
            Err(NativeBuildError::Api(
                "create CUDA retrieval index returned null handle".into(),
            ))
        } else {
            Ok(Self {
                raw,
                dimensions,
                nlist,
                nprobe,
                vector_count: index.len(),
            })
        }
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn nlist(&self) -> usize {
        self.nlist
    }

    pub fn nprobe(&self) -> usize {
        self.nprobe
    }

    pub fn len(&self) -> usize {
        self.vector_count
    }

    pub fn is_empty(&self) -> bool {
        self.vector_count == 0
    }

    /// Blend HuBERT features in place. `index_rate` is a one-element device
    /// buffer, so changing the ratio does not invalidate a captured CUDA graph.
    pub fn blend(
        &self,
        features: &CudaBuffer,
        frames: usize,
        index_rate: &CudaBuffer,
        stream: &CudaStream,
    ) -> Result<(), NativeBuildError> {
        if frames == 0 {
            return Ok(());
        }
        validate_blend_buffers(frames, self.dimensions, features.bytes, index_rate.bytes)?;
        let mut error = ptr::null_mut();
        let code = unsafe {
            ffi::vocal_faiss_gpu_blend(
                self.raw,
                features.device_ptr() as *mut f32,
                frames,
                i32::try_from(self.dimensions).map_err(|_| {
                    NativeBuildError::Argument("retrieval width exceeds i32".into())
                })?,
                index_rate.device_ptr() as *const f32,
                stream.native_handle(),
                &mut error,
            )
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "CUDA retrieval blend failed",
            )))
        } else {
            Ok(())
        }
    }
}

fn validate_blend_buffers(
    frames: usize,
    dimensions: usize,
    feature_bytes: usize,
    rate_bytes: usize,
) -> Result<(), NativeBuildError> {
    let required = frames
        .checked_mul(dimensions)
        .and_then(|n| n.checked_mul(size_of::<f32>()));
    if frames > i32::MAX as usize
        || required.is_none_or(|bytes| bytes > feature_bytes)
        || rate_bytes < size_of::<f32>()
    {
        return Err(NativeBuildError::Argument(
            "retrieval shape exceeds CUDA grid or buffer capacity".into(),
        ));
    }
    Ok(())
}

impl Drop for GpuRetrievalIndex {
    fn drop(&mut self) {
        unsafe { ffi::vocal_faiss_gpu_destroy(self.raw) }
    }
}
