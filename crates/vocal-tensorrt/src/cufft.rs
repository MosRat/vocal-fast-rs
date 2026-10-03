//! cuFFT-backed RMVPE STFT frontend.
//!
//! The frontend owns one fixed-size cuFFT plan and scratch buffers. It emits
//! an FP32 `[513, frames]` magnitude tensor directly into device memory; the
//! RMVPE TensorRT graph consumes that tensor without a layout transpose and
//! still owns mel projection, log compression, and all learned layers.

use std::ptr;

use crate::{
    error::NativeBuildError,
    ffi,
    runtime::{CudaBuffer, CudaStream},
};

pub(crate) struct CufftFrontend {
    raw: *mut ffi::CufftFrontend,
}

const MIN_REFLECT_SAMPLES: usize = 1024;

unsafe impl Send for CufftFrontend {}

impl CufftFrontend {
    pub(crate) fn new(samples: usize, stream: &CudaStream) -> Result<Self, NativeBuildError> {
        if samples < MIN_REFLECT_SAMPLES || samples % 160 != 0 {
            return Err(NativeBuildError::Argument(
                "cuFFT frontend samples must be a 160-sample multiple >= 1024".into(),
            ));
        }
        let samples_i32 = i32::try_from(samples)
            .map_err(|_| NativeBuildError::Argument("cuFFT sample count exceeds i32".into()))?;
        let (mut raw, mut error) = (ptr::null_mut(), ptr::null_mut());
        let code = unsafe { ffi::vocal_cufft_frontend_create(samples_i32, &mut raw, &mut error) };
        if code != 0 {
            return Err(NativeBuildError::Api(ffi::native_message(
                error,
                "cuFFT frontend creation failed",
            )));
        }
        if raw.is_null() {
            return Err(NativeBuildError::Api(
                "cuFFT frontend creation returned null handle".into(),
            ));
        }
        let mut result = Self { raw };
        if let Err(error) = result.set_stream(stream) {
            unsafe { ffi::vocal_cufft_frontend_destroy(result.raw) };
            result.raw = ptr::null_mut();
            return Err(error);
        }
        Ok(result)
    }

    fn set_stream(&mut self, stream: &CudaStream) -> Result<(), NativeBuildError> {
        let mut error = ptr::null_mut();
        let code = unsafe {
            ffi::vocal_cufft_frontend_set_stream(self.raw, stream.native_handle(), &mut error)
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "cuFFT frontend stream setup failed",
            )))
        } else {
            Ok(())
        }
    }

    pub(crate) fn enqueue(
        &mut self,
        waveform: *const std::ffi::c_void,
        waveform_bytes: usize,
        magnitude: &CudaBuffer,
        stream: &CudaStream,
    ) -> Result<(), NativeBuildError> {
        let mut error = ptr::null_mut();
        let code = unsafe {
            ffi::vocal_cufft_frontend_enqueue(
                self.raw,
                waveform,
                waveform_bytes,
                magnitude.native_handle(),
                stream.native_handle(),
                &mut error,
            )
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "cuFFT frontend enqueue failed",
            )))
        } else {
            Ok(())
        }
    }
}

impl Drop for CufftFrontend {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::vocal_cufft_frontend_destroy(self.raw) };
        }
    }
}

/// Return the unique fixed shapes needed by the current realtime/full-window
/// profile. Keeping plan creation outside the hot path avoids cuFFT plan
/// rebuilds and makes the stage safe to include in explicit CUDA graph capture.
pub(crate) fn unique_shapes(values: impl IntoIterator<Item = usize>) -> Vec<usize> {
    let mut result = Vec::new();
    for value in values {
        if value > 0 && !result.contains(&value) {
            result.push(value);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::unique_shapes;

    #[test]
    fn fixed_plan_shapes_are_deduplicated_in_stable_order() {
        assert_eq!(
            unique_shapes([4_960, 44_960, 4_960, 11_200, 0, 44_960]),
            vec![4_960, 44_960, 11_200]
        );
    }

    #[test]
    fn empty_and_zero_shape_inputs_do_not_create_plans() {
        assert!(unique_shapes([0, 0]).is_empty());
        assert!(unique_shapes(std::iter::empty()).is_empty());
    }
}
