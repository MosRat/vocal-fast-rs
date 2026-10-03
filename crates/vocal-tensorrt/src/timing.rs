//! Stream-ordered GPU timing. Read elapsed times only after stream completion.
use crate::{CudaStream, NativeBuildError, ffi};
use std::{os::raw::c_void, ptr};

pub(crate) struct CudaEvent(*mut c_void);
impl CudaEvent {
    pub fn new() -> Result<Self, NativeBuildError> {
        let (mut raw, mut err) = (ptr::null_mut(), ptr::null_mut());
        let rc = unsafe { ffi::vocal_cuda_event_create(&mut raw, &mut err) };
        check(rc, err)?;
        Ok(Self(raw))
    }
    pub fn record(&self, stream: &CudaStream) -> Result<(), NativeBuildError> {
        let mut err = ptr::null_mut();
        let rc = unsafe { ffi::vocal_cuda_event_record(self.0, stream.device_ptr(), &mut err) };
        check(rc, err)
    }
    pub fn elapsed(&self, end: &Self) -> Result<f64, NativeBuildError> {
        let (mut ms, mut err) = (0.0, ptr::null_mut());
        let rc = unsafe { ffi::vocal_cuda_event_elapsed(self.0, end.0, &mut ms, &mut err) };
        check(rc, err)?;
        Ok(f64::from(ms))
    }
}
fn check(rc: i32, err: *mut std::os::raw::c_char) -> Result<(), NativeBuildError> {
    if rc == 0 {
        Ok(())
    } else {
        Err(NativeBuildError::Api(ffi::native_message(
            err,
            "CUDA event error",
        )))
    }
}
impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe { ffi::vocal_cuda_event_free(self.0) }
    }
}
