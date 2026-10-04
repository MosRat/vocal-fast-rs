//! Safe ownership wrappers for TensorRT runtime/context and CUDA resources.

use std::{collections::HashMap, fs, os::raw::c_void, path::Path, ptr, sync::Arc};

use crate::{error::NativeBuildError, ffi};

/// Deserialized TensorRT engine. A context borrows this value and therefore
/// cannot outlive the engine that owns it.
pub struct Runtime {
    raw: *mut ffi::Runtime,
}

unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl Runtime {
    pub fn shared(path: impl AsRef<Path>) -> Result<Arc<Self>, NativeBuildError> {
        Ok(Arc::new(Self::load(path)?))
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self, NativeBuildError> {
        let bytes = fs::read(path)?;
        let (mut raw, mut error) = (ptr::null_mut(), ptr::null_mut());
        let code = unsafe {
            ffi::vocal_trt_runtime_load(bytes.as_ptr(), bytes.len(), &mut raw, &mut error)
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "runtime load failed",
            )))
        } else if raw.is_null() {
            Err(NativeBuildError::Api(
                "runtime load returned null handle".into(),
            ))
        } else {
            Ok(Self { raw })
        }
    }

    pub fn context(&self) -> Result<Context<'_>, NativeBuildError> {
        let (mut raw, mut error) = (ptr::null_mut(), ptr::null_mut());
        let code = unsafe { ffi::vocal_trt_context_create(self.raw, &mut raw, &mut error) };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "context creation failed",
            )))
        } else if raw.is_null() {
            Err(NativeBuildError::Api(
                "context creation returned null handle".into(),
            ))
        } else {
            Ok(Context {
                raw,
                _runtime: self,
            })
        }
    }

    /// Create a context with TensorRT-RTX's internal CUDA graph strategy.
    /// Strategy `1` is `kWHOLE_GRAPH_CAPTURE`; `false` explicitly disables
    /// the strategy so callers can use the application-level CUDA Graph path.
    pub fn context_with_cuda_graphs(&self, enabled: bool) -> Result<Context<'_>, NativeBuildError> {
        Context::create_with_cuda_graphs(self, enabled)
    }
}

/// A context that owns a reference-counted runtime. This is the form used by
/// multi-engine pipelines, where several engines must live in one struct.
pub struct OwnedContext {
    raw: *mut ffi::Context,
    _runtime: Arc<Runtime>,
    // TensorRT keeps tensor addresses in the execution context.  Repeating
    // the FFI call for every binding on every callback is unnecessary when a
    // fixed profile reuses the same device allocations.  The cache is
    // invalidated by set_input_shape, because a shape transition is the
    // point where callers must establish the complete binding set again.
    bound_addresses: HashMap<String, usize>,
}

unsafe impl Send for OwnedContext {}

impl OwnedContext {
    /// Profiling is deliberately opt-in: TensorRT's layer profiler adds overhead.
    pub fn enable_profiling(&mut self, enabled: bool) {
        unsafe { ffi::vocal_trt_profile_enable(self.raw, enabled as i32) }
    }
    pub fn layer_timings(&mut self) -> Vec<(String, f64)> {
        unsafe extern "C" fn collect(
            user: *mut c_void,
            name: *const std::os::raw::c_char,
            ms: f64,
        ) {
            let rows = unsafe { &mut *(user as *mut Vec<(String, f64)>) };
            rows.push((
                unsafe { std::ffi::CStr::from_ptr(name) }
                    .to_string_lossy()
                    .into_owned(),
                ms,
            ));
        }
        let mut rows = Vec::<(String, f64)>::new();
        unsafe {
            ffi::vocal_trt_profile_read(self.raw, &mut rows as *mut _ as *mut c_void, collect)
        };
        rows
    }
    pub fn create(runtime: Arc<Runtime>) -> Result<Self, NativeBuildError> {
        Self::create_with_cuda_graphs(runtime, false)
    }

    /// Create a context using TensorRT-RTX's JIT CUDA graph strategy when
    /// requested. The strategy owns capture/replay inside `enqueueV3` and does
    /// not require an externally captured CUDA stream.
    pub fn create_with_cuda_graphs(
        runtime: Arc<Runtime>,
        enabled: bool,
    ) -> Result<Self, NativeBuildError> {
        let (mut raw, mut error) = (ptr::null_mut(), ptr::null_mut());
        let code = unsafe {
            ffi::vocal_trt_context_create_with_options(
                runtime.raw,
                enabled as i32,
                &mut raw,
                &mut error,
            )
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "context creation failed",
            )))
        } else if raw.is_null() {
            Err(NativeBuildError::Api(
                "context creation returned null handle".into(),
            ))
        } else {
            Ok(Self {
                raw,
                _runtime: runtime,
                bound_addresses: HashMap::new(),
            })
        }
    }

    /// Query whether the next enqueue can be wrapped by an application-level
    /// CUDA stream capture. TensorRT-RTX internal graph mode generally makes
    /// this unnecessary; the query is used only for the explicit fallback.
    pub fn is_stream_capturable(&mut self, stream: &CudaStream) -> Result<bool, NativeBuildError> {
        let (mut capturable, mut error) = (0_i32, ptr::null_mut());
        let code = unsafe {
            ffi::vocal_trt_context_is_stream_capturable(
                self.raw,
                stream.device_ptr(),
                &mut capturable,
                &mut error,
            )
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "query stream capturability failed",
            )))
        } else {
            Ok(capturable != 0)
        }
    }

    pub fn set_input_shape(&mut self, name: &str, dims: &[i64]) -> Result<(), NativeBuildError> {
        let name = crate::cstring(name)?;
        let mut error = ptr::null_mut();
        let code = unsafe {
            ffi::vocal_trt_set_input_shape(
                self.raw,
                name.as_ptr(),
                dims.as_ptr(),
                i32::try_from(dims.len())
                    .map_err(|_| NativeBuildError::Argument("rank exceeds i32".into()))?,
                &mut error,
            )
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "set input shape failed",
            )))
        } else {
            self.bound_addresses.clear();
            Ok(())
        }
    }

    pub fn set_tensor_address(
        &mut self,
        name: &str,
        address: *mut c_void,
    ) -> Result<(), NativeBuildError> {
        let name = crate::cstring(name)?;
        let mut error = ptr::null_mut();
        let code = unsafe {
            ffi::vocal_trt_set_tensor_address(self.raw, name.as_ptr(), address, &mut error)
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "set tensor address failed",
            )))
        } else {
            Ok(())
        }
    }

    /// Set a binding only when its device pointer changed since the last
    /// shape configuration.  TensorRT retains addresses across enqueue calls,
    /// so this removes repeated name conversion and native FFI work from the
    /// fixed-profile hot path while preserving the explicit setter for callers
    /// that need unconditional rebinding.
    pub fn set_tensor_address_cached(
        &mut self,
        name: &str,
        address: *mut c_void,
    ) -> Result<(), NativeBuildError> {
        let key = address as usize;
        if self.bound_addresses.get(name).copied() == Some(key) {
            return Ok(());
        }
        self.set_tensor_address(name, address)?;
        self.bound_addresses.insert(name.to_owned(), key);
        Ok(())
    }

    pub fn tensor_shape(&mut self, name: &str) -> Result<Vec<i64>, NativeBuildError> {
        let name = crate::cstring(name)?;
        let mut dimensions = [0_i64; 8];
        let (mut rank, mut error) = (0_i32, ptr::null_mut());
        let code = unsafe {
            ffi::vocal_trt_get_tensor_shape(
                self.raw,
                name.as_ptr(),
                dimensions.as_mut_ptr(),
                dimensions.len() as i32,
                &mut rank,
                &mut error,
            )
        };
        if code != 0 {
            return Err(NativeBuildError::Api(ffi::native_message(
                error,
                "get tensor shape failed",
            )));
        }
        Ok(dimensions[..usize::try_from(rank)
            .map_err(|_| NativeBuildError::Api("negative tensor rank".into()))?]
            .to_vec())
    }

    pub fn enqueue(&mut self, stream: &CudaStream) -> Result<(), NativeBuildError> {
        self.enqueue_named(stream, None)
    }

    /// Enqueue while emitting an NVTX host range for Nsight Systems. The range
    /// surrounds only the TensorRT enqueue call; CUDA events remain the source
    /// of GPU elapsed time.
    pub fn enqueue_named(
        &mut self,
        stream: &CudaStream,
        range_name: Option<&str>,
    ) -> Result<(), NativeBuildError> {
        let range_name = range_name.map(crate::cstring).transpose()?;
        let mut error = ptr::null_mut();
        let code = unsafe {
            ffi::vocal_trt_enqueue_named(
                self.raw,
                stream.device_ptr(),
                range_name.as_ref().map_or(ptr::null(), |v| v.as_ptr()),
                &mut error,
            )
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "enqueue failed",
            )))
        } else {
            Ok(())
        }
    }
}

impl Drop for OwnedContext {
    fn drop(&mut self) {
        unsafe { ffi::vocal_trt_context_free(self.raw) }
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe { ffi::vocal_trt_runtime_free(self.raw) }
    }
}

pub struct Context<'runtime> {
    raw: *mut ffi::Context,
    _runtime: &'runtime Runtime,
}

impl Context<'_> {
    pub fn set_input_shape(&mut self, name: &str, dims: &[i64]) -> Result<(), NativeBuildError> {
        let name = crate::cstring(name)?;
        let mut error = ptr::null_mut();
        let code = unsafe {
            ffi::vocal_trt_set_input_shape(
                self.raw,
                name.as_ptr(),
                dims.as_ptr(),
                i32::try_from(dims.len())
                    .map_err(|_| NativeBuildError::Argument("rank exceeds i32".into()))?,
                &mut error,
            )
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "set input shape failed",
            )))
        } else {
            Ok(())
        }
    }

    /// Bind a device allocation returned by [`CudaBuffer::device_ptr`].
    ///
    /// The pointer must remain allocated and refer to the tensor's declared
    /// device type until the next enqueue completes.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn set_tensor_address(
        &mut self,
        name: &str,
        address: *mut c_void,
    ) -> Result<(), NativeBuildError> {
        let name = crate::cstring(name)?;
        let mut error = ptr::null_mut();
        let code = unsafe {
            ffi::vocal_trt_set_tensor_address(self.raw, name.as_ptr(), address, &mut error)
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "set tensor address failed",
            )))
        } else {
            Ok(())
        }
    }

    pub fn tensor_shape(&mut self, name: &str) -> Result<Vec<i64>, NativeBuildError> {
        let name = crate::cstring(name)?;
        let mut dimensions = [0_i64; 8];
        let (mut rank, mut error) = (0_i32, ptr::null_mut());
        let code = unsafe {
            ffi::vocal_trt_get_tensor_shape(
                self.raw,
                name.as_ptr(),
                dimensions.as_mut_ptr(),
                dimensions.len() as i32,
                &mut rank,
                &mut error,
            )
        };
        if code != 0 {
            return Err(NativeBuildError::Api(ffi::native_message(
                error,
                "get tensor shape failed",
            )));
        }
        Ok(dimensions[..usize::try_from(rank)
            .map_err(|_| NativeBuildError::Api("negative tensor rank".into()))?]
            .to_vec())
    }

    pub fn enqueue(&mut self, stream: &CudaStream) -> Result<(), NativeBuildError> {
        self.enqueue_named(stream, None)
    }

    pub fn enqueue_named(
        &mut self,
        stream: &CudaStream,
        range_name: Option<&str>,
    ) -> Result<(), NativeBuildError> {
        let range_name = range_name.map(crate::cstring).transpose()?;
        let mut error = ptr::null_mut();
        let code = unsafe {
            ffi::vocal_trt_enqueue_named(
                self.raw,
                stream.device_ptr(),
                range_name.as_ref().map_or(ptr::null(), |v| v.as_ptr()),
                &mut error,
            )
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "enqueue failed",
            )))
        } else {
            Ok(())
        }
    }
}

impl<'runtime> Context<'runtime> {
    fn create_with_cuda_graphs(
        runtime: &'runtime Runtime,
        enabled: bool,
    ) -> Result<Self, NativeBuildError> {
        let (mut raw, mut error) = (ptr::null_mut(), ptr::null_mut());
        let code = unsafe {
            ffi::vocal_trt_context_create_with_options(
                runtime.raw,
                enabled as i32,
                &mut raw,
                &mut error,
            )
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "context creation failed",
            )))
        } else if raw.is_null() {
            Err(NativeBuildError::Api(
                "context creation returned null handle".into(),
            ))
        } else {
            Ok(Self {
                raw,
                _runtime: runtime,
            })
        }
    }
}

impl Drop for Context<'_> {
    fn drop(&mut self) {
        unsafe { ffi::vocal_trt_context_free(self.raw) }
    }
}

pub struct CudaStream {
    raw: *mut ffi::Stream,
}

impl CudaStream {
    pub fn new() -> Result<Self, NativeBuildError> {
        let (mut raw, mut error) = (ptr::null_mut(), ptr::null_mut());
        let code = unsafe { ffi::vocal_cuda_stream_create(&mut raw, &mut error) };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "cuda stream creation failed",
            )))
        } else if raw.is_null() {
            Err(NativeBuildError::Api(
                "cuda stream creation returned null handle".into(),
            ))
        } else {
            Ok(Self { raw })
        }
    }

    pub fn synchronize(&self) -> Result<(), NativeBuildError> {
        let mut error = ptr::null_mut();
        let code = unsafe { ffi::vocal_cuda_stream_sync(self.raw, &mut error) };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "cuda stream synchronize failed",
            )))
        } else {
            Ok(())
        }
    }

    /// Begin a CUDA stream capture.  All TensorRT enqueues and device copies
    /// issued until [`CudaGraphCapture::finish`] become graph nodes.
    pub fn begin_capture(&self) -> Result<CudaGraphCapture, NativeBuildError> {
        let mut error = ptr::null_mut();
        let code = unsafe { ffi::vocal_cuda_graph_begin(self.raw, &mut error) };
        if code != 0 {
            return Err(NativeBuildError::Api(ffi::native_message(
                error,
                "CUDA graph capture begin failed",
            )));
        }
        Ok(CudaGraphCapture {
            stream: self.raw,
            active: true,
        })
    }

    pub(crate) fn device_ptr(&self) -> *mut c_void {
        unsafe { ffi::vocal_cuda_stream_ptr(self.raw) }
    }

    pub(crate) fn native_handle(&self) -> *mut ffi::Stream {
        self.raw
    }
}

/// RAII guard for an active CUDA stream capture.  Dropping an unfinished
/// capture aborts it so a TensorRT enqueue error cannot poison the stream for
/// the next realtime block.
pub struct CudaGraphCapture {
    stream: *mut ffi::Stream,
    active: bool,
}

impl CudaGraphCapture {
    pub fn finish(mut self) -> Result<CudaGraphExec, NativeBuildError> {
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let code = unsafe { ffi::vocal_cuda_graph_end(self.stream, &mut raw, &mut error) };
        self.active = false;
        if code != 0 {
            return Err(NativeBuildError::Api(ffi::native_message(
                error,
                "CUDA graph capture end failed",
            )));
        }
        if raw.is_null() {
            return Err(NativeBuildError::Api(
                "CUDA graph capture returned null executable".into(),
            ));
        }
        Ok(CudaGraphExec { raw })
    }
}

impl Drop for CudaGraphCapture {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut error = ptr::null_mut();
        unsafe {
            let _ = ffi::vocal_cuda_graph_abort(self.stream, &mut error);
            if !error.is_null() {
                ffi::vocal_trt_free_error(error);
            }
        }
    }
}

/// Instantiated CUDA graph executable.  It is shape- and pointer-specific;
/// callers must recapture after changing TensorRT dynamic shapes or bindings.
pub struct CudaGraphExec {
    raw: *mut ffi::CudaGraphExec,
}

unsafe impl Send for CudaGraphExec {}

impl CudaGraphExec {
    pub fn launch(&self, stream: &CudaStream) -> Result<(), NativeBuildError> {
        let mut error = ptr::null_mut();
        let code = unsafe { ffi::vocal_cuda_graph_launch(self.raw, stream.raw, &mut error) };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "CUDA graph launch failed",
            )))
        } else {
            Ok(())
        }
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        unsafe { ffi::vocal_cuda_graph_exec_destroy(self.raw) }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe { ffi::vocal_cuda_stream_destroy(self.raw) }
    }
}

pub struct CudaBuffer {
    raw: *mut ffi::Memory,
    pub bytes: usize,
}

/// Page-locked host allocation suitable for `cudaMemcpyAsync`. The allocation
/// is owned by Rust and remains live until the stream has consumed its bytes.
pub struct PinnedHostBuffer {
    raw: *mut ffi::HostMemory,
    pub bytes: usize,
}

unsafe impl Send for PinnedHostBuffer {}

impl PinnedHostBuffer {
    pub fn new(bytes: usize) -> Result<Self, NativeBuildError> {
        if bytes == 0 {
            return Err(NativeBuildError::Argument(
                "pinned host allocation size is zero".into(),
            ));
        }
        let (mut raw, mut error) = (ptr::null_mut(), ptr::null_mut());
        let code = unsafe { ffi::vocal_cuda_host_alloc(bytes, &mut raw, &mut error) };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "pinned host allocation failed",
            )))
        } else if raw.is_null() {
            Err(NativeBuildError::Api(
                "pinned host allocation returned null handle".into(),
            ))
        } else {
            Ok(Self { raw, bytes })
        }
    }

    pub fn copy_from_f32(&mut self, data: &[f32]) -> Result<(), NativeBuildError> {
        let bytes = std::mem::size_of_val(data);
        if bytes > self.bytes {
            return Err(NativeBuildError::Argument(
                "pinned host copy exceeds allocation".into(),
            ));
        }
        if !cfg!(target_endian = "little") {
            return Err(NativeBuildError::Argument(
                "pinned host f32 staging requires a little-endian host".into(),
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                ffi::vocal_cuda_host_ptr(self.raw) as *mut u8,
                bytes,
            );
        }
        Ok(())
    }

    fn raw(&self) -> *mut ffi::HostMemory {
        self.raw
    }

    fn copy_to_bytes(&self, destination: &mut [u8]) -> Result<(), NativeBuildError> {
        if destination.len() > self.bytes {
            return Err(NativeBuildError::Argument(
                "pinned host copy exceeds allocation".into(),
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                ffi::vocal_cuda_host_ptr(self.raw) as *const u8,
                destination.as_mut_ptr(),
                destination.len(),
            );
        }
        Ok(())
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        unsafe { ffi::vocal_cuda_host_free(self.raw) }
    }
}

/// Small rotating staging pool. A stream synchronization at the end of each
/// pipeline call makes a three-slot ring sufficient for producer/consumer
/// overlap while also avoiding a heap allocation per callback.
pub struct PinnedHostRing {
    slots: Vec<PinnedHostBuffer>,
    next: usize,
}

impl PinnedHostRing {
    pub fn new(slot_count: usize, bytes: usize) -> Result<Self, NativeBuildError> {
        if slot_count == 0 {
            return Err(NativeBuildError::Argument(
                "pinned host ring must contain at least one slot".into(),
            ));
        }
        let mut slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            slots.push(PinnedHostBuffer::new(bytes)?);
        }
        Ok(Self { slots, next: 0 })
    }

    pub fn upload_f32(
        &mut self,
        destination: &CudaBuffer,
        data: &[f32],
        stream: &CudaStream,
    ) -> Result<(), NativeBuildError> {
        let index = self.next;
        self.next = (self.next + 1) % self.slots.len();
        let slot = &mut self.slots[index];
        slot.copy_from_f32(data)?;
        destination.upload_async_pinned(slot, std::mem::size_of_val(data), stream)
    }

    /// Queue a device-to-host copy into the next pinned slot. The caller must
    /// wait for `stream` before copying the slot into pageable output memory;
    /// the pipeline does that once for the complete graph and the final D2H.
    pub fn download_bytes_async(
        &mut self,
        source: &CudaBuffer,
        bytes: usize,
        stream: &CudaStream,
    ) -> Result<usize, NativeBuildError> {
        let index = self.next;
        self.next = (self.next + 1) % self.slots.len();
        let slot = &self.slots[index];
        if bytes > slot.bytes {
            return Err(NativeBuildError::Argument(
                "pinned host download exceeds allocation".into(),
            ));
        }
        source.download_async_pinned(slot, bytes, stream)?;
        Ok(index)
    }

    /// Copy a completed pinned slot into the caller's owned output buffer.
    pub fn copy_bytes(&self, index: usize, destination: &mut [u8]) -> Result<(), NativeBuildError> {
        let slot = self.slots.get(index).ok_or_else(|| {
            NativeBuildError::Argument("pinned host ring slot index out of range".into())
        })?;
        slot.copy_to_bytes(destination)
    }
}

impl CudaBuffer {
    pub fn new(bytes: usize) -> Result<Self, NativeBuildError> {
        if bytes == 0 {
            return Err(NativeBuildError::Argument(
                "CUDA allocation size is zero".into(),
            ));
        }
        let (mut raw, mut error) = (ptr::null_mut(), ptr::null_mut());
        let code = unsafe { ffi::vocal_cuda_malloc(bytes, &mut raw, &mut error) };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "cuda allocation failed",
            )))
        } else if raw.is_null() {
            Err(NativeBuildError::Api(
                "cuda allocation returned null handle".into(),
            ))
        } else {
            Ok(Self { raw, bytes })
        }
    }

    pub fn device_ptr(&self) -> *mut c_void {
        unsafe { ffi::vocal_cuda_ptr(self.raw) }
    }

    pub(crate) fn native_handle(&self) -> *mut ffi::Memory {
        self.raw
    }

    pub(crate) fn device_ptr_offset(&self, offset_bytes: usize) -> *mut c_void {
        assert!(
            offset_bytes <= self.bytes,
            "CUDA buffer pointer offset exceeds allocation"
        );
        unsafe { (self.device_ptr() as *mut u8).add(offset_bytes) as *mut c_void }
    }

    pub fn zero(&self) -> Result<(), NativeBuildError> {
        let mut error = ptr::null_mut();
        let code = unsafe { ffi::vocal_cuda_memset(self.raw, 0, self.bytes, &mut error) };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "cuda memset failed",
            )))
        } else {
            Ok(())
        }
    }

    pub fn upload_bytes(&self, data: &[u8]) -> Result<(), NativeBuildError> {
        if data.len() > self.bytes {
            return Err(NativeBuildError::Argument(
                "host upload exceeds CUDA buffer".into(),
            ));
        }
        let mut error = ptr::null_mut();
        let code = unsafe {
            ffi::vocal_cuda_upload(
                self.raw,
                data.as_ptr() as *const c_void,
                data.len(),
                &mut error,
            )
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "cuda upload failed",
            )))
        } else {
            Ok(())
        }
    }

    /// Upload a native-endian f32 slice without first allocating a temporary
    /// byte vector. All TensorRT pipeline public float bindings are FP32, and
    /// the CUDA shim copies the exact byte representation to the device.
    pub fn upload_f32(&self, data: &[f32]) -> Result<(), NativeBuildError> {
        if !cfg!(target_endian = "little") {
            let bytes = data
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            return self.upload_bytes(&bytes);
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        self.upload_bytes(bytes)
    }

    /// Upload one i32 shape/control value without heap allocation.
    pub fn upload_i32(&self, value: i32) -> Result<(), NativeBuildError> {
        let bytes = value.to_le_bytes();
        self.upload_bytes(&bytes)
    }

    pub(crate) fn upload_async_pinned(
        &self,
        source: &PinnedHostBuffer,
        bytes: usize,
        stream: &CudaStream,
    ) -> Result<(), NativeBuildError> {
        if bytes > self.bytes || bytes > source.bytes {
            return Err(NativeBuildError::Argument(
                "async pinned upload exceeds allocation".into(),
            ));
        }
        let mut error = ptr::null_mut();
        let code = unsafe {
            ffi::vocal_cuda_upload_async(
                self.raw,
                0,
                source.raw(),
                0,
                bytes,
                stream.raw,
                &mut error,
            )
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "async pinned CUDA upload failed",
            )))
        } else {
            Ok(())
        }
    }

    pub(crate) fn download_async_pinned(
        &self,
        destination: &PinnedHostBuffer,
        bytes: usize,
        stream: &CudaStream,
    ) -> Result<(), NativeBuildError> {
        if bytes > self.bytes || bytes > destination.bytes {
            return Err(NativeBuildError::Argument(
                "async pinned download exceeds allocation".into(),
            ));
        }
        let mut error = ptr::null_mut();
        let code = unsafe {
            ffi::vocal_cuda_download_async(
                self.raw,
                0,
                destination.raw,
                0,
                bytes,
                stream.raw,
                &mut error,
            )
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "async pinned CUDA download failed",
            )))
        } else {
            Ok(())
        }
    }

    pub fn download_bytes(&self, data: &mut [u8]) -> Result<(), NativeBuildError> {
        if data.len() > self.bytes {
            return Err(NativeBuildError::Argument(
                "host download exceeds CUDA buffer".into(),
            ));
        }
        let mut error = ptr::null_mut();
        let code = unsafe {
            ffi::vocal_cuda_download(
                self.raw,
                data.as_mut_ptr() as *mut c_void,
                data.len(),
                &mut error,
            )
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "cuda download failed",
            )))
        } else {
            Ok(())
        }
    }

    pub(crate) fn device_copy_from(
        &self,
        source: &Self,
        destination_offset: usize,
        source_offset: usize,
        bytes: usize,
        stream: &CudaStream,
    ) -> Result<(), NativeBuildError> {
        let mut error = ptr::null_mut();
        let code = unsafe {
            ffi::vocal_cuda_device_copy(
                self.raw,
                destination_offset,
                source.raw,
                source_offset,
                bytes,
                stream.raw,
                &mut error,
            )
        };
        if code != 0 {
            Err(NativeBuildError::Api(ffi::native_message(
                error,
                "CUDA device copy failed",
            )))
        } else {
            Ok(())
        }
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe { ffi::vocal_cuda_free(self.raw) }
    }
}
