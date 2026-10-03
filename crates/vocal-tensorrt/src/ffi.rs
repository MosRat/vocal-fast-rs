//! Raw symbols exported by the C++ TensorRT/CUDA shim.
//!
//! This module is intentionally boring: no ownership or policy lives here.
//! Safe wrappers in `runtime` and `builder` are the only consumers.

use std::{
    ffi::CStr,
    os::raw::{c_char, c_void},
};

use crate::abi::{NativeConfig, NativeOptions, NativeWeight};

/// Opaque native handle categories exposed by the C ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandleKind {
    Builder,
    Runtime,
    Context,
    Stream,
    DeviceBuffer,
}

#[repr(C)]
pub(crate) struct Build {
    _private: [u8; 0],
}
#[repr(C)]
pub(crate) struct Runtime {
    _private: [u8; 0],
}
#[repr(C)]
pub(crate) struct Context {
    _private: [u8; 0],
}
#[repr(C)]
pub(crate) struct Memory {
    _private: [u8; 0],
}
#[repr(C)]
pub(crate) struct HostMemory {
    _private: [u8; 0],
}
#[repr(C)]
pub(crate) struct Stream {
    _private: [u8; 0],
}
#[repr(C)]
pub(crate) struct CudaGraphExec {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub(crate) fn vocal_cuda_event_create(out: *mut *mut c_void, error: *mut *mut c_char) -> i32;
    pub(crate) fn vocal_cuda_event_free(event: *mut c_void);
    pub(crate) fn vocal_cuda_event_record(
        event: *mut c_void,
        stream: *mut c_void,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_cuda_event_elapsed(
        start: *mut c_void,
        end: *mut c_void,
        ms: *mut f32,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_trt_profile_enable(context: *mut Context, enabled: i32);
    pub(crate) fn vocal_trt_profile_read(
        context: *mut Context,
        user: *mut c_void,
        callback: unsafe extern "C" fn(*mut c_void, *const c_char, f64),
    );
    #[allow(dead_code)]
    pub(crate) fn vocal_trt_build_onnx(
        path: *const c_char,
        workspace: u64,
        fp16: i32,
        min_frames: i32,
        opt_frames: i32,
        max_frames: i32,
        out: *mut *mut Build,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_trt_build_onnx_profile(
        path: *const c_char,
        workspace: u64,
        fp16: i32,
        min_frames: i32,
        opt_frames: i32,
        max_frames: i32,
        min_samples: i32,
        opt_samples: i32,
        max_samples: i32,
        profile_kind: i32,
        out: *mut *mut Build,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_trt_build_rvc_frontend(
        workspace: u64,
        min_frames: i32,
        opt_frames: i32,
        max_frames: i32,
        out: *mut *mut Build,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_trt_build_rvc(
        config: *const NativeConfig,
        weights: *const NativeWeight,
        count: usize,
        options: *const NativeOptions,
        out: *mut *mut Build,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_trt_build_hubert(
        weights: *const NativeWeight,
        count: usize,
        options: *const NativeOptions,
        out: *mut *mut Build,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_trt_build_rmvpe(
        weights: *const NativeWeight,
        count: usize,
        options: *const NativeOptions,
        out: *mut *mut Build,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_trt_build_bytes(
        handle: *mut Build,
        data: *mut *mut u8,
        bytes: *mut usize,
    ) -> i32;
    pub(crate) fn vocal_trt_build_free(handle: *mut Build);
    pub(crate) fn vocal_trt_free_bytes(data: *mut u8, bytes: usize);
    pub(crate) fn vocal_trt_free_error(error: *mut c_char);

    pub(crate) fn vocal_trt_runtime_load(
        data: *const u8,
        bytes: usize,
        out: *mut *mut Runtime,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_trt_runtime_free(handle: *mut Runtime);
    pub(crate) fn vocal_trt_context_create(
        runtime: *mut Runtime,
        out: *mut *mut Context,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_trt_context_create_with_options(
        runtime: *mut Runtime,
        cuda_graph_strategy: i32,
        out: *mut *mut Context,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_trt_context_free(handle: *mut Context);
    pub(crate) fn vocal_trt_context_is_stream_capturable(
        context: *mut Context,
        stream: *mut c_void,
        capturable: *mut i32,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_trt_set_input_shape(
        context: *mut Context,
        name: *const c_char,
        dims: *const i64,
        rank: i32,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_trt_set_tensor_address(
        context: *mut Context,
        name: *const c_char,
        address: *mut c_void,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_trt_get_tensor_shape(
        context: *mut Context,
        name: *const c_char,
        dimensions: *mut i64,
        capacity: i32,
        rank: *mut i32,
        error: *mut *mut c_char,
    ) -> i32;
    #[allow(dead_code)]
    pub(crate) fn vocal_trt_enqueue(
        context: *mut Context,
        stream: *mut c_void,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_trt_enqueue_named(
        context: *mut Context,
        stream: *mut c_void,
        range_name: *const c_char,
        error: *mut *mut c_char,
    ) -> i32;

    pub(crate) fn vocal_cuda_stream_create(out: *mut *mut Stream, error: *mut *mut c_char) -> i32;
    pub(crate) fn vocal_cuda_stream_destroy(handle: *mut Stream);
    pub(crate) fn vocal_cuda_stream_sync(handle: *mut Stream, error: *mut *mut c_char) -> i32;
    pub(crate) fn vocal_cuda_stream_ptr(handle: *mut Stream) -> *mut c_void;
    pub(crate) fn vocal_cuda_graph_begin(stream: *mut Stream, error: *mut *mut c_char) -> i32;
    pub(crate) fn vocal_cuda_graph_end(
        stream: *mut Stream,
        out: *mut *mut CudaGraphExec,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_cuda_graph_abort(stream: *mut Stream, error: *mut *mut c_char) -> i32;
    pub(crate) fn vocal_cuda_graph_launch(
        graph: *mut CudaGraphExec,
        stream: *mut Stream,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_cuda_graph_exec_destroy(graph: *mut CudaGraphExec);
    pub(crate) fn vocal_cuda_malloc(
        bytes: usize,
        out: *mut *mut Memory,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_cuda_free(handle: *mut Memory);
    pub(crate) fn vocal_cuda_ptr(handle: *mut Memory) -> *mut c_void;
    pub(crate) fn vocal_cuda_host_alloc(
        bytes: usize,
        out: *mut *mut HostMemory,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_cuda_host_free(handle: *mut HostMemory);
    pub(crate) fn vocal_cuda_host_ptr(handle: *mut HostMemory) -> *mut c_void;
    pub(crate) fn vocal_cuda_upload_async(
        destination: *mut Memory,
        destination_offset: usize,
        source: *mut HostMemory,
        source_offset: usize,
        bytes: usize,
        stream: *mut Stream,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_cuda_memset(
        handle: *mut Memory,
        value: i32,
        bytes: usize,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_cuda_upload(
        handle: *mut Memory,
        source: *const c_void,
        bytes: usize,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_cuda_download(
        handle: *mut Memory,
        destination: *mut c_void,
        bytes: usize,
        error: *mut *mut c_char,
    ) -> i32;
    pub(crate) fn vocal_cuda_device_copy(
        destination: *mut Memory,
        destination_offset: usize,
        source: *mut Memory,
        source_offset: usize,
        bytes: usize,
        stream: *mut Stream,
        error: *mut *mut c_char,
    ) -> i32;
}

pub(crate) fn native_message(error: *mut c_char, fallback: &str) -> String {
    if error.is_null() {
        fallback.into()
    } else {
        let message = unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned();
        unsafe { vocal_trt_free_error(error) };
        message
    }
}

pub const ABI_VERSION: u32 = 2;
