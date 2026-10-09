use std::ffi::{c_char, c_void};

#[repr(C)]
pub struct Node {
    pub op: u32,
    pub dtype: u32,
    pub rank: u32,
    pub shape: [u32; 8],
    pub inputs: *const u32,
    pub input_count: usize,
    pub attrs: *const i32,
    pub attr_count: usize,
    pub values: [f32; 4],
    pub data: *const u8,
    pub data_bytes: usize,
}
#[repr(C)]
pub struct Binding {
    pub data: *mut u8,
    pub bytes: usize,
}

#[repr(C)]
pub struct Feedback {
    pub input: u32,
    pub output: u32,
}

#[repr(C)]
pub struct CompileOptions {
    pub pooled_io: u32,
    pub contiguous_views: u32,
    pub fuse_gemm_bias: u32,
    pub pooled_weights: u32,
    pub weight_staging_mib: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct AdapterDesc {
    pub id: *const c_char,
    pub name: *const c_char,
    pub reason: *const c_char,
    pub dedicated_memory: u64,
    pub shared_memory: u64,
    pub index: u32,
    pub vendor_id: u32,
    pub available: u32,
    pub software: u32,
}

#[cfg(windows)]
unsafe extern "C" {
    pub fn vdml_adapters_enumerate(out: *mut *mut c_void, error: *mut *mut c_char) -> i32;
    pub fn vdml_adapters_len(adapters: *const c_void) -> usize;
    pub fn vdml_adapters_get(adapters: *const c_void, index: usize, out: *mut AdapterDesc);
    pub fn vdml_adapters_free(adapters: *mut c_void);
    pub fn vdml_device_create(
        adapter: u32,
        selected_id: *const c_char,
        debug: u32,
        out: *mut *mut c_void,
        error: *mut *mut c_char,
    ) -> i32;
    pub fn vdml_device_free(device: *mut c_void);
    pub fn vdml_device_name(device: *mut c_void) -> *const c_char;
    pub fn vdml_device_status(device: *mut c_void, error: *mut *mut c_char) -> i32;
    pub fn vdml_compile(
        device: *mut c_void,
        nodes: *const Node,
        count: usize,
        outputs: *const u32,
        output_count: usize,
        feedback: *const Feedback,
        feedback_count: usize,
        options: *const CompileOptions,
        out: *mut *mut c_void,
        error: *mut *mut c_char,
    ) -> i32;
    pub fn vdml_graph_free(graph: *mut c_void);
    pub fn vdml_graph_memory(graph: *const c_void, out: *mut crate::runtime::GraphMemory);
    pub fn vdml_reset_states(graph: *mut c_void, error: *mut *mut c_char) -> i32;
    pub fn vdml_run(
        graph: *mut c_void,
        inputs: *const Binding,
        input_count: usize,
        outputs: *const Binding,
        output_count: usize,
        timings: *mut crate::runtime::GpuTimings,
        error: *mut *mut c_char,
    ) -> i32;
    pub fn vdml_error_free(error: *mut c_char);
}
