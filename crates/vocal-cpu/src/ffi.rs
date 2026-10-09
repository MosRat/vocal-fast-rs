use std::ffi::{CStr, c_char, c_void};
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
    pub id: u32,
    pub data: *const u8,
    pub bytes: usize,
}
#[repr(C)]
pub struct Quantization {
    pub id: u32,
    pub mode: u32,
    pub smooth_alpha: f32,
    pub input_max: *const f32,
    pub channels: usize,
}
#[repr(C)]
pub struct Attention {
    pub mode: u32,
    pub query_tile: u32,
    pub key_tile: u32,
    pub token_major_output: bool,
}
#[repr(C)]
pub struct Convolution {
    pub mode: u32,
    pub fuse_gelu: bool,
    pub fuse_residual: bool,
}
#[repr(C)]
#[derive(Default, Clone, Debug, serde::Serialize)]
pub struct CpuMemory {
    pub tensor_bytes: usize,
    pub workspace_bytes: usize,
    pub packed_bytes: usize,
    pub auxiliary_bytes: usize,
    pub tensor_allocations: usize,
    pub workspace_allocations: usize,
}
unsafe extern "C" {
    pub fn vocal_cpu_create(
        nodes: *const Node,
        count: usize,
        outputs: *const u32,
        output_count: usize,
        threads: usize,
        spin_wait_us: u32,
        pin_to_performance_cores: bool,
        avx2: bool,
        profile: bool,
        channels_last: bool,
        fuse: bool,
        fuse_linear_gelu: bool,
        fast_tensor_io: bool,
        linear_mode: u32,
        convolution: *const Convolution,
        attention: *const Attention,
        quantization: *const Quantization,
        quantization_count: usize,
    ) -> *mut c_void;
    pub fn vocal_cpu_calibration(
        graph: *mut c_void,
        id: u32,
        data: *mut f32,
        channels: usize,
    ) -> i32;
    pub fn vocal_cpu_implementation(graph: *mut c_void, id: u32) -> *const c_char;
    pub fn vocal_cpu_run(graph: *mut c_void, bindings: *const Binding, count: usize) -> i32;
    pub fn vocal_cpu_output(graph: *mut c_void, id: u32, data: *mut u8, bytes: usize) -> i32;
    pub fn vocal_cpu_node_ms(graph: *mut c_void, id: u32) -> f64;
    pub fn vocal_cpu_memory(graph: *mut c_void, memory: *mut CpuMemory);
    pub fn vocal_cpu_error() -> *const c_char;
    pub fn vocal_cpu_version(version: *mut u32);
    pub fn vocal_cpu_destroy(graph: *mut c_void);
}
pub fn error() -> crate::CpuError {
    crate::CpuError::Runtime(
        unsafe { CStr::from_ptr(vocal_cpu_error()) }
            .to_string_lossy()
            .into_owned(),
    )
}
