//! VFRM CPU graphs: Rust model lowering and ownership, standalone oneDNN operators,
//! and runtime-dispatched AVX2 fused elementwise kernels. No Python execution.
mod attention;
mod config;
mod convolution;
mod ffi;
mod fft;
mod linear;
mod pipeline;
mod pitch;
mod planner;
mod quantization;
mod recipes;
mod retrieval;
mod runtime;
mod validation;
pub use attention::{CpuAttentionConfig, CpuAttentionMode};
pub use config::{CpuConfig, CpuF0Config};
pub use convolution::{CpuConvolutionConfig, CpuConvolutionMode};
pub use ffi::CpuMemory;
pub use linear::CpuLinearMode;
pub use pipeline::{CpuPipeline, CpuPipelineOutput, CpuSources};
pub use quantization::{CpuQuantization, CpuQuantizationRule, CpuQuantizedOperator};
pub use recipes::CpuQuantizationPreset;
pub use runtime::{CpuGraph, CpuNodeTiming, CpuOutput};
pub use vocal_graph::{Precision, PrecisionConfig};

/// Version of the loaded standalone oneDNN runtime (not the build SDK path).
pub fn operator_version() -> String {
    let mut version = [0u32; 3];
    // SAFETY: Native function writes exactly three integers to this owned array.
    unsafe {
        ffi::vocal_cpu_version(version.as_mut_ptr());
    }
    format!("{}.{}.{}", version[0], version[1], version[2])
}

#[derive(Debug, thiserror::Error)]
pub enum CpuError {
    #[error("CPU configuration: {0}")]
    Config(String),
    #[error("CPU graph execution: {0}")]
    Runtime(String),
    #[error(transparent)]
    Graph(#[from] vocal_graph::GraphError),
    #[error(transparent)]
    Model(#[from] vocal_model::ModelError),
}
