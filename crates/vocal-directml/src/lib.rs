//! Native DirectML graphs built from canonical VFRM weights.
//! Rust owns model contracts and precision; D3D12/DirectML own GPU execution.
mod adapters;
mod config;
mod error;
mod f0;
mod ffi;
mod graph;
pub mod models;
mod offline;
pub mod pipeline;
mod pitch_matching;
mod runtime;
mod sources;
mod stages;

pub use adapters::{DirectMlAdapter, enumerate_adapters};
pub use config::{
    DirectMlConfig, ExecutionConfig, F0Config, IoBuffers, Precision, PrecisionConfig,
    RetrievalSearch,
};
pub use error::DirectMlError;
pub use offline::{DirectMlOfflineGraph, DirectMlOfflineOutput};
pub use pipeline::{DirectMlControls, DirectMlPipeline, DirectMlPipelineOutput, DirectMlSources};
pub use runtime::{DirectMlDevice, DirectMlGraph, GpuTimings, GraphMemory, GraphOutput};
