//! TensorRT-RTX integration for the VFRM RVC graph.
//!
//! Public modules are organized by responsibility:
//!
//! * [`builder`] compiles ONNX or canonical VFRM packages;
//! * [`runtime`] owns deserialized engines, execution contexts, streams, and
//!   device allocations;
//! * [`graph`] validates the source-independent RVC graph contract;
//! * [`buffer`] contains host weight tables and device arenas;
//! * [`ffi`] contains only raw C symbols and opaque handles.

mod abi;
pub mod acoustic;
mod builder;
mod cache_io;
mod config;
mod cufft;
mod error;
mod options;
mod pitch_matching;
mod plan_cache;
mod retrieval;
mod runtime;
mod timing;

pub mod buffer;
pub mod cuda;
pub mod ffi;
pub mod graph;
pub mod pipeline;

pub use builder::NativeTensorRtBuilder;
pub use config::{
    DiagnosticsConfig, GraphReplayMode, PipelineRunConfig, PipelineRuntimeConfig, TensorRtConfig,
};
pub use error::{ConfigError, NativeBuildError};
pub use options::{
    DEFAULT_TRT_ROOT, DispatchTune, NativeBuildOptions, OnnxProfileKind, PrecisionMode,
};
pub use pipeline::{
    PlanBuildProgress, RMVPE_REALTIME_SAMPLES, RVC_F0_CACHE_FRAMES, RvcCudaGraphStats,
    RvcRealtimeConfig, RvcTrtOutput, RvcTrtPipeline, RvcTrtPipelineArtifacts, RvcTrtProfile,
    RvcTrtTimings,
};
pub use retrieval::GpuRetrievalIndex;
pub use runtime::{
    Context, CudaBuffer, CudaGraphCapture, CudaGraphExec, CudaStream, OwnedContext,
    PinnedHostBuffer, PinnedHostRing, Runtime,
};

/// Identifies the statically linked native graph builder and SDK revision.
pub const BUILDER_FINGERPRINT: &str = env!("VOCAL_TRT_BUILDER_FINGERPRINT");

pub(crate) fn cstring(value: &str) -> Result<std::ffi::CString, NativeBuildError> {
    std::ffi::CString::new(value.as_bytes())
        .map_err(|_| NativeBuildError::Argument("tensor name contains NUL".into()))
}
