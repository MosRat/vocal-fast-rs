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
mod error;
mod options;
mod runtime;
mod timing;

pub mod buffer;
pub mod cuda;
pub mod ffi;
pub mod graph;
pub mod pipeline;

pub use builder::NativeTensorRtBuilder;
pub use error::NativeBuildError;
pub use options::{
    DEFAULT_TRT_ROOT, DispatchTune, NativeBuildOptions, OnnxProfileKind, PrecisionMode,
};
pub use pipeline::{
    RMVPE_REALTIME_SAMPLES, RVC_F0_CACHE_FRAMES, RvcCudaGraphStats, RvcRealtimeConfig,
    RvcTrtOutput, RvcTrtPipeline, RvcTrtPipelineArtifacts, RvcTrtProfile, RvcTrtTimings,
};
pub use runtime::{
    Context, CudaBuffer, CudaGraphCapture, CudaGraphExec, CudaStream, OwnedContext,
    PinnedHostBuffer, PinnedHostRing, Runtime,
};

pub(crate) fn cstring(value: &str) -> Result<std::ffi::CString, NativeBuildError> {
    std::ffi::CString::new(value.as_bytes())
        .map_err(|_| NativeBuildError::Argument("tensor name contains NUL".into()))
}
