//! Shared application runtime used by CLI and Tauri controllers.
//!
//! The public context is a bounded control-channel handle. A single actor
//! owns lifecycle/configuration state and the idle engine. During an active
//! audio session the engine is transferred to the model worker, while capture
//! and render remain on their own SPSC rings. Controllers provide presentation
//! and transport: Tauri owns device views, while CLI owns argument parsing and
//! output.

mod capabilities;
mod config;
#[cfg(feature = "cpu-graph")]
mod cpu;
#[cfg(feature = "directml")]
mod directml;
mod error;
mod memory;
pub mod model_library;
mod state;

pub use capabilities::{
    BackendAvailability, CpuCapabilities, ProcessingBackendCatalog, ProcessingDevice,
    ProcessingDeviceKind, detect_processing_backends, detect_tensor_rt_backend,
};
pub use config::{
    ConfigStore, ConfigUpdateKind, ModelBackendSelection, RUNTIME_CONFIG_SCHEMA_VERSION,
    RuntimeBackend, RuntimeConfig, TensorRtPipelineConfig, classify_config_change,
};
pub use error::{RuntimeError, RuntimeResult};
pub use state::{
    ModelCatalog, ModelCatalogEntry, ModelCatalogIssue, ModelSummary, RuntimeContext,
    RuntimeEngineStatus, RuntimeErrorInfo, RuntimeMetrics, RuntimeSnapshot,
};
pub use vocal_model::ModelArtifactKind;
