//! Shared application runtime used by CLI and Tauri controllers.
//!
//! This crate owns configuration, model loading, engine construction and the
//! processing state machine. Controllers provide presentation and transport:
//! Tauri owns channels/device views, while CLI owns argument parsing/output.

mod config;
mod error;
mod state;

pub use config::{ConfigStore, RUNTIME_CONFIG_SCHEMA_VERSION, RuntimeBackend, RuntimeConfig};
pub use error::{RuntimeError, RuntimeResult};
pub use state::{
    ModelSummary, RuntimeContext, RuntimeEngineStatus, RuntimeErrorInfo, RuntimeMetrics,
    RuntimeSnapshot,
};
