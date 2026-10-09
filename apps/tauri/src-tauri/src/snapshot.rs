//! Stable view models exposed to the Tauri control plane.
//!
//! Runtime ownership stays in [`crate::state`]. Keeping these DTOs in a
//! separate module prevents command handlers and the state machine from
//! becoming coupled to the JSON shape consumed by the frontend.

use serde::Serialize;
use vocal_types::LifecycleState;

use crate::{audio::AudioTelemetry, config::GuiConfig, events::ErrorReport};

pub use vocal_runtime::ModelSummary;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UiSnapshot {
    pub boot: UiBootStatus,
    pub lifecycle: LifecycleState,
    /// Per-subsystem states avoid making the aggregate lifecycle carry model,
    /// engine and device failures at the same time. `lifecycle` is retained
    /// as the compact headline status used by older clients.
    pub status: UiStatus,
    pub config: GuiConfig,
    pub config_revision: u64,
    pub last_config_update: vocal_runtime::ConfigUpdateKind,
    pub model: Option<ModelSummary>,
    pub engine: Option<UiEngineStatus>,
    pub last_error: Option<ErrorReport>,
    pub audio: AudioTelemetry,
    pub tensorrt_pipeline: TensorRtPipelineStatus,
    pub preparation: crate::preparation::PreparationStatus,
    pub model_restore_pending: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum TensorRtPipelinePhase {
    #[default]
    Idle,
    CheckingCache,
    Ready,
    Failed,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TensorRtPipelineStatus {
    pub phase: TensorRtPipelinePhase,
    pub message: Option<String>,
    pub artifacts_path: Option<String>,
    pub cache_hit: bool,
    pub elapsed_ms: u64,
}

/// Application startup is intentionally separate from the runtime lifecycle.
/// Ready means the shell and device/file catalogs are initialized. Model
/// weights, cached plans and execution contexts belong to model preparation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum BootPhase {
    Starting,
    RestoringConfig,
    InitializingIo,
    DetectingDevices,
    Ready,
    Failed,
    ShuttingDown,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UiBootStatus {
    pub revision: u64,
    pub phase: BootPhase,
    pub message: Option<String>,
    pub elapsed_ms: u64,
    pub recoverable: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UiStatus {
    pub session: LifecycleState,
    pub model: LifecycleState,
    pub engine: LifecycleState,
    pub audio: LifecycleState,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UiEngineStatus {
    pub state: LifecycleState,
    pub last_error: Option<String>,
}

impl From<vocal_runtime::RuntimeEngineStatus> for UiEngineStatus {
    fn from(status: vocal_runtime::RuntimeEngineStatus) -> Self {
        Self {
            state: status.state,
            last_error: status.last_error,
        }
    }
}
