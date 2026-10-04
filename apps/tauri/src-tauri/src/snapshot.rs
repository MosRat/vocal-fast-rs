//! Stable view models exposed to the Tauri control plane.
//!
//! Runtime ownership stays in [`crate::state`]. Keeping these DTOs in a
//! separate module prevents command handlers and the state machine from
//! becoming coupled to the JSON shape consumed by the frontend.

use serde::Serialize;
use vocal_engine::EngineStatus;
use vocal_model::RvcModelManifest;
use vocal_types::{LatencyMetrics, LifecycleState};

use crate::{
    audio::AudioTelemetry,
    config::GuiConfig,
    events::ErrorReport,
};

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSummary {
    pub id: String,
    pub display_name: String,
    pub architecture: String,
    pub version: String,
    pub sample_rate: u32,
    pub feature_dims: usize,
    pub uses_f0: bool,
}

impl From<&RvcModelManifest> for ModelSummary {
    fn from(manifest: &RvcModelManifest) -> Self {
        Self {
            id: manifest.id.clone(),
            display_name: manifest.display_name.clone(),
            architecture: manifest.architecture.clone(),
            version: manifest.version.clone(),
            sample_rate: manifest.sample_rate,
            feature_dims: manifest.feature_dims(),
            uses_f0: manifest.uses_f0(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UiSnapshot {
    pub lifecycle: LifecycleState,
    /// Per-subsystem states avoid making the aggregate lifecycle carry model,
    /// engine and device failures at the same time. `lifecycle` is retained
    /// as the compact headline status used by older clients.
    pub status: UiStatus,
    pub config: GuiConfig,
    pub config_revision: u64,
    pub model: Option<ModelSummary>,
    pub metrics: UiMetrics,
    pub engine: Option<UiEngineStatus>,
    pub last_error: Option<ErrorReport>,
    pub audio: AudioTelemetry,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UiStatus {
    pub session: LifecycleState,
    pub model: LifecycleState,
    pub engine: LifecycleState,
    pub audio: LifecycleState,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UiMetrics {
    pub input_frames: u64,
    pub processing_micros: u64,
    pub output_frames: u64,
}

impl From<LatencyMetrics> for UiMetrics {
    fn from(metrics: LatencyMetrics) -> Self {
        Self {
            input_frames: metrics.input_frames,
            processing_micros: metrics.processing_micros,
            output_frames: metrics.output_frames,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UiEngineStatus {
    pub state: LifecycleState,
    pub processed_chunks: u64,
    pub failures: u64,
    pub last_error: Option<String>,
}

impl From<EngineStatus> for UiEngineStatus {
    fn from(status: EngineStatus) -> Self {
        Self {
            state: status.state,
            processed_chunks: status.processed_chunks,
            failures: status.failures,
            last_error: status.last_error,
        }
    }
}
