//! Progress never queries the runtime actor: native engine creation can occupy it for seconds.
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::Serialize;

use crate::events::{BackendEvent, EventSink};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum PreparationPhase {
    #[default]
    Idle,
    Inspecting,
    Extracting,
    LoadingWeights,
    CheckingCache,
    Building,
    Publishing,
    LoadingEngine,
    Ready,
    Failed,
}

impl PreparationPhase {
    fn active(self) -> bool {
        !matches!(self, Self::Idle | Self::Ready | Self::Failed)
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum PreparationKind {
    LoadModel,
    PrepareTensorRt,
    UpdateConfig,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PreparationStatus {
    pub operation_id: u64,
    /// Orders stage updates independently of elapsed time and command snapshots.
    pub revision: u64,
    pub kind: Option<PreparationKind>,
    pub phase: PreparationPhase,
    pub message: String,
    pub elapsed_ms: u64,
    /// Counts completed plans, never an estimate of how long native compilation will take.
    pub completed_steps: u32,
    pub total_steps: Option<u32>,
    pub cache_hit: bool,
}

#[derive(Default)]
struct ProgressState {
    status: PreparationStatus,
    started: Option<Instant>,
}

pub struct PreparationTracker {
    state: Mutex<ProgressState>,
    events: Arc<EventSink>,
}

impl PreparationTracker {
    pub fn new(events: Arc<EventSink>) -> Self {
        Self {
            state: Mutex::new(ProgressState::default()),
            events,
        }
    }

    pub fn snapshot(&self) -> PreparationStatus {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let mut status = state.status.clone();
        if status.phase.active() {
            status.elapsed_ms = elapsed(state.started);
        }
        status
    }

    pub fn begin(self: &Arc<Self>, kind: PreparationKind) -> PreparationOperation {
        let status = {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            state.status = PreparationStatus {
                operation_id: state.status.operation_id.saturating_add(1),
                revision: 1,
                kind: Some(kind),
                phase: PreparationPhase::Inspecting,
                message: "Inspecting the selected model and configuration".into(),
                ..Default::default()
            };
            state.started = Some(Instant::now());
            state.status.clone()
        };
        let id = status.operation_id;
        self.events.publish(BackendEvent::preparation(status));
        PreparationOperation {
            tracker: self.clone(),
            id,
        }
    }
}

pub struct PreparationOperation {
    tracker: Arc<PreparationTracker>,
    id: u64,
}

impl PreparationOperation {
    pub fn report(&self, phase: PreparationPhase, message: impl Into<String>) {
        self.report_steps(phase, message, 0, None);
    }

    pub fn report_steps(
        &self,
        phase: PreparationPhase,
        message: impl Into<String>,
        completed_steps: u32,
        total_steps: Option<u32>,
    ) {
        let status = {
            let mut state = self
                .tracker
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if state.status.operation_id != self.id {
                return;
            }
            state.status.phase = phase;
            state.status.revision = state.status.revision.saturating_add(1);
            state.status.message = message.into();
            state.status.completed_steps = completed_steps;
            state.status.total_steps = total_steps;
            state.status.elapsed_ms = elapsed(state.started);
            state.status.clone()
        };
        self.tracker
            .events
            .publish(BackendEvent::preparation(status));
    }

    pub fn cache_hit(&self, value: bool) {
        let mut state = self
            .tracker
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if state.status.operation_id == self.id {
            state.status.cache_hit = value;
            state.status.revision = state.status.revision.saturating_add(1);
        }
    }
}

impl Drop for PreparationOperation {
    fn drop(&mut self) {
        let unfinished = {
            let state = self
                .tracker
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            state.status.operation_id == self.id && state.status.phase.active()
        };
        if unfinished {
            self.report(
                PreparationPhase::Failed,
                "Preparation was interrupted. Retry the operation.",
            );
        }
    }
}

fn elapsed(started: Option<Instant>) -> u64 {
    started
        .map(|time| time.elapsed().as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interrupted_operations_finish_and_old_handles_cannot_overwrite_new_progress() {
        let tracker = Arc::new(PreparationTracker::new(Arc::new(EventSink::default())));
        let old = tracker.begin(PreparationKind::LoadModel);
        let current = tracker.begin(PreparationKind::PrepareTensorRt);
        drop(old);
        assert_eq!(tracker.snapshot().phase, PreparationPhase::Inspecting);
        current.report_steps(PreparationPhase::Building, "Synth", 4, Some(5));
        assert_eq!(tracker.snapshot().completed_steps, 4);
        drop(current);
        assert_eq!(tracker.snapshot().phase, PreparationPhase::Failed);
    }

    #[test]
    fn cache_hit_and_completed_status_survive_guard_drop() {
        let tracker = Arc::new(PreparationTracker::new(Arc::new(EventSink::default())));
        {
            let operation = tracker.begin(PreparationKind::LoadModel);
            operation.cache_hit(true);
            operation.report(PreparationPhase::Ready, "Model ready");
        }
        let value = serde_json::to_value(tracker.snapshot()).unwrap();
        assert_eq!(value["phase"], "ready");
        assert_eq!(value["cacheHit"], true);
        assert_eq!(value["operationId"], 1);
    }
}
