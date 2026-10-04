use serde::{Deserialize, Serialize};

/// Common lifecycle vocabulary used by model, engine and device adapters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum LifecycleState {
    #[default]
    Created,
    Loading,
    Ready,
    Running,
    Stopping,
    Stopped,
    Failed,
}

impl LifecycleState {
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Stopped | Self::Failed)
    }

    pub fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Created, Self::Loading | Self::Failed)
                | (Self::Loading, Self::Ready | Self::Failed)
                | (Self::Ready, Self::Running | Self::Stopping | Self::Failed)
                | (Self::Running, Self::Stopping | Self::Failed)
                | (Self::Stopping, Self::Stopped | Self::Failed)
        ) || self == next
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProcessingStatus {
    pub state: LifecycleState,
    pub operations: u64,
    pub failures: u64,
}

impl ProcessingStatus {
    pub fn transition(&mut self, next: LifecycleState) -> bool {
        if !self.state.can_transition_to(next) {
            return false;
        }
        self.state = next;
        true
    }

    pub fn record_success(&mut self) {
        self.operations = self.operations.saturating_add(1);
    }

    pub fn record_failure(&mut self) {
        self.failures = self.failures.saturating_add(1);
        self.state = LifecycleState::Failed;
    }
}
