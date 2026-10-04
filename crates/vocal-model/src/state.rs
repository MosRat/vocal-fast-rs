use serde::{Deserialize, Serialize};

pub use vocal_types::LifecycleState as ModelLoadState;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelLoadStatus {
    pub state: ModelLoadState,
    pub loads: u64,
    pub failures: u64,
}

impl ModelLoadStatus {
    pub fn begin(&mut self) {
        self.state = ModelLoadState::Loading;
    }

    pub fn ready(&mut self) {
        self.state = ModelLoadState::Ready;
        self.loads = self.loads.saturating_add(1);
    }

    pub fn failed(&mut self) {
        self.state = ModelLoadState::Failed;
        self.failures = self.failures.saturating_add(1);
    }
}
