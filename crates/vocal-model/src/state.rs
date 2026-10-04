use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ModelLoadState {
    Created,
    Loading,
    Ready,
    Failed,
}

impl Default for ModelLoadState {
    fn default() -> Self {
        Self::Created
    }
}

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
