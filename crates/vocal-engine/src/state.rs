use serde::{Deserialize, Serialize};

pub use vocal_types::LifecycleState as EngineLifecycle;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EngineStatus {
    pub state: EngineLifecycle,
    pub processed_chunks: u64,
    pub failures: u64,
    pub last_error: Option<String>,
}

impl EngineStatus {
    pub fn ready(&mut self) {
        self.state = EngineLifecycle::Ready;
        self.last_error = None;
    }

    pub fn begin(&mut self) -> Result<(), &'static str> {
        match self.state {
            EngineLifecycle::Ready | EngineLifecycle::Running => {
                self.state = EngineLifecycle::Running;
                Ok(())
            }
            EngineLifecycle::Created | EngineLifecycle::Loading => {
                Err("engine has not completed initialization")
            }
            EngineLifecycle::Failed => Err("engine is failed"),
            EngineLifecycle::Stopping | EngineLifecycle::Stopped => Err("engine is stopped"),
        }
    }

    pub fn success(&mut self) {
        self.state = EngineLifecycle::Running;
        self.processed_chunks = self.processed_chunks.saturating_add(1);
        self.last_error = None;
    }

    pub fn failure(&mut self, error: impl Into<String>) {
        self.state = EngineLifecycle::Failed;
        self.failures = self.failures.saturating_add(1);
        self.last_error = Some(error.into());
    }
}
