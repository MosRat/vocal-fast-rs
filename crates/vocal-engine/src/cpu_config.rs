//! Portable CPU execution policy shared by controllers, independent of the SDK.
use serde::{Deserialize, Serialize};

use crate::EngineError;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct CpuExecutionConfig {
    /// Zero selects a bounded pool instead of taking all logical processors.
    pub threads: usize,
    pub spin_wait_us: u32,
    pub pin_to_performance_cores: bool,
    pub profile: bool,
}

impl Default for CpuExecutionConfig {
    fn default() -> Self {
        Self {
            threads: 0,
            spin_wait_us: 0,
            pin_to_performance_cores: false,
            profile: false,
        }
    }
}

impl CpuExecutionConfig {
    pub fn recommended_threads() -> usize {
        std::thread::available_parallelism().map_or(1, |n| n.get().min(6))
    }

    pub fn resolved_threads(&self) -> usize {
        if self.threads == 0 {
            Self::recommended_threads()
        } else {
            self.threads
        }
    }

    pub fn validate(&self) -> Result<(), EngineError> {
        if self.threads > 256 || self.spin_wait_us > 1000 {
            return Err(EngineError::InvalidConfig(
                "CPU threads must be 0 (automatic) or 1..=256; spin wait must be <= 1000 us".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_pool_is_bounded_and_scheduling_is_validated() {
        let mut config = CpuExecutionConfig::default();
        assert!((1..=6).contains(&config.resolved_threads()));
        assert!(config.validate().is_ok());
        config.threads = 257;
        assert!(config.validate().is_err());
        config.threads = 2;
        config.spin_wait_us = 1001;
        assert!(config.validate().is_err());
        config.spin_wait_us = 0;
        assert!(config.validate().is_ok());
    }
}
