use serde::{Deserialize, Serialize};

/// Dense oneDNN or cache tiled online softmax. Both compute full bidirectional
/// attention in FP32, without a cross-window KV cache.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuAttentionMode {
    #[default]
    Dense,
    Online,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CpuAttentionConfig {
    /// Applied to attention patterns lowered with `CpuConfig::fuse` enabled.
    pub mode: CpuAttentionMode,
    /// Work grouping and live output rows; independent of sequence length.
    pub query_tile: usize,
    pub key_tile: usize,
    /// Store [B,H,T,D] in BTHD order so the following transpose/reshape is a view.
    pub token_major_output: bool,
}

impl Default for CpuAttentionConfig {
    fn default() -> Self {
        Self {
            mode: CpuAttentionMode::Dense,
            query_tile: 4,
            key_tile: 32,
            token_major_output: true,
        }
    }
}

impl CpuAttentionConfig {
    pub(crate) fn validate(&self) -> Result<(), crate::CpuError> {
        if !(1..=32).contains(&self.query_tile) || !(1..=256).contains(&self.key_tile) {
            return Err(crate::CpuError::Config(
                "attention query tile must be 1..=32 and key tile 1..=256".into(),
            ));
        }
        Ok(())
    }
}
