use serde::{Deserialize, Serialize};

/// Constant-weight FP32 projections. Batched/dynamic matrix multiplication
/// keeps its ordinary MatMul implementation under either policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuLinearMode {
    #[default]
    Matmul,
    /// Experimental InnerProduct; prefer a supported packed BRGEMM kernel.
    InnerProduct,
    /// Experimental constant-weight projection lowered to a packed 1x1 Conv.
    Pointwise1d,
    /// Experimental shape dispatch: retain MatMul for long, wide reductions;
    /// use packed pointwise projections for the remaining eligible layouts.
    Hybrid,
}
