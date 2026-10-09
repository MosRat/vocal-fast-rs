use serde::{Deserialize, Serialize};

/// FP32 convolution implementation policy. Unsupported shapes retain oneDNN
/// direct convolution; this does not alter the graph's logical tensor layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuConvolutionMode {
    #[default]
    Direct,
    /// Experimental unfold + packed MatMul for ungrouped, NHWC 1D convolution.
    /// Expansion is bounded to 96 MiB of shared graph workspace.
    Matmul1d,
    /// Experimental unfold + oneDNN pointwise convolution using BRGEMM.
    Pointwise1d,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CpuConvolutionConfig {
    pub mode: CpuConvolutionMode,
    /// Exact erf GELU, only when the pre-activation has one unmarked consumer.
    pub fuse_gelu: bool,
    /// Fuse an exclusively consumed convolution's residual Add/Leaky when the
    /// other input is already available. Marked intermediates remain intact.
    pub fuse_residual: bool,
}
