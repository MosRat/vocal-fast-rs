use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalSearch {
    #[default]
    Auto,
    Dense,
    Sparse,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    #[default]
    Fp32,
    Fp16,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PrecisionConfig {
    pub hubert: Precision,
    pub hubert_features: Precision,
    pub rmvpe_encoder: Precision,
    pub rmvpe_decoder: Precision,
    pub synth_decoder: Precision,
    pub synth_fp32_stage_mask: u32,
    pub synth_noise_fp16_stage_mask: u32,
    pub synth_input_scale: f32,
    pub synth_post: Precision,
}
impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            hubert: Precision::Fp16,
            hubert_features: Precision::Fp32,
            rmvpe_encoder: Precision::Fp32,
            rmvpe_decoder: Precision::Fp32,
            synth_decoder: Precision::Fp16,
            synth_fp32_stage_mask: 8,
            synth_noise_fp16_stage_mask: 7,
            synth_input_scale: 1.,
            synth_post: Precision::Fp32,
        }
    }
}
impl PrecisionConfig {
    pub fn fp32() -> Self {
        Self {
            hubert: Precision::Fp32,
            synth_decoder: Precision::Fp32,
            synth_fp32_stage_mask: 0,
            synth_noise_fp16_stage_mask: 0,
            ..Self::default()
        }
    }
}
