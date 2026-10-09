//! Measured, opt-in static W8A8 selections. Presets contain operator rules, not
//! calibration data: callers must calibrate the same models and graph settings.
use crate::{CpuError, CpuQuantization, CpuQuantizationRule, CpuQuantizedOperator};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Experimental CPU precision presets. Ordinary pipeline construction remains
/// FP32. Names and SmoothQuant alphas follow the held-out native experiments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuQuantizationPreset {
    /// 33 RMVPE intermediate convolutions; keep encoder, decoder and head FP32.
    RmvpeIntermediate,
    /// 49 HuBERT attention/feature-projection linears; keep both FFN linears,
    /// attention arithmetic, RMVPE and Synth FP32.
    HubertAttention,
    /// 61 HuBERT projection/FFN-input linears; keep RMVPE and Synth FP32.
    HubertMixed,
    /// Add 61 HuBERT projection/FFN-input linears. Attention arithmetic, FFN
    /// outputs, norms, STFT, GRU and the complete Synth remain FP32.
    HubertMixedRmvpeIntermediate,
}

impl CpuQuantizationPreset {
    /// Uncalibrated rules for `CpuPipeline::from_sources_calibration`.
    pub fn recipe(self) -> CpuQuantization {
        self.recipe_for_rvc(vocal_model::RvcVersion::V2)
    }

    /// v1 has nine HuBERT blocks; its learned output head remains FP32.
    pub fn recipe_for_rvc(self, version: vocal_model::RvcVersion) -> CpuQuantization {
        // Canonical VFRM names of the fixed RMVPE intermediate residual stack.
        // Coverage validation rejects graph/schema drift rather than silently
        // falling back to a partially quantized preset.
        const RMVPE_INTERMEDIATE: [u16; 33] = [
            138, 140, 142, 144, 146, 149, 151, 154, 156, 159, 161, 164, 166, 169, 171, 174, 176,
            179, 181, 184, 186, 189, 191, 194, 196, 199, 201, 204, 206, 209, 211, 214, 216,
        ];
        let mut rules = RMVPE_INTERMEDIATE
            .iter()
            .filter(|_| {
                matches!(
                    self,
                    Self::RmvpeIntermediate | Self::HubertMixedRmvpeIntermediate
                )
            })
            .map(|id| CpuQuantizationRule {
                prefix: format!("rmvpe.node.{id}"),
                smooth_alpha: None,
            })
            .collect::<Vec<_>>();
        if self != Self::RmvpeIntermediate {
            rules.push(CpuQuantizationRule {
                prefix: "hubert.model.feature_projection.projection".into(),
                smooth_alpha: Some(0.5),
            });
            for block in 0..version.hubert_output_layer() {
                let prefix = format!("hubert.model.encoder.layers.{block}");
                for (suffix, smooth_alpha) in [
                    ("attention.q_proj", None),
                    ("attention.k_proj", None),
                    ("attention.v_proj", Some(0.5)),
                    ("attention.out_proj", Some(0.5)),
                    ("feed_forward.intermediate_dense", Some(0.7)),
                ] {
                    if self == Self::HubertAttention && suffix.starts_with("feed_forward.") {
                        continue;
                    }
                    rules.push(CpuQuantizationRule {
                        prefix: format!("{prefix}.{suffix}"),
                        smooth_alpha,
                    });
                }
            }
        }
        CpuQuantization {
            rules,
            ..CpuQuantization::default()
        }
    }

    /// Require observations for every selected operator before loading graphs.
    /// Superset calibration is accepted to share an explicitly bound calibration
    /// corpus between presets. Channel sizes are checked against the real graph
    /// during compilation. Model/config provenance remains the caller's duty.
    pub fn calibrated_recipe(
        self,
        calibration: BTreeMap<String, Vec<f32>>,
    ) -> Result<CpuQuantization, CpuError> {
        self.calibrated_recipe_for_rvc(calibration, vocal_model::RvcVersion::V2)
    }

    pub fn calibrated_recipe_for_rvc(
        self,
        calibration: BTreeMap<String, Vec<f32>>,
        version: vocal_model::RvcVersion,
    ) -> Result<CpuQuantization, CpuError> {
        let mut recipe = self.recipe_for_rvc(version);
        recipe.calibration = calibration;
        recipe.validate()?;
        let missing = recipe
            .rules
            .iter()
            .filter(|rule| !recipe.calibration.contains_key(&rule.prefix))
            .map(|rule| rule.prefix.as_str())
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(CpuError::Config(format!(
                "missing CPU preset calibration: {}",
                missing.join(", ")
            )));
        }
        Ok(recipe)
    }

    /// Check actual native coverage, including extra and duplicate operators.
    pub fn validate_coverage(self, operators: &[CpuQuantizedOperator]) -> Result<(), CpuError> {
        self.validate_coverage_for_rvc(operators, vocal_model::RvcVersion::V2)
    }

    pub fn validate_coverage_for_rvc(
        self,
        operators: &[CpuQuantizedOperator],
        version: vocal_model::RvcVersion,
    ) -> Result<(), CpuError> {
        let recipe = self.recipe_for_rvc(version);
        let expected = recipe
            .rules
            .iter()
            .map(|rule| rule.prefix.as_str())
            .collect::<BTreeSet<_>>();
        let actual = operators
            .iter()
            .map(|operator| operator.name.as_str())
            .collect::<BTreeSet<_>>();
        if actual != expected || actual.len() != operators.len() {
            let missing = expected.difference(&actual).copied().collect::<Vec<_>>();
            let unexpected = actual.difference(&expected).copied().collect::<Vec<_>>();
            return Err(CpuError::Config(format!(
                "CPU preset coverage mismatch: missing [{}], unexpected [{}], duplicates {}",
                missing.join(", "),
                unexpected.join(", "),
                operators.len() - actual.len()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_quantization_coverage_stops_after_nine_blocks() {
        let preset = CpuQuantizationPreset::HubertMixed;
        let recipe = preset.recipe_for_rvc(vocal_model::RvcVersion::V1);
        assert_eq!(recipe.rules.len(), 46);
        assert!(
            !recipe
                .rules
                .iter()
                .any(|r| r.prefix.contains("layers.9.") || r.prefix.contains("final_proj"))
        );
        let operators = recipe
            .rules
            .into_iter()
            .map(|r| CpuQuantizedOperator {
                name: r.prefix,
                weight_elements: 1,
                implementation: "test".into(),
            })
            .collect::<Vec<_>>();
        preset
            .validate_coverage_for_rvc(&operators, vocal_model::RvcVersion::V1)
            .unwrap();
        assert!(preset.validate_coverage(&operators).is_err());
    }

    #[test]
    fn drift_and_duplicate_coverage_are_rejected() {
        let preset = CpuQuantizationPreset::RmvpeIntermediate;
        let mut operators = preset
            .recipe()
            .rules
            .into_iter()
            .map(|rule| CpuQuantizedOperator {
                name: rule.prefix,
                weight_elements: 1,
                implementation: "test".into(),
            })
            .collect::<Vec<_>>();
        preset.validate_coverage(&operators).unwrap();
        // Same count does not imply the expected layer coverage.
        operators[0].name = "rmvpe.node.1380".into();
        let error = preset
            .validate_coverage(&operators)
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing [rmvpe.node.138]"));
        assert!(error.contains("unexpected [rmvpe.node.1380]"));
        operators[0].name = operators[1].name.clone();
        assert!(preset.validate_coverage(&operators).is_err());
    }

    #[test]
    fn incomplete_or_nonfinite_calibration_is_rejected() {
        let preset = CpuQuantizationPreset::HubertMixedRmvpeIntermediate;
        assert!(preset.calibrated_recipe(BTreeMap::new()).is_err());
        let mut calibration = preset
            .recipe()
            .rules
            .into_iter()
            .map(|rule| (rule.prefix, vec![1.]))
            .collect::<BTreeMap<_, _>>();
        calibration.insert("unused.observed.layer".into(), vec![2.]);
        preset.calibrated_recipe(calibration.clone()).unwrap();
        calibration.insert("rmvpe.node.138".into(), vec![f32::NAN]);
        assert!(preset.calibrated_recipe(calibration).is_err());
    }

    #[test]
    fn hubert_only_preset_is_an_exact_subset_without_rmvpe_or_synth() {
        let only = CpuQuantizationPreset::HubertMixed.recipe();
        let combined = CpuQuantizationPreset::HubertMixedRmvpeIntermediate.recipe();
        assert_eq!(only.rules.len(), 61);
        assert_eq!(combined.rules.len(), 94);
        let attention = CpuQuantizationPreset::HubertAttention.recipe();
        assert_eq!(attention.rules.len(), 49);
        assert!(
            attention
                .rules
                .iter()
                .all(|r| !r.prefix.contains("feed_forward.")
                    && r.prefix.starts_with("hubert.model."))
        );
        for rule in only.rules {
            assert!(rule.prefix.starts_with("hubert.model."));
            assert!(
                combined
                    .rules
                    .iter()
                    .any(|r| r.prefix == rule.prefix && r.smooth_alpha == rule.smooth_alpha)
            );
        }
    }
}
