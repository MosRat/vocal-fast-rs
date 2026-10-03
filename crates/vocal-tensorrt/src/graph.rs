//! VFRM to TensorRT graph contract. This module owns graph-level validation;
//! the FFI layer only receives primitive dimensions and never parses files.
use crate::NativeBuildError;
use vocal_model::ModelPackage;
use vocal_model::TensorDType;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageKind {
    Native,
    Plugin,
}

#[derive(Clone, Debug)]
pub struct GraphStage {
    pub name: String,
    pub kind: StageKind,
    pub op: &'static str,
    pub weight_prefix: Option<String>,
}

#[derive(Clone, Debug)]
pub struct NativeGraphPlan {
    pub spec: VfrmGraphSpec,
    pub stages: Vec<GraphStage>,
    pub native_count: usize,
    pub plugin_count: usize,
}

impl NativeGraphPlan {
    pub fn from_package(package: &ModelPackage) -> Result<Self, NativeBuildError> {
        let spec = VfrmGraphSpec::from_package(package)?;
        let config = package
            .rvc_synth_config()
            .map_err(|e| NativeBuildError::Api(e.to_string()))?
            .ok_or_else(|| {
                NativeBuildError::Argument("canonical package has no rvc_config".into())
            })?;
        let mut stages = vec![
            GraphStage {
                name: "enc_p.emb_phone".into(),
                kind: StageKind::Native,
                op: "linear",
                weight_prefix: Some("enc_p.emb_phone".into()),
            },
            GraphStage {
                name: "enc_p.encoder.attention".into(),
                kind: StageKind::Plugin,
                op: "multi_head_attention",
                weight_prefix: Some("enc_p.encoder.attn_layers".into()),
            },
            GraphStage {
                name: "enc_p.encoder.ffn".into(),
                kind: StageKind::Native,
                op: "feed_forward",
                weight_prefix: Some("enc_p.encoder.ffn_layers".into()),
            },
        ];
        for i in 0..config.flow_blocks {
            stages.push(GraphStage {
                name: format!("flow.flows.{}.reverse", i * 2),
                kind: StageKind::Plugin,
                op: "residual_coupling_reverse",
                weight_prefix: Some(format!("flow.flows.{}", i * 2)),
            });
        }
        for i in 0..config.upsample_rates.len() {
            stages.push(GraphStage {
                name: format!("dec.ups.{i}"),
                kind: StageKind::Native,
                op: "conv_transpose1d",
                weight_prefix: Some(format!("dec.ups.{i}")),
            });
            for j in 0..config.resblock_kernels.len() {
                stages.push(GraphStage {
                    name: format!("dec.resblocks.{}", i * config.resblock_kernels.len() + j),
                    kind: StageKind::Plugin,
                    op: "residual_block",
                    weight_prefix: Some(format!(
                        "dec.resblocks.{}",
                        i * config.resblock_kernels.len() + j
                    )),
                });
            }
        }
        if config.uses_f0 {
            stages.push(GraphStage {
                name: "dec.m_source".into(),
                kind: StageKind::Plugin,
                op: "nsf_source",
                weight_prefix: Some("dec.m_source".into()),
            });
        }
        stages.push(GraphStage {
            name: "dec.conv_post".into(),
            kind: StageKind::Native,
            op: "conv1d",
            weight_prefix: Some("dec.conv_post".into()),
        });
        let native_count = stages
            .iter()
            .filter(|s| s.kind == StageKind::Native)
            .count();
        let plugin_count = stages.len() - native_count;
        Ok(Self {
            spec,
            stages,
            native_count,
            plugin_count,
        })
    }
    pub fn validate_weights(&self, package: &ModelPackage) -> Result<(), NativeBuildError> {
        let shapes = package
            .rvc_synth_config()
            .map_err(|e| NativeBuildError::Api(e.to_string()))?
            .ok_or_else(|| NativeBuildError::Argument("missing rvc config".into()))?
            .tensor_shapes()
            .map_err(|e| NativeBuildError::Api(e.to_string()))?;
        for (name, shape) in shapes {
            let t = package
                .tensor(&name)
                .ok_or_else(|| NativeBuildError::Argument(format!("missing VFRM weight {name}")))?;
            if t.shape != shape
                || !matches!(
                    t.dtype,
                    TensorDType::F16 | TensorDType::F32 | TensorDType::BF16
                )
            {
                return Err(NativeBuildError::Argument(format!(
                    "invalid VFRM weight {name}"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use vocal_model::{
        ModelPackage, RVC_TENSOR_SCHEMA, RvcModelManifest, RvcSynthConfig, Tensor, TensorDType,
    };

    fn package() -> ModelPackage {
        let manifest = RvcModelManifest {
            id: "test".into(),
            display_name: "test".into(),
            architecture: "rvc-v2-nono".into(),
            version: "v2".into(),
            sample_rate: 16_000,
            model_file: "x".into(),
            index_file: None,
            source: "test".into(),
            notes: None,
        };
        let c = RvcSynthConfig {
            feature_width: 768,
            uses_f0: false,
            sample_rate: 16_000,
            hidden_channels: 32,
            latent_channels: 16,
            filter_channels: 64,
            attention_heads: 4,
            encoder_layers: 1,
            encoder_kernel: 3,
            relative_window: 2,
            relative_heads: 4,
            norm_epsilon: 1e-5,
            speakers: 2,
            speaker_channels: 8,
            decoder_channels: 32,
            resblock_type: 1,
            resblock_kernels: vec![3],
            resblock_dilations: vec![vec![1, 3, 5]],
            upsample_rates: vec![2],
            upsample_kernels: vec![4],
            flow_blocks: 1,
            flow_layers: 3,
            flow_kernel: 5,
            flow_dilations: vec![1; 3],
        };
        let tensors = c
            .tensor_shapes()
            .unwrap()
            .into_iter()
            .map(|(name, shape)| Tensor {
                name,
                dtype: TensorDType::F32,
                data: vec![0; shape.iter().product::<u64>() as usize * 4],
                shape,
            })
            .collect();
        ModelPackage {
            manifest,
            metadata: json!({"canonical_schema":RVC_TENSOR_SCHEMA,"rvc_config":c}),
            tensors,
        }
    }
    #[test]
    fn plan_marks_composites_as_plugins() {
        let p = NativeGraphPlan::from_package(&package()).unwrap();
        assert!(p.plugin_count > 0);
        assert!(p.native_count > 0);
        assert!(
            p.stages
                .iter()
                .any(|s| s.op == "residual_coupling_reverse" && s.kind == StageKind::Plugin)
        );
    }
    #[test]
    fn plan_validates_all_weights() {
        let p = package();
        let plan = NativeGraphPlan::from_package(&p).unwrap();
        plan.validate_weights(&p).unwrap();
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VfrmGraphSpec {
    pub feature_width: i32,
    pub latent_channels: i32,
    pub output_scale: i32,
}

impl VfrmGraphSpec {
    pub fn from_package(package: &ModelPackage) -> Result<Self, NativeBuildError> {
        let c = package
            .validate_rvc_synth()
            .map_err(|e| NativeBuildError::Api(e.to_string()))?;
        let scale = c
            .upsample_rates
            .iter()
            .try_fold(1u64, |a, &v| a.checked_mul(v))
            .ok_or_else(|| NativeBuildError::Argument("upsample product overflow".into()))?;
        if scale > i32::MAX as u64 {
            return Err(NativeBuildError::Argument(
                "upsample product exceeds TensorRT dimension range".into(),
            ));
        }
        Ok(Self {
            feature_width: c.feature_width as i32,
            latent_channels: c.latent_channels as i32,
            output_scale: scale as i32,
        })
    }
}
