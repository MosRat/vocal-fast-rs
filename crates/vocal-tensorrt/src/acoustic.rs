//! Source-independent acoustic graph contract.
//!
//! The converter stores the graph as a compact operator IR in the acoustic
//! VFRM metadata.  This module is the Rust side of the native builder seam:
//! it validates the IR before any C++ call and classifies operators into
//! TensorRT primitives, fused candidates, and explicit plugin kernels.

use crate::error::NativeBuildError;
use serde_json::Value;
use std::collections::BTreeMap;
use vocal_model::{AcousticConfig, AcousticKind, ModelPackage};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AcousticStageKind {
    Primitive,
    Fused,
    Kernel,
}

#[derive(Clone, Debug)]
pub struct AcousticStage {
    pub id: usize,
    pub op: String,
    pub kind: AcousticStageKind,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct AcousticGraphPlan {
    pub config: AcousticConfig,
    pub stages: Vec<AcousticStage>,
    pub op_counts: BTreeMap<String, usize>,
}

impl AcousticGraphPlan {
    pub fn from_package(
        package: &ModelPackage,
        expected: AcousticKind,
    ) -> Result<Self, NativeBuildError> {
        let config = package
            .validate_acoustic(Some(expected))
            .map_err(|error| NativeBuildError::Api(error.to_string()))?;
        let nodes = package
            .metadata
            .get("nodes")
            .and_then(Value::as_array)
            .ok_or_else(|| NativeBuildError::Argument("acoustic VFRM has no operator IR".into()))?;
        let mut stages = Vec::with_capacity(nodes.len());
        let mut op_counts = BTreeMap::new();
        for node in nodes {
            let id = node.get("id").and_then(Value::as_u64).ok_or_else(|| {
                NativeBuildError::Argument("acoustic node has no numeric id".into())
            })? as usize;
            let op = node
                .get("op")
                .and_then(Value::as_str)
                .ok_or_else(|| NativeBuildError::Argument("acoustic node has no op".into()))?;
            let name = node
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(op)
                .to_owned();
            let kind = classify(expected, op);
            *op_counts.entry(op.to_owned()).or_default() += 1;
            stages.push(AcousticStage {
                id,
                op: op.to_owned(),
                kind,
                name,
            });
        }
        if stages.len() != config.node_count {
            return Err(NativeBuildError::Argument(format!(
                "acoustic node count mismatch: metadata={} IR={}",
                config.node_count,
                stages.len()
            )));
        }
        Ok(Self {
            config,
            stages,
            op_counts,
        })
    }

    pub fn kernel_count(&self) -> usize {
        self.stages
            .iter()
            .filter(|stage| stage.kind == AcousticStageKind::Kernel)
            .count()
    }

    pub fn fused_count(&self) -> usize {
        self.stages
            .iter()
            .filter(|stage| stage.kind == AcousticStageKind::Fused)
            .count()
    }
}

fn classify(kind: AcousticKind, op: &str) -> AcousticStageKind {
    match (kind, op) {
        (_, "LayerNormalization" | "BatchNormalization") => AcousticStageKind::Fused,
        (AcousticKind::Hubert, "Softmax") => AcousticStageKind::Fused,
        (AcousticKind::Rmvpe, "STFT" | "GRU") => AcousticStageKind::Kernel,
        (
            _,
            "Conv" | "ConvTranspose" | "MatMul" | "Add" | "Mul" | "Div" | "Relu" | "Sigmoid"
            | "Tanh" | "Erf" | "Sqrt" | "Log" | "Pow" | "ReduceSum" | "ReduceMax" | "Reshape"
            | "Transpose" | "Slice" | "Concat" | "Gather" | "AveragePool",
        ) => AcousticStageKind::Primitive,
        _ => AcousticStageKind::Primitive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vocal_model::{AcousticKind, load_cache};

    #[test]
    fn shipped_hubert_ir_is_classified() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/shared/hubert_base/hubert_features.vfr");
        let package = load_cache(path);
        if let Ok(package) = package {
            let plan = AcousticGraphPlan::from_package(&package, AcousticKind::Hubert).unwrap();
            assert_eq!(plan.stages.len(), 583);
            assert!(plan.fused_count() >= 12);
        }
    }

    #[test]
    fn shipped_rmvpe_ir_has_explicit_frontend_and_gru_kernels() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/shared/rmvpe/rmvpe.vfr");
        let package = load_cache(path);
        if let Ok(package) = package {
            let plan = AcousticGraphPlan::from_package(&package, AcousticKind::Rmvpe).unwrap();
            assert_eq!(plan.stages.len(), 390);
            assert_eq!(plan.kernel_count(), 2);
            assert_eq!(plan.op_counts.get("Conv"), Some(&124));
            assert_eq!(plan.op_counts.get("GRU"), Some(&1));
        }
    }
}
