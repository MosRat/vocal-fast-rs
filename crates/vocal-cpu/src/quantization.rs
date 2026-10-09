//! Experimental static W8A8 recipes. Calibration observes the unmodified FP32
//! graph; replay uses packed S8 weights, scalar S8 inputs and FP32 outputs.
use crate::CpuError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use vocal_graph::graph::{Node, Op};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpuQuantizationRule {
    pub prefix: String,
    /// Optional SmoothQuant alpha in [0, 1]. First matching rule wins.
    pub smooth_alpha: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpuQuantization {
    pub rules: Vec<CpuQuantizationRule>,
    /// Canonical operator name -> FP32 calibration input channel absmax.
    pub calibration: BTreeMap<String, Vec<f32>>,
    /// Align selected ConvTranspose/BN PTQ with the reference's folded weights.
    #[serde(default = "fold_deconv_default")]
    pub fold_deconvolution_batchnorm: bool,
}
fn fold_deconv_default() -> bool {
    true
}
impl Default for CpuQuantization {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            calibration: BTreeMap::new(),
            fold_deconvolution_batchnorm: true,
        }
    }
}

impl CpuQuantization {
    pub fn validate(&self) -> Result<(), CpuError> {
        if self.rules.iter().any(|r| {
            r.prefix.is_empty()
                || r.smooth_alpha
                    .is_some_and(|a| !a.is_finite() || !(0. ..=1.).contains(&a))
        }) || self
            .calibration
            .values()
            .any(|v| v.is_empty() || v.iter().any(|x| !x.is_finite() || *x < 0.))
        {
            return Err(CpuError::Config(
                "invalid CPU quantization recipe/calibration".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn rule(&self, name: &str) -> Option<&CpuQuantizationRule> {
        self.rules.iter().find(|r| {
            name.strip_prefix(&r.prefix).is_some_and(|tail| {
                tail.is_empty() || r.prefix.ends_with('.') || tail.starts_with('.')
            })
        })
    }
}

pub(crate) fn operator_name(nodes: &[Node], n: &Node) -> Option<String> {
    match n.op {
        Op::Conv | Op::Deconv => Some(n.name.clone()),
        Op::Matmul | Op::LinearBias => {
            let weight = &nodes[n.inputs[1] as usize];
            if weight.tensor.shape.len() != 2 || weight.name.is_empty() {
                return None;
            }
            Some(
                weight
                    .name
                    .strip_suffix(".weight")
                    .unwrap_or(&weight.name)
                    .into(),
            )
        }
        _ => None,
    }
}

/// Match the reference's ConvTranspose->BatchNorm PTQ contract. Fold only
/// selected, exclusive FP32 parameters; the default graph is untouched.
pub(crate) fn fold_deconv_batchnorm(nodes: &mut [Node], outputs: &[u32], recipe: &CpuQuantization) {
    let mut users = vec![0; nodes.len()];
    for n in nodes.iter() {
        for id in &n.inputs {
            users[*id as usize] += 1;
        }
    }
    for id in outputs {
        users[*id as usize] += 1;
    }
    for id in 0..nodes.len() {
        let add = &nodes[id];
        if !matches!(add.op, Op::Add) {
            continue;
        }
        let mul_id = add.inputs[0] as usize;
        let shift_id = add.inputs[1] as usize;
        let mul = &nodes[mul_id];
        if !matches!(mul.op, Op::Mul) || users[mul_id] != 1 {
            continue;
        }
        let conv_id = mul.inputs[0] as usize;
        let scale_id = mul.inputs[1] as usize;
        let conv = &nodes[conv_id];
        if !matches!(conv.op, Op::Deconv)
            || conv.inputs.len() != 2
            || users[conv_id] != 1
            || recipe.rule(&conv.name).is_none()
        {
            continue;
        }
        let weight_id = conv.inputs[1] as usize;
        let weight = &nodes[weight_id];
        let scale = &nodes[scale_id];
        let shift = &nodes[shift_id];
        let channels = conv.tensor.shape[1];
        if users[weight_id] != 1
            || !matches!(weight.op, Op::Constant)
            || !matches!(scale.op, Op::Constant)
            || !matches!(shift.op, Op::Constant)
            || [weight, scale, shift]
                .iter()
                .any(|n| n.tensor.dtype != vocal_graph::graph::DType::F32)
            || scale.tensor.shape != [1, channels, 1, 1]
            || shift.tensor.shape != [1, channels, 1, 1]
            || weight.data.len() != weight.tensor.elements() * 4
            || scale.data.len() != channels * 4
        {
            continue;
        }
        let factors = scale
            .data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        let groups = conv.attrs[8] as usize;
        let co = channels / groups;
        let ci = weight.tensor.shape[0] / groups;
        let kernel = weight.tensor.shape[2] * weight.tensor.shape[3];
        let mut replacement = Node {
            op: Op::Deconv,
            tensor: add.tensor.clone(),
            inputs: conv.inputs.clone(),
            attrs: conv.attrs.clone(),
            values: conv.values,
            data: conv.data.clone(),
            name: conv.name.clone(),
        };
        replacement.inputs.push(shift_id as u32);
        let mut folded = weight.data.clone();
        for (i, b) in folded.chunks_exact_mut(4).enumerate() {
            let channel = (i / kernel) % co + ((i / (kernel * co)) / ci) * co;
            let value = f32::from_le_bytes(b[..].try_into().unwrap()) * factors[channel];
            b.copy_from_slice(&value.to_le_bytes());
        }
        nodes[weight_id].data = folded;
        nodes[id] = replacement;
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CpuQuantizedOperator {
    pub name: String,
    pub weight_elements: usize,
    pub implementation: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numeric_layer_prefix_does_not_select_another_layer() {
        let recipe = CpuQuantization {
            rules: vec![CpuQuantizationRule {
                prefix: "rmvpe.node.34".into(),
                smooth_alpha: None,
            }],
            ..CpuQuantization::default()
        };
        assert!(recipe.rule("rmvpe.node.34").is_some());
        assert!(recipe.rule("rmvpe.node.34.weight").is_some());
        assert!(recipe.rule("rmvpe.node.344").is_none());
    }
}
