//! Validate public IR before lowering or handing descriptors to native code.
use crate::CpuError;
use vocal_graph::graph::{Node, Op};
pub fn validate(nodes: &[Node], outputs: &[u32]) -> Result<(), CpuError> {
    let mut names = std::collections::HashSet::new();
    for (id, n) in nodes.iter().enumerate() {
        let bad = || {
            CpuError::Config(format!(
                "invalid CPU graph descriptor at node {id} ({:?})",
                n.op
            ))
        };
        if n.tensor.id as usize != id
            || n.inputs.iter().any(|edge| *edge as usize >= id)
            || n.tensor.shape.is_empty()
            || n.tensor.shape.len() > 8
            || n.tensor
                .shape
                .iter()
                .any(|d| *d == 0 || *d > u32::MAX as usize)
            || n.tensor
                .shape
                .iter()
                .try_fold(1usize, |a, d| a.checked_mul(*d))
                .and_then(|s| s.checked_mul(n.tensor.dtype.bytes()))
                .is_none()
        {
            return Err(bad());
        }
        let (input_count, attrs) = match n.op {
            Op::Input | Op::Constant => (0, 0),
            Op::Transpose => (1, n.tensor.shape.len()),
            Op::Matmul => (2, 1),
            Op::Conv | Op::Deconv => (n.inputs.len(), 9),
            Op::Softmax => (1, 1),
            Op::Sum | Op::Mean | Op::ReduceMax => (1, n.attrs.len()),
            Op::Gather => (2, 2),
            Op::GatherElements => (2, 1),
            Op::Slice => (1, 3),
            Op::Concat => (n.inputs.len(), 1),
            Op::Cumsum => (1, n.attrs.len()),
            Op::Gru => (4, 0),
            Op::TopK => (1, 3),
            Op::Select => (3, 0),
            Op::Norm => (3, 1),
            Op::LinearBias => (3, 1),
            Op::Attention => (3, 0),
            Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::Max
            | Op::Min
            | Op::Greater
            | Op::Less
            | Op::Equals
            | Op::StftMel
            | Op::AddLeaky => (2, 0),
            _ => (1, 0),
        };
        if n.inputs.len() != input_count || n.attrs.len() != attrs {
            return Err(bad());
        }
        match n.op {
            Op::Input if n.name.is_empty() || !names.insert(n.name.clone()) => return Err(bad()),
            Op::Constant if n.data.len() != n.tensor.elements() * n.tensor.dtype.bytes() => {
                return Err(bad());
            }
            Op::Conv | Op::Deconv
                if !(2..=3).contains(&n.inputs.len())
                    || n.attrs[..4].iter().any(|v| *v <= 0)
                    || n.attrs[4..8].iter().any(|v| *v < 0)
                    || n.attrs[8] <= 0
                    || n.tensor.shape.len() != 4 =>
            {
                return Err(bad());
            }
            Op::Gru if n.tensor.shape.len() != 4 => return Err(bad()),
            Op::Pool if n.tensor.shape.len() != 4 => return Err(bad()),
            Op::Cumsum if !(1..=2).contains(&n.attrs.len()) => return Err(bad()),
            Op::Concat if n.inputs.is_empty() => return Err(bad()),
            Op::Slice
                if n.attrs[1] < 0
                    || n.attrs[2] == 0
                    || n.tensor.shape.len() != nodes[n.inputs[0] as usize].tensor.shape.len() =>
            {
                return Err(bad());
            }
            Op::Sum | Op::Mean | Op::ReduceMax if n.attrs.is_empty() => return Err(bad()),
            Op::Matmul | Op::LinearBias if ![0, 1].contains(&n.attrs[0]) => return Err(bad()),
            Op::TopK if n.attrs[0] <= 0 => return Err(bad()),
            _ => {}
        }
        match n.op {
            Op::Transpose | Op::Sum | Op::Mean | Op::ReduceMax => {
                let rank = nodes[n.inputs[0] as usize].tensor.shape.len();
                if n.attrs.iter().any(|v| *v < 0 || *v as usize >= rank) {
                    return Err(bad());
                }
            }
            Op::Softmax
            | Op::Gather
            | Op::GatherElements
            | Op::Slice
            | Op::Concat
            | Op::Cumsum
            | Op::Norm => {
                let rank = nodes[n.inputs[0] as usize].tensor.shape.len();
                if n.attrs[0] < 0 || n.attrs[0] as usize >= rank {
                    return Err(bad());
                }
            }
            _ => {}
        }
    }
    if outputs.iter().any(|id| *id as usize >= nodes.len()) {
        return Err(CpuError::Config("invalid marked CPU output".into()));
    }
    Ok(())
}
