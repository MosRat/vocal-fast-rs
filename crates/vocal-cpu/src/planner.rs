//! Lower local patterns, then prune unreachable tensors before native allocation.
use vocal_graph::graph::{DType, Node, Op};

fn norm(nodes: &[Node], id: usize, channel_norm: bool) -> Option<(Vec<u32>, Vec<i32>, f32)> {
    let n = &nodes[id];
    if !matches!(n.op, Op::Add) || n.tensor.dtype != DType::F32 {
        return None;
    }
    let mul = &nodes[n.inputs[0] as usize];
    if !matches!(mul.op, Op::Mul) {
        return None;
    }
    let div = &nodes[mul.inputs[0] as usize];
    if !matches!(div.op, Op::Div) {
        return None;
    }
    let center = &nodes[div.inputs[0] as usize];
    let sqrt = &nodes[div.inputs[1] as usize];
    if !matches!(center.op, Op::Sub) || !matches!(sqrt.op, Op::Sqrt) {
        return None;
    }
    let mean = &nodes[center.inputs[1] as usize];
    let add = &nodes[sqrt.inputs[0] as usize];
    if !matches!(mean.op, Op::Mean) || mean.attrs.len() != 1 || !matches!(add.op, Op::Add) {
        return None;
    }
    let var = &nodes[add.inputs[0] as usize];
    let eps = &nodes[add.inputs[1] as usize];
    if !matches!(var.op, Op::Mean)
        || var.attrs != mean.attrs
        || !matches!(eps.op, Op::Constant)
        || eps.data.len() != 4
    {
        return None;
    }
    let square = &nodes[var.inputs[0] as usize];
    if !matches!(square.op, Op::Mul)
        || square.inputs != [div.inputs[0], div.inputs[0]]
        || mean.inputs != [center.inputs[0]]
    {
        return None;
    }
    let axis = mean.attrs[0] as usize;
    let width = center.tensor.shape[axis];
    let gamma = &nodes[mul.inputs[1] as usize];
    let beta = &nodes[n.inputs[1] as usize];
    if [
        nodes[center.inputs[0] as usize].tensor.dtype,
        gamma.tensor.dtype,
        beta.tensor.dtype,
    ]
    .iter()
    .any(|dtype| *dtype != DType::F32)
    {
        return None;
    }
    let mut affine = vec![1; center.tensor.shape.len()];
    affine[axis] = width;
    let attrs = if gamma.tensor.shape == affine && beta.tensor.shape == affine {
        vec![axis as i32]
    } else if channel_norm
        && center.tensor.shape.len() == 3
        && center.tensor.shape[0] == 1
        && axis == 2
        && gamma.tensor.shape == [1, center.tensor.shape[1], 1]
        && beta.tensor.shape == gamma.tensor.shape
    {
        // HuBERT extractor GroupNorm: one independent group per channel,
        // statistics over time, affine parameters over channels.
        vec![2, 1]
    } else {
        return None;
    };
    Some((
        vec![center.inputs[0], mul.inputs[1], n.inputs[1]],
        attrs,
        f32::from_le_bytes(eps.data.as_slice().try_into().ok()?),
    ))
}
pub fn lower(
    mut nodes: Vec<Node>,
    output_ids: &[u32],
    fuse: bool,
    channel_norm: bool,
    conv_residual: bool,
) -> (Vec<Node>, Vec<u32>) {
    let mut users = vec![0; nodes.len()];
    if conv_residual {
        for node in &nodes {
            for input in &node.inputs {
                users[*input as usize] += 1;
            }
        }
        for output in output_ids {
            users[*output as usize] += 1;
        }
    }
    if fuse {
        for id in 0..nodes.len() {
            if let Some((inputs, scale)) = attention(&nodes, id) {
                nodes[id].op = Op::Attention;
                nodes[id].inputs = inputs;
                nodes[id].attrs.clear();
                nodes[id].values[0] = scale;
            } else if let Some((inputs, attrs, epsilon)) = norm(&nodes, id, channel_norm) {
                let n = &mut nodes[id];
                n.op = Op::Norm;
                n.inputs = inputs;
                n.attrs = attrs;
                n.values[0] = epsilon;
                if n.attrs.len() == 2 {
                    n.name = "channel_temporal_norm".into();
                }
            } else if matches!(nodes[id].op, Op::Leaky) {
                let add = &nodes[nodes[id].inputs[0] as usize];
                // With convolution residual post-ops, duplicating a shared Add
                // into AddLeaky prevents the producer from fusing the sum.
                // Retain one Add and its standalone activation in that case.
                if matches!(add.op, Op::Add)
                    && add.tensor.dtype == DType::F32
                    && (!conv_residual || users[nodes[id].inputs[0] as usize] == 1)
                {
                    let inputs = add.inputs.clone();
                    nodes[id].op = Op::AddLeaky;
                    nodes[id].inputs = inputs;
                }
            } else if matches!(nodes[id].op, Op::Add) {
                if let Some((matmul, bias_id)) = linear_bias_inputs(&nodes, id) {
                    let mut inputs = matmul.inputs.clone();
                    inputs.push(bias_id);
                    let attrs = matmul.attrs.clone();
                    nodes[id].op = Op::LinearBias;
                    nodes[id].inputs = inputs;
                    nodes[id].attrs = attrs;
                }
            }
        }
    }
    let mut active = vec![false; nodes.len()];
    let mut stack = output_ids.to_vec();
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut active[id as usize], true) {
            continue;
        }
        stack.extend(&nodes[id as usize].inputs);
    }
    let mut remap = vec![0; nodes.len()];
    let mut out = Vec::new();
    for (id, mut node) in nodes.into_iter().enumerate() {
        if !active[id] {
            continue;
        }
        remap[id] = out.len() as u32;
        node.tensor.id = out.len() as u32;
        for edge in &mut node.inputs {
            *edge = remap[*edge as usize];
        }
        out.push(node);
    }
    let outputs = output_ids.iter().map(|id| remap[*id as usize]).collect();
    (out, outputs)
}

fn attention(nodes: &[Node], id: usize) -> Option<(Vec<u32>, f32)> {
    let context = &nodes[id];
    if !matches!(context.op, Op::Matmul) || context.attrs != [0] || context.tensor.shape.len() != 4
    {
        return None;
    }
    let softmax = &nodes[context.inputs[0] as usize];
    if !matches!(softmax.op, Op::Softmax) || softmax.attrs != [3] {
        return None;
    }
    let scaled = &nodes[softmax.inputs[0] as usize];
    if !matches!(scaled.op, Op::Mul) {
        return None;
    }
    let (dot_id, scale_id) = if matches!(nodes[scaled.inputs[0] as usize].op, Op::Matmul)
        && matches!(nodes[scaled.inputs[1] as usize].op, Op::Constant)
    {
        (scaled.inputs[0], scaled.inputs[1])
    } else if matches!(nodes[scaled.inputs[1] as usize].op, Op::Matmul)
        && matches!(nodes[scaled.inputs[0] as usize].op, Op::Constant)
    {
        (scaled.inputs[1], scaled.inputs[0])
    } else {
        return None;
    };
    let dot = &nodes[dot_id as usize];
    let scale = &nodes[scale_id as usize];
    if !matches!(dot.op, Op::Matmul)
        || dot.attrs != [1]
        || !matches!(scale.op, Op::Constant)
        || scale.data.len() != 4
        || scale.tensor.dtype != DType::F32
    {
        return None;
    }
    let q = &nodes[dot.inputs[0] as usize];
    let k = &nodes[dot.inputs[1] as usize];
    let v = &nodes[context.inputs[1] as usize];
    if q.tensor.dtype != dot.tensor.dtype
        || k.tensor.dtype != dot.tensor.dtype
        || v.tensor.dtype != context.tensor.dtype
        || q.tensor.shape.len() < 2
        || q.tensor.shape.len() != k.tensor.shape.len()
        || q.tensor.shape.len() != v.tensor.shape.len()
        || dot.tensor.shape != softmax.tensor.shape
        || context.tensor.shape.len() != q.tensor.shape.len()
        || q.tensor.shape[..q.tensor.shape.len() - 2] != k.tensor.shape[..k.tensor.shape.len() - 2]
        || q.tensor.shape[..q.tensor.shape.len() - 2] != v.tensor.shape[..v.tensor.shape.len() - 2]
        || q.tensor.shape[q.tensor.shape.len() - 1] != k.tensor.shape[k.tensor.shape.len() - 1]
        || q.tensor.shape[q.tensor.shape.len() - 2] != v.tensor.shape[v.tensor.shape.len() - 2]
        || q.tensor.shape[q.tensor.shape.len() - 1]
            != context.tensor.shape[context.tensor.shape.len() - 1]
    {
        return None;
    }
    let value = f32::from_le_bytes(scale.data.as_slice().try_into().ok()?);
    value
        .is_finite()
        .then_some((vec![dot.inputs[0], dot.inputs[1], context.inputs[1]], value))
}

fn linear_bias_inputs(nodes: &[Node], id: usize) -> Option<(&Node, u32)> {
    let add = &nodes[id];
    if add.inputs.len() != 2 || add.tensor.shape.is_empty() {
        return None;
    }
    for (candidate_id, bias_id) in [
        (add.inputs[0], add.inputs[1]),
        (add.inputs[1], add.inputs[0]),
    ] {
        let matmul = &nodes[candidate_id as usize];
        let bias = &nodes[bias_id as usize];
        if matches!(matmul.op, Op::Matmul)
            && matmul.inputs.len() == 2
            && nodes[matmul.inputs[1] as usize].tensor.shape.len() == 2
            && matmul.tensor.shape == add.tensor.shape
            && matmul.tensor.dtype == add.tensor.dtype
            && bias.tensor.elements() == *add.tensor.shape.last().unwrap()
            && bias.tensor.dtype == add.tensor.dtype
            && bias.tensor.shape.len() <= add.tensor.shape.len()
            && bias
                .tensor
                .shape
                .iter()
                .rev()
                .zip(add.tensor.shape.iter().rev())
                .all(|(bias_dim, output_dim)| *bias_dim == 1 || bias_dim == output_dim)
        {
            return Some((matmul, bias_id));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use vocal_graph::graph::GraphBuilder;

    #[test]
    fn linear_bias_fusion_is_commutative_and_preserves_bias_slot() {
        let g = GraphBuilder::default();
        let x = g.input("x", &[1, 2, 3], DType::F32);
        let w = g.input("w", &[3, 4], DType::F32);
        let bias = g.constant(&[0.; 4], &[4], vocal_graph::Precision::Fp32);
        let mm = g.matmul(&x, &w, false);
        let add = g.add(&bias, &mm);
        let (nodes, outputs) = lower(g.nodes.into_inner(), &[add.id], true, true, false);
        let fused = &nodes[outputs[0] as usize];
        assert!(matches!(fused.op, Op::LinearBias));
        assert_eq!(fused.inputs, vec![x.id, w.id, bias.id]);
    }

    #[test]
    fn attention_fusion_accepts_either_scale_operand() {
        let g = GraphBuilder::default();
        let q = g.input("q", &[1, 2, 3, 4], DType::F32);
        let k = g.input("k", &[1, 2, 3, 4], DType::F32);
        let v = g.input("v", &[1, 2, 3, 4], DType::F32);
        let dot = g.matmul(&q, &k, true);
        let scale = g.scalar(0.5, vocal_graph::Precision::Fp32);
        let scores = g.mul(&scale, &dot);
        let probs = g.softmax(&scores, 3);
        let out = g.matmul(&probs, &v, false);
        let (nodes, outputs) = lower(g.nodes.into_inner(), &[out.id], true, true, false);
        assert!(matches!(nodes[outputs[0] as usize].op, Op::Attention));
    }
}
