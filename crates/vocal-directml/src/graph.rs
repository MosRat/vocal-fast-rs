use crate::ffi;
pub use vocal_graph::graph::*;
pub fn raw(g: &GraphBuilder) -> Vec<ffi::Node> {
    g.nodes
        .borrow()
        .iter()
        .map(|n| {
            let mut shape = [1; 8];
            for (out, value) in shape.iter_mut().zip(&n.tensor.shape) {
                *out = *value as u32;
            }
            ffi::Node {
                op: n.op as u32,
                dtype: n.tensor.dtype as u32,
                rank: n.tensor.shape.len() as u32,
                shape,
                inputs: n.inputs.as_ptr(),
                input_count: n.inputs.len(),
                attrs: n.attrs.as_ptr(),
                attr_count: n.attrs.len(),
                values: n.values,
                data: n.data.as_ptr(),
                data_bytes: n.data.len(),
            }
        })
        .collect()
}
