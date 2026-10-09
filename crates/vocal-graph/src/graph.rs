//! Fixed-shape, source-independent operator graph. Shape errors are rejected
//! before DirectML receives descriptors; constants have their execution dtype.
use crate::{GraphError, Precision};
use std::cell::RefCell;
use std::collections::HashMap;
use vocal_model::{ModelPackage, TensorDType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DType {
    F32 = 1,
    F16 = 2,
    I32 = 3,
    U32 = 4,
    U8 = 5,
}
impl From<Precision> for DType {
    fn from(p: Precision) -> Self {
        match p {
            Precision::Fp32 => Self::F32,
            Precision::Fp16 => Self::F16,
        }
    }
}
impl DType {
    pub fn bytes(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::U8 => 1,
            _ => 4,
        }
    }
}
#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum Op {
    Input = 0,
    Constant,
    Reshape,
    Transpose,
    Cast,
    Add,
    Sub,
    Mul,
    Div,
    Max = 10,
    Min,
    Relu,
    Leaky,
    Tanh,
    Sigmoid,
    Gelu,
    Exp,
    Log,
    Sqrt,
    Floor = 21,
    Sin,
    Matmul,
    Conv,
    Deconv,
    Softmax,
    Sum,
    Mean,
    ReduceMax,
    Gather,
    GatherElements,
    Slice,
    Concat,
    Cumsum = 35,
    Greater,
    Less,
    Select,
    Gru = 40,
    Pool,
    TopK,
    Round,
    Equals = 45,
    StftMel = 50,
    Norm = 51,
    AddLeaky = 52,
    LinearBias = 53,
    Attention = 54,
}
#[derive(Clone, Debug)]
pub struct Tensor {
    pub id: u32,
    pub shape: Vec<usize>,
    pub dtype: DType,
}
impl Tensor {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}
pub struct Node {
    pub op: Op,
    pub tensor: Tensor,
    pub inputs: Vec<u32>,
    pub attrs: Vec<i32>,
    pub values: [f32; 4],
    pub data: Vec<u8>,
    pub name: String,
}
#[derive(Default)]
pub struct GraphBuilder {
    pub nodes: RefCell<Vec<Node>>,
    pub outputs: RefCell<Vec<(String, Tensor)>>,
    pub feedback: RefCell<Vec<(Tensor, usize)>>,
    parameters: RefCell<HashMap<(String, u32), Tensor>>,
}
impl GraphBuilder {
    /// An internal graph output is copied back to this input after dispatch.
    /// It is neither uploaded nor read back during inference.
    pub fn feedback(&self, input: &Tensor, next: &Tensor) {
        assert_eq!(input.shape, next.shape);
        assert_eq!(input.dtype, next.dtype);
        let output = self.outputs.borrow().len();
        self.outputs
            .borrow_mut()
            .push((format!("__state_{output}"), next.clone()));
        self.feedback.borrow_mut().push((input.clone(), output));
    }
    pub fn node(
        &self,
        op: Op,
        inputs: &[&Tensor],
        shape: Vec<usize>,
        dtype: DType,
        attrs: Vec<i32>,
        values: [f32; 4],
        data: Vec<u8>,
        name: String,
    ) -> Tensor {
        assert!(!shape.is_empty() && shape.len() <= 8 && shape.iter().all(|n| *n > 0));
        assert!(
            shape
                .iter()
                .try_fold(1usize, |a, b| a.checked_mul(*b))
                .is_some()
        );
        let mut nodes = self.nodes.borrow_mut();
        let t = Tensor {
            id: nodes.len() as u32,
            shape,
            dtype,
        };
        nodes.push(Node {
            op,
            tensor: t.clone(),
            inputs: inputs.iter().map(|t| t.id).collect(),
            attrs,
            values,
            data,
            name,
        });
        t
    }
    pub fn input(&self, name: &str, shape: &[usize], dtype: DType) -> Tensor {
        self.node(
            Op::Input,
            &[],
            shape.to_vec(),
            dtype,
            vec![],
            [0.; 4],
            vec![],
            name.into(),
        )
    }
    pub fn constant(&self, data: &[f32], shape: &[usize], p: Precision) -> Tensor {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        let bytes = match p {
            Precision::Fp32 => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Precision::Fp16 => data
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
                .collect(),
        };
        self.node(
            Op::Constant,
            &[],
            shape.to_vec(),
            p.into(),
            vec![],
            [0.; 4],
            bytes,
            String::new(),
        )
    }
    pub fn integers(&self, data: &[i32], shape: &[usize]) -> Tensor {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        self.node(
            Op::Constant,
            &[],
            shape.to_vec(),
            DType::I32,
            vec![],
            [0.; 4],
            data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            String::new(),
        )
    }
    pub fn scalar(&self, value: f32, p: Precision) -> Tensor {
        self.constant(&[value], &[1], p)
    }
    pub fn parameter(
        &self,
        package: &ModelPackage,
        name: &str,
        p: Precision,
    ) -> Result<Tensor, GraphError> {
        let key = (name.to_string(), DType::from(p) as u32);
        if let Some(t) = self.parameters.borrow().get(&key) {
            return Ok(t.clone());
        }
        let weight = resolve_weight(package, name)?;
        let shape = weight.shape.iter().map(|n| *n as usize).collect::<Vec<_>>();
        let t = if (weight.dtype == TensorDType::F16 && p == Precision::Fp16)
            || (weight.dtype == TensorDType::F32 && p == Precision::Fp32)
        {
            self.node(
                Op::Constant,
                &[],
                shape,
                p.into(),
                vec![],
                [0.; 4],
                weight.data.clone(),
                name.into(),
            )
        } else {
            let tensor = self.constant(&weight_f32(package, name)?, &shape, p);
            // Keep canonical names across dtype conversion so backend recipes
            // and diagnostics can identify the same parameter in every cache.
            self.nodes.borrow_mut()[tensor.id as usize].name = name.into();
            tensor
        };
        self.parameters.borrow_mut().insert(key, t.clone());
        Ok(t)
    }
    pub fn reshape(&self, x: &Tensor, shape: &[usize]) -> Tensor {
        assert_eq!(x.elements(), shape.iter().product::<usize>());
        self.node(
            Op::Reshape,
            &[x],
            shape.to_vec(),
            x.dtype,
            vec![],
            [0.; 4],
            vec![],
            String::new(),
        )
    }
    pub fn transpose(&self, x: &Tensor, order: &[usize]) -> Tensor {
        assert_eq!(x.shape.len(), order.len());
        self.node(
            Op::Transpose,
            &[x],
            order.iter().map(|i| x.shape[*i]).collect(),
            x.dtype,
            order.iter().map(|i| *i as i32).collect(),
            [0.; 4],
            vec![],
            String::new(),
        )
    }
    pub fn cast(&self, x: &Tensor, dtype: DType) -> Tensor {
        if x.dtype == dtype {
            return x.clone();
        }
        self.node(
            Op::Cast,
            &[x],
            x.shape.clone(),
            dtype,
            vec![],
            [0.; 4],
            vec![],
            String::new(),
        )
    }
    pub fn unary(&self, op: Op, x: &Tensor) -> Tensor {
        self.node(
            op,
            &[x],
            x.shape.clone(),
            x.dtype,
            vec![],
            [0.1, 0., 0., 0.],
            vec![],
            String::new(),
        )
    }
    pub fn leaky(&self, x: &Tensor, alpha: f32) -> Tensor {
        self.node(
            Op::Leaky,
            &[x],
            x.shape.clone(),
            x.dtype,
            vec![],
            [alpha, 0., 0., 0.],
            vec![],
            String::new(),
        )
    }
    pub fn binary(&self, op: Op, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(a.dtype, b.dtype, "binary {op:?} dtype mismatch");
        let shape = broadcast_shape(&a.shape, &b.shape);
        let dtype = if matches!(op, Op::Greater | Op::Less | Op::Equals) {
            DType::U8
        } else {
            a.dtype
        };
        self.node(
            op,
            &[a, b],
            shape,
            dtype,
            vec![],
            [0.; 4],
            vec![],
            String::new(),
        )
    }
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        self.binary(Op::Add, a, b)
    }
    pub fn sub(&self, a: &Tensor, b: &Tensor) -> Tensor {
        self.binary(Op::Sub, a, b)
    }
    pub fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        self.binary(Op::Mul, a, b)
    }
    pub fn div(&self, a: &Tensor, b: &Tensor) -> Tensor {
        self.binary(Op::Div, a, b)
    }
    pub fn scale(&self, x: &Tensor, s: f32) -> Tensor {
        let p = if x.dtype == DType::F16 {
            Precision::Fp16
        } else {
            Precision::Fp32
        };
        self.mul(x, &self.scalar(s, p))
    }
    pub fn reduce(&self, op: Op, x: &Tensor, axes: &[usize]) -> Tensor {
        let mut shape = x.shape.clone();
        for a in axes {
            shape[*a] = 1;
        }
        self.node(
            op,
            &[x],
            shape,
            x.dtype,
            axes.iter().map(|i| *i as i32).collect(),
            [0.; 4],
            vec![],
            String::new(),
        )
    }
    pub fn softmax(&self, x: &Tensor, axis: usize) -> Tensor {
        self.node(
            Op::Softmax,
            &[x],
            x.shape.clone(),
            x.dtype,
            vec![axis as i32],
            [0.; 4],
            vec![],
            String::new(),
        )
    }
    pub fn matmul(&self, a: &Tensor, b: &Tensor, transpose_b: bool) -> Tensor {
        assert_eq!(a.dtype, b.dtype);
        let ar = a.shape.len();
        let br = b.shape.len();
        let k = b.shape[br - if transpose_b { 1 } else { 2 }];
        assert_eq!(a.shape[ar - 1], k);
        let mut shape = broadcast_shape(&a.shape[..ar - 2], &b.shape[..br - 2]);
        shape.extend([
            a.shape[ar - 2],
            b.shape[br - if transpose_b { 2 } else { 1 }],
        ]);
        self.node(
            Op::Matmul,
            &[a, b],
            shape,
            a.dtype,
            vec![i32::from(transpose_b)],
            [0.; 4],
            vec![],
            String::new(),
        )
    }
    pub fn slice(
        &self,
        x: &Tensor,
        axis: usize,
        start: usize,
        count: usize,
        stride: i32,
    ) -> Tensor {
        assert!(count > 0 && start < x.shape[axis]);
        let end = start as i64 + (count as i64 - 1) * stride as i64;
        assert!(end >= 0 && end < x.shape[axis] as i64);
        let mut shape = x.shape.clone();
        shape[axis] = count;
        self.node(
            Op::Slice,
            &[x],
            shape,
            x.dtype,
            vec![axis as i32, start as i32, stride],
            [0.; 4],
            vec![],
            String::new(),
        )
    }
    pub fn concat(&self, xs: &[Tensor], axis: usize) -> Tensor {
        let mut shape = xs[0].shape.clone();
        shape[axis] = 0;
        for x in xs {
            for a in 0..shape.len() {
                if a != axis {
                    assert_eq!(shape[a], x.shape[a]);
                }
            }
            assert_eq!(x.dtype, xs[0].dtype);
            shape[axis] += x.shape[axis];
        }
        self.node(
            Op::Concat,
            &xs.iter().collect::<Vec<_>>(),
            shape,
            xs[0].dtype,
            vec![axis as i32],
            [0.; 4],
            vec![],
            String::new(),
        )
    }
    pub fn gather(&self, x: &Tensor, indices: &Tensor, axis: usize) -> Tensor {
        let mut shape = x.shape[..axis].to_vec();
        shape.extend(&indices.shape);
        shape.extend(&x.shape[axis + 1..]);
        self.node(
            Op::Gather,
            &[x, indices],
            shape,
            x.dtype,
            vec![axis as i32, indices.shape.len() as i32],
            [0.; 4],
            vec![],
            String::new(),
        )
    }
    pub fn gather_elements(&self, x: &Tensor, indices: &Tensor, axis: usize) -> Tensor {
        assert_eq!(x.shape.len(), indices.shape.len());
        self.node(
            Op::GatherElements,
            &[x, indices],
            indices.shape.clone(),
            x.dtype,
            vec![axis as i32],
            [0.; 4],
            vec![],
            String::new(),
        )
    }
    pub fn select(&self, c: &Tensor, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(a.dtype, b.dtype);
        let shape = broadcast_shape(&c.shape, &broadcast_shape(&a.shape, &b.shape));
        self.node(
            Op::Select,
            &[c, a, b],
            shape,
            a.dtype,
            vec![],
            [0.; 4],
            vec![],
            String::new(),
        )
    }
    pub fn conv(
        &self,
        x: &Tensor,
        w: &Tensor,
        b: Option<&Tensor>,
        strides: [usize; 2],
        dilations: [usize; 2],
        padding: [usize; 2],
        groups: usize,
        deconv: bool,
        output_padding: [usize; 2],
    ) -> Tensor {
        assert_eq!(x.shape.len(), 4);
        assert_eq!(w.shape.len(), 4);
        assert_eq!(x.dtype, w.dtype);
        let mut shape = vec![
            x.shape[0],
            if deconv {
                w.shape[1] * groups
            } else {
                w.shape[0]
            },
        ];
        for i in 0..2 {
            let kernel = (w.shape[i + 2] - 1) * dilations[i] + 1;
            shape.push(if deconv {
                (x.shape[i + 2] - 1) * strides[i] + kernel + output_padding[i] - padding[i] * 2
            } else {
                (x.shape[i + 2] + padding[i] * 2 - kernel) / strides[i] + 1
            });
        }
        let mut inputs = vec![x, w];
        if let Some(b) = b {
            inputs.push(b);
        }
        let attrs = strides
            .into_iter()
            .chain(dilations)
            .chain(padding)
            .chain(output_padding)
            .chain([groups])
            .map(|n| n as i32)
            .collect();
        self.node(
            if deconv { Op::Deconv } else { Op::Conv },
            &inputs,
            shape,
            x.dtype,
            attrs,
            [0.; 4],
            vec![],
            String::new(),
        )
    }
    pub fn topk(&self, x: &Tensor, k: usize, largest: bool, indices: bool) -> Tensor {
        let mut shape = x.shape.clone();
        *shape.last_mut().unwrap() = k;
        self.node(
            Op::TopK,
            &[x],
            shape,
            if indices { DType::U32 } else { x.dtype },
            vec![k as i32, i32::from(largest), i32::from(indices)],
            [0.; 4],
            vec![],
            String::new(),
        )
    }
    pub fn mark(&self, name: &str, x: &Tensor) {
        self.outputs.borrow_mut().push((name.into(), x.clone()));
    }
}

pub fn weight_f32(package: &ModelPackage, name: &str) -> Result<Vec<f32>, GraphError> {
    let t = resolve_weight(package, name)?;
    match t.dtype {
        TensorDType::F32 => Ok(t
            .data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()),
        TensorDType::F16 => Ok(t
            .data
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect()),
        TensorDType::BF16 => Ok(t
            .data
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect()),
        _ => Err(GraphError::Graph(format!("non-floating weight {name}"))),
    }
}
fn resolve_weight<'a>(
    package: &'a ModelPackage,
    name: &str,
) -> Result<&'a vocal_model::Tensor, GraphError> {
    if let Some(tensor) = package.tensor(name) {
        return Ok(tensor);
    }
    if name == "rmvpe.mel_basis" {
        return package
            .tensor("rmvpe.mel_extractor.mel_basis")
            .ok_or_else(|| GraphError::Graph("missing RMVPE mel basis".into()));
    }
    if let Some((id, suffix)) = name
        .strip_prefix("rmvpe.node.")
        .and_then(|s| s.split_once('.'))
    {
        let id = id
            .parse::<u64>()
            .map_err(|_| GraphError::Graph(format!("invalid weight {name}")))?;
        let node = package
            .metadata
            .get("nodes")
            .and_then(serde_json::Value::as_array)
            .and_then(|nodes| nodes.iter().find(|n| n["id"].as_u64() == Some(id)))
            .ok_or_else(|| GraphError::Graph(format!("missing RMVPE mapping for {name}")))?;
        let slot = match suffix {
            "weight" | "W" => 1,
            "bias" if id == 361 => 0,
            "bias" | "R" => 2,
            "mean" | "B" => 3,
            "var" => 4,
            _ => {
                return Err(GraphError::Graph(format!("invalid RMVPE parameter {name}")));
            }
        };
        if let Some(tensor) = node["inputs"]
            .get(slot)
            .and_then(serde_json::Value::as_str)
            .and_then(|source| package.tensor(source))
        {
            return Ok(tensor);
        }
    }
    Err(GraphError::Graph(format!("missing weight {name}")))
}
fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let rank = a.len().max(b.len());
    let mut s = vec![1; rank];
    for dims in [a, b] {
        for (i, n) in dims.iter().rev().enumerate() {
            let dst = &mut s[rank - 1 - i];
            assert!(
                *dst == 1 || *n == 1 || *n == *dst,
                "invalid broadcast {a:?} {b:?}"
            );
            *dst = (*dst).max(*n);
        }
    }
    s
}
