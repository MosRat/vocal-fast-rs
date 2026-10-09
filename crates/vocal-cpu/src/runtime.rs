use crate::{
    CpuConfig, CpuError, CpuQuantization, CpuQuantizedOperator, ffi, planner, quantization,
};
use std::{
    collections::BTreeMap,
    ffi::{CStr, c_void},
    ptr::NonNull,
};
use vocal_graph::graph::{GraphBuilder, Tensor};

pub struct CpuGraph {
    native: NonNull<c_void>,
    inputs: BTreeMap<String, Tensor>,
    outputs: BTreeMap<String, Tensor>,
    nodes: Vec<(String, String)>,
    profile: bool,
    quantized: Vec<(u32, String, usize, usize)>,
    calibration_only: bool,
}
// The graph is exclusively owned and all execution methods require &mut self.
unsafe impl Send for CpuGraph {}
impl Drop for CpuGraph {
    fn drop(&mut self) {
        unsafe {
            ffi::vocal_cpu_destroy(self.native.as_ptr());
        }
    }
}
#[derive(Default)]
pub struct CpuOutput {
    pub tensors: BTreeMap<String, Vec<u8>>,
    dtypes: BTreeMap<String, vocal_graph::graph::DType>,
}
impl CpuOutput {
    pub fn f32(&self, name: &str) -> Result<Vec<f32>, CpuError> {
        if self.dtypes.get(name) != Some(&vocal_graph::graph::DType::F32) {
            return Err(CpuError::Runtime(format!("output {name} is not FP32")));
        }
        let data = self
            .tensors
            .get(name)
            .ok_or_else(|| CpuError::Runtime(format!("missing output {name}")))?;
        Ok(data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct CpuNodeTiming {
    /// Empty for standalone graphs; set when graph timings join a pipeline.
    pub stage: String,
    pub id: usize,
    pub operator: String,
    pub name: String,
    pub implementation: String,
    pub ms: f64,
}
impl CpuGraph {
    /// Owned tensor/workspace/parameter bytes. Excludes primitive JIT metadata,
    /// allocator overhead, FFT plans, thread stacks, and model file mappings.
    pub fn memory_usage(&self) -> ffi::CpuMemory {
        let mut memory = ffi::CpuMemory::default();
        // SAFETY: Graph is alive and the ABI writes a complete owned structure.
        unsafe { ffi::vocal_cpu_memory(self.native.as_ptr(), &mut memory) };
        memory
    }
    pub fn compile(builder: GraphBuilder, config: &CpuConfig) -> Result<Self, CpuError> {
        Self::compile_impl(builder, config, None, false)
    }
    pub fn compile_calibration(
        builder: GraphBuilder,
        config: &CpuConfig,
        recipe: &CpuQuantization,
    ) -> Result<Self, CpuError> {
        Self::compile_impl(builder, config, Some(recipe), true)
    }
    pub fn compile_quantized(
        builder: GraphBuilder,
        config: &CpuConfig,
        recipe: &CpuQuantization,
    ) -> Result<Self, CpuError> {
        Self::compile_impl(builder, config, Some(recipe), false)
    }
    fn compile_impl(
        builder: GraphBuilder,
        config: &CpuConfig,
        recipe: Option<&CpuQuantization>,
        calibration_only: bool,
    ) -> Result<Self, CpuError> {
        config.validate()?;
        if let Some(recipe) = recipe {
            recipe.validate()?;
        }
        if !builder.feedback.borrow().is_empty() {
            return Err(CpuError::Config(
                "CPU state feedback must be managed by the pipeline".into(),
            ));
        }
        let marks = builder.outputs.into_inner();
        if marks.is_empty() {
            return Err(CpuError::Config("graph has no outputs".into()));
        }
        let original_ids = marks.iter().map(|(_, t)| t.id).collect::<Vec<_>>();
        let mut nodes = builder.nodes.into_inner();
        crate::validation::validate(&nodes, &original_ids)?;
        if let Some(recipe) = recipe.filter(|r| r.fold_deconvolution_batchnorm) {
            quantization::fold_deconv_batchnorm(&mut nodes, &original_ids, recipe);
        }
        let (nodes, ids) = planner::lower(
            nodes,
            &original_ids,
            config.fuse,
            config.fuse_channel_norm,
            config.convolution.fuse_residual,
        );
        let mut quantized = Vec::new();
        let mut quantization = Vec::new();
        if let Some(recipe) = recipe {
            for (id, n) in nodes.iter().enumerate() {
                let Some(name) = quantization::operator_name(&nodes, n) else {
                    continue;
                };
                let Some(rule) = recipe.rule(&name) else {
                    continue;
                };
                let x = &nodes[n.inputs[0] as usize].tensor;
                let channels = if matches!(
                    n.op,
                    vocal_graph::graph::Op::Conv | vocal_graph::graph::Op::Deconv
                ) {
                    x.shape[1]
                } else {
                    *x.shape.last().unwrap()
                };
                let values = if calibration_only {
                    &[][..]
                } else {
                    recipe
                        .calibration
                        .get(&name)
                        .filter(|v| v.len() == channels)
                        .ok_or_else(|| {
                            CpuError::Config(format!(
                                "missing or incompatible calibration for {name}"
                            ))
                        })?
                        .as_slice()
                };
                quantization.push(ffi::Quantization {
                    id: id as u32,
                    mode: if calibration_only { 1 } else { 2 },
                    smooth_alpha: rule.smooth_alpha.unwrap_or(-1.),
                    input_max: values.as_ptr(),
                    channels,
                });
                quantized.push((
                    id as u32,
                    name,
                    channels,
                    nodes[n.inputs[1] as usize].tensor.elements(),
                ));
            }
        }
        let raw = nodes
            .iter()
            .map(|n| {
                let mut shape = [1; 8];
                for (dst, src) in shape.iter_mut().zip(&n.tensor.shape) {
                    *dst = *src as u32;
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
            .collect::<Vec<_>>();
        #[cfg(target_arch = "x86_64")]
        let avx2 = config.avx2
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;
        let attention = ffi::Attention {
            mode: match config.attention.mode {
                crate::CpuAttentionMode::Dense => 0,
                crate::CpuAttentionMode::Online => 1,
            },
            query_tile: config.attention.query_tile as u32,
            key_tile: config.attention.key_tile as u32,
            token_major_output: config.attention.token_major_output,
        };
        let native = NonNull::new(unsafe {
            ffi::vocal_cpu_create(
                raw.as_ptr(),
                raw.len(),
                ids.as_ptr(),
                ids.len(),
                config.threads,
                config.spin_wait_us,
                config.pin_to_performance_cores,
                avx2,
                config.profile,
                config.channels_last,
                config.fuse,
                config.fuse_linear_gelu,
                config.fast_tensor_io,
                match config.linear_mode {
                    crate::CpuLinearMode::Matmul => 0,
                    crate::CpuLinearMode::InnerProduct => 1,
                    crate::CpuLinearMode::Pointwise1d => 2,
                    crate::CpuLinearMode::Hybrid => 3,
                },
                &ffi::Convolution {
                    mode: match config.convolution.mode {
                        crate::CpuConvolutionMode::Direct => 0,
                        crate::CpuConvolutionMode::Matmul1d => 1,
                        crate::CpuConvolutionMode::Pointwise1d => 2,
                    },
                    fuse_gelu: config.convolution.fuse_gelu,
                    fuse_residual: config.convolution.fuse_residual,
                },
                &attention,
                quantization.as_ptr(),
                quantization.len(),
            )
        })
        .ok_or_else(ffi::error)?;
        let inputs = nodes
            .iter()
            .filter(|n| matches!(n.op, vocal_graph::graph::Op::Input))
            .map(|n| (n.name.clone(), n.tensor.clone()))
            .collect();
        let outputs = marks
            .into_iter()
            .zip(ids)
            .map(|((name, mut t), id)| {
                t.id = id;
                (name, t)
            })
            .collect();
        Ok(Self {
            native,
            inputs,
            outputs,
            nodes: nodes
                .iter()
                .map(|n| {
                    (
                        format!("{:?}", n.op),
                        quantization::operator_name(&nodes, n).unwrap_or_else(|| n.name.clone()),
                    )
                })
                .collect(),
            profile: config.profile,
            quantized,
            calibration_only,
        })
    }
    /// Read per-channel absmax collected before FP32 operators execute. This
    /// diagnostic path is excluded from quantized replay and latency benchmarks.
    pub fn calibration(&self) -> Result<BTreeMap<String, Vec<f32>>, CpuError> {
        if !self.calibration_only {
            return Err(CpuError::Config(
                "graph is not collecting calibration".into(),
            ));
        }
        self.quantized
            .iter()
            .map(|(id, name, channels, _)| {
                let mut values = vec![0.; *channels];
                if unsafe {
                    ffi::vocal_cpu_calibration(
                        self.native.as_ptr(),
                        *id,
                        values.as_mut_ptr(),
                        *channels,
                    )
                } != 0
                {
                    return Err(ffi::error());
                }
                Ok((name.clone(), values))
            })
            .collect()
    }
    pub fn quantized_operators(&self) -> Vec<CpuQuantizedOperator> {
        if self.calibration_only {
            return Vec::new();
        }
        self.quantized
            .iter()
            .map(|(id, name, _, weight_elements)| CpuQuantizedOperator {
                name: name.clone(),
                weight_elements: *weight_elements,
                // SAFETY: Owned native kernel string lives until graph destruction.
                implementation: unsafe {
                    CStr::from_ptr(ffi::vocal_cpu_implementation(self.native.as_ptr(), *id))
                }
                .to_string_lossy()
                .into_owned(),
            })
            .collect()
    }
    pub fn input_names(&self) -> impl Iterator<Item = &str> {
        self.inputs.keys().map(String::as_str)
    }
    fn execute(&mut self, bindings: &[(&str, &[u8])]) -> Result<(), CpuError> {
        let mut seen = std::collections::HashSet::new();
        let raw = bindings
            .iter()
            .map(|(name, data)| {
                let t = self
                    .inputs
                    .get(*name)
                    .ok_or_else(|| CpuError::Config(format!("unknown input {name}")))?;
                if data.len() != t.elements() * t.dtype.bytes() || !seen.insert(*name) {
                    return Err(CpuError::Config(format!("invalid input binding {name}")));
                }
                Ok(ffi::Binding {
                    id: t.id,
                    data: data.as_ptr(),
                    bytes: data.len(),
                })
            })
            .collect::<Result<Vec<_>, CpuError>>()?;
        if seen.len() != self.inputs.len() {
            return Err(CpuError::Config("missing CPU input binding".into()));
        }
        if unsafe { ffi::vocal_cpu_run(self.native.as_ptr(), raw.as_ptr(), raw.len()) } != 0 {
            return Err(ffi::error());
        }
        Ok(())
    }
    /// Copy a marked FP32 output directly into caller-owned reusable storage.
    /// No temporary byte tensor or byte-to-float conversion is allocated.
    pub fn run_f32_into(
        &mut self,
        bindings: &[(&str, &[u8])],
        name: &str,
        destination: &mut Vec<f32>,
    ) -> Result<(), CpuError> {
        let tensor = self
            .outputs
            .get(name)
            .ok_or_else(|| CpuError::Config(format!("unknown CPU output {name}")))?;
        if tensor.dtype != vocal_graph::graph::DType::F32 {
            return Err(CpuError::Config(format!("CPU output {name} is not FP32")));
        }
        let (id, elements) = (tensor.id, tensor.elements());
        self.execute(bindings)?;
        destination.resize(elements, 0.);
        // SAFETY: The native copy writes exactly `elements * sizeof(f32)`
        // bytes to this initialized, exclusively borrowed FP32 allocation.
        if unsafe {
            ffi::vocal_cpu_output(
                self.native.as_ptr(),
                id,
                destination.as_mut_ptr().cast(),
                std::mem::size_of_val(destination.as_slice()),
            )
        } != 0
        {
            return Err(ffi::error());
        }
        Ok(())
    }
    pub fn run(&mut self, bindings: &[(&str, &[u8])]) -> Result<CpuOutput, CpuError> {
        self.execute(bindings)?;
        let mut result = CpuOutput::default();
        for (name, t) in &self.outputs {
            let mut data = vec![0; t.elements() * t.dtype.bytes()];
            if unsafe {
                ffi::vocal_cpu_output(self.native.as_ptr(), t.id, data.as_mut_ptr(), data.len())
            } != 0
            {
                return Err(ffi::error());
            }
            result.tensors.insert(name.clone(), data);
            result.dtypes.insert(name.clone(), t.dtype);
        }
        Ok(result)
    }
    pub fn timings(&self) -> Vec<CpuNodeTiming> {
        if !self.profile {
            return vec![];
        }
        self.nodes
            .iter()
            .enumerate()
            .map(|(id, (operator, name))| CpuNodeTiming {
                stage: String::new(),
                id,
                operator: operator.clone(),
                name: name.clone(),
                // Diagnostic only; the returned string belongs to this graph.
                implementation: unsafe {
                    CStr::from_ptr(ffi::vocal_cpu_implementation(
                        self.native.as_ptr(),
                        id as u32,
                    ))
                }
                .to_string_lossy()
                .into_owned(),
                ms: unsafe { ffi::vocal_cpu_node_ms(self.native.as_ptr(), id as u32) },
            })
            .filter(|n| n.ms > 0.)
            .collect()
    }
}
pub(crate) fn bytes(values: &[f32]) -> &[u8] {
    // Byte slice is borrowed only during a synchronous native run.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}
