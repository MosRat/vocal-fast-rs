use crate::{
    DirectMlError, ExecutionConfig, IoBuffers, ffi,
    graph::{DType, GraphBuilder, Op},
};
use std::{
    ffi::{CStr, CString, c_char, c_void},
    ptr,
    sync::{Arc, Mutex},
    time::Instant,
};

struct DeviceHandle {
    raw: *mut c_void,
    lock: Mutex<()>,
    name: String,
}
unsafe impl Send for DeviceHandle {}
unsafe impl Sync for DeviceHandle {}
impl Drop for DeviceHandle {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            ffi::vdml_device_free(self.raw)
        }
    }
}
#[derive(Clone)]
pub struct DirectMlDevice(Arc<DeviceHandle>, ExecutionConfig);
impl DirectMlDevice {
    /// Stable adapter identity, weak device reuse and graph execution policy.
    /// Compiled weights and mutable history always belong to individual graphs.
    pub fn from_config(config: &crate::DirectMlConfig) -> Result<Self, DirectMlError> {
        config.execution.validate()?;
        let device = if config.execution.reuse_device {
            device_pool::acquire(config)?
        } else {
            Self::with_adapter(
                config.adapter_index,
                config.adapter_id.as_deref(),
                config.debug_layer,
            )?
        };
        Ok(device.with_execution(config.execution.clone()))
    }
    pub fn new(adapter: u32) -> Result<Self, DirectMlError> {
        Self::with_debug_layer(adapter, false)
    }
    pub fn with_debug_layer(adapter: u32, debug: bool) -> Result<Self, DirectMlError> {
        Self::with_adapter(adapter, None, debug)
    }
    pub fn with_adapter(
        adapter: u32,
        id: Option<&str>,
        debug: bool,
    ) -> Result<Self, DirectMlError> {
        #[cfg(windows)]
        {
            let (mut raw, mut error) = (ptr::null_mut(), ptr::null_mut());
            let id = id.map(CString::new).transpose().map_err(|_| {
                DirectMlError::Config("DirectML device ID contains a null character".into())
            })?;
            check(
                unsafe {
                    ffi::vdml_device_create(
                        adapter,
                        id.as_ref().map_or(ptr::null(), |id| id.as_ptr()),
                        u32::from(debug),
                        &mut raw,
                        &mut error,
                    )
                },
                error,
            )?;
            Ok(Self(
                Arc::new(DeviceHandle {
                    raw,
                    lock: Mutex::new(()),
                    name: unsafe { CStr::from_ptr(ffi::vdml_device_name(raw)) }
                        .to_string_lossy()
                        .into_owned(),
                }),
                ExecutionConfig::default(),
            ))
        }
        #[cfg(not(windows))]
        {
            let _ = (adapter, id, debug);
            Err(DirectMlError::Runtime("DirectML requires Windows".into()))
        }
    }
    pub fn adapter_name(&self) -> &str {
        &self.0.name
    }
    /// Clone the device/queue while selecting independent graph compile options.
    pub fn with_execution(mut self, execution: ExecutionConfig) -> Self {
        self.1 = execution;
        self
    }
    pub fn execution_config(&self) -> &ExecutionConfig {
        &self.1
    }
    pub(crate) fn compile(&self, g: &GraphBuilder) -> Result<DirectMlGraph, DirectMlError> {
        self.1.validate()?;
        #[cfg(windows)]
        {
            let _guard = self
                .0
                .lock
                .lock()
                .map_err(|_| DirectMlError::Runtime("DirectML device lock poisoned".into()))?;
            let source = g.nodes.borrow();
            let output = g.outputs.borrow();
            let mut active = vec![false; source.len()];
            let mut pending = output
                .iter()
                .map(|(_, t)| t.id as usize)
                .collect::<Vec<_>>();
            while let Some(id) = pending.pop() {
                if !active[id] {
                    active[id] = true;
                    pending.extend(source[id].inputs.iter().map(|id| *id as usize));
                }
            }
            let mut remap = vec![0u32; source.len()];
            let mut count = 0;
            for (id, used) in active.iter().enumerate() {
                if *used {
                    remap[id] = count;
                    count += 1;
                }
            }
            let edges = source
                .iter()
                .enumerate()
                .filter(|(i, _)| active[*i])
                .map(|(_, n)| {
                    n.inputs
                        .iter()
                        .map(|i| remap[*i as usize])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut nodes = crate::graph::raw(g)
                .into_iter()
                .enumerate()
                .filter(|(i, _)| active[*i])
                .map(|(_, n)| n)
                .collect::<Vec<_>>();
            for (n, edges) in nodes.iter_mut().zip(&edges) {
                n.inputs = edges.as_ptr();
                n.input_count = edges.len();
            }
            let ids = output
                .iter()
                .map(|(_, t)| remap[t.id as usize])
                .collect::<Vec<_>>();
            let states = g.feedback.borrow();
            let mut slot_for_node = vec![u32::MAX; source.len()];
            let mut slot = 0;
            for (id, n) in source.iter().enumerate() {
                if active[id] && matches!(n.op, Op::Input | Op::Constant) {
                    slot_for_node[id] = slot;
                    slot += 1;
                }
            }
            let feedback = states
                .iter()
                .map(|(input, output)| ffi::Feedback {
                    input: slot_for_node[input.id as usize],
                    output: *output as u32,
                })
                .collect::<Vec<_>>();
            let (mut raw, mut error) = (ptr::null_mut(), ptr::null_mut());
            let options = ffi::CompileOptions {
                pooled_io: u32::from(self.1.io_buffers == IoBuffers::Pooled),
                contiguous_views: u32::from(self.1.contiguous_views),
                fuse_gemm_bias: u32::from(self.1.fuse_gemm_bias),
                pooled_weights: u32::from(self.1.pooled_weights),
                weight_staging_mib: self.1.weight_staging_mib,
            };
            check(
                unsafe {
                    ffi::vdml_compile(
                        self.0.raw,
                        nodes.as_ptr(),
                        nodes.len(),
                        ids.as_ptr(),
                        ids.len(),
                        feedback.as_ptr(),
                        feedback.len(),
                        &options,
                        &mut raw,
                        &mut error,
                    )
                },
                error,
            )?;
            let graph = DirectMlGraph {
                raw,
                device: self.clone(),
                binding_workspace: Vec::new(),
                inputs: source
                    .iter()
                    .enumerate()
                    .filter(|(i, n)| {
                        active[*i]
                            && matches!(n.op, Op::Input)
                            && !states.iter().any(|(t, _)| t.id as usize == *i)
                    })
                    .map(|(_, n)| (n.name.clone(), n.tensor.elements() * n.tensor.dtype.bytes()))
                    .collect(),
                outputs: output
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !states.iter().any(|(_, output)| output == i))
                    .map(|(_, (name, t))| (name.clone(), t.elements() * t.dtype.bytes(), t.dtype))
                    .collect(),
            };
            tracing::info!(resources = ?graph.memory(), "DirectML graph resource budget");
            Ok(graph)
        }
        #[cfg(not(windows))]
        {
            let _ = g;
            Err(DirectMlError::Runtime("DirectML requires Windows".into()))
        }
    }
}
#[path = "device_pool.rs"]
mod device_pool;
pub struct DirectMlGraph {
    raw: *mut c_void,
    device: DirectMlDevice,
    inputs: Vec<(String, usize)>,
    outputs: Vec<(String, usize, DType)>,
    binding_workspace: Vec<ffi::Binding>,
}
/// Resource budgets, in bytes. `*_allocation_bytes` reports D3D12 allocation
/// sizes including heap alignment; other byte fields are requested payloads.
/// Driver overhead and process/adapter usage are not included. Constants,
/// staging and initializer scratch are transient and released after loading.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct GraphMemory {
    pub constant_bytes: u64,
    pub constant_buffer_bytes: u64,
    pub staging_bytes: u64,
    pub initialization_temporary_bytes: u64,
    pub execution_temporary_bytes: u64,
    pub persistent_bytes: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub host_io_bytes: u64,
    pub io_resource_count: u64,
    pub io_buffer_bytes: u64,
    pub upload_copies: u64,
    pub readback_copies: u64,
    pub source_nodes: u64,
    pub lowered_nodes: u64,
    pub gemm_bias_fusions: u64,
    pub constant_resource_count: u64,
    pub constant_allocation_bytes: u64,
    pub io_allocation_bytes: u64,
}
unsafe impl Send for DirectMlGraph {}
/// Native GPU timestamps, without diagnostic tensor downloads. Dispatch also
/// includes the GPU feedback copies. Queue scheduling/waits are host time.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct GpuTimings {
    pub upload_ms: f64,
    pub dispatch_ms: f64,
    pub readback_ms: f64,
}
impl GpuTimings {
    pub fn total_ms(self) -> f64 {
        self.upload_ms + self.dispatch_ms + self.readback_ms
    }
}
impl std::ops::AddAssign for GpuTimings {
    fn add_assign(&mut self, rhs: Self) {
        self.upload_ms += rhs.upload_ms;
        self.dispatch_ms += rhs.dispatch_ms;
        self.readback_ms += rhs.readback_ms;
    }
}
#[derive(Debug)]
pub struct GraphOutput {
    pub tensors: Vec<(String, Vec<u8>)>,
    dtypes: Vec<DType>,
    pub gpu_ms: f64,
    pub total_ms: f64,
    pub gpu_timings: GpuTimings,
}
impl GraphOutput {
    pub(crate) fn push_f32(&mut self, name: &str, values: &[f32]) {
        self.tensors.push((
            name.into(),
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
        self.dtypes.push(DType::F32);
    }
    pub(crate) fn timings(gpu_ms: f64, total_ms: f64) -> Self {
        Self {
            tensors: Vec::new(),
            dtypes: Vec::new(),
            gpu_ms,
            total_ms,
            gpu_timings: GpuTimings {
                dispatch_ms: gpu_ms,
                ..Default::default()
            },
        }
    }
    pub fn f32(&self, name: &str) -> Result<Vec<f32>, DirectMlError> {
        let position = self
            .tensors
            .iter()
            .position(|(n, _)| n == name)
            .ok_or_else(|| DirectMlError::Graph(format!("missing output {name}")))?;
        let b = &self.tensors[position].1;
        if self.dtypes.get(position) != Some(&DType::F32) || b.len() % 4 != 0 {
            return Err(DirectMlError::Graph("output is not F32".into()));
        }
        Ok(b.chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect())
    }
}
impl DirectMlGraph {
    pub fn memory(&self) -> GraphMemory {
        let mut memory = GraphMemory::default();
        #[cfg(windows)]
        unsafe {
            ffi::vdml_graph_memory(self.raw, &mut memory);
        }
        memory
    }
    /// Clear GPU history without recompiling the graph or changing its weights.
    pub fn reset_states(&mut self) -> Result<(), DirectMlError> {
        #[cfg(windows)]
        {
            let _guard = self
                .device
                .0
                .lock
                .lock()
                .map_err(|_| DirectMlError::Runtime("DirectML device lock poisoned".into()))?;
            let mut error = ptr::null_mut();
            check(
                unsafe { ffi::vdml_reset_states(self.raw, &mut error) },
                error,
            )
        }
        #[cfg(not(windows))]
        Err(DirectMlError::Runtime("DirectML requires Windows".into()))
    }
    pub fn input_names(&self) -> impl Iterator<Item = &str> {
        self.inputs.iter().map(|(n, _)| n.as_str())
    }
    pub fn run(&mut self, inputs: &[(&str, &[u8])]) -> Result<GraphOutput, DirectMlError> {
        let mut tensors = self
            .outputs
            .iter()
            .map(|(name, size, _)| (name.clone(), vec![0u8; *size]))
            .collect::<Vec<_>>();
        let out = tensors
            .iter_mut()
            .map(|(_, b)| ffi::Binding {
                data: b.as_mut_ptr(),
                bytes: b.len(),
            })
            .collect::<Vec<_>>();
        let (gpu_timings, total_ms) = self.execute(inputs, &out)?;
        Ok(GraphOutput {
            tensors,
            dtypes: self.outputs.iter().map(|(_, _, d)| *d).collect(),
            gpu_ms: gpu_timings.dispatch_ms,
            total_ms,
            gpu_timings,
        })
    }
    /// Bind the final F32 output directly to its Rust destination, avoiding an
    /// intermediate byte vector and a second audio copy/decoding pass.
    pub(crate) fn run_f32(
        &mut self,
        name: &str,
        inputs: &[(&str, &[u8])],
    ) -> Result<(Vec<f32>, GraphOutput), DirectMlError> {
        if self.outputs.len() != 1 || self.outputs[0].0 != name || self.outputs[0].2 != DType::F32 {
            return Err(DirectMlError::Graph(
                "direct audio output requires one F32 host output".into(),
            ));
        }
        let mut values = vec![0f32; self.outputs[0].1 / 4];
        let out = [ffi::Binding {
            data: values.as_mut_ptr().cast(),
            bytes: std::mem::size_of_val(values.as_slice()),
        }];
        let (gpu_timings, total_ms) = self.execute(inputs, &out)?;
        let mut result = GraphOutput::timings(gpu_timings.dispatch_ms, total_ms);
        result.gpu_timings = gpu_timings;
        Ok((values, result))
    }
    fn execute(
        &mut self,
        inputs: &[(&str, &[u8])],
        out: &[ffi::Binding],
    ) -> Result<(GpuTimings, f64), DirectMlError> {
        let ordered = &mut self.binding_workspace;
        ordered.clear();
        if inputs.len() != self.inputs.len() {
            return Err(DirectMlError::Graph("input count mismatch".into()));
        }
        for (index, (name, size)) in self.inputs.iter().enumerate() {
            let (_, data) = inputs
                .get(index)
                .filter(|(n, _)| *n == name)
                .or_else(|| inputs.iter().find(|(n, _)| *n == name))
                .ok_or_else(|| DirectMlError::Graph(format!("missing input {name}")))?;
            if data.len() != *size {
                return Err(DirectMlError::Graph(format!(
                    "{name}: expected {size} bytes, got {}",
                    data.len()
                )));
            }
            ordered.push(ffi::Binding {
                data: data.as_ptr() as *mut u8,
                bytes: data.len(),
            });
        }
        let mut timings = GpuTimings::default();
        let started = Instant::now();
        #[cfg(windows)]
        {
            let _guard = self
                .device
                .0
                .lock
                .lock()
                .map_err(|_| DirectMlError::Runtime("DirectML device lock poisoned".into()))?;
            let mut error = ptr::null_mut();
            check(
                unsafe {
                    ffi::vdml_run(
                        self.raw,
                        ordered.as_ptr(),
                        ordered.len(),
                        out.as_ptr(),
                        out.len(),
                        &mut timings,
                        &mut error,
                    )
                },
                error,
            )?;
        }
        #[cfg(not(windows))]
        {
            return Err(DirectMlError::Runtime("DirectML requires Windows".into()));
        }
        Ok((timings, started.elapsed().as_secs_f64() * 1000.))
    }
}
impl Drop for DirectMlGraph {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            ffi::vdml_graph_free(self.raw)
        }
    }
}
#[cfg(windows)]
pub(crate) fn check(code: i32, error: *mut c_char) -> Result<(), DirectMlError> {
    if code == 0 {
        return Ok(());
    }
    let message = if error.is_null() {
        "native DirectML failure".into()
    } else {
        let s = unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned();
        unsafe { ffi::vdml_error_free(error) };
        s
    };
    Err(DirectMlError::Runtime(message))
}

#[cfg(all(test, windows))]
#[path = "runtime_tests.rs"]
mod optimization_tests;

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::Precision;

    fn bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    #[ignore = "requires a DirectML hardware adapter"]
    fn owned_weights_mixed_gemm_and_replay() {
        let g = GraphBuilder::default();
        let x = g.input("x", &[2, 3], DType::F32);
        let w = g.constant(&[1., 2., 3., 4., 5., 6.], &[3, 2], Precision::Fp16);
        let y = g.matmul(&g.cast(&x, DType::F16), &w, false);
        g.mark("y", &g.cast(&y, DType::F32));
        let mut graph = DirectMlDevice::new(0).unwrap().compile(&g).unwrap();
        drop(g); // No compiled graph may retain pointers into source constants.
        for (input, expected) in [
            (vec![1., 2., 3., 4., 5., 6.], vec![22., 28., 49., 64.]),
            (vec![2., 0., 0., 0., 0., 3.], vec![2., 4., 15., 18.]),
        ] {
            let output = graph.run(&[("x", &bytes(&input))]).unwrap();
            assert_eq!(output.f32("y").unwrap(), expected);
            assert!(output.gpu_ms.is_finite() && output.gpu_ms > 0.);
        }
    }

    #[test]
    #[ignore = "requires a DirectML hardware adapter"]
    fn bounded_weight_upload_spans_batches_and_pads_half_scalars() {
        let g = GraphBuilder::default();
        let elements = 16 * 1024 * 1024 / 4 + 3;
        let x = g.input("x", &[1, 1], DType::F32);
        let weights = g.constant(&vec![0.5; elements], &[1, elements], Precision::Fp32);
        let bias = g.scalar(0.25, Precision::Fp16);
        let y = g.add(&g.add(&x, &weights), &g.cast(&bias, DType::F32));
        g.mark("y", &g.reduce(Op::Mean, &y, &[1]));
        let mut graph = DirectMlDevice::with_debug_layer(0, true)
            .unwrap()
            .compile(&g)
            .unwrap();
        let memory = graph.memory();
        assert_eq!(memory.staging_bytes, 16 * 1024 * 1024);
        assert_eq!(memory.constant_bytes, elements as u64 * 4 + 2);
        assert_eq!(memory.constant_buffer_bytes, elements as u64 * 4 + 4);
        drop(g);
        for (input, expected) in [(2., 2.75), (-1., -0.25)] {
            let out = graph.run(&[("x", &bytes(&[input]))]).unwrap();
            assert!((out.f32("y").unwrap()[0] - expected).abs() < 1e-4);
        }
    }

    #[test]
    #[ignore = "requires a DirectML hardware adapter"]
    fn invalid_constant_extent_does_not_poison_device() {
        let device = DirectMlDevice::new(0).unwrap();
        let g = GraphBuilder::default();
        let x = g.input("x", &[1], DType::F32);
        let bias = g.scalar(0.5, Precision::Fp16);
        g.mark("y", &g.add(&x, &g.cast(&bias, DType::F32)));
        g.nodes.borrow_mut()[bias.id as usize].data.extend([0; 4]);
        let error = device.compile(&g).err().unwrap();
        assert!(error.to_string().contains("invalid constant data extent"));
        g.nodes.borrow_mut()[bias.id as usize].data.truncate(2);
        let mut graph = device.compile(&g).unwrap();
        drop(g);
        let out = graph.run(&[("x", &bytes(&[2.]))]).unwrap();
        assert_eq!(out.f32("y").unwrap(), [2.5]);
    }

    #[test]
    #[ignore = "requires a DirectML hardware adapter"]
    fn device_feedback_replay_direct_audio_and_reset() {
        let g = GraphBuilder::default();
        let x = g.input("x", &[1, 2], DType::F32);
        let history = g.input("history", &[1, 8], DType::F32);
        let next = g.concat(&[g.slice(&history, 1, 2, 6, 1), x], 1);
        g.feedback(&history, &next);
        g.mark("y", &g.slice(&next, 1, 4, 4, 1));
        let mut graph = DirectMlDevice::with_debug_layer(0, true)
            .unwrap()
            .compile(&g)
            .unwrap();
        assert_eq!(graph.input_names().collect::<Vec<_>>(), ["x"]);
        let out = graph.run(&[("x", &bytes(&[1., 2.]))]).unwrap();
        assert_eq!(out.tensors.len(), 1);
        assert_eq!(out.f32("y").unwrap(), [0., 0., 1., 2.]);
        assert!(graph.run(&[("x", &bytes(&[9.]))]).is_err());
        let (audio, _) = graph.run_f32("y", &[("x", &bytes(&[3., 4.]))]).unwrap();
        assert_eq!(audio, [1., 2., 3., 4.]);
        graph.reset_states().unwrap();
        let (audio, _) = graph.run_f32("y", &[("x", &bytes(&[9., 10.]))]).unwrap();
        assert_eq!(audio, [0., 0., 9., 10.]);
    }

    #[test]
    #[ignore = "requires a DirectML hardware adapter"]
    fn sparse_ivf_empty_lists_short_lists_and_multiple_probes() {
        use crate::RetrievalSearch;
        let device = DirectMlDevice::with_debug_layer(0, true).unwrap();
        assert!(!device.adapter_name().is_empty());
        let queries = [0., 1., 10., 1.2, 21., 0., 30., 1.5];
        for fixture in [
            include_bytes!("../tests/fixtures/ivf-small-probe1.index").as_slice(),
            include_bytes!("../tests/fixtures/ivf-small-probe2.index").as_slice(),
        ] {
            let index = vocal_model::RetrievalIndex::from_faiss(fixture).unwrap();
            let mut dense = device
                .compile_retrieval_with_search(&index, 4, RetrievalSearch::Dense)
                .unwrap();
            let mut sparse = device
                .compile_retrieval_with_search(&index, 4, RetrievalSearch::Sparse)
                .unwrap();
            let query_bytes = bytes(&queries);
            let rate = 1f32.to_le_bytes();
            let bindings = [
                ("features", query_bytes.as_slice()),
                ("index_ratio", rate.as_slice()),
            ];
            let d = dense.run(&bindings).unwrap().f32("features").unwrap();
            let s = sparse.run(&bindings).unwrap().f32("features").unwrap();
            assert!(
                d.iter().zip(&s).all(|(a, b)| (a - b).abs() < 2e-4),
                "{d:?} != {s:?}"
            );
            assert_eq!(&s[..2], &[0., 1.]);
            if index.nprobe() == 1 {
                assert_eq!(&s[4..6], &[21., 0.]);
            } else {
                assert!((s[4] - 30.).abs() < 2e-5);
            }
        }
    }

    #[test]
    #[ignore = "requires a DirectML hardware adapter"]
    fn transpose_gather_reverse_broadcast_and_conv() {
        let g = GraphBuilder::default();
        let x = g.input("x", &[1, 2, 3], DType::F32);
        let reversed = g.slice(&x, 2, 2, 3, -1);
        let transposed = g.transpose(&reversed, &[0, 2, 1]);
        let picked = g.gather(&transposed, &g.integers(&[2, 0], &[2]), 1);
        let bias = g.constant(&[10., 20.], &[2], Precision::Fp32);
        g.mark("picked", &g.add(&picked, &bias));
        let w = g.constant(&[1., 2.], &[1, 1, 1, 2], Precision::Fp32);
        let y = g.conv(
            &g.reshape(&x, &[1, 1, 2, 3]),
            &w,
            None,
            [1, 1],
            [1, 1],
            [0, 0],
            1,
            false,
            [0, 0],
        );
        g.mark("conv", &y);
        let mut graph = DirectMlDevice::new(0).unwrap().compile(&g).unwrap();
        let output = graph
            .run(&[("x", &bytes(&[1., 2., 3., 4., 5., 6.]))])
            .unwrap();
        assert_eq!(output.f32("picked").unwrap(), [11., 24., 13., 26.]);
        assert_eq!(output.f32("conv").unwrap(), [5., 8., 14., 17.]);
    }

    #[test]
    #[ignore = "requires a DirectML hardware adapter"]
    fn transposed_convolution_output_padding_matches_torch() {
        let g = GraphBuilder::default();
        let x = g.input("x", &[1, 1, 2, 2], DType::F32);
        let w = g.constant(
            &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
            &[1, 1, 3, 3],
            Precision::Fp32,
        );
        let y = g.conv(&x, &w, None, [2, 2], [1, 1], [1, 1], 1, true, [1, 1]);
        g.mark("y", &y);
        let mut graph = DirectMlDevice::new(0).unwrap().compile(&g).unwrap();
        let result = graph.run(&[("x", &bytes(&[1., 2., 3., 4.]))]).unwrap();
        assert_eq!(
            result.f32("y").unwrap(),
            [
                5., 14., 10., 12., 14., 36., 24., 30., 15., 34., 20., 24., 24., 55., 32., 36.
            ]
        );
    }
}
