//! Canonical RVC Synthesizer graph construction.
//!
//! This module is intentionally independent of the TensorRT SDK. It expands
//! one validated VFRM package into a deterministic graph IR. The optional
//! `vocal-tensorrt` crate consumes the IR on machines that have TensorRT-RTX.

use std::{
    collections::BTreeSet,
    env, fmt, fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use vocal_model::{ModelPackage, RVC_TENSOR_SCHEMA, RvcSynthConfig, TensorDType};

use crate::EngineError;

pub const TENSORRT_GRAPH_SCHEMA: &str = "rvc-tensorrt-graph-v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphDim {
    Static { value: u32 },
    Dynamic { min: u32, opt: u32, max: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphDType {
    Float32,
    Float16,
    Int64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphTensor {
    pub name: String,
    pub dtype: GraphDType,
    pub shape: Vec<GraphDim>,
    pub input: bool,
    pub output: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphNode {
    pub name: String,
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub attributes: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorRtGraph {
    pub schema: String,
    pub source_graph_schema: String,
    pub feature_width: u32,
    pub uses_f0: bool,
    pub tensors: Vec<GraphTensor>,
    pub nodes: Vec<GraphNode>,
    pub weights: Vec<String>,
}

impl TensorRtGraph {
    pub fn input(&self, name: &str) -> Option<&GraphTensor> {
        self.tensors.iter().find(|x| x.input && x.name == name)
    }
    pub fn output(&self, name: &str) -> Option<&GraphTensor> {
        self.tensors.iter().find(|x| x.output && x.name == name)
    }
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, GraphError> {
        serde_json::to_vec_pretty(self).map_err(GraphError::Json)
    }
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<(), GraphError> {
        fs::write(path, self.to_json_bytes()?).map_err(GraphError::Io)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRtGraphOptions {
    pub min_frames: u32,
    pub opt_frames: u32,
    pub max_frames: u32,
}
impl Default for TensorRtGraphOptions {
    fn default() -> Self {
        Self {
            min_frames: 8,
            opt_frames: 192,
            max_frames: 2048,
        }
    }
}
impl TensorRtGraphOptions {
    fn validate(&self) -> Result<(), GraphError> {
        if self.min_frames == 0
            || self.min_frames > self.opt_frames
            || self.opt_frames > self.max_frames
        {
            return Err(GraphError::InvalidConfig(format!(
                "invalid frame profile [{}, {}, {}]",
                self.min_frames, self.opt_frames, self.max_frames
            )));
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("invalid TensorRT graph configuration: {0}")]
    InvalidConfig(String),
    #[error("canonical model error: {0}")]
    Model(#[from] vocal_model::ModelError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TensorRT compiler is unavailable: {0}")]
    CompilerUnavailable(String),
    #[error("TensorRT compiler failed: {0}")]
    CompilerFailed(String),
}
impl From<GraphError> for EngineError {
    fn from(e: GraphError) -> Self {
        EngineError::BackendUnavailable(e.to_string())
    }
}

pub fn build_rvc_graph(
    package: &ModelPackage,
    options: TensorRtGraphOptions,
) -> Result<TensorRtGraph, GraphError> {
    options.validate()?;
    if package
        .metadata
        .get("canonical_schema")
        .and_then(Value::as_str)
        != Some(RVC_TENSOR_SCHEMA)
    {
        return Err(GraphError::InvalidConfig(
            "package is not canonical rvc-synth-v1".into(),
        ));
    }
    let config = package.validate_rvc_synth()?;
    let source_graph_schema = package
        .metadata
        .get("graph_schema")
        .and_then(Value::as_str)
        .unwrap_or(match (config.feature_width, config.uses_f0) {
            (256, true) => "rvc-v1-f0-feats",
            (256, false) => "rvc-v1-nof0-feats",
            (_, true) => "rvc-v2-f0-feats",
            (_, false) => "rvc-v2-nof0-feats",
        })
        .to_owned();
    let shapes = config.tensor_shapes()?;
    let mut weights = Vec::with_capacity(shapes.len());
    for (name, shape) in &shapes {
        let tensor = package
            .tensor(name)
            .ok_or_else(|| GraphError::InvalidConfig(format!("missing canonical weight {name}")))?;
        if tensor.shape != *shape
            || !matches!(
                tensor.dtype,
                TensorDType::F16 | TensorDType::F32 | TensorDType::BF16
            )
        {
            return Err(GraphError::InvalidConfig(format!(
                "invalid canonical weight {name}"
            )));
        }
        weights.push(name.clone());
    }
    let mut b = Builder {
        config: &config,
        options,
        graph: TensorRtGraph {
            schema: TENSORRT_GRAPH_SCHEMA.into(),
            source_graph_schema,
            feature_width: config.feature_width as u32,
            uses_f0: config.uses_f0,
            tensors: Vec::new(),
            nodes: Vec::new(),
            weights: weights.clone(),
        },
    };
    b.inputs();
    b.encoder();
    b.flow();
    b.decoder();
    b.outputs()?;
    let referenced = b
        .graph
        .nodes
        .iter()
        .flat_map(|n| n.inputs.iter())
        .filter_map(|x| x.strip_prefix("@weight:"))
        .collect::<BTreeSet<_>>();
    let declared = weights.iter().map(String::as_str).collect::<BTreeSet<_>>();
    if referenced != declared {
        return Err(GraphError::InvalidConfig(format!(
            "graph weight bindings do not exactly match VFRM; missing={:?}, unknown={:?}",
            declared.difference(&referenced).collect::<Vec<_>>(),
            referenced.difference(&declared).collect::<Vec<_>>()
        )));
    }
    Ok(b.graph)
}

struct Builder<'a> {
    config: &'a RvcSynthConfig,
    options: TensorRtGraphOptions,
    graph: TensorRtGraph,
}
impl<'a> Builder<'a> {
    fn batch() -> GraphDim {
        GraphDim::Static { value: 1 }
    }
    fn frames(&self) -> GraphDim {
        GraphDim::Dynamic {
            min: self.options.min_frames,
            opt: self.options.opt_frames,
            max: self.options.max_frames,
        }
    }
    fn tensor(
        &mut self,
        name: &str,
        dtype: GraphDType,
        shape: Vec<GraphDim>,
        input: bool,
        output: bool,
    ) {
        self.graph.tensors.push(GraphTensor {
            name: name.into(),
            dtype,
            shape,
            input,
            output,
        });
    }
    fn node(
        &mut self,
        name: &str,
        op: &str,
        inputs: &[&str],
        output: &str,
        weights: &[String],
        attrs: Map<String, Value>,
    ) {
        let mut all = inputs.iter().map(|x| (*x).into()).collect::<Vec<_>>();
        all.extend(weights.iter().map(|x| format!("@weight:{x}")));
        self.graph.nodes.push(GraphNode {
            name: name.into(),
            op: op.into(),
            inputs: all,
            outputs: vec![output.into()],
            attributes: attrs,
        });
    }
    fn attrs(items: impl IntoIterator<Item = (&'static str, Value)>) -> Map<String, Value> {
        items.into_iter().map(|(k, v)| (k.into(), v)).collect()
    }
    fn weights(&self, names: impl IntoIterator<Item = String>) -> Vec<String> {
        names.into_iter().collect()
    }
    fn inputs(&mut self) {
        let f = self.frames();
        self.tensor(
            "feats",
            GraphDType::Float32,
            vec![
                Self::batch(),
                f.clone(),
                GraphDim::Static {
                    value: self.config.feature_width as u32,
                },
            ],
            true,
            false,
        );
        self.tensor("p_len", GraphDType::Int64, vec![Self::batch()], true, false);
        self.tensor("sid", GraphDType::Int64, vec![Self::batch()], true, false);
        if self.config.uses_f0 {
            self.tensor(
                "pitch",
                GraphDType::Int64,
                vec![Self::batch(), f.clone()],
                true,
                false,
            );
            self.tensor(
                "pitchf",
                GraphDType::Float32,
                vec![Self::batch(), f.clone()],
                true,
                false,
            );
        }
        self.tensor(
            "rnd",
            GraphDType::Float32,
            vec![
                Self::batch(),
                GraphDim::Static {
                    value: self.config.latent_channels as u32,
                },
                f,
            ],
            true,
            false,
        );
    }
    fn encoder(&mut self) {
        self.node(
            "enc_p.emb_phone",
            "linear",
            &["feats"],
            "enc_phone",
            &self.weights(vec![
                "enc_p.emb_phone.weight".into(),
                "enc_p.emb_phone.bias".into(),
            ]),
            Map::new(),
        );
        let mut x = "enc_phone";
        if self.config.uses_f0 {
            self.node(
                "enc_p.emb_pitch",
                "embedding",
                &[x, "pitch"],
                "enc_pitch",
                &self.weights(vec!["enc_p.emb_pitch.weight".into()]),
                Map::new(),
            );
            self.node(
                "enc_p.add_phone_pitch",
                "add",
                &[x, "enc_pitch"],
                "enc_input",
                &[],
                Map::new(),
            );
            x = "enc_input";
        }
        self.node(
            "enc_p.scale",
            "scale",
            &[x],
            "enc_scaled",
            &[],
            Self::attrs([("scale", json!((self.config.hidden_channels as f64).sqrt()))]),
        );
        self.node(
            "enc_p.leaky_relu",
            "leaky_relu",
            &["enc_scaled"],
            "enc_activation",
            &[],
            Map::new(),
        );
        self.node(
            "enc_p.transpose",
            "transpose",
            &["enc_activation"],
            "enc_hidden",
            &[],
            Self::attrs([("perm", json!([0, 2, 1]))]),
        );
        self.node(
            "enc_p.mask",
            "sequence_mask",
            &["p_len", "enc_hidden"],
            "enc_mask",
            &[],
            Self::attrs([("axis", json!(2))]),
        );
        self.node(
            "enc_p.mask_input",
            "mul",
            &["enc_hidden", "enc_mask"],
            "enc_masked",
            &[],
            Map::new(),
        );
        for i in 0..self.config.encoder_layers {
            let inp = if i == 0 {
                "enc_masked".to_owned()
            } else {
                format!("enc_l{}_in", i)
            };
            let attn = format!("enc_l{i}_attn");
            let out = format!("enc_l{i}_out");
            let p = format!("enc_p.encoder.attn_layers.{i}");
            let w = self.weights(
                ["conv_q", "conv_k", "conv_v", "conv_o"]
                    .into_iter()
                    .flat_map(|q| [format!("{p}.{q}.weight"), format!("{p}.{q}.bias")])
                    .chain([
                        format!("{p}.emb_rel_k"),
                        format!("{p}.emb_rel_v"),
                        format!("enc_p.encoder.norm_layers_1.{i}.gamma"),
                        format!("enc_p.encoder.norm_layers_1.{i}.beta"),
                        format!("enc_p.encoder.norm_layers_2.{i}.gamma"),
                        format!("enc_p.encoder.norm_layers_2.{i}.beta"),
                    ]),
            );
            self.node(
                &format!("enc_p.encoder.{i}.attention"),
                "multi_head_attention",
                &[&inp],
                &attn,
                &w,
                Self::attrs([
                    ("heads", json!(self.config.attention_heads)),
                    ("relative_window", json!(self.config.relative_window)),
                ]),
            );
            let q = format!("enc_p.encoder.ffn_layers.{i}");
            let w = self.weights([
                format!("{q}.conv_1.weight"),
                format!("{q}.conv_1.bias"),
                format!("{q}.conv_2.weight"),
                format!("{q}.conv_2.bias"),
            ]);
            self.node(
                &format!("enc_p.encoder.{i}.ffn"),
                "feed_forward",
                &[&attn],
                &out,
                &w,
                Self::attrs([("kernel", json!(self.config.encoder_kernel))]),
            );
            if i + 1 < self.config.encoder_layers {
                self.node(
                    &format!("enc_p.encoder.{i}.next"),
                    "identity",
                    &[&out],
                    &format!("enc_l{}_in", i + 1),
                    &[],
                    Map::new(),
                );
            }
        }
        let last = format!("enc_l{}_out", self.config.encoder_layers - 1);
        self.node(
            "enc_p.proj",
            "conv1d",
            &[&last],
            "stats",
            &self.weights(vec!["enc_p.proj.weight".into(), "enc_p.proj.bias".into()]),
            Map::new(),
        );
        self.node(
            "enc_p.split_m",
            "split",
            &["stats"],
            "m_p",
            &[],
            Self::attrs([("part", json!(0))]),
        );
        self.node(
            "enc_p.split_logs",
            "split",
            &["stats"],
            "logs_p",
            &[],
            Self::attrs([("part", json!(1))]),
        );
        self.node("enc_p.exp", "exp", &["logs_p"], "exp_logs", &[], Map::new());
        self.node(
            "enc_p.noise",
            "mul",
            &["rnd", "exp_logs"],
            "noise",
            &[],
            Self::attrs([("scale", json!(0.66666))]),
        );
        self.node(
            "enc_p.add_noise",
            "add",
            &["m_p", "noise"],
            "z_p",
            &[],
            Map::new(),
        );
        self.node(
            "enc_p.mask",
            "mul",
            &["z_p", "enc_mask"],
            "z_masked",
            &[],
            Map::new(),
        );
    }
    fn flow(&mut self) {
        self.node(
            "emb_g.lookup",
            "embedding",
            &["sid"],
            "speaker",
            &self.weights(vec!["emb_g.weight".into()]),
            Map::new(),
        );
        self.node(
            "emb_g.unsqueeze",
            "unsqueeze",
            &["speaker"],
            "speaker_3d",
            &[],
            Self::attrs([("axis", json!(2))]),
        );
        let mut x = "z_masked".to_owned();
        for i in (0..self.config.flow_blocks).rev() {
            let block = i * 2;
            let flip = format!("flow_{i}_flip");
            self.node(
                &format!("flow.flows.{}.flip", block + 1),
                "channel_reverse",
                &[&x],
                &flip,
                &[],
                Map::new(),
            );
            let base = format!("flow.flows.{block}");
            let mut w = vec![
                format!("{base}.pre.weight"),
                format!("{base}.pre.bias"),
                format!("{base}.post.weight"),
                format!("{base}.post.bias"),
                format!("{base}.enc.cond_layer.weight"),
                format!("{base}.enc.cond_layer.bias"),
            ];
            for j in 0..self.config.flow_layers {
                w.extend([
                    format!("{base}.enc.in_layers.{j}.weight"),
                    format!("{base}.enc.in_layers.{j}.bias"),
                    format!("{base}.enc.res_skip_layers.{j}.weight"),
                    format!("{base}.enc.res_skip_layers.{j}.bias"),
                ]);
            }
            let out = format!("flow_{i}");
            self.node(
                &format!("{base}.reverse"),
                "residual_coupling_reverse",
                &[&flip, "enc_mask", "speaker_3d"],
                &out,
                &w,
                Self::attrs([
                    ("kernel", json!(self.config.flow_kernel)),
                    ("layers", json!(self.config.flow_layers)),
                ]),
            );
            x = out;
        }
        self.node("flow.mask", "mul", &[&x, "enc_mask"], "z", &[], Map::new());
    }
    fn decoder(&mut self) {
        self.node(
            "dec.conv_pre",
            "conv1d",
            &["z"],
            "dec_pre",
            &self.weights(vec![
                "dec.conv_pre.weight".into(),
                "dec.conv_pre.bias".into(),
            ]),
            Map::new(),
        );
        self.node(
            "dec.cond",
            "conv1d",
            &["speaker_3d"],
            "dec_cond",
            &self.weights(vec!["dec.cond.weight".into(), "dec.cond.bias".into()]),
            Map::new(),
        );
        self.node(
            "dec.add_cond",
            "add",
            &["dec_pre", "dec_cond"],
            "dec_input",
            &[],
            Map::new(),
        );
        let mut x = "dec_input".to_owned();
        if self.config.uses_f0 {
            self.node(
                "dec.m_source",
                "nsf_source",
                &["pitchf"],
                "harmonic_source",
                &self.weights(vec![
                    "dec.m_source.l_linear.weight".into(),
                    "dec.m_source.l_linear.bias".into(),
                ]),
                Self::attrs([("sample_rate", json!(self.config.sample_rate))]),
            );
        }
        for i in 0..self.config.upsample_rates.len() {
            let act = format!("dec_up{i}_act");
            self.node(
                &format!("dec.ups.{i}.act"),
                "leaky_relu",
                &[&x],
                &act,
                &[],
                Map::new(),
            );
            let up = format!("dec_up{i}");
            self.node(
                &format!("dec.ups.{i}"),
                "conv_transpose1d",
                &[&act],
                &up,
                &self.weights(vec![
                    format!("dec.ups.{i}.weight"),
                    format!("dec.ups.{i}.bias"),
                ]),
                Self::attrs([
                    ("stride", json!(self.config.upsample_rates[i])),
                    ("kernel", json!(self.config.upsample_kernels[i])),
                ]),
            );
            let source = if self.config.uses_f0 {
                let s = format!("dec_source{i}");
                self.node(
                    &format!("dec.noise_convs.{i}"),
                    "conv1d",
                    &["harmonic_source"],
                    &s,
                    &self.weights(vec![
                        format!("dec.noise_convs.{i}.weight"),
                        format!("dec.noise_convs.{i}.bias"),
                    ]),
                    Map::new(),
                );
                let a = format!("dec_up{i}_source");
                self.node(
                    &format!("dec.ups.{i}.source_add"),
                    "add",
                    &[&up, &s],
                    &a,
                    &[],
                    Map::new(),
                );
                a
            } else {
                up
            };
            let mut outs = Vec::new();
            for j in 0..self.config.resblock_kernels.len() {
                let b = i * self.config.resblock_kernels.len() + j;
                let lists: &[&str] = if self.config.resblock_type == 1 {
                    &["convs1", "convs2"]
                } else {
                    &["convs"]
                };
                let mut w = Vec::new();
                for list in lists {
                    for k in 0..self.config.resblock_dilations[j].len() {
                        w.extend([
                            format!("dec.resblocks.{b}.{list}.{k}.weight"),
                            format!("dec.resblocks.{b}.{list}.{k}.bias"),
                        ]);
                    }
                }
                let o = format!("dec_res_{i}_{j}");
                self.node(
                    &format!("dec.resblocks.{b}"),
                    "residual_block",
                    &[&source],
                    &o,
                    &w,
                    Self::attrs([("kernel", json!(self.config.resblock_kernels[j]))]),
                );
                outs.push(o);
            }
            let refs = outs.iter().map(String::as_str).collect::<Vec<_>>();
            let avg = format!("dec_avg{i}");
            self.node(
                &format!("dec.ups.{i}.average"),
                "mean",
                &refs,
                &avg,
                &[],
                Self::attrs([("count", json!(outs.len()))]),
            );
            x = avg;
        }
        self.node(
            "dec.leaky_relu",
            "leaky_relu",
            &[&x],
            "dec_act",
            &[],
            Map::new(),
        );
        self.node(
            "dec.conv_post",
            "conv1d",
            &["dec_act"],
            "audio_pre_tanh",
            &self.weights(vec!["dec.conv_post.weight".into()]),
            Map::new(),
        );
        self.node(
            "dec.tanh",
            "tanh",
            &["audio_pre_tanh"],
            "audio",
            &[],
            Map::new(),
        );
    }
    fn outputs(&mut self) -> Result<(), GraphError> {
        let up = self
            .config
            .upsample_rates
            .iter()
            .try_fold(1u32, |a, &x| a.checked_mul(x as u32))
            .ok_or_else(|| GraphError::InvalidConfig("upsample product overflow".into()))?;
        let scale = |x: u32| {
            x.checked_mul(up)
                .ok_or_else(|| GraphError::InvalidConfig("output profile overflow".into()))
        };
        self.tensor(
            "audio",
            GraphDType::Float32,
            vec![
                Self::batch(),
                GraphDim::Static { value: 1 },
                GraphDim::Dynamic {
                    min: scale(self.options.min_frames)?,
                    opt: scale(self.options.opt_frames)?,
                    max: scale(self.options.max_frames)?,
                },
            ],
            false,
            true,
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRtCompileReport {
    pub engine_path: Option<PathBuf>,
    pub graph_path: PathBuf,
    pub compiler: String,
}
pub trait TensorRtCompiler {
    fn compile(
        &self,
        graph: &TensorRtGraph,
        output: impl AsRef<Path>,
    ) -> Result<TensorRtCompileReport, GraphError>;
}
#[derive(Clone, Debug)]
pub struct TrtExecCompiler {
    pub executable: PathBuf,
    pub workspace_bytes: u64,
    pub fp16: bool,
    pub onnx_source: Option<PathBuf>,
}
impl TrtExecCompiler {
    pub fn discover() -> Option<Self> {
        let executable = env::var_os("VOCAL_TRTEXEC")
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("PATH").and_then(|p| {
                    env::split_paths(&p)
                        .map(|x| x.join("trtexec.exe"))
                        .find(|x| x.is_file())
                })
            })?;
        Some(Self {
            executable,
            workspace_bytes: 1 << 30,
            fp16: true,
            onnx_source: None,
        })
    }
    pub fn with_onnx(mut self, path: impl Into<PathBuf>) -> Self {
        self.onnx_source = Some(path.into());
        self
    }
}
impl TensorRtCompiler for TrtExecCompiler {
    fn compile(
        &self,
        graph: &TensorRtGraph,
        output: impl AsRef<Path>,
    ) -> Result<TensorRtCompileReport, GraphError> {
        let output = output.as_ref();
        let graph_path = output.with_extension("graph.json");
        graph.write_json(&graph_path)?;
        let onnx = self.onnx_source.as_ref().ok_or_else(|| {
            GraphError::CompilerUnavailable(
                "trtexec needs an ONNX source; native VFRM builder requires the vocal-tensorrt SDK/plugin boundary".into(),
            )
        })?;
        let out = Command::new(&self.executable)
            .arg(format!("--onnx={}", onnx.display()))
            .arg(format!("--saveEngine={}", output.display()))
            .arg(format!(
                "--memPoolSize=workspace:{}",
                self.workspace_bytes / (1 << 20)
            ))
            .args(self.fp16.then_some("--fp16"))
            .output()?;
        if !out.status.success() {
            return Err(GraphError::CompilerFailed(
                String::from_utf8_lossy(&out.stderr).trim().into(),
            ));
        }
        Ok(TensorRtCompileReport {
            engine_path: Some(output.into()),
            graph_path,
            compiler: self.executable.display().to_string(),
        })
    }
}
impl fmt::Display for TensorRtGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} nodes, {} tensors, {} weights",
            self.schema,
            self.nodes.len(),
            self.tensors.len(),
            self.weights.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use vocal_model::{RvcModelManifest, Tensor};
    fn package() -> ModelPackage {
        let manifest = RvcModelManifest {
            id: "x".into(),
            display_name: "x".into(),
            architecture: "rvc-v2-nono".into(),
            version: "v2".into(),
            sample_rate: 16000,
            model_file: "x".into(),
            index_file: None,
            source: "test".into(),
            notes: None,
        };
        let c = RvcSynthConfig {
            feature_width: 768,
            uses_f0: false,
            sample_rate: 16000,
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
            metadata: json!({"canonical_schema":RVC_TENSOR_SCHEMA,"graph_schema":"rvc-v2-nof0-feats","rvc_config":c}),
            tensors,
        }
    }
    #[test]
    fn graph_binds_all_weights() {
        let p = package();
        let g = build_rvc_graph(&p, TensorRtGraphOptions::default()).unwrap();
        assert_eq!(
            g.weights.iter().collect::<BTreeSet<_>>(),
            p.rvc_synth_config()
                .unwrap()
                .unwrap()
                .tensor_shapes()
                .unwrap()
                .keys()
                .collect::<BTreeSet<_>>()
        );
        assert!(g.nodes.iter().any(|x| x.op == "residual_coupling_reverse"));
    }
    #[test]
    fn profile_rejects_invalid() {
        let p = package();
        assert!(
            build_rvc_graph(
                &p,
                TensorRtGraphOptions {
                    min_frames: 4,
                    opt_frames: 3,
                    max_frames: 2
                }
            )
            .is_err()
        );
    }
}
