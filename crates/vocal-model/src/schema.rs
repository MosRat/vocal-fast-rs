//! Source-independent contract for the RVC inference graph (training encoder excluded).
use crate::{ModelError, RvcModelManifest, Tensor, TensorDType};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const RVC_TENSOR_SCHEMA: &str = "rvc-synth-v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RvcSynthConfig {
    pub feature_width: u64,
    pub uses_f0: bool,
    pub sample_rate: u32,
    pub hidden_channels: u64,
    pub latent_channels: u64,
    pub filter_channels: u64,
    pub attention_heads: u64,
    pub encoder_layers: usize,
    pub encoder_kernel: u64,
    pub relative_window: u64,
    pub relative_heads: u64,
    pub norm_epsilon: f32,
    pub speakers: u64,
    pub speaker_channels: u64,
    pub decoder_channels: u64,
    pub resblock_type: u8,
    pub resblock_kernels: Vec<u64>,
    pub resblock_dilations: Vec<Vec<u64>>,
    pub upsample_rates: Vec<u64>,
    pub upsample_kernels: Vec<u64>,
    pub flow_blocks: usize,
    pub flow_layers: usize,
    pub flow_kernel: u64,
    pub flow_dilations: Vec<u64>,
}

fn bad(message: impl Into<String>) -> ModelError {
    ModelError::UnsupportedModel(message.into())
}

impl RvcSynthConfig {
    /// The exact tensor directory required by the fixed inference graph.
    /// Linear: [in,out]; Conv1d: [out,in,k]; ConvTranspose1d: [in,out,k].
    pub fn tensor_shapes(&self) -> Result<BTreeMap<String, Vec<u64>>, ModelError> {
        self.validate()?;
        let mut shapes = BTreeMap::new();
        let h = self.hidden_channels;
        let z = self.latent_channels;
        let g = self.speaker_channels;
        let d = self.decoder_channels;
        let mut affine = |base: &str, shape: Vec<u64>, bias: Option<u64>| {
            shapes.insert(format!("{base}.weight"), shape);
            if let Some(channels) = bias {
                shapes.insert(format!("{base}.bias"), vec![channels]);
            }
        };
        affine("enc_p.emb_phone", vec![self.feature_width, h], Some(h));
        if self.uses_f0 {
            affine("enc_p.emb_pitch", vec![256, h], None);
        }
        affine("emb_g", vec![self.speakers, g], None);
        affine("enc_p.proj", vec![2 * z, h, 1], Some(2 * z));
        for i in 0..self.encoder_layers {
            for projection in ["q", "k", "v", "o"] {
                affine(
                    &format!("enc_p.encoder.attn_layers.{i}.conv_{projection}"),
                    vec![h, h, 1],
                    Some(h),
                );
            }
            affine(
                &format!("enc_p.encoder.ffn_layers.{i}.conv_1"),
                vec![self.filter_channels, h, self.encoder_kernel],
                Some(self.filter_channels),
            );
            affine(
                &format!("enc_p.encoder.ffn_layers.{i}.conv_2"),
                vec![h, self.filter_channels, self.encoder_kernel],
                Some(h),
            );
        }
        affine("dec.conv_pre", vec![d, z, 7], Some(d));
        affine("dec.cond", vec![d, g, 1], Some(d));
        if self.uses_f0 {
            affine("dec.m_source.l_linear", vec![1, 1], Some(1));
        }
        for i in 0..self.upsample_rates.len() {
            let input = d / (1 << i);
            let output = d / (1 << (i + 1));
            affine(
                &format!("dec.ups.{i}"),
                vec![input, output, self.upsample_kernels[i]],
                Some(output),
            );
            if self.uses_f0 {
                let tail: u64 = self.upsample_rates[i + 1..].iter().product();
                let kernel = if i + 1 == self.upsample_rates.len() {
                    1
                } else {
                    2 * tail
                };
                affine(
                    &format!("dec.noise_convs.{i}"),
                    vec![output, 1, kernel],
                    Some(output),
                );
            }
            for j in 0..self.resblock_kernels.len() {
                let b = i * self.resblock_kernels.len() + j;
                for k in 0..self.resblock_dilations[j].len() {
                    let lists: &[&str] = if self.resblock_type == 1 {
                        &["convs1", "convs2"]
                    } else {
                        &["convs"]
                    };
                    for list in lists {
                        affine(
                            &format!("dec.resblocks.{b}.{list}.{k}"),
                            vec![output, output, self.resblock_kernels[j]],
                            Some(output),
                        );
                    }
                }
            }
        }
        affine(
            "dec.conv_post",
            vec![1, d / (1 << self.upsample_rates.len()), 7],
            None,
        );
        for i in 0..self.flow_blocks {
            let base = format!("flow.flows.{}", i * 2);
            affine(&format!("{base}.pre"), vec![h, z / 2, 1], Some(h));
            affine(&format!("{base}.post"), vec![z / 2, h, 1], Some(z / 2));
            affine(
                &format!("{base}.enc.cond_layer"),
                vec![2 * h * self.flow_layers as u64, g, 1],
                Some(2 * h * self.flow_layers as u64),
            );
            for j in 0..self.flow_layers {
                affine(
                    &format!("{base}.enc.in_layers.{j}"),
                    vec![2 * h, h, self.flow_kernel],
                    Some(2 * h),
                );
                let out = if j + 1 == self.flow_layers { h } else { 2 * h };
                affine(
                    &format!("{base}.enc.res_skip_layers.{j}"),
                    vec![out, h, 1],
                    Some(out),
                );
            }
        }
        for i in 0..self.encoder_layers {
            for kind in ["k", "v"] {
                shapes.insert(
                    format!("enc_p.encoder.attn_layers.{i}.emb_rel_{kind}"),
                    vec![
                        self.relative_heads,
                        2 * self.relative_window + 1,
                        h / self.attention_heads,
                    ],
                );
            }
            for norm in [1, 2] {
                for parameter in ["gamma", "beta"] {
                    shapes.insert(
                        format!("enc_p.encoder.norm_layers_{norm}.{i}.{parameter}"),
                        vec![h],
                    );
                }
            }
        }
        Ok(shapes)
    }

    pub(crate) fn validate(&self) -> Result<(), ModelError> {
        if ![256, 768].contains(&self.feature_width)
            || ![16_000, 32_000, 40_000, 48_000].contains(&self.sample_rate)
            || !(1..=4096).contains(&self.hidden_channels)
            || !(2..=4096).contains(&self.latent_channels)
            || !self.latent_channels.is_multiple_of(2)
            || !(1..=16384).contains(&self.filter_channels)
            || !(1..=32).contains(&self.attention_heads)
            || !self.hidden_channels.is_multiple_of(self.attention_heads)
            || !(1..=64).contains(&self.encoder_layers)
            || !(1..=31).contains(&self.encoder_kernel)
            || self.encoder_kernel.is_multiple_of(2)
            || self.relative_window > 128
            || !(self.relative_heads == 1 || self.relative_heads == self.attention_heads)
            || !self.norm_epsilon.is_finite()
            || self.norm_epsilon <= 0.0
            || self.norm_epsilon > 0.1
            || !(1..=100000).contains(&self.speakers)
            || !(1..=4096).contains(&self.speaker_channels)
            || !(1..=4096).contains(&self.decoder_channels)
            || !(1..=8).contains(&self.upsample_rates.len())
            || self.upsample_rates.len() != self.upsample_kernels.len()
            || !(1..=8).contains(&self.resblock_kernels.len())
            || self.resblock_kernels.len() != self.resblock_dilations.len()
            || ![1, 2].contains(&self.resblock_type)
            || !(1..=16).contains(&self.flow_blocks)
            || !(1..=16).contains(&self.flow_layers)
            || self.flow_dilations.len() != self.flow_layers
            || self.flow_dilations.iter().any(|&x| x == 0 || x > 128)
            || !(1..=31).contains(&self.flow_kernel)
            || self.flow_kernel.is_multiple_of(2)
        {
            return Err(bad(
                "RVC canonical configuration is outside supported bounds",
            ));
        }
        if self.decoder_channels / (1 << self.upsample_rates.len()) == 0
            || self
                .upsample_rates
                .iter()
                .zip(&self.upsample_kernels)
                .any(|(&r, &k)| r == 0 || r > 64 || k < r || k > 128)
            || self
                .resblock_kernels
                .iter()
                .any(|&k| k == 0 || k > 127 || k.is_multiple_of(2))
            || self.resblock_dilations.iter().any(|d| {
                d.len() != if self.resblock_type == 1 { 3 } else { 2 }
                    || d.iter().any(|&v| v == 0 || v > 128)
            })
        {
            return Err(bad("unsupported RVC decoder configuration"));
        }
        Ok(())
    }

    pub(crate) fn from_onnx_parts(
        manifest: &RvcModelManifest,
        tensors: &[Tensor],
        upsample_rates: Vec<u64>,
        upsample_kernels: Vec<u64>,
        resblock_dilations: Vec<Vec<u64>>,
        resblock_type: u8,
    ) -> Result<Self, ModelError> {
        let shape = |name: &str| {
            tensors
                .iter()
                .find(|t| t.name == name)
                .map(|t| t.shape.clone())
                .ok_or_else(|| bad(format!("missing canonical tensor {name}")))
        };
        let phone = shape("enc_p.emb_phone.weight")?;
        let hidden = shape("enc_p.emb_phone.bias")?.first().copied().unwrap_or(0);
        let latent = shape("enc_p.proj.weight")?.first().copied().unwrap_or(0) / 2;
        let filter = shape("enc_p.encoder.ffn_layers.0.conv_1.weight")?
            .first()
            .copied()
            .unwrap_or(0);
        let rel = shape("enc_p.encoder.attn_layers.0.emb_rel_k")?;
        let speakers = shape("emb_g.weight")?;
        let decoder = shape("dec.conv_pre.weight")?;
        let encoder_layers = tensors
            .iter()
            .filter_map(|t| {
                t.name
                    .strip_prefix("enc_p.encoder.attn_layers.")
                    .and_then(|s| s.split('.').next()?.parse::<usize>().ok())
            })
            .max()
            .map_or(0, |x| x + 1);
        let flow_ids = tensors
            .iter()
            .filter_map(|t| {
                t.name
                    .strip_prefix("flow.flows.")
                    .and_then(|s| s.split('.').next()?.parse::<usize>().ok())
            })
            .collect::<std::collections::BTreeSet<_>>();
        let flow_layers = tensors
            .iter()
            .filter_map(|t| {
                t.name.strip_prefix("flow.flows.").and_then(|s| {
                    s.split(".enc.in_layers.")
                        .nth(1)?
                        .split('.')
                        .next()?
                        .parse::<usize>()
                        .ok()
                })
            })
            .max()
            .map_or(0, |x| x + 1);
        let config = Self {
            feature_width: *phone
                .first()
                .ok_or_else(|| bad("phone projection has no dimensions"))?,
            uses_f0: manifest.uses_f0(),
            sample_rate: manifest.sample_rate,
            hidden_channels: hidden,
            latent_channels: latent,
            filter_channels: filter,
            attention_heads: if rel.len() == 3 && rel[2] > 0 {
                hidden / rel[2]
            } else {
                0
            },
            encoder_layers,
            encoder_kernel: shape("enc_p.encoder.ffn_layers.0.conv_1.weight")?
                .last()
                .copied()
                .unwrap_or(0),
            relative_window: if rel.len() == 3 { (rel[1] - 1) / 2 } else { 0 },
            relative_heads: rel.first().copied().unwrap_or(0),
            norm_epsilon: 1e-5,
            speakers: speakers.first().copied().unwrap_or(0),
            speaker_channels: speakers.get(1).copied().unwrap_or(0),
            decoder_channels: decoder.first().copied().unwrap_or(0),
            resblock_type,
            resblock_kernels: (0..resblock_dilations.len())
                .map(|j| {
                    let prefix = if resblock_type == 1 {
                        format!("dec.resblocks.{j}.convs1.0.weight")
                    } else {
                        format!("dec.resblocks.{j}.convs.0.weight")
                    };
                    shape(&prefix).map(|s| *s.last().unwrap_or(&0))
                })
                .collect::<Result<_, _>>()?,
            resblock_dilations,
            upsample_rates,
            upsample_kernels,
            flow_blocks: flow_ids.len(),
            flow_layers,
            flow_kernel: tensors
                .iter()
                .find(|t| t.name.contains(".enc.in_layers.0.weight"))
                .and_then(|t| t.shape.last())
                .copied()
                .unwrap_or(0),
            flow_dilations: vec![1; flow_layers],
        };
        config.validate()?;
        Ok(config)
    }

    pub(crate) fn from_pth(
        manifest: &RvcModelManifest,
        metadata: &Value,
        tensors: &[Tensor],
    ) -> Result<Self, ModelError> {
        let c = metadata
            .get("config")
            .and_then(Value::as_array)
            .filter(|c| c.len() == 18)
            .ok_or_else(|| bad("RVC checkpoint needs an 18-element config"))?;
        let integer = |i: usize| {
            c[i].as_u64()
                .ok_or_else(|| bad(format!("RVC config[{i}] must be an unsigned integer")))
        };
        let list = |i: usize| {
            serde_json::from_value::<Vec<u64>>(c[i].clone())
                .map_err(|_| bad(format!("RVC config[{i}] must be an integer array")))
        };
        let speaker = tensors
            .iter()
            .find(|t| t.name == "emb_g.weight")
            .filter(|t| t.shape.len() == 2)
            .ok_or_else(|| bad("speaker embedding must have rank 2"))?;
        let rel = tensors
            .iter()
            .find(|t| t.name == "enc_p.encoder.attn_layers.0.emb_rel_k")
            .filter(|t| t.shape.len() == 3 && t.shape[1] > 0)
            .ok_or_else(|| bad("missing relative attention embedding"))?;
        let resblock_type = c[9]
            .as_str()
            .and_then(|s| s.parse::<u8>().ok())
            .or_else(|| c[9].as_u64().and_then(|n| u8::try_from(n).ok()))
            .ok_or_else(|| bad("invalid RVC resblock type"))?;
        let config = Self {
            feature_width: manifest.feature_dims() as u64,
            uses_f0: manifest.uses_f0(),
            sample_rate: manifest.sample_rate,
            latent_channels: integer(2)?,
            hidden_channels: integer(3)?,
            filter_channels: integer(4)?,
            attention_heads: integer(5)?,
            encoder_layers: usize::try_from(integer(6)?)
                .map_err(|_| bad("encoder layers overflow"))?,
            encoder_kernel: integer(7)?,
            relative_window: (rel.shape[1] - 1) / 2,
            relative_heads: rel.shape[0],
            norm_epsilon: 1e-5,
            speakers: speaker.shape[0],
            speaker_channels: integer(16)?,
            decoder_channels: integer(13)?,
            resblock_type,
            resblock_kernels: list(10)?,
            resblock_dilations: serde_json::from_value(c[11].clone())
                .map_err(|_| bad("invalid RVC dilation configuration"))?,
            upsample_rates: list(12)?,
            upsample_kernels: list(14)?,
            flow_blocks: 4,
            flow_layers: 3,
            flow_kernel: 5,
            flow_dilations: vec![1; 3],
        };
        config.validate()?;
        Ok(config)
    }
}

pub(crate) fn validate_tensors(
    config: &RvcSynthConfig,
    manifest: &RvcModelManifest,
    tensors: &[Tensor],
) -> Result<(), ModelError> {
    if config.feature_width != manifest.feature_dims() as u64
        || config.uses_f0 != manifest.uses_f0()
        || config.sample_rate != manifest.sample_rate
    {
        return Err(bad("canonical RVC config disagrees with manifest"));
    }
    let expected = config.tensor_shapes()?;
    let mut actual = BTreeMap::new();
    for t in tensors {
        t.validate()?;
        if t.name == "aux.retrieval.faiss" && t.dtype == TensorDType::U8 && t.shape.len() == 1 {
            continue;
        }
        if !matches!(
            t.dtype,
            TensorDType::F16 | TensorDType::F32 | TensorDType::BF16
        ) {
            return Err(bad(format!(
                "RVC parameter {} must be floating point",
                t.name
            )));
        }
        if actual.insert(t.name.as_str(), t).is_some() {
            return Err(bad(format!("duplicate canonical parameter {}", t.name)));
        }
        let shape = expected
            .get(&t.name)
            .ok_or_else(|| bad(format!("unknown canonical parameter {}", t.name)))?;
        if &t.shape != shape {
            return Err(bad(format!(
                "canonical parameter {} shape {:?}, expected {:?}",
                t.name, t.shape, shape
            )));
        }
    }
    for name in expected.keys() {
        if !actual.contains_key(name.as_str()) {
            return Err(bad(format!("missing canonical parameter {name}")));
        }
    }
    Ok(())
}
