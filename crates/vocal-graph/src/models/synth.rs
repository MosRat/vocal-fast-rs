use super::*;
use crate::PrecisionConfig;
use vocal_model::RvcSynthConfig;

fn attention(
    l: &Layers<'_>,
    x: &Tensor,
    mask: &Tensor,
    c: &RvcSynthConfig,
    layer: usize,
) -> Result<Tensor, GraphError> {
    let g = l.g;
    let p = format!("enc_p.encoder.attn_layers.{layer}");
    let h = c.attention_heads as usize;
    let d = c.hidden_channels as usize / h;
    let t = x.shape[2];
    let w = c.relative_window as usize;
    let r = 2 * w + 1;
    let mut heads = Vec::new();
    for name in ["q", "k", "v"] {
        let y = l.conv1(
            x,
            &format!("{p}.conv_{name}"),
            Precision::Fp32,
            1,
            1,
            None,
            false,
            true,
            1.,
        )?;
        heads.push(g.transpose(&g.reshape(&y, &[h, d, t]), &[0, 2, 1]));
    }
    let q = g.scale(&heads[0], 1. / (d as f32).sqrt());
    let mut scores = g.matmul(&q, &heads[1], true);
    let mut relative = None;
    if c.relative_heads > 0 {
        let rel_k = l.parameter(&format!("{p}.emb_rel_k"), Precision::Fp32)?;
        let dot = g.matmul(&q, &rel_k, true);
        let mut indices = Vec::new();
        let mut valid = Vec::new();
        for _ in 0..h {
            for row in 0..t {
                for col in 0..t {
                    let delta = col as i32 - row as i32 + w as i32;
                    indices.push(delta.clamp(0, r as i32 - 1));
                    valid.push(if (0..r as i32).contains(&delta) {
                        1.
                    } else {
                        0.
                    });
                }
            }
        }
        let local = g.gather_elements(&dot, &g.integers(&indices, &[h, t, t]), 2);
        scores = g.add(
            &scores,
            &g.mul(&local, &g.constant(&valid, &[h, t, t], Precision::Fp32)),
        );
        let mut columns = Vec::new();
        let mut valid = Vec::new();
        for _ in 0..h {
            for row in 0..t {
                for k in 0..r {
                    let col = row as i32 + k as i32 - w as i32;
                    columns.push(col.clamp(0, t as i32 - 1));
                    valid.push(if (0..t as i32).contains(&col) { 1. } else { 0. });
                }
            }
        }
        relative = Some((
            g.integers(&columns, &[h, t, r]),
            g.constant(&valid, &[h, t, r], Precision::Fp32),
        ));
    }
    let allowed = g.mul(&g.transpose(mask, &[0, 2, 1]), mask);
    let allowed = g.binary(Op::Greater, &allowed, &g.scalar(0., Precision::Fp32));
    let prob = g.softmax(
        &g.select(&allowed, &scores, &g.scalar(-1e4, Precision::Fp32)),
        2,
    );
    let mut y = g.matmul(&prob, &heads[2], false);
    if let Some((columns, valid)) = relative {
        let local = g.mul(&g.gather_elements(&prob, &columns, 2), &valid);
        y = g.add(
            &y,
            &g.matmul(
                &local,
                &l.parameter(&format!("{p}.emb_rel_v"), Precision::Fp32)?,
                false,
            ),
        );
    }
    l.conv1(
        &g.reshape(&g.transpose(&y, &[0, 2, 1]), &[1, h * d, t]),
        &format!("{p}.conv_o"),
        Precision::Fp32,
        1,
        1,
        None,
        false,
        true,
        1.,
    )
}
fn encoder(
    l: &Layers<'_>,
    feats: &Tensor,
    pitch: &Tensor,
    length: &Tensor,
    rnd: &Tensor,
    c: &RvcSynthConfig,
) -> Result<(Tensor, Tensor), GraphError> {
    let g = l.g;
    let t = feats.shape[1];
    let hidden = c.hidden_channels as usize;
    let latent = c.latent_channels as usize;
    let mut x = l.linear(feats, "enc_p.emb_phone", Precision::Fp32)?;
    if c.uses_f0 {
        x = g.add(
            &x,
            &g.gather(
                &l.parameter("enc_p.emb_pitch.weight", Precision::Fp32)?,
                pitch,
                0,
            ),
        );
    }
    x = g.transpose(
        &g.leaky(&g.scale(&x, (hidden as f32).sqrt()), 0.1),
        &[0, 2, 1],
    );
    let positions = g.integers(&(0..t as i32).collect::<Vec<_>>(), &[1, 1, t]);
    let mask = g.cast(
        &g.binary(Op::Less, &positions, &g.reshape(length, &[1, 1, 1])),
        DType::F32,
    );
    x = g.mul(&x, &mask);
    for i in 0..c.encoder_layers {
        let y = attention(l, &x, &mask, c, i)?;
        x = l.norm(
            &g.add(&x, &y),
            &format!("enc_p.encoder.norm_layers_1.{i}"),
            1,
            c.norm_epsilon,
            Precision::Fp32,
            "gamma",
            "beta",
        )?;
        let prefix = format!("enc_p.encoder.ffn_layers.{i}");
        let y = g.unary(
            Op::Relu,
            &l.conv1(
                &g.mul(&x, &mask),
                &format!("{prefix}.conv_1"),
                Precision::Fp32,
                1,
                1,
                None,
                false,
                true,
                1.,
            )?,
        );
        let y = g.mul(
            &l.conv1(
                &g.mul(&y, &mask),
                &format!("{prefix}.conv_2"),
                Precision::Fp32,
                1,
                1,
                None,
                false,
                true,
                1.,
            )?,
            &mask,
        );
        x = l.norm(
            &g.add(&x, &y),
            &format!("enc_p.encoder.norm_layers_2.{i}"),
            1,
            c.norm_epsilon,
            Precision::Fp32,
            "gamma",
            "beta",
        )?;
    }
    let stats = g.mul(
        &l.conv1(
            &g.mul(&x, &mask),
            "enc_p.proj",
            Precision::Fp32,
            1,
            1,
            None,
            false,
            true,
            1.,
        )?,
        &mask,
    );
    let mean = g.slice(&stats, 1, 0, latent, 1);
    let logs = g.slice(&stats, 1, latent, latent, 1);
    let z = g.mul(
        &g.add(
            &mean,
            &g.scale(&g.mul(&g.unary(Op::Exp, &logs), rnd), 0.66666),
        ),
        &mask,
    );
    Ok((z, mask))
}
fn flow(
    l: &Layers<'_>,
    z: &Tensor,
    mask: &Tensor,
    speaker: &Tensor,
    c: &RvcSynthConfig,
) -> Result<Tensor, GraphError> {
    let g = l.g;
    let latent = c.latent_channels as usize;
    let hidden = c.hidden_channels as usize;
    let mut z = z.clone();
    for i in (0..c.flow_blocks).rev() {
        z = g.slice(&z, 1, latent - 1, latent, -1);
        let prefix = format!("flow.flows.{}", i * 2);
        let a = g.slice(&z, 1, 0, latent / 2, 1);
        let b = g.slice(&z, 1, latent / 2, latent / 2, 1);
        let mut x = g.mul(
            &l.conv1(
                &a,
                &format!("{prefix}.pre"),
                Precision::Fp32,
                1,
                1,
                None,
                false,
                true,
                1.,
            )?,
            mask,
        );
        let cond = l.conv1(
            speaker,
            &format!("{prefix}.enc.cond_layer"),
            Precision::Fp32,
            1,
            1,
            None,
            false,
            true,
            1.,
        )?;
        let mut total = None;
        for j in 0..c.flow_layers {
            let gates = g.add(
                &l.conv1(
                    &x,
                    &format!("{prefix}.enc.in_layers.{j}"),
                    Precision::Fp32,
                    1,
                    c.flow_dilations[j] as usize,
                    None,
                    false,
                    true,
                    1.,
                )?,
                &g.slice(&cond, 1, j * 2 * hidden, 2 * hidden, 1),
            );
            let y = g.mul(
                &g.unary(Op::Tanh, &g.slice(&gates, 1, 0, hidden, 1)),
                &g.unary(Op::Sigmoid, &g.slice(&gates, 1, hidden, hidden, 1)),
            );
            let mut y = l.conv1(
                &y,
                &format!("{prefix}.enc.res_skip_layers.{j}"),
                Precision::Fp32,
                1,
                1,
                None,
                false,
                true,
                1.,
            )?;
            if j + 1 < c.flow_layers {
                x = g.mul(&g.add(&x, &g.slice(&y, 1, 0, hidden, 1)), mask);
                y = g.slice(&y, 1, hidden, hidden, 1);
            }
            total = Some(if let Some(prev) = total {
                g.add(&prev, &y)
            } else {
                y
            });
        }
        let mean = g.mul(
            &l.conv1(
                &g.mul(&total.unwrap(), mask),
                &format!("{prefix}.post"),
                Precision::Fp32,
                1,
                1,
                None,
                false,
                true,
                1.,
            )?,
            mask,
        );
        z = g.concat(&[a, g.mul(&g.sub(&b, &mean), mask)], 1);
    }
    Ok(g.mul(&z, mask))
}
fn source(
    l: &Layers<'_>,
    pitchf: &Tensor,
    noise: &Tensor,
    c: &RvcSynthConfig,
) -> Result<Tensor, GraphError> {
    let g = l.g;
    let up = c.upsample_rates.iter().product::<u64>() as usize;
    let frames = pitchf.elements();
    let f = g.reshape(pitchf, &[1, frames, 1]);
    let cycle = g.scale(&f, up as f32 / c.sample_rate as f32);
    let centered = g.sub(
        &cycle,
        &g.unary(Op::Floor, &g.add(&cycle, &g.scalar(0.5, Precision::Fp32))),
    );
    let phase = g.node(
        Op::Cumsum,
        &[&centered],
        centered.shape.clone(),
        DType::F32,
        vec![1, 1],
        [0.; 4],
        vec![],
        String::new(),
    );
    let phase = g.sub(&phase, &g.unary(Op::Floor, &phase));
    let offsets = (1..=up)
        .map(|i| i as f32 / c.sample_rate as f32)
        .collect::<Vec<_>>();
    let radians = g.add(
        &g.mul(&f, &g.constant(&offsets, &[1, 1, up], Precision::Fp32)),
        &phase,
    );
    let sine = g.scale(
        &g.unary(
            Op::Sin,
            &g.scale(
                &g.reshape(&radians, &[1, 1, frames * up]),
                std::f32::consts::TAU,
            ),
        ),
        0.1,
    );
    let uv = g.cast(
        &g.binary(
            Op::Greater,
            &g.reshape(pitchf, &[1, 1, frames]),
            &g.scalar(0., Precision::Fp32),
        ),
        DType::F32,
    );
    let indices = (0..frames * up)
        .map(|i| (i / up) as i32)
        .collect::<Vec<_>>();
    let uv = g.gather(&uv, &g.integers(&indices, &[frames * up]), 2);
    let amplitude = g.add(
        &g.scale(&uv, 0.003),
        &g.scale(&g.sub(&g.scalar(1., Precision::Fp32), &uv), 0.1 / 3.),
    );
    let source = g.add(&g.mul(&sine, &uv), &g.mul(noise, &amplitude));
    let w = l.parameter("dec.m_source.l_linear.weight", Precision::Fp32)?;
    let b = l.parameter("dec.m_source.l_linear.bias", Precision::Fp32)?;
    Ok(g.unary(
        Op::Tanh,
        &g.add(
            &g.mul(&source, &g.reshape(&w, &[1, 1, 1])),
            &g.reshape(&b, &[1, 1, 1]),
        ),
    ))
}
fn decoder(
    l: &Layers<'_>,
    z: &Tensor,
    speaker: &Tensor,
    pitchf: &Tensor,
    noise: &Tensor,
    c: &RvcSynthConfig,
    p: &PrecisionConfig,
    debug: bool,
) -> Result<Tensor, GraphError> {
    let g = l.g;
    let pre = l.conv1(
        z,
        "dec.conv_pre",
        Precision::Fp32,
        1,
        1,
        None,
        false,
        true,
        1.,
    )?;
    let cond = l.conv1(
        speaker,
        "dec.cond",
        Precision::Fp32,
        1,
        1,
        None,
        false,
        true,
        1.,
    )?;
    let mut x = g.add(&pre, &cond);
    if p.synth_decoder == Precision::Fp16 {
        x = g.scale(&x, p.synth_input_scale);
    }
    let harmonic = if c.uses_f0 {
        Some(source(l, pitchf, noise, c)?)
    } else {
        None
    };
    let mut harmonic_low = None;
    if let Some(src) = &harmonic {
        if p.synth_noise_fp16_stage_mask != 0 {
            harmonic_low = Some(g.cast(src, DType::F16));
        }
    }
    for i in 0..c.upsample_rates.len() {
        let precision = if (p.synth_fp32_stage_mask >> i) & 1 != 0 {
            Precision::Fp32
        } else {
            p.synth_decoder
        };
        x = g.cast(&x, precision.into());
        let up = c.upsample_rates[i] as usize;
        let kernel = c.upsample_kernels[i] as usize;
        let scale = if i == 0 && p.synth_decoder == Precision::Fp16 {
            1. / p.synth_input_scale
        } else {
            1.
        };
        x = l.conv1(
            &g.leaky(&x, 0.1),
            &format!("dec.ups.{i}"),
            precision,
            up,
            1,
            Some((kernel - up) / 2),
            true,
            true,
            scale,
        )?;
        if let Some(src) = &harmonic {
            let tail = c.upsample_rates[i + 1..].iter().product::<u64>() as usize;
            let noise_precision =
                if precision == Precision::Fp16 && (p.synth_noise_fp16_stage_mask >> i) & 1 != 0 {
                    Precision::Fp16
                } else {
                    Precision::Fp32
                };
            let src = if noise_precision == Precision::Fp16 {
                harmonic_low.as_ref().unwrap()
            } else {
                src
            };
            let y = l.conv1(
                src,
                &format!("dec.noise_convs.{i}"),
                noise_precision,
                tail,
                1,
                Some(if i + 1 == c.upsample_rates.len() {
                    0
                } else {
                    tail / 2
                }),
                false,
                true,
                1.,
            )?;
            x = g.cast(
                &g.add(&g.cast(&x, noise_precision.into()), &y),
                precision.into(),
            );
        }
        let mut sum = None;
        for j in 0..c.resblock_kernels.len() {
            let prefix = format!("dec.resblocks.{}", i * c.resblock_kernels.len() + j);
            let mut y = x.clone();
            for (k, dilation) in c.resblock_dilations[j].iter().enumerate() {
                let suffix = if c.resblock_type == 1 {
                    "convs1"
                } else {
                    "convs"
                };
                let mut a = l.conv1(
                    &g.leaky(&y, 0.1),
                    &format!("{prefix}.{suffix}.{k}"),
                    precision,
                    1,
                    *dilation as usize,
                    None,
                    false,
                    true,
                    1.,
                )?;
                if c.resblock_type == 1 {
                    a = l.conv1(
                        &g.leaky(&a, 0.1),
                        &format!("{prefix}.convs2.{k}"),
                        precision,
                        1,
                        1,
                        None,
                        false,
                        true,
                        1.,
                    )?;
                }
                y = g.add(&y, &a);
            }
            sum = Some(if let Some(s) = sum { g.add(&s, &y) } else { y });
        }
        x = g.scale(&sum.unwrap(), 1. / c.resblock_kernels.len() as f32);
        if debug {
            g.mark(&format!("decoder_res_{i}"), &g.cast(&x, DType::F32));
        }
    }
    let y = l.conv1(
        &g.leaky(&g.cast(&x, p.synth_post.into()), 0.01),
        "dec.conv_post",
        p.synth_post,
        1,
        1,
        None,
        false,
        false,
        1.,
    )?;
    Ok(g.unary(Op::Tanh, &g.cast(&y, DType::F32)))
}
pub fn synth(
    g: &GraphBuilder,
    package: &ModelPackage,
    feats: &Tensor,
    pitch: &Tensor,
    pitchf: &Tensor,
    length: &Tensor,
    sid: &Tensor,
    rnd: &Tensor,
    noise: &Tensor,
    skip: usize,
    p: &PrecisionConfig,
    debug: bool,
) -> Result<Tensor, GraphError> {
    let c = package.validate_rvc_synth()?;
    let l = Layers { g, package };
    let frames = feats.shape[1];
    let (z, mask) = encoder(&l, feats, pitch, length, rnd, &c)?;
    if debug {
        g.mark("encoder_z", &z);
    }
    let speaker = g.reshape(
        &g.gather(&l.parameter("emb_g.weight", Precision::Fp32)?, sid, 0),
        &[1, c.speaker_channels as usize, 1],
    );
    let flow_head = skip.saturating_sub(24);
    let count = frames - flow_head;
    let z = flow(
        &l,
        &g.slice(&z, 2, flow_head, count, 1),
        &g.slice(&mask, 2, flow_head, count, 1),
        &speaker,
        &c,
    )?;
    if debug {
        g.mark("flow_z", &z);
    }
    let z = g.slice(&z, 2, skip - flow_head, frames - skip, 1);
    let pitchf = g.slice(pitchf, 1, skip, frames - skip, 1);
    let up = c.upsample_rates.iter().product::<u64>() as usize;
    let noise = g.slice(noise, 2, skip * up, (frames - skip) * up, 1);
    decoder(&l, &z, &speaker, &pitchf, &noise, &c, p, debug)
}
