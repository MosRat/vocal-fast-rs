use super::*;
use crate::PrecisionConfig;

pub fn hubert(
    g: &GraphBuilder,
    package: &ModelPackage,
    waveform: &Tensor,
    p: &PrecisionConfig,
) -> Result<Tensor, GraphError> {
    hubert_for_rvc(g, package, waveform, p, vocal_model::RvcVersion::V2)
}

/// Select the RVC feature layer before interpolation and index retrieval.
pub fn hubert_for_rvc(
    g: &GraphBuilder,
    package: &ModelPackage,
    waveform: &Tensor,
    p: &PrecisionConfig,
    version: vocal_model::RvcVersion,
) -> Result<Tensor, GraphError> {
    package.validate_hubert_for_rvc(version)?;
    let l = Layers { g, package };
    let features = p.hubert_features;
    let mut x = g.reshape(
        &g.cast(waveform, features.into()),
        &[1, 1, waveform.elements()],
    );
    for (i, stride) in [5, 2, 2, 2, 2, 2, 2].into_iter().enumerate() {
        x = l.conv1(
            &x,
            &format!("hubert.model.feature_extractor.conv_layers.{i}.conv"),
            features,
            stride,
            1,
            Some(0),
            false,
            false,
            1.,
        )?;
        if i == 0 {
            x = g.cast(&x, DType::F32);
            let mean = g.reduce(Op::Mean, &x, &[2]);
            let center = g.sub(&x, &mean);
            let variance = g.reduce(Op::Mean, &g.mul(&center, &center), &[2]);
            x = g.div(
                &center,
                &g.unary(
                    Op::Sqrt,
                    &g.add(&variance, &g.scalar(1e-5, Precision::Fp32)),
                ),
            );
            for (scale, bias) in [
                ("fef951e6c76ad6a0", "e5a00aa9991ac8a5"),
                ("d8b9ba9bce6ed02c", "685ea624c050bb77"),
            ] {
                let a = g.reshape(
                    &l.parameter(&format!("hubert.const.{scale}"), Precision::Fp32)?,
                    &[1, 512, 1],
                );
                let b = g.reshape(
                    &l.parameter(&format!("hubert.const.{bias}"), Precision::Fp32)?,
                    &[1, 512, 1],
                );
                x = g.add(&g.mul(&x, &a), &b);
            }
            x = g.cast(&x, features.into());
        }
        x = g.unary(Op::Gelu, &x);
    }
    x = g.cast(&g.transpose(&x, &[0, 2, 1]), p.hubert.into());
    x = l.norm(
        &x,
        "hubert.model.feature_projection.layer_norm",
        2,
        1e-5,
        p.hubert,
        "weight",
        "bias",
    )?;
    x = l.linear(&x, "hubert.model.feature_projection.projection", p.hubert)?;
    let frames = x.shape[1];
    let tx = g.transpose(&x, &[0, 2, 1]);
    let w = l.parameter("hubert.model.encoder.pos_conv_embed.conv.weight", p.hubert)?;
    let w = g.reshape(&w, &[768, 48, 1, 128]);
    let b = g.reshape(
        &l.parameter("hubert.model.encoder.pos_conv_embed.conv.bias", p.hubert)?,
        &[1, 768, 1, 1],
    );
    let pos = g.conv(
        &g.reshape(&tx, &[1, 768, 1, frames]),
        &w,
        Some(&b),
        [1, 1],
        [1, 1],
        [0, 64],
        16,
        false,
        [0, 0],
    );
    let pos = g.slice(&pos, 3, 0, frames, 1);
    let pos = g.transpose(&g.reshape(&pos, &[1, 768, frames]), &[0, 2, 1]);
    x = l.norm(
        &g.add(&x, &g.unary(Op::Gelu, &pos)),
        "hubert.model.encoder.layer_norm",
        2,
        1e-5,
        p.hubert,
        "weight",
        "bias",
    )?;
    for i in 0..version.hubert_output_layer() {
        let prefix = format!("hubert.model.encoder.layers.{i}");
        let mut qkv = Vec::new();
        for name in ["q", "k", "v"] {
            let y = l.linear(&x, &format!("{prefix}.attention.{name}_proj"), p.hubert)?;
            qkv.push(g.transpose(&g.reshape(&y, &[1, frames, 12, 64]), &[0, 2, 1, 3]));
        }
        let scores = g.scale(&g.matmul(&qkv[0], &qkv[1], true), 1. / 8.);
        let context = g.matmul(&g.softmax(&scores, 3), &qkv[2], false);
        let context = g.reshape(&g.transpose(&context, &[0, 2, 1, 3]), &[1, frames, 768]);
        let attention = l.linear(&context, &format!("{prefix}.attention.out_proj"), p.hubert)?;
        let residual = l.norm(
            &g.add(&x, &attention),
            &format!("{prefix}.layer_norm"),
            2,
            1e-5,
            p.hubert,
            "weight",
            "bias",
        )?;
        let ff = g.unary(
            Op::Gelu,
            &l.linear(
                &residual,
                &format!("{prefix}.feed_forward.intermediate_dense"),
                p.hubert,
            )?,
        );
        let ff = l.linear(
            &ff,
            &format!("{prefix}.feed_forward.output_dense"),
            p.hubert,
        )?;
        x = l.norm(
            &g.add(&residual, &ff),
            &format!("{prefix}.final_layer_norm"),
            2,
            1e-5,
            p.hubert,
            "weight",
            "bias",
        )?;
    }
    if version == vocal_model::RvcVersion::V1 {
        x = l.linear(&x, "hubert.model.final_proj", p.hubert)?;
    }
    Ok(g.cast(&x, DType::F32))
}
