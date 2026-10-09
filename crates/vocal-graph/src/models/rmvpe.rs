use super::*;
use crate::PrecisionConfig;

fn fft_mel(g: &GraphBuilder, l: &Layers<'_>, wave: &Tensor) -> Result<Tensor, GraphError> {
    let samples = wave.elements();
    let frames = samples / 160 + 1;
    let flat = g.reshape(wave, &[samples]);
    let padded = g.concat(
        &[
            g.slice(&flat, 0, 512, 512, -1),
            flat.clone(),
            g.slice(&flat, 0, samples - 2, 512, -1),
        ],
        0,
    );
    let indices = (0..frames)
        .flat_map(|f| (0..1024).map(move |i| (f * 160 + i) as i32))
        .collect::<Vec<_>>();
    let windows = g.gather(&padded, &g.integers(&indices, &[frames, 1024]), 0);
    let hann = (0..1024)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / 1024.).cos())
        .collect::<Vec<_>>();
    let bitreverse = (0..1024u32)
        .map(|i| i.reverse_bits() >> 22)
        .map(|i| i as i32)
        .collect::<Vec<_>>();
    let mut real = g.gather(
        &g.mul(&windows, &g.constant(&hann, &[1, 1024], Precision::Fp32)),
        &g.integers(&bitreverse, &[1024]),
        1,
    );
    let mut imag = g.scale(&real, 0.);
    for stage in 1..=10 {
        let width = 1usize << stage;
        let half = width / 2;
        let groups = 1024 / width;
        let r = g.reshape(&real, &[frames, groups, width]);
        let im = g.reshape(&imag, &[frames, groups, width]);
        let ar = g.slice(&r, 2, 0, half, 1);
        let br = g.slice(&r, 2, half, half, 1);
        let ai = g.slice(&im, 2, 0, half, 1);
        let bi = g.slice(&im, 2, half, half, 1);
        let phase = (0..half)
            .map(|i| std::f32::consts::TAU * i as f32 / width as f32)
            .collect::<Vec<_>>();
        let wr = g.constant(
            &phase.iter().map(|v| v.cos()).collect::<Vec<_>>(),
            &[1, 1, half],
            Precision::Fp32,
        );
        let wi = g.constant(
            &phase.iter().map(|v| -v.sin()).collect::<Vec<_>>(),
            &[1, 1, half],
            Precision::Fp32,
        );
        let tr = g.sub(&g.mul(&br, &wr), &g.mul(&bi, &wi));
        let ti = g.add(&g.mul(&br, &wi), &g.mul(&bi, &wr));
        real = g.reshape(
            &g.concat(&[g.add(&ar, &tr), g.sub(&ar, &tr)], 2),
            &[frames, 1024],
        );
        imag = g.reshape(
            &g.concat(&[g.add(&ai, &ti), g.sub(&ai, &ti)], 2),
            &[frames, 1024],
        );
    }
    real = g.slice(&real, 1, 0, 513, 1);
    imag = g.slice(&imag, 1, 0, 513, 1);
    let magnitude = g.transpose(
        &g.unary(Op::Sqrt, &g.add(&g.mul(&real, &real), &g.mul(&imag, &imag))),
        &[1, 0],
    );
    let mel = g.matmul(
        &l.parameter("rmvpe.mel_basis", Precision::Fp32)?,
        &magnitude,
        false,
    );
    Ok(g.reshape(
        &g.unary(
            Op::Log,
            &g.binary(Op::Max, &mel, &g.scalar(1e-5, Precision::Fp32)),
        ),
        &[1, 128, frames],
    ))
}
fn batchnorm(l: &Layers<'_>, x: &Tensor, node: usize, p: Precision) -> Result<Tensor, GraphError> {
    let prefix = format!("rmvpe.node.{node}");
    let g = l.g;
    let gamma = crate::graph::weight_f32(l.package, &format!("{prefix}.weight"))?;
    let beta = crate::graph::weight_f32(l.package, &format!("{prefix}.bias"))?;
    let mean = crate::graph::weight_f32(l.package, &format!("{prefix}.mean"))?;
    let var = crate::graph::weight_f32(l.package, &format!("{prefix}.var"))?;
    let scale = gamma
        .iter()
        .zip(var)
        .map(|(g, v)| g / (v + 1e-5).sqrt())
        .collect::<Vec<_>>();
    let shift = beta
        .iter()
        .zip(mean)
        .zip(&scale)
        .map(|((b, m), s)| b - m * s)
        .collect::<Vec<_>>();
    Ok(g.add(
        &g.mul(
            &g.cast(x, p.into()),
            &g.constant(&scale, &[1, scale.len(), 1, 1], p),
        ),
        &g.constant(&shift, &[1, shift.len(), 1, 1], p),
    ))
}
fn residual(
    l: &Layers<'_>,
    x: &Tensor,
    a: usize,
    b: usize,
    shortcut: i32,
    p: Precision,
) -> Result<Tensor, GraphError> {
    let g = l.g;
    let y = g.unary(
        Op::Relu,
        &l.conv2(x, &format!("rmvpe.node.{a}"), p, 1, 1, false, 0, true)?,
    );
    let y = g.unary(
        Op::Relu,
        &l.conv2(&y, &format!("rmvpe.node.{b}"), p, 1, 1, false, 0, true)?,
    );
    let skip = if shortcut >= 0 {
        l.conv2(
            x,
            &format!("rmvpe.node.{shortcut}"),
            p,
            1,
            0,
            false,
            0,
            true,
        )?
    } else {
        g.cast(x, p.into())
    };
    Ok(g.add(&y, &skip))
}
pub fn rmvpe(
    g: &GraphBuilder,
    package: &ModelPackage,
    wave: &Tensor,
    p: &PrecisionConfig,
    threshold: &Tensor,
    debug: bool,
) -> Result<Tensor, GraphError> {
    package.validate_acoustic(Some(vocal_model::AcousticKind::Rmvpe))?;
    let l = Layers { g, package };
    let mel = fft_mel(g, &l, wave)?;
    rmvpe_from_mel(g, package, &mel, p, threshold, debug)
}
pub fn rmvpe_from_mel(
    g: &GraphBuilder,
    package: &ModelPackage,
    mel: &Tensor,
    p: &PrecisionConfig,
    threshold: &Tensor,
    debug: bool,
) -> Result<Tensor, GraphError> {
    package.validate_acoustic(Some(vocal_model::AcousticKind::Rmvpe))?;
    let l = Layers { g, package };
    let mel = mel.clone();
    let frames = mel.shape[2];
    if debug {
        g.mark("rmvpe_mel", &mel);
    }
    let padded = frames.div_ceil(32) * 32;
    let pad = padded - frames;
    let mel = if pad > 0 {
        g.concat(&[mel.clone(), g.slice(&mel, 2, frames - 2, pad, -1)], 2)
    } else {
        mel
    };
    let mut x = g.cast(
        &g.reshape(&g.transpose(&mel, &[0, 2, 1]), &[1, 1, padded, 128]),
        p.rmvpe_encoder.into(),
    );
    x = batchnorm(&l, &x, 27, p.rmvpe_encoder)?;
    let encoder = [
        [[28, 30, 32], [34, 36, -1], [39, 41, -1], [44, 46, -1]],
        [[50, 52, 54], [56, 58, -1], [61, 63, -1], [66, 68, -1]],
        [[72, 74, 76], [78, 80, -1], [83, 85, -1], [88, 90, -1]],
        [[94, 96, 98], [100, 102, -1], [105, 107, -1], [110, 112, -1]],
        [
            [116, 118, 120],
            [122, 124, -1],
            [127, 129, -1],
            [132, 134, -1],
        ],
    ];
    let mut skips = Vec::new();
    for (i, blocks) in encoder.into_iter().enumerate() {
        for [a, b, shortcut] in blocks {
            x = residual(&l, &x, a as usize, b as usize, shortcut, p.rmvpe_encoder)?;
        }
        skips.push(x.clone());
        if debug {
            g.mark(&format!("rmvpe_encoder_{i}"), &g.cast(&x, DType::F32));
        }
        let shape = vec![1, x.shape[1], x.shape[2] / 2, x.shape[3] / 2];
        x = g.node(
            Op::Pool,
            &[&x],
            shape,
            x.dtype,
            vec![],
            [0.; 4],
            vec![],
            String::new(),
        );
    }
    x = residual(&l, &x, 138, 140, 142, p.rmvpe_encoder)?;
    for a in [
        144, 149, 154, 159, 164, 169, 174, 179, 184, 189, 194, 199, 204, 209, 214,
    ] {
        x = residual(&l, &x, a, a + 2, -1, p.rmvpe_encoder)?;
    }
    if debug {
        g.mark("rmvpe_bottleneck", &g.cast(&x, DType::F32));
    }
    let decoder = [
        [219, 223, 225, 227, 229, 231, 234, 236, 239, 241],
        [244, 248, 250, 252, 254, 256, 259, 261, 264, 266],
        [269, 273, 275, 277, 279, 281, 284, 286, 289, 291],
        [294, 298, 300, 302, 304, 306, 309, 311, 314, 316],
        [319, 323, 325, 327, 329, 331, 334, 336, 339, 341],
    ];
    for (i, nodes) in decoder.into_iter().enumerate() {
        x = l.conv2(
            &x,
            &format!("rmvpe.node.{}", nodes[0]),
            p.rmvpe_decoder,
            2,
            1,
            true,
            1,
            false,
        )?;
        x = g.unary(
            Op::Relu,
            &batchnorm(&l, &x, [220, 245, 270, 295, 320][i], p.rmvpe_decoder)?,
        );
        x = g.concat(&[x, g.cast(&skips[4 - i], p.rmvpe_decoder.into())], 1);
        x = residual(&l, &x, nodes[1], nodes[2], nodes[3] as i32, p.rmvpe_decoder)?;
        for b in 0..3 {
            x = residual(
                &l,
                &x,
                nodes[4 + b * 2],
                nodes[5 + b * 2],
                -1,
                p.rmvpe_decoder,
            )?;
        }
        if debug {
            g.mark(&format!("rmvpe_decoder_{i}"), &g.cast(&x, DType::F32));
        }
    }
    x = g.cast(&x, DType::F32);
    if debug {
        g.mark("rmvpe_unet", &x);
    }
    x = l.conv2(&x, "rmvpe.node.344", Precision::Fp32, 1, 1, false, 0, true)?;
    let sequence = g.reshape(&g.transpose(&x, &[2, 0, 1, 3]), &[1, padded, 1, 384]);
    let w = g.reshape(
        &l.parameter("rmvpe.node.356.W", Precision::Fp32)?,
        &[1, 2, 768, 384],
    );
    let r = g.reshape(
        &l.parameter("rmvpe.node.356.R", Precision::Fp32)?,
        &[1, 2, 768, 256],
    );
    let b = g.reshape(
        &l.parameter("rmvpe.node.356.B", Precision::Fp32)?,
        &[1, 1, 2, 1536],
    );
    let gru = g.node(
        Op::Gru,
        &[&sequence, &w, &r, &b],
        vec![padded, 2, 1, 256],
        DType::F32,
        vec![],
        [0.; 4],
        vec![],
        String::new(),
    );
    let both = g.reshape(&gru, &[1, padded, 512]);
    if debug {
        g.mark("rmvpe_gru", &both);
    }
    let logits = g.add(
        &g.matmul(
            &both,
            &l.parameter("rmvpe.node.360.weight", Precision::Fp32)?,
            false,
        ),
        &l.parameter("rmvpe.node.361.bias", Precision::Fp32)?,
    );
    if debug {
        g.mark("rmvpe_logits", &logits);
    }
    let salience = g.unary(Op::Sigmoid, &g.slice(&logits, 1, 0, frames, 1));
    let index = g.cast(&g.topk(&salience, 1, true, true), DType::F32);
    let offsets = g.constant(
        &[-4., -3., -2., -1., 0., 1., 2., 3., 4.],
        &[1, 1, 9],
        Precision::Fp32,
    );
    let centres = g.add(&g.add(&index, &g.scalar(4., Precision::Fp32)), &offsets);
    let local_indices = g.cast(&centres, DType::I32);
    let zero = g.constant(&vec![0.; frames * 4], &[1, frames, 4], Precision::Fp32);
    let padded_salience = g.concat(&[zero.clone(), salience.clone(), zero], 2);
    let local = g.gather_elements(&padded_salience, &local_indices, 2);
    let cents = (0..368)
        .map(|i| {
            if (4..364).contains(&i) {
                1997.3794 + 20. * (i - 4) as f32
            } else {
                0.
            }
        })
        .collect::<Vec<_>>();
    let cents = g.constant(&cents, &[368], Precision::Fp32);
    let local_cents = g.gather(&cents, &local_indices, 0);
    let weighted = g.reduce(Op::Sum, &g.mul(&local, &local_cents), &[2]);
    let sum = g.reduce(Op::Sum, &local, &[2]);
    let average = g.div(
        &weighted,
        &g.binary(Op::Max, &sum, &g.scalar(1e-12, Precision::Fp32)),
    );
    let hz = g.scale(
        &g.unary(Op::Exp, &g.scale(&average, std::f32::consts::LN_2 / 1200.)),
        10.,
    );
    let confident = g.binary(
        Op::Greater,
        &g.reduce(Op::ReduceMax, &salience, &[2]),
        threshold,
    );
    Ok(g.reshape(
        &g.select(&confident, &hz, &g.scalar(0., Precision::Fp32)),
        &[1, frames],
    ))
}
