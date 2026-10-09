use vocal_cpu::{CpuConfig, CpuGraph, CpuLinearMode};
use vocal_graph::{
    Precision,
    graph::{DType, GraphBuilder, Op},
};

// Independently integrate the Gaussian in FP64 rather than reuse a native GELU.
fn gelu_reference(x: f64) -> f64 {
    let z = x / 2f64.sqrt();
    let step = z / 128.;
    let mut sum = 1. + (-z * z).exp();
    for i in 1..128 {
        let t = i as f64 * step;
        sum += (if i % 2 == 0 { 2. } else { 4. }) * (-t * t).exp();
    }
    let erf = 2. / std::f64::consts::PI.sqrt() * step / 3. * sum;
    0.5 * x * (1. + erf)
}

#[test]
fn packed_projections_preserve_bias_weight_orientation_and_gelu() {
    for mode in [CpuLinearMode::InnerProduct, CpuLinearMode::Pointwise1d] {
        for transpose in [false, true] {
            for fused_gelu in [false, true] {
                let g = GraphBuilder::default();
                let x = g.input("x", &[2, 17, 13], DType::F32);
                let weights = (0..13 * 19)
                    .map(|i| (i as f32 * 0.07).sin() * 0.02)
                    .collect::<Vec<_>>();
                let shape = if transpose { [19, 13] } else { [13, 19] };
                let w = g.constant(&weights, &shape, Precision::Fp32);
                let bias = (0..19).map(|i| i as f32 * 0.03 - 0.2).collect::<Vec<_>>();
                let b = g.constant(&bias, &[19], Precision::Fp32);
                let y = g.binary(Op::Add, &g.matmul(&x, &w, transpose), &b);
                if !fused_gelu {
                    g.mark("pre", &y);
                }
                g.mark("y", &g.unary(Op::Gelu, &y));
                let mut graph = CpuGraph::compile(
                    g,
                    &CpuConfig {
                        threads: 3,
                        profile: true,
                        linear_mode: mode,
                        fuse_linear_gelu: fused_gelu,
                        ..Default::default()
                    },
                )
                .unwrap();
                for gain in [1., 0.001, -0.4] {
                    let x = (0..2 * 17 * 13)
                        .map(|i| (i as f32 * 0.03).cos() * gain)
                        .collect::<Vec<_>>();
                    let data = x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
                    let output = graph.run(&[("x", &data)]).unwrap();
                    let activated = output.f32("y").unwrap();
                    let pre = (!fused_gelu).then(|| output.f32("pre").unwrap());
                    for m in 0..34 {
                        for n in 0..19 {
                            let expected = bias[n] as f64
                                + (0..13)
                                    .map(|k| {
                                        let wi = if transpose { n * 13 + k } else { k * 19 + n };
                                        x[m * 13 + k] as f64 * weights[wi] as f64
                                    })
                                    .sum::<f64>();
                            if let Some(pre) = &pre {
                                assert!((pre[m * 19 + n] as f64 - expected).abs() < 1e-6);
                            }
                            assert!(
                                (activated[m * 19 + n] as f64 - gelu_reference(expected)).abs()
                                    < 1e-6
                            );
                        }
                    }
                    assert!(graph.timings().iter().any(|n| n.implementation.starts_with(
                        if mode == CpuLinearMode::InnerProduct {
                            "inner-product:"
                        } else {
                            "pointwise-linear:"
                        }
                    )));
                    assert_eq!(
                        graph.timings().iter().any(|n| n.operator == "Gelu"),
                        !fused_gelu
                    );
                }
            }
        }
    }
}

#[test]
fn hybrid_keeps_long_wide_reductions_on_matmul() {
    for rows in [63, 64] {
        let g = GraphBuilder::default();
        let x = g.input("x", &[1, rows, 1536], DType::F32);
        let w = g.constant(&vec![1. / 1024.; 1536 * 2], &[1536, 2], Precision::Fp32);
        g.mark("y", &g.matmul(&x, &w, false));
        let mut graph = CpuGraph::compile(
            g,
            &CpuConfig {
                threads: 2,
                profile: true,
                linear_mode: CpuLinearMode::Hybrid,
                ..Default::default()
            },
        )
        .unwrap();
        let data = vec![1f32; rows * 1536]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let output = graph.run(&[("x", &data)]).unwrap().f32("y").unwrap();
        assert!(output.iter().all(|v| *v == 1.5));
        assert_eq!(
            graph
                .timings()
                .iter()
                .any(|n| n.implementation.starts_with("pointwise-linear:")),
            rows < 64
        );
    }
}

#[test]
fn packed_policy_retains_dynamic_batched_matmul() {
    for mode in [
        CpuLinearMode::InnerProduct,
        CpuLinearMode::Pointwise1d,
        CpuLinearMode::Hybrid,
    ] {
        let g = GraphBuilder::default();
        let x = g.input("x", &[2, 3, 4], DType::F32);
        let w = g.input("w", &[2, 4, 5], DType::F32);
        g.mark("y", &g.matmul(&x, &w, false));
        let mut graph = CpuGraph::compile(
            g,
            &CpuConfig {
                threads: 2,
                linear_mode: mode,
                ..Default::default()
            },
        )
        .unwrap();
        for weight in [0.5f32, -0.25] {
            let a = vec![1f32; 24]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            let b = vec![weight; 40]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            let y = graph
                .run(&[("x", &a), ("w", &b)])
                .unwrap()
                .f32("y")
                .unwrap();
            assert!(y.iter().all(|v| *v == 4. * weight));
        }
    }
}
