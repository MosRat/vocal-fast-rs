use vocal_cpu::{CpuConfig, CpuGraph};
use vocal_graph::{
    Precision,
    graph::{DType, GraphBuilder, Op},
};
fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn build(fuse: bool, avx2: bool) -> CpuGraph {
    let g = GraphBuilder::default();
    let x = g.input("x", &[1, 3, 17], DType::F32);
    let y = g.input("y", &x.shape, DType::F32);
    g.mark("activation", &g.leaky(&g.add(&x, &y), 0.1));
    let mean = g.reduce(Op::Mean, &x, &[1]);
    let center = g.sub(&x, &mean);
    let var = g.reduce(Op::Mean, &g.mul(&center, &center), &[1]);
    let gamma = g.constant(&[1., 1.5, 2.], &[1, 3, 1], Precision::Fp32);
    let beta = g.constant(&[0.1, 0.2, 0.3], &[1, 3, 1], Precision::Fp32);
    let normalized = g.add(
        &g.mul(
            &g.div(
                &center,
                &g.unary(Op::Sqrt, &g.add(&var, &g.scalar(1e-5, Precision::Fp32))),
            ),
            &gamma,
        ),
        &beta,
    );
    g.mark("norm", &normalized);
    CpuGraph::compile(
        g,
        &CpuConfig {
            threads: 6,
            fuse,
            avx2,
            ..CpuConfig::default()
        },
    )
    .unwrap()
}
#[test]
fn fused_norm_and_unaligned_avx2_match_unfused() {
    let x = bytes(&(0..51).map(|i| (i as f32 - 20.) * 0.1).collect::<Vec<_>>());
    let y = bytes(&(0..51).map(|i| (i % 7) as f32 * 0.02).collect::<Vec<_>>());
    let reference = build(false, false).run(&[("x", &x), ("y", &y)]).unwrap();
    for simd in [false, true] {
        let actual = build(true, simd).run(&[("x", &x), ("y", &y)]).unwrap();
        for name in ["activation", "norm"] {
            for (a, b) in actual
                .f32(name)
                .unwrap()
                .iter()
                .zip(reference.f32(name).unwrap())
            {
                assert!((a - b).abs() < 2e-6, "{name}: {a} vs {b}");
            }
        }
    }
}

// Temporal statistics with affine parameters on a different axis, as in the
// first HuBERT convolution. Include NHWC views, partial AVX2 tiles, constants,
// almost constant rows and a marked intermediate that must survive pruning.
#[test]
fn channel_temporal_norm_preserves_layout_rounding_and_marked_consumers() {
    fn graph(fused: bool, avx2: bool, channels_last: bool) -> CpuGraph {
        let g = GraphBuilder::default();
        let input = g.input("x", &[1, 67, 1, 137], DType::F32);
        let x = if channels_last {
            let w = g.constant(&vec![1.; 67], &[67, 1, 1, 1], Precision::Fp32);
            let conv = g.conv(&input, &w, None, [1, 1], [1, 1], [0, 0], 67, false, [0, 0]);
            g.reshape(&conv, &[1, 67, 137])
        } else {
            g.reshape(&input, &[1, 67, 137])
        };
        let mean = g.reduce(Op::Mean, &x, &[2]);
        let center = g.sub(&x, &mean);
        let var = g.reduce(Op::Mean, &g.mul(&center, &center), &[2]);
        let scale = g.constant(
            &(0..67).map(|c| c as f32 * 0.03 - 0.8).collect::<Vec<_>>(),
            &[1, 67, 1],
            Precision::Fp32,
        );
        let bias = g.constant(&vec![0.02; 67], &[1, 67, 1], Precision::Fp32);
        let normalized = g.add(
            &g.mul(
                &g.div(
                    &center,
                    &g.unary(Op::Sqrt, &g.add(&var, &g.scalar(1e-5, Precision::Fp32))),
                ),
                &scale,
            ),
            &bias,
        );
        g.mark("center", &center);
        g.mark("norm", &normalized);
        CpuGraph::compile(
            g,
            &CpuConfig {
                threads: 3,
                fuse_channel_norm: fused,
                avx2,
                channels_last,
                profile: true,
                ..Default::default()
            },
        )
        .unwrap()
    }
    for channels_last in [false, true] {
        let mut baseline = graph(false, false, channels_last);
        for avx2 in [false, true] {
            let mut candidate = graph(true, avx2, channels_last);
            for gain in [1., 1e-4, 0.] {
                let x = bytes(
                    &(0..67 * 137)
                        .map(|i| gain * ((i % 17) as f32 - 8.) + (i / 137) as f32 * 0.125)
                        .collect::<Vec<_>>(),
                );
                let a = baseline.run(&[("x", &x)]).unwrap();
                let b = candidate.run(&[("x", &x)]).unwrap();
                for name in ["center", "norm"] {
                    let left = a.f32(name).unwrap();
                    let right = b.f32(name).unwrap();
                    for (i, (a, b)) in left.iter().zip(&right).enumerate() {
                        assert!(
                            (a - b).abs() <= 2e-6,
                            "{name}[{i}] {a} vs {b} simd={avx2} nhwc={channels_last} gain={gain}"
                        );
                    }
                }
                assert!(
                    candidate
                        .timings()
                        .iter()
                        .any(|n| n.name == "channel_temporal_norm")
                );
            }
        }
    }
}
#[test]
fn binding_errors_do_not_poison_graph() {
    let mut graph = build(true, true);
    let a = bytes(&[1.; 51]);
    assert!(graph.run(&[("x", &a)]).is_err());
    assert!(graph.run(&[("x", &a), ("x", &a)]).is_err());
    assert!(graph.run(&[("x", &a), ("y", &a[..4])]).is_err());
    assert!(graph.run(&[("x", &a), ("y", &a)]).is_ok());
}
#[test]
fn index_and_exclusive_phase_semantics() {
    let g = GraphBuilder::default();
    let x = g.constant(&[0.1, 0.2, 0.3, 0.4], &[1, 4], Precision::Fp32);
    let phase = g.node(
        Op::Cumsum,
        &[&x],
        x.shape.clone(),
        DType::F32,
        vec![1, 1],
        [0.; 4],
        vec![],
        String::new(),
    );
    g.mark("phase", &phase);
    g.mark("reverse", &g.slice(&x, 1, 3, 4, -1));
    let top = g.topk(&x, 2, true, true);
    g.mark("index", &g.cast(&top, DType::I32));
    let mut graph = CpuGraph::compile(
        g,
        &CpuConfig {
            threads: 6,
            ..CpuConfig::default()
        },
    )
    .unwrap();
    let out = graph.run(&[]).unwrap();
    assert_eq!(out.f32("reverse").unwrap(), [0.4, 0.3, 0.2, 0.1]);
    assert_eq!(out.f32("phase").unwrap(), [0., 0.1, 0.3, 0.6]);
    let index = out.tensors["index"]
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(index, [3, 2]);
}

#[test]
fn fused_attention_preserves_fp32_context() {
    fn graph(fuse: bool) -> CpuGraph {
        let g = GraphBuilder::default();
        let shape = [1, 2, 17, 8];
        let q = g.input("q", &shape, DType::F32);
        let k = g.input("k", &shape, DType::F32);
        let v = g.input("v", &shape, DType::F32);
        let logits = g.scale(&g.matmul(&q, &k, true), 1. / 8f32.sqrt());
        g.mark("context", &g.matmul(&g.softmax(&logits, 3), &v, false));
        CpuGraph::compile(
            g,
            &CpuConfig {
                threads: 6,
                fuse,
                ..CpuConfig::default()
            },
        )
        .unwrap()
    }
    let data = bytes(&(0..272).map(|i| (i as f32 * 0.1).sin()).collect::<Vec<_>>());
    let bindings = [
        ("q", data.as_slice()),
        ("k", data.as_slice()),
        ("v", data.as_slice()),
    ];
    let a = graph(false).run(&bindings).unwrap().f32("context").unwrap();
    let b = graph(true).run(&bindings).unwrap().f32("context").unwrap();
    let max = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max < 1e-6,
        "attention max abs: {max}, unfused={:?}, fused={:?}",
        &a[..8],
        &b[..8]
    );
}

#[test]
fn malformed_native_descriptor_is_rejected_before_ffi() {
    let g = GraphBuilder::default();
    let x = g.input("x", &[1, 8], DType::F32);
    let y = g.softmax(&x, 1);
    g.nodes.borrow_mut()[y.id as usize].attrs.clear();
    g.mark("y", &y);
    assert!(
        CpuGraph::compile(
            g,
            &CpuConfig {
                threads: 6,
                ..CpuConfig::default()
            }
        )
        .is_err()
    );
}

#[test]
fn input_storage_accepts_unaligned_byte_slices_and_is_reusable() {
    let g = GraphBuilder::default();
    let x = g.input("x", &[3], DType::F32);
    g.mark("y", &g.unary(Op::Relu, &x));
    let mut graph = CpuGraph::compile(
        g,
        &CpuConfig {
            threads: 6,
            ..CpuConfig::default()
        },
    )
    .unwrap();
    let mut data = vec![0];
    data.extend(bytes(&[-1., 2., 3.]));
    let first = graph.run(&[("x", &data[1..])]).unwrap().f32("y").unwrap();
    assert_eq!(first, [0., 2., 3.]);
    drop(data);
    assert_eq!(
        graph
            .run(&[("x", &bytes(&[4., 5., 6.]))])
            .unwrap()
            .f32("y")
            .unwrap(),
        [4., 5., 6.]
    );
    assert_eq!(first, [0., 2., 3.]);
}

#[test]
fn linear_gelu_post_op_preserves_bias_and_marked_pre_activation() {
    fn graph(fused: bool, marked: bool) -> CpuGraph {
        let g = GraphBuilder::default();
        let x = g.input("x", &[1, 17, 8], DType::F32);
        let weights = (0..88).map(|i| (i as f32 * 0.37).sin()).collect::<Vec<_>>();
        let w = g.constant(&weights, &[11, 8], Precision::Fp32);
        let b = g.constant(
            &[-2., -1., -0.5, -0.1, 0., 0.1, 0.5, 1., 2., 3., 4.],
            &[11],
            Precision::Fp32,
        );
        let linear = g.add(&g.matmul(&x, &w, true), &b);
        if marked {
            g.mark("linear", &linear);
        }
        g.mark("gelu", &g.unary(Op::Gelu, &linear));
        CpuGraph::compile(
            g,
            &CpuConfig {
                threads: 2,
                fuse_linear_gelu: fused,
                ..Default::default()
            },
        )
        .unwrap()
    }
    for marked in [false, true] {
        let mut baseline = graph(false, marked);
        let mut fused = graph(true, marked);
        for amplitude in [0.001, 1., 100.] {
            let input = bytes(
                &(0..136)
                    .map(|i| (i as f32 * 0.19).cos() * amplitude)
                    .collect::<Vec<_>>(),
            );
            let a = baseline.run(&[("x", &input)]).unwrap();
            let b = fused.run(&[("x", &input)]).unwrap();
            let expected = a.f32("gelu").unwrap();
            let actual = b.f32("gelu").unwrap();
            assert!(
                expected
                    .iter()
                    .zip(actual)
                    .all(|(a, b)| (a - b).abs() < 4e-6 * (1. + a.abs()))
            );
            if marked {
                assert_eq!(a.f32("linear").unwrap(), b.f32("linear").unwrap());
                assert!(b.f32("linear").unwrap().iter().any(|v| *v < -0.1));
            }
        }
    }
}
