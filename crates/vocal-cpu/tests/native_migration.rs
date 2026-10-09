//! ABI, layout and model-specific contracts that can break when changing providers.
use vocal_cpu::{CpuConfig, CpuGraph};
use vocal_graph::{
    Precision,
    graph::{DType, GraphBuilder, Op},
};
fn bytes(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn config() -> CpuConfig {
    CpuConfig {
        threads: 3,
        ..CpuConfig::default()
    }
}

#[test]
fn loaded_runtime_is_standalone_onednn() {
    assert!(vocal_cpu::operator_version().starts_with("3."));
}

#[test]
fn conv_activation_postop_preserves_reshape_and_negative_values() {
    for fuse in [false, true] {
        let g = GraphBuilder::default();
        let x = g.input("x", &[1, 2, 1, 5], DType::F32);
        let w = g.constant(&[1., -0.5, 0.25, 2.], &[2, 2, 1, 1], Precision::Fp32);
        let y = g.conv(&x, &w, None, [1, 1], [1, 1], [0, 0], 1, false, [0, 0]);
        g.mark("y", &g.leaky(&g.reshape(&y, &[1, 2, 5]), 0.1));
        let mut cfg = config();
        cfg.fuse = fuse;
        let input = [-2., -1., 0., 1., 2., -1., 2., 0.5, -3., 1.];
        let out = CpuGraph::compile(g, &cfg)
            .unwrap()
            .run(&[("x", &bytes(&input))])
            .unwrap()
            .f32("y")
            .unwrap();
        for c in 0..2 {
            for t in 0..5 {
                let v = if c == 0 {
                    input[t] - 0.5 * input[t + 5]
                } else {
                    0.25 * input[t] + 2. * input[t + 5]
                };
                let expected = if v < 0. { v * 0.1 } else { v };
                assert_eq!(out[c * 5 + t], expected);
            }
        }
    }
}

#[test]
fn grouped_deconv_preserves_iohw_weights_and_output_padding() {
    let input = (0..20).map(|i| (i as f32 - 7.) * 0.1).collect::<Vec<_>>();
    let weights = (0..24).map(|i| (i as f32 - 12.) * 0.02).collect::<Vec<_>>();
    let mut expected = vec![0f32; 40];
    for group in 0..2 {
        for ic in 0..2 {
            for oc in 0..2 {
                for t in 0..5 {
                    for k in 0..3 {
                        let p = (t * 2 + k) as isize - 1;
                        if (0..10).contains(&p) {
                            expected[(group * 2 + oc) * 10 + p as usize] += input
                                [(group * 2 + ic) * 5 + t]
                                * weights[((group * 2 + ic) * 2 + oc) * 3 + k];
                        }
                    }
                }
            }
        }
    }
    for channels_last in [false, true] {
        let g = GraphBuilder::default();
        let x = g.input("x", &[1, 4, 1, 5], DType::F32);
        let w = g.constant(&weights, &[4, 2, 1, 3], Precision::Fp32);
        g.mark(
            "y",
            &g.conv(&x, &w, None, [1, 2], [1, 1], [0, 1], 2, true, [0, 1]),
        );
        let mut cfg = config();
        cfg.channels_last = channels_last;
        let out = CpuGraph::compile(g, &cfg)
            .unwrap()
            .run(&[("x", &bytes(&input))])
            .unwrap()
            .f32("y")
            .unwrap();
        for (a, b) in out.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}

#[test]
fn worker_gather_exception_returns_to_rust_and_graph_recovers() {
    let g = GraphBuilder::default();
    let x = g.input("x", &[2], DType::F32);
    let indices = g.input("indices", &[100000], DType::I32);
    g.mark("y", &g.gather(&x, &indices, 0));
    let mut graph = CpuGraph::compile(g, &config()).unwrap();
    let invalid = vec![2i32; 100000]
        .iter()
        .flat_map(|i| i.to_le_bytes())
        .collect::<Vec<_>>();
    assert!(
        graph
            .run(&[("x", &bytes(&[0.5, 1.])), ("indices", &invalid)])
            .is_err()
    );
    let valid = vec![0u8; 400000];
    assert_eq!(
        graph
            .run(&[("x", &bytes(&[0.5, 1.])), ("indices", &valid)])
            .unwrap()
            .f32("y")
            .unwrap(),
        vec![0.5; 100000]
    );
}

#[test]
fn nhwc_conv_views_branching_and_replay_preserve_canonical_outputs() {
    for fuse in [false, true] {
        for channels_last in [false, true] {
            let g = GraphBuilder::default();
            let x = g.input("x", &[1, 2, 1, 5], DType::F32);
            let weights = g.constant(&[1., -0.5, 0.25, 2.], &[2, 2, 1, 1], Precision::Fp32);
            let conv = g.conv(&x, &weights, None, [1, 1], [1, 1], [0, 0], 1, false, [0, 0]);
            let flat = g.reshape(&conv, &[1, 2, 5]);
            let activated = g.leaky(&flat, 0.1);
            let tail = g.slice(&activated, 2, 2, 3, 1);
            let concat = g.concat(&[tail.clone(), tail], 2);
            // Marking the intermediate must disable destructive Conv+activation fusion.
            g.mark("conv", &conv);
            g.mark("concat", &concat);
            g.mark("mean", &g.reduce(Op::Mean, &activated, &[1]));
            let mut cfg = config();
            cfg.fuse = fuse;
            cfg.channels_last = channels_last;
            let mut graph = CpuGraph::compile(g, &cfg).unwrap();
            for offset in [0., 3.] {
                let values = (0..10).map(|i| i as f32 - 4. + offset).collect::<Vec<_>>();
                let out = graph.run(&[("x", &bytes(&values))]).unwrap();
                let expected = (0..2)
                    .flat_map(|c| {
                        let values = &values;
                        (0..5).map(move |t| {
                            if c == 0 {
                                values[t] - 0.5 * values[t + 5]
                            } else {
                                0.25 * values[t] + 2. * values[t + 5]
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                assert_eq!(out.f32("conv").unwrap(), expected);
                let activation = expected
                    .iter()
                    .map(|&v| if v < 0. { v * 0.1 } else { v })
                    .collect::<Vec<_>>();
                let concat = (0..2)
                    .flat_map(|c| activation[c * 5 + 2..c * 5 + 5].repeat(2))
                    .collect::<Vec<_>>();
                assert_eq!(out.f32("concat").unwrap(), concat);
                for (t, v) in out.f32("mean").unwrap().iter().enumerate() {
                    assert!((*v - (activation[t] + activation[t + 5]) * 0.5).abs() < 1e-6);
                }
            }
        }
    }
}

#[test]
fn lbr_gru_matches_scalar_onnx_gates_and_bidirectional_order() {
    let g = GraphBuilder::default();
    let x = g.input("x", &[3, 1, 2], DType::F32);
    let w = (0..24).map(|i| (i as f32 - 8.) * 0.02).collect::<Vec<_>>();
    let r = (0..24)
        .map(|i| (i as f32 - 13.) * 0.015)
        .collect::<Vec<_>>();
    let bias = (0..24).map(|i| (i as f32 - 6.) * 0.01).collect::<Vec<_>>();
    let wt = g.constant(&w, &[2, 6, 2], Precision::Fp32);
    let rt = g.constant(&r, &[2, 6, 2], Precision::Fp32);
    let bt = g.constant(&bias, &[2, 12], Precision::Fp32);
    let y = g.node(
        Op::Gru,
        &[&x, &wt, &rt, &bt],
        vec![3, 2, 1, 2],
        DType::F32,
        vec![],
        [0.; 4],
        vec![],
        String::new(),
    );
    g.mark("y", &y);
    let mut graph = CpuGraph::compile(g, &config()).unwrap();
    let input = [0.2, -0.1, 0.4, 0.3, -0.2, 0.5];
    let mut expected = vec![0.; 12];
    for d in 0..2 {
        let mut h = [0f32; 2];
        for step in 0..3 {
            let t = if d == 0 { step } else { 2 - step };
            let linear = |gate: usize, o: usize| {
                let mut a = bias[d * 12 + gate * 2 + o];
                let mut b = bias[d * 12 + 6 + gate * 2 + o];
                for i in 0..2 {
                    a += w[(d * 6 + gate * 2 + o) * 2 + i] * input[t * 2 + i];
                    b += r[(d * 6 + gate * 2 + o) * 2 + i] * h[i];
                }
                (a, b)
            };
            let mut next = [0.; 2];
            for o in 0..2 {
                let (a, b) = linear(0, o);
                let z = 1. / (1. + (-(a + b)).exp());
                let (a, b) = linear(1, o);
                let reset = 1. / (1. + (-(a + b)).exp());
                let (a, b) = linear(2, o);
                let candidate = (a + reset * b).tanh();
                next[o] = z * h[o] + (1. - z) * candidate;
                expected[(t * 2 + d) * 2 + o] = next[o];
            }
            h = next;
        }
    }
    // Initial hidden state must be zero again on subsequent calls.
    for _ in 0..2 {
        let out = graph
            .run(&[("x", &bytes(&input))])
            .unwrap()
            .f32("y")
            .unwrap();
        for (a, b) in out.iter().zip(&expected) {
            assert!((a - b).abs() < 2e-6, "GRU {a} != {b}");
        }
    }
}

#[test]
fn stft_periodic_hann_reflection_and_magnitude_match_independent_dft() {
    let samples = 640;
    let frames = samples / 160 + 1;
    let wave = (0..samples)
        .map(|i| (i as f32 * 0.017).sin() * 0.3 + (i as f32 * 0.09).cos() * 0.2)
        .collect::<Vec<_>>();
    let mut basis = vec![0.; 128 * 513];
    for m in 0..128 {
        basis[m * 513 + 17] = 1.;
    }
    let g = GraphBuilder::default();
    let x = g.input("x", &[1, samples], DType::F32);
    let mel = g.constant(&basis, &[128, 513], Precision::Fp32);
    let y = g.node(
        Op::StftMel,
        &[&x, &mel],
        vec![1, 128, frames],
        DType::F32,
        vec![],
        [0.; 4],
        vec![],
        String::new(),
    );
    g.mark("y", &y);
    let out = CpuGraph::compile(g, &config())
        .unwrap()
        .run(&[("x", &bytes(&wave))])
        .unwrap()
        .f32("y")
        .unwrap();
    for t in 0..frames {
        let (mut re, mut im) = (0f64, 0f64);
        for i in 0..1024 {
            let p = (t * 160 + i) as isize - 512;
            let p = if p < 0 {
                -p
            } else if p >= samples as isize {
                2 * samples as isize - 2 - p
            } else {
                p
            } as usize;
            let window = 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / 1024.).cos();
            let v = (wave[p] * window) as f64;
            let phase = std::f64::consts::TAU * 17. * i as f64 / 1024.;
            re += v * phase.cos();
            im -= v * phase.sin();
        }
        let expected = ((re * re + im * im).sqrt().max(1e-5)).ln() as f32;
        assert!(
            (out[t] - expected).abs() < 2e-5,
            "STFT {} != {expected}",
            out[t]
        );
    }
}
