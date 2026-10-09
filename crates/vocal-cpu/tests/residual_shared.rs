//! A shared residual sum must remain available before the activation, while
//! its producer can still fuse the sum. This mirrors the Synth resblock DAG.
use vocal_cpu::{CpuConfig, CpuConvolutionConfig, CpuGraph};
use vocal_graph::{
    Precision,
    graph::{DType, GraphBuilder, Op},
};

fn build(fused: bool) -> CpuGraph {
    let g = GraphBuilder::default();
    let x = g.transpose(&g.input("x", &[1, 67, 1, 8], DType::F32), &[0, 3, 2, 1]);
    let x3 = g.reshape(&x, &[1, 8, 67]);
    let w = g.constant(
        &(0..8 * 8 * 3)
            .map(|i| (i as f32 * 0.11).sin() * 0.05)
            .collect::<Vec<_>>(),
        &[8, 8, 1, 3],
        Precision::Fp32,
    );
    let a = g.conv(&x, &w, None, [1, 1], [1, 1], [0, 1], 1, false, [0, 0]);
    let a3 = g.reshape(&a, &[1, 8, 67]);
    let sum = g.binary(Op::Add, &x3, &a3);
    // Both the next convolution and its residual path consume this sum.
    let activated = g.leaky(&sum, 0.1);
    let b = g.conv(
        &g.reshape(&activated, &[1, 8, 1, 67]),
        &w,
        None,
        [1, 1],
        [1, 3],
        [0, 3],
        1,
        false,
        [0, 0],
    );
    g.mark("y", &g.binary(Op::Add, &sum, &g.reshape(&b, &[1, 8, 67])));
    CpuGraph::compile(
        g,
        &CpuConfig {
            threads: 3,
            profile: true,
            convolution: CpuConvolutionConfig {
                fuse_residual: fused,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn shared_sum_replay_matches_unfused_and_uses_fast_convolution() {
    let mut baseline = build(false);
    let mut candidate = build(true);
    let memory = candidate.memory_usage();
    for gain in [1., -0.003, 0.] {
        let data = (0..8 * 67)
            .map(|i| (i as f32 * 0.09).cos() * gain)
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let a = baseline.run(&[("x", &data)]).unwrap().f32("y").unwrap();
        let b = candidate.run(&[("x", &data)]).unwrap().f32("y").unwrap();
        for (a, b) in a.iter().zip(&b) {
            assert!((a - b).abs() < 1e-6);
        }
        let timings = candidate.timings();
        assert!(
            !timings
                .iter()
                .any(|t| t.operator == "Add" || t.operator == "AddLeaky")
        );
        assert_eq!(
            timings
                .iter()
                .filter(|t| t.implementation.contains("binary_add"))
                .count(),
            2
        );
        assert!(timings.iter().all(|t| !t.implementation.contains("ref:")));
        assert_eq!(memory.tensor_bytes, candidate.memory_usage().tensor_bytes);
    }
}
