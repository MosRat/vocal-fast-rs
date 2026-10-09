use vocal_cpu::{CpuAttentionConfig, CpuAttentionMode, CpuConfig, CpuGraph};
use vocal_graph::graph::{DType, GraphBuilder};

fn build(qt: usize, kt: usize, online: bool, simd: bool, token: bool) -> CpuGraph {
    let g = GraphBuilder::default();
    // Unequal Q/K lengths and D/V widths, with non-unit feature strides and
    // BTHD views, exercise bounds and the generic stride-aware fallback.
    let q = g.transpose(&g.input("q", &[2, 19, 3, 12], DType::F32), &[0, 2, 1, 3]);
    let k = g.transpose(&g.input("k", &[2, 37, 3, 12], DType::F32), &[0, 2, 1, 3]);
    let v = g.transpose(&g.input("v", &[2, 37, 3, 13], DType::F32), &[0, 2, 1, 3]);
    // Slice every second channel: last-axis stride is two instead of one.
    let q = g.slice(&q, 3, 0, 6, 2);
    let k = g.slice(&k, 3, 0, 6, 2);
    let scores = g.scale(&g.matmul(&q, &k, true), 1. / 6f32.sqrt());
    let y = g.matmul(&g.softmax(&scores, 3), &v, false);
    g.mark(
        "y",
        &g.reshape(&g.transpose(&y, &[0, 2, 1, 3]), &[2, 19, 39]),
    );
    CpuGraph::compile(
        g,
        &CpuConfig {
            threads: 3,
            avx2: simd,
            attention: CpuAttentionConfig {
                mode: if online {
                    CpuAttentionMode::Online
                } else {
                    CpuAttentionMode::Dense
                },
                query_tile: qt,
                key_tile: kt,
                token_major_output: token,
            },
            ..CpuConfig::default()
        },
    )
    .unwrap()
}

#[test]
fn online_attention_preserves_views_tails_and_bidirectional_context() {
    let data = |n: usize| {
        (0..n)
            .flat_map(|i| ((i as f32 * 0.13).sin() * 7.).to_le_bytes())
            .collect::<Vec<_>>()
    };
    let q = data(2 * 19 * 3 * 12);
    let k = data(2 * 37 * 3 * 12);
    let v = data(2 * 37 * 3 * 13);
    let bindings = [
        ("q", q.as_slice()),
        ("k", k.as_slice()),
        ("v", v.as_slice()),
    ];
    let reference = build(4, 32, false, false, false)
        .run(&bindings)
        .unwrap()
        .f32("y")
        .unwrap();
    for (qt, kt) in [(1, 1), (4, 16), (8, 32)] {
        for token in [false, true] {
            let mut g = build(qt, kt, true, true, token);
            for _ in 0..2 {
                let actual = g.run(&bindings).unwrap().f32("y").unwrap();
                let max = actual
                    .iter()
                    .zip(&reference)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(max < 5e-5, "{qt}/{kt}/{token}: {max}");
            }
        }
    }
}

#[test]
fn simd_attention_is_stable_with_large_logits_and_odd_width() {
    fn builder() -> GraphBuilder {
        let g = GraphBuilder::default();
        let q = g.input("q", &[1, 2, 35, 65], DType::F32);
        let k = g.input("k", &q.shape, DType::F32);
        let v = g.input("v", &q.shape, DType::F32);
        let s = g.scale(&g.matmul(&q, &k, true), 0.125);
        g.mark("y", &g.matmul(&g.softmax(&s, 3), &v, false));
        g
    }
    let configs = [
        CpuConfig {
            threads: 2,
            ..CpuConfig::default()
        },
        CpuConfig {
            threads: 2,
            attention: CpuAttentionConfig {
                mode: CpuAttentionMode::Online,
                key_tile: 16,
                token_major_output: true,
                ..CpuAttentionConfig::default()
            },
            ..CpuConfig::default()
        },
    ];
    let mut dense = CpuGraph::compile(builder(), &configs[0]).unwrap();
    let mut online = CpuGraph::compile(builder(), &configs[1]).unwrap();
    let q = (0..4550)
        .flat_map(|i| (((i as f32 * 0.23).sin() + 0.5) * 16.).to_le_bytes())
        .collect::<Vec<_>>();
    let v = (0..4550)
        .flat_map(|i| (i as f32 * 0.19).cos().to_le_bytes())
        .collect::<Vec<_>>();
    let bindings = [
        ("q", q.as_slice()),
        ("k", q.as_slice()),
        ("v", v.as_slice()),
    ];
    let a = dense.run(&bindings).unwrap().f32("y").unwrap();
    let b = online.run(&bindings).unwrap().f32("y").unwrap();
    assert!(b.iter().all(|v| v.is_finite()));
    let max = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max < 3e-4, "large logit max error: {max}");
    assert!(online.memory_usage().workspace_bytes < dense.memory_usage().workspace_bytes);
}

#[test]
fn old_config_defaults_and_tile_validation() {
    let c: CpuConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(c.attention, CpuAttentionConfig::default());
    let mut c = c;
    c.attention.key_tile = 0;
    assert!(c.validate().is_err());
    c.attention.key_tile = 32;
    c.attention.query_tile = 33;
    assert!(c.validate().is_err());
}
