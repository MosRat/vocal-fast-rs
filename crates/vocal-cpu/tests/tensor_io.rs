use vocal_cpu::{CpuConfig, CpuGraph};
use vocal_graph::graph::{DType, GraphBuilder};

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn materialized_reshape_preserves_canonical_order_and_marked_views() {
    fn graph(fast: bool) -> CpuGraph {
        let g = GraphBuilder::default();
        let x = g.input("x", &[2, 3, 5], DType::F32);
        let view = g.transpose(&x, &[0, 2, 1]);
        g.mark("view", &view);
        g.mark("flat", &g.reshape(&view, &[2, 15]));
        CpuGraph::compile(
            g,
            &CpuConfig {
                fast_tensor_io: fast,
                ..Default::default()
            },
        )
        .unwrap()
    }
    let mut reference = graph(false);
    let mut candidate = graph(true);
    let memory = candidate.memory_usage();
    for gain in [1., -0.125, 0.] {
        let values = (0..30).map(|i| (i as f32 - 12.) * gain).collect::<Vec<_>>();
        let input = floats(&values);
        let a = reference.run(&[("x", &input)]).unwrap();
        let b = candidate.run(&[("x", &input)]).unwrap();
        assert_eq!(a.tensors, b.tensors);
        let expected = (0..2)
            .flat_map(|batch| {
                let values = &values;
                (0..5).flat_map(move |t| (0..3).map(move |c| values[batch * 15 + c * 5 + t]))
            })
            .collect::<Vec<_>>();
        assert_eq!(b.f32("flat").unwrap(), expected);
        assert_eq!(
            candidate.memory_usage().workspace_bytes,
            memory.workspace_bytes
        );
    }
}

#[test]
fn row_gather_handles_negative_indices_strided_fallback_and_error_recovery() {
    fn graph(fast: bool, transpose: bool) -> CpuGraph {
        let g = GraphBuilder::default();
        let x = g.input("x", &[2, 4, 7], DType::F32);
        let x = if transpose {
            g.transpose(&x, &[0, 2, 1])
        } else {
            x
        };
        let i = g.input("indices", &[2, x.shape[1], 11], DType::I32);
        g.mark("y", &g.gather_elements(&x, &i, 2));
        CpuGraph::compile(
            g,
            &CpuConfig {
                threads: 3,
                fast_tensor_io: fast,
                ..Default::default()
            },
        )
        .unwrap()
    }
    for transpose in [false, true] {
        let width = if transpose { 4 } else { 7 };
        let count = 2 * if transpose { 7 } else { 4 } * 11;
        let mut reference = graph(false, transpose);
        let mut candidate = graph(true, transpose);
        for offset in [0., -0.25] {
            let x = floats(&(0..56).map(|i| i as f32 + offset).collect::<Vec<_>>());
            let indices = (0..count)
                .map(|i| if i % 3 == 0 { -1 } else { i % width })
                .collect::<Vec<i32>>();
            let bytes = |values: &[i32]| {
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()
            };
            let mut invalid = indices.clone();
            invalid[3] = width;
            assert!(
                candidate
                    .run(&[("x", &x), ("indices", &bytes(&invalid))])
                    .is_err()
            );
            invalid[3] = -width - 1;
            assert!(
                candidate
                    .run(&[("x", &x), ("indices", &bytes(&invalid))])
                    .is_err()
            );
            let indices = bytes(&indices);
            let a = reference.run(&[("x", &x), ("indices", &indices)]).unwrap();
            let b = candidate.run(&[("x", &x), ("indices", &indices)]).unwrap();
            assert_eq!(a.tensors, b.tensors);
        }
    }
}
