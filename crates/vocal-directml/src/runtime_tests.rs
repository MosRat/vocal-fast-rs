//! GPU regressions for binding arenas, view lowering and optional GEMM fusion.
use super::*;
use crate::{ExecutionConfig, IoBuffers, Precision};

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
#[ignore = "requires a DirectML hardware adapter"]
fn weight_arenas_span_custom_staging_batches_and_pad_half_constants() {
    let device = DirectMlDevice::with_debug_layer(0, true).unwrap();
    let g = GraphBuilder::default();
    let elements = 1024 * 1024 / 4 + 3;
    let x = g.input("x", &[1, 1], DType::F32);
    let weights = g.constant(&vec![0.5; elements], &[1, elements], Precision::Fp32);
    let bias = g.scalar(0.25, Precision::Fp16);
    g.mark(
        "y",
        &g.reduce(
            Op::Mean,
            &g.add(&g.add(&x, &weights), &g.cast(&bias, DType::F32)),
            &[1],
        ),
    );
    let mut allocations = Vec::new();
    for pooled_weights in [false, true] {
        let mut graph = device
            .clone()
            .with_execution(ExecutionConfig {
                pooled_weights,
                weight_staging_mib: 1,
                ..Default::default()
            })
            .compile(&g)
            .unwrap();
        let memory = graph.memory();
        assert_eq!(memory.staging_bytes, 1024 * 1024);
        assert_eq!(
            memory.constant_resource_count,
            if pooled_weights { 1 } else { 2 }
        );
        allocations.push(memory.constant_allocation_bytes);
        assert!(
            (graph
                .run(&[("x", &bytes(&[2.]))])
                .unwrap()
                .f32("y")
                .unwrap()[0]
                - 2.75)
                .abs()
                < 1e-4
        );
    }
    assert!(allocations[1] < allocations[0]);
}

#[test]
#[ignore = "requires a DirectML hardware adapter"]
fn pooled_offsets_multiple_outputs_and_two_feedback_states() {
    let device = DirectMlDevice::with_debug_layer(0, true).unwrap();
    for io_buffers in [IoBuffers::Dedicated, IoBuffers::Pooled] {
        let g = GraphBuilder::default();
        let x = g.input("x", &[1, 2], DType::F32);
        let history = g.input("history", &[1, 2], DType::F32);
        let tag = g.input("tag", &[1], DType::I32);
        let counter = g.input("counter", &[1], DType::F32);
        let half = g.input("half", &[1], DType::F16);
        let next = g.add(&history, &x);
        let next_counter = g.add(&counter, &g.scalar(1., Precision::Fp32));
        g.feedback(&history, &next);
        g.mark("half", &half);
        g.feedback(&counter, &next_counter);
        g.mark("audio", &g.add(&next, &next_counter));
        g.mark("tag", &g.cast(&tag, DType::F32));
        let mut graph = device
            .clone()
            .with_execution(ExecutionConfig {
                io_buffers,
                ..Default::default()
            })
            .compile(&g)
            .unwrap();
        drop(g);
        let half = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let tag = 37i32.to_le_bytes();
        let input = bytes(&[1., 2.]);
        // Named binding order may differ from the compiled input order.
        let inputs = [("half", &half[..]), ("x", &input), ("tag", &tag[..])];
        assert!(graph.run(&inputs[..2]).is_err());
        for expected in [[2., 3.], [4., 6.]] {
            let out = graph.run(&inputs).unwrap();
            assert_eq!(out.f32("audio").unwrap(), expected);
            assert_eq!(out.f32("tag").unwrap(), [37.]);
            assert!(out.f32("half").is_err());
            assert_eq!(
                out.tensors.iter().find(|(n, _)| n == "half").unwrap().1,
                half
            );
        }
        graph.reset_states().unwrap();
        assert_eq!(graph.run(&inputs).unwrap().f32("audio").unwrap(), [2., 3.]);
        let memory = graph.memory();
        if io_buffers == IoBuffers::Pooled {
            assert_eq!(memory.io_resource_count, 4);
            assert_eq!(memory.upload_copies, 1);
            assert_eq!(memory.readback_copies, 1);
        } else {
            assert_eq!(memory.io_resource_count, 10);
            assert_eq!(memory.upload_copies, 3);
            assert_eq!(memory.readback_copies, 3);
        }
    }
}

#[test]
#[ignore = "requires a DirectML hardware adapter"]
fn gemm_bias_fusion_preserves_marked_and_branched_results() {
    let device = DirectMlDevice::with_debug_layer(0, true).unwrap();
    for precision in [Precision::Fp32, Precision::Fp16] {
        for mark_product in [false, true] {
            let g = GraphBuilder::default();
            let x = g.input("x", &[2, 3], DType::F32);
            let w = g.constant(&[1., 2., 3., 4., 5., 6.], &[3, 2], precision);
            let product = g.matmul(&g.cast(&x, precision.into()), &w, false);
            if mark_product {
                g.mark("product", &g.cast(&product, DType::F32));
            }
            // The bias is deliberately declared after the product.
            let bias = g.constant(&[10., 20.], &[2], precision);
            g.mark("y", &g.cast(&g.add(&bias, &product), DType::F32));
            let mut graph = device
                .clone()
                .with_execution(ExecutionConfig {
                    fuse_gemm_bias: true,
                    ..Default::default()
                })
                .compile(&g)
                .unwrap();
            assert_eq!(graph.memory().gemm_bias_fusions, u64::from(!mark_product));
            let out = graph
                .run(&[("x", &bytes(&[1., 2., 3., 4., 5., 6.]))])
                .unwrap();
            assert_eq!(out.f32("y").unwrap(), [32., 48., 59., 84.]);
            if mark_product {
                assert_eq!(out.f32("product").unwrap(), [22., 28., 49., 64.]);
            }
        }
    }
}

#[test]
#[ignore = "requires a DirectML hardware adapter"]
fn contiguous_views_preserve_transpose_reshape_and_broadcast() {
    let device = DirectMlDevice::with_debug_layer(0, true).unwrap();
    let g = GraphBuilder::default();
    let x = g.input("x", &[2, 3], DType::F32);
    let transposed = g.transpose(&x, &[1, 0]);
    let flattened = g.reshape(&transposed, &[1, 6]);
    let padded = g.reshape(&flattened, &[1, 1, 6]);
    let scalar = g.scalar(0.5, Precision::Fp32);
    g.mark("y", &g.reshape(&g.add(&padded, &scalar), &[6]));
    let mut counts = Vec::new();
    for execution in [
        ExecutionConfig::legacy(),
        ExecutionConfig {
            contiguous_views: true,
            ..Default::default()
        },
    ] {
        let mut graph = device
            .clone()
            .with_execution(execution)
            .compile(&g)
            .unwrap();
        counts.push(graph.memory().lowered_nodes);
        assert_eq!(
            graph
                .run(&[("x", &bytes(&[1., 2., 3., 4., 5., 6.]))])
                .unwrap()
                .f32("y")
                .unwrap(),
            [1.5, 4.5, 2.5, 5.5, 3.5, 6.5]
        );
    }
    assert!(counts[1] < counts[0]);
}
