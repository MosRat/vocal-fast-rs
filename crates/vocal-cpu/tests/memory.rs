use vocal_cpu::{CpuConfig, CpuGraph};
use vocal_graph::{
    Precision,
    graph::{DType, GraphBuilder},
};

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn packed_weights_release_originals_but_preserve_marked_and_shared_consumers() {
    for mark_weights in [false, true] {
        let g = GraphBuilder::default();
        let x = g.input("x", &[2, 64], DType::F32);
        let weights = vec![0.125; 64 * 64];
        let w = g.constant(&weights, &[64, 64], Precision::Fp32);
        g.mark("y", &g.matmul(&x, &w, false));
        if mark_weights {
            g.mark("w", &w);
        }
        let mut graph = CpuGraph::compile(g, &CpuConfig::default()).unwrap();
        let before = graph.memory_usage();
        assert!(before.packed_bytes >= weights.len() * 4);
        if !mark_weights {
            assert!(before.tensor_bytes < weights.len() * 4);
        }
        for value in [1., 2.] {
            let result = graph.run(&[("x", &bytes(&vec![value; 128]))]).unwrap();
            assert_eq!(result.f32("y").unwrap(), vec![value * 8.; 128]);
            if mark_weights {
                assert_eq!(result.f32("w").unwrap(), weights);
            }
            assert_eq!(graph.memory_usage().workspace_bytes, before.workspace_bytes);
        }
    }
    let g = GraphBuilder::default();
    let x = g.input("x", &[4, 4], DType::F32);
    let w = g.constant(&[0.25; 16], &[4, 4], Precision::Fp32);
    g.mark("gemm", &g.matmul(&x, &w, false));
    // The same constant is still required by a native elementwise consumer.
    g.mark("add", &g.add(&x, &w));
    let mut graph = CpuGraph::compile(g, &CpuConfig::default()).unwrap();
    let result = graph.run(&[("x", &bytes(&[1.; 16]))]).unwrap();
    assert_eq!(result.f32("gemm").unwrap(), vec![1.; 16]);
    assert_eq!(result.f32("add").unwrap(), vec![1.25; 16]);
}

#[test]
fn bounded_spin_pool_can_move_threads_recover_and_drop() {
    for spin_wait_us in [0, 50, 200] {
        let g = GraphBuilder::default();
        let x = g.input("x", &[100000], DType::F32);
        let index = g.input("index", &[100000], DType::I32);
        g.mark("y", &g.gather(&x, &index, 0));
        let graph = CpuGraph::compile(
            g,
            &CpuConfig {
                threads: 3,
                spin_wait_us,
                pin_to_performance_cores: spin_wait_us > 0,
                ..CpuConfig::default()
            },
        )
        .unwrap();
        std::thread::spawn(move || {
            let mut graph = graph;
            let x = bytes(&vec![2.; 100000]);
            let invalid = vec![100000i32; 100000]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            assert!(graph.run(&[("x", &x), ("index", &invalid)]).is_err());
            let valid = vec![0; 400000];
            for _ in 0..3 {
                assert_eq!(
                    graph
                        .run(&[("x", &x), ("index", &valid)])
                        .unwrap()
                        .f32("y")
                        .unwrap(),
                    vec![2.; 100000]
                );
            }
        })
        .join()
        .unwrap();
    }
}

#[test]
fn reusable_fp32_output_validates_dtype_and_preserves_storage_on_replay() {
    let g = GraphBuilder::default();
    let x = g.input("x", &[4], DType::F32);
    g.mark("y", &g.scale(&x, 2.));
    g.mark("index", &g.cast(&x, DType::I32));
    let mut graph = CpuGraph::compile(g, &CpuConfig::default()).unwrap();
    let mut output = vec![0.; 4];
    let pointer = output.as_ptr();
    for value in [1., 3.] {
        graph
            .run_f32_into(&[("x", &bytes(&[value; 4]))], "y", &mut output)
            .unwrap();
        assert_eq!(output, vec![value * 2.; 4]);
        assert_eq!(pointer, output.as_ptr());
    }
    assert!(
        graph
            .run_f32_into(&[("x", &bytes(&[1.; 4]))], "index", &mut output)
            .is_err()
    );
    assert!(graph.run_f32_into(&[], "y", &mut output).is_err());
    assert!(
        graph
            .run_f32_into(&[("x", &bytes(&[1.; 4]))], "missing", &mut output)
            .is_err()
    );
}

#[cfg(windows)]
#[test]
fn performance_core_policy_restores_the_callers_affinity() {
    #[repr(C)]
    #[derive(Default)]
    struct GroupAffinity {
        mask: usize,
        group: u16,
        reserved: [u16; 3],
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> *mut std::ffi::c_void;
        fn GetThreadGroupAffinity(thread: *mut std::ffi::c_void, result: *mut GroupAffinity)
        -> i32;
    }
    let affinity = || {
        let mut result = GroupAffinity::default();
        // SAFETY: A valid current-thread pseudo handle and writable ABI structure.
        assert_ne!(
            unsafe { GetThreadGroupAffinity(GetCurrentThread(), &mut result) },
            0
        );
        (result.mask, result.group)
    };
    let before = affinity();
    let g = GraphBuilder::default();
    let x = g.input("x", &[4], DType::F32);
    g.mark("y", &g.scale(&x, 2.));
    let mut graph = CpuGraph::compile(
        g,
        &CpuConfig {
            pin_to_performance_cores: true,
            ..CpuConfig::default()
        },
    )
    .unwrap();
    assert_eq!(affinity(), before);
    graph.run(&[("x", &bytes(&[1.; 4]))]).unwrap();
    assert_eq!(affinity(), before);
}
