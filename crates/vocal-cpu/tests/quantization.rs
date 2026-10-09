use std::collections::BTreeMap;
use vocal_cpu::{CpuConfig, CpuGraph, CpuQuantization, CpuQuantizationRule};
use vocal_graph::{
    Precision,
    graph::{DType, GraphBuilder, Op},
};
fn bytes(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn recipe(alpha: Option<f32>) -> CpuQuantization {
    CpuQuantization {
        rules: vec![CpuQuantizationRule {
            prefix: "test".into(),
            smooth_alpha: alpha,
        }],
        calibration: BTreeMap::new(),
        ..CpuQuantization::default()
    }
}
fn convolution(deconv: bool, groups: usize) -> GraphBuilder {
    let g = GraphBuilder::default();
    let x = g.input("x", &[1, 4, 2, 5], DType::F32);
    let shape = if deconv {
        [4, 6 / groups, 1, 3]
    } else {
        [6, 4 / groups, 1, 3]
    };
    let values = (0..shape.iter().product())
        .map(|i| ((i % 11) as f32 - 5.) * 0.13)
        .collect::<Vec<_>>();
    let w = g.constant(&values, &shape, Precision::Fp32);
    let b = g.constant(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[6], Precision::Fp32);
    let y = g.conv(
        &x,
        &w,
        Some(&b),
        [1, 1],
        [1, 1],
        [0, 1],
        groups,
        deconv,
        [0, 0],
    );
    g.nodes.borrow_mut()[y.id as usize].name = "test.conv".into();
    g.mark("y", &y);
    g
}
#[test]
fn static_quantization_calibration_replay_and_grouped_layouts() {
    let config = CpuConfig {
        threads: 2,
        ..CpuConfig::default()
    };
    let data = bytes(&(0..40).map(|i| (i as f32 * 0.17).sin()).collect::<Vec<_>>());
    for deconv in [false, true] {
        for groups in [1, 2] {
            for alpha in [None, Some(0.7)] {
                let mut r = recipe(alpha);
                let mut observer =
                    CpuGraph::compile_calibration(convolution(deconv, groups), &config, &r)
                        .unwrap();
                assert!(observer.calibration().is_err());
                let expected = observer.run(&[("x", &data)]).unwrap().f32("y").unwrap();
                r.calibration = observer.calibration().unwrap();
                assert_eq!(r.calibration["test.conv"].len(), 4);
                let mut quantized =
                    match CpuGraph::compile_quantized(convolution(deconv, groups), &config, &r) {
                        Ok(g) => g,
                        Err(e) if e.to_string().contains("requires a VNNI/AMX") => return,
                        Err(e) => panic!("{e}"),
                    };
                assert!(
                    quantized.quantized_operators()[0]
                        .implementation
                        .contains("vnni")
                        || quantized.quantized_operators()[0]
                            .implementation
                            .contains("amx")
                );
                for _ in 0..3 {
                    let actual = quantized.run(&[("x", &data)]).unwrap().f32("y").unwrap();
                    let max = expected
                        .iter()
                        .zip(&actual)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0., f32::max);
                    assert!(
                        max < 0.04,
                        "deconv={deconv} groups={groups} alpha={alpha:?}: {max}"
                    );
                }
            }
        }
    }
}
#[test]
fn invalid_calibration_is_rejected() {
    let config = CpuConfig::default();
    let mut r = recipe(None);
    assert!(CpuGraph::compile_quantized(convolution(false, 1), &config, &r).is_err());
    r.calibration.insert("test.conv".into(), vec![f32::NAN; 4]);
    assert!(CpuGraph::compile_quantized(convolution(false, 1), &config, &r).is_err());
    r.calibration.insert("test.conv".into(), vec![1.; 3]);
    assert!(CpuGraph::compile_quantized(convolution(false, 1), &config, &r).is_err());
}

#[test]
fn int8_gemm_preserves_bias_and_transposed_weight_axis() {
    let config = CpuConfig {
        threads: 2,
        ..CpuConfig::default()
    };
    fn builder(transpose: bool) -> GraphBuilder {
        let g = GraphBuilder::default();
        let x = g.input("x", &[1, 3, 4], DType::F32);
        let values = (0..24).map(|i| (i as f32 * 0.21).sin()).collect::<Vec<_>>();
        let w = g.constant(
            &values,
            if transpose { &[6, 4] } else { &[4, 6] },
            Precision::Fp32,
        );
        g.nodes.borrow_mut()[w.id as usize].name = "test.linear.weight".into();
        let b = g.constant(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[6], Precision::Fp32);
        g.mark("y", &g.add(&g.matmul(&x, &w, transpose), &b));
        g
    }
    let data = bytes(&(0..12).map(|i| (i as f32 * 0.17).cos()).collect::<Vec<_>>());
    for transpose in [false, true] {
        let mut r = recipe(Some(0.7));
        let mut observer = CpuGraph::compile_calibration(builder(transpose), &config, &r).unwrap();
        let expected = observer.run(&[("x", &data)]).unwrap().f32("y").unwrap();
        r.calibration = observer.calibration().unwrap();
        let mut graph = match CpuGraph::compile_quantized(builder(transpose), &config, &r) {
            Ok(g) => g,
            Err(e) if e.to_string().contains("requires a VNNI/AMX") => return,
            Err(e) => panic!("{e}"),
        };
        let actual = graph.run(&[("x", &data)]).unwrap().f32("y").unwrap();
        assert!(
            expected
                .iter()
                .zip(actual)
                .all(|(a, b)| (a - b).abs() < 0.04)
        );
    }
}

#[test]
fn selected_deconvolution_batchnorm_uses_effective_weight_and_bias() {
    fn builder() -> GraphBuilder {
        let g = convolution(true, 1);
        let x = g.outputs.borrow_mut().pop().unwrap().1;
        g.nodes.borrow_mut()[x.id as usize].inputs.truncate(2);
        let scale = g.constant(
            &[0.01, 0.2, 1., 5., -0.7, 0.03],
            &[1, 6, 1, 1],
            Precision::Fp32,
        );
        let shift = g.constant(
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            &[1, 6, 1, 1],
            Precision::Fp32,
        );
        g.mark("y", &g.unary(Op::Relu, &g.add(&g.mul(&x, &scale), &shift)));
        g
    }
    let config = CpuConfig {
        threads: 2,
        ..CpuConfig::default()
    };
    let data = bytes(&(0..40).map(|i| (i as f32 * 0.17).sin()).collect::<Vec<_>>());
    let expected = CpuGraph::compile(builder(), &config)
        .unwrap()
        .run(&[("x", &data)])
        .unwrap()
        .f32("y")
        .unwrap();
    let mut r = recipe(Some(0.7));
    let mut observer = CpuGraph::compile_calibration(builder(), &config, &r).unwrap();
    let observed = observer.run(&[("x", &data)]).unwrap().f32("y").unwrap();
    assert!(
        expected
            .iter()
            .zip(&observed)
            .all(|(a, b)| (a - b).abs() < 4e-6)
    );
    r.calibration = observer.calibration().unwrap();
    let mut quantized = match CpuGraph::compile_quantized(builder(), &config, &r) {
        Ok(g) => g,
        Err(e) if e.to_string().contains("requires a VNNI/AMX") => return,
        Err(e) => panic!("{e}"),
    };
    assert_eq!(quantized.quantized_operators().len(), 1);
    let actual = quantized.run(&[("x", &data)]).unwrap().f32("y").unwrap();
    assert!(
        expected
            .iter()
            .zip(actual)
            .all(|(a, b)| (a - b).abs() < 0.3)
    );
    r.fold_deconvolution_batchnorm = false;
    let mut preserved = CpuGraph::compile_quantized(builder(), &config, &r).unwrap();
    let before_fold = preserved.run(&[("x", &data)]).unwrap().f32("y").unwrap();
    let mut folded = CpuGraph::compile_quantized(
        builder(),
        &config,
        &CpuQuantization {
            fold_deconvolution_batchnorm: true,
            ..r
        },
    )
    .unwrap();
    let after_fold = folded.run(&[("x", &data)]).unwrap().f32("y").unwrap();
    assert!(
        after_fold
            .iter()
            .zip(before_fold)
            .any(|(a, b)| (a - b).abs() > 1e-5)
    );
}

#[test]
fn quantized_linear_gelu_post_op_matches_unfused_int8() {
    fn builder() -> GraphBuilder {
        let g = GraphBuilder::default();
        let x = g.input("x", &[1, 17, 16], DType::F32);
        let weights = (0..512)
            .map(|i| (i as f32 * 0.21).sin() * 0.1)
            .collect::<Vec<_>>();
        let w = g.constant(&weights, &[32, 16], Precision::Fp32);
        g.nodes.borrow_mut()[w.id as usize].name = "test.linear.weight".into();
        let bias = (0..32).map(|i| (i as f32 - 16.) * 0.1).collect::<Vec<_>>();
        let b = g.constant(&bias, &[32], Precision::Fp32);
        let linear = g.add(&g.matmul(&x, &w, true), &b);
        g.mark("y", &g.unary(Op::Gelu, &linear));
        g
    }
    let config = CpuConfig {
        threads: 2,
        fuse_linear_gelu: false,
        ..Default::default()
    };
    let input = bytes(
        &(0..272)
            .map(|i| (i as f32 * 0.17).cos())
            .collect::<Vec<_>>(),
    );
    let mut r = recipe(Some(0.7));
    let mut observer = CpuGraph::compile_calibration(builder(), &config, &r).unwrap();
    observer.run(&[("x", &input)]).unwrap();
    r.calibration = observer.calibration().unwrap();
    let mut baseline = match CpuGraph::compile_quantized(builder(), &config, &r) {
        Ok(g) => g,
        Err(e) if e.to_string().contains("requires a VNNI/AMX") => return,
        Err(e) => panic!("{e}"),
    };
    let mut fused = CpuGraph::compile_quantized(
        builder(),
        &CpuConfig {
            fuse_linear_gelu: true,
            ..config
        },
        &r,
    )
    .unwrap();
    for _ in 0..3 {
        let a = baseline.run(&[("x", &input)]).unwrap().f32("y").unwrap();
        let b = fused.run(&[("x", &input)]).unwrap().f32("y").unwrap();
        assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() < 2e-6));
        assert!(
            a.iter().any(|v| *v < -0.05),
            "GELU must retain negative outputs"
        );
    }
}
