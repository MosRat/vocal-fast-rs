use vocal_cpu::{CpuConfig, CpuConvolutionConfig, CpuConvolutionMode, CpuGraph};
use vocal_graph::{
    Precision,
    graph::{DType, GraphBuilder, Op},
};

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn unfolded_conv_preserves_padding_dilation_strides_bias_and_replay() {
    // Includes directly viewed non-overlapping patches, an odd batch boundary,
    // and canonical / NHWC source views. Every path compares independent math.
    for (samples, width, stride, dilation, pad) in
        [(19, 3, 2, 1, 0), (20, 2, 2, 1, 0), (19, 5, 1, 3, 6)]
    {
        for (nhwc, mode) in [false, true].into_iter().flat_map(|nhwc| {
            [
                CpuConvolutionMode::Matmul1d,
                CpuConvolutionMode::Pointwise1d,
            ]
            .into_iter()
            .map(move |mode| (nhwc, mode))
        }) {
            let (batch, ic, oc) = (2, 7, 9);
            let time = (samples + 2 * pad - (width - 1) * dilation - 1) / stride + 1;
            let g = GraphBuilder::default();
            let input = if nhwc {
                g.transpose(
                    &g.input("x", &[batch, samples, 1, ic], DType::F32),
                    &[0, 3, 2, 1],
                )
            } else {
                g.input("x", &[batch, ic, 1, samples], DType::F32)
            };
            let weights = (0..oc * ic * width)
                .map(|i| (i as f32 * 0.17).sin() * 0.03)
                .collect::<Vec<_>>();
            let bias = (0..oc).map(|i| i as f32 * 0.07 - 0.2).collect::<Vec<_>>();
            let w = g.constant(&weights, &[oc, ic, 1, width], Precision::Fp32);
            let b = g.constant(&bias, &[oc], Precision::Fp32);
            let y = g.conv(
                &input,
                &w,
                Some(&b),
                [1, stride],
                [1, dilation],
                [0, pad],
                1,
                false,
                [0, 0],
            );
            g.mark("y", &y);
            let mut graph = CpuGraph::compile(
                g,
                &CpuConfig {
                    threads: 3,
                    profile: true,
                    convolution: CpuConvolutionConfig {
                        mode,
                        fuse_gelu: false,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();
            let memory = graph.memory_usage();
            for gain in [1., -0.25, 0.] {
                let canonical = (0..batch * ic * samples)
                    .map(|i| (i as f32 * 0.13).cos() * gain)
                    .collect::<Vec<_>>();
                let values = if nhwc {
                    (0..batch)
                        .flat_map(|b| {
                            let canonical = &canonical;
                            (0..samples).flat_map(move |t| {
                                (0..ic).map(move |c| canonical[(b * ic + c) * samples + t])
                            })
                        })
                        .collect()
                } else {
                    canonical.clone()
                };
                let y = graph
                    .run(&[("x", &bytes(&values))])
                    .unwrap()
                    .f32("y")
                    .unwrap();
                for b in 0..batch {
                    for o in 0..oc {
                        for t in 0..time {
                            let mut expected = bias[o] as f64;
                            for c in 0..ic {
                                for k in 0..width {
                                    let p = (t * stride + k * dilation) as isize - pad as isize;
                                    if (0..samples as isize).contains(&p) {
                                        expected += canonical[(b * ic + c) * samples + p as usize]
                                            as f64
                                            * weights[(o * ic + c) * width + k] as f64;
                                    }
                                }
                            }
                            assert!((y[(b * oc + o) * time + t] as f64 - expected).abs() < 2e-6);
                        }
                    }
                }
                assert!(
                    graph
                        .timings()
                        .iter()
                        .any(|t| t.implementation.starts_with("unfold1d"))
                );
                assert_eq!(graph.memory_usage().workspace_bytes, memory.workspace_bytes);
            }
        }
    }
}

#[test]
fn residual_postop_preserves_views_branches_and_operand_order() {
    fn graph(
        fused: bool,
        mark_pre: bool,
        swap: bool,
        late: bool,
        mode: CpuConvolutionMode,
    ) -> CpuGraph {
        let g = GraphBuilder::default();
        let x = g.transpose(&g.input("x", &[2, 23, 1, 5], DType::F32), &[0, 3, 2, 1]);
        let early_residual = g.reshape(&x, &[2, 5, 23]);
        let weights = (0..75)
            .map(|i| (i as f32 * 0.31).sin() * 0.07)
            .collect::<Vec<_>>();
        let w = g.constant(&weights, &[5, 5, 1, 3], Precision::Fp32);
        let y = g.conv(&x, &w, None, [1, 1], [1, 2], [0, 2], 1, false, [0, 0]);
        let view = g.reshape(&y, &[2, 5, 23]);
        let residual = if late {
            g.reshape(&g.scale(&x, 0.5), &[2, 5, 23])
        } else {
            early_residual
        };
        let add = if swap {
            g.binary(Op::Add, &residual, &view)
        } else {
            g.binary(Op::Add, &view, &residual)
        };
        g.mark("y", &g.leaky(&add, 0.1));
        if mark_pre {
            g.mark("pre", &view);
        }
        CpuGraph::compile(
            g,
            &CpuConfig {
                threads: 3,
                profile: true,
                convolution: CpuConvolutionConfig {
                    mode,
                    fuse_residual: fused,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap()
    }
    for mode in [
        CpuConvolutionMode::Direct,
        CpuConvolutionMode::Matmul1d,
        CpuConvolutionMode::Pointwise1d,
    ] {
        for (marked, swap, late) in [
            (false, false, false),
            (false, true, false),
            (true, false, false),
            (false, true, true),
        ] {
            let mut a = graph(false, marked, swap, late, mode);
            let mut b = graph(true, marked, swap, late, mode);
            for gain in [1., 0.001, -0.3] {
                let x = bytes(
                    &(0..230)
                        .map(|i| (i as f32 * 0.2).sin() * gain)
                        .collect::<Vec<_>>(),
                );
                let expected = a.run(&[("x", &x)]).unwrap();
                let actual = b.run(&[("x", &x)]).unwrap();
                for (a, b) in expected
                    .f32("y")
                    .unwrap()
                    .iter()
                    .zip(actual.f32("y").unwrap())
                {
                    assert!((a - b).abs() < 1e-6, "{mode:?} {a} != {b}");
                }
                if marked {
                    assert_eq!(expected.tensors["pre"], actual.tensors["pre"]);
                }
                assert_eq!(
                    b.timings().iter().any(|t| t.operator == "AddLeaky"),
                    marked || late
                );
            }
        }
    }
}

#[test]
fn conv_gelu_fusion_preserves_marked_pre_activation_and_shared_consumers() {
    fn graph(fused: bool, marked: bool, branch: bool) -> CpuGraph {
        let g = GraphBuilder::default();
        let x = g.input("x", &[1, 3, 1, 17], DType::F32);
        let w = g.constant(
            &(0..45)
                .map(|i| (i as f32 - 20.) * 0.013)
                .collect::<Vec<_>>(),
            &[5, 3, 1, 3],
            Precision::Fp32,
        );
        let conv = g.conv(&x, &w, None, [1, 1], [1, 1], [0, 1], 1, false, [0, 0]);
        let view = g.reshape(&conv, &[1, 5, 17]);
        g.mark("y", &g.unary(Op::Gelu, &view));
        if marked {
            g.mark("pre", &view);
        }
        if branch {
            g.mark("branch", &g.scale(&conv, 0.3));
        }
        CpuGraph::compile(
            g,
            &CpuConfig {
                profile: true,
                convolution: CpuConvolutionConfig {
                    fuse_gelu: fused,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap()
    }
    for (marked, branch) in [(false, false), (true, false), (false, true)] {
        let mut reference = graph(false, marked, branch);
        let mut candidate = graph(true, marked, branch);
        for gain in [1., -1., 0.001] {
            let x = bytes(&(0..51).map(|i| (i as f32 - 22.) * gain).collect::<Vec<_>>());
            let a = reference.run(&[("x", &x)]).unwrap();
            let b = candidate.run(&[("x", &x)]).unwrap();
            if marked {
                assert_eq!(a.tensors["pre"], b.tensors["pre"]);
            }
            if branch {
                assert_eq!(a.tensors["branch"], b.tensors["branch"]);
            }
            for (a, b) in a.f32("y").unwrap().iter().zip(b.f32("y").unwrap()) {
                assert!((a - b).abs() < 2e-6);
            }
            assert_eq!(
                candidate.timings().iter().any(|n| n.operator == "Gelu"),
                marked || branch
            );
        }
    }
}
