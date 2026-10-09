//! Paired convolution experiments on actual VFRM operator captures. Profile
//! clocks and numerical comparisons are outside production timing passes.
use std::{error, path::Path, time::Instant};
use vocal_cpu::{CpuConfig, CpuConvolutionConfig, CpuConvolutionMode, CpuGraph};
use vocal_graph::{
    Precision,
    graph::{DType, GraphBuilder, Op},
};

#[derive(serde::Deserialize)]
struct Manifest {
    cases: Vec<Case>,
}
#[derive(serde::Deserialize, serde::Serialize)]
struct Case {
    id: String,
    stage: String,
    name: String,
    op: u32,
    attrs: Vec<i32>,
    input_shapes: Vec<Vec<usize>>,
    output_shape: Vec<usize>,
    occurrences: usize,
}
type Error = Box<dyn error::Error>;
fn floats(path: &Path) -> Result<Vec<f32>, Error> {
    let data = std::fs::read(path)?;
    if data.len() % 4 != 0 {
        return Err("invalid FP32 fixture".into());
    }
    Ok(data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn bytes(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}
fn nhwc(data: &[f32], shape: &[usize]) -> Vec<f32> {
    let [n, c, h, w]: [usize; 4] = shape.try_into().unwrap();
    (0..n)
        .flat_map(|n| {
            (0..h * w).flat_map(move |t| (0..c).map(move |c0| data[(n * c + c0) * h * w + t]))
        })
        .collect()
}
fn build(
    c: &Case,
    inputs: &[Vec<f32>],
    cfg: &CpuConfig,
    gelu: bool,
    residual: bool,
) -> Result<CpuGraph, Error> {
    let g = GraphBuilder::default();
    let [n, ic, h, t]: [usize; 4] = c.input_shapes[0].as_slice().try_into()?;
    let x = g.transpose(&g.input("x", &[n, h, t, ic], DType::F32), &[0, 3, 1, 2]);
    let w = g.constant(&inputs[1], &c.input_shapes[1], Precision::Fp32);
    let bias = inputs
        .get(2)
        .map(|b| g.constant(b, &c.input_shapes[2], Precision::Fp32));
    let a = &c.attrs;
    let y = g.conv(
        &x,
        &w,
        bias.as_ref(),
        [a[0] as usize, a[1] as usize],
        [a[2] as usize, a[3] as usize],
        [a[4] as usize, a[5] as usize],
        a[8] as usize,
        false,
        [0, 0],
    );
    g.nodes.borrow_mut()[y.id as usize].name = c.name.clone();
    let y = if gelu { g.unary(Op::Gelu, &y) } else { y };
    let y = if residual {
        g.leaky(&g.binary(Op::Add, &x, &y), 0.1)
    } else {
        y
    };
    // Expose NHWC output contiguously so canonical output copying doesn't
    // dominate isolated compute, unlike a production pipeline stage boundary.
    g.mark("y", &g.transpose(&y, &[0, 2, 3, 1]));
    Ok(CpuGraph::compile(g, cfg)?)
}
fn error(a: &[f32], b: &[f32]) -> serde_json::Value {
    assert_eq!(a.len(), b.len());
    let squared: f64 = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum();
    let max = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .fold(0f64, f64::max);
    serde_json::json!({"rmse": (squared/a.len() as f64).sqrt(), "max_abs": max})
}
fn main() -> Result<(), Error> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err(
            "usage: convolution_benchmark fixtures-directory output.json iterations".into(),
        );
    }
    let directory = Path::new(&args[0]);
    let iterations: usize = args[2].parse()?;
    if iterations < 16 {
        return Err("at least 16 alternating pairs required".into());
    }
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(directory.join("manifest.json"))?)?;
    let mut rows = Vec::new();
    for case in manifest.cases.iter().filter(|c| {
        c.op == 24 && c.attrs[8] == 1 && c.input_shapes[0][2] == 1 && c.input_shapes[1][2] == 1
    }) {
        let inputs = (0..case.input_shapes.len())
            .map(|i| floats(&directory.join(format!("{}_input_{i}.f32", case.id))))
            .collect::<Result<Vec<_>, _>>()?;
        let x = nhwc(&inputs[0], &case.input_shapes[0]);
        let bindings = [("x", bytes(&x))];
        let mut variants = vec![(
            "matmul1d",
            CpuConvolutionConfig {
                mode: CpuConvolutionMode::Matmul1d,
                fuse_gelu: false,
                ..Default::default()
            },
            false,
            false,
        )];
        variants.push((
            "pointwise1d",
            CpuConvolutionConfig {
                mode: CpuConvolutionMode::Pointwise1d,
                ..Default::default()
            },
            false,
            false,
        ));
        if case.input_shapes[0] == case.output_shape {
            variants.push((
                "residual",
                CpuConvolutionConfig {
                    fuse_residual: true,
                    ..Default::default()
                },
                false,
                true,
            ));
        }
        if case.name.contains("feature_extractor") && !case.name.contains("layers.0.") {
            variants.push((
                "conv_gelu",
                CpuConvolutionConfig {
                    mode: CpuConvolutionMode::Direct,
                    fuse_gelu: true,
                    ..Default::default()
                },
                true,
                false,
            ));
            variants.push((
                "matmul1d_gelu",
                CpuConvolutionConfig {
                    mode: CpuConvolutionMode::Matmul1d,
                    fuse_gelu: true,
                    ..Default::default()
                },
                true,
                false,
            ));
        }
        for (variant, convolution, gelu, residual) in variants {
            let baseline_config = CpuConfig {
                threads: 6,
                ..Default::default()
            };
            let candidate_config = CpuConfig {
                convolution,
                ..baseline_config.clone()
            };
            let mut baseline = build(case, &inputs, &baseline_config, gelu, residual)?;
            let mut candidate = build(case, &inputs, &candidate_config, gelu, residual)?;
            let mut a = Vec::new();
            let mut b = Vec::new();
            for _ in 0..8 {
                baseline.run_f32_into(&bindings, "y", &mut a)?;
                candidate.run_f32_into(&bindings, "y", &mut b)?;
            }
            let mut base_ms = Vec::new();
            let mut candidate_ms = Vec::new();
            for i in 0..iterations {
                let run = |g: &mut CpuGraph,
                           y: &mut Vec<f32>,
                           times: &mut Vec<f64>|
                 -> Result<(), Error> {
                    let start = Instant::now();
                    g.run_f32_into(&bindings, "y", y)?;
                    times.push(start.elapsed().as_secs_f64() * 1000.);
                    Ok(())
                };
                if i % 2 == 0 {
                    run(&mut baseline, &mut a, &mut base_ms)?;
                    run(&mut candidate, &mut b, &mut candidate_ms)?;
                } else {
                    run(&mut candidate, &mut b, &mut candidate_ms)?;
                    run(&mut baseline, &mut a, &mut base_ms)?;
                }
            }
            let fp32_error = error(&a, &b);
            let mut diagnostic = build(
                case,
                &inputs,
                &CpuConfig {
                    profile: true,
                    ..candidate_config.clone()
                },
                gelu,
                residual,
            )?;
            diagnostic.run_f32_into(&bindings, "y", &mut b)?;
            rows.push(serde_json::json!({"case":case, "variant":variant, "baseline_ms":base_ms, "candidate_ms":candidate_ms,
                "fp32_error":fp32_error, "config":candidate_config.convolution, "diagnostic":diagnostic.timings(),
                "baseline_memory":baseline.memory_usage(), "candidate_memory":candidate.memory_usage()}));
        }
        std::fs::write(
            &args[1],
            serde_json::to_vec_pretty(
                &serde_json::json!({"iterations":iterations, "operator_version":vocal_cpu::operator_version(), "rows":rows,
            "note":"6 threads; real FP32 VFRM captures; prepared NHWC source, packed weights; binding/output copies included; clocks collected separately; isolated operators, not pipeline latency"}),
            )?,
        )?;
        eprintln!("measured {}", case.name);
    }
    Ok(())
}
