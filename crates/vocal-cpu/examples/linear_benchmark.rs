//! Paired constant-weight FP32 projection implementations on VFRM captures.
use std::{error::Error, path::Path, time::Instant};
use vocal_cpu::{CpuConfig, CpuGraph, CpuLinearMode};
use vocal_graph::{
    Precision,
    graph::{DType, GraphBuilder, Op},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

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
fn floats(path: &Path) -> Result<Vec<f32>> {
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
fn build(case: &Case, inputs: &[Vec<f32>], mode: CpuLinearMode, profile: bool) -> Result<CpuGraph> {
    let g = GraphBuilder::default();
    let x = g.input("x", &case.input_shapes[0], DType::F32);
    let w = g.constant(&inputs[1], &case.input_shapes[1], Precision::Fp32);
    let mut y = g.matmul(&x, &w, case.attrs[0] != 0);
    if let Some(bias) = inputs.get(2) {
        let b = g.constant(bias, &case.input_shapes[2], Precision::Fp32);
        y = g.binary(Op::Add, &y, &b);
    }
    g.mark("y", &y);
    Ok(CpuGraph::compile(
        g,
        &CpuConfig {
            threads: 6,
            profile,
            linear_mode: mode,
            ..Default::default()
        },
    )?)
}
fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: linear_benchmark fixtures-directory output.json iterations".into());
    }
    let dir = Path::new(&args[0]);
    let iterations: usize = args[2].parse()?;
    if iterations < 16 {
        return Err("at least 16 alternating pairs required".into());
    }
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
    let mut rows = Vec::new();
    for case in manifest
        .cases
        .iter()
        .filter(|c| (c.op == 23 || c.op == 53) && c.input_shapes[1].len() == 2)
    {
        for mode in [CpuLinearMode::InnerProduct, CpuLinearMode::Pointwise1d] {
            let inputs = (0..case.input_shapes.len())
                .map(|i| floats(&dir.join(format!("{}_input_{i}.f32", case.id))))
                .collect::<Result<Vec<_>>>()?;
            let bindings = [("x", bytes(&inputs[0]))];
            let mut baseline = build(case, &inputs, CpuLinearMode::Matmul, false)?;
            let mut candidate = build(case, &inputs, mode, false)?;
            let mut a = Vec::new();
            let mut b = Vec::new();
            for _ in 0..8 {
                baseline.run_f32_into(&bindings, "y", &mut a)?;
                candidate.run_f32_into(&bindings, "y", &mut b)?;
            }
            let mut base_ms = Vec::new();
            let mut candidate_ms = Vec::new();
            for i in 0..iterations {
                let run =
                    |g: &mut CpuGraph, y: &mut Vec<f32>, times: &mut Vec<f64>| -> Result<()> {
                        let started = Instant::now();
                        g.run_f32_into(&bindings, "y", y)?;
                        times.push(started.elapsed().as_secs_f64() * 1000.);
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
            let squared: f64 = a
                .iter()
                .zip(&b)
                .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                .sum();
            let max = a
                .iter()
                .zip(&b)
                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                .fold(0f64, f64::max);
            let mut diagnostic = build(case, &inputs, mode, true)?;
            diagnostic.run_f32_into(&bindings, "y", &mut b)?;
            rows.push(serde_json::json!({"case":case, "variant":mode, "baseline_ms":base_ms, "candidate_ms":candidate_ms,
            "fp32_error":{"rmse":(squared/a.len() as f64).sqrt(), "max_abs":max}, "diagnostic":diagnostic.timings(),
            "baseline_memory":baseline.memory_usage(), "candidate_memory":candidate.memory_usage()}));
        }
    }
    std::fs::write(
        &args[1],
        serde_json::to_vec_pretty(
            &serde_json::json!({"iterations":iterations, "operator_version":vocal_cpu::operator_version(), "rows":rows,
        "note":"6 threads; real constant FP32 projections; copies included; clocks collected separately; isolated operators, not pipeline latency"}),
        )?,
    )?;
    Ok(())
}
