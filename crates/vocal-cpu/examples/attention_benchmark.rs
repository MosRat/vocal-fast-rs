//! Paired FP32 attention math + bindings + output-layout benchmark on HF Q/K/V.
use std::{path::Path, time::Instant};
use vocal_cpu::{CpuAttentionConfig, CpuAttentionMode, CpuConfig, CpuGraph};
use vocal_graph::graph::{DType, GraphBuilder};

type Error = Box<dyn std::error::Error>;
#[derive(serde::Deserialize)]
struct Cases {
    cases: Vec<Case>,
}
#[derive(serde::Deserialize, serde::Serialize)]
struct Case {
    stem: String,
    samples: usize,
    layer: usize,
    shape: [usize; 4],
    scale: f32,
}

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
fn token_major(data: &[f32], shape: [usize; 4]) -> Vec<f32> {
    let [b, h, t, d] = shape;
    let mut out = vec![0.; data.len()];
    for batch in 0..b {
        for head in 0..h {
            for frame in 0..t {
                let src = ((batch * h + head) * t + frame) * d;
                let dst = ((batch * t + frame) * h + head) * d;
                out[dst..dst + d].copy_from_slice(&data[src..src + d]);
            }
        }
    }
    out
}
fn bytes(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}
fn build(c: &Case, attention: CpuAttentionConfig, profile: bool) -> Result<CpuGraph, Error> {
    let [b, h, t, d] = c.shape;
    let g = GraphBuilder::default();
    let q = g.transpose(&g.input("q", &[b, t, h, d], DType::F32), &[0, 2, 1, 3]);
    let k = g.transpose(&g.input("k", &[b, t, h, d], DType::F32), &[0, 2, 1, 3]);
    let v = g.transpose(&g.input("v", &[b, t, h, d], DType::F32), &[0, 2, 1, 3]);
    let scores = g.scale(&g.matmul(&q, &k, true), c.scale);
    let y = g.matmul(&g.softmax(&scores, 3), &v, false);
    g.mark(
        "context",
        &g.reshape(&g.transpose(&y, &[0, 2, 1, 3]), &[b, t, h * d]),
    );
    Ok(CpuGraph::compile(
        g,
        &CpuConfig {
            threads: 6,
            profile,
            attention,
            ..CpuConfig::default()
        },
    )?)
}
fn error(a: &[f32], b: &[f32]) -> serde_json::Value {
    let mut squared = 0.;
    let mut max = 0f64;
    for (a, b) in a.iter().zip(b) {
        let d = *a as f64 - *b as f64;
        squared += d * d;
        max = max.max(d.abs());
    }
    serde_json::json!({"rmse":(squared/a.len() as f64).sqrt(),"max_abs":max})
}
fn main() -> Result<(), Error> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: attention_benchmark cases-directory output.json iterations".into());
    }
    let directory = Path::new(&args[0]);
    let iterations: usize = args[2].parse()?;
    if iterations < 16 {
        return Err("at least 16 pairs required".into());
    }
    let cases: Cases =
        serde_json::from_slice(&std::fs::read(directory.join("attention-cases.json"))?)?;
    let mut rows = Vec::new();
    let variants = [
        (
            "dense_token",
            CpuAttentionConfig {
                token_major_output: true,
                ..CpuAttentionConfig::default()
            },
        ),
        (
            "online_4_32",
            CpuAttentionConfig {
                mode: CpuAttentionMode::Online,
                token_major_output: true,
                ..CpuAttentionConfig::default()
            },
        ),
        (
            "online_4_64",
            CpuAttentionConfig {
                mode: CpuAttentionMode::Online,
                key_tile: 64,
                token_major_output: true,
                ..CpuAttentionConfig::default()
            },
        ),
        (
            "online_8_64",
            CpuAttentionConfig {
                mode: CpuAttentionMode::Online,
                query_tile: 8,
                key_tile: 64,
                token_major_output: true,
                ..CpuAttentionConfig::default()
            },
        ),
    ];
    for case in cases.cases {
        let inputs =
            ["q", "k", "v"].map(|n| floats(&directory.join(format!("{}-{n}.f32", case.stem))));
        let [q, k, v] = inputs;
        let q = token_major(&q?, case.shape);
        let k = token_major(&k?, case.shape);
        let v = token_major(&v?, case.shape);
        let bindings = [("q", bytes(&q)), ("k", bytes(&k)), ("v", bytes(&v))];
        let reference = token_major(
            &floats(&directory.join(format!("{}-reference.f32", case.stem)))?,
            case.shape,
        );
        for (name, attention) in &variants {
            let mut baseline = build(
                &case,
                CpuAttentionConfig {
                    token_major_output: false,
                    ..CpuAttentionConfig::default()
                },
                false,
            )?;
            let mut candidate = build(&case, attention.clone(), false)?;
            let mut a = Vec::new();
            let mut b = Vec::new();
            for _ in 0..8 {
                baseline.run_f32_into(&bindings, "context", &mut a)?;
                candidate.run_f32_into(&bindings, "context", &mut b)?;
            }
            let mut base_ms = Vec::new();
            let mut actual_ms = Vec::new();
            for i in 0..iterations {
                let run = |g: &mut CpuGraph,
                           y: &mut Vec<f32>,
                           times: &mut Vec<f64>|
                 -> Result<(), Error> {
                    let start = Instant::now();
                    g.run_f32_into(&bindings, "context", y)?;
                    times.push(start.elapsed().as_secs_f64() * 1000.);
                    Ok(())
                };
                if i % 2 == 0 {
                    run(&mut baseline, &mut a, &mut base_ms)?;
                    run(&mut candidate, &mut b, &mut actual_ms)?;
                } else {
                    run(&mut candidate, &mut b, &mut actual_ms)?;
                    run(&mut baseline, &mut a, &mut base_ms)?;
                }
            }
            let mut diagnostic = build(&case, attention.clone(), true)?;
            diagnostic.run_f32_into(&bindings, "context", &mut b)?;
            rows.push(
                serde_json::json!({"case":case,"variant":name,"config":attention,
                "baseline_ms":base_ms,"candidate_ms":actual_ms,"fp32_error":error(&a,&b),
                "reference_error":error(&reference,&b),"baseline_memory":baseline.memory_usage(),
                "candidate_memory":candidate.memory_usage(),"diagnostic":diagnostic.timings()}),
            );
        }
    }
    let report = serde_json::json!({"note":"6 threads, 8 warmup pairs, alternating order; real HF FP32 Q/K/V, BTHD views; complete CpuGraph run includes binding copies, reshape and output copy; node timings collected separately",
        "operator_version":vocal_cpu::operator_version(),"iterations":iterations,"rows":rows});
    std::fs::write(&args[1], serde_json::to_vec_pretty(&report)?)?;
    println!(
        "measured {} attention pairs",
        report["rows"].as_array().unwrap().len()
    );
    Ok(())
}
