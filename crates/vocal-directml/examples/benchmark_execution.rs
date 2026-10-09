//! Paired production comparisons with identical noise and bounded GPU memory.
//! benchmark_execution <sources.json> <wave.f32> <report.json> [iterations]
use serde_json::{Value, json};
use std::{error::Error, time::Instant};
use vocal_directml::{
    DirectMlControls, DirectMlPipeline, DirectMlPipelineOutput, DirectMlSources, ExecutionConfig,
    IoBuffers,
};

fn summary(values: &[f64]) -> Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    json!({"mean": values.iter().sum::<f64>() / values.len() as f64,
        "p50": sorted[sorted.len()/2], "p95": sorted[(sorted.len()*95/100).min(sorted.len()-1)]})
}
#[derive(Default)]
struct Samples {
    gpu: Vec<f64>,
    host: Vec<f64>,
    upload: Vec<f64>,
    readback: Vec<f64>,
}
impl Samples {
    fn push(&mut self, output: &DirectMlPipelineOutput) {
        self.gpu.push(output.gpu_ms);
        self.host.push(output.total_ms);
        self.upload.push(output.debug.gpu_timings.upload_ms);
        self.readback.push(output.debug.gpu_timings.readback_ms);
    }
    fn report(&self) -> Value {
        json!({"gpu_ms": self.gpu, "host_ms": self.host,
            "gpu_summary_ms": summary(&self.gpu), "host_summary_ms": summary(&self.host),
            "upload_summary_ms": summary(&self.upload), "readback_summary_ms": summary(&self.readback)})
    }
}
fn metric(reference: &[f32], actual: &[f32]) -> Value {
    assert_eq!(reference.len(), actual.len());
    let (mut square, mut dot, mut aa, mut bb, mut max) = (0., 0., 0., 0., 0f64);
    for (&a, &b) in reference.iter().zip(actual) {
        let (a, b) = (a as f64, b as f64);
        square += (a - b).powi(2);
        max = max.max((a - b).abs());
        dot += a * b;
        aa += a * a;
        bb += b * b;
    }
    json!({"rmse": (square/reference.len() as f64).sqrt(), "max_abs": max,
        "cosine": if aa*bb > 0. { dot/(aa*bb).sqrt() } else { f64::from(max == 0.) }})
}
fn write_audio(report: &str, window: usize, name: &str, audio: &[f32]) -> std::io::Result<()> {
    let path = std::path::Path::new(report).with_file_name(format!("{window}-{name}.f32"));
    std::fs::write(
        path,
        audio
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 3 {
        return Err("benchmark_execution <sources> <wave.f32> <report> [iterations]".into());
    }
    let mut sources = DirectMlSources::load(&args[0])?;
    sources.config.debug_outputs = false;
    sources.config.debug_layer = false;
    let input = std::fs::read(&args[1])?;
    if input.len() % 4 != 0 {
        return Err("input must contain complete F32 samples".into());
    }
    let wave = input
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    if wave.len() != sources.config.window_samples {
        return Err("wave length must match the fixed window".into());
    }
    let iterations = args
        .get(3)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(40usize)
        .max(1);
    let variants = [
        ("default", ExecutionConfig::default()),
        (
            "pooled",
            ExecutionConfig {
                io_buffers: IoBuffers::Pooled,
                ..Default::default()
            },
        ),
        (
            "views",
            ExecutionConfig {
                contiguous_views: true,
                io_buffers: IoBuffers::Pooled,
                ..Default::default()
            },
        ),
        (
            "gemm_bias",
            ExecutionConfig {
                fuse_gemm_bias: true,
                io_buffers: IoBuffers::Pooled,
                ..Default::default()
            },
        ),
    ];
    sources.config.execution = ExecutionConfig::legacy();
    let mut baseline = DirectMlPipeline::from_sources(&sources)?;
    let baseline_metadata = json!({"name": "legacy", "config": sources.config,
        "memory": baseline.memory(), "build_ms": baseline.build_ms});
    let controls = DirectMlControls::new(baseline.frames);
    let mut reports = Vec::new();
    for (name, execution) in variants {
        sources.config.execution = execution;
        let started = Instant::now();
        let mut candidate = DirectMlPipeline::from_sources(&sources)?;
        eprintln!(
            "{name}: compiled in {:.2}s; {:?}",
            started.elapsed().as_secs_f64(),
            candidate.memory()
        );
        baseline.reset()?;
        candidate.reset()?;
        let (mut reference, mut actual) = (Samples::default(), Samples::default());
        let (mut worst_rmse, mut worst_max) = (0f64, 0f64);
        let mut last = Value::Null;
        for step in 0..iterations + 4 {
            // Keep exactly two resident graphs and alternate order. More full
            // 44960 graphs can exceed the WDDM memory budget and mistake
            // residency/eviction overhead for operator performance.
            let (a, b) = if step % 2 == 0 {
                let a = baseline.process(&wave, &controls)?;
                (a, candidate.process(&wave, &controls)?)
            } else {
                let b = candidate.process(&wave, &controls)?;
                (baseline.process(&wave, &controls)?, b)
            };
            if step >= 4 {
                reference.push(&a);
                actual.push(&b);
            }
            last = metric(&a.audio, &b.audio);
            worst_rmse = worst_rmse.max(last["rmse"].as_f64().unwrap());
            worst_max = worst_max.max(last["max_abs"].as_f64().unwrap());
            if step == iterations + 3 {
                write_audio(&args[2], sources.config.window_samples, "legacy", &a.audio)?;
                write_audio(&args[2], sources.config.window_samples, name, &b.audio)?;
            }
        }
        let mut report = actual.report();
        report["name"] = json!(name);
        report["config"] = json!(sources.config);
        report["memory"] = json!(candidate.memory());
        report["build_ms"] = json!(candidate.build_ms);
        report["paired_reference"] = reference.report();
        report["parity_last_vs_legacy"] = last;
        report["worst_rmse_vs_legacy"] = json!(worst_rmse);
        report["worst_max_abs_vs_legacy"] = json!(worst_max);
        eprintln!(
            "{name}: GPU {} vs {}; host {}; parity {}",
            report["gpu_summary_ms"],
            report["paired_reference"]["gpu_summary_ms"],
            report["host_summary_ms"],
            report["parity_last_vs_legacy"]
        );
        reports.push(report);
        // Release weights, descriptors and scratch before the next compile.
        // The device pool holds weak references only.
        drop(candidate);
    }
    std::fs::write(
        &args[2],
        serde_json::to_vec_pretty(&json!({"warmup": 4, "iterations": iterations,
        "measurement": "serial alternating pairs; at most two resident graphs; no diagnostic downloads; identical noise per iteration",
        "baseline": baseline_metadata, "variants": reports}))?,
    )?;
    Ok(())
}
