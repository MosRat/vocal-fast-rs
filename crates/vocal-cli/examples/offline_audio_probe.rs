//! Replay the saved runtime engine at its actual public callback format.
//! No capture/render device is opened and no persisted configuration is edited.
use std::{error::Error, path::PathBuf, time::Instant};

use serde_json::json;
use vocal_audio::{read_wav, write_wav_pcm16};
use vocal_runtime::{RuntimeConfig, RuntimeContext};

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * percentile).round() as usize]
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args.len() != 4 {
        return Err(
            "usage: offline_audio_probe <runtime.json> <input.wav> <output.wav> <report.json>"
                .into(),
        );
    }
    let config: RuntimeConfig = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    config.validate()?;
    let input = read_wav(&args[1])?;
    if input.sample_rate != config.input_format.sample_rate
        || input.channels != config.input_format.channels
        || input.channels != 1
        || config.input_format != config.output_format
    {
        return Err("probe requires matching mono WAV and runtime input/output formats".into());
    }
    let model = config
        .model_path
        .clone()
        .ok_or("configuration has no model")?;
    let load_started = Instant::now();
    let runtime = RuntimeContext::new(config.clone())?;
    runtime.load_model(model)?;
    runtime.start_processing()?;
    let (generation, mut engine) = runtime.take_engine_for_worker()?;
    runtime.activate_worker(generation)?;
    let preparation_ms = load_started.elapsed().as_secs_f64() * 1000.;
    let block_samples = config.block_frames * usize::from(input.channels);
    let callback_ms = config.block_frames as f64 * 1000. / input.sample_rate as f64;
    let mut output = Vec::with_capacity(input.samples.len());
    let mut callbacks = Vec::new();
    let started = Instant::now();
    for (sequence, block) in input.samples.chunks(block_samples).enumerate() {
        let begin = Instant::now();
        let result = engine.process_samples(
            config.input_format,
            block,
            sequence as u64,
            config.conversion,
        )?;
        let elapsed_ms = begin.elapsed().as_secs_f64() * 1000.;
        if result.samples.len() != block.len() || result.samples.iter().any(|x| !x.is_finite()) {
            return Err(format!("invalid callback output at {sequence}").into());
        }
        output.extend_from_slice(&result.samples);
        callbacks.push(
            json!({"sequence": sequence, "input_time_ms": sequence as f64 * callback_ms,
            "processing_ms": elapsed_ms, "samples": block.len()}),
        );
    }
    let processing_ms = started.elapsed().as_secs_f64() * 1000.;
    let metrics = engine.metrics();
    runtime.complete_worker(generation, engine, None)?;
    runtime.stop_processing()?;
    if let Some(parent) = args[2].parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_wav_pcm16(&args[2], input.sample_rate, &output)?;
    let mut times: Vec<_> = callbacks
        .iter()
        .map(|c| c["processing_ms"].as_f64().unwrap())
        .collect();
    times.sort_by(f64::total_cmp);
    let mut bursts: Vec<_> = times
        .iter()
        .copied()
        .filter(|t| *t >= callback_ms)
        .collect();
    bursts.sort_by(f64::total_cmp);
    let report = json!({
        "mode": "unpaced_file_replay_of_runtime_callbacks", "device_io": false,
        "input": args[1], "output": args[2], "runtime_config": config,
        "input_samples": input.samples.len(), "output_samples": output.len(),
        "input_seconds": input.frames() as f64 / input.sample_rate as f64,
        "callback_ms": callback_ms, "preparation_ms": preparation_ms, "processing_ms": processing_ms,
        "engine_metrics": metrics, "callbacks": callbacks.len(),
        "callback_processing_ms": {"p50": percentile(&times, 0.5), "p95": percentile(&times, 0.95),
            "p99": percentile(&times, 0.99), "max": times.last()},
        "callbacks_over_period": bursts.len(),
        "over_period_processing_ms": (!bursts.is_empty()).then(|| json!({
            "p50": percentile(&bursts, 0.5), "p95": percentile(&bursts, 0.95),
            "p99": percentile(&bursts, 0.99), "max": bursts.last()})),
        "nominal_output_ring_ms": callback_ms * config.queue_blocks.clamp(1, 2) as f64,
        "callback_trace": callbacks,
    });
    if let Some(parent) = args[3].parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args[3], serde_json::to_vec_pretty(&report)?)?;
    println!(
        "{}",
        json!({"output": args[2], "report": args[3], "processing_ms": processing_ms,
        "callback_processing_ms": report["callback_processing_ms"],
        "callbacks_over_period": report["callbacks_over_period"],
        "over_period_processing_ms": report["over_period_processing_ms"]})
    );
    Ok(())
}
