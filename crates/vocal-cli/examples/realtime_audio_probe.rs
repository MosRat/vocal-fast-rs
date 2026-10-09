//! Pace real inference through the GUI's batch scheduler and SPSC queues.
//! File data stands in for devices; persisted GUI configuration is not edited.
use std::{
    error::Error,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use serde_json::json;
use vocal_audio::{
    AudioError, BlockDrainer, DuplexWorkerAdapter, RealtimeBufferConfig, StreamConfig, StreamIo,
    WorkerInputPolicy, WorkerSchedule, read_wav, write_wav_pcm16,
};
use vocal_runtime::{RuntimeConfig, RuntimeContext};
use vocal_types::AudioFormat;

fn percentiles(times: &mut [f64]) -> serde_json::Value {
    if times.is_empty() {
        return serde_json::Value::Null;
    }
    times.sort_by(f64::total_cmp);
    let at = |p: f64| times[((times.len() - 1) as f64 * p).round() as usize];
    json!({"p50": at(0.5), "p95": at(0.95), "p99": at(0.99), "max": times.last()})
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args.len() != 4 {
        return Err(
            "usage: realtime_audio_probe <runtime.json> <input.wav> <output.wav> <report.json>"
                .into(),
        );
    }
    let config: RuntimeConfig = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    config.validate()?;
    if config.output_format.channels != 1 {
        return Err("probe WAV writer requires mono logical output".into());
    }
    let input = read_wav(&args[1])?;
    let runtime = RuntimeContext::new(config.clone())?;
    runtime.load_model(config.model_path.clone().ok_or("missing model path")?)?;
    runtime.start_processing()?;
    let nominal = StreamConfig {
        format: config.input_format,
        block_frames: config.block_frames,
        queue_blocks: config.queue_blocks,
    };
    let buffering = RealtimeBufferConfig::for_batch(nominal, runtime.processing_batch_frames()?)?;
    let capture = nominal.for_endpoint(AudioFormat {
        sample_rate: input.sample_rate,
        channels: input.channels,
    })?;
    let render = nominal.for_endpoint(config.output_format)?;
    let (mut device, mut worker) = StreamIo::new_duplex(capture, render, buffering.queues)?.split();
    let mut adapter = DuplexWorkerAdapter::with_batch_frames(
        &worker,
        config.input_format,
        config.output_format,
        buffering.batch_frames,
    )?;
    let conversion_delay_micros = adapter.conversion_delay_micros();
    let (generation, mut engine) = runtime.take_engine_for_worker()?;
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    let controls = config.conversion;
    let worker_runtime = runtime.clone();
    let join = thread::spawn(move || {
        let mut times = Vec::new();
        let mut gaps = 0_u64;
        let result = (|| -> Result<(), AudioError> {
            engine
                .reset_stream()
                .map_err(|error| AudioError::Processing(error.to_string()))?;
            worker_runtime
                .activate_worker(generation)
                .map_err(|error| AudioError::Processing(error.to_string()))?;
            let schedule = WorkerSchedule {
                input_policy: WorkerInputPolicy::Fifo,
                timeout: Some(Duration::from_millis(20)),
            };
            let mut sequence = 0;
            while !thread_stop.load(Ordering::Acquire) {
                match adapter.process_batch_once(&mut worker, schedule, |format, samples, gap| {
                    if gap {
                        gaps += 1;
                        engine
                            .reset_stream()
                            .map_err(|error| AudioError::Processing(error.to_string()))?;
                    }
                    let started = Instant::now();
                    let output = engine
                        .process_samples(format, samples, sequence, controls)
                        .map_err(|error| AudioError::Processing(error.to_string()))?;
                    times.push(started.elapsed().as_secs_f64() * 1000.);
                    sequence += 1;
                    Ok(output.samples)
                }) {
                    Ok(_) | Err(AudioError::Timeout | AudioError::QueueEmpty) => {}
                    Err(error) => return Err(error),
                }
            }
            Ok(())
        })();
        (engine, times, gaps, result.map_err(|e| e.to_string()))
    });

    let period = capture.block_duration();
    let file_ticks = input.samples.len().div_ceil(capture.block_samples());
    // Feed trailing silence so the final partial model batch and converter
    // history are processed. This is included explicitly in the report/WAV.
    let tail_ticks = Duration::from_secs(1)
        .as_nanos()
        .div_ceil(period.as_nanos()) as usize;
    let mut capture_buffer = vec![0.; capture.block_samples()];
    let mut render_buffer = vec![0.; render.block_samples()];
    let mut drainer = BlockDrainer::new(render)?;
    let mut output = Vec::with_capacity((file_ticks + tail_ticks) * render.block_samples());
    let mut underflow_ticks = Vec::new();
    let mut late_ticks = 0;
    let started = Instant::now();
    for tick in 0..file_ticks + tail_ticks {
        let deadline = started + period.mul_f64(tick as f64);
        thread::sleep(deadline.saturating_duration_since(Instant::now()));
        if Instant::now().saturating_duration_since(deadline) > period {
            late_ticks += 1;
        }
        if join.is_finished() {
            break;
        }
        capture_buffer.fill(0.);
        let offset = tick * capture_buffer.len();
        if offset < input.samples.len() {
            let take = capture_buffer.len().min(input.samples.len() - offset);
            capture_buffer[..take].copy_from_slice(&input.samples[offset..offset + take]);
        }
        match device.input.try_push(&capture_buffer) {
            Ok(_) | Err(AudioError::QueueFull) => {}
            Err(error) => return Err(error.into()),
        }
        let report = drainer.drain(&mut render_buffer, &mut device.output)?;
        if report.underflow_samples > 0 {
            underflow_ticks.push(tick);
        }
        output.extend_from_slice(&render_buffer);
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.;
    stop.store(true, Ordering::Release);
    let (engine, mut times, gaps, result) = join.join().map_err(|_| "probe worker panicked")?;
    let metrics = engine.metrics();
    runtime.complete_worker(
        generation,
        engine,
        result
            .as_ref()
            .err()
            .map(|e| ("probe.worker".into(), e.clone())),
    )?;
    runtime.stop_processing()?;
    result.map_err(|error| format!("real-time model worker: {error}"))?;
    if output.iter().any(|value| !value.is_finite()) {
        return Err("non-finite output".into());
    }
    if let Some(parent) = args[2].parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_wav_pcm16(&args[2], render.format.sample_rate, &output)?;
    let stats = device.stats();
    let report = json!({
        "mode": "paced_real_inference_with_gui_batch_scheduler", "physical_devices": false,
        "neural_inference": true, "input": args[1], "output": args[2],
        "runtime_config": config, "capture_rate": capture.format.sample_rate,
        "capture_channels": capture.format.channels, "render_rate": render.format.sample_rate,
        "device_period_ms": period.as_secs_f64() * 1000., "processing_batch_frames": buffering.batch_frames,
        "input_queue_blocks": buffering.queues.input_blocks, "output_queue_blocks": buffering.queues.output_blocks,
        "render_prefill_blocks": buffering.queues.render_prefill_blocks,
        "conversion_delay_micros": conversion_delay_micros, "model_batches": times.len(),
        "batch_processing_ms": percentiles(&mut times), "engine_metrics": metrics,
        "input_seconds": input.frames() as f64 / input.sample_rate as f64,
        "output_seconds": output.len() as f64 / render.format.sample_rate as f64,
        "trailing_capture_silence_seconds": tail_ticks as f64 * period.as_secs_f64(),
        "elapsed_ms": elapsed_ms, "late_device_ticks": late_ticks,
        "input_dropped_full": stats.input.dropped_full, "input_dropped_stale": stats.input.dropped_stale,
        "output_dropped_full": stats.output.dropped_full, "model_discontinuities": gaps,
        "input_high_water_blocks": stats.input.high_water_blocks, "output_high_water_blocks": stats.output.high_water_blocks,
        "underflow_samples": drainer.stats().underflow_samples, "underflow_ticks": underflow_ticks,
    });
    if let Some(parent) = args[3].parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args[3], serde_json::to_vec_pretty(&report)?)?;
    println!(
        "{}",
        json!({"report": args[3], "underflow_samples": report["underflow_samples"],
        "input_dropped_full": report["input_dropped_full"], "input_dropped_stale": report["input_dropped_stale"],
        "output_dropped_full": report["output_dropped_full"], "batch_processing_ms": report["batch_processing_ms"]})
    );
    Ok(())
}
