//! Replay measured callback durations through the real SPSC/duplex scheduler.
//! This tests software pacing; it does not open devices or run neural inference.
use std::{
    error::Error,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use serde_json::{Value, json};
use vocal_audio::{
    AudioError, BlockDrainer, DuplexWorkerAdapter, StreamConfig, StreamIo, StreamQueueConfig,
    WorkerInputPolicy, WorkerSchedule,
};
use vocal_runtime::RuntimeConfig;

fn run(config: &RuntimeConfig, times: Vec<f64>, seconds: f64) -> Result<Value, Box<dyn Error>> {
    let stream = StreamConfig {
        format: config.input_format,
        block_frames: config.block_frames,
        queue_blocks: config.queue_blocks,
    };
    let (mut device, mut worker) = StreamIo::new_with_queue_config(
        stream,
        StreamQueueConfig::new(config.queue_blocks, config.queue_blocks.clamp(1, 2)),
    )?
    .split();
    let silence = vec![0.; stream.block_samples()];
    for _ in 0..config.queue_blocks.clamp(1, 2) {
        worker.output.try_push(&silence)?;
    }
    let mut adapter =
        DuplexWorkerAdapter::with_formats(&worker, config.input_format, config.output_format)?;
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = Arc::clone(&stop);
    let join = thread::spawn(move || -> Result<usize, String> {
        let mut processed = 0;
        let schedule = WorkerSchedule {
            input_policy: WorkerInputPolicy::Latest,
            timeout: Some(Duration::from_millis(20)),
        };
        while !worker_stop.load(Ordering::Acquire) {
            match adapter.process_once(&mut worker, schedule, |_, input| {
                let ms = times.get(processed).copied().unwrap_or(0.);
                if ms > 0. {
                    thread::sleep(Duration::from_secs_f64(ms / 1000.));
                }
                processed += 1;
                Ok(input.to_vec())
            }) {
                Ok(_)
                | Err(AudioError::Timeout | AudioError::QueueEmpty | AudioError::QueueFull) => {}
                Err(error) => return Err(error.to_string()),
            }
        }
        Ok(processed)
    });
    let period = Duration::from_secs_f64(
        config.block_frames as f64 / config.input_format.sample_rate as f64,
    );
    let count = (seconds / period.as_secs_f64()).round() as usize;
    let input = vec![0.25; stream.block_samples()];
    let mut output = silence;
    let mut drainer = BlockDrainer::new(stream)?;
    let mut underruns = Vec::new();
    let mut late_ticks = 0;
    let started = Instant::now();
    for tick in 0..count {
        let deadline = started + period.mul_f64(tick as f64);
        thread::sleep(deadline.saturating_duration_since(Instant::now()));
        if Instant::now().saturating_duration_since(deadline) > period {
            late_ticks += 1;
        }
        match device.input.try_push(&input) {
            Ok(_) | Err(AudioError::QueueFull) => {}
            Err(error) => return Err(error.into()),
        }
        let report = drainer.drain(&mut output, &mut device.output)?;
        if report.underflow_samples > 0 {
            underruns.push(tick as f64 * period.as_secs_f64() * 1000.);
        }
    }
    stop.store(true, Ordering::Release);
    let processed = join.join().map_err(|_| "pacing worker panicked")??;
    let stats = device.stats();
    let underflow_ms = drainer.stats().underflow_samples as f64 * 1000.
        / config.input_format.sample_rate as f64
        / config.input_format.channels as f64;
    Ok(
        json!({"seconds":seconds,"device_ticks":count,"processed_blocks":processed,
        "input_dropped_full":stats.input.dropped_full,"input_dropped_stale":stats.input.dropped_stale,
        "input_high_water_blocks":stats.input.high_water_blocks,"output_high_water_blocks":stats.output.high_water_blocks,
        "underflow_blocks":underruns.len(),"underflow_ms":underflow_ms,"underflow_times_ms":underruns,
        "device_ticks_late_by_more_than_one_period":late_ticks}),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: replay_queue_timing <offline_audio_probe.json> <report.json>".into());
    }
    let trace: Value = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let config: RuntimeConfig = serde_json::from_value(trace["runtime_config"].clone())?;
    config.validate()?;
    if config.input_format != config.output_format {
        return Err("timing replay requires matching formats".into());
    }
    let times: Vec<_> = trace["callback_trace"]
        .as_array()
        .ok_or("missing callback trace")?
        .iter()
        .map(|c| c["processing_ms"].as_f64().ok_or("missing processing time"))
        .collect::<Result<_, _>>()?;
    if times.iter().any(|t| !t.is_finite() || *t < 0.) {
        return Err("invalid processing times".into());
    }
    let seconds = 10.;
    let baseline = run(&config, Vec::new(), seconds)?;
    let measured_timing = run(&config, times, seconds)?;
    let result = json!({"mode":"real_spsc_queues_with_recorded_processing_sleep",
        "physical_devices":false,"neural_inference":false,"runtime_config":config,
        "output_prefill_blocks":config.queue_blocks.clamp(1, 2),
        "baseline":baseline,"recorded_timing":measured_timing});
    std::fs::write(&args[1], serde_json::to_vec_pretty(&result)?)?;
    println!(
        "{}",
        json!({"baseline":result["baseline"],"recorded_timing":{
        "underflow_blocks":result["recorded_timing"]["underflow_blocks"],
        "underflow_ms":result["recorded_timing"]["underflow_ms"],
        "input_dropped_full":result["recorded_timing"]["input_dropped_full"],
        "input_dropped_stale":result["recorded_timing"]["input_dropped_stale"]}})
    );
    Ok(())
}
