//! Real CPU integration probe. Uses a private config store and file audio.
use serde_json::json;
use std::{error::Error, path::PathBuf, time::Instant};
use vocal_audio::{read_wav, write_wav_pcm16};
use vocal_runtime::{ConfigStore, ConfigUpdateKind, RuntimeBackend, RuntimeConfig, RuntimeContext};
use vocal_types::{AudioFormat, LifecycleState};

fn replay(runtime: &RuntimeContext, input: &[f32]) -> Result<(Vec<f32>, Vec<f64>), Box<dyn Error>> {
    let snapshot = runtime.snapshot()?;
    runtime.start_processing()?;
    let (generation, mut engine) = runtime.take_engine_for_worker()?;
    runtime.activate_worker(generation)?;
    let frames = engine.preferred_batch_frames(AudioFormat::MONO_16K);
    let mut blocked = snapshot.config.clone();
    blocked.cpu.threads = 1;
    assert!(
        runtime
            .update_config(blocked, snapshot.config_revision)
            .is_err()
    );
    let mut audio = Vec::new();
    let mut durations = Vec::new();
    for (sequence, block) in input.chunks_exact(frames).enumerate() {
        let started = Instant::now();
        let output = engine.process_samples(
            AudioFormat::MONO_16K,
            block,
            sequence as u64,
            snapshot.config.conversion,
        )?;
        durations.push(started.elapsed().as_secs_f64() * 1000.);
        assert_eq!(output.samples.len(), block.len());
        assert!(output.samples.iter().all(|n| n.is_finite()));
        audio.extend(output.samples);
    }
    runtime.stop_processing()?;
    runtime.complete_worker(generation, engine, None)?;
    assert_eq!(runtime.snapshot()?.lifecycle, LifecycleState::Stopped);
    assert!(audio.iter().any(|n| n.abs() > 1e-6));
    Ok((audio, durations))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args.len() != 3 {
        return Err(
            "usage: cpu_runtime_probe <complete.vfrm> <input.wav> <report-directory>".into(),
        );
    }
    std::fs::create_dir_all(&args[2])?;
    let input = read_wav(&args[1])?;
    let mono: Vec<_> = input
        .samples
        .chunks_exact(usize::from(input.channels))
        .map(|frame| frame.iter().sum::<f32>() / f32::from(input.channels))
        .collect();
    let input = vocal_engine::pipeline::resample_linear(&mono, input.sample_rate, 16000);
    let input = &input[..input.len().min(6 * 16000)];
    let config = RuntimeConfig {
        backend: RuntimeBackend::RvcCpu,
        backend_auto: false,
        model_path: Some(args[0].display().to_string()),
        input_format: AudioFormat::MONO_16K,
        output_format: AudioFormat::MONO_16K,
        ..Default::default()
    };
    let store = ConfigStore::new(args[2].join("runtime.json"));
    store.save(&config)?;
    let runtime = RuntimeContext::new(config.clone())?;
    runtime.set_config_store(store.clone())?;
    let load = Instant::now();
    runtime.load_model(args[0].display().to_string())?;
    let load_ms = load.elapsed().as_secs_f64() * 1000.;
    let (audio, times) = replay(&runtime, input)?;
    write_wav_pcm16(args[2].join("cpu-auto.wav"), 16000, &audio)?;
    let before = runtime.snapshot()?;
    let mut controls = before.config.clone();
    controls.conversion.protect = 0.2;
    let saved = runtime.update_config(controls, before.config_revision)?;
    assert_eq!(saved.last_config_update, ConfigUpdateKind::HotReload);
    assert_eq!(
        saved.engine.as_ref().unwrap().processed_chunks,
        before.engine.as_ref().unwrap().processed_chunks
    );
    let (changed, _) = replay(&runtime, input)?;
    write_wav_pcm16(args[2].join("cpu-protect.wav"), 16000, &changed)?;
    let before = runtime.snapshot()?;
    let mut candidate = before.config.clone();
    candidate.cpu.threads = 2;
    candidate.cpu.pin_to_performance_cores = true;
    let rebuild = Instant::now();
    let saved = runtime.update_config(candidate, before.config_revision)?;
    let rebuild_ms = rebuild.elapsed().as_secs_f64() * 1000.;
    assert_eq!(saved.last_config_update, ConfigUpdateKind::RebuildEngine);
    assert_eq!(saved.engine.as_ref().unwrap().processed_chunks, 0);
    assert_eq!(store.load()?.unwrap().cpu, saved.config.cpu);
    let mut bad = saved.config.clone();
    bad.speaker_id = i32::MAX;
    assert!(runtime.update_config(bad, saved.config_revision).is_err());
    let preserved = runtime.snapshot()?;
    assert_eq!(preserved.config_revision, saved.config_revision);
    assert_eq!(preserved.config.cpu, saved.config.cpu);
    let (audio, two_thread_times) = replay(&runtime, input)?;
    write_wav_pcm16(args[2].join("cpu-two-threads.wav"), 16000, &audio)?;
    runtime.unload_model()?;
    assert!(runtime.snapshot()?.engine.is_none());
    let report = json!({
        "mode": "unpaced_file_replay", "device_io": false, "model": args[0], "input": args[1],
        "cpu": config.cpu, "resolved_threads": config.cpu.resolved_threads(),
        "load_ms": load_ms, "rebuild_ms": rebuild_ms, "chunk_ms": 512,
        "auto_chunk_processing_ms": times, "two_thread_chunk_processing_ms": two_thread_times,
        "checks": ["complete bundle loading", "finite audible output", "worker handoff", "active update rejected", "protect hot reload preserves engine", "CPU settings rebuild and persist", "invalid speaker retains engine and revision", "restart and unload"],
    });
    std::fs::write(
        args[2].join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
