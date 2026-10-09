//! Exercise real backends across idle control updates and worker ownership handoffs.
//! Uses a private config store and file input; never edits persisted GUI settings.
use std::{error::Error, path::PathBuf};

use serde_json::json;
use vocal_audio::{read_wav, write_wav_pcm16};
use vocal_runtime::{ConfigStore, ConfigUpdateKind, RuntimeBackend, RuntimeConfig, RuntimeContext};
use vocal_types::{AudioFormat, LifecycleState};

fn process_session(runtime: &RuntimeContext, input: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
    let config = runtime.config()?;
    runtime.start_processing()?;
    let frames = runtime.processing_batch_frames()?.max(4000);
    let samples = frames * config.input_format.channels as usize;
    let (generation, mut engine) = runtime.take_engine_for_worker()?;
    runtime.activate_worker(generation)?;
    // Updates remain forbidden even when the worker, rather than the actor,
    // owns the engine. A rejected update must not poison that engine.
    let snapshot = runtime.snapshot()?;
    assert!(
        runtime
            .update_config(config.clone(), snapshot.config_revision)
            .is_err()
    );
    let mut audio = Vec::new();
    let result = (|| -> Result<(), Box<dyn Error>> {
        for (sequence, block) in input.chunks_exact(samples).enumerate() {
            let output = engine.process_samples(
                config.input_format,
                block,
                sequence as u64,
                config.conversion,
            )?;
            assert_eq!(output.samples.len(), block.len());
            assert!(output.samples.iter().all(|sample| sample.is_finite()));
            audio.extend(output.samples);
        }
        Ok(())
    })();
    runtime.stop_processing()?;
    runtime.complete_worker(
        generation,
        engine,
        result
            .as_ref()
            .err()
            .map(|e| ("probe".into(), e.to_string())),
    )?;
    result?;
    assert_eq!(runtime.snapshot()?.lifecycle, LifecycleState::Stopped);
    Ok(audio)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args.len() != 3 {
        return Err(
            "usage: idle_reload_probe <runtime.json> <mono16k.wav> <report-directory>".into(),
        );
    }
    let config: RuntimeConfig = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    assert_eq!(config.input_format, AudioFormat::MONO_16K);
    let wave = read_wav(&args[1])?;
    assert_eq!((wave.sample_rate, wave.channels), (16000, 1));
    let input = &wave.samples[..wave.samples.len().min(6 * 16000)];
    std::fs::create_dir_all(&args[2])?;
    let store = ConfigStore::new(args[2].join("runtime.json"));
    store.save(&config)?;
    let runtime = RuntimeContext::new(config.clone())?;
    runtime.set_config_store(store.clone())?;
    runtime.load_model(config.model_path.clone().ok_or("missing model path")?)?;
    let mut passes = Vec::new();
    let mut reference = None;
    let mut disabled = None;
    for protect in [0.33, 0.0, 0.5, 1.0, 0.33] {
        let before = runtime.snapshot()?;
        let before_chunks = before.engine.as_ref().unwrap().processed_chunks;
        let mut next = before.config.clone();
        next.conversion.protect = protect;
        let saved = runtime.update_config(next, before.config_revision)?;
        assert!(matches!(
            saved.last_config_update,
            ConfigUpdateKind::Unchanged | ConfigUpdateKind::HotReload
        ));
        assert_eq!(
            saved.engine.as_ref().unwrap().processed_chunks,
            before_chunks
        );
        assert_eq!(store.load()?.unwrap().conversion.protect, protect);
        let started = std::time::Instant::now();
        let audio = process_session(&runtime, input)?;
        let energy: f64 = audio.iter().map(|v| f64::from(*v).powi(2)).sum();
        assert!(energy > 1e-10, "no publishable audio after warmup");
        let difference = reference.as_ref().map(|old: &Vec<f32>| {
            old.iter()
                .zip(&audio)
                .map(|(a, b)| (*a - *b).abs())
                .fold(0.0_f32, f32::max)
        });
        if reference.is_none() {
            reference = Some(audio.clone());
        }
        if protect == 0.5 {
            disabled = Some(audio.clone());
        }
        if protect == 1.0 {
            let difference = disabled
                .as_ref()
                .unwrap()
                .iter()
                .zip(&audio)
                .map(|(a, b)| (*a - *b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                difference < 1e-4,
                "disabled protection differs at 0.5 and 1: {difference}"
            );
        }
        let after = runtime.snapshot()?;
        assert!(after.engine.as_ref().unwrap().processed_chunks > before_chunks);
        assert_eq!(after.engine.as_ref().unwrap().failures, 0);
        write_wav_pcm16(
            args[2].join(format!("protect-{protect:.2}-{}.wav", passes.len())),
            16000,
            &audio,
        )?;
        passes.push(json!({"protect":protect,"update":saved.last_config_update,"chunks_before":before_chunks,"chunks_after":after.engine.as_ref().unwrap().processed_chunks,"energy":energy,"max_difference_from_first":difference,"elapsed_ms":started.elapsed().as_secs_f64()*1000.}));
    }

    let mut checks = Vec::new();
    let mut control_passes = Vec::new();
    for (name, index, rms, pitch, format) in [
        ("index off", 0.0, 1.0, 0.0, AudioFormat::MONO_16K),
        ("index on", 0.7, 1.0, 0.0, AudioFormat::MONO_16K),
        ("RMS mix", 0.5, 0.3, 0.0, AudioFormat::MONO_16K),
        (
            "format 44.1k stereo",
            0.5,
            1.0,
            0.0,
            AudioFormat {
                sample_rate: 44100,
                channels: 2,
            },
        ),
        (
            "format restore + pitch",
            0.5,
            1.0,
            2.0,
            AudioFormat::MONO_16K,
        ),
    ] {
        let before = runtime.snapshot()?;
        let mut candidate = before.config;
        candidate.conversion.index_ratio = index;
        candidate.conversion.rms_mix_rate = rms;
        candidate.conversion.pitch_semitones = pitch;
        candidate.input_format = format;
        candidate.output_format = format;
        let saved = runtime.update_config(candidate, before.config_revision)?;
        assert_eq!(saved.last_config_update, ConfigUpdateKind::HotReload);
        assert_eq!(
            saved.engine.as_ref().unwrap().processed_chunks,
            before.engine.as_ref().unwrap().processed_chunks
        );
        let input = vocal_engine::pipeline::resample_linear(input, 16000, format.sample_rate);
        let input = input
            .iter()
            .flat_map(|v| std::iter::repeat_n(*v, format.channels as usize))
            .collect::<Vec<_>>();
        let audio = process_session(&runtime, &input)?;
        assert!(audio.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() > 1e-10);
        control_passes.push(json!({"name":name,"format":format,"index":index,"rms":rms,"pitch":pitch,"engine":runtime.snapshot()?.engine}));
    }
    checks.push(
        "index, RMS, supported pitch and 44.1k stereo/16k mono edits preserve the idle engine",
    );
    let original = runtime.snapshot()?;
    if original.config.backend == RuntimeBackend::TensorRtRtx {
        // Invalid controls fail at Save while the old idle engine remains usable.
        let mut next = original.config.clone();
        next.conversion.pitch_semitones = f32::NAN;
        assert!(
            runtime
                .update_config(next, original.config_revision)
                .is_err()
        );
        let mut next = original.config.clone();
        next.tensorrt.artifacts_path =
            Some(args[2].join("missing-pipeline.json").display().to_string());
        assert!(
            runtime
                .update_config(next, original.config_revision)
                .is_err()
        );
        assert_eq!(
            runtime.snapshot()?.config_revision,
            original.config_revision
        );
        checks.push("invalid pitch and missing pipeline preserve prior engine/configuration");
    }

    let mut pitch_passes = Vec::new();
    let mut matched_reference: Option<Vec<f32>> = None;
    for (enabled, target_hz, manual) in [
        (true, 255., 0.),
        (false, 255., 0.),
        (true, 155., 0.),
        (true, 255., 2.),
        (true, 255., 0.),
    ] {
        let before = runtime.snapshot()?;
        let mut candidate = before.config.clone();
        candidate.conversion.auto_pitch = vocal_types::AutoPitchConfig { enabled, target_hz };
        candidate.conversion.pitch_semitones = manual;
        let saved = runtime.update_config(candidate, before.config_revision)?;
        assert!(matches!(
            saved.last_config_update,
            ConfigUpdateKind::Unchanged | ConfigUpdateKind::HotReload
        ));
        assert_eq!(
            saved.engine.as_ref().unwrap().processed_chunks,
            before.engine.as_ref().unwrap().processed_chunks
        );
        let audio = process_session(&runtime, input)?;
        let difference = matched_reference.as_ref().map(|reference| {
            reference
                .iter()
                .zip(&audio)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        });
        if pitch_passes.is_empty() {
            matched_reference = Some(audio.clone());
        }
        if pitch_passes.len() == 4 {
            assert!(
                difference.unwrap() < 1e-4,
                "pitch state leaked across restarts"
            );
        }
        let persisted = store.load()?.unwrap();
        assert_eq!(
            persisted.conversion.auto_pitch,
            saved.config.conversion.auto_pitch
        );
        write_wav_pcm16(
            args[2].join(format!("pitch-match-{}.wav", pitch_passes.len())),
            16000,
            &audio,
        )?;
        pitch_passes.push(
            json!({"enabled":enabled,"target_hz":target_hz,"manual":manual,
            "update":saved.last_config_update,"max_difference_from_first":difference,
            "engine":runtime.snapshot()?.engine}),
        );
    }
    checks.push("automatic pitch toggle/target/manual edits preserve idle engine; restart recalibrates deterministically");

    // Persistence failure must discard a prepared replacement, including TRT
    // execution contexts. The same revision remains valid for a later retry.
    let before = runtime.snapshot()?;
    let mut next = before.config.clone();
    next.input_format = AudioFormat::MONO_16K;
    next.output_format = AudioFormat::MONO_16K;
    if next.backend == RuntimeBackend::TensorRtRtx {
        next.tensorrt.graph_cache_size += 1;
    } else {
        next.conversion.protect = 0.0;
    }
    std::fs::remove_file(store.path())?;
    std::fs::create_dir(store.path())?;
    let saved = runtime.update_config(next.clone(), before.config_revision);
    assert!(saved.is_err());
    let failed = runtime.snapshot()?;
    assert_eq!(failed.config_revision, before.config_revision);
    assert_eq!(
        failed.engine.as_ref().unwrap().processed_chunks,
        before.engine.as_ref().unwrap().processed_chunks
    );
    assert_eq!(failed.lifecycle, before.lifecycle);
    std::fs::remove_dir(store.path())?;
    let retry = runtime.update_config(next, before.config_revision)?;
    checks.push("save failure preserves engine/config/revision and retry succeeds");
    let audio = process_session(&runtime, input)?;
    assert!(audio.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() > 1e-10);
    let report = json!({"backend":config.backend,"physical_devices":false,"real_backend_execution":true,"input":args[1],"passes":passes,"control_passes":control_passes,"pitch_passes":pitch_passes,"checks":checks,"retry_update":retry.last_config_update,"final":runtime.snapshot()?});
    std::fs::write(
        args[2].join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "{}",
        json!({"backend":config.backend,"passes":passes.len(),"checks":checks,"report":args[2].join("report.json")})
    );
    Ok(())
}
