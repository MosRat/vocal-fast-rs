//! Measure the same Runtime actor and allocator as the GUI with a private configuration.
use serde_json::json;
use std::{
    error::Error,
    path::PathBuf,
    time::{Duration, Instant},
};
use vocal_runtime::{RuntimeConfig, RuntimeContext};

#[cfg(windows)]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(windows)]
fn memory() -> serde_json::Value {
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        faults: u32,
        peak_working_set: usize,
        working_set: usize,
        peak_paged_pool: usize,
        paged_pool: usize,
        peak_nonpaged_pool: usize,
        nonpaged_pool: usize,
        pagefile: usize,
        peak_pagefile: usize,
        private_bytes: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut std::ffi::c_void,
            counters: *mut Counters,
            size: u32,
        ) -> i32;
    }
    let mut c = Counters {
        cb: std::mem::size_of::<Counters>() as u32,
        ..Default::default()
    };
    assert_ne!(
        unsafe {
            GetProcessMemoryInfo(
                GetCurrentProcess(),
                &mut c,
                std::mem::size_of::<Counters>() as u32,
            )
        },
        0
    );
    json!({"private_bytes":c.private_bytes,"peak_commit_bytes":c.peak_pagefile,"working_set_bytes":c.working_set,"peak_working_set_bytes":c.peak_working_set})
}

#[cfg(not(windows))]
fn memory() -> serde_json::Value {
    serde_json::Value::Null
}

fn mark(stages: &mut Vec<serde_json::Value>, name: &str, start: Instant) {
    let stage =
        json!({"stage":name,"elapsed_ms":start.elapsed().as_secs_f64()*1000.,"memory":memory()});
    println!("{stage}");
    stages.push(stage);
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args.len() != 3 {
        return Err("usage: model_memory_probe <runtime.json> <mono16k.wav> <report.json>".into());
    }
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter("vocal_directml=info,vocal_tensorrt::runtime=info")
        .init();
    let start = Instant::now();
    let mut stages = Vec::new();
    mark(&mut stages, "before_config", start);
    let config: RuntimeConfig = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let runtime = RuntimeContext::new(config.clone())?;
    mark(&mut stages, "before_load", start);
    runtime.load_model(config.model_path.clone().ok_or("model path missing")?)?;
    mark(&mut stages, "after_load", start);
    std::thread::sleep(Duration::from_millis(400));
    mark(&mut stages, "loaded_idle", start);
    let wave = vocal_audio::read_wav(&args[1])?;
    assert_eq!((wave.sample_rate, wave.channels), (16000, 1));
    let input = vocal_engine::pipeline::resample_linear(
        &wave.samples[..wave.samples.len().min(6 * 16000)],
        16000,
        config.input_format.sample_rate,
    );
    let input = input
        .iter()
        .flat_map(|v| std::iter::repeat_n(*v, config.input_format.channels as usize))
        .collect::<Vec<_>>();
    runtime.start_processing()?;
    let block =
        runtime.processing_batch_frames()?.max(4000) * config.input_format.channels as usize;
    let (generation, mut engine) = runtime.take_engine_for_worker()?;
    runtime.activate_worker(generation)?;
    let mut energy = 0.0_f64;
    let mut converted = Vec::new();
    for (sequence, samples) in input.chunks_exact(block).enumerate() {
        let output = engine.process_samples(
            config.input_format,
            samples,
            sequence as u64,
            config.conversion,
        )?;
        assert!(output.samples.iter().all(|v| v.is_finite()));
        energy += output
            .samples
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>();
        converted.extend(output.samples);
    }
    assert!(energy > 1e-10);
    let audio_path = args[2].with_extension("wav");
    if let Some(parent) = audio_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    vocal_audio::write_wav_pcm16(
        &audio_path,
        config.input_format.sample_rate,
        &converted
            .chunks_exact(config.input_format.channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
            .collect::<Vec<_>>(),
    )?;
    runtime.stop_processing()?;
    runtime.complete_worker(generation, engine, None)?;
    mark(&mut stages, "after_inference", start);
    runtime.unload_model()?;
    std::thread::sleep(Duration::from_millis(400));
    mark(&mut stages, "after_unload", start);
    runtime.load_model(config.model_path.clone().unwrap())?;
    std::thread::sleep(Duration::from_millis(400));
    mark(&mut stages, "second_loaded_idle", start);
    runtime.unload_model()?;
    std::thread::sleep(Duration::from_millis(400));
    mark(&mut stages, "second_unloaded", start);
    let report = json!({"config":config,"stages":stages,"output_energy":energy,"physical_devices":false,"allocator":"mimalloc","pid":std::process::id()});
    if let Some(parent) = args[2].parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args[2], serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
