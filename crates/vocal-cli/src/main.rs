use std::{env, fs, path::PathBuf, process::ExitCode};
#[cfg(feature = "tensorrt-rtx")]
use std::{io::Read, time::Instant};
use vocal_audio::{AudioRing, StreamConfig, read_wav, resample_linear, write_wav_pcm16};
#[cfg(not(feature = "tensorrt-rtx"))]
use vocal_engine::tensorrt::{TensorRtCompiler, TrtExecCompiler};
use vocal_engine::tensorrt::{TensorRtGraphOptions, build_rvc_graph};
use vocal_engine::{
    BackendKind, EngineConfig, InferenceEngine,
    offline::{EngineOfflinePipeline, OfflineRunConfig, run_offline},
};
use vocal_model::{
    ACOUSTIC_TENSOR_SCHEMA, load_cache, load_directory, load_manifest, load_rvc_directory,
    load_rvc_onnx, load_rvc_pth, write_cache,
};
use vocal_runtime::{RuntimeBackend, RuntimeConfig, RuntimeContext};
use vocal_types::{AudioChunk, AudioFormat, ModelDescriptor};

// Keep allocation policy at the executable boundary. Library crates remain
// allocator-neutral so embedders can choose their own heap implementation.
#[cfg(windows)]
#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod cli_config;
mod cli_error;
mod cli_logging;
mod cli_state;

use cli_config::CliConfig;
use cli_error::CliResult;
use cli_state::CliStatus;

fn usage() {
    eprintln!(
        "vocal-fast baseline\n\nCommands:\n  doctor\n  model inspect <manifest.json>\n  model inspect-cache <model.vfr>\n  model pack <model-dir> <manifest.json> <output.vfr>\n  model pack-rvc <model-dir> <manifest.json> <output.vfr>\n  model convert-pth <model-dir> <manifest.json> <checkpoint.pth> <output.vfr>\n  model convert-onnx <model-dir> <manifest.json> <model.onnx> <output.vfr>\n  model build-graph <model.vfr> <output.graph.json>\n  model compile-trt <model.vfr> <output.plan> [source.onnx]\n  model compile-trt-pipeline <model.vfr> <hubert.onnx> <rmvpe.onnx> <output-dir>\n  model test-trt-pipeline <pipeline.json> <model.vfr> <input.wav|input.f32>\n  model run-trt-pipeline <pipeline.json> <model.vfr> <input.wav|input.f32> <output.wav>\n  model test-pipeline <passthrough|cpu|tensorrt> <model.vfr> <input.wav|input.f32> [pipeline.json]\n  model run-pipeline <passthrough|cpu|tensorrt> <model.vfr> <input.wav|input.f32> <output.wav> [pipeline.json]\n  model profile-trt-pipeline <pipeline.json> <model.vfr> <input.wav|input.f32>\n  model compare-trt-pipeline <pipeline.json> <model.vfr> <input.wav|input.f32> [iterations]\n  model test-trt <output.plan> [model.vfr]\n  demo [blocks]\n"
    );
    eprintln!("  model benchmark-trt <output.plan> <input.vin> <output.bin> [iterations]");
}

fn main() -> ExitCode {
    cli_logging::init();
    let args: Vec<String> = env::args().skip(1).collect();
    let command = args
        .first()
        .map(String::as_str)
        .unwrap_or("help")
        .to_owned();
    let mut status = CliStatus::default();
    status.start();
    cli_logging::command_started(&command);
    match run(args) {
        Ok(()) => {
            status.complete();
            cli_logging::command_finished(&command);
            ExitCode::SUCCESS
        }
        Err(error) => {
            status.fail();
            cli_logging::command_failed(&command, &error);
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> CliResult<()> {
    let cli_config = CliConfig::default();
    match args.as_slice() {
        [command] if command == "doctor" => doctor(),
        [command, subcommand, manifest] if command == "model" && subcommand == "inspect" => {
            inspect(PathBuf::from(manifest))
        }
        [command, subcommand, cache] if command == "model" && subcommand == "inspect-cache" => {
            inspect_cache(PathBuf::from(cache))
        }
        [command, subcommand, cache] if command == "model" && subcommand == "inspect-acoustic" => {
            inspect_acoustic(PathBuf::from(cache))
        }
        [command, subcommand, model_dir, manifest, output]
            if command == "model" && subcommand == "pack" =>
        {
            pack(model_dir.into(), manifest.into(), output.into())
        }
        [command, subcommand, model_dir, manifest, output]
            if command == "model" && subcommand == "pack-rvc" =>
        {
            pack_rvc(model_dir.into(), manifest.into(), output.into())
        }
        [command, subcommand, model_dir, manifest, checkpoint, output]
            if command == "model" && subcommand == "convert-pth" =>
        {
            convert_pth(
                model_dir.into(),
                manifest.into(),
                checkpoint.into(),
                output.into(),
            )
        }
        [command, subcommand, model_dir, manifest, onnx, output]
            if command == "model" && subcommand == "convert-onnx" =>
        {
            convert_onnx(
                model_dir.into(),
                manifest.into(),
                onnx.into(),
                output.into(),
            )
        }
        [command, subcommand, cache, output]
            if command == "model" && subcommand == "build-graph" =>
        {
            build_graph(cache.into(), output.into())
        }
        [command, subcommand, cache, output]
            if command == "model" && subcommand == "compile-trt" =>
        {
            compile_trt(cache.into(), output.into(), None)
        }
        [command, subcommand, cache, output]
            if command == "model" && subcommand == "compile-hubert-trt" =>
        {
            compile_hubert_trt(cache.into(), output.into())
        }
        [command, subcommand, plan, input]
            if command == "model" && subcommand == "test-hubert-trt" =>
        {
            test_hubert_trt(plan.into(), input.into())
        }
        [command, subcommand, cache, output, onnx]
            if command == "model" && subcommand == "compile-trt" =>
        {
            compile_trt(cache.into(), output.into(), Some(onnx.into()))
        }
        [command, subcommand, cache, hubert, rmvpe, output]
            if command == "model" && subcommand == "compile-trt-pipeline" =>
        {
            compile_trt_pipeline(cache.into(), hubert.into(), rmvpe.into(), output.into())
        }
        [command, subcommand, artifacts, cache, input]
            if command == "model" && subcommand == "test-trt-pipeline" =>
        {
            test_trt_pipeline(artifacts.into(), cache.into(), input.into(), false)
        }
        [command, subcommand, backend, cache, input]
            if command == "model" && subcommand == "test-pipeline" =>
        {
            run_offline_pipeline(backend, cache.into(), input.into(), None, None)
        }
        [command, subcommand, backend, cache, input, artifacts]
            if command == "model" && subcommand == "test-pipeline" =>
        {
            run_offline_pipeline(
                backend,
                cache.into(),
                input.into(),
                None,
                Some(artifacts.into()),
            )
        }
        [command, subcommand, backend, cache, input, output]
            if command == "model" && subcommand == "run-pipeline" =>
        {
            run_offline_pipeline(
                backend,
                cache.into(),
                input.into(),
                Some(output.into()),
                None,
            )
        }
        [
            command,
            subcommand,
            backend,
            cache,
            input,
            output,
            artifacts,
        ] if command == "model" && subcommand == "run-pipeline" => run_offline_pipeline(
            backend,
            cache.into(),
            input.into(),
            Some(output.into()),
            Some(artifacts.into()),
        ),
        [command, subcommand, artifacts, cache, input, output]
            if command == "model" && subcommand == "run-trt-pipeline" =>
        {
            run_trt_pipeline(artifacts.into(), cache.into(), input.into(), output.into())
        }
        [command, subcommand, artifacts, cache, input]
            if command == "model" && subcommand == "profile-trt-pipeline" =>
        {
            test_trt_pipeline(artifacts.into(), cache.into(), input.into(), true)
        }
        [command, subcommand, artifacts, cache, input]
            if command == "model" && subcommand == "compare-trt-pipeline" =>
        {
            compare_trt_pipeline(artifacts.into(), cache.into(), input.into(), None)
        }
        [command, subcommand, artifacts, cache, input, iterations]
            if command == "model" && subcommand == "compare-trt-pipeline" =>
        {
            compare_trt_pipeline(
                artifacts.into(),
                cache.into(),
                input.into(),
                Some(iterations.parse()?),
            )
        }
        [command, subcommand, plan] if command == "model" && subcommand == "test-trt" => {
            test_trt(plan.into(), None)
        }
        [command, subcommand, plan, cache] if command == "model" && subcommand == "test-trt" => {
            test_trt(plan.into(), Some(cache.into()))
        }
        [command, subcommand, plan, input, output]
            if command == "model" && subcommand == "benchmark-trt" =>
        {
            benchmark_trt(
                plan.into(),
                input.into(),
                output.into(),
                cli_config.default_benchmark_iterations,
            )
        }
        [command, subcommand, plan, input, output, iterations]
            if command == "model" && subcommand == "benchmark-trt" =>
        {
            benchmark_trt(
                plan.into(),
                input.into(),
                output.into(),
                cli_config.benchmark_iterations(Some(iterations.parse()?)),
            )
        }
        [command] if command == "demo" => demo(cli_config.default_demo_blocks),
        [command, count] if command == "demo" => demo(count.parse()?),
        _ => {
            usage();
            Ok(())
        }
    }
}

#[cfg(feature = "tensorrt-rtx")]
fn compile_trt_pipeline(
    cache: PathBuf,
    hubert: PathBuf,
    rmvpe: PathBuf,
    output: PathBuf,
) -> CliResult<()> {
    let package = load_cache(cache)?;
    let trt_config = vocal_tensorrt::TensorRtConfig::from_env()?;
    let profile = trt_config.profile.clone();
    let artifacts = vocal_tensorrt::RvcTrtPipelineArtifacts::compile_with_config(
        &trt_config,
        &package,
        hubert,
        rmvpe,
        &output,
    )?;
    let manifest = output.join("pipeline.json");
    fs::write(&manifest, serde_json::to_vec_pretty(&artifacts)?)?;
    println!(
        "compiled HuBERT/RMVPE/frontend/Synth plans; samples=[{}..={}..={}], hubert=[{}..={}..={}], pitch=[{}..={}..={}], rvc=[{}..={}..={}]; manifest={}",
        profile.min_samples,
        profile.opt_samples,
        profile.max_samples,
        profile.min_hubert_frames,
        profile.opt_hubert_frames,
        profile.max_hubert_frames,
        profile.min_pitch_frames,
        profile.opt_pitch_frames,
        profile.max_pitch_frames,
        profile.min_rvc_frames,
        profile.opt_rvc_frames,
        profile.max_rvc_frames,
        manifest.display()
    );
    Ok(())
}

#[cfg(not(feature = "tensorrt-rtx"))]
fn compile_trt_pipeline(
    _cache: PathBuf,
    _hubert: PathBuf,
    _rmvpe: PathBuf,
    _output: PathBuf,
) -> CliResult<()> {
    Err("TensorRT-RTX feature is disabled; rebuild with --features tensorrt-rtx".into())
}

#[cfg(feature = "tensorrt-rtx")]
fn test_trt_pipeline(
    artifacts_path: PathBuf,
    cache: PathBuf,
    input: PathBuf,
    nsight_profile: bool,
) -> CliResult<()> {
    let artifacts: vocal_tensorrt::RvcTrtPipelineArtifacts =
        serde_json::from_slice(&fs::read(&artifacts_path)?)?;
    let package = load_cache(cache)?;
    let config = package
        .rvc_synth_config()?
        .ok_or("cache has no rvc_synth config")?;
    let samples = read_f32_audio(&input)?;
    let trt_config = vocal_tensorrt::TensorRtConfig::from_env()?;
    let runtime_config = trt_config.runtime.clone();
    let mut diagnostics = trt_config.diagnostics.clone();
    // Nsight runs intentionally measure the production path without host
    // debug copies, while retaining the typed runtime graph policy.
    diagnostics.debug_intermediates &= !nsight_profile;
    diagnostics.layer_profile &= !nsight_profile;
    let mut run_config = trt_config.run.clone();
    if nsight_profile {
        if run_config.iterations == 1 {
            run_config.iterations = 100;
        }
        if run_config.warmup == 0 {
            run_config.warmup = 10;
        }
    }
    let mut pipeline = vocal_tensorrt::RvcTrtPipeline::load_with_runtime_config(
        &artifacts,
        config,
        runtime_config.clone(),
        diagnostics.clone(),
    )?;
    // Intermediate D2H copies are opt-in for parity diagnostics. Nsight mode
    // keeps all activations device-resident and measures the production path.
    let debug_intermediates = diagnostics.debug_intermediates;
    pipeline.enable_debug_intermediates(debug_intermediates);
    let profile_layers = diagnostics.layer_profile;
    if profile_layers {
        pipeline.enable_layer_profiling(true);
    }
    let iterations = run_config.iterations;
    let warmups = run_config.warmup;
    let realtime_f0 = run_config.realtime_f0;
    let realtime_advance = run_config
        .realtime_advance_frames
        .unwrap_or((samples.len() / 160).min(25).max(1));
    let run_once = |pipeline: &mut vocal_tensorrt::RvcTrtPipeline|
                   -> Result<vocal_tensorrt::RvcTrtOutput, vocal_tensorrt::NativeBuildError> {
        if realtime_f0 {
            pipeline.process_realtime(&samples, 0, realtime_advance)
        } else {
            pipeline.process(&samples, 0)
        }
    };
    for _ in 0..warmups {
        let _ = run_once(&mut pipeline)?;
    }
    let mut results = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        results.push(run_once(&mut pipeline)?);
    }
    let result = results.last().ok_or("pipeline produced no result")?;
    let debug_path = artifacts_path.with_file_name("trt-debug.json");
    if debug_intermediates {
        fs::write(
            &debug_path,
            serde_json::to_vec(&serde_json::json!({
                "hubert_frames": result.hubert_frames, "pitch_frames": result.pitch_frames,
                "rvc_frames": result.rvc_frames, "hubert_features": result.hubert_features,
                "rvc_features": result.rvc_features, "pitchf": result.pitchf,
                "aligned_pitchf": result.aligned_pitchf, "pitch": result.pitch,
                "rmvpe_mel": result.rmvpe_mel, "rmvpe_gru": result.rmvpe_gru,
                "rmvpe_bottleneck": result.rmvpe_bottleneck,
                "rmvpe_unet": result.rmvpe_unet,
                "rmvpe_logits": result.rmvpe_logits,
                "audio": result.audio,
            }))?,
        )?;
    }
    let mean_total = results.iter().map(|r| r.timings.total_ms).sum::<f64>() / iterations as f64;
    println!(
        "{}",
        serde_json::json!({
            "iterations": iterations, "mean_total_ms": mean_total,
            "input_samples": samples.len(), "hubert_frames": result.hubert_frames,
            "pitch_frames": result.pitch_frames, "rvc_frames": result.rvc_frames,
            "output_samples": result.audio.len(), "timings_ms": {
                "upload": result.timings.upload_ms, "hubert": result.timings.hubert_enqueue_ms,
                "rmvpe": result.timings.rmvpe_enqueue_ms, "frontend": result.timings.frontend_enqueue_ms,
                "synth": result.timings.synth_enqueue_ms, "total": result.timings.total_ms,
            },
            "warmups": warmups, "nsight_profile": nsight_profile,
            "realtime_f0": realtime_f0, "realtime_advance_frames": realtime_advance,
            "cuda_graph": pipeline.cuda_graph_stats(),
            "debug_intermediates": debug_intermediates,
            "layer_profile": if profile_layers { serde_json::to_value(pipeline.layer_timings())? } else { serde_json::Value::Null }
        })
    );
    if debug_intermediates {
        println!("debug tensors written to {}", debug_path.display());
    }
    Ok(())
}

#[cfg(feature = "tensorrt-rtx")]
fn run_trt_pipeline(
    artifacts_path: PathBuf,
    cache: PathBuf,
    input: PathBuf,
    output: PathBuf,
) -> CliResult<()> {
    let artifacts: vocal_tensorrt::RvcTrtPipelineArtifacts =
        serde_json::from_slice(&fs::read(&artifacts_path)?)?;
    let package = load_cache(cache)?;
    let synth_config = package
        .rvc_synth_config()?
        .ok_or("cache has no rvc_synth config")?;
    let samples = read_f32_audio(&input)?;
    if samples.is_empty() {
        return Err("offline input contains no samples".into());
    }
    let trt_config = vocal_tensorrt::TensorRtConfig::from_env()?;
    let mut pipeline = vocal_tensorrt::RvcTrtPipeline::load_with_runtime_config(
        &artifacts,
        synth_config.clone(),
        trt_config.runtime,
        trt_config.diagnostics,
    )?;
    let profile = artifacts.profile;
    let min_samples = profile.min_samples.max(160) as usize;
    let max_samples = profile.max_samples.max(min_samples as i32) as usize;
    let preferred_samples = profile.opt_samples.max(min_samples as i32) as usize;
    let mut output_audio = Vec::new();
    let mut input_offset = 0usize;
    let started = Instant::now();
    let mut windows = 0usize;
    while input_offset < samples.len() {
        let remaining = samples.len() - input_offset;
        let source_len = remaining.min(max_samples);
        let aligned_len = round_up_audio_samples(source_len, 160);
        let window_len = if aligned_len < min_samples {
            preferred_samples.max(aligned_len)
        } else {
            aligned_len.min(max_samples)
        };
        let valid_len = source_len.min(window_len);
        let mut window = vec![0.0f32; window_len];
        window[..valid_len].copy_from_slice(&samples[input_offset..input_offset + valid_len]);
        let result = pipeline.process(&window, 0)?;
        let keep = if valid_len < window_len {
            result.audio.len().saturating_mul(valid_len) / window_len
        } else {
            result.audio.len()
        };
        output_audio.extend_from_slice(&result.audio[..keep]);
        input_offset += source_len;
        windows += 1;
    }
    write_wav_pcm16(&output, synth_config.sample_rate, &output_audio)
        .map_err(|error| format!("write output WAV {}: {error}", output.display()))?;
    println!(
        "{}",
        serde_json::json!({
            "input": input,
            "output": output,
            "input_samples_16k": samples.len(),
            "output_samples": output_audio.len(),
            "output_sample_rate": synth_config.sample_rate,
            "windows": windows,
            "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
            "cuda_graph": pipeline.cuda_graph_stats(),
        })
    );
    Ok(())
}

/// Run any backend through the shared offline window runner.  The optional
/// TensorRT artifact path is only interpreted for the `tensorrt` backend;
/// passthrough and CPU RVC use the same command and model package format.
fn run_offline_pipeline(
    backend_name: &str,
    cache: PathBuf,
    input: PathBuf,
    output: Option<PathBuf>,
    artifacts: Option<PathBuf>,
) -> CliResult<()> {
    let backend = parse_offline_backend(backend_name)?;
    let package = load_cache(&cache)?;
    let descriptor = package.manifest.descriptor();
    let (engine, window_samples) = create_offline_engine(
        backend,
        descriptor.clone(),
        cache.clone(),
        artifacts.clone(),
    )?;
    let samples = read_f32_audio(&input)?;
    if samples.is_empty() {
        return Err("offline input contains no samples".into());
    }
    let mut pipeline = EngineOfflinePipeline::new(backend, engine, AudioFormat::MONO_16K)?;
    let mut config = OfflineRunConfig::default();
    config.window_samples = window_samples;
    let (audio_16k, report) = run_offline(&mut pipeline, &samples, &config)?;
    let output_rate = descriptor.sample_rate;
    if let Some(output_path) = output {
        let audio = resample_linear(&audio_16k, 16_000, output_rate);
        write_wav_pcm16(&output_path, output_rate, &audio)
            .map_err(|error| format!("write output WAV {}: {error}", output_path.display()))?;
    }
    println!(
        "{}",
        serde_json::json!({
            "backend": backend_name,
            "model": cache,
            "input": input,
            "output_sample_rate": output_rate,
            "input_samples_16k": report.input_samples,
            "output_samples_16k": report.output_samples,
            "windows": report.windows,
            "elapsed_ms": report.elapsed_ms,
            "engine_metrics": report.metrics,
            "artifacts": artifacts,
        })
    );
    Ok(())
}

fn parse_offline_backend(value: &str) -> CliResult<BackendKind> {
    match value.to_ascii_lowercase().as_str() {
        "passthrough" | "pass" => Ok(BackendKind::Passthrough),
        "cpu" | "rvc-cpu" | "rvc_cpu" => Ok(BackendKind::RvcCpuPipeline),
        "tensorrt" | "trt" | "tensorrt-rtx" | "tensorrt_rtx" => Ok(BackendKind::TensorRtRtx),
        _ => Err(format!(
            "unknown offline backend {value}; expected passthrough, cpu, or tensorrt"
        )
        .into()),
    }
}

fn create_offline_engine(
    backend: BackendKind,
    descriptor: ModelDescriptor,
    cache: PathBuf,
    artifacts: Option<PathBuf>,
) -> CliResult<(Box<dyn InferenceEngine>, usize)> {
    match backend {
        BackendKind::Passthrough | BackendKind::RvcCpuPipeline => {
            let engine = vocal_engine::create_engine(EngineConfig::portable(backend, descriptor))?;
            Ok((engine, offline_window_from_env()))
        }
        BackendKind::TensorRtRtx => {
            #[cfg(feature = "tensorrt-rtx")]
            {
                let artifacts_path = artifacts.ok_or(
                    "TensorRT offline execution requires the compiled pipeline.json artifact",
                )?;
                let mut config =
                    EngineConfig::with_tensorrt(descriptor, artifacts_path.clone(), cache);
                let profile: vocal_tensorrt::RvcTrtPipelineArtifacts =
                    serde_json::from_slice(&fs::read(&artifacts_path)?)?;
                let window_samples = offline_window_from_env()
                    .max(profile.profile.min_samples.max(1) as usize)
                    .min(profile.profile.max_samples.max(1) as usize);
                if let Some(spec) = config.tensorrt.as_mut() {
                    spec.window_samples = Some(window_samples);
                }
                let engine = vocal_engine::create_engine(config)?;
                Ok((engine, window_samples))
            }
            #[cfg(not(feature = "tensorrt-rtx"))]
            {
                let _ = (descriptor, cache, artifacts);
                Err("TensorRT offline execution requires --features tensorrt-rtx".into())
            }
        }
    }
}

fn offline_window_from_env() -> usize {
    env::var("VOCAL_OFFLINE_WINDOW_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value: &usize| *value > 0)
        .unwrap_or(vocal_engine::offline::DEFAULT_WINDOW_SAMPLES)
}

#[cfg(not(feature = "tensorrt-rtx"))]
fn run_trt_pipeline(
    _artifacts: PathBuf,
    _cache: PathBuf,
    _input: PathBuf,
    _output: PathBuf,
) -> CliResult<()> {
    Err("TensorRT-RTX feature is disabled; rebuild with --features tensorrt-rtx".into())
}

#[cfg(feature = "tensorrt-rtx")]
fn round_up_audio_samples(samples: usize, hop: usize) -> usize {
    samples
        .saturating_add(hop.saturating_sub(1))
        .checked_div(hop)
        .unwrap_or(0)
        .saturating_mul(hop)
}

#[cfg(feature = "tensorrt-rtx")]
fn compare_trt_pipeline(
    artifacts_path: PathBuf,
    cache: PathBuf,
    input: PathBuf,
    requested_iterations: Option<usize>,
) -> CliResult<()> {
    let artifacts: vocal_tensorrt::RvcTrtPipelineArtifacts =
        serde_json::from_slice(&fs::read(&artifacts_path)?)?;
    let package = load_cache(cache)?;
    let synth_config = package
        .rvc_synth_config()?
        .ok_or("cache has no rvc_synth config")?;
    let samples = read_f32_audio(&input)?;
    let base_config = vocal_tensorrt::TensorRtConfig::from_env()?;
    let warmups = base_config.run.warmup.max(3);
    let iterations = requested_iterations
        .unwrap_or(base_config.run.iterations.max(20))
        .max(1);
    let realtime_f0 = base_config.run.realtime_f0;
    let realtime_advance = base_config
        .run
        .realtime_advance_frames
        .unwrap_or((samples.len() / 160).min(25).max(1));
    let modes = [
        vocal_tensorrt::GraphReplayMode::NativeSubgraphs,
        vocal_tensorrt::GraphReplayMode::WholePipeline,
        vocal_tensorrt::GraphReplayMode::HybridSubgraphs,
    ];
    let mut baseline_audio: Option<Vec<f32>> = None;
    let mut reports = Vec::with_capacity(modes.len());

    for mode in modes {
        let mut config = base_config.clone();
        config.runtime.graph_mode = mode;
        // Keep the legacy fields explicit so this command also exercises the
        // compatibility resolver used by older embedding callers.
        config.runtime.cuda_graph = true;
        config.runtime.native_cuda_graph = mode == vocal_tensorrt::GraphReplayMode::NativeSubgraphs;
        config.diagnostics.debug_intermediates = false;
        config.diagnostics.layer_profile = false;
        let mut pipeline = vocal_tensorrt::RvcTrtPipeline::load_with_runtime_config(
            &artifacts,
            synth_config.clone(),
            config.runtime,
            config.diagnostics,
        )?;
        let run_once = |pipeline: &mut vocal_tensorrt::RvcTrtPipeline| {
            if realtime_f0 {
                pipeline.process_realtime(&samples, 0, realtime_advance)
            } else {
                pipeline.process(&samples, 0)
            }
        };
        for _ in 0..warmups {
            let _ = run_once(&mut pipeline)?;
        }
        let mut outputs = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            outputs.push(run_once(&mut pipeline)?);
        }
        let result = outputs.pop().ok_or("graph comparison produced no result")?;
        let audio = result.audio.clone();
        let delta = baseline_audio
            .as_deref()
            .map(|reference| audio_delta(reference, &audio))
            .unwrap_or(serde_json::Value::Null);
        if baseline_audio.is_none() {
            baseline_audio = Some(audio.clone());
        }
        let mean = |select: fn(&vocal_tensorrt::RvcTrtOutput) -> f64| {
            outputs
                .iter()
                .map(select)
                .chain(std::iter::once(select(&result)))
                .sum::<f64>()
                / iterations as f64
        };
        reports.push(serde_json::json!({
            "mode": mode,
            "iterations": iterations,
            "warmups": warmups,
            "mean_total_ms": mean(|r| r.timings.total_ms),
            "mean_stage_ms": {
                "upload": mean(|r| r.timings.upload_ms),
                "hubert": mean(|r| r.timings.hubert_enqueue_ms),
                "rmvpe": mean(|r| r.timings.rmvpe_enqueue_ms),
                "frontend": mean(|r| r.timings.frontend_enqueue_ms),
                "synth": mean(|r| r.timings.synth_enqueue_ms),
            },
            "output_samples": audio.len(),
            "output_delta_vs_native": delta,
            "cuda_graph": pipeline.cuda_graph_stats(),
        }));
    }

    println!(
        "{}",
        serde_json::json!({
            "input_samples": samples.len(),
            "realtime_f0": realtime_f0,
            "realtime_advance_frames": realtime_advance,
            "comparison": reports,
        })
    );
    Ok(())
}

#[cfg(feature = "tensorrt-rtx")]
fn audio_delta(reference: &[f32], candidate: &[f32]) -> serde_json::Value {
    let count = reference.len().min(candidate.len());
    if count == 0 {
        return serde_json::json!({
            "reference_samples": reference.len(),
            "candidate_samples": candidate.len(),
            "max_abs": null,
            "rmse": null,
            "cosine": null,
        });
    }
    let mut max_abs = 0.0_f64;
    let mut squared = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut reference_norm = 0.0_f64;
    let mut candidate_norm = 0.0_f64;
    for (&lhs, &rhs) in reference.iter().zip(candidate).take(count) {
        let lhs = lhs as f64;
        let rhs = rhs as f64;
        let difference = (lhs - rhs).abs();
        max_abs = max_abs.max(difference);
        squared += difference * difference;
        dot += lhs * rhs;
        reference_norm += lhs * lhs;
        candidate_norm += rhs * rhs;
    }
    let cosine = if reference_norm > 0.0 && candidate_norm > 0.0 {
        dot / (reference_norm.sqrt() * candidate_norm.sqrt())
    } else {
        0.0
    };
    serde_json::json!({
        "reference_samples": reference.len(),
        "candidate_samples": candidate.len(),
        "compared_samples": count,
        "max_abs": max_abs,
        "rmse": (squared / count as f64).sqrt(),
        "cosine": cosine,
    })
}

#[cfg(not(feature = "tensorrt-rtx"))]
fn test_trt_pipeline(
    _artifacts: PathBuf,
    _cache: PathBuf,
    _input: PathBuf,
    _nsight_profile: bool,
) -> CliResult<()> {
    Err("TensorRT-RTX feature is disabled; rebuild with --features tensorrt-rtx".into())
}

#[cfg(not(feature = "tensorrt-rtx"))]
fn compare_trt_pipeline(
    _artifacts: PathBuf,
    _cache: PathBuf,
    _input: PathBuf,
    _iterations: Option<usize>,
) -> CliResult<()> {
    Err("TensorRT-RTX feature is disabled; rebuild with --features tensorrt-rtx".into())
}

fn read_f32_audio(path: &PathBuf) -> CliResult<Vec<f32>> {
    let bytes = fs::read(path)?;
    if bytes.len() >= 44 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return Ok(read_wav(path)
            .map_err(|error| format!("decode WAV {}: {error}", path.display()))?
            .mono_16k());
    }
    if !bytes.len().is_multiple_of(4) {
        return Err("raw f32 input size must be a multiple of four bytes".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect())
}

fn doctor() -> CliResult<()> {
    let config = StreamConfig::default();
    let ring = AudioRing::new(config)?;
    println!(
        "audio: ok ({} Hz, {} channel, {} frame blocks)",
        config.format.sample_rate, config.format.channels, config.block_frames
    );
    println!("engine: CPU RVC pipeline + passthrough baseline ready");
    println!("cache: VFRM v1 tensor container ready");
    println!("ring capacity: {} blocks", config.queue_blocks);
    drop(ring);
    Ok(())
}

fn inspect(path: PathBuf) -> CliResult<()> {
    let manifest = load_manifest(path)?;
    println!("{} ({})", manifest.display_name, manifest.id);
    println!(
        "architecture={} version={} sample_rate={}",
        manifest.architecture, manifest.version, manifest.sample_rate
    );
    println!(
        "weights={} index={}",
        manifest.model_file,
        manifest.index_file.as_deref().unwrap_or("none")
    );
    Ok(())
}

fn pack(model_dir: PathBuf, manifest_path: PathBuf, output: PathBuf) -> CliResult<()> {
    let manifest = load_manifest(&manifest_path)?;
    let package = load_directory(model_dir, manifest)?;
    write_cache(&output, &package)?;
    println!(
        "wrote {} bytes to {}",
        fs::metadata(&output)?.len(),
        output.display()
    );
    Ok(())
}

fn pack_rvc(model_dir: PathBuf, manifest_path: PathBuf, output: PathBuf) -> CliResult<()> {
    let package = load_rvc_directory(model_dir, load_manifest(manifest_path)?)?;
    write_cache(&output, &package)?;
    println!(
        "wrote {} typed tensors to {}",
        package.tensors.len(),
        output.display()
    );
    Ok(())
}

fn inspect_cache(path: PathBuf) -> CliResult<()> {
    let package = load_cache(path)?;
    println!(
        "{} ({}) tensors={} bytes={}",
        package.manifest.display_name,
        package.manifest.id,
        package.tensors.len(),
        package
            .tensors
            .iter()
            .map(|tensor| tensor.data.len())
            .sum::<usize>()
    );
    if let Some(schema) = package
        .metadata
        .get("canonical_schema")
        .and_then(serde_json::Value::as_str)
    {
        println!(
            "canonical_schema={} graph_schema={} tensor_map={}",
            schema,
            package
                .metadata
                .get("graph_schema")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown"),
            package
                .metadata
                .get("canonical_tensor_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
        );
    }
    for tensor in package.tensors {
        println!(
            "  {} {:?} {:?} {} bytes",
            tensor.name,
            tensor.dtype,
            tensor.shape,
            tensor.data.len()
        );
    }
    Ok(())
}

fn inspect_acoustic(path: PathBuf) -> CliResult<()> {
    let package = load_cache(path)?;
    let config = package.validate_acoustic(None)?;
    let nodes = package
        .metadata
        .get("nodes")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    let ops = package
        .metadata
        .get("nodes")
        .and_then(serde_json::Value::as_array)
        .map(|nodes| {
            let mut counts = std::collections::BTreeMap::<String, usize>::new();
            for node in nodes {
                if let Some(op) = node.get("op").and_then(serde_json::Value::as_str) {
                    *counts.entry(op.to_owned()).or_default() += 1;
                }
            }
            counts
        })
        .unwrap_or_default();
    println!(
        "kind={} schema={} nodes={} tensors={} inputs={} outputs={}",
        config.kind.as_str(),
        ACOUSTIC_TENSOR_SCHEMA,
        nodes,
        package.tensors.len(),
        config.inputs.len(),
        config.outputs.len()
    );
    println!("operators={}", serde_json::to_string(&ops)?);
    for port in config.inputs {
        println!("input {} {:?} dtype={}", port.name, port.shape, port.dtype);
    }
    for port in config.outputs {
        println!("output {} {:?} dtype={}", port.name, port.shape, port.dtype);
    }
    Ok(())
}

fn convert_pth(
    model_dir: PathBuf,
    manifest_path: PathBuf,
    checkpoint: PathBuf,
    output: PathBuf,
) -> CliResult<()> {
    let package = load_rvc_pth(checkpoint, load_manifest(manifest_path)?)?;
    write_cache(&output, &package)?;
    println!(
        "converted {} tensors to {}",
        package.tensors.len(),
        output.display()
    );
    let _ = model_dir;
    Ok(())
}

fn convert_onnx(
    model_dir: PathBuf,
    manifest_path: PathBuf,
    onnx: PathBuf,
    output: PathBuf,
) -> CliResult<()> {
    let package = load_rvc_onnx(onnx, load_manifest(manifest_path)?)?;
    write_cache(&output, &package)?;
    println!(
        "converted {} ONNX initializers to {}",
        package.tensors.len(),
        output.display()
    );
    let _ = model_dir;
    Ok(())
}

fn build_graph(cache: PathBuf, output: PathBuf) -> CliResult<()> {
    let package = load_cache(cache)?;
    let graph = build_rvc_graph(&package, TensorRtGraphOptions::default())?;
    graph.write_json(&output)?;
    println!("wrote {graph} to {}", output.display());
    Ok(())
}

#[allow(clippy::needless_return, clippy::field_reassign_with_default)]
fn compile_trt(cache: PathBuf, output: PathBuf, onnx: Option<PathBuf>) -> CliResult<()> {
    #[cfg(feature = "tensorrt-rtx")]
    {
        let package = load_cache(cache)?;
        let mut options = vocal_tensorrt::TensorRtConfig::from_env()?.build;
        options.output = output;
        let builder = vocal_tensorrt::NativeTensorRtBuilder::new(options);
        let path = match onnx {
            Some(onnx) => builder.compile_onnx_file(onnx)?,
            None => builder.compile_vfrm(&package)?,
        };
        println!(
            "compiled {} with TensorRT-RTX native builder",
            path.display()
        );
        return Ok(());
    }
    #[cfg(not(feature = "tensorrt-rtx"))]
    {
        let package = load_cache(cache)?;
        let graph = build_rvc_graph(&package, TensorRtGraphOptions::default())?;
        let mut compiler = TrtExecCompiler::discover()
            .ok_or("trtexec was not found; install TensorRT or set VOCAL_TRTEXEC to its path")?;
        if let Some(onnx) = onnx {
            compiler = compiler.with_onnx(onnx);
        }
        let report = compiler.compile(&graph, &output)?;
        println!(
            "compiled {} using {}; graph={}",
            report.engine_path.as_ref().map_or_else(
                || "descriptor-only".to_owned(),
                |path| path.display().to_string()
            ),
            report.compiler,
            report.graph_path.display()
        );
        Ok(())
    }
}

#[cfg(feature = "tensorrt-rtx")]
fn compile_hubert_trt(cache: PathBuf, output: PathBuf) -> CliResult<()> {
    let package = load_cache(cache)?;
    let mut options = vocal_tensorrt::TensorRtConfig::from_env()?.build;
    options.output = output;
    let path = vocal_tensorrt::NativeTensorRtBuilder::new(options).compile_hubert_vfrm(&package)?;
    println!("compiled native HuBERT plan {}", path.display());
    Ok(())
}

#[cfg(not(feature = "tensorrt-rtx"))]
fn compile_hubert_trt(_cache: PathBuf, _output: PathBuf) -> CliResult<()> {
    Err("TensorRT-RTX feature is disabled; rebuild with --features tensorrt-rtx".into())
}

#[cfg(feature = "tensorrt-rtx")]
fn test_hubert_trt(plan: PathBuf, input: PathBuf) -> CliResult<()> {
    let trt_config = vocal_tensorrt::TensorRtConfig::from_env()?;
    let samples = read_f32_audio(&input)?;
    let runtime = vocal_tensorrt::Runtime::load(&plan)?;
    let mut context = runtime.context()?;
    let stream = vocal_tensorrt::CudaStream::new()?;
    let waveform = vocal_tensorrt::CudaBuffer::new(samples.len() * 4)?;
    let max_frames = (samples.len() / 320).saturating_add(8);
    let features = vocal_tensorrt::CudaBuffer::new(max_frames * 768 * 4)?;
    let bytes = samples
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    waveform.upload_bytes(&bytes)?;
    context.set_input_shape("waveform", &[1, samples.len() as i64])?;
    context.set_tensor_address("waveform", waveform.device_ptr())?;
    context.set_tensor_address("features", features.device_ptr())?;
    let start = Instant::now();
    context.enqueue_named(&stream, Some("rvc.hubert.native"))?;
    stream.synchronize()?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let shape = context.tensor_shape("features")?;
    let count = shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d as usize))
        .ok_or("feature shape overflow")?;
    let mut raw = vec![0_u8; count * 4];
    features.download_bytes(&mut raw)?;
    let values = raw
        .chunks_exact(4)
        .map(|x| <[u8; 4]>::try_from(x).map(f32::from_le_bytes))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(path) = trt_config.diagnostics.dump_hubert {
        fs::write(path, &raw)?;
    }
    let finite = values.iter().all(|v| v.is_finite());
    println!(
        "{}",
        serde_json::json!({"plan": plan, "input_samples": samples.len(), "shape": shape, "latency_ms": elapsed_ms, "finite": finite, "mean_abs": values.iter().map(|v| v.abs() as f64).sum::<f64>() / values.len().max(1) as f64})
    );
    if !finite {
        return Err("native HuBERT produced non-finite features".into());
    }
    Ok(())
}

#[cfg(not(feature = "tensorrt-rtx"))]
fn test_hubert_trt(_plan: PathBuf, _input: PathBuf) -> CliResult<()> {
    Err("TensorRT-RTX feature is disabled; rebuild with --features tensorrt-rtx".into())
}

#[cfg(feature = "tensorrt-rtx")]
fn test_trt(plan: PathBuf, cache: Option<PathBuf>) -> CliResult<()> {
    let trt_config = vocal_tensorrt::TensorRtConfig::from_env()?;
    let runtime = vocal_tensorrt::Runtime::load(&plan)?;
    let mut context = runtime.context()?;
    let stream = vocal_tensorrt::CudaStream::new()?;
    let Some(cache) = cache else {
        let buffer = vocal_tensorrt::CudaBuffer::new(4096)?;
        println!(
            "loaded {} and created TensorRT execution context; CUDA allocation={} bytes",
            plan.display(),
            buffer.bytes
        );
        return Ok(());
    };
    let package = vocal_model::load_cache(cache)?;
    let config = package
        .rvc_synth_config()?
        .ok_or("cache has no rvc_synth config")?;
    let frames = i64::try_from(trt_config.run.test_frames)?;
    let up = config
        .upsample_rates
        .iter()
        .try_fold(1_i64, |value, rate| value.checked_mul(*rate as i64))
        .ok_or("upsample product overflow")?;
    let alloc = |elements: i64| -> CliResult<vocal_tensorrt::CudaBuffer> {
        let bytes = usize::try_from(elements.checked_mul(4).ok_or("buffer size overflow")?)?;
        let buffer = vocal_tensorrt::CudaBuffer::new(bytes)?;
        buffer.zero()?;
        Ok(buffer)
    };
    let feats = alloc(frames * config.feature_width as i64)?;
    let p_len = alloc(1)?;
    let sid = alloc(1)?;
    let rnd = alloc(config.latent_channels as i64 * frames)?;
    let pitch = config.uses_f0.then(|| alloc(frames)).transpose()?;
    let pitchf = config.uses_f0.then(|| alloc(frames)).transpose()?;
    let noise = config.uses_f0.then(|| alloc(frames * up)).transpose()?;
    let audio = alloc(frames * up + 4096)?;
    let feature_values = (0..(frames as usize * config.feature_width as usize))
        .map(|index| 0.02_f32 * ((index as f32) * 0.013).sin())
        .collect::<Vec<_>>();
    feats.upload_f32(&feature_values)?;
    let rnd_values = (0..(frames as usize * config.latent_channels as usize))
        .map(|index| 0.01_f32 * ((index as f32) * 0.017).cos())
        .collect::<Vec<_>>();
    rnd.upload_f32(&rnd_values)?;
    let length_bytes = (frames as i32).to_le_bytes();
    p_len.upload_bytes(&length_bytes)?;
    let sid_bytes = 0_i32.to_le_bytes();
    sid.upload_bytes(&sid_bytes)?;
    if let (Some(pitch), Some(pitchf)) = (&pitch, &pitchf) {
        let pitch_bytes = (0..frames)
            .flat_map(|_| 220_i32.to_le_bytes())
            .collect::<Vec<_>>();
        pitch.upload_bytes(&pitch_bytes)?;
        let pitchf_bytes = (0..frames)
            .flat_map(|_| 220.0_f32.to_le_bytes())
            .collect::<Vec<_>>();
        pitchf.upload_bytes(&pitchf_bytes)?;
    }
    if let Some(noise) = &noise {
        let noise_values = (0..(frames as usize * up as usize))
            .map(|index| 0.001_f32 * ((index as f32) * 0.011).sin())
            .collect::<Vec<_>>();
        noise.upload_f32(&noise_values)?;
    }
    tracing::debug!(target: cli_logging::LOG_TARGET, up, "test buffers allocated");
    context.set_input_shape("feats", &[1, frames, config.feature_width as i64])?;
    context.set_input_shape("rnd", &[1, config.latent_channels as i64, frames])?;
    if config.uses_f0 {
        context.set_input_shape("pitch", &[1, frames])?;
        context.set_input_shape("pitchf", &[1, frames])?;
        context.set_input_shape("nsf_noise", &[1, 1, frames * up])?;
    }
    tracing::debug!(target: cli_logging::LOG_TARGET, "test input shapes set");
    let mut addresses = vec![
        ("feats", &feats),
        ("p_len", &p_len),
        ("sid", &sid),
        ("rnd", &rnd),
        ("audio", &audio),
    ];
    if let (Some(pitch), Some(pitchf), Some(noise)) = (&pitch, &pitchf, &noise) {
        addresses.extend([("pitch", pitch), ("pitchf", pitchf), ("nsf_noise", noise)]);
    }
    for (name, buffer) in addresses {
        context.set_tensor_address(name, buffer.device_ptr())?;
    }
    tracing::debug!(target: cli_logging::LOG_TARGET, "test tensor addresses set");
    let start = Instant::now();
    context.enqueue(&stream)?;
    tracing::debug!(target: cli_logging::LOG_TARGET, "test enqueue returned");
    stream.synchronize()?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut audio_bytes = vec![0_u8; audio.bytes];
    audio.download_bytes(&mut audio_bytes)?;
    let (audio_chunks, audio_remainder) = audio_bytes.as_chunks::<4>();
    let values = audio_chunks
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    if let Some(path) = trt_config.diagnostics.dump_audio {
        fs::write(path, &audio_bytes)?;
    }
    if !audio_remainder.is_empty() || values.iter().any(|sample| !sample.is_finite()) {
        return Err("native TensorRT output contains a non-finite sample".into());
    }
    let max_abs = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    let mean_abs =
        values.iter().map(|value| value.abs() as f64).sum::<f64>() / values.len().max(1) as f64;
    println!(
        "{}",
        serde_json::json!({
            "plan": plan,
            "frames": frames,
            "upsample": up,
            "output_capacity_bytes": audio.bytes,
            "elapsed_ms": elapsed_ms,
            "finite": true,
            "max_abs": max_abs,
            "mean_abs": mean_abs
        })
    );
    Ok(())
}

#[cfg(feature = "tensorrt-rtx")]
struct BenchmarkInput {
    frames: i64,
    feature_width: i64,
    latent: i64,
    upsample: i64,
    feats: Vec<u8>,
    p_len: Vec<u8>,
    sid: Vec<u8>,
    rnd: Vec<u8>,
    pitch: Vec<u8>,
    pitchf: Vec<u8>,
    noise: Vec<u8>,
}

#[cfg(feature = "tensorrt-rtx")]
fn take_bytes<'a>(bytes: &mut &'a [u8], count: usize) -> CliResult<&'a [u8]> {
    if bytes.len() < count {
        return Err("benchmark input is truncated".into());
    }
    let (head, tail) = bytes.split_at(count);
    *bytes = tail;
    Ok(head)
}

#[cfg(feature = "tensorrt-rtx")]
fn read_benchmark_input(path: &PathBuf) -> CliResult<BenchmarkInput> {
    let mut raw = Vec::new();
    fs::File::open(path)?.read_to_end(&mut raw)?;
    let mut bytes = raw.as_slice();
    if take_bytes(&mut bytes, 8)? != b"VFRTEST1" {
        return Err("invalid benchmark input magic".into());
    }
    let read_u32 = |bytes: &mut &[u8]| -> CliResult<u32> {
        Ok(u32::from_le_bytes(take_bytes(bytes, 4)?.try_into()?))
    };
    let frames = i64::from(read_u32(&mut bytes)?);
    let feature_width = i64::from(read_u32(&mut bytes)?);
    let latent = i64::from(read_u32(&mut bytes)?);
    let upsample = i64::from(read_u32(&mut bytes)?);
    let _p_len = read_u32(&mut bytes)?;
    let _sid = read_u32(&mut bytes)?;
    if !(1..=2048).contains(&frames)
        || !(1..=4096).contains(&feature_width)
        || !(1..=4096).contains(&latent)
        || !(1..=4096).contains(&upsample)
    {
        return Err("benchmark dimensions are outside supported bounds".into());
    }
    let bytes_for = |elements: i64| -> CliResult<usize> {
        usize::try_from(elements.checked_mul(4).ok_or("benchmark size overflow")?)
            .map_err(|_| "benchmark size exceeds host usize".into())
    };
    let feats = take_bytes(&mut bytes, bytes_for(frames * feature_width)?)?.to_vec();
    let p_len = take_bytes(&mut bytes, 4)?.to_vec();
    let sid = take_bytes(&mut bytes, 4)?.to_vec();
    let rnd = take_bytes(&mut bytes, bytes_for(frames * latent)?)?.to_vec();
    let pitch = take_bytes(&mut bytes, bytes_for(frames)?)?.to_vec();
    let pitchf = take_bytes(&mut bytes, bytes_for(frames)?)?.to_vec();
    let noise = take_bytes(&mut bytes, bytes_for(frames * upsample)?)?.to_vec();
    if !bytes.is_empty() {
        return Err("benchmark input has trailing bytes".into());
    }
    Ok(BenchmarkInput {
        frames,
        feature_width,
        latent,
        upsample,
        feats,
        p_len,
        sid,
        rnd,
        pitch,
        pitchf,
        noise,
    })
}

#[cfg(feature = "tensorrt-rtx")]
fn benchmark_trt(
    plan: PathBuf,
    input: PathBuf,
    output: PathBuf,
    iterations: usize,
) -> CliResult<()> {
    let trt_config = vocal_tensorrt::TensorRtConfig::from_env()?;
    if iterations == 0 {
        return Err("benchmark iterations must be positive".into());
    }
    let input = read_benchmark_input(&input)?;
    let runtime = vocal_tensorrt::Runtime::load(&plan)?;
    let mut context = runtime.context()?;
    let stream = vocal_tensorrt::CudaStream::new()?;
    let alloc = |bytes: usize| -> CliResult<vocal_tensorrt::CudaBuffer> {
        Ok(vocal_tensorrt::CudaBuffer::new(bytes.max(4))?)
    };
    let feats = alloc(input.feats.len())?;
    let p_len = alloc(input.p_len.len())?;
    let sid = alloc(input.sid.len())?;
    let rnd = alloc(input.rnd.len())?;
    let pitch = alloc(input.pitch.len())?;
    let pitchf = alloc(input.pitchf.len())?;
    let noise = alloc(input.noise.len())?;
    let debug_outputs = trt_config.diagnostics.debug_graph_outputs;
    let mut debug_buffers: Vec<(String, vocal_tensorrt::CudaBuffer, usize)> = Vec::new();
    let output_bytes = usize::try_from(
        input
            .frames
            .checked_mul(input.upsample)
            .and_then(|x| x.checked_mul(4))
            .ok_or("benchmark output size overflow")?,
    )?;
    let audio = alloc(output_bytes + 16 * 1024)?;
    for (buffer, bytes) in [
        (&feats, input.feats.as_slice()),
        (&p_len, input.p_len.as_slice()),
        (&sid, input.sid.as_slice()),
        (&rnd, input.rnd.as_slice()),
        (&pitch, input.pitch.as_slice()),
        (&pitchf, input.pitchf.as_slice()),
        (&noise, input.noise.as_slice()),
    ] {
        buffer.upload_bytes(bytes)?;
    }
    context.set_input_shape("feats", &[1, input.frames, input.feature_width])?;
    context.set_input_shape("rnd", &[1, input.latent, input.frames])?;
    context.set_input_shape("pitch", &[1, input.frames])?;
    context.set_input_shape("pitchf", &[1, input.frames])?;
    context.set_input_shape("nsf_noise", &[1, 1, input.frames * input.upsample])?;
    if debug_outputs {
        let names = [
            "encoder_z",
            "flow_z",
            "nsf_source",
            "decoder_pre",
            "decoder_up_0",
            "decoder_noise_0",
            "decoder_res_0",
            "decoder_up_1",
            "decoder_noise_1",
            "decoder_res_1",
            "decoder_up_2",
            "decoder_noise_2",
            "decoder_res_2",
            "decoder_up_3",
            "decoder_noise_3",
            "decoder_res_3",
            "audio_pre_tanh",
            "decoder_post_input",
        ];
        for name in names {
            let shape = context.tensor_shape(name)?;
            let elements = shape
                .iter()
                .try_fold(1_usize, |acc, dimension| {
                    usize::try_from(*dimension)
                        .ok()
                        .and_then(|value| acc.checked_mul(value))
                })
                .ok_or_else(|| format!("invalid debug output shape for {name}"))?;
            let bytes = elements
                .checked_mul(4)
                .ok_or("debug output size overflow")?;
            debug_buffers.push((name.to_owned(), alloc(bytes)?, bytes));
        }
    }
    for (name, buffer) in [
        ("feats", &feats),
        ("p_len", &p_len),
        ("sid", &sid),
        ("rnd", &rnd),
        ("pitch", &pitch),
        ("pitchf", &pitchf),
        ("nsf_noise", &noise),
        ("audio", &audio),
    ] {
        context.set_tensor_address(name, buffer.device_ptr())?;
    }
    for (name, buffer, _) in &debug_buffers {
        context.set_tensor_address(name, buffer.device_ptr())?;
    }
    for _ in 0..5 {
        context.enqueue(&stream)?;
        stream.synchronize()?;
    }
    let mut timings = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        context.enqueue(&stream)?;
        stream.synchronize()?;
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let mut sorted = timings.clone();
    sorted.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| -> f64 {
        let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
        sorted[index]
    };
    let mean = timings.iter().sum::<f64>() / timings.len() as f64;
    let mut output_bytes_host = vec![0_u8; output_bytes];
    audio.download_bytes(&mut output_bytes_host)?;
    fs::write(&output, output_bytes_host)?;
    for (name, buffer, bytes) in &debug_buffers {
        let mut values = vec![0_u8; *bytes];
        buffer.download_bytes(&mut values)?;
        fs::write(output.with_extension(format!("{name}.f32")), &values)?;
    }
    let max_ms = sorted
        .last()
        .copied()
        .ok_or("benchmark produced no timing samples")?;
    println!(
        "{}",
        serde_json::json!({
            "backend": "tensorrt-rtx-native",
            "frames": input.frames,
            "output_samples": input.frames * input.upsample,
            "iterations": iterations,
            "mean_ms": mean,
            "median_ms": percentile(0.50),
            "p95_ms": percentile(0.95),
            "min_ms": sorted[0],
            "max_ms": max_ms,
            "debug_outputs": debug_outputs,
        })
    );
    Ok(())
}

#[cfg(not(feature = "tensorrt-rtx"))]
fn test_trt(_plan: PathBuf, _cache: Option<PathBuf>) -> CliResult<()> {
    Err("TensorRT-RTX feature is disabled; rebuild with --features tensorrt-rtx".into())
}

#[cfg(not(feature = "tensorrt-rtx"))]
fn benchmark_trt(
    _plan: PathBuf,
    _input: PathBuf,
    _output: PathBuf,
    _iterations: usize,
) -> CliResult<()> {
    Err("TensorRT-RTX feature is disabled; rebuild with --features tensorrt-rtx".into())
}

fn demo(blocks: usize) -> CliResult<()> {
    let format = AudioFormat::MONO_16K;
    let context = RuntimeContext::new(RuntimeConfig {
        input_format: format,
        output_format: format,
        backend: RuntimeBackend::RvcCpu,
        ..RuntimeConfig::default()
    })?;
    context.load_model_descriptor(ModelDescriptor {
        id: "fixture".into(),
        display_name: "Fixture".into(),
        architecture: "rvc-v2".into(),
        sample_rate: format.sample_rate,
        version: "0.1".into(),
    })?;
    context.start_processing()?;
    for sequence in 0..blocks {
        let input = AudioChunk {
            format,
            samples: vec![0.0; 160],
            sequence: sequence as u64,
        };
        let _output = context.process(&input)?;
    }
    let snapshot = context.snapshot()?;
    println!(
        "processed {} blocks; metrics={:?}",
        blocks, snapshot.metrics
    );
    Ok(())
}
