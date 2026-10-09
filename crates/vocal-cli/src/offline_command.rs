//! WAV/CLI boundary for native-rate, long-segment offline conversion.
use super::cli_error::CliResult;
use std::path::PathBuf;
use vocal_engine::BackendKind;

pub fn run(
    backend: BackendKind,
    cache: PathBuf,
    input: PathBuf,
    output: Option<PathBuf>,
    artifacts: Option<PathBuf>,
) -> CliResult<()> {
    #[cfg(any(feature = "cpu-graph", feature = "directml", feature = "tensorrt-rtx"))]
    {
        #[cfg(feature = "tensorrt-rtx")]
        if backend == BackendKind::TensorRtRtx
            && std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0")
        {
            // RTX 1.6's native FP32 path needs the NVIDIA process override in
            // addition to clearFlag(kTF32) for parity. Start before CUDA loads;
            // never mutate the environment of a multithreaded Rust host.
            let status = std::process::Command::new(std::env::current_exe()?)
                .args(std::env::args_os().skip(1))
                .env("NVIDIA_TF32_OVERRIDE", "0")
                .status()?;
            return if status.success() {
                Ok(())
            } else {
                Err(format!("strict FP32 offline worker exited with {status}").into())
            };
        }
        use vocal_engine::offline::{
            AlignedOfflineConfig, OfflineModelSources, OfflineVoicePipeline, run_aligned_offline,
        };
        let mut config = AlignedOfflineConfig::default();
        config.conversion.pitch_semitones = env_float("VOCAL_OFFLINE_PITCH", 0.)?;
        config.conversion.auto_pitch.enabled = env_bool("VOCAL_AUTO_PITCH", true)?;
        config.conversion.auto_pitch.target_hz = env_float("VOCAL_AUTO_PITCH_TARGET", 255.)?;
        config.conversion.index_ratio = env_float("VOCAL_INDEX_RATIO", 0.)?;
        config.conversion.protect = env_float("VOCAL_OFFLINE_PROTECT", 0.5)?;
        config.conversion.rms_mix_rate = env_float("VOCAL_OFFLINE_RMS_MIX", 1.)?;
        config
            .conversion
            .validate()
            .map_err(|error| vocal_engine::EngineError::InvalidConfig(error.to_string()))?;
        let samples = super::read_f32_audio(&input)?;
        let root = OfflineModelSources::default_cache_root();
        // Source manifests are usable across backends; pipeline.json contains
        // compiled TRT plans and requires the complete VFRM for rebuilding a
        // different offline shape. Never substitute unrelated voice weights.
        let supplied = artifacts
            .as_ref()
            .map(|p| -> CliResult<serde_json::Value> {
                Ok(serde_json::from_slice(&std::fs::read(p)?)?)
            })
            .transpose()?;
        let sources = if supplied
            .as_ref()
            .is_some_and(|v| v.get("hubert").is_some() && v.get("rmvpe").is_some())
        {
            let value = supplied.as_ref().unwrap();
            let resolve = |name: &str| -> CliResult<PathBuf> {
                let path = PathBuf::from(
                    value[name]
                        .as_str()
                        .ok_or("source manifest path must be a string")?,
                );
                Ok(if path.is_absolute() {
                    path
                } else {
                    artifacts
                        .as_ref()
                        .unwrap()
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .join(path)
                })
            };
            OfflineModelSources {
                synth: cache.clone(),
                hubert: resolve("hubert")?,
                rmvpe: resolve("rmvpe")?,
            }
        } else {
            OfflineModelSources::from_bundle(&cache, &root)?
        };
        let trace = std::env::var_os("VOCAL_OFFLINE_TRACE_DIR").map(PathBuf::from);
        let (mut pipeline, backend_config): (Box<dyn OfflineVoicePipeline>, serde_json::Value) =
            match backend {
                #[cfg(feature = "cpu-graph")]
                BackendKind::RvcCpuPipeline => {
                    let mut cpu = vocal_cpu::CpuConfig::default();
                    if let Ok(value) = std::env::var("VOCAL_CPU_THREADS") {
                        cpu.threads = value
                            .parse()
                            .map_err(|_| "VOCAL_CPU_THREADS must be an integer")?;
                    }
                    cpu.profile = env_bool("VOCAL_CPU_PROFILE", false)?;
                    cpu.fuse = env_bool("VOCAL_CPU_FUSE", true)?;
                    cpu.channels_last = env_bool("VOCAL_CPU_CHANNELS_LAST", true)?;
                    let policy = serde_json::json!({"execution": "cpu_graph", "operator_library": {"name": "oneDNN", "version": vocal_cpu::operator_version()}, "precision": "fp32", "config": cpu});
                    let mut p =
                        vocal_engine::offline::CpuOfflinePipeline::new(sources.clone(), cpu)?;
                    p.trace_directory = trace.clone();
                    (Box::new(p), policy)
                }

                #[cfg(feature = "directml")]
                BackendKind::DirectMl => {
                    let dml = supplied
                        .as_ref()
                        .and_then(|v| v.get("config"))
                        .map(|v| serde_json::from_value(v.clone()))
                        .transpose()?
                        .unwrap_or_default();
                    let mut p =
                        vocal_engine::offline::DirectMlOfflinePipeline::new(sources.clone(), dml)?;
                    p.trace_directory = trace.clone();
                    let policy = serde_json::json!({"execution": "serial_stages", "precision": p.config.precision, "retrieval_search": p.config.retrieval_search});
                    (Box::new(p), policy)
                }
                #[cfg(feature = "tensorrt-rtx")]
                BackendKind::TensorRtRtx => {
                    let mut trt = vocal_tensorrt::TensorRtConfig::from_env()?;
                    // Long offline activations exceed the realtime builder's
                    // 1 GiB workspace. A supplied typed budget remains explicit.
                    trt.build.workspace_bytes = 2 << 30;
                    // Preserve precision from a supplied prepared artifact. The
                    // waveform geometry and dispatch belong to the new offline
                    // graph and are selected by the library for each segment.
                    if let Some(value) = &supplied {
                        let mut build = serde_json::to_value(&trt.build)?;
                        for (name, value) in value
                            .as_object()
                            .ok_or("artifact manifest must be an object")?
                        {
                            if build.get(name).is_some() {
                                build[name] = value.clone();
                            }
                        }
                        trt.build = serde_json::from_value(build)?;
                    }
                    trt.build.rmvpe_fft_cufft = true;
                    trt.build.rmvpe_fft_radix2 = false;
                    let mut p = vocal_engine::offline::TensorRtOfflinePipeline::new(
                        sources.clone(),
                        root.clone(),
                        trt,
                    )?;
                    let policy = serde_json::json!({"build": p.config.build, "runtime": p.config.runtime, "nvidia_tf32_override": "0"});
                    p.trace_directory = trace.clone();
                    (Box::new(p), policy)
                }
                _ => {
                    return Err(
                        "requested offline backend is not compiled into this executable".into(),
                    );
                }
            };
        let (audio, report) = run_aligned_offline(&mut *pipeline, &samples, &config)?;
        if let Some(path) = &output {
            vocal_audio::write_wav_pcm16(path, report.output_sample_rate, &audio)?;
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "mode": "applio_offline", "backend": backend, "input": input, "output": output,
                "sources": sources, "cache_root": root, "config": config, "backend_config": backend_config, "trace_directory": trace,
                "report": report, "elapsed_ms": report.processing_ms,
                "output_sample_rate": report.output_sample_rate, "output_samples": report.output_samples,
            }))?
        );
        return Ok(());
    }
    #[cfg(not(any(feature = "cpu-graph", feature = "directml", feature = "tensorrt-rtx")))]
    {
        let _ = (backend, cache, input, output, artifacts);
        Err("native offline conversion requires --features directml or tensorrt-rtx".into())
    }
}

#[cfg(any(feature = "cpu-graph", feature = "directml", feature = "tensorrt-rtx"))]
fn env_bool(name: &str, default: bool) -> CliResult<bool> {
    match std::env::var(name) {
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "on" => Ok(true),
            "0" | "false" | "off" => Ok(false),
            _ => Err(format!("{name} must be true/false or 1/0").into()),
        },
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("{name}: {error}").into()),
    }
}

#[cfg(any(feature = "cpu-graph", feature = "directml", feature = "tensorrt-rtx"))]
fn env_float(name: &str, default: f32) -> CliResult<f32> {
    match std::env::var(name) {
        Ok(value) => Ok(value
            .parse()
            .map_err(|_| format!("{name} must be a floating-point number"))?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("{name}: {error}").into()),
    }
}
