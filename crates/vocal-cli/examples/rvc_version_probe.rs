//! Real-model feature selection and full-pipeline parity for all native backends.
//! Run backends serially; outputs retain exact stochastic inputs for Python.
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};

fn decode(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 6 {
        return Err("rvc_version_probe <cpu|directml|tensorrt> synth.vfrm hubert.vfrm rmvpe.vfrm wave.f32 report.json [fp32|mixed] [index_ratio] [iterations]".into());
    }
    // Set the strict FP32 policy before CUDA initialization; library code must
    // never mutate an embedding application's process environment.
    if args[0] == "tensorrt" && std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args(std::env::args_os().skip(1))
            .env("NVIDIA_TF32_OVERRIDE", "0")
            .status()?;
        return if status.success() {
            Ok(())
        } else {
            Err(format!("TensorRT probe exited with {status}").into())
        };
    }
    let wave = decode(&std::fs::read(&args[4])?);
    let mixed = match args.get(6).map(String::as_str).unwrap_or("fp32") {
        "fp32" => false,
        "mixed" => true,
        _ => return Err("precision must be fp32 or mixed".into()),
    };
    let ratio = args.get(7).map(|s| s.parse()).transpose()?.unwrap_or(0f32);
    let iterations = args
        .get(8)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(5usize)
        .max(1);
    let synth = vocal_model::load_voice_cache_shared(&args[1])?;
    let sc = synth.validate_rvc_synth()?;
    let version = synth.manifest.rvc_version()?;
    let mut tensors = BTreeMap::<String, Vec<f32>>::new();
    let pitch: Vec<i32>;
    let config: Value;
    let mut latencies = Vec::new();
    let mut graph_stats = Value::Null;
    let controls = vocal_types::ConversionConfig {
        auto_pitch: vocal_types::AutoPitchConfig::disabled(),
        index_ratio: ratio,
        protect: 0.33,
        ..Default::default()
    };
    match args[0].as_str() {
        "cpu" => {
            if mixed {
                return Err(
                    "CPU probe supports fp32 only; use CPU recipe benchmarks for W8A8".into(),
                );
            }
            let policy = vocal_cpu::CpuConfig {
                window_samples: wave.len(),
                skip_head_frames: 0,
                append_hubert_frame: false,
                ..Default::default()
            };
            config = serde_json::to_value(&policy)?;
            drop(synth);
            let mut p = vocal_cpu::CpuPipeline::from_sources(
                &vocal_cpu::CpuSources {
                    synth: PathBuf::from(&args[1]),
                    hubert: PathBuf::from(&args[2]),
                    rmvpe: PathBuf::from(&args[3]),
                },
                policy,
            )?;
            for _ in 0..2 {
                p.process(&wave, controls, 0, false)?;
            }
            for _ in 0..iterations {
                latencies.push(p.process(&wave, controls, 0, false)?.total_ms);
            }
            let out = p.process(&wave, controls, 0, true)?;
            pitch = out.debug["pitch"]
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            tensors.extend(
                out.debug
                    .into_iter()
                    .filter(|(k, _)| k != "pitch")
                    .map(|(k, v)| (k, decode(&v))),
            );
        }
        "directml" => {
            let policy = vocal_directml::DirectMlConfig {
                window_samples: wave.len(),
                skip_head_frames: 0,
                append_hubert_frame: false,
                debug_outputs: true,
                precision: if mixed {
                    Default::default()
                } else {
                    vocal_directml::PrecisionConfig::fp32()
                },
                ..Default::default()
            };
            config = serde_json::to_value(&policy)?;
            drop(synth);
            let mut p =
                vocal_directml::DirectMlPipeline::from_sources(&vocal_directml::DirectMlSources {
                    synth: PathBuf::from(&args[1]),
                    hubert: PathBuf::from(&args[2]),
                    rmvpe: PathBuf::from(&args[3]),
                    enable_retrieval: ratio > 0.,
                    config: policy,
                })?;
            let mut controls = vocal_directml::DirectMlControls::new(p.frames);
            controls.index_ratio = ratio;
            for _ in 0..2 {
                p.process(&wave, &controls)?;
            }
            for _ in 0..iterations {
                latencies.push(p.process(&wave, &controls)?.total_ms);
            }
            let out = p.process(&wave, &controls)?;
            let pitch_bytes = &out
                .debug
                .tensors
                .iter()
                .find(|(n, _)| n == "pitch")
                .ok_or("missing pitch")?
                .1;
            pitch = pitch_bytes
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            tensors.extend(
                out.debug
                    .tensors
                    .into_iter()
                    .filter(|(k, _)| k != "pitch")
                    .map(|(k, v)| (k, decode(&v))),
            );
        }
        "tensorrt" => {
            use vocal_tensorrt::{
                PrecisionMode, RvcTrtPipeline, RvcTrtPipelineArtifacts, RvcTrtProfile,
                TensorRtConfig,
            };
            let mut policy = TensorRtConfig::default();
            policy.build.rvc_version = version;
            policy.profile = RvcTrtProfile::for_offline(wave.len())?;
            policy.build.rmvpe_fft_cufft = true;
            if mixed {
                policy.build.hubert_precision = PrecisionMode::Fp16;
            } else {
                policy.build.synth_precision = PrecisionMode::Fp32;
                policy.build.synth_resblock_precision = PrecisionMode::Fp32;
                policy.build.synth_resblock_stage_mask = 0;
                policy.build.synth_fp16_noise = false;
            }
            config = json!({"window_samples":wave.len(), "skip_head_frames":0, "append_hubert_frame":false,
                "f0":{"mode":"full_window"}, "tensorrt":policy});
            let output = PathBuf::from(&args[5]);
            let plan_dir = output.with_extension("plans");
            let artifacts = RvcTrtPipelineArtifacts::compile_with_config(
                &policy, &synth, &args[2], &args[3], &plan_dir,
            )?;
            std::fs::write(
                plan_dir.join("pipeline.json"),
                serde_json::to_vec_pretty(&artifacts)?,
            )?;
            let index = vocal_model::RetrievalIndex::from_package(&synth)?;
            drop(synth);
            let mut p = RvcTrtPipeline::load_with_runtime_config_and_retrieval(
                &artifacts,
                sc.clone(),
                policy.runtime,
                policy.diagnostics,
                index.as_ref(),
            )?;
            p.set_retrieval_ratio(ratio)?;
            p.set_protect(controls.protect)?;
            p.set_pitch_controls(0., controls.auto_pitch)?;
            for _ in 0..2 {
                p.process(&wave, 0)?;
            }
            for _ in 0..iterations {
                latencies.push(p.process(&wave, 0)?.timings.total_ms);
            }
            p.enable_debug_intermediates(true);
            let out = p.process(&wave, 0)?;
            pitch = out.pitch.ok_or("missing pitch")?;
            let features = out.hubert_features.ok_or("missing features")?;
            if ratio == 0. {
                tensors.insert("hubert_raw".into(), features.clone());
            }
            tensors.insert("hubert_features".into(), features);
            tensors.insert("feats".into(), out.rvc_features.ok_or("missing feats")?);
            tensors.insert("rmvpe_pitchf".into(), out.pitchf.ok_or("missing f0")?);
            tensors.insert(
                "pitchf".into(),
                out.aligned_pitchf.ok_or("missing aligned f0")?,
            );
            tensors.insert(
                "synth_random".into(),
                out.synth_random.ok_or("missing random")?,
            );
            tensors.insert(
                "synth_noise".into(),
                out.synth_noise.ok_or("missing noise")?,
            );
            tensors.insert("audio".into(), out.audio);
            graph_stats = serde_json::to_value(p.cuda_graph_stats())?;
        }
        _ => return Err("unknown backend".into()),
    }
    if tensors.values().flatten().any(|v| !v.is_finite()) {
        return Err("non-finite pipeline output".into());
    }
    let frames = tensors["feats"].len() / version.feature_width();
    let report = json!({"backend":args[0], "rvc_version":version, "feature_width":version.feature_width(),
        "sample_rate":sc.sample_rate, "frames":frames, "config":config, "pitch":pitch,
        "controls":controls, "gpu_ms":latencies, "latency_kind":if args[0] == "directml" {"host including diagnostic readback"} else {"host production"},
        "graph_stats":graph_stats, "tensors":tensors});
    std::fs::write(&args[5], serde_json::to_vec(&report)?)?;
    println!(
        "{}",
        json!({"backend":args[0], "version":version, "frames":frames, "report":args[5]})
    );
    Ok(())
}
