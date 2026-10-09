//! Real five-stage builds and cache hits, without changing the GUI configuration.
use serde_json::json;
use std::{error::Error, path::PathBuf, time::Instant};
use vocal_runtime::RuntimeConfig;
use vocal_tensorrt::{PrecisionMode, RvcTrtPipelineArtifacts, RvcTrtProfile, TensorRtConfig};

#[cfg(windows)]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args.len() != 3 {
        return Err(
            "usage: plan_build_probe <runtime.json> <private-cache-directory> <report.json>".into(),
        );
    }
    let runtime: RuntimeConfig = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let gui = &runtime.tensorrt;
    let start = Instant::now();
    let package =
        vocal_model::load_voice_cache_shared(runtime.model_path.as_ref().ok_or("missing model")?)?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    let mut config = TensorRtConfig::from_env()?;
    config.profile = RvcTrtProfile::from_name(&gui.profile)?;
    config.build.precision = PrecisionMode::from_name(&gui.precision);
    config.build.fp16 = config.build.precision != PrecisionMode::Fp32;
    config.build.hubert_precision = PrecisionMode::from_name(&gui.hubert_precision);
    config.build.hubert_feature_precision = config.build.hubert_precision;
    config.build.rmvpe_precision = PrecisionMode::from_name(&gui.rmvpe_precision);
    config.build.rmvpe_encoder_precision = config.build.rmvpe_precision;
    config.build.rmvpe_decoder_precision = config.build.rmvpe_precision;
    config.build.synth_precision = PrecisionMode::from_name(&gui.synth_precision);
    config.build.synth_full_fp16 = gui.synth_full_fp16;
    config.build.rmvpe_fft_cufft = gui.rmvpe_fft.eq_ignore_ascii_case("cufft");
    config.build.rmvpe_fft_radix2 = gui.rmvpe_fft.eq_ignore_ascii_case("radix2");
    let cache = args[1].join("plans");
    let output = args[1].join("output");
    let mut runs = Vec::new();
    for name in ["first", "hit", "synth-change"] {
        if name == "synth-change" {
            config.diagnostics.debug_graph_outputs = true;
        }
        let start = Instant::now();
        let mut stages = Vec::new();
        let artifacts = RvcTrtPipelineArtifacts::compile_with_config_and_plan_cache(
            &config,
            &package,
            gui.hubert_source.as_ref().ok_or("missing HuBERT")?,
            gui.rmvpe_source.as_ref().ok_or("missing RMVPE")?,
            &output,
            Some(&cache),
            |stage| {
                if stage.finished {
                    let row = json!({"stage":stage.name,"cache_hit":stage.cache_hit,"elapsed_ms":stage.elapsed_ms});
                    println!("{row}");
                    stages.push(row);
                }
            },
        )?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
        let manifest = args[1].join(format!("{name}-pipeline.json"));
        std::fs::write(&manifest, serde_json::to_vec_pretty(&artifacts)?)?;
        let row = json!({"run":name,"elapsed_ms":elapsed_ms,"stages":stages,"manifest":manifest});
        println!("{row}");
        runs.push(row);
    }
    if let Some(parent) = args[2].parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        &args[2],
        serde_json::to_vec_pretty(
            &json!({"model_load_ms":load_ms,"builder":vocal_tensorrt::BUILDER_FINGERPRINT,"runs":runs}),
        )?,
    )?;
    Ok(())
}
