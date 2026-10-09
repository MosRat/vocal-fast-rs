//! Tauri-facing TensorRT pipeline preparation.
//!
//! The GUI only sends semantic paths and profile choices. This module owns
//! cache-key construction, VFRM loading, native plan compilation, and atomic
//! manifest publication so the application state machine never exposes a
//! half-built pipeline.

use std::path::PathBuf;

use serde::Deserialize;

use crate::error::{GuiError, GuiResult};
use crate::preparation::PreparationOperation;
#[cfg(feature = "tensorrt-rtx")]
use crate::preparation::PreparationPhase;

#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PrepareTensorRtRequest {
    pub expected_revision: u64,
    pub model_path: Option<String>,
    pub hubert_source: Option<String>,
    pub rmvpe_source: Option<String>,
    pub cache_directory: Option<String>,
}

#[derive(Clone, Debug)]
pub struct PreparedTensorRt {
    pub artifacts_path: PathBuf,
    pub model_path: PathBuf,
    pub config: vocal_runtime::TensorRtPipelineConfig,
    pub cache_hit: bool,
}

#[cfg(not(feature = "tensorrt-rtx"))]
pub fn prepare(
    _config: &vocal_runtime::RuntimeConfig,
    _request: PrepareTensorRtRequest,
    _progress: &PreparationOperation,
) -> GuiResult<PreparedTensorRt> {
    Err(GuiError::Engine(
        "TensorRT pipeline preparation requires the Tauri tensorrt-rtx feature".into(),
    ))
}

#[cfg(feature = "tensorrt-rtx")]
pub fn prepare(
    config: &vocal_runtime::RuntimeConfig,
    request: PrepareTensorRtRequest,
    progress: &PreparationOperation,
) -> GuiResult<PreparedTensorRt> {
    use std::{
        collections::hash_map::DefaultHasher,
        fs,
        hash::{Hash, Hasher},
        path::Path,
        time::SystemTime,
    };

    use vocal_model::{
        ModelLoader, inspect_model_artifact, load_voice_cache_shared, materialize_acoustic_sources,
        resolve_model_path,
    };
    use vocal_tensorrt::{
        GraphReplayMode, PrecisionMode, RvcTrtPipelineArtifacts, RvcTrtProfile, TensorRtConfig,
    };

    let mut gui = config.tensorrt.clone();
    if let Some(value) = request.hubert_source {
        gui.hubert_source = Some(value);
    }
    if let Some(value) = request.rmvpe_source {
        gui.rmvpe_source = Some(value);
    }
    if let Some(value) = request.cache_directory {
        gui.cache_directory = Some(value);
    }
    let model_path = request
        .model_path
        .or_else(|| config.model_path.clone())
        .ok_or_else(|| GuiError::InvalidConfig("a VFRM model path is required".into()))?;
    let cache_root = gui
        .cache_directory
        .clone()
        .ok_or_else(|| GuiError::InvalidConfig("a TensorRT cache directory is required".into()))?;

    progress.report(
        PreparationPhase::Inspecting,
        "Inspecting the model bundle and acoustic components",
    );
    let model_path = resolve_model_path(Path::new(&model_path)).map_err(GuiError::from)?;
    let inspection = inspect_model_artifact(&model_path).map_err(GuiError::from)?;
    if inspection.components.automatic_tensor_rt {
        progress.report(
            PreparationPhase::Extracting,
            "Checking cached acoustic sources and extracting HuBERT / RMVPE when needed",
        );
        let [hubert, rmvpe] =
            materialize_acoustic_sources(&model_path, &cache_root).map_err(GuiError::from)?;
        gui.hubert_source = Some(hubert.display().to_string());
        gui.rmvpe_source = Some(rmvpe.display().to_string());
    }
    let hubert_source = resolve_existing_file(gui.hubert_source.as_deref().ok_or_else(|| GuiError::InvalidConfig("this voice model has no embedded HuBERT weights; select a HuBERT source or a complete bundle".into()))?, "HuBERT source")?;
    let rmvpe_source = resolve_existing_file(gui.rmvpe_source.as_deref().ok_or_else(|| GuiError::InvalidConfig("this voice model has no embedded RMVPE weights; select an RMVPE source or a complete bundle".into()))?, "RMVPE source")?;
    gui.hubert_source = Some(hubert_source.display().to_string());
    gui.rmvpe_source = Some(rmvpe_source.display().to_string());
    let profile = RvcTrtProfile::from_name(&gui.profile)
        .map_err(|error| GuiError::InvalidConfig(error.to_string()))?;
    let precision = parse_precision(&gui.precision)?;
    let hubert_precision = parse_precision(&gui.hubert_precision)?;
    let rmvpe_precision = parse_precision(&gui.rmvpe_precision)?;
    let synth_precision = parse_precision(&gui.synth_precision)?;
    let graph_mode = GraphReplayMode::parse(&gui.graph_mode).ok_or_else(|| {
        GuiError::InvalidConfig(format!("invalid TensorRT graph mode: {}", gui.graph_mode))
    })?;
    let mut trt =
        TensorRtConfig::from_env().map_err(|error| GuiError::InvalidConfig(error.to_string()))?;
    let version = inspection.manifest.rvc_version().map_err(GuiError::from)?;
    trt.build.rvc_version = version;
    trt.profile = profile.clone();
    trt.build.precision = precision;
    trt.build.fp16 = precision != PrecisionMode::Fp32;
    trt.build.hubert_precision = hubert_precision;
    trt.build.hubert_feature_precision = hubert_precision;
    trt.build.rmvpe_precision = rmvpe_precision;
    trt.build.rmvpe_encoder_precision = rmvpe_precision;
    trt.build.rmvpe_decoder_precision = rmvpe_precision;
    trt.build.synth_precision = synth_precision;
    trt.build.synth_full_fp16 = gui.synth_full_fp16;
    trt.build.min_samples = profile.min_samples;
    trt.build.opt_samples = profile.opt_samples;
    trt.build.max_samples = profile.max_samples;
    trt.build.min_frames = profile.min_rvc_frames;
    trt.build.opt_frames = profile.opt_rvc_frames;
    trt.build.max_frames = profile.max_rvc_frames;
    trt.build.rmvpe_fft_radix2 = gui.rmvpe_fft.eq_ignore_ascii_case("radix2");
    trt.build.rmvpe_fft_cufft =
        gui.rmvpe_fft.eq_ignore_ascii_case("cufft") || gui.rmvpe_fft.eq_ignore_ascii_case("cuda");
    if !matches!(
        gui.rmvpe_fft.to_ascii_lowercase().as_str(),
        "none" | "radix2" | "cufft" | "cuda"
    ) {
        return Err(GuiError::InvalidConfig(format!(
            "invalid RMVPE FFT mode: {}",
            gui.rmvpe_fft
        )));
    }
    trt.runtime.graph_mode = graph_mode;
    trt.runtime.cuda_graph = gui.cuda_graph;
    trt.runtime.native_cuda_graph = gui.native_cuda_graph;
    trt.runtime.graph_cache_size = gui.graph_cache_size;
    trt.runtime.parallel_acoustic = gui.parallel_acoustic;
    trt.runtime.async_audio_d2h = gui.async_audio_d2h;
    trt.validate()
        .map_err(|error| GuiError::InvalidConfig(error.to_string()))?;

    let mut hasher = DefaultHasher::new();
    "vocal-trt-pipeline-cache-v3".hash(&mut hasher);
    std::env::var_os("NVIDIA_TF32_OVERRIDE").hash(&mut hasher);
    vocal_tensorrt::BUILDER_FINGERPRINT.hash(&mut hasher);
    serde_json::to_vec(&trt.build)?.hash(&mut hasher);
    serde_json::to_vec(&trt.profile)?.hash(&mut hasher);
    let backend = vocal_runtime::detect_tensor_rt_backend();
    if !backend.available {
        return Err(GuiError::Engine(
            backend
                .reason
                .unwrap_or_else(|| "TensorRT RTX is unavailable".into()),
        ));
    }
    backend.device.hash(&mut hasher);
    for sdk in ["tensorrt_rtx_1_6.dll"] {
        if let Ok(exe) = std::env::current_exe() {
            let path = exe.parent().unwrap_or(Path::new(".")).join(sdk);
            if let Ok(metadata) = fs::metadata(path) {
                metadata.len().hash(&mut hasher);
                metadata.modified().ok().hash(&mut hasher);
            }
        }
    }
    inspection.manifest.id.hash(&mut hasher);
    gui.profile.hash(&mut hasher);
    gui.precision.hash(&mut hasher);
    gui.hubert_precision.hash(&mut hasher);
    gui.rmvpe_precision.hash(&mut hasher);
    gui.synth_precision.hash(&mut hasher);
    gui.rmvpe_fft.hash(&mut hasher);
    gui.synth_full_fp16.hash(&mut hasher);
    for source in [&model_path, &hubert_source, &rmvpe_source] {
        source.display().to_string().hash(&mut hasher);
        let metadata = fs::metadata(source).map_err(GuiError::from)?;
        metadata.len().hash(&mut hasher);
        metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos())
            .hash(&mut hasher);
    }
    let fingerprint = hasher.finish();
    let model_id = safe_component(&inspection.manifest.id);
    let output_dir = PathBuf::from(&cache_root).join(model_id).join(format!(
        "{}-{fingerprint:016x}",
        safe_component(&gui.profile)
    ));
    let manifest_path = output_dir.join("pipeline.json");
    progress.report(
        PreparationPhase::CheckingCache,
        "Checking the pipeline fingerprint and cached plan files",
    );
    if load_valid_artifacts(&manifest_path).is_some_and(|a| a.rvc_version == version) {
        progress.cache_hit(true);
        progress.report(
            PreparationPhase::CheckingCache,
            "Cache hit · reusing all compiled plans",
        );
        let mut ready = gui;
        ready.artifacts_path = Some(manifest_path.display().to_string());
        return Ok(PreparedTensorRt {
            artifacts_path: manifest_path,
            model_path,
            config: ready,
            cache_hit: true,
        });
    }

    // Inspect/fingerprint first: a complete pipeline hit does not need to
    // decode hundreds of MiB of Synth weights or deserialize its FAISS index.
    progress.report(
        PreparationPhase::LoadingWeights,
        "Reading Synth weights for the plans that need rebuilding",
    );
    let mut loader = ModelLoader::new(Default::default()).map_err(GuiError::from)?;
    let package = if model_path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("vfr") || extension.eq_ignore_ascii_case("vfrm")
        }) {
        load_voice_cache_shared(&model_path).map_err(GuiError::from)?
    } else {
        let manifest = loader.load_manifest(&model_path).map_err(GuiError::from)?;
        let root = model_path.parent().unwrap_or_else(|| Path::new("."));
        std::sync::Arc::new(
            loader
                .load_rvc_directory(root, manifest)
                .map_err(GuiError::from)?,
        )
    };
    package
        .validate_rvc_synth()
        .map_err(|error| GuiError::InvalidConfig(error.to_string()))?;

    let parent = output_dir
        .parent()
        .ok_or_else(|| GuiError::Internal("invalid TensorRT cache path".into()))?;
    fs::create_dir_all(parent).map_err(GuiError::from)?;
    let temp_dir = parent.join(format!(".build-{}-{fingerprint:016x}", std::process::id()));
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir).map_err(GuiError::from)?;
    }
    fs::create_dir_all(&temp_dir).map_err(GuiError::from)?;
    let plan_cache = PathBuf::from(&cache_root).join("plans");
    let mut reused_all = true;
    let result = (|| -> GuiResult<()> {
        let artifacts = RvcTrtPipelineArtifacts::compile_with_config_and_plan_cache(
            &trt,
            &package,
            &hubert_source,
            &rmvpe_source,
            &temp_dir,
            Some(&plan_cache),
            |stage| {
                if stage.finished {
                    reused_all &= stage.cache_hit;
                }
                progress.report_steps(
                    PreparationPhase::Building,
                    format!(
                        "{} · {}",
                        if stage.cache_hit {
                            "Reused from cache"
                        } else if stage.finished {
                            "Compiled"
                        } else {
                            "Compiling"
                        },
                        stage.name
                    ),
                    stage.completed,
                    Some(stage.total),
                );
            },
        )
        .map_err(|error| GuiError::Engine(error.to_string()))?;
        progress.report(
            PreparationPhase::Publishing,
            "Validating and publishing the compiled pipeline to the cache",
        );
        let artifacts = relocate_artifacts(artifacts, &temp_dir, &output_dir);
        atomic_json_write(&temp_dir.join("pipeline.json"), &artifacts)?;
        if output_dir.exists() {
            fs::remove_dir_all(&output_dir).map_err(GuiError::from)?;
        }
        fs::rename(&temp_dir, &output_dir).map_err(GuiError::from)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&temp_dir);
    }
    result?;
    let mut ready = gui;
    ready.artifacts_path = Some(manifest_path.display().to_string());
    Ok(PreparedTensorRt {
        artifacts_path: manifest_path,
        model_path,
        config: ready,
        cache_hit: reused_all,
    })
}

#[cfg(feature = "tensorrt-rtx")]
fn resolve_existing_file(value: &str, label: &str) -> GuiResult<PathBuf> {
    let path = PathBuf::from(value);
    if !path.is_file() {
        return Err(GuiError::InvalidConfig(format!(
            "{label} does not exist: {}",
            path.display()
        )));
    }
    std::fs::canonicalize(path).map_err(GuiError::from)
}

#[cfg(feature = "tensorrt-rtx")]
fn parse_precision(value: &str) -> GuiResult<vocal_tensorrt::PrecisionMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "fp32" | "float32" | "float" => Ok(vocal_tensorrt::PrecisionMode::Fp32),
        "fp16" | "float16" | "half" => Ok(vocal_tensorrt::PrecisionMode::Fp16),
        "bf16" | "bfloat16" => Ok(vocal_tensorrt::PrecisionMode::Bf16),
        _ => Err(GuiError::InvalidConfig(format!(
            "invalid TensorRT precision: {value}"
        ))),
    }
}

#[cfg(feature = "tensorrt-rtx")]
fn safe_component(value: &str) -> String {
    let value: String = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect();
    if value.is_empty() {
        "model".into()
    } else {
        value
    }
}

#[cfg(feature = "tensorrt-rtx")]
fn load_valid_artifacts(
    manifest_path: &std::path::Path,
) -> Option<vocal_tensorrt::RvcTrtPipelineArtifacts> {
    let artifacts: vocal_tensorrt::RvcTrtPipelineArtifacts =
        serde_json::from_slice(&std::fs::read(manifest_path).ok()?).ok()?;
    let mut plans = vec![
        &artifacts.hubert_plan,
        &artifacts.rmvpe_plan,
        &artifacts.frontend_plan,
        &artifacts.synth_plan,
    ];
    if let Some(path) = artifacts.rmvpe_realtime_plan.as_ref() {
        plans.push(path);
    }
    if let Some(path) = artifacts.rmvpe_full_plan.as_ref() {
        plans.push(path);
    }
    if plans
        .iter()
        .all(|path| std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.len() > 0))
    {
        Some(artifacts)
    } else {
        None
    }
}

#[cfg(feature = "tensorrt-rtx")]
fn relocate_artifacts(
    mut artifacts: vocal_tensorrt::RvcTrtPipelineArtifacts,
    from: &std::path::Path,
    to: &std::path::Path,
) -> vocal_tensorrt::RvcTrtPipelineArtifacts {
    fn move_path(path: &mut PathBuf, from: &std::path::Path, to: &std::path::Path) {
        if let Ok(relative) = path.strip_prefix(from) {
            *path = to.join(relative);
        }
    }
    move_path(&mut artifacts.hubert_plan, from, to);
    move_path(&mut artifacts.rmvpe_plan, from, to);
    if let Some(path) = artifacts.rmvpe_realtime_plan.as_mut() {
        move_path(path, from, to);
    }
    if let Some(path) = artifacts.rmvpe_full_plan.as_mut() {
        move_path(path, from, to);
    }
    move_path(&mut artifacts.frontend_plan, from, to);
    move_path(&mut artifacts.synth_plan, from, to);
    artifacts
}

#[cfg(feature = "tensorrt-rtx")]
fn atomic_json_write<T: serde::Serialize>(path: &std::path::Path, value: &T) -> GuiResult<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    let temp = path.with_extension("json.tmp");
    std::fs::write(&temp, bytes).map_err(GuiError::from)?;
    std::fs::rename(&temp, path).map_err(GuiError::from)?;
    Ok(())
}

#[cfg(all(test, feature = "tensorrt-rtx"))]
mod tests {
    use super::*;
    use crate::{
        events::EventSink,
        preparation::{PreparationKind, PreparationTracker},
    };
    use std::{sync::Arc, time::Instant};

    #[test]
    #[ignore = "requires complete local weights and a TensorRT-RTX GPU; uses private test paths"]
    fn profile_tensor_rt_prepare_cache() {
        let config_path =
            std::env::var_os("VOCAL_TEST_TRT_CONFIG").expect("set VOCAL_TEST_TRT_CONFIG");
        let cache =
            std::env::var("VOCAL_TEST_TRT_CACHE_DIR").expect("set VOCAL_TEST_TRT_CACHE_DIR");
        let report = std::env::var_os("VOCAL_TEST_TRT_REPORT").expect("set VOCAL_TEST_TRT_REPORT");
        let mut config: vocal_runtime::RuntimeConfig =
            serde_json::from_slice(&std::fs::read(config_path).unwrap()).unwrap();
        let tracker = Arc::new(PreparationTracker::new(Arc::new(EventSink::default())));
        let mut runs = Vec::new();
        let mut first_path = None;
        for name in ["first", "hit", "runtime-change"] {
            if name == "runtime-change" {
                config.tensorrt.graph_mode = "disabled".into();
            }
            let operation = tracker.begin(PreparationKind::PrepareTensorRt);
            let start = Instant::now();
            let ready = prepare(
                &config,
                PrepareTensorRtRequest {
                    expected_revision: 0,
                    model_path: None,
                    hubert_source: None,
                    rmvpe_source: None,
                    cache_directory: Some(cache.clone()),
                },
                &operation,
            )
            .unwrap();
            let row = serde_json::json!({"run":name,"elapsed_ms":start.elapsed().as_secs_f64()*1000.,"cache_hit":ready.cache_hit,"path":ready.artifacts_path});
            println!("{row}");
            runs.push(row);
            if name != "first" {
                assert!(ready.cache_hit);
                assert_eq!(Some(&ready.artifacts_path), first_path.as_ref());
                assert_eq!(tracker.snapshot().phase, PreparationPhase::CheckingCache);
            } else {
                first_path = Some(ready.artifacts_path.clone());
            }
            assert_eq!(ready.config.graph_mode, config.tensorrt.graph_mode);
            operation.report(PreparationPhase::Ready, "Cache preparation verified");
        }
        std::fs::write(report, serde_json::to_vec_pretty(&runs).unwrap()).unwrap();
    }
}
