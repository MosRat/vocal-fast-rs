//! Explicit manifests override automatic extraction from a complete bundle.
use crate::{RuntimeConfig, RuntimeError, RuntimeResult};
use vocal_directml::{DirectMlConfig, DirectMlSources};
use vocal_engine::{DirectMlEngine, DirectMlStreamConfig, EngineConfig, InferenceEngine};

pub(crate) fn create(
    config: &RuntimeConfig,
    model: &vocal_types::ModelDescriptor,
) -> RuntimeResult<Box<dyn InferenceEngine>> {
    let speaker = usize::try_from(config.speaker_id)
        .map_err(|_| RuntimeError::Config("DirectML speaker ID must be nonnegative".into()))?;
    let adapters = vocal_directml::enumerate_adapters().map_err(|error| {
        vocal_engine::EngineError::BackendUnavailable(format!("DirectML device discovery: {error}"))
    })?;
    let adapter = if let Some(id) = config.directml_device_id.as_deref() {
        adapters.iter().find(|a| a.id.eq_ignore_ascii_case(id)).ok_or_else(|| {
            vocal_engine::EngineError::BackendUnavailable("Selected DirectML GPU is no longer available. Refresh devices and select a GPU.".into())
        })?
    } else {
        adapters
            .iter()
            .find(|a| {
                a.available
                    && crate::capabilities::directml_device_kind(&a.id, a.software)
                        == crate::ProcessingDeviceKind::Gpu
            })
            .or_else(|| adapters.iter().find(|a| a.available))
            .ok_or_else(|| {
                vocal_engine::EngineError::BackendUnavailable(
                    "No compatible DirectML GPU found".into(),
                )
            })?
    };
    if !adapter.available {
        return Err(vocal_engine::EngineError::BackendUnavailable(
            adapter
                .reason
                .clone()
                .unwrap_or_else(|| "Selected DirectML GPU is unavailable".into()),
        )
        .into());
    }
    if let Some(path) = config
        .directml_sources_path
        .as_deref()
        .filter(|p| !p.is_empty())
    {
        let mut engine = EngineConfig::with_directml(model.clone(), path);
        let spec = engine.directml.as_mut().unwrap();
        spec.speaker_id = speaker;
        spec.adapter_id = Some(adapter.id.clone());
        return Ok(vocal_engine::create_engine(engine)?);
    }
    let path = config.model_path.as_deref().ok_or_else(|| {
        RuntimeError::Config("DirectML requires a complete VFRM bundle or a source manifest".into())
    })?;
    let (sources, manifest) = DirectMlSources::prepare_bundle(
        path,
        DirectMlSources::default_cache_root(),
        DirectMlConfig {
            adapter_id: Some(adapter.id.clone()),
            ..DirectMlConfig::realtime_44960()
        },
    )
    .map_err(|error| {
        vocal_engine::EngineError::BackendUnavailable(format!(
            "DirectML bundle preparation: {error}"
        ))
    })?;
    tracing::info!(path = %manifest.display(), "prepared DirectML bundle sources");
    Ok(Box::new(DirectMlEngine::from_sources(
        model.clone(),
        &sources,
        DirectMlStreamConfig::default(),
        speaker,
    )?))
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires DirectML and the prepared real bundle/Chinese F32 input"]
    fn bundle_loading_worker_handoff_and_small_callbacks() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let input = std::fs::read(root.join("target/dml-chinese-stream.f32")).unwrap();
        let wave = input
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        let config = RuntimeConfig {
            backend: crate::RuntimeBackend::DirectMl,
            backend_auto: false,
            input_format: vocal_types::AudioFormat::MONO_16K,
            output_format: vocal_types::AudioFormat::MONO_16K,
            conversion: vocal_types::ConversionConfig {
                index_ratio: 0.5,
                ..Default::default()
            },
            ..Default::default()
        };
        let controls = config.conversion;
        let runtime = crate::RuntimeContext::new(config).unwrap();
        let loaded = runtime
            .load_model(
                root.join("target/default-complete-with-index.vfrm")
                    .display()
                    .to_string(),
            )
            .unwrap();
        assert!(loaded.config.directml_sources_path.is_none());
        runtime.start_processing().unwrap();
        let (generation, mut engine) = runtime.take_engine_for_worker().unwrap();
        runtime.activate_worker(generation).unwrap();
        let started = std::time::Instant::now();
        let mut energy = 0f64;
        for (sequence, block) in wave.chunks(160).enumerate() {
            let audio = engine
                .process_samples(
                    vocal_types::AudioFormat::MONO_16K,
                    block,
                    sequence as u64,
                    controls,
                )
                .unwrap();
            assert_eq!(audio.samples.len(), block.len());
            assert!(audio.samples.iter().all(|v| v.is_finite()));
            energy += audio
                .samples
                .iter()
                .map(|v| f64::from(*v).powi(2))
                .sum::<f64>();
        }
        assert!(
            energy > 1e-8,
            "no converted audio after startup context filled"
        );
        let metrics = engine.metrics();
        assert_eq!(engine.status().failures, 0);
        runtime.complete_worker(generation, engine, None).unwrap();
        runtime.stop_processing().unwrap();
        runtime.unload_model().unwrap();
        std::fs::write(root.join("target/dml-runtime-stream.json"), serde_json::to_vec_pretty(&serde_json::json!({"callback_samples":160,"input_samples":wave.len(),"output_energy":energy,"elapsed_ms":started.elapsed().as_secs_f64()*1000.,"metrics":metrics})).unwrap()).unwrap();
    }
}
