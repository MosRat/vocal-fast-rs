//! Idle preparation of CPU graphs from the same model bundle used by GPU paths.
use crate::{RuntimeConfig, RuntimeError, RuntimeResult};
use vocal_engine::{CpuEngineSpec, EngineConfig, InferenceEngine};

pub(crate) fn create(
    config: &RuntimeConfig,
    descriptor: &vocal_types::ModelDescriptor,
) -> RuntimeResult<Box<dyn InferenceEngine>> {
    let model_path = config.model_path.as_ref().ok_or_else(|| {
        RuntimeError::Config("CPU inference requires a complete VFRM model path".into())
    })?;
    let sources = vocal_engine::offline::OfflineModelSources::from_bundle(
        model_path,
        vocal_engine::offline::OfflineModelSources::default_cache_root(),
    )?;
    tracing::info!(
        threads = config.cpu.resolved_threads(),
        window = config.cpu_window_samples(),
        pin = config.cpu.pin_to_performance_cores,
        "preparing standalone oneDNN CPU engine"
    );
    Ok(vocal_engine::create_engine(EngineConfig::with_cpu_graph(
        descriptor.clone(),
        CpuEngineSpec {
            synth: sources.synth,
            hubert: sources.hubert,
            rmvpe: sources.rmvpe,
            execution: config.cpu.clone(),
            window_samples: config.cpu_window_samples(),
            speaker_id: usize::try_from(config.speaker_id)
                .map_err(|_| RuntimeError::Config("CPU speaker ID must be nonnegative".into()))?,
        },
    ))?)
}
