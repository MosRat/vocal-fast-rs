use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use vocal_engine::{EngineError, PreparedEngineReload};
use vocal_types::{AudioFormat, ConversionConfig};

struct ProbeEngine {
    applied: Arc<AtomicUsize>,
    status: EngineStatus,
}

struct ProbeReload<'a> {
    engine: &'a mut ProbeEngine,
    value: usize,
}

impl PreparedEngineReload for ProbeReload<'_> {
    fn commit(self: Box<Self>) {
        self.engine.applied.store(self.value, Ordering::SeqCst);
    }
}

impl InferenceEngine for ProbeEngine {
    fn validate_config(&self, config: ConversionConfig) -> Result<(), EngineError> {
        vocal_engine::validate_conversion(config)?;
        if config.index_ratio > 0.0 {
            return Err(EngineError::InvalidConfig("fixture has no index".into()));
        }
        Ok(())
    }

    fn prepare_reload_config(
        &mut self,
        config: EngineReloadConfig,
    ) -> Result<Box<dyn PreparedEngineReload + '_>, EngineError> {
        let trt = config.tensorrt.unwrap();
        if trt.artifacts_path == Path::new("missing") {
            return Err(EngineError::BackendUnavailable("missing pipeline".into()));
        }
        Ok(Box::new(ProbeReload {
            engine: self,
            value: trt.graph_cache_size,
        }))
    }

    fn process(
        &mut self,
        input: &AudioChunk,
        _: ConversionConfig,
    ) -> Result<AudioChunk, EngineError> {
        self.status.success();
        Ok(input.clone())
    }

    fn metrics(&self) -> LatencyMetrics {
        LatencyMetrics::default()
    }

    fn status(&self) -> EngineStatus {
        self.status.clone()
    }
}

fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: "reload-fixture".into(),
        display_name: "Reload fixture".into(),
        architecture: "rvc-v2".into(),
        sample_rate: 40000,
        version: "v2".into(),
    }
}

fn probe_runtime() -> (RuntimeContext, Arc<AtomicUsize>) {
    let runtime = RuntimeContext::new(RuntimeConfig::default()).unwrap();
    runtime.load_model_descriptor(descriptor()).unwrap();
    let applied = Arc::new(AtomicUsize::new(2));
    let probe = applied.clone();
    runtime
        .request(move |state| {
            state.config.backend = RuntimeBackend::TensorRtRtx;
            state.config.tensorrt.artifacts_path = Some("pipeline".into());
            state.engine_config = Some(state.config.clone());
            state.engine = Some(Box::new(ProbeEngine {
                applied: probe,
                status: EngineStatus {
                    state: LifecycleState::Ready,
                    ..Default::default()
                },
            }));
            Ok(())
        })
        .unwrap();
    (runtime, applied)
}

fn temp_directory() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "vocal-idle-reload-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&path).unwrap();
    path
}

#[test]
fn validation_and_preparation_fail_before_persistence_or_revision_changes() {
    let (runtime, applied) = probe_runtime();
    let directory = temp_directory();
    let store = crate::ConfigStore::new(directory.join("config.json"));
    store.save(&runtime.config().unwrap()).unwrap();
    let saved = fs::read(store.path()).unwrap();
    runtime
        .request(move |state| {
            state.config_store = Some(store);
            Ok(())
        })
        .unwrap();
    let original = runtime.snapshot().unwrap();
    for missing_pipeline in [false, true] {
        let mut candidate = original.config.clone();
        if missing_pipeline {
            candidate.tensorrt.artifacts_path = Some("missing".into());
        } else {
            candidate.conversion.index_ratio = 0.5;
        }
        assert!(
            runtime
                .update_config(candidate, original.config_revision)
                .is_err()
        );
        let actual = runtime.snapshot().unwrap();
        assert_eq!(actual.config_revision, original.config_revision);
        assert_eq!(actual.lifecycle, original.lifecycle);
        assert_eq!(actual.engine.unwrap().failures, 0);
        assert_eq!(applied.load(Ordering::SeqCst), 2);
        assert_eq!(fs::read(directory.join("config.json")).unwrap(), saved);
    }
    fs::remove_file(directory.join("config.json")).unwrap();
    fs::remove_dir(directory).unwrap();
}

#[test]
fn save_failure_discards_prepared_reload_and_retry_commits_once() {
    let (runtime, applied) = probe_runtime();
    let directory = temp_directory();
    let path = directory.join("config.json");
    fs::create_dir(&path).unwrap(); // Atomic replacement must fail against a directory.
    let store = crate::ConfigStore::new(&path);
    runtime
        .request(move |state| {
            state.config_store = Some(store);
            Ok(())
        })
        .unwrap();
    let original = runtime.snapshot().unwrap();
    let mut candidate = original.config.clone();
    candidate.tensorrt.graph_cache_size = 3;
    assert!(matches!(
        runtime.update_config(candidate.clone(), original.config_revision),
        Err(RuntimeError::Io { .. })
    ));
    assert_eq!(applied.load(Ordering::SeqCst), 2);
    assert_eq!(
        runtime.snapshot().unwrap().config_revision,
        original.config_revision
    );
    assert_eq!(runtime.config().unwrap().tensorrt.graph_cache_size, 2);
    fs::remove_dir(&path).unwrap();
    let committed = runtime
        .update_config(candidate, original.config_revision)
        .unwrap();
    assert_eq!(
        committed.last_config_update,
        ConfigUpdateKind::ReloadTensorRtPipeline
    );
    assert_eq!(applied.load(Ordering::SeqCst), 3);
    assert_eq!(
        crate::ConfigStore::new(&path)
            .load()
            .unwrap()
            .unwrap()
            .tensorrt
            .graph_cache_size,
        3
    );
    fs::remove_file(path).unwrap();
    fs::remove_dir(directory).unwrap();
}

#[test]
fn pending_plan_survives_control_edits_and_can_be_reverted_or_published() {
    let (runtime, applied) = probe_runtime();
    let original = runtime.snapshot().unwrap();
    let mut candidate = original.config.clone();
    candidate.tensorrt.hubert_precision = "fp32".into();
    candidate.tensorrt.graph_cache_size = 3;
    let pending = runtime
        .update_config(candidate.clone(), original.config_revision)
        .unwrap();
    assert_eq!(
        pending.last_config_update,
        ConfigUpdateKind::PlanBuildRequired
    );
    candidate.conversion.protect = 0.5;
    let pending = runtime
        .update_config(candidate.clone(), pending.config_revision)
        .unwrap();
    assert_eq!(
        pending.last_config_update,
        ConfigUpdateKind::PlanBuildRequired
    );
    assert_eq!(applied.load(Ordering::SeqCst), 2);
    let reverted = runtime
        .update_config(original.config.clone(), pending.config_revision)
        .unwrap();
    assert_eq!(reverted.last_config_update, ConfigUpdateKind::Unchanged);
    let pending = runtime
        .update_config(candidate.clone(), reverted.config_revision)
        .unwrap();
    candidate.tensorrt.artifacts_path = Some("published-pipeline".into());
    let published = runtime
        .update_config(candidate, pending.config_revision)
        .unwrap();
    assert_eq!(
        published.last_config_update,
        ConfigUpdateKind::ReloadTensorRtPipeline
    );
    assert_eq!(applied.load(Ordering::SeqCst), 3);
}

#[test]
fn every_portable_backend_keeps_engine_identity_and_clears_stream_on_restart() {
    for backend in [RuntimeBackend::Passthrough, RuntimeBackend::RvcCpu] {
        let runtime = RuntimeContext::new(RuntimeConfig {
            backend,
            ..Default::default()
        })
        .unwrap();
        runtime.load_model_descriptor(descriptor()).unwrap();
        for (iteration, protect) in [0.0, 0.33, 0.5, 1.0, 0.0].into_iter().enumerate() {
            let before = runtime.snapshot().unwrap();
            let mut config = before.config;
            config.conversion.protect = protect;
            config.conversion.pitch_semitones = iteration as f32;
            config.conversion.auto_pitch = vocal_types::AutoPitchConfig {
                enabled: iteration % 2 == 0,
                target_hz: if iteration % 2 == 0 { 255. } else { 155. },
            };
            config.input_format = if iteration % 2 == 0 {
                AudioFormat::MONO_16K
            } else {
                AudioFormat::MONO_48K
            };
            let snapshot = runtime
                .update_config(config.clone(), before.config_revision)
                .unwrap();
            assert_eq!(snapshot.engine.unwrap().processed_chunks, iteration as u64);
            runtime.start_processing().unwrap();
            let output = runtime
                .process(&AudioChunk {
                    format: config.input_format,
                    samples: vec![0.1; 480],
                    sequence: iteration as u64,
                })
                .unwrap();
            assert!(output.samples.iter().all(|v| v.is_finite()));
            assert!(
                runtime
                    .update_config(config, snapshot.config_revision)
                    .is_err()
            );
            runtime.stop_processing().unwrap();
        }
    }
}

#[test]
fn automatic_pitch_policy_is_transactional_idle_data_not_an_engine_reload() {
    let (runtime, applied) = probe_runtime();
    // The fake engine exercises actor ownership, without requiring the
    // TensorRT Cargo feature merely to enter the Running lifecycle.
    runtime
        .request(|state| {
            state.config.backend = RuntimeBackend::Passthrough;
            state.engine_config = Some(state.config.clone());
            Ok(())
        })
        .unwrap();
    for (enabled, target_hz) in [(false, 255.), (true, 155.), (true, 255.)] {
        let before = runtime.snapshot().unwrap();
        let mut next = before.config.clone();
        next.conversion.auto_pitch = vocal_types::AutoPitchConfig { enabled, target_hz };
        let changed = runtime
            .update_config(next.clone(), before.config_revision)
            .unwrap();
        assert_eq!(changed.last_config_update, ConfigUpdateKind::HotReload);
        assert_eq!(
            changed.config.conversion.auto_pitch,
            next.conversion.auto_pitch
        );
        assert_eq!(applied.load(Ordering::SeqCst), 2);
        runtime.start_processing().unwrap();
        next.conversion.auto_pitch.enabled = !enabled;
        assert!(
            runtime
                .update_config(next, changed.config_revision)
                .is_err()
        );
        assert_eq!(
            runtime.config().unwrap().conversion.auto_pitch,
            changed.config.conversion.auto_pitch
        );
        runtime.stop_processing().unwrap();
    }
}

#[test]
fn unloading_or_rebinding_store_cannot_destroy_state_when_save_fails() {
    let runtime = RuntimeContext::new(RuntimeConfig::default()).unwrap();
    runtime.load_model_descriptor(descriptor()).unwrap();
    assert!(
        runtime
            .set_config_store(crate::ConfigStore::new("unused-config.json"))
            .is_err()
    );
    let directory = temp_directory();
    let path = directory.join("config.json");
    fs::create_dir(&path).unwrap();
    let store = crate::ConfigStore::new(&path);
    runtime
        .request(move |state| {
            state.config_store = Some(store);
            Ok(())
        })
        .unwrap();
    let original = runtime.snapshot().unwrap();
    let mut replacement = original.config.clone();
    replacement.backend = RuntimeBackend::RvcCpu;
    assert!(
        runtime
            .update_config(replacement, original.config_revision)
            .is_err()
    );
    assert_eq!(
        runtime.config().unwrap().backend,
        RuntimeBackend::Passthrough
    );
    assert!(runtime.unload_model().is_err());
    let actual = runtime.snapshot().unwrap();
    assert_eq!(actual.config_revision, original.config_revision);
    assert!(actual.model.is_some());
    assert!(actual.engine.is_some());
    fs::remove_dir(path).unwrap();
    fs::remove_dir(directory).unwrap();
}

#[test]
fn worker_ownership_blocks_idle_updates_until_the_engine_returns() {
    let runtime = RuntimeContext::new(RuntimeConfig::default()).unwrap();
    runtime.load_model_descriptor(descriptor()).unwrap();
    runtime.start_processing().unwrap();
    let (generation, engine) = runtime.take_engine_for_worker().unwrap();
    runtime.stop_processing().unwrap();
    let mut config = runtime.config().unwrap();
    config.conversion.protect = 0.5;
    assert!(runtime.update_config(config.clone(), 1).is_err());
    runtime.complete_worker(generation, engine, None).unwrap();
    assert_eq!(
        runtime.update_config(config, 1).unwrap().last_config_update,
        ConfigUpdateKind::HotReload
    );
}

#[test]
fn duplicate_start_or_error_clear_cannot_reset_an_active_actor_engine() {
    let runtime = RuntimeContext::new(RuntimeConfig::default()).unwrap();
    runtime.load_model_descriptor(descriptor()).unwrap();
    runtime.start_processing().unwrap();
    assert!(runtime.start_processing().is_err());
    assert!(runtime.clear_error().is_err());
    assert_eq!(
        runtime.snapshot().unwrap().lifecycle,
        LifecycleState::Running
    );
    runtime
        .process(&AudioChunk {
            format: AudioFormat::MONO_16K,
            samples: vec![0.1; 160],
            sequence: 0,
        })
        .unwrap();
}

#[test]
fn prepared_build_at_same_manifest_path_commits_internal_reload() {
    let (runtime, applied) = probe_runtime();
    let original = runtime.snapshot().unwrap();
    let mut config = original.config.clone();
    config.tensorrt.hubert_precision = "fp32".into();
    config.tensorrt.graph_cache_size = 3;
    let pending = runtime
        .update_config(config.clone(), original.config_revision)
        .unwrap();
    assert_eq!(
        pending.last_config_update,
        ConfigUpdateKind::PlanBuildRequired
    );
    let reloaded = runtime
        .apply_prepared_tensorrt_pipeline(config, pending.config_revision)
        .unwrap();
    assert_eq!(
        reloaded.last_config_update,
        ConfigUpdateKind::ReloadTensorRtPipeline
    );
    assert_eq!(applied.load(Ordering::SeqCst), 3);
    assert_eq!(reloaded.lifecycle, LifecycleState::Ready);
}
