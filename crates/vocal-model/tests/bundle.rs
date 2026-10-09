use std::{
    fs,
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
};
use vocal_model::*;

struct Fixture {
    root: PathBuf,
    synth: PathBuf,
    hubert: PathBuf,
    rmvpe: PathBuf,
    bundle: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        Self::with_speakers(1)
    }
    fn with_speakers(speakers: u64) -> Self {
        let root = std::env::temp_dir().join(format!(
            "vocal-bundle-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let mut fixture = Self {
            synth: root.join("synth.vfrm"),
            hubert: root.join("hubert.vfrm"),
            rmvpe: root.join("rmvpe.vfrm"),
            bundle: root.join("complete.vfrm"),
            root,
        };
        fixture.write(speakers);
        fixture
    }
    fn write(&mut self, speakers: u64) {
        let manifest = RvcModelManifest {
            id: "test".into(),
            display_name: "Voice".into(),
            architecture: "rvc-v2".into(),
            version: "v2".into(),
            sample_rate: 32000,
            model_file: "voice.pth".into(),
            index_file: None,
            source: "test".into(),
            notes: None,
        };
        let config = RvcSynthConfig {
            feature_width: 768,
            uses_f0: true,
            sample_rate: 32000,
            hidden_channels: 2,
            latent_channels: 2,
            filter_channels: 2,
            attention_heads: 1,
            encoder_layers: 1,
            encoder_kernel: 1,
            relative_window: 1,
            relative_heads: 1,
            norm_epsilon: 1e-5,
            speakers,
            speaker_channels: 1,
            decoder_channels: 2,
            resblock_type: 2,
            resblock_kernels: vec![1],
            resblock_dilations: vec![vec![1, 1]],
            upsample_rates: vec![2],
            upsample_kernels: vec![2],
            flow_blocks: 1,
            flow_layers: 1,
            flow_kernel: 1,
            flow_dilations: vec![1],
        };
        let mut tensors: Vec<_> = config
            .tensor_shapes()
            .unwrap()
            .into_iter()
            .map(|(name, shape)| {
                let len = shape.iter().product::<u64>() as usize * 4;
                Tensor {
                    name,
                    dtype: TensorDType::F32,
                    shape,
                    data: vec![0; len],
                }
            })
            .collect();
        tensors.push(Tensor::byte_tensor("aux.retrieval.faiss", vec![1, 2, 3, 4]));
        write_cache(&self.synth, &ModelPackage { manifest: manifest.clone(), metadata: serde_json::json!({"canonical_schema": RVC_TENSOR_SCHEMA, "rvc_config": config}), tensors }).unwrap();
        for (path, kind) in [(&self.hubert, "hubert"), (&self.rmvpe, "rmvpe")] {
            let name = format!("{kind}.weight");
            let tensor = Tensor {
                name: name.clone(),
                dtype: TensorDType::F32,
                shape: vec![2],
                data: vec![0; 8],
            };
            let metadata = serde_json::json!({"canonical_schema": ACOUSTIC_TENSOR_SCHEMA, "kind": kind, "graph_schema": "test", "node_count": 1, "inputs": [{"name":"audio","shape":[1,160],"dtype":1}], "outputs":[{"name":"result","shape":[1,2],"dtype":1}], "tensor_map":[{"canonical":name,"source":"weight"}]});
            write_cache(
                path,
                &ModelPackage {
                    manifest: manifest.clone(),
                    metadata,
                    tensors: vec![tensor],
                },
            )
            .unwrap();
        }
        bundle_rvc_cache(&self.synth, &self.hubert, &self.rmvpe, &self.bundle).unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn v1_bundle_requires_projection_and_preserves_feature_identity() {
    let fixture = Fixture::new();
    let mut synth = load_cache(&fixture.synth).unwrap();
    synth.manifest.version = "v1".into();
    synth.manifest.architecture = "rvc-v1".into();
    synth.metadata["rvc_config"]["feature_width"] = 256.into();
    let phone = synth
        .tensors
        .iter_mut()
        .find(|t| t.name == "enc_p.emb_phone.weight")
        .unwrap();
    phone.shape[0] = 256;
    phone.data.resize(256 * 2 * 4, 0);
    write_cache(&fixture.synth, &synth).unwrap();
    let output = fixture.root.join("v1-complete.vfrm");
    let error =
        bundle_rvc_cache(&fixture.synth, &fixture.hubert, &fixture.rmvpe, &output).unwrap_err();
    assert!(error.to_string().contains("final_proj"));
    assert!(!output.exists());
    let mut hubert = load_cache(&fixture.hubert).unwrap();
    for (name, shape) in [
        ("hubert.model.final_proj.weight", vec![768, 256]),
        ("hubert.model.final_proj.bias", vec![256]),
    ] {
        hubert.tensors.push(Tensor {
            name: name.into(),
            dtype: TensorDType::F32,
            data: vec![0; shape.iter().product::<u64>() as usize * 4],
            shape,
        });
    }
    synth.metadata["graph_schema"] = "rvc-v1-f0-feats".into();
    write_cache(&fixture.synth, &synth).unwrap();
    write_cache(&fixture.hubert, &hubert).unwrap();
    bundle_rvc_cache(&fixture.synth, &fixture.hubert, &fixture.rmvpe, &output).unwrap();
    let inspection = inspect_model_artifact(&output).unwrap();
    assert_eq!(inspection.manifest.rvc_version().unwrap(), RvcVersion::V1);
    assert!(inspection.components.automatic_tensor_rt);
    assert_eq!(
        load_cache(&output)
            .unwrap()
            .validate_rvc_synth()
            .unwrap()
            .feature_width,
        256
    );
    let mut graph_mismatch = load_cache(&output).unwrap();
    graph_mismatch.metadata["graph_schema"] = "rvc-v2-f0-feats".into();
    let graph_mismatch_path = fixture.root.join("v1-graph-mismatch.vfrm");
    let mut unchecked = ModelLoadConfig::default();
    unchecked.validate_canonical = false;
    write_cache_with_config(&graph_mismatch_path, &graph_mismatch, &unchecked).unwrap();
    assert!(load_cache(&graph_mismatch_path).is_err());
    let paths = materialize_acoustic_sources(&output, fixture.root.join("cache")).unwrap();
    load_cache(&paths[0])
        .unwrap()
        .validate_hubert_for_rvc(RvcVersion::V1)
        .unwrap();
    let mut mismatched = synth.clone();
    mismatched.manifest.version = "v2".into();
    assert!(mismatched.validate_rvc_synth().is_err());
    // An explicitly v2-only acoustic package must not be silently selected
    // for v1 even when its tensor directory happens to contain the projection.
    hubert.metadata["rvc_feature_modes"] = serde_json::json!([{
        "version":"v2", "output_layer":12, "feature_width":768, "projection":null
    }]);
    assert!(hubert.validate_hubert_for_rvc(RvcVersion::V1).is_err());
    write_cache(&fixture.hubert, &hubert).unwrap();
    assert!(
        bundle_rvc_cache(
            &fixture.synth,
            &fixture.hubert,
            &fixture.rmvpe,
            fixture.root.join("unsupported.vfrm")
        )
        .is_err()
    );
}

#[test]
fn concurrent_cache_loads_share_ownership_and_release_after_unload() {
    use std::sync::{Arc, Barrier};
    let fixture = Fixture::new();
    let cache = ModelPackageCache::new();
    let barrier = Barrier::new(8);
    let packages = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    cache
                        .load_voice(&fixture.bundle, &ModelLoadConfig::default())
                        .unwrap()
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(
        packages
            .iter()
            .all(|package| Arc::ptr_eq(&packages[0], package))
    );
    assert_eq!(cache.len(), 1);
    drop(packages);
    assert!(cache.is_empty());
    let broken = fixture.root.join("broken.vfrm");
    fs::write(&broken, b"invalid cache").unwrap();
    assert!(
        cache
            .load_voice(&broken, &ModelLoadConfig::default())
            .is_err()
    );
    fs::copy(&fixture.bundle, &broken).unwrap();
    assert!(
        cache
            .load_voice(&broken, &ModelLoadConfig::default())
            .is_ok()
    );
}

#[test]
fn inventory_selective_load_and_extraction_preserve_components() {
    let fixture = Fixture::new();
    let inspection = inspect_model_artifact(&fixture.bundle).unwrap();
    assert!(inspection.components.automatic_tensor_rt);
    assert!(inspection.components.retrieval.present);
    assert_eq!(inspection.components.hubert.size_bytes, 8);
    let cache = ModelPackageCache::new();
    let complete = cache
        .load(&fixture.bundle, &ModelLoadConfig::default())
        .unwrap();
    let voice = cache
        .load_voice(&fixture.bundle, &ModelLoadConfig::default())
        .unwrap();
    assert!(complete.tensor(HUBERT_BUNDLE_TENSOR).is_some());
    assert!(voice.tensor(HUBERT_BUNDLE_TENSOR).is_none());
    assert!(voice.tensor(RMVPE_BUNDLE_TENSOR).is_none());
    assert_eq!(
        voice.tensor("aux.retrieval.faiss").unwrap().data,
        [1, 2, 3, 4]
    );
    assert!(voice.validate_rvc_synth().is_ok());
    assert_eq!(
        cache.len(),
        2,
        "full and voice-only cache entries must be distinct"
    );
    let paths = materialize_acoustic_sources(&fixture.bundle, fixture.root.join("cache")).unwrap();
    assert_eq!(
        fs::read(&paths[0]).unwrap(),
        fs::read(&fixture.hubert).unwrap()
    );
    assert_eq!(
        fs::read(&paths[1]).unwrap(),
        fs::read(&fixture.rmvpe).unwrap()
    );
    let modified = fs::metadata(&paths[0]).unwrap().modified().unwrap();
    assert_eq!(
        materialize_acoustic_sources(&fixture.bundle, fixture.root.join("cache")).unwrap(),
        paths
    );
    assert_eq!(
        fs::metadata(&paths[0]).unwrap().modified().unwrap(),
        modified
    );
    fs::write(&paths[0], b"truncated").unwrap();
    materialize_acoustic_sources(&fixture.bundle, fixture.root.join("cache")).unwrap();
    assert_eq!(
        fs::read(&paths[0]).unwrap(),
        fs::read(&fixture.hubert).unwrap()
    );
}

#[test]
fn incomplete_and_wrong_kind_sources_do_not_claim_automatic_readiness() {
    let fixture = Fixture::new();
    let voice = inspect_model_artifact(&fixture.synth).unwrap();
    assert!(!voice.components.automatic_tensor_rt);
    assert!(!voice.components.hubert.present);
    assert!(voice.components.synth.present);
    assert!(
        bundle_rvc_cache(
            &fixture.synth,
            &fixture.rmvpe,
            &fixture.hubert,
            fixture.root.join("bad.vfrm")
        )
        .is_err()
    );
    assert!(
        bundle_rvc_cache(
            &fixture.synth,
            &fixture.hubert,
            &fixture.rmvpe,
            &fixture.synth
        )
        .is_err()
    );
    assert!(
        !inspect_model_artifact(&fixture.hubert)
            .unwrap()
            .components
            .synth
            .present
    );
}

#[test]
fn inventory_reports_canonical_speaker_count_and_leaves_unknown_counts_unset() {
    for speakers in [1, 3] {
        let fixture = Fixture::with_speakers(speakers);
        for path in [&fixture.synth, &fixture.bundle] {
            let inspection = inspect_model_artifact(path).unwrap();
            assert_eq!(inspection.components.speaker_count, Some(speakers));
            assert_eq!(
                serde_json::to_value(&inspection.components).unwrap()["speakerCount"],
                speakers
            );
        }
        assert_eq!(
            inspect_model_artifact(&fixture.hubert)
                .unwrap()
                .components
                .speaker_count,
            None
        );
        let legacy = fixture.root.join("legacy.vfrm");
        let package = load_cache(&fixture.synth).unwrap();
        write_cache(
            &legacy,
            &ModelPackage {
                manifest: package.manifest,
                metadata: serde_json::Value::Null,
                tensors: vec![Tensor::byte_tensor("rvc.pth", vec![1, 2])],
            },
        )
        .unwrap();
        let legacy = inspect_model_artifact(legacy).unwrap();
        assert!(legacy.components.synth.present);
        assert_eq!(legacy.components.speaker_count, None);
    }
}

#[test]
fn inventory_rejects_oversized_directory_and_truncated_payload_before_loading() {
    let fixture = Fixture::new();
    let path = fixture.root.join("broken.vfrm");
    fs::copy(&fixture.bundle, &path).unwrap();
    let mut file = fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(32)).unwrap();
    file.write_all(&u64::MAX.to_le_bytes()).unwrap();
    assert!(inspect_model_artifact(&path).is_err());
    drop(file);
    fs::copy(&fixture.bundle, &path).unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(1024)
        .unwrap();
    assert!(inspect_model_artifact(&path).is_err());
    assert!(load_cache(&path).is_err());
}
