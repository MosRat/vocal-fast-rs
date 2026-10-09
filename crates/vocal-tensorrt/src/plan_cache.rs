//! Per-stage immutable plans. Runtime controls and retrieval indices are not
//! graph inputs; unchanged acoustic plans can be reused across voice models.
use std::{
    collections::hash_map::DefaultHasher,
    fs,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use vocal_model::ModelPackage;

use crate::{NativeBuildError, NativeBuildOptions, cache_io};

pub(crate) fn root() -> Option<PathBuf> {
    if let Some(root) = std::env::var_os("VOCAL_TRT_PLAN_CACHE_DIR") {
        return (!root.is_empty()).then(|| root.into());
    }
    Some(
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("vocal-fast/cache/tensorrt/plans"),
    )
}

fn effective_options(stage: &str, options: &NativeBuildOptions) -> serde_json::Value {
    let mut value = serde_json::to_value(options).expect("build options are serializable");
    let fields = value.as_object_mut().expect("build options are an object");
    fields.retain(|field, _| {
        if matches!(
            field.as_str(),
            "sdk_root" | "output" | "timing_cache" | "log_build_options"
        ) {
            return false;
        }
        if stage == "frontend" {
            return matches!(
                field.as_str(),
                "workspace_bytes" | "min_frames" | "opt_frames" | "max_frames" | "rvc_version"
            );
        }
        if field == "rvc_version" && stage != "hubert" {
            return false;
        }
        if field.starts_with("synth_") && stage != "synth" {
            return false;
        }
        if field.starts_with("hubert_") && stage != "hubert" {
            return false;
        }
        if field.starts_with("rmvpe_") && stage != "rmvpe" {
            return false;
        }
        if matches!(field.as_str(), "min_frames" | "opt_frames" | "max_frames") && stage != "synth"
        {
            return false;
        }
        if matches!(
            field.as_str(),
            "min_samples" | "opt_samples" | "max_samples"
        ) && stage == "synth"
        {
            return false;
        }
        // The HuBERT and RVC fusion switches affect different graphs.
        if field == "fused_rvc_qkv" && stage != "synth" {
            return false;
        }
        if field == "fused_qkv" && stage != "hubert" {
            return false;
        }
        if field == "debug_outputs" && stage != "synth" {
            return false;
        }
        true
    });
    value
}

pub(crate) fn synth_signature(package: &ModelPackage) -> Result<u64, NativeBuildError> {
    let mut hash = DefaultHasher::new();
    let config = package
        .rvc_synth_config()
        .map_err(|e| NativeBuildError::Api(e.to_string()))?;
    serde_json::to_vec(&config)
        .map_err(|e| NativeBuildError::Api(e.to_string()))?
        .hash(&mut hash);
    for tensor in package
        .tensors
        .iter()
        .filter(|t| !t.name.starts_with("aux."))
    {
        tensor.name.hash(&mut hash);
        (tensor.dtype as u8).hash(&mut hash);
        tensor.shape.hash(&mut hash);
        tensor.data.hash(&mut hash);
    }
    Ok(hash.finish())
}

/// Memoize the content signature by immutable file identity. Complete voice
/// bundles extract identical acoustic files to different directories; their
/// plan keys must depend on bytes, rather than those voice-specific paths.
pub(crate) fn source_signature(root: &Path, source: &Path) -> Result<u64, NativeBuildError> {
    let path = fs::canonicalize(source)?;
    let metadata = fs::metadata(&path)?;
    let mut identity = DefaultHasher::new();
    "vocal-trt-source-signature-v1".hash(&mut identity);
    path.hash(&mut identity);
    metadata.len().hash(&mut identity);
    metadata.modified()?.hash(&mut identity);
    let key = identity.finish();
    let receipt_path = root.join("sources").join(format!("{key:016x}.json"));
    if fs::metadata(&receipt_path).is_ok_and(|m| m.len() <= 4096)
        && let Ok(bytes) = fs::read(&receipt_path)
        && let Ok(receipt) = serde_json::from_slice::<Receipt>(&bytes)
        && receipt.key == key
        && receipt.bytes == metadata.len()
    {
        return Ok(receipt.hash);
    }
    let bytes = cache_io::map(&path)?;
    let mut hash = DefaultHasher::new();
    bytes[..].hash(&mut hash);
    let hash = hash.finish();
    let receipt = Receipt {
        key,
        bytes: bytes.len() as u64,
        hash,
    };
    cache_io::publish(
        &receipt_path,
        &serde_json::to_vec(&receipt).map_err(|e| NativeBuildError::Api(e.to_string()))?,
    )?;
    Ok(hash)
}

pub(crate) fn key(
    stage: &str,
    options: &NativeBuildOptions,
    source: Option<u64>,
    synth_signature: Option<u64>,
) -> Result<u64, NativeBuildError> {
    let mut hash = DefaultHasher::new();
    "vocal-trt-stage-cache-v3".hash(&mut hash);
    crate::BUILDER_FINGERPRINT.hash(&mut hash);
    std::env::consts::OS.hash(&mut hash);
    std::env::consts::ARCH.hash(&mut hash);
    // RTX serializes this process policy into plans. A plan built with TF32
    // allowed can deserialize but cannot create a context under override=0.
    std::env::var_os("NVIDIA_TF32_OVERRIDE").hash(&mut hash);
    stage.hash(&mut hash);
    effective_options(stage, options)
        .to_string()
        .hash(&mut hash);
    source.hash(&mut hash);
    synth_signature.hash(&mut hash);
    Ok(hash.finish())
}

#[derive(Serialize, Deserialize)]
struct Receipt {
    key: u64,
    bytes: u64,
    hash: u64,
}

pub(crate) struct Entry {
    pub path: PathBuf,
    key: u64,
}

impl Entry {
    pub fn new(root: &Path, stage: &str, key: u64) -> Self {
        Self {
            path: root.join(format!("{stage}-{key:016x}.plan")),
            key,
        }
    }

    fn receipt_path(&self) -> PathBuf {
        self.path.with_extension("json")
    }

    pub fn valid(&self) -> bool {
        let check = || -> Option<()> {
            let path = self.receipt_path();
            if fs::metadata(&path).ok()?.len() > 4096 {
                return None;
            }
            let receipt: Receipt = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
            if receipt.key != self.key
                || receipt.bytes == 0
                || fs::metadata(&self.path).ok()?.len() != receipt.bytes
            {
                return None;
            }
            let bytes = cache_io::map(&self.path).ok()?;
            let mut hash = DefaultHasher::new();
            bytes[..].hash(&mut hash);
            (hash.finish() == receipt.hash).then_some(())
        };
        check().is_some()
    }

    pub fn seal(&self) -> Result<(), NativeBuildError> {
        let bytes = cache_io::map(&self.path)?;
        if bytes.is_empty() {
            return Err(NativeBuildError::Argument("empty compiled plan".into()));
        }
        let mut hash = DefaultHasher::new();
        bytes[..].hash(&mut hash);
        let receipt = Receipt {
            key: self.key,
            bytes: bytes.len() as u64,
            hash: hash.finish(),
        };
        cache_io::publish(
            &self.receipt_path(),
            &serde_json::to_vec(&receipt).map_err(|e| NativeBuildError::Api(e.to_string()))?,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_keys_ignore_unrelated_controls_and_track_graph_inputs() {
        let options = NativeBuildOptions::default();
        let mut changed = options.clone();
        changed.output = "another.plan".into();
        changed.log_build_options = true;
        changed.synth_full_fp16 = !changed.synth_full_fp16;
        assert_eq!(
            key("hubert", &options, None, None).unwrap(),
            key("hubert", &changed, None, None).unwrap()
        );
        assert_ne!(
            key("synth", &options, None, Some(1)).unwrap(),
            key("synth", &changed, None, Some(1)).unwrap()
        );
        changed = options.clone();
        changed.rmvpe_fft_cufft = true;
        assert_eq!(
            key("synth", &options, None, Some(1)).unwrap(),
            key("synth", &changed, None, Some(1)).unwrap()
        );
        assert_ne!(
            key("rmvpe", &options, None, None).unwrap(),
            key("rmvpe", &changed, None, None).unwrap()
        );
        assert_ne!(
            key("synth", &options, None, Some(1)).unwrap(),
            key("synth", &options, None, Some(2)).unwrap()
        );
    }

    #[test]
    fn corrupt_or_partial_plan_is_a_miss() {
        let root = std::env::temp_dir().join(format!("vocal-plan-cache-{}", std::process::id()));
        let entry = Entry::new(&root, "frontend", 42);
        assert!(!entry.valid());
        cache_io::publish(&entry.path, b"plan bytes").unwrap();
        assert!(!entry.valid());
        entry.seal().unwrap();
        assert!(entry.valid());
        cache_io::publish(&entry.path, b"bad! bytes").unwrap();
        assert!(!entry.valid());
        cache_io::publish(&entry.path, b"plan bytes").unwrap();
        entry.seal().unwrap();
        assert!(entry.valid());
        fs::remove_file(&entry.path).unwrap();
        fs::remove_file(entry.receipt_path()).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn rvc_versions_separate_features_and_frontend_but_share_pitch_plans() {
        let v2 = NativeBuildOptions::default();
        let mut v1 = v2.clone();
        v1.rvc_version = vocal_model::RvcVersion::V1;
        for stage in ["hubert", "frontend"] {
            assert_ne!(
                key(stage, &v1, Some(7), None).unwrap(),
                key(stage, &v2, Some(7), None).unwrap()
            );
        }
        assert_eq!(
            key("rmvpe", &v1, Some(7), None).unwrap(),
            key("rmvpe", &v2, Some(7), None).unwrap()
        );
    }

    #[test]
    fn identical_acoustic_content_reuses_keys_across_voice_directories() {
        let root =
            std::env::temp_dir().join(format!("vocal-source-signatures-{}", std::process::id()));
        let a = root.join("voice-a/hubert.vfrm");
        let b = root.join("voice-b/hubert.vfrm");
        cache_io::publish(&a, b"same acoustic weights").unwrap();
        cache_io::publish(&b, b"same acoustic weights").unwrap();
        let first = source_signature(&root, &a).unwrap();
        let second = source_signature(&root, &b).unwrap();
        assert_eq!(first, second);
        assert_eq!(source_signature(&root, &a).unwrap(), first);
        cache_io::publish(&b, b"different acoustic weights").unwrap();
        assert_ne!(first, source_signature(&root, &b).unwrap());
        fs::remove_dir_all(root).unwrap();
    }
}
