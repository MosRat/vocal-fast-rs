//! RVC model metadata and the VFRM v1 tensor container.
//!
//! VFRM v1 is close to GGUF's layout: a fixed header, compact JSON
//! configuration, a binary tensor directory, and aligned tensor bytes.

use serde::Serialize;
use serde_json::Value;
use std::{
    fs,
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
    sync::Arc,
};

mod acoustic;
mod bundle;
mod cache;
mod canonical;
mod config;
mod error;
mod import;
mod inspection;
mod logging;
mod metadata;
mod onnx;
mod package;
mod pickle;
mod retrieval;
mod rvc_version;
mod schema;
mod state;

pub use acoustic::{
    ACOUSTIC_TENSOR_SCHEMA, AcousticConfig, AcousticKind, AcousticPort, AcousticTensorMap,
    HubertFeatureMode,
};
pub use bundle::{bundle_rvc_cache, materialize_acoustic_sources};
pub use cache::{ModelPackageCache, shared_model_cache};
pub use config::{
    CACHE_ALIGNMENT, CACHE_HEADER_BYTES, CACHE_MAGIC, DEFAULT_MAX_METADATA_BYTES,
    DEFAULT_MAX_TENSOR_BYTES, DEFAULT_MAX_TENSORS, ModelLoadConfig,
};
pub use error::ModelError;
pub use import::load_rvc_pth_auto;
pub use inspection::{
    HUBERT_BUNDLE_TENSOR, ModelArtifactInspection, ModelComponent, ModelComponents,
    RMVPE_BUNDLE_TENSOR, inspect_model_artifact, inspect_model_artifact_with_config,
};
pub use metadata::{
    ModelMetadataEdit, ModelMetadataView, edit_model_metadata, read_model_metadata,
};
pub use package::{ModelPackage, RvcModelManifest, Tensor, TensorDType};
pub use retrieval::RetrievalIndex;
pub use rvc_version::{RvcFeatureSpec, RvcVersion};
pub use schema::{RVC_TENSOR_SCHEMA, RvcSynthConfig};
pub use state::{ModelLoadState, ModelLoadStatus};

/// The artifact selected by a model path. Directory inputs are resolved to a
/// manifest or a unique VFRM cache before loading, so callers share one path
/// policy instead of implementing subtly different directory heuristics.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelArtifactKind {
    Vfrm,
    Manifest,
}

pub fn model_artifact_kind(path: impl AsRef<Path>) -> Option<ModelArtifactKind> {
    let extension = path
        .as_ref()
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase);
    match extension.as_deref() {
        Some("vfr") | Some("vfrm") => Some(ModelArtifactKind::Vfrm),
        Some("json") => Some(ModelArtifactKind::Manifest),
        _ => None,
    }
}

/// Resolve a user supplied model file or model directory to one canonical
/// artifact. A directory must contain a well-known manifest, or exactly one
/// VFRM cache. This keeps startup restoration and GUI selection consistent.
pub fn resolve_model_path(path: impl AsRef<Path>) -> Result<std::path::PathBuf, ModelError> {
    let input = path.as_ref();
    if input.is_file() {
        return Ok(fs::canonicalize(input)?);
    }
    if !input.is_dir() {
        return Err(ModelError::MissingFile(input.to_path_buf()));
    }
    for name in ["manifest.json", "model.json", "vocal-fast.json"] {
        let candidate = input.join(name);
        if candidate.is_file() {
            return Ok(fs::canonicalize(candidate)?);
        }
    }
    let mut cache = None;
    for entry in fs::read_dir(input)? {
        let candidate = entry?.path();
        if !candidate.is_file()
            || !matches!(
                model_artifact_kind(&candidate),
                Some(ModelArtifactKind::Vfrm)
            )
        {
            continue;
        }
        if cache.replace(candidate).is_some() {
            return Err(ModelError::UnsupportedModel(format!(
                "model directory has more than one VFRM cache: {}",
                input.display()
            )));
        }
    }
    if let Some(path) = cache {
        return Ok(fs::canonicalize(path)?);
    }
    Err(ModelError::UnsupportedModel(format!(
        "model directory has no manifest.json, model.json, or unique VFRM cache: {}",
        input.display()
    )))
}

/// Stateful model access facade. It keeps policy and lifecycle information
/// together while the free functions remain available for simple callers.
pub struct ModelLoader {
    config: ModelLoadConfig,
    status: ModelLoadStatus,
}

impl ModelLoader {
    pub fn new(config: ModelLoadConfig) -> Result<Self, ModelError> {
        config.validate()?;
        Ok(Self {
            config,
            status: ModelLoadStatus::default(),
        })
    }

    pub fn config(&self) -> &ModelLoadConfig {
        &self.config
    }

    pub fn status(&self) -> &ModelLoadStatus {
        &self.status
    }

    pub fn load_manifest(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<RvcModelManifest, ModelError> {
        let config = self.config.clone();
        self.run("manifest", || load_manifest_with_config(path, &config))
    }

    pub fn load_cache(&mut self, path: impl AsRef<Path>) -> Result<ModelPackage, ModelError> {
        let config = self.config.clone();
        self.run("cache", || load_cache_with_config(path, &config))
    }

    /// Load a parsed package through the process-wide weak cache. The returned
    /// `Arc` can be shared by multiple builders without cloning tensor bytes.
    pub fn load_cache_shared(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<Arc<ModelPackage>, ModelError> {
        let config = self.config.clone();
        self.run("cache-shared", || shared_model_cache().load(path, &config))
    }

    /// Load Synth and retrieval tensors without retaining bundled acoustic files.
    pub fn load_voice_cache_shared(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<Arc<ModelPackage>, ModelError> {
        let config = self.config.clone();
        self.run("voice-cache-shared", || {
            shared_model_cache().load_voice(path, &config)
        })
    }

    pub fn load_directory(
        &mut self,
        root: impl AsRef<Path>,
        manifest: RvcModelManifest,
    ) -> Result<ModelPackage, ModelError> {
        let config = self.config.clone();
        self.run("directory", || {
            load_directory_with_config(root, manifest, &config)
        })
    }

    pub fn load_rvc_directory(
        &mut self,
        root: impl AsRef<Path>,
        manifest: RvcModelManifest,
    ) -> Result<ModelPackage, ModelError> {
        let config = self.config.clone();
        self.run("rvc-directory", || {
            load_rvc_directory_with_config(root, manifest, &config)
        })
    }

    pub fn write_cache(
        &mut self,
        path: impl AsRef<Path>,
        package: &ModelPackage,
    ) -> Result<(), ModelError> {
        let config = self.config.clone();
        self.run("cache-write", || {
            write_cache_with_config(path, package, &config)
        })
    }

    fn run<T>(
        &mut self,
        operation: &'static str,
        load: impl FnOnce() -> Result<T, ModelError>,
    ) -> Result<T, ModelError> {
        self.status.begin();
        logging::load_started(operation);
        match load() {
            Ok(value) => {
                self.status.ready();
                logging::load_finished(operation, 0);
                Ok(value)
            }
            Err(error) => {
                self.status.failed();
                logging::load_failed(operation);
                Err(error)
            }
        }
    }
}

pub fn load_manifest(path: impl AsRef<Path>) -> Result<RvcModelManifest, ModelError> {
    load_manifest_with_config(path, &ModelLoadConfig::default())
}

pub fn load_manifest_with_config(
    path: impl AsRef<Path>,
    config: &ModelLoadConfig,
) -> Result<RvcModelManifest, ModelError> {
    config.validate()?;
    let path = path.as_ref();
    let size = fs::metadata(path)?.len();
    if size > config.max_metadata_bytes {
        return Err(ModelError::InvalidCache(format!(
            "manifest is {size} bytes, limit is {}",
            config.max_metadata_bytes
        )));
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

/// Read only the manifest embedded in a VFRM cache.
///
/// Directory views use this instead of `load_cache_with_config`, so scanning a
/// model library never allocates the model tensors just to render metadata.
pub fn inspect_cache_manifest_with_config(
    path: impl AsRef<Path>,
    load_config: &ModelLoadConfig,
) -> Result<RvcModelManifest, ModelError> {
    load_config.validate()?;
    let path = path.as_ref();
    let file_len = fs::metadata(path)?.len();
    let mut input = io::BufReader::new(fs::File::open(path)?);
    let mut magic = [0_u8; 8];
    input.read_exact(&mut magic)?;
    if &magic != CACHE_MAGIC {
        return Err(ModelError::InvalidCache("bad magic/version".into()));
    }
    let _flags = read_u32(&mut input)?;
    let header_size = read_u32(&mut input)? as u64;
    let metadata_len = read_u64(&mut input)?;
    let tensor_count = read_u64(&mut input)?;
    let directory_len = read_u64(&mut input)?;
    let data_offset = read_u64(&mut input)?;
    let _reserved0 = read_u64(&mut input)?;
    let _reserved1 = read_u64(&mut input)?;
    let metadata_end = CACHE_HEADER_BYTES
        .checked_add(metadata_len)
        .ok_or_else(|| ModelError::InvalidCache("metadata range overflows".into()))?;
    let directory_end = metadata_end
        .checked_add(directory_len)
        .ok_or_else(|| ModelError::InvalidCache("directory range overflows".into()))?;
    if header_size != CACHE_HEADER_BYTES
        || metadata_len > load_config.max_metadata_bytes
        || tensor_count > load_config.max_tensors
        || directory_len > load_config.max_tensor_bytes
        || data_offset < directory_end
        || data_offset > file_len
    {
        return Err(ModelError::InvalidCache("invalid header bounds".into()));
    }
    let mut config_bytes = vec![0; metadata_len as usize];
    input.read_exact(&mut config_bytes)?;
    let config: Value = serde_json::from_slice(&config_bytes)?;
    serde_json::from_value(
        config
            .get("manifest")
            .cloned()
            .ok_or_else(|| ModelError::InvalidCache("config has no manifest".into()))?,
    )
    .map_err(ModelError::from)
}

/// Inspect a supported model artifact without loading its tensor payload.
pub fn inspect_model_manifest_with_config(
    path: impl AsRef<Path>,
    load_config: &ModelLoadConfig,
) -> Result<RvcModelManifest, ModelError> {
    let path = path.as_ref();
    match path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("vfr") | Some("vfrm") => inspect_cache_manifest_with_config(path, load_config),
        _ => load_manifest_with_config(path, load_config),
    }
}

pub fn load_directory(
    root: impl AsRef<Path>,
    manifest: RvcModelManifest,
) -> Result<ModelPackage, ModelError> {
    load_directory_with_config(root, manifest, &ModelLoadConfig::default())
}

pub fn load_directory_with_config(
    root: impl AsRef<Path>,
    manifest: RvcModelManifest,
    load_config: &ModelLoadConfig,
) -> Result<ModelPackage, ModelError> {
    load_config.validate()?;
    let root = root.as_ref();
    let model_path = manifest_path(root, &manifest.model_file)?;
    let mut tensors = vec![Tensor::byte_tensor(
        "rvc.pth",
        read_bounded_file(&model_path, load_config.max_tensor_bytes)?,
    )];
    if let Some(file) = &manifest.index_file {
        let path = manifest_path(root, file)?;
        tensors.push(Tensor::byte_tensor(
            "aux.retrieval.faiss",
            read_bounded_file(&path, load_config.max_tensor_bytes)?,
        ));
    }
    Ok(ModelPackage {
        manifest,
        metadata: Value::Object(Default::default()),
        tensors,
    })
}

/// Parse an RVC PyTorch checkpoint without executing pickle reducers. Only the
/// CPU/standard RVC state_dict subset is accepted; unknown pickle globals,
/// tensor views and storage types fail closed.
pub fn load_rvc_pth(
    path: impl AsRef<Path>,
    manifest: RvcModelManifest,
) -> Result<ModelPackage, ModelError> {
    load_rvc_pth_with_config(path, manifest, &ModelLoadConfig::default())
}

pub fn load_rvc_pth_with_config(
    path: impl AsRef<Path>,
    manifest: RvcModelManifest,
    load_config: &ModelLoadConfig,
) -> Result<ModelPackage, ModelError> {
    load_config.validate()?;
    ensure_file_size(path.as_ref(), load_config.max_tensor_bytes)?;
    let (metadata, tensors) = pickle::load_rvc_pth(path.as_ref())?;
    validate_rvc_checkpoint_metadata(&manifest, &metadata)?;
    let (metadata, tensors) = canonical::normalize_pth(&manifest, metadata, tensors)?;
    validate_rvc_tensors(&manifest, &tensors)?;
    Ok(ModelPackage {
        manifest,
        metadata,
        tensors,
    })
}

/// Parse a model directory when the checkpoint itself is a supported RVC
/// `.pth` or Synthesizer `.onnx`, then attach its optional FAISS index.
pub fn load_rvc_directory(
    root: impl AsRef<Path>,
    manifest: RvcModelManifest,
) -> Result<ModelPackage, ModelError> {
    load_rvc_directory_with_config(root, manifest, &ModelLoadConfig::default())
}

pub fn load_rvc_directory_with_config(
    root: impl AsRef<Path>,
    manifest: RvcModelManifest,
    load_config: &ModelLoadConfig,
) -> Result<ModelPackage, ModelError> {
    load_config.validate()?;
    let root = root.as_ref();
    let model_path = manifest_path(root, &manifest.model_file)?;
    let mut package = match model_path
        .extension()
        .and_then(|x| x.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("pth") => load_rvc_pth_with_config(&model_path, manifest.clone(), load_config)?,
        Some("onnx") => {
            ensure_file_size(&model_path, load_config.max_tensor_bytes)?;
            load_rvc_onnx(&model_path, manifest.clone())?
        }
        _ => {
            return Err(ModelError::UnsupportedModel(
                "RVC model must end in .pth or .onnx".into(),
            ));
        }
    };
    if let Some(file) = &manifest.index_file {
        let path = manifest_path(root, file)?;
        package.tensors.push(Tensor::byte_tensor(
            "aux.retrieval.faiss",
            read_bounded_file(&path, load_config.max_tensor_bytes)?,
        ));
    }
    Ok(package)
}

fn manifest_path(root: &Path, relative: &str) -> Result<std::path::PathBuf, ModelError> {
    let root = fs::canonicalize(root)?;
    let candidate = root.join(relative);
    let canonical = match fs::canonicalize(&candidate) {
        Ok(path) => path,
        Err(_) => return Err(ModelError::MissingFile(candidate)),
    };
    if !canonical.starts_with(&root) {
        return Err(ModelError::UnsupportedModel(format!(
            "manifest file escapes model directory: {relative}"
        )));
    }
    Ok(canonical)
}

/// Parse a known RVC Synthesizer ONNX export. The graph is represented by a
/// fixed `graph_schema` in metadata; initializers become typed VFRM tensors.
pub fn load_rvc_onnx(
    path: impl AsRef<Path>,
    manifest: RvcModelManifest,
) -> Result<ModelPackage, ModelError> {
    let (metadata, tensors) = onnx::load_rvc_onnx(path.as_ref(), &manifest)?;
    validate_rvc_tensors(&manifest, &tensors)?;
    Ok(ModelPackage {
        manifest,
        metadata,
        tensors,
    })
}

fn validate_rvc_tensors(manifest: &RvcModelManifest, tensors: &[Tensor]) -> Result<(), ModelError> {
    let phone = tensors.iter().find(|t| t.name == "enc_p.emb_phone.weight");
    if let Some(phone) = phone {
        let last = *phone.shape.last().ok_or_else(|| {
            ModelError::UnsupportedModel("RVC phone projection has no shape".into())
        })?;
        if phone.shape != vec![manifest.feature_dims() as u64, 192] {
            return Err(ModelError::UnsupportedModel(format!(
                "feature width {} does not match manifest {}",
                last,
                manifest.feature_dims()
            )));
        }
    } else if !tensors.iter().any(|t| t.name.contains("emb_phone")) {
        // ONNX exporters may fold the phone projection into a graph constant
        // such as `/enc_p/emb_phone/Transpose_output_0`; the ONNX signature
        // validator has already checked its feature width.
        if !tensors.iter().any(|t| t.name.contains("/enc_p/emb_phone/")) {
            return Err(ModelError::UnsupportedModel(
                "RVC encoder phone projection is missing".into(),
            ));
        }
    }
    if !tensors.iter().any(|t| t.name == "emb_g.weight") {
        return Err(ModelError::UnsupportedModel(
            "RVC speaker embedding is missing".into(),
        ));
    }
    Ok(())
}

fn validate_rvc_checkpoint_metadata(
    manifest: &RvcModelManifest,
    metadata: &Value,
) -> Result<(), ModelError> {
    if let Some(version) = metadata.get("version").and_then(Value::as_str) {
        let checkpoint_version = match version.to_ascii_lowercase().as_str() {
            "v1" | "1" => RvcVersion::V1,
            "v2" | "2" => RvcVersion::V2,
            _ => {
                return Err(ModelError::UnsupportedModel(format!(
                    "unsupported checkpoint RVC version {version}"
                )));
            }
        };
        if checkpoint_version != manifest.rvc_version()? {
            return Err(ModelError::UnsupportedModel(format!(
                "checkpoint version {version} does not match manifest {}",
                manifest.version
            )));
        }
    }
    if let Some(f0) = metadata.get("f0").and_then(|v| v.as_i64())
        && (f0 != 0) != manifest.uses_f0()
    {
        return Err(ModelError::UnsupportedModel(
            "checkpoint F0 flag does not match manifest".into(),
        ));
    }
    if let Some(sr) = metadata.get("sr") {
        let rate = sr.as_u64().or_else(|| {
            sr.as_str().and_then(|s| {
                s.trim_end_matches('k')
                    .parse::<u64>()
                    .ok()
                    .map(|x| x * 1000)
            })
        });
        if let Some(rate) = rate
            && rate != manifest.sample_rate as u64
        {
            return Err(ModelError::UnsupportedModel(format!(
                "checkpoint sample rate {rate} does not match manifest {}",
                manifest.sample_rate
            )));
        }
    }
    Ok(())
}

/// VFRM v1: fixed header, config JSON, binary tensor directory, aligned data.
pub fn write_cache(path: impl AsRef<Path>, package: &ModelPackage) -> Result<(), ModelError> {
    write_cache_with_config(path, package, &ModelLoadConfig::default())
}

pub fn write_cache_with_config(
    path: impl AsRef<Path>,
    package: &ModelPackage,
    load_config: &ModelLoadConfig,
) -> Result<(), ModelError> {
    load_config.validate()?;
    if load_config.validate_canonical {
        validate_canonical_package(package)?;
    }
    let config = serde_json::to_vec(
        &serde_json::json!({"manifest": package.manifest, "metadata": package.metadata}),
    )?;
    if config.len() as u64 > load_config.max_metadata_bytes
        || package.tensors.len() as u64 > load_config.max_tensors
    {
        return Err(ModelError::InvalidCache(
            "cache metadata or tensor count too large".into(),
        ));
    }
    for tensor in &package.tensors {
        tensor.validate()?;
    }
    let directory_len = package
        .tensors
        .iter()
        .map(|t| 8 + t.name.len() as u64 + 8 * t.shape.len() as u64 + 16)
        .sum::<u64>();
    let directory_offset = CACHE_HEADER_BYTES + config.len() as u64;
    let data_offset = align_up(directory_offset + directory_len, CACHE_ALIGNMENT);
    let mut offsets = Vec::with_capacity(package.tensors.len());
    let mut cursor = data_offset;
    for tensor in &package.tensors {
        cursor = align_up(cursor, CACHE_ALIGNMENT);
        offsets.push(cursor);
        cursor = cursor
            .checked_add(tensor.data.len() as u64)
            .ok_or_else(|| ModelError::InvalidCache("cache size overflows".into()))?;
    }
    let mut out = io::BufWriter::new(fs::File::create(path)?);
    out.write_all(CACHE_MAGIC)?;
    write_u32(&mut out, 0)?;
    write_u32(&mut out, CACHE_HEADER_BYTES as u32)?;
    write_u64(&mut out, config.len() as u64)?;
    write_u64(&mut out, package.tensors.len() as u64)?;
    write_u64(&mut out, directory_len)?;
    write_u64(&mut out, data_offset)?;
    write_u64(&mut out, 0)?;
    write_u64(&mut out, 0)?;
    out.write_all(&config)?;
    for (tensor, &offset) in package.tensors.iter().zip(&offsets) {
        write_u32(&mut out, tensor.name.len() as u32)?;
        out.write_all(&[tensor.dtype as u8, tensor.shape.len() as u8])?;
        write_u16(&mut out, 0)?;
        out.write_all(tensor.name.as_bytes())?;
        for &dim in &tensor.shape {
            write_u64(&mut out, dim)?;
        }
        write_u64(&mut out, offset)?;
        write_u64(&mut out, tensor.data.len() as u64)?;
    }
    pad_to(&mut out, data_offset - directory_offset - directory_len)?;
    for (tensor, &offset) in package.tensors.iter().zip(&offsets) {
        let current = out.stream_position()?;
        pad_to(&mut out, offset - current)?;
        out.write_all(&tensor.data)?;
    }
    out.flush()?;
    Ok(())
}

pub fn load_cache(path: impl AsRef<Path>) -> Result<ModelPackage, ModelError> {
    load_cache_with_config(path, &ModelLoadConfig::default())
}

pub fn load_cache_with_config(
    path: impl AsRef<Path>,
    load_config: &ModelLoadConfig,
) -> Result<ModelPackage, ModelError> {
    load_cache_selected(path.as_ref(), load_config, false)
}

pub(crate) fn load_cache_selected(
    path: &Path,
    load_config: &ModelLoadConfig,
    voice_only: bool,
) -> Result<ModelPackage, ModelError> {
    load_config.validate()?;
    let mut input = io::BufReader::new(fs::File::open(path)?);
    let directory =
        inspection::read_directory(&mut input, 0, fs::metadata(path)?.len(), load_config)?;
    if voice_only
        || load_config.validate_canonical
            && directory
                .tensors
                .iter()
                .any(|t| matches!(t.name.as_str(), HUBERT_BUNDLE_TENSOR | RMVPE_BUNDLE_TENSOR))
    {
        inspection::inspect_directory(
            &mut input,
            &directory,
            &path.display().to_string(),
            load_config,
        )?;
    }
    let mut tensors = Vec::with_capacity(directory.tensors.len());
    for entry in directory.tensors {
        if voice_only
            && matches!(
                entry.name.as_str(),
                HUBERT_BUNDLE_TENSOR | RMVPE_BUNDLE_TENSOR
            )
        {
            continue;
        }
        input.seek(SeekFrom::Start(entry.offset))?;
        let mut data = vec![0; entry.len as usize];
        input.read_exact(&mut data)?;
        let tensor = Tensor {
            name: entry.name,
            dtype: entry.dtype,
            shape: entry.shape,
            data,
        };
        tensor.validate()?;
        tensors.push(tensor);
    }
    let package = ModelPackage {
        manifest: directory.manifest,
        metadata: directory.metadata,
        tensors,
    };
    if load_config.validate_canonical {
        validate_canonical_package(&package)?;
    }
    Ok(package)
}

pub fn load_voice_cache_shared(path: impl AsRef<Path>) -> Result<Arc<ModelPackage>, ModelError> {
    shared_model_cache().load_voice(path, &ModelLoadConfig::default())
}

/// Load a VFRM package through the process-wide weak cache.
pub fn load_cache_shared(path: impl AsRef<Path>) -> Result<Arc<ModelPackage>, ModelError> {
    load_cache_shared_with_config(path, &ModelLoadConfig::default())
}

pub fn load_cache_shared_with_config(
    path: impl AsRef<Path>,
    load_config: &ModelLoadConfig,
) -> Result<Arc<ModelPackage>, ModelError> {
    shared_model_cache().load(path, load_config)
}

/// Drop weak cache entries. Packages still held by callers remain valid and
/// are released normally when their final `Arc` is dropped.
pub fn clear_shared_model_cache() {
    shared_model_cache().clear();
}

fn validate_canonical_package(package: &ModelPackage) -> Result<(), ModelError> {
    if package
        .metadata
        .get("canonical_schema")
        .and_then(Value::as_str)
        == Some(RVC_TENSOR_SCHEMA)
    {
        let manifest_version = package.manifest.rvc_version()?;
        if let Some(graph_schema) = package.metadata.get("graph_schema").and_then(Value::as_str) {
            let graph_version = if graph_schema.starts_with("rvc-v1-") {
                Some(RvcVersion::V1)
            } else if graph_schema.starts_with("rvc-v2-") {
                Some(RvcVersion::V2)
            } else {
                None
            };
            if graph_version.is_some_and(|version| version != manifest_version) {
                return Err(ModelError::InvalidCache(
                    "canonical graph_schema disagrees with RVC manifest version".into(),
                ));
            }
        }
        let config: RvcSynthConfig =
            serde_json::from_value(package.metadata.get("rvc_config").cloned().ok_or_else(
                || ModelError::InvalidCache("canonical cache has no rvc_config".into()),
            )?)?;
        schema::validate_tensors(&config, &package.manifest, &package.tensors)?;
        inspection::validate_embedded_acoustics(package)?;
    }
    if package
        .metadata
        .get("canonical_schema")
        .and_then(Value::as_str)
        == Some(ACOUSTIC_TENSOR_SCHEMA)
    {
        package.validate_acoustic(None)?;
    }
    Ok(())
}

fn ensure_file_size(path: &Path, limit: u64) -> Result<(), ModelError> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > limit {
        return Err(ModelError::InvalidCache(format!(
            "file {} is {} bytes, limit is {limit}",
            path.display(),
            metadata.len()
        )));
    }
    Ok(())
}

fn read_bounded_file(path: &Path, limit: u64) -> Result<Vec<u8>, ModelError> {
    ensure_file_size(path, limit)?;
    Ok(fs::read(path)?)
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}
fn pad_to(out: &mut impl Write, count: u64) -> io::Result<()> {
    const ZEROES: [u8; 64] = [0; 64];
    let mut remaining = count;
    while remaining > 0 {
        let n = remaining.min(64) as usize;
        out.write_all(&ZEROES[..n])?;
        remaining -= n as u64;
    }
    Ok(())
}
fn write_u16(out: &mut impl Write, value: u16) -> io::Result<()> {
    out.write_all(&value.to_le_bytes())
}
fn write_u32(out: &mut impl Write, value: u32) -> io::Result<()> {
    out.write_all(&value.to_le_bytes())
}
fn write_u64(out: &mut impl Write, value: u64) -> io::Result<()> {
    out.write_all(&value.to_le_bytes())
}
fn read_u32(input: &mut impl Read) -> io::Result<u32> {
    let mut b = [0; 4];
    input.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(input: &mut impl Read) -> io::Result<u64> {
    let mut b = [0; 8];
    input.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> ModelPackage {
        ModelPackage {
            manifest: RvcModelManifest {
                id: "fixture".into(),
                display_name: "Fixture".into(),
                architecture: "rvc-v2".into(),
                version: "1".into(),
                sample_rate: 32_000,
                model_file: "model.pth".into(),
                index_file: Some("model.index".into()),
                source: "test".into(),
                notes: None,
            },
            metadata: serde_json::json!({"config": [1, 2, 3], "f0": 1}),
            tensors: vec![
                Tensor {
                    name: "enc.weight".into(),
                    dtype: TensorDType::F32,
                    shape: vec![2],
                    data: vec![0, 0, 128, 63, 0, 0, 0, 64],
                },
                Tensor::byte_tensor("retrieval.faiss", vec![4, 5]),
            ],
        }
    }
    #[test]
    fn cache_round_trip_preserves_tensor_directory() {
        let package = fixture();
        let path = std::env::temp_dir().join(format!("vocal-fast-{}.vfr", std::process::id()));
        write_cache(&path, &package).unwrap();
        let restored = load_cache(&path).unwrap();
        assert_eq!(restored.manifest.id, package.manifest.id);
        assert_eq!(restored.metadata, package.metadata);
        assert_eq!(restored.tensors.len(), 2);
        assert_eq!(restored.tensors[0].name, "enc.weight");
        assert_eq!(restored.tensors[0].shape, vec![2]);
        assert_eq!(restored.tensors[1].data, vec![4, 5]);
        let _ = fs::remove_file(path);
    }
    #[test]
    fn shared_cache_reuses_live_package_without_cloning_tensors() {
        let package = fixture();
        let path =
            std::env::temp_dir().join(format!("vocal-fast-shared-{}.vfr", std::process::id()));
        write_cache(&path, &package).unwrap();
        clear_shared_model_cache();
        let first = load_cache_shared_with_config(&path, &ModelLoadConfig::default()).unwrap();
        let second = load_cache_shared_with_config(&path, &ModelLoadConfig::default()).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(shared_model_cache().len(), 1);
        clear_shared_model_cache();
        assert_eq!(shared_model_cache().len(), 0);
        drop(first);
        drop(second);
        assert_eq!(shared_model_cache().len(), 0);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn inspect_cache_manifest_does_not_load_tensor_payload() {
        let package = fixture();
        let path = std::env::temp_dir().join(format!(
            "vocal-fast-inspect-{}-{}.vfrm",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_cache(&path, &package).unwrap();
        let manifest = inspect_cache_manifest_with_config(&path, &ModelLoadConfig::default())
            .expect("inspect manifest");
        assert_eq!(manifest.id, package.manifest.id);
        assert_eq!(manifest.model_file, package.manifest.model_file);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn resolve_model_path_accepts_directory_manifest_and_unique_cache() {
        let root = std::env::temp_dir().join(format!(
            "vocal-fast-resolve-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let manifest_path = root.join("manifest.json");
        fs::write(&manifest_path, b"{}").unwrap();
        assert_eq!(
            resolve_model_path(&root).unwrap(),
            fs::canonicalize(&manifest_path).unwrap()
        );
        fs::remove_file(&manifest_path).unwrap();
        let cache_path = root.join("voice.vfrm");
        write_cache(&cache_path, &fixture()).unwrap();
        assert_eq!(
            resolve_model_path(&root).unwrap(),
            fs::canonicalize(&cache_path).unwrap()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_shape_byte_mismatch() {
        let tensor = Tensor {
            name: "bad".into(),
            dtype: TensorDType::F32,
            shape: vec![2],
            data: vec![0; 4],
        };
        assert!(tensor.validate().is_err());
    }
}
