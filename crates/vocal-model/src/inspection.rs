//! Bounded directory inspection. Tensor payloads are never allocated here.
use crate::{
    ACOUSTIC_TENSOR_SCHEMA, AcousticConfig, AcousticKind, CACHE_HEADER_BYTES, CACHE_MAGIC,
    ModelArtifactKind, ModelError, ModelLoadConfig, ModelPackage, RVC_TENSOR_SCHEMA,
    RvcModelManifest, RvcSynthConfig, TensorDType, load_manifest_with_config, manifest_path,
    model_artifact_kind, read_u32, read_u64, resolve_model_path,
};
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    io::{self, Read, Seek, SeekFrom},
    path::Path,
};

pub const HUBERT_BUNDLE_TENSOR: &str = "aux.hubert.vfrm";
pub const RMVPE_BUNDLE_TENSOR: &str = "aux.rmvpe.vfrm";
const MAX_DIRECTORY_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelComponent {
    pub present: bool,
    pub embedded: bool,
    pub tensor_count: usize,
    pub size_bytes: u64,
    pub source: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelComponents {
    pub synth: ModelComponent,
    /// Number of rows in the canonical Synth speaker embedding, when known.
    pub speaker_count: Option<u64>,
    pub hubert: ModelComponent,
    pub rmvpe: ModelComponent,
    pub retrieval: ModelComponent,
    pub automatic_tensor_rt: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelArtifactInspection {
    pub path: String,
    pub manifest: RvcModelManifest,
    pub components: ModelComponents,
}

pub(crate) struct TensorEntry {
    pub name: String,
    pub dtype: TensorDType,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub len: u64,
}

pub(crate) struct CacheDirectory {
    pub manifest: RvcModelManifest,
    pub metadata: Value,
    pub tensors: Vec<TensorEntry>,
}

fn bad(message: &str) -> ModelError {
    ModelError::InvalidCache(message.into())
}

/// `base` and `length` bound a standalone file or one embedded acoustic file.
pub(crate) fn read_directory(
    input: &mut (impl Read + Seek),
    base: u64,
    length: u64,
    config: &ModelLoadConfig,
) -> Result<CacheDirectory, ModelError> {
    config.validate()?;
    if length < CACHE_HEADER_BYTES || base.checked_add(length).is_none() {
        return Err(bad("invalid cache range"));
    }
    input.seek(SeekFrom::Start(base))?;
    let mut header = [0; CACHE_HEADER_BYTES as usize];
    input.read_exact(&mut header)?;
    let mut header = io::Cursor::new(header);
    let mut magic = [0; 8];
    header.read_exact(&mut magic)?;
    if &magic != CACHE_MAGIC {
        return Err(bad("bad magic/version"));
    }
    let _flags = read_u32(&mut header)?;
    let header_size = read_u32(&mut header)? as u64;
    let metadata_len = read_u64(&mut header)?;
    let count = read_u64(&mut header)?;
    let directory_len = read_u64(&mut header)?;
    let data_offset = read_u64(&mut header)?;
    let directory_offset = CACHE_HEADER_BYTES
        .checked_add(metadata_len)
        .ok_or_else(|| bad("metadata range overflows"))?;
    let directory_end = directory_offset
        .checked_add(directory_len)
        .ok_or_else(|| bad("directory range overflows"))?;
    if header_size != CACHE_HEADER_BYTES
        || metadata_len > config.max_metadata_bytes
        || count > config.max_tensors
        || directory_len > MAX_DIRECTORY_BYTES
        || data_offset < directory_end
        || data_offset > length
    {
        return Err(bad("invalid header bounds"));
    }
    let mut bytes = vec![0; metadata_len as usize];
    input.read_exact(&mut bytes)?;
    let json: Value = serde_json::from_slice(&bytes)?;
    let manifest = serde_json::from_value(
        json.get("manifest")
            .cloned()
            .ok_or_else(|| bad("config has no manifest"))?,
    )?;
    let metadata = json.get("metadata").cloned().unwrap_or(Value::Null);
    let mut bytes = vec![0; directory_len as usize];
    input.read_exact(&mut bytes)?;
    let mut directory = io::Cursor::new(bytes);
    let mut tensors = Vec::new();
    let mut names = HashSet::new();
    for _ in 0..count {
        let name_len = read_u32(&mut directory)? as usize;
        let mut attrs = [0; 4];
        directory.read_exact(&mut attrs)?;
        let dtype = TensorDType::from_byte(attrs[0])?;
        let rank = attrs[1] as usize;
        if name_len == 0 || name_len > u16::MAX as usize || rank > 64 {
            return Err(bad("invalid tensor directory entry"));
        }
        let mut name = vec![0; name_len];
        directory.read_exact(&mut name)?;
        let name = String::from_utf8(name).map_err(|_| bad("tensor name is not UTF-8"))?;
        if !names.insert(name.clone()) {
            return Err(bad("duplicate tensor name"));
        }
        let shape = (0..rank)
            .map(|_| read_u64(&mut directory))
            .collect::<Result<Vec<_>, _>>()?;
        let offset = read_u64(&mut directory)?;
        let len = read_u64(&mut directory)?;
        let expected = shape
            .iter()
            .try_fold(dtype.element_size() as u64, |a, &b| a.checked_mul(b));
        if expected != Some(len)
            || len > config.max_tensor_bytes
            || offset < data_offset
            || offset.checked_add(len).is_none_or(|end| end > length)
        {
            return Err(bad("invalid tensor shape or data range"));
        }
        tensors.push(TensorEntry {
            name,
            dtype,
            shape,
            offset: base + offset,
            len,
        });
    }
    if directory.position() != directory_len {
        return Err(bad("tensor directory length mismatch"));
    }
    let mut ranges: Vec<_> = tensors
        .iter()
        .map(|t| (t.offset, t.offset + t.len))
        .collect();
    ranges.sort_unstable();
    if ranges.windows(2).any(|r| r[0].1 > r[1].0) {
        return Err(bad("overlapping tensor data"));
    }
    Ok(CacheDirectory {
        manifest,
        metadata,
        tensors,
    })
}

fn component(entries: &[&TensorEntry], source: String) -> ModelComponent {
    ModelComponent {
        present: !entries.is_empty(),
        embedded: true,
        tensor_count: entries.len(),
        size_bytes: entries.iter().map(|t| t.len).sum(),
        source: Some(source),
    }
}

pub(crate) fn validate_acoustic_directory(
    directory: &CacheDirectory,
    kind: Option<AcousticKind>,
) -> Result<AcousticConfig, ModelError> {
    if directory
        .metadata
        .get("canonical_schema")
        .and_then(Value::as_str)
        != Some(ACOUSTIC_TENSOR_SCHEMA)
    {
        return Err(bad("embedded source must be acoustic-v1"));
    }
    let config: AcousticConfig = serde_json::from_value(directory.metadata.clone())?;
    config.validate_feature_modes()?;
    for mode in &config.rvc_feature_modes {
        crate::rvc_version::validate_hubert_feature_tensors(
            mode.version,
            directory
                .tensors
                .iter()
                .map(|t| (t.name.as_str(), t.shape.as_slice(), t.dtype)),
        )?;
    }
    if kind.is_some_and(|kind| kind != config.kind)
        || config.node_count == 0
        || config.node_count > 10_000
        || config.inputs.is_empty()
        || config.outputs.is_empty()
        || directory.tensors.is_empty()
    {
        return Err(bad("invalid acoustic component"));
    }
    let prefix = format!("{}.", config.kind.as_str());
    let names: HashSet<_> = directory.tensors.iter().map(|t| t.name.as_str()).collect();
    if directory
        .tensors
        .iter()
        .any(|t| !t.name.starts_with(&prefix))
        || config
            .tensor_map
            .iter()
            .any(|t| !names.contains(t.canonical.as_str()))
    {
        return Err(bad("invalid acoustic tensor mapping"));
    }
    Ok(config)
}

pub(crate) fn inspect_directory(
    input: &mut (impl Read + Seek),
    directory: &CacheDirectory,
    path: &str,
    config: &ModelLoadConfig,
) -> Result<ModelComponents, ModelError> {
    let mut result = ModelComponents::default();
    let schema = directory
        .metadata
        .get("canonical_schema")
        .and_then(Value::as_str);
    if schema == Some(RVC_TENSOR_SCHEMA) {
        let synth: RvcSynthConfig = serde_json::from_value(
            directory
                .metadata
                .get("rvc_config")
                .cloned()
                .ok_or_else(|| bad("missing rvc_config"))?,
        )?;
        let expected = synth.tensor_shapes()?;
        crate::rvc_version::validate_feature_metadata(
            &directory.metadata,
            directory.manifest.rvc_version()?,
        )?;
        if synth.feature_width != directory.manifest.rvc_version()?.feature_width() as u64
            || synth.uses_f0 != directory.manifest.uses_f0()
            || synth.sample_rate != directory.manifest.sample_rate
        {
            return Err(bad("canonical RVC config disagrees with manifest"));
        }
        let parameters: BTreeMap<_, _> = directory
            .tensors
            .iter()
            .filter(|t| !t.name.starts_with("aux."))
            .map(|t| (&t.name, t))
            .collect();
        if expected.len() != parameters.len()
            || expected.iter().any(|(name, shape)| {
                parameters.get(name).is_none_or(|t| {
                    &t.shape != shape
                        || !matches!(
                            t.dtype,
                            TensorDType::F16 | TensorDType::F32 | TensorDType::BF16
                        )
                })
            })
        {
            return Err(bad("incomplete canonical Synth weights"));
        }
        result.synth = component(
            &parameters.values().copied().collect::<Vec<_>>(),
            path.into(),
        );
        result.speaker_count = Some(synth.speakers);
        for entry in directory
            .tensors
            .iter()
            .filter(|t| t.name.starts_with("aux."))
        {
            if !matches!(
                entry.name.as_str(),
                "aux.retrieval.faiss" | HUBERT_BUNDLE_TENSOR | RMVPE_BUNDLE_TENSOR
            ) || entry.dtype != TensorDType::U8
                || entry.shape.len() != 1
            {
                return Err(bad("unsupported RVC auxiliary tensor"));
            }
            if entry.name == "aux.retrieval.faiss" {
                result.retrieval = component(&[entry], format!("{path}#retrieval"));
            } else {
                let kind = if entry.name == HUBERT_BUNDLE_TENSOR {
                    AcousticKind::Hubert
                } else {
                    AcousticKind::Rmvpe
                };
                let nested = read_directory(input, entry.offset, entry.len, config)?;
                let acoustic = validate_acoustic_directory(&nested, Some(kind))?;
                if kind == AcousticKind::Hubert {
                    acoustic.validate_rvc_version(directory.manifest.rvc_version()?)?;
                    crate::rvc_version::validate_hubert_feature_tensors(
                        directory.manifest.rvc_version()?,
                        nested
                            .tensors
                            .iter()
                            .map(|t| (t.name.as_str(), t.shape.as_slice(), t.dtype)),
                    )?;
                }
                let value = component(
                    &nested.tensors.iter().collect::<Vec<_>>(),
                    format!("{path}#{}", kind.as_str()),
                );
                match kind {
                    AcousticKind::Hubert => result.hubert = value,
                    AcousticKind::Rmvpe => result.rmvpe = value,
                }
            }
        }
    } else if schema == Some(ACOUSTIC_TENSOR_SCHEMA) {
        let acoustic = validate_acoustic_directory(directory, None)?;
        let value = component(&directory.tensors.iter().collect::<Vec<_>>(), path.into());
        match acoustic.kind {
            AcousticKind::Hubert => result.hubert = value,
            AcousticKind::Rmvpe => result.rmvpe = value,
        }
    } else {
        if let Some(entry) = directory.tensors.iter().find(|t| t.name == "rvc.pth") {
            result.synth = component(&[entry], path.into());
        }
        if let Some(entry) = directory
            .tensors
            .iter()
            .find(|t| t.name == "aux.retrieval.faiss")
        {
            result.retrieval = component(&[entry], format!("{path}#retrieval"));
        }
    }
    result.automatic_tensor_rt = schema == Some(RVC_TENSOR_SCHEMA)
        && result.synth.present
        && result.hubert.present
        && result.rmvpe.present;
    Ok(result)
}

pub fn inspect_model_artifact(
    path: impl AsRef<Path>,
) -> Result<ModelArtifactInspection, ModelError> {
    inspect_model_artifact_with_config(path, &ModelLoadConfig::default())
}

pub fn inspect_model_artifact_with_config(
    path: impl AsRef<Path>,
    config: &ModelLoadConfig,
) -> Result<ModelArtifactInspection, ModelError> {
    let path = resolve_model_path(path)?;
    let source = path.display().to_string();
    let (manifest, components) = if model_artifact_kind(&path) == Some(ModelArtifactKind::Vfrm) {
        let mut input = io::BufReader::new(fs::File::open(&path)?);
        let directory = read_directory(&mut input, 0, fs::metadata(&path)?.len(), config)?;
        let components = inspect_directory(&mut input, &directory, &source, config)?;
        (directory.manifest, components)
    } else {
        let manifest = load_manifest_with_config(&path, config)?;
        let root = path.parent().unwrap_or(Path::new("."));
        let external = |relative: &str| -> ModelComponent {
            let candidate = root.join(relative);
            let metadata = manifest_path(root, relative)
                .ok()
                .and_then(|p| fs::metadata(p).ok())
                .filter(|m| m.is_file());
            ModelComponent {
                present: metadata.is_some(),
                size_bytes: metadata.map_or(0, |m| m.len()),
                source: Some(candidate.display().to_string()),
                ..Default::default()
            }
        };
        let components = ModelComponents {
            synth: external(&manifest.model_file),
            retrieval: manifest
                .index_file
                .as_deref()
                .map(external)
                .unwrap_or_default(),
            ..Default::default()
        };
        (manifest, components)
    };
    Ok(ModelArtifactInspection {
        path: source,
        manifest,
        components,
    })
}

pub(crate) fn validate_embedded_acoustics(package: &ModelPackage) -> Result<(), ModelError> {
    for (name, kind) in [
        (HUBERT_BUNDLE_TENSOR, AcousticKind::Hubert),
        (RMVPE_BUNDLE_TENSOR, AcousticKind::Rmvpe),
    ] {
        if let Some(tensor) = package.tensor(name) {
            let mut input = io::Cursor::new(&tensor.data);
            let directory = read_directory(
                &mut input,
                0,
                tensor.data.len() as u64,
                &ModelLoadConfig::default(),
            )?;
            let acoustic = validate_acoustic_directory(&directory, Some(kind))?;
            if kind == AcousticKind::Hubert {
                acoustic.validate_rvc_version(package.manifest.rvc_version()?)?;
                crate::rvc_version::validate_hubert_feature_tensors(
                    package.manifest.rvc_version()?,
                    directory
                        .tensors
                        .iter()
                        .map(|t| (t.name.as_str(), t.shape.as_slice(), t.dtype)),
                )?;
            }
        }
    }
    Ok(())
}
