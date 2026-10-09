//! Stream complete voice bundles and materialize their acoustic sources.
use crate::{
    AcousticKind, CACHE_ALIGNMENT, CACHE_HEADER_BYTES, CACHE_MAGIC, HUBERT_BUNDLE_TENSOR,
    ModelError, ModelLoadConfig, RMVPE_BUNDLE_TENSOR, RVC_TENSOR_SCHEMA, TensorDType, align_up,
    inspect_model_artifact,
    inspection::{TensorEntry, read_directory},
    pad_to, resolve_model_path, write_u16, write_u32, write_u64,
};
use std::{
    collections::hash_map::DefaultHasher,
    fs,
    hash::{Hash, Hasher},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::UNIX_EPOCH,
};

static TEMP_ID: AtomicU64 = AtomicU64::new(1);
fn temporary(path: &Path) -> PathBuf {
    path.with_extension(format!(
        "tmp-{}-{}.vfrm",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

fn stream_range(
    source: &Path,
    offset: u64,
    length: u64,
    output: &mut impl Write,
) -> Result<(), ModelError> {
    let mut input = io::BufReader::new(fs::File::open(source)?);
    input.seek(SeekFrom::Start(offset))?;
    if io::copy(&mut input.take(length), output)? != length {
        return Err(ModelError::InvalidCache(
            "component source was truncated".into(),
        ));
    }
    Ok(())
}

/// Create a Synth + index + HuBERT + RMVPE container using bounded memory.
/// Existing output files are never overwritten. Sources must be canonical VFRM.
pub fn bundle_rvc_cache(
    synth: impl AsRef<Path>,
    hubert: impl AsRef<Path>,
    rmvpe: impl AsRef<Path>,
    output: impl AsRef<Path>,
) -> Result<(), ModelError> {
    let synth = resolve_model_path(synth)?;
    let hubert = resolve_model_path(hubert)?;
    let rmvpe = resolve_model_path(rmvpe)?;
    let output = output.as_ref();
    if output.exists() {
        return Err(ModelError::InvalidConfig(
            "bundle output already exists".into(),
        ));
    }
    let config = ModelLoadConfig::default();
    let voice = inspect_model_artifact(&synth)?;
    let acoustic = [
        inspect_model_artifact(&hubert)?,
        inspect_model_artifact(&rmvpe)?,
    ];
    if !voice.components.synth.present
        || !acoustic[0].components.hubert.present
        || !acoustic[1].components.rmvpe.present
    {
        return Err(ModelError::UnsupportedModel(
            "bundle requires Synth, HuBERT and RMVPE weights".into(),
        ));
    }
    let mut input = io::BufReader::new(fs::File::open(&synth)?);
    let mut directory = read_directory(&mut input, 0, fs::metadata(&synth)?.len(), &config)?;
    if directory
        .metadata
        .get("canonical_schema")
        .and_then(serde_json::Value::as_str)
        != Some(RVC_TENSOR_SCHEMA)
    {
        return Err(ModelError::UnsupportedModel(
            "bundle Synth must use rvc-synth-v1".into(),
        ));
    }
    let manifest_version = directory.manifest.rvc_version()?;
    if let Some(graph_schema) = directory
        .metadata
        .get("graph_schema")
        .and_then(serde_json::Value::as_str)
    {
        let graph_version = if graph_schema.starts_with("rvc-v1-") {
            Some(crate::RvcVersion::V1)
        } else if graph_schema.starts_with("rvc-v2-") {
            Some(crate::RvcVersion::V2)
        } else {
            None
        };
        if graph_version.is_some_and(|version| version != manifest_version) {
            return Err(ModelError::InvalidCache(
                "bundle Synth graph_schema disagrees with RVC manifest version".into(),
            ));
        }
    }
    directory
        .tensors
        .retain(|t| !matches!(t.name.as_str(), HUBERT_BUNDLE_TENSOR | RMVPE_BUNDLE_TENSOR));
    let mut sources: Vec<_> = directory
        .tensors
        .iter()
        .map(|t| (synth.clone(), t.offset, t.len))
        .collect();
    for (path, name) in [
        (&hubert, HUBERT_BUNDLE_TENSOR),
        (&rmvpe, RMVPE_BUNDLE_TENSOR),
    ] {
        let len = fs::metadata(path)?.len();
        directory.tensors.push(TensorEntry {
            name: name.into(),
            dtype: TensorDType::U8,
            shape: vec![len],
            offset: 0,
            len,
        });
        sources.push((path.clone(), 0, len));
    }
    let metadata = serde_json::to_vec(
        &serde_json::json!({"manifest": directory.manifest, "metadata": directory.metadata}),
    )?;
    let directory_len = directory
        .tensors
        .iter()
        .map(|t| 24 + t.name.len() as u64 + 8 * t.shape.len() as u64)
        .sum::<u64>();
    let data_offset = align_up(
        CACHE_HEADER_BYTES + metadata.len() as u64 + directory_len,
        CACHE_ALIGNMENT,
    );
    let mut cursor = data_offset;
    for tensor in &mut directory.tensors {
        tensor.offset = align_up(cursor, CACHE_ALIGNMENT);
        cursor = tensor
            .offset
            .checked_add(tensor.len)
            .ok_or_else(|| ModelError::InvalidCache("bundle size overflows".into()))?;
    }
    let temp = temporary(output);
    let result = (|| -> Result<(), ModelError> {
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        let mut out = io::BufWriter::new(file);
        out.write_all(CACHE_MAGIC)?;
        write_u32(&mut out, 0)?;
        write_u32(&mut out, CACHE_HEADER_BYTES as u32)?;
        write_u64(&mut out, metadata.len() as u64)?;
        write_u64(&mut out, directory.tensors.len() as u64)?;
        write_u64(&mut out, directory_len)?;
        write_u64(&mut out, data_offset)?;
        write_u64(&mut out, 0)?;
        write_u64(&mut out, 0)?;
        out.write_all(&metadata)?;
        for t in &directory.tensors {
            write_u32(&mut out, t.name.len() as u32)?;
            out.write_all(&[t.dtype as u8, t.shape.len() as u8])?;
            write_u16(&mut out, 0)?;
            out.write_all(t.name.as_bytes())?;
            for &dim in &t.shape {
                write_u64(&mut out, dim)?;
            }
            write_u64(&mut out, t.offset)?;
            write_u64(&mut out, t.len)?;
        }
        for (t, (source, offset, len)) in directory.tensors.iter().zip(sources) {
            let position = out.stream_position()?;
            pad_to(&mut out, t.offset - position)?;
            stream_range(&source, offset, len, &mut out)?;
        }
        out.flush()?;
        out.get_ref().sync_all()?;
        drop(out);
        inspect_model_artifact(&temp)?;
        fs::rename(&temp, output)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

/// Extract embedded acoustic VFRM files atomically. Returns real paths usable
/// by the existing native builders and by configuration editors.
pub fn materialize_acoustic_sources(
    model: impl AsRef<Path>,
    cache_root: impl AsRef<Path>,
) -> Result<[PathBuf; 2], ModelError> {
    let model = resolve_model_path(model)?;
    let inspection = inspect_model_artifact(&model)?;
    if !inspection.components.automatic_tensor_rt {
        return Err(ModelError::UnsupportedModel(
            "model does not contain all TensorRT weights".into(),
        ));
    }
    let metadata = fs::metadata(&model)?;
    let mut hash = DefaultHasher::new();
    "vocal-acoustic-sources-v1".hash(&mut hash);
    model.hash(&mut hash);
    metadata.len().hash(&mut hash);
    metadata.modified().unwrap_or(UNIX_EPOCH).hash(&mut hash);
    let root = cache_root
        .as_ref()
        .join("sources")
        .join(format!("{:016x}", hash.finish()));
    fs::create_dir_all(&root)?;
    let root = fs::canonicalize(root)?;
    let mut input = io::BufReader::new(fs::File::open(&model)?);
    let directory = read_directory(&mut input, 0, metadata.len(), &ModelLoadConfig::default())?;
    let mut paths = [PathBuf::new(), PathBuf::new()];
    for (index, (name, kind)) in [
        (HUBERT_BUNDLE_TENSOR, AcousticKind::Hubert),
        (RMVPE_BUNDLE_TENSOR, AcousticKind::Rmvpe),
    ]
    .into_iter()
    .enumerate()
    {
        let entry = directory
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| ModelError::InvalidCache(format!("missing {name}")))?;
        let path = root.join(format!("{}.vfrm", kind.as_str()));
        let valid = || -> bool {
            fs::metadata(&path).is_ok_and(|m| m.len() == entry.len)
                && inspect_model_artifact(&path).is_ok_and(|i| match kind {
                    AcousticKind::Hubert => i.components.hubert.present,
                    AcousticKind::Rmvpe => i.components.rmvpe.present,
                })
        };
        if !valid() {
            let temp = temporary(&path);
            let result = (|| -> Result<(), ModelError> {
                let mut out = io::BufWriter::new(
                    fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&temp)?,
                );
                stream_range(&model, entry.offset, entry.len, &mut out)?;
                out.flush()?;
                out.get_ref().sync_all()?;
                drop(out);
                // Inspect using the VFRM reader regardless of the temp extension.
                let mut source = io::BufReader::new(fs::File::open(&temp)?);
                let nested =
                    read_directory(&mut source, 0, entry.len, &ModelLoadConfig::default())?;
                crate::inspection::validate_acoustic_directory(&nested, Some(kind))?;
                if path.exists() {
                    fs::remove_file(&path)?;
                }
                fs::rename(&temp, &path)?;
                Ok(())
            })();
            if result.is_err() {
                let _ = fs::remove_file(&temp);
            }
            result?;
        }
        paths[index] = path;
    }
    Ok(paths)
}
