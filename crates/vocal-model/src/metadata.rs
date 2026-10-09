//! Display metadata edits preserve the tensor directory and stream payload bytes.
use crate::{
    CACHE_ALIGNMENT, CACHE_HEADER_BYTES, ModelArtifactInspection, ModelError,
    inspect_model_artifact,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::UNIX_EPOCH,
};

static EDIT_LOCK: Mutex<()> = Mutex::new(());
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelMetadataEdit {
    pub display_name: String,
    pub source: String,
    pub notes: Option<String>,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelMetadataView {
    pub inspection: ModelArtifactInspection,
    pub revision: String,
    pub editable: bool,
    pub fields: ModelMetadataEdit,
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::InvalidConfig(message.into())
}
fn header(input: &mut File) -> Result<([u8; 64], serde_json::Value, Vec<u8>), ModelError> {
    let len = input.metadata()?.len();
    crate::inspection::read_directory(input, 0, len, &crate::ModelLoadConfig::default())?;
    input.rewind()?;
    let mut header = [0; 64];
    input.read_exact(&mut header)?;
    let n = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let d = u64::from_le_bytes(header[32..40].try_into().unwrap());
    let mut json = vec![0; n as usize];
    input.read_exact(&mut json)?;
    let mut directory = vec![0; d as usize];
    input.read_exact(&mut directory)?;
    Ok((header, serde_json::from_slice(&json)?, directory))
}

fn revision(input: &mut File) -> Result<String, ModelError> {
    let stat = input.metadata()?;
    let (_, json, _) = header(input)?;
    let mut hash = Sha256::new();
    hash.update(stat.len().to_le_bytes());
    hash.update(
        stat.modified()?
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .to_le_bytes(),
    );
    hash.update(serde_json::to_vec(&json)?);
    Ok(format!("{:x}", hash.finalize()))
}

pub fn read_model_metadata(path: impl AsRef<Path>) -> Result<ModelMetadataView, ModelError> {
    let inspection = inspect_model_artifact(path)?;
    let editable =
        crate::model_artifact_kind(&inspection.path) == Some(crate::ModelArtifactKind::Vfrm);
    let mut fields = ModelMetadataEdit {
        display_name: inspection.manifest.display_name.clone(),
        source: inspection.manifest.source.clone(),
        notes: inspection.manifest.notes.clone(),
        ..Default::default()
    };
    let revision = if editable {
        let mut file = File::open(&inspection.path)?;
        let (_, json, _) = header(&mut file)?;
        if let Some(info) = json.pointer("/metadata/libraryInfo") {
            fields.author = info["author"].as_str().unwrap_or_default().into();
            fields.language = info["language"].as_str().unwrap_or_default().into();
            fields.license = info["license"].as_str().unwrap_or_default().into();
            fields.tags = info["tags"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_owned))
                        .collect()
                })
                .unwrap_or_default();
        }
        revision(&mut file)?
    } else {
        String::new()
    };
    Ok(ModelMetadataView {
        inspection,
        revision,
        editable,
        fields,
    })
}

pub fn edit_model_metadata(
    path: impl AsRef<Path>,
    expected_revision: &str,
    mut edit: ModelMetadataEdit,
) -> Result<ModelMetadataView, ModelError> {
    let _lock = EDIT_LOCK
        .lock()
        .map_err(|_| invalid("metadata editor unavailable"))?;
    let path = crate::resolve_model_path(path)?;
    if crate::model_artifact_kind(&path) != Some(crate::ModelArtifactKind::Vfrm) {
        return Err(invalid("only VFRM metadata is editable"));
    }
    edit.display_name = edit.display_name.trim().into();
    if edit.display_name.is_empty() || edit.display_name.len() > 512 {
        return Err(invalid("model name must contain 1–512 UTF-8 bytes"));
    }
    if edit.source.len() > 4096
        || edit.notes.as_ref().is_some_and(|v| v.len() > 16384)
        || [&edit.author, &edit.language, &edit.license]
            .iter()
            .any(|v| v.len() > 512)
        || edit.tags.len() > 32
        || edit.tags.iter().any(|v| v.len() > 128)
    {
        return Err(invalid("model display metadata exceeds limits"));
    }
    let mut input = File::open(&path)?;
    if revision(&mut input)? != expected_revision {
        return Err(invalid(
            "model changed on disk; reload its metadata before saving",
        ));
    }
    let (mut hdr, mut json, mut directory) = header(&mut input)?;
    json["manifest"]["display_name"] = edit.display_name.into();
    json["manifest"]["source"] = edit.source.into();
    json["manifest"]["notes"] = serde_json::to_value(edit.notes)?;
    if !json["metadata"].is_object() && !json["metadata"].is_null() {
        return Err(invalid(
            "model computational metadata is not an object; cannot add library information safely",
        ));
    }
    if json["metadata"].is_null() {
        json["metadata"] = serde_json::json!({});
    }
    json["metadata"]["libraryInfo"] = serde_json::json!({"author":edit.author,"language":edit.language,"license":edit.license,"tags":edit.tags});
    let bytes = serde_json::to_vec(&json)?;
    let old_data = u64::from_le_bytes(hdr[40..48].try_into().unwrap());
    let new_data = (CACHE_HEADER_BYTES + bytes.len() as u64 + directory.len() as u64)
        .div_ceil(CACHE_ALIGNMENT)
        * CACHE_ALIGNMENT;
    let count = u64::from_le_bytes(hdr[24..32].try_into().unwrap());
    let mut cursor = 0;
    for _ in 0..count {
        let name_len =
            u32::from_le_bytes(directory[cursor..cursor + 4].try_into().unwrap()) as usize;
        let rank = directory[cursor + 5] as usize;
        cursor += 8 + name_len + rank * 8;
        let old = u64::from_le_bytes(directory[cursor..cursor + 8].try_into().unwrap());
        let offset = (old - old_data)
            .checked_add(new_data)
            .ok_or_else(|| invalid("tensor offset overflow"))?;
        directory[cursor..cursor + 8].copy_from_slice(&offset.to_le_bytes());
        cursor += 16;
    }
    hdr[16..24].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
    hdr[40..48].copy_from_slice(&new_data.to_le_bytes());
    let temp = path.with_file_name(format!(
        ".metadata-{}-{}.tmp.vfrm",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temp)?;
    let result = (|| -> Result<(), ModelError> {
        output.write_all(&hdr)?;
        output.write_all(&bytes)?;
        output.write_all(&directory)?;
        let position = output.stream_position()?;
        output.write_all(&vec![0; (new_data - position) as usize])?;
        input.seek(SeekFrom::Start(old_data))?;
        std::io::copy(&mut input, &mut output)?;
        output.sync_all()?;
        drop(output);
        inspect_model_artifact(&temp)?;
        let mut current = File::open(&path)?;
        if revision(&mut current)? != expected_revision {
            return Err(invalid("model changed while metadata was being saved"));
        }
        drop(current);
        drop(input);
        fs::rename(&temp, &path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result?;
    read_model_metadata(path)
}
