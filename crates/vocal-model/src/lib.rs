//! RVC model metadata and the VFRM v1 tensor container.
//!
//! VFRM v1 is close to GGUF's layout: a fixed header, compact JSON
//! configuration, a binary tensor directory, and aligned tensor bytes.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs,
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};
use thiserror::Error;
use vocal_types::ModelDescriptor;

mod acoustic;
mod canonical;
mod onnx;
mod pickle;
mod schema;

pub use acoustic::{
    ACOUSTIC_TENSOR_SCHEMA, AcousticConfig, AcousticKind, AcousticPort, AcousticTensorMap,
};
pub use schema::{RVC_TENSOR_SCHEMA, RvcSynthConfig};

pub const CACHE_MAGIC: &[u8; 8] = b"VFRM\0\0\0\x01";
pub const CACHE_HEADER_BYTES: u64 = 64;
pub const CACHE_ALIGNMENT: u64 = 64;
const MAX_METADATA_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TENSORS: u64 = 100_000;
const MAX_TENSOR_BYTES: u64 = 16 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RvcModelManifest {
    pub id: String,
    pub display_name: String,
    pub architecture: String,
    pub version: String,
    pub sample_rate: u32,
    pub model_file: String,
    pub index_file: Option<String>,
    pub source: String,
    pub notes: Option<String>,
}

impl RvcModelManifest {
    pub fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            id: self.id.clone(),
            display_name: self.display_name.clone(),
            architecture: self.architecture.clone(),
            sample_rate: self.sample_rate,
            version: self.version.clone(),
        }
    }
    pub fn feature_dims(&self) -> usize {
        if self.version.to_ascii_lowercase().contains("v2")
            || self.architecture.to_ascii_lowercase().contains("v2")
        {
            768
        } else {
            256
        }
    }
    pub fn uses_f0(&self) -> bool {
        !self.architecture.to_ascii_lowercase().contains("nono")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TensorDType {
    U8 = 1,
    I32 = 2,
    I64 = 3,
    F16 = 4,
    F32 = 5,
    BF16 = 6,
}

impl TensorDType {
    fn from_byte(byte: u8) -> Result<Self, ModelError> {
        match byte {
            1 => Ok(Self::U8),
            2 => Ok(Self::I32),
            3 => Ok(Self::I64),
            4 => Ok(Self::F16),
            5 => Ok(Self::F32),
            6 => Ok(Self::BF16),
            _ => Err(ModelError::InvalidCache(format!(
                "unknown tensor dtype {byte}"
            ))),
        }
    }
    pub fn element_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 => 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub dtype: TensorDType,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn byte_tensor(name: impl Into<String>, data: Vec<u8>) -> Self {
        let len = data.len() as u64;
        Self {
            name: name.into(),
            dtype: TensorDType::U8,
            shape: vec![len],
            data,
        }
    }
    pub(crate) fn validate(&self) -> Result<(), ModelError> {
        if self.name.is_empty() || self.name.len() > u16::MAX as usize {
            return Err(ModelError::InvalidCache(
                "invalid tensor name length".into(),
            ));
        }
        if self.shape.len() > u8::MAX as usize {
            return Err(ModelError::InvalidCache("tensor rank exceeds u8".into()));
        }
        let elements = self.shape.iter().try_fold(1_u64, |a, &b| a.checked_mul(b));
        let expected = elements
            .and_then(|n| n.checked_mul(self.dtype.element_size() as u64))
            .ok_or_else(|| ModelError::InvalidCache("tensor shape overflows".into()))?;
        if expected != self.data.len() as u64 {
            return Err(ModelError::InvalidCache(format!(
                "tensor {} has {} bytes, expected {}",
                self.name,
                self.data.len(),
                expected
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ModelPackage {
    pub manifest: RvcModelManifest,
    pub metadata: Value,
    pub tensors: Vec<Tensor>,
}

impl ModelPackage {
    /// Check the complete canonical Synthesizer parameter directory before building a graph.
    pub fn validate_rvc_synth(&self) -> Result<RvcSynthConfig, ModelError> {
        if self
            .metadata
            .get("canonical_schema")
            .and_then(Value::as_str)
            != Some(RVC_TENSOR_SCHEMA)
        {
            return Err(ModelError::UnsupportedModel(
                "expected canonical rvc-synth-v1 package".into(),
            ));
        }
        let config = self.rvc_synth_config()?.ok_or_else(|| {
            ModelError::InvalidCache("canonical package has no rvc_config".into())
        })?;
        schema::validate_tensors(&config, &self.manifest, &self.tensors)?;
        Ok(config)
    }
    /// Find a tensor by its canonical RVC parameter name.
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    pub fn rvc_synth_config(&self) -> Result<Option<RvcSynthConfig>, ModelError> {
        self.metadata
            .get("rvc_config")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(ModelError::from)
    }
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("manifest JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid cache: {0}")]
    InvalidCache(String),
    #[error("missing model file: {0}")]
    MissingFile(PathBuf),
    #[error("unsupported RVC model: {0}")]
    UnsupportedModel(String),
}

pub fn load_manifest(path: impl AsRef<Path>) -> Result<RvcModelManifest, ModelError> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

pub fn load_directory(
    root: impl AsRef<Path>,
    manifest: RvcModelManifest,
) -> Result<ModelPackage, ModelError> {
    let root = root.as_ref();
    let model_path = root.join(&manifest.model_file);
    if !model_path.is_file() {
        return Err(ModelError::MissingFile(model_path));
    }
    let mut tensors = vec![Tensor::byte_tensor("rvc.pth", fs::read(model_path)?)];
    if let Some(file) = &manifest.index_file {
        let path = root.join(file);
        if !path.is_file() {
            return Err(ModelError::MissingFile(path));
        }
        tensors.push(Tensor::byte_tensor("aux.retrieval.faiss", fs::read(path)?));
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
    let root = root.as_ref();
    let model_path = root.join(&manifest.model_file);
    let mut package = match model_path
        .extension()
        .and_then(|x| x.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("pth") => load_rvc_pth(&model_path, manifest.clone())?,
        Some("onnx") => load_rvc_onnx(&model_path, manifest.clone())?,
        _ => {
            return Err(ModelError::UnsupportedModel(
                "RVC model must end in .pth or .onnx".into(),
            ));
        }
    };
    if let Some(file) = &manifest.index_file {
        let path = root.join(file);
        if !path.is_file() {
            return Err(ModelError::MissingFile(path));
        }
        package
            .tensors
            .push(Tensor::byte_tensor("aux.retrieval.faiss", fs::read(path)?));
    }
    Ok(package)
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
        let is_v2 = version.to_ascii_lowercase().contains('2');
        let expected_v2 = manifest.feature_dims() == 768;
        if is_v2 != expected_v2 {
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
    validate_canonical_package(package)?;
    let config = serde_json::to_vec(
        &serde_json::json!({"manifest": package.manifest, "metadata": package.metadata}),
    )?;
    if config.len() as u64 > MAX_METADATA_BYTES || package.tensors.len() as u64 > MAX_TENSORS {
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
    if header_size != CACHE_HEADER_BYTES
        || metadata_len > MAX_METADATA_BYTES
        || tensor_count > MAX_TENSORS
        || directory_len > MAX_TENSOR_BYTES
        || data_offset < CACHE_HEADER_BYTES + metadata_len + directory_len
    {
        return Err(ModelError::InvalidCache("invalid header bounds".into()));
    }
    let mut config_bytes = vec![0; metadata_len as usize];
    input.read_exact(&mut config_bytes)?;
    let config: Value = serde_json::from_slice(&config_bytes)?;
    let manifest: RvcModelManifest = serde_json::from_value(
        config
            .get("manifest")
            .cloned()
            .ok_or_else(|| ModelError::InvalidCache("config has no manifest".into()))?,
    )?;
    let metadata = config.get("metadata").cloned().unwrap_or(Value::Null);
    let directory_end = CACHE_HEADER_BYTES + metadata_len + directory_len;
    let mut descriptors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name_len = read_u32(&mut input)? as usize;
        let mut attrs = [0_u8; 4];
        input.read_exact(&mut attrs)?;
        let dtype = TensorDType::from_byte(attrs[0])?;
        let rank = attrs[1] as usize;
        if name_len > u16::MAX as usize || rank > 64 {
            return Err(ModelError::InvalidCache(
                "invalid tensor directory entry".into(),
            ));
        }
        let mut name_bytes = vec![0; name_len];
        input.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|_| ModelError::InvalidCache("tensor name is not UTF-8".into()))?;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(read_u64(&mut input)?);
        }
        let offset = read_u64(&mut input)?;
        let len = read_u64(&mut input)?;
        if offset < data_offset || len > MAX_TENSOR_BYTES || offset.checked_add(len).is_none() {
            return Err(ModelError::InvalidCache(
                "tensor data range is invalid".into(),
            ));
        }
        descriptors.push((name, dtype, shape, offset, len));
    }
    if input.stream_position()? > directory_end {
        return Err(ModelError::InvalidCache(
            "tensor directory exceeds declared length".into(),
        ));
    }
    let mut tensors = Vec::with_capacity(descriptors.len());
    for (name, dtype, shape, offset, len) in descriptors {
        input.seek(SeekFrom::Start(offset))?;
        let mut data = vec![0; len as usize];
        input.read_exact(&mut data)?;
        let tensor = Tensor {
            name,
            dtype,
            shape,
            data,
        };
        tensor.validate()?;
        tensors.push(tensor);
    }
    let package = ModelPackage {
        manifest,
        metadata,
        tensors,
    };
    validate_canonical_package(&package)?;
    Ok(package)
}

fn validate_canonical_package(package: &ModelPackage) -> Result<(), ModelError> {
    if package
        .metadata
        .get("canonical_schema")
        .and_then(Value::as_str)
        == Some(RVC_TENSOR_SCHEMA)
    {
        let config: RvcSynthConfig =
            serde_json::from_value(package.metadata.get("rvc_config").cloned().ok_or_else(
                || ModelError::InvalidCache("canonical cache has no rvc_config".into()),
            )?)?;
        schema::validate_tensors(&config, &package.manifest, &package.tensors)?;
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
