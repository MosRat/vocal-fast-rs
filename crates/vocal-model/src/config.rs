use crate::ModelError;

pub const CACHE_MAGIC: &[u8; 8] = b"VFRM\0\0\0\x01";
pub const CACHE_HEADER_BYTES: u64 = 64;
pub const CACHE_ALIGNMENT: u64 = 64;

pub const DEFAULT_MAX_METADATA_BYTES: u64 = 64 * 1024 * 1024;
pub const DEFAULT_MAX_TENSORS: u64 = 100_000;
pub const DEFAULT_MAX_TENSOR_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// Bounds and validation policy applied while reading or writing a model package.
#[derive(Clone, Debug)]
pub struct ModelLoadConfig {
    pub max_metadata_bytes: u64,
    pub max_tensors: u64,
    pub max_tensor_bytes: u64,
    pub validate_canonical: bool,
}

impl Default for ModelLoadConfig {
    fn default() -> Self {
        Self {
            max_metadata_bytes: DEFAULT_MAX_METADATA_BYTES,
            max_tensors: DEFAULT_MAX_TENSORS,
            max_tensor_bytes: DEFAULT_MAX_TENSOR_BYTES,
            validate_canonical: true,
        }
    }
}

impl ModelLoadConfig {
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.max_metadata_bytes == 0 || self.max_tensors == 0 || self.max_tensor_bytes == 0 {
            return Err(ModelError::InvalidConfig(
                "model cache bounds must be non-zero".into(),
            ));
        }
        if self.max_metadata_bytes > DEFAULT_MAX_METADATA_BYTES
            || self.max_tensors > DEFAULT_MAX_TENSORS
            || self.max_tensor_bytes > DEFAULT_MAX_TENSOR_BYTES
        {
            return Err(ModelError::InvalidConfig(
                "model cache bounds exceed the supported safety ceiling".into(),
            ));
        }
        Ok(())
    }
}
