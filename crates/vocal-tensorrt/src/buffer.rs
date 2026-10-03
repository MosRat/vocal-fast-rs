//! Device buffer ownership and coarse grained sub-allocation.
use crate::{error::NativeBuildError, runtime::CudaBuffer};
use std::os::raw::c_void;
use vocal_model::{ModelPackage, TensorDType};

#[derive(Clone, Debug)]
pub struct WeightSegment {
    pub name: String,
    pub offset: usize,
    pub bytes: usize,
    pub shape: Vec<u64>,
    pub dtype: TensorDType,
}

#[derive(Clone, Debug, Default)]
pub struct WeightTable {
    pub storage: Vec<u8>,
    pub segments: Vec<WeightSegment>,
}

impl WeightTable {
    pub fn from_package(package: &ModelPackage) -> Result<Self, NativeBuildError> {
        package
            .validate_rvc_synth()
            .map_err(|e| NativeBuildError::Api(e.to_string()))?;
        let mut table = Self::default();
        for tensor in &package.tensors {
            if tensor.name.starts_with("aux.") {
                continue;
            }
            let aligned = (table.storage.len() + 63) & !63;
            table.storage.resize(aligned, 0);
            let offset = table.storage.len();
            table.storage.extend_from_slice(&tensor.data);
            table.segments.push(WeightSegment {
                name: tensor.name.clone(),
                offset,
                bytes: tensor.data.len(),
                shape: tensor.shape.clone(),
                dtype: tensor.dtype,
            });
        }
        Ok(table)
    }
    pub fn segment(&self, name: &str) -> Option<&WeightSegment> {
        self.segments.iter().find(|s| s.name == name)
    }
    pub fn bytes(&self, segment: &WeightSegment) -> &[u8] {
        &self.storage[segment.offset..segment.offset + segment.bytes]
    }
}

pub struct DeviceArena {
    chunks: Vec<CudaBuffer>,
    chunk_bytes: usize,
    used: usize,
}
pub struct DeviceSlice {
    pub ptr: *mut c_void,
    pub bytes: usize,
}
impl DeviceArena {
    pub fn new(chunk_bytes: usize) -> Result<Self, NativeBuildError> {
        if chunk_bytes == 0 {
            return Err(NativeBuildError::Argument(
                "arena chunk size is zero".into(),
            ));
        }
        Ok(Self {
            chunks: Vec::new(),
            chunk_bytes,
            used: 0,
        })
    }
    pub fn alloc(&mut self, bytes: usize) -> Result<DeviceSlice, NativeBuildError> {
        if bytes == 0 {
            return Err(NativeBuildError::Argument(
                "device allocation is zero".into(),
            ));
        }
        if bytes > self.chunk_bytes || self.used + bytes > self.chunk_bytes {
            self.chunks
                .push(CudaBuffer::new(self.chunk_bytes.max(bytes))?);
            self.used = 0;
        }
        let base = self.chunks.last().expect("chunk was created").device_ptr();
        let ptr = unsafe { (base as *mut u8).add(self.used) as *mut c_void };
        self.used += bytes;
        Ok(DeviceSlice { ptr, bytes })
    }
    pub fn chunks(&self) -> usize {
        self.chunks.len()
    }
}
