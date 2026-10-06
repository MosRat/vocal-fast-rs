//! Bounded reader for the FAISS Flat-L2 / IVF-Flat-L2 subset used by RVC.
//! No FAISS runtime or Python is needed. Vectors remain in inverted-list order.

use crate::{ModelError, ModelPackage, TensorDType};

#[derive(Clone, Debug)]
pub struct RetrievalIndex {
    dimensions: usize,
    nprobe: usize,
    centroids: Vec<f32>,
    offsets: Vec<i32>,
    vectors: Vec<f32>,
    ids: Vec<i32>,
}

impl RetrievalIndex {
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    pub fn nprobe(&self) -> usize {
        self.nprobe
    }
    pub fn nlist(&self) -> usize {
        self.offsets.len() - 1
    }
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }
    pub fn offsets(&self) -> &[i32] {
        &self.offsets
    }
    pub fn vectors(&self) -> &[f32] {
        &self.vectors
    }
    pub fn ids(&self) -> &[i32] {
        &self.ids
    }

    pub fn from_package(package: &ModelPackage) -> Result<Option<Self>, ModelError> {
        let Some(tensor) = package.tensor("aux.retrieval.faiss") else {
            return Ok(None);
        };
        if tensor.dtype != TensorDType::U8 || tensor.shape != [tensor.data.len() as u64] {
            return Err(bad("expected a byte tensor for aux.retrieval.faiss"));
        }
        let index = Self::from_faiss(&tensor.data)?;
        if index.dimensions != package.manifest.feature_dims() {
            return Err(bad("index feature width does not match the voice model"));
        }
        Ok(Some(index))
    }

    /// Accept only little-endian, 64-bit FAISS Flat/IVFFlat with L2 distance,
    /// Flat-L2 coarse quantizer and in-memory ArrayInvertedLists. Reject all
    /// transforms, PQ, external files and non-contiguous user IDs explicitly.
    pub fn from_faiss(bytes: &[u8]) -> Result<Self, ModelError> {
        let mut r = Reader { bytes, position: 0 };
        let kind = r.tag()?;
        let (dimensions, count) = r.header()?;
        let result = match &kind {
            b"IxF2" => {
                let vectors = r.floats(
                    dimensions
                        .checked_mul(count)
                        .ok_or_else(|| bad("shape overflow"))?,
                    true,
                )?;
                Self {
                    dimensions,
                    nprobe: 1,
                    centroids: vec![0.0; dimensions],
                    offsets: vec![0, count as i32],
                    vectors,
                    ids: (0..count as i32).collect(),
                }
            }
            b"IwFl" => {
                let nlist = r.usize()?;
                let nprobe = r.usize()?;
                if nlist == 0 || nprobe == 0 || nprobe > nlist || nlist > bytes.len() / 8 {
                    return Err(bad("invalid IVF list/probe count"));
                }
                if r.tag()? != *b"IxF2" {
                    return Err(bad("IVF requires a Flat-L2 coarse quantizer"));
                }
                let (coarse_dimensions, coarse_count) = r.header()?;
                if coarse_dimensions != dimensions || coarse_count != nlist {
                    return Err(bad("coarse quantizer shape mismatch"));
                }
                let centroids = r.floats(
                    nlist
                        .checked_mul(dimensions)
                        .ok_or_else(|| bad("centroid shape overflow"))?,
                    true,
                )?;
                // Direct maps are load-time lookup accelerators, not search data.
                let map_kind = r.take(1)?[0];
                let map_len = r.usize()?;
                if map_kind > 2
                    || (map_kind == 0 && map_len != 0)
                    || (map_kind == 1 && map_len != count)
                {
                    return Err(bad("invalid direct map"));
                }
                r.skip_elements(map_len, 8)?;
                if map_kind == 2 {
                    let n = r.usize()?;
                    r.skip_elements(n, 16)?;
                }
                if r.tag()? != *b"ilar" {
                    return Err(bad("only ArrayInvertedLists is supported"));
                }
                if r.usize()? != nlist || r.usize()? != dimensions * 4 {
                    return Err(bad("inverted-list layout mismatch"));
                }
                let sizes_kind = r.tag()?;
                let sizes_len = r.usize()?;
                // Bound allocation by the bytes that encode the centroids.
                let mut sizes = vec![0usize; nlist];
                match &sizes_kind {
                    b"full" if sizes_len == nlist => {
                        for size in &mut sizes {
                            *size = r.usize()?;
                        }
                    }
                    b"sprs" if sizes_len % 2 == 0 && sizes_len / 2 <= nlist => {
                        for _ in 0..sizes_len / 2 {
                            let list = r.usize()?;
                            let size = r.usize()?;
                            if list >= nlist || size == 0 || sizes[list] != 0 {
                                return Err(bad("invalid sparse list directory"));
                            }
                            sizes[list] = size;
                        }
                    }
                    _ => return Err(bad("invalid inverted-list size directory")),
                }
                let total = sizes
                    .iter()
                    .try_fold(0usize, |a, &b| a.checked_add(b))
                    .ok_or_else(|| bad("list size overflow"))?;
                if total != count {
                    return Err(bad("list sizes disagree with ntotal"));
                }
                let payload = count
                    .checked_mul(dimensions * 4 + 8)
                    .ok_or_else(|| bad("payload overflow"))?;
                if payload != r.remaining() {
                    return Err(bad("truncated or trailing IVF payload"));
                }
                let mut vectors = Vec::with_capacity(count * dimensions);
                let mut ids = Vec::with_capacity(count);
                let mut offsets = Vec::with_capacity(nlist + 1);
                let mut seen = vec![false; count];
                offsets.push(0);
                for size in sizes {
                    vectors.extend(r.floats(size * dimensions, false)?);
                    for _ in 0..size {
                        let id = r.usize()?;
                        if id >= count || seen[id] {
                            return Err(bad("RVC requires unique IDs in 0..ntotal"));
                        }
                        seen[id] = true;
                        ids.push(id as i32);
                    }
                    offsets.push(ids.len() as i32);
                }
                Self {
                    dimensions,
                    nprobe,
                    centroids,
                    offsets,
                    vectors,
                    ids,
                }
            }
            _ => {
                return Err(bad(
                    "supported FAISS types are IxF2 and IwFl (L2 Flat/IVFFlat)",
                ));
            }
        };
        if r.remaining() != 0 {
            return Err(bad("unexpected trailing data"));
        }
        Ok(result)
    }
}

fn bad(message: &str) -> ModelError {
    ModelError::UnsupportedModel(format!("RVC retrieval index: {message}"))
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }
    fn take(&mut self, count: usize) -> Result<&'a [u8], ModelError> {
        let end = self
            .position
            .checked_add(count)
            .filter(|&n| n <= self.bytes.len())
            .ok_or_else(|| bad("truncated index"))?;
        let result = &self.bytes[self.position..end];
        self.position = end;
        Ok(result)
    }
    fn tag(&mut self) -> Result<[u8; 4], ModelError> {
        Ok(self.take(4)?.try_into().unwrap())
    }
    fn usize(&mut self) -> Result<usize, ModelError> {
        usize::try_from(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
            .map_err(|_| bad("size overflow"))
    }
    fn skip_elements(&mut self, count: usize, size: usize) -> Result<(), ModelError> {
        self.take(
            count
                .checked_mul(size)
                .ok_or_else(|| bad("size overflow"))?,
        )?;
        Ok(())
    }
    fn header(&mut self) -> Result<(usize, usize), ModelError> {
        let d = i32::from_le_bytes(self.tag()?);
        let count = self.usize()?;
        self.take(16)?; // legacy dummy fields
        let trained = self.take(1)?[0];
        let metric = i32::from_le_bytes(self.tag()?);
        if !(1..=4096).contains(&d)
            || count == 0
            || count > i32::MAX as usize
            || trained != 1
            || metric != 1
        {
            return Err(bad(
                "requires a populated, trained L2 index, width <= 4096 and i32 row count",
            ));
        }
        Ok((d as usize, count))
    }
    fn floats(&mut self, count: usize, length_prefix: bool) -> Result<Vec<f32>, ModelError> {
        if length_prefix && self.usize()? != count {
            return Err(bad("float vector length mismatch"));
        }
        let bytes = self.take(
            count
                .checked_mul(4)
                .ok_or_else(|| bad("float shape overflow"))?,
        )?;
        let values: Vec<_> = bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(bad("non-finite vector"));
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn header(out: &mut Vec<u8>, kind: &[u8; 4], d: i32, n: u64) {
        out.extend(kind);
        out.extend(d.to_le_bytes());
        out.extend(n.to_le_bytes());
        out.extend([0; 16]);
        out.push(1);
        out.extend(1i32.to_le_bytes());
    }
    fn flat() -> Vec<u8> {
        let mut b = Vec::new();
        header(&mut b, b"IxF2", 2, 2);
        b.extend(4u64.to_le_bytes());
        for v in [0f32, 1., 2., 3.] {
            b.extend(v.to_le_bytes());
        }
        b
    }
    #[test]
    fn flat_and_malformed_inputs() {
        let b = flat();
        let ix = RetrievalIndex::from_faiss(&b).unwrap();
        assert_eq!(ix.vectors(), &[0., 1., 2., 3.]);
        assert_eq!(ix.ids(), &[0, 1]);
        for n in 0..b.len() {
            assert!(RetrievalIndex::from_faiss(&b[..n]).is_err());
        }
        let mut bad = b.clone();
        bad.extend([0]);
        assert!(RetrievalIndex::from_faiss(&bad).is_err());
        let mut bad = b;
        bad[45..49].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(RetrievalIndex::from_faiss(&bad).is_err());
    }
    #[test]
    fn ivf_lists_preserve_vectors_and_ids() {
        let mut b = Vec::new();
        header(&mut b, b"IwFl", 2, 2);
        b.extend(2u64.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        b.extend(flat());
        b.push(0);
        b.extend(0u64.to_le_bytes());
        b.extend(b"ilar");
        b.extend(2u64.to_le_bytes());
        b.extend(8u64.to_le_bytes());
        b.extend(b"full");
        b.extend(2u64.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        for (v, id) in [(4f32, 1u64), (8., 0)] {
            b.extend(v.to_le_bytes());
            b.extend(v.to_le_bytes());
            b.extend(id.to_le_bytes());
        }
        let ix = RetrievalIndex::from_faiss(&b).unwrap();
        assert_eq!(ix.offsets(), &[0, 1, 2]);
        assert_eq!(ix.ids(), &[1, 0]);
        assert_eq!(ix.vectors(), &[4., 4., 8., 8.]);
        for n in 0..b.len() {
            assert!(RetrievalIndex::from_faiss(&b[..n]).is_err());
        }
        let end = b.len();
        b[end - 8..].copy_from_slice(&1u64.to_le_bytes());
        assert!(RetrievalIndex::from_faiss(&b).is_err());
    }
}
