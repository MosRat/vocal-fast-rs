//! CUDA resource policy shared by stream and buffer owners.
pub const DEFAULT_ARENA_CHUNK_BYTES: usize = 64 * 1024 * 1024;

pub fn checked_bytes(elements: usize, element_size: usize) -> Option<usize> {
    elements.checked_mul(element_size)
}
