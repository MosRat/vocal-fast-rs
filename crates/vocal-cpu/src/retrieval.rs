//! IVF-Flat k=8 search without materializing a frame-by-index distance matrix.
use crate::CpuError;
use vocal_model::RetrievalIndex;

fn scalar_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum()
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_l2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= a.len() {
        let delta = _mm256_sub_ps(unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) }, unsafe {
            _mm256_loadu_ps(b.as_ptr().add(i))
        });
        sum = _mm256_add_ps(sum, _mm256_mul_ps(delta, delta));
        i += 8;
    }
    let mut lanes = [0.; 8];
    unsafe {
        _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    }
    lanes.iter().sum::<f32>() + scalar_l2(&a[i..], &b[i..])
}
pub(crate) fn blend(
    index: &RetrievalIndex,
    row: &mut [f32],
    ratio: f32,
    simd: bool,
) -> Result<(), CpuError> {
    if row.len() != index.dimensions() || index.is_empty() {
        return Err(CpuError::Config(
            "invalid retrieval dimensions or empty index".into(),
        ));
    }
    let mut distance: fn(&[f32], &[f32]) -> f32 = scalar_l2;
    #[cfg(target_arch = "x86_64")]
    if simd && std::arch::is_x86_feature_detected!("avx2") {
        distance = |a, b| unsafe { avx2_l2(a, b) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = simd;
    let mut lists = index
        .centroids()
        .chunks_exact(row.len())
        .enumerate()
        .map(|(id, c)| (distance(row, c), id))
        .collect::<Vec<_>>();
    lists.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut best = [(f32::INFINITY, usize::MAX); 8];
    for (_, list) in lists.into_iter().take(index.nprobe()) {
        for id in index.offsets()[list] as usize..index.offsets()[list + 1] as usize {
            let score = distance(row, &index.vectors()[id * row.len()..(id + 1) * row.len()]);
            let entry = (score, id);
            if entry.0 >= best[7].0 {
                continue;
            }
            let position = best.partition_point(|v| v.0 <= score);
            best.copy_within(position..7, position + 1);
            best[position] = entry;
        }
    }
    let count = best.iter().take_while(|v| v.1 != usize::MAX).count();
    if count == 0 {
        return Err(CpuError::Runtime("IVF search selected no candidate".into()));
    }
    let mut weights = [0f64; 8];
    for (w, (score, _)) in weights.iter_mut().zip(&best[..count]) {
        *w = 1. / f64::from(score.max(1e-12)).powi(2);
    }
    let sum = weights.iter().sum::<f64>();
    for (dim, value) in row.iter_mut().enumerate() {
        let retrieved = best[..count]
            .iter()
            .zip(weights)
            .map(|((_, id), w)| f64::from(index.vectors()[id * index.dimensions() + dim]) * w / sum)
            .sum::<f64>() as f32;
        *value = *value * (1. - ratio) + retrieved * ratio;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vector_distance_handles_unaligned_tail() {
        let a = (0..771).map(|x| x as f32 / 100.).collect::<Vec<_>>();
        let b = (0..771).map(|x| (x % 13) as f32).collect::<Vec<_>>();
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            let actual = unsafe { avx2_l2(&a[1..], &b[1..]) };
            let expected = scalar_l2(&a[1..], &b[1..]);
            assert!((actual - expected).abs() < expected * 1e-5);
        }
    }
}
