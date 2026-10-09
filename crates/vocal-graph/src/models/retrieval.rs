use super::*;
use vocal_model::RetrievalIndex;

fn distances(g: &GraphBuilder, query: &Tensor, vectors: &Tensor) -> Tensor {
    let q = g.reduce(Op::Sum, &g.mul(query, query), &[1]);
    let v = g.transpose(&g.reduce(Op::Sum, &g.mul(vectors, vectors), &[1]), &[1, 0]);
    let dot = g.scale(&g.matmul(query, vectors, true), 2.);
    g.binary(
        Op::Max,
        &g.sub(&g.add(&q, &v), &dot),
        &g.scalar(0., Precision::Fp32),
    )
}
pub fn retrieval(
    g: &GraphBuilder,
    query: &Tensor,
    index: &RetrievalIndex,
    rate: &Tensor,
    search: crate::RetrievalSearch,
) -> Result<Tensor, GraphError> {
    if index.is_empty() || query.shape[2] != index.dimensions() || index.nprobe() > 32 {
        return Err(GraphError::Graph(
            "unsupported or mismatched retrieval index".into(),
        ));
    }
    let frames = query.shape[1];
    let width = index.dimensions();
    let count = index.len();
    let q = g.reshape(query, &[frames, width]);
    let vectors = g.constant(index.vectors(), &[count, width], Precision::Fp32);
    let capacity = index
        .offsets()
        .windows(2)
        .map(|x| (x[1] - x[0]) as usize)
        .max()
        .unwrap_or(0);
    let candidates = capacity * index.nprobe();
    let sparse_bytes = frames
        .checked_mul(candidates)
        .and_then(|n| n.checked_mul(width * 4));
    let sparse_supported = index.nlist() > 1
        && candidates < count
        && sparse_bytes.is_some_and(|n| n <= 256 * 1024 * 1024);
    let sparse =
        match search {
            crate::RetrievalSearch::Dense => false,
            crate::RetrievalSearch::Auto => sparse_supported,
            crate::RetrievalSearch::Sparse if sparse_supported => true,
            _ => return Err(GraphError::Config(
                "sparse IVF candidates exceed the 256 MiB window bound or cannot prune this index"
                    .into(),
            )),
        };
    let (scores, candidates_map) = if sparse {
        sparse_scores(g, &q, &vectors, index, capacity)
    } else {
        (dense_scores(g, &q, &vectors, index), None)
    };
    tracing::info!(
        sparse,
        frames,
        vectors = count,
        candidates = scores.shape[1],
        "building DirectML retrieval search"
    );
    let k = scores.shape[1].min(8);
    let local_ids = g.cast(&g.topk(&scores, k, false, true), DType::I32);
    let ids = if let Some(map) = candidates_map {
        g.gather_elements(&map, &local_ids, 1)
    } else {
        local_ids
    };
    let scores = g.topk(&scores, k, false, false);
    let first = g.slice(&scores, 1, 0, 1, 1);
    let exact = g.binary(Op::Less, &first, &g.scalar(1e-12, Precision::Fp32));
    let scores = g.binary(Op::Max, &scores, &g.scalar(1e-12, Precision::Fp32));
    let inverse = g.div(&g.scalar(1., Precision::Fp32), &g.mul(&scores, &scores));
    let sum = g.reduce(Op::Sum, &inverse, &[1]);
    let normal = g.div(
        &inverse,
        &g.binary(Op::Max, &sum, &g.scalar(1e-30, Precision::Fp32)),
    );
    let mut onehot = vec![0.; k];
    onehot[0] = 1.;
    let weights = g.select(
        &exact,
        &g.constant(&onehot, &[1, k], Precision::Fp32),
        &normal,
    );
    let neighbors = g.gather(&vectors, &ids, 0);
    let retrieved = g.reshape(
        &g.reduce(
            Op::Sum,
            &g.mul(&neighbors, &g.reshape(&weights, &[frames, k, 1])),
            &[1],
        ),
        &[1, frames, width],
    );
    let retrieved = g.select(
        &g.reshape(
            &g.binary(Op::Greater, &sum, &g.scalar(0., Precision::Fp32)),
            &[1, frames, 1],
        ),
        &retrieved,
        query,
    );
    Ok(g.add(
        &g.mul(&retrieved, rate),
        &g.mul(query, &g.sub(&g.scalar(1., Precision::Fp32), rate)),
    ))
}

/// Gather just the selected IVF lists. Padding stores a safe vector index and
/// an explicit validity mask, so empty/small lists never gather out of bounds.
fn sparse_scores(
    g: &GraphBuilder,
    q: &Tensor,
    vectors: &Tensor,
    index: &RetrievalIndex,
    capacity: usize,
) -> (Tensor, Option<Tensor>) {
    let frames = q.shape[0];
    let width = index.dimensions();
    let candidates = capacity * index.nprobe();
    let centroids = g.constant(index.centroids(), &[index.nlist(), width], Precision::Fp32);
    let probes = g.topk(&distances(g, q, &centroids), index.nprobe(), false, true);
    let mut table = vec![0; index.nlist() * capacity];
    let mut valid = vec![0.; table.len()];
    for list in 0..index.nlist() {
        let begin = index.offsets()[list] as usize;
        let end = index.offsets()[list + 1] as usize;
        for (slot, row) in (begin..end).enumerate() {
            table[list * capacity + slot] = row as i32;
            valid[list * capacity + slot] = 1.;
        }
    }
    let ids = g.reshape(
        &g.gather(&g.integers(&table, &[index.nlist(), capacity]), &probes, 0),
        &[frames, candidates],
    );
    let valid = g.reshape(
        &g.gather(
            &g.constant(&valid, &[index.nlist(), capacity], Precision::Fp32),
            &probes,
            0,
        ),
        &[frames, candidates],
    );
    let bank = g.gather(vectors, &ids, 0);
    // Norms depend only on weights. Compute once with F64 accumulation and
    // gather a small scalar table, avoiding another full candidate-bank pass.
    let norms = index
        .vectors()
        .chunks_exact(width)
        .map(|row| row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() as f32)
        .collect::<Vec<_>>();
    let norms = g.gather(
        &g.constant(&norms, &[index.len()], Precision::Fp32),
        &ids,
        0,
    );
    let qnorm = g.reduce(Op::Sum, &g.mul(q, q), &[1]);
    let dot = g.reshape(
        &g.matmul(&g.reshape(q, &[frames, 1, width]), &bank, true),
        &[frames, candidates],
    );
    let scores = g.binary(
        Op::Max,
        &g.sub(&g.add(&qnorm, &norms), &g.scale(&dot, 2.)),
        &g.scalar(0., Precision::Fp32),
    );
    let scores = g.select(
        &g.binary(Op::Greater, &valid, &g.scalar(0., Precision::Fp32)),
        &scores,
        &g.scalar(1e30, Precision::Fp32),
    );
    (scores, Some(ids))
}

fn dense_scores(g: &GraphBuilder, q: &Tensor, vectors: &Tensor, index: &RetrievalIndex) -> Tensor {
    let count = index.len();
    let width = index.dimensions();
    let mut scores = distances(g, q, vectors);
    if index.nlist() > 1 {
        let centroids = g.constant(index.centroids(), &[index.nlist(), width], Precision::Fp32);
        let probes = g.topk(&distances(g, q, &centroids), index.nprobe(), false, true);
        let mut list_ids = vec![0; count];
        for list in 0..index.nlist() {
            list_ids[index.offsets()[list] as usize..index.offsets()[list + 1] as usize]
                .fill(list as i32);
        }
        let labels = g.integers(&list_ids, &[1, count]);
        let mut selected = g.cast(
            &g.binary(
                Op::Equals,
                &labels,
                &g.cast(&g.slice(&probes, 1, 0, 1, 1), DType::I32),
            ),
            DType::F32,
        );
        for p in 1..index.nprobe() {
            let mask = g.cast(
                &g.binary(
                    Op::Equals,
                    &labels,
                    &g.cast(&g.slice(&probes, 1, p, 1, 1), DType::I32),
                ),
                DType::F32,
            );
            selected = g.add(&selected, &mask);
        }
        scores = g.select(
            &g.binary(Op::Greater, &selected, &g.scalar(0., Precision::Fp32)),
            &scores,
            &g.scalar(1e30, Precision::Fp32),
        );
    }
    scores
}
