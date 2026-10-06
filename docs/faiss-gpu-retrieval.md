# GPU FAISS retrieval

The TensorRT-RTX RVC path can load the normalized `aux.retrieval.faiss`
payload directly into device memory.  It accepts the bounded FAISS formats
parsed by `vocal-model`: Flat-L2 (`IxF2`) and IVF-Flat-L2 (`IwFl` with a Flat
coarse quantizer).  The shipped RVC index is `IVF511,Flat,nprobe=1,d=768`.
No FAISS runtime or Python package is needed at inference time.

## Placement and semantics

Retrieval runs on the same CUDA stream after HuBERT and before the RVC
frontend.  HuBERT features are blended in place, so the RVC graph receives the
same tensor binding whether an index is enabled or not.  `index_rate` is held
in a one-element device buffer; changing it does not invalidate a captured
CUDA graph.

For every feature frame the implementation preserves the Applio/RVC rule:

1. search the configured IVF lists with squared L2 distance;
2. retain the nearest eight vectors;
3. use `weight = (1 / distance)^2`, normalized across the eight rows;
4. blend `index_rate * retrieved + (1 - index_rate) * query`.

An exact zero-distance match uses one-hot retrieval to avoid the inverse-square
singularity.  Non-contiguous FAISS IDs are retained by the model parser; the
CUDA search walks the same inverted-list vector order and therefore uses the
same vector values as `index.reconstruct_n` in the reference implementation.

## Kernel layout

The hot path uses one warp per HuBERT frame.  Each lane computes a strided
portion of the 768-dimensional distance and a shuffle reduction combines the
partial sums.  Lane zero performs the stable top-k insertion in the same scan
order as the scalar implementation, while all lanes participate in candidate
distance evaluation and the final feature blend.  `nprobe=1` has a dedicated
single-centroid insertion path; compatible indexes with up to 32 probes still
use the general path.

This avoids intermediate host downloads and replaces the former one-thread
scan over every centroid and vector.  The kernel is launched with four warps
per block, enough for the fixed 11200/44960 profiles without creating a large
block-level shared-memory requirement.

## Measured baseline

On the RTX 3060 with the real `IVF511,Flat,nprobe=1,d=768` index and a fixed
44960-sample profile, the warp implementation measured:

| measurement | result |
| --- | ---: |
| retrieval CUDA event (`index_rate=0.5`) | 1.92 ms |
| complete pipeline mean (10 iterations) | 78.2 ms |
| retrieved feature max absolute error vs Python FAISS | 2.98e-7 |
| retrieved feature RMSE vs Python FAISS | 6.70e-9 |
| retrieved feature cosine | 1.0 |

The previous scalar kernel measured about 39.8 ms for the same retrieval
stage.  The complete pipeline timing includes the selected HuBERT/RMVPE/Synth
plans and should be treated separately from the retrieval event.

At runtime set `ConversionConfig.index_ratio` from Rust, or use
`VOCAL_INDEX_RATIO=0..1` for the CLI test commands.  A non-zero ratio without
an index in the VFRM package is rejected during pipeline setup.
