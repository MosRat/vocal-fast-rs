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
singularity. FAISS IDs are retained by the model parser; the CUDA search walks
inverted-list vector order and uses the corresponding vector values. The
parser requires the IDs to form a permutation of `0..ntotal` (arbitrary sparse
external IDs remain unsupported).

## Kernel layout

For `nprobe=1`, one block owns a query frame. Warps scan disjoint centroids,
choose the nearest list, scan disjoint candidate rows and merge their sorted
top-8 lists. Distance ties retain the original inverted-list order. The blend
is fused into the same kernel. FP32 arithmetic and distance reduction order
are preserved; no quantization or approximate coarse search is introduced.

The 256/768-dimensional RVC paths cache the query in registers and unroll its
distance loop. Other accepted widths use the same block algorithm with a
shared query. Measured dispatch uses 16 warps for batches of at most 64 frames
with at least 128 lists, and 8 warps otherwise. A 32-warp block was slower for
both target shapes. Multi-probe indexes (`nprobe=2..32`) retain the original
warp implementation.

There is no additional persistent workspace, device allocation, host transfer,
or kernel launch per call. The index remains immutable, the blend stays in
place, and a captured graph observes updates to the ratio buffer. Dispatch is
determined by index/shape, so the existing graph shape/lifetime rules apply.

Implementation boundaries:

- `native/retrieval.cu`: index ownership, validation, C ABI and dispatch;
- `native/retrieval_kernels.cuh`: FP32 primitives and multi-probe fallback;
- `native/retrieval_block.cuh`: cooperative search, top-k merge and blend;
- `src/retrieval.rs`: Rust ownership and checked buffer capacities.

## RTX 3060 measurements (2026-10-07)

Real `default-v2` index: 19,949 vectors, `IVF511,Flat,nprobe=1,d=768`, ratio 0.5.
The independent benchmark uses a real 140-frame raw HuBERT snapshot; its first
34 frames provide the 11200-profile-sized case. This controls input values
across versions; it is not a separate full 11200-sample pipeline benchmark.

CUDA events surround retrieval only. Each iteration resets features by D2D
outside the events; accuracy downloads occur after measurement. Events are
recorded outside graph capture so replay refreshes their timestamps. The
numbers include device-side launch gaps; Nsight separately measures kernels.

| query frames / profile size | old replay median | new replay median | speedup | new mean / p95 |
| --- | ---: | ---: | ---: | ---: |
| 34 / 11200 | 1.925 ms | 0.127 ms | 15.2x | 0.130 / 0.144 ms |
| 140 / 44960 | 2.003 ms | 0.242 ms | 8.3x | 0.295 / 0.412 ms |

20 warmups; 200 measured iterations before and 500 after. All samples,
including scheduling outliers, are retained. These are retrieval-stage gains.

Nsight Systems node tracing confirms replay kernel medians of **0.127 ms**
(34 frames) and **0.271 ms** (140 frames); the old 140-frame kernel measured
**2.050 ms**. The 140-frame launch changes from 35 blocks of 128 threads to
140 blocks of 256 threads. Registers/thread fall from 62 to 56, with zero
local memory/thread reported. The new shared allocation is only 3,664 bytes
per block (4,176 for the 16-warp variant). Thus more queries/candidates can
execute concurrently without a global scratch buffer.

Nsight Compute counters were unavailable (`ERR_NVGPUCTRPERM`); no measured SM
utilization or memory-bandwidth claim is made. Reports are under
`target/compare/retrieval-perf/`, including `nsight-summary.json`.

Full 44960 realtime pipeline validation retained native CUDA Graph replay
(2 captures, 38 replays). Retrieval measured 2.239 ms before and 0.279 ms after
in the final stage events. Whole-call 30-iteration means were 33.91 and
37.32 ms, with varying HuBERT/RMVPE/Synth scheduling; this run does **not**
establish a whole-pipeline latency improvement. The isolated measurements
above establish the retrieval improvement.

## Accuracy and regression coverage

- Real 34/140-frame outputs are bit-identical to the previous CUDA kernel.
- 140-frame blend vs Python FAISS: max absolute error **4.768e-7**, RMSE
  **1.643e-8**, cosine **0.9999999999999982**.
- The full 44960 diagnostic retains identical HuBERT, frontend and audio
  outputs. Original PTH / Applio FP32 Synth on the exact native inputs/noise
  and 31-frame tail crop: max **0.0008821**, RMSE **0.0001886**, cosine
  **0.999999833**. This is Synth-stage parity, not independent offline acoustic
  pipeline parity.
- FAISS fixtures cover Flat/IVF, widths 33/256/768/4096, short lists, an empty
  nearest list, exact matches, duplicate vectors and `nprobe=1/4/8`.
- Replay ratio changes to zero preserve the query bitwise without recapture.
- Compute Sanitizer memcheck passes real 34-frame and odd-width fixtures
  with zero errors. Rust tests reject undersized buffers and size overflow.

Reproduce (test executable hash varies with Cargo's dependency configuration):

```powershell
rtk cargo test -p vocal-tensorrt --lib --no-run
rtk .venv/Scripts/python.exe scripts/benchmark-retrieval.py `
  --snapshot target/compare/retrieval-fixed/zero-debug.json `
  --index models/voices/default-v2/added_IVF511_Flat_nprobe_1_default_v2.index `
  --binary target/debug/deps/vocal_tensorrt-804daac88a329e23.exe `
  --output target/compare/retrieval-perf/recheck --iterations 500
```

The snapshot must have ratio zero (raw HuBERT features). Omitting `--binary`
prepares fixtures only. Python/FAISS are validation tools, not runtime
dependencies. Timing summaries with raw samples are saved in `report.json`.

At runtime set `ConversionConfig.index_ratio` from Rust, or use
`VOCAL_INDEX_RATIO=0..1` for the CLI test commands.  A non-zero ratio without
an index in the VFRM package is rejected during pipeline setup.
