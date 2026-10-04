# Graph replay and model handoff cache

## Replay scope

The default `native_cuda_graph=true` path asks TensorRT-RTX to use
`kWHOLE_GRAPH_CAPTURE` for each execution context. The SDK owns capture and
replay for HuBERT, RMVPE, frontend, and Synth separately; the Rust counters
named `native_captures` and `native_replays` only record shape specialization
transitions observed by the pipeline. TensorRT-RTX does not expose its internal
graph cache counters.

The explicit fallback (`VOCAL_TRT_NATIVE_CUDA_GRAPH=0`) captures the complete
stage sequence on one stream. It can include cuFFT, all TensorRT enqueues, the
device F0 cache update, and event records. Waveform H2D, shape/address setup,
and final audio D2H stay outside the graph because they use callback-owned host
memory or dynamic shape state.

The mode can be selected directly:

```text
VOCAL_TRT_GRAPH_MODE=native_subgraphs
VOCAL_TRT_GRAPH_MODE=whole_pipeline
VOCAL_TRT_GRAPH_MODE=hybrid_subgraphs
VOCAL_TRT_GRAPH_MODE=disabled
```

The old `VOCAL_TRT_CUDA_GRAPH` and `VOCAL_TRT_NATIVE_CUDA_GRAPH` variables are
still accepted. If no explicit mode is supplied, their old combination maps to
native subgraphs, whole pipeline, or eager execution.

## Larger graphs and shape changes

The largest safe capture is therefore the whole device-resident pipeline. A
larger input shape increases the work inside one replay but does not remove the
quadratic/long-sequence cost of HuBERT attention or the linear convolution and
GRU work in RMVPE/Synth. The measured application shapes remain 11200 and
44960 samples; choosing 8192 merely to make a power-of-two graph would model a
different host window.

The fallback now retains a bounded LRU of shape-keyed executables. The default
capacity is two, covering the voice-changer and WebUI/Applio windows. Configure
it with:

```text
VOCAL_TRT_GRAPH_CACHE_SIZE=2
```

The key includes waveform/RMVPE/HuBERT/pitch/RVC lengths, streaming-F0 mode,
and F0-cache advance. A hit launches the existing executable; a miss captures
once and inserts it. When full, the least recently used executable is destroyed.
`cuda_graph_stats()` reports `graph_cache_entries`, `graph_cache_capacity`,
`graph_cache_hits`, `graph_cache_misses`, and `graph_evictions` in addition to
the legacy capture/replay counters.

The external cache is only used when all stages report stream-capturable and
the native strategy is disabled. If capture fails, the pipeline aborts that
capture, runs the same sequence eagerly, and disables the fallback for the
instance. This preserves the existing TensorRT-RTX safety behavior.

In hybrid mode the first graph contains serial cuFFT/HuBERT/RMVPE/F0-cache
work and the second contains Frontend/Synth. The two graph launches use the
same stream, so the dependency is implicit without a host synchronization.
Each shape key stores the available executables, and a miss captures and
launches each accepted graph before returning the first result. If only one
capture is accepted, the cache stores that executable and the other half
remains eager; if both are rejected,
the complete sequence falls back eagerly for the pipeline instance.

## Comparison command and current RTX result

The CLI compares the three graph modes with identical input, warmups and
iterations:

```powershell
rtk cargo run -p vocal-fast --features tensorrt-rtx -- model compare-trt-pipeline `
  target/opt-op16-s2-c2-11200/pipeline.json target/tsukuyomi-canonical.vfr `
  target/compare/sine-11200-input.f32 20
```

On the installed RTX 3060 plans, the 11200-sample run produced the following
mean timings (three warmups, two measured iterations in the smoke run):

| mode | total ms | graph status | output vs native |
| --- | ---: | --- | --- |
| native subgraphs | 17.17 | native replay active | reference |
| whole pipeline | 21.72 | external capture rejected, eager fallback | cosine 1.0, RMSE 0 |
| hybrid subgraphs | 25.14 | external capture rejected, eager fallback | cosine 1.0, RMSE 0 |

The whole/hybrid rows are valid compatibility and parity measurements, but they
do not claim an external graph speedup: `cuda_graph.enabled=false` and
`invalidations=1` show that the installed plans reported a non-capturable
TensorRT context. A future SDK/plan that reports capturable contexts will use
the same command to measure actual graph replay. The parity values in this run
are exact because those two rows executed the same eager kernels as the native
reference after native graph strategy was disabled.

The boundary-only update was smoke-tested on the same 11200-sample input with
five measured iterations. NativeSubgraphs reported `upload=0.0465 ms` and
`total=17.12 ms`; the outputs remained bit-identical across the compared modes.
The run also confirms that WholePipeline and HybridSubgraphs retain their
existing eager fallback behavior on these plans.

## Host and plan IO

`PinnedHostRing` keeps the recurring waveform upload and final audio download
asynchronous. Scalar `sid` and `p_len` updates use a second pinned ring on the
inference stream, avoiding a default-stream synchronization. Final FP32 audio
is copied directly from the pinned slot into the output vector, so the hot path
does not allocate a temporary byte buffer or parse samples one by one. The
remaining load path is now shared:

* `vocal_model::load_cache_shared` uses a weak process-wide cache keyed by
  canonical path, file size, modification time, and load policy. Repeated
  VFRM loads reuse one `Arc<ModelPackage>` and do not clone tensor bytes.
* `Runtime::shared` uses the same file identity strategy for serialized
  TensorRT plans. A shared `Arc<Runtime>` owns the deserialized engine; every
  pipeline still creates its own execution contexts and device buffers.

Both caches have explicit clear APIs for a model reload or device reset. A
weak entry never keeps an unused model or engine alive, and a changed file
automatically receives a new key. VFRM conversion still owns its F32 handoff
buffers only for the duration of a build, so plan compilation remains isolated
from runtime replay state.
