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

## Host and plan IO

`PinnedHostRing` already keeps the recurring waveform upload and final audio
download asynchronous. The remaining load path is now shared:

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

