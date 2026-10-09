# DirectML execution, transfers and resource management

## Configuration and interfaces

Execution settings live in `DirectMlConfig.execution` and roundtrip through
`DirectMlSources` JSON. Old manifests receive these defaults:

```json
{
  "execution": {
    "reuse_device": true,
    "io_buffers": "dedicated",
    "pooled_weights": true,
    "weight_staging_mib": 16,
    "contiguous_views": false,
    "fuse_gemm_bias": false
  }
}
```

`weight_staging_mib` accepts 1–64 MiB. Unknown execution fields and invalid
capacities fail validation. Changing execution settings requires recompilation;
speaker, pitch, retrieval blend and valid frame count remain runtime controls.

```rust
use vocal_directml::{DirectMlConfig, DirectMlPipeline, ExecutionConfig, IoBuffers};

sources.config = DirectMlConfig {
    execution: ExecutionConfig {
        io_buffers: IoBuffers::Pooled,
        weight_staging_mib: 8,
        ..ExecutionConfig::default()
    },
    ..DirectMlConfig::realtime_44960()
};
let pipeline = DirectMlPipeline::from_sources(&sources)?;
```

For standalone stages, use `DirectMlDevice::from_config(&config)` or
`device.clone().with_execution(execution)`. The latter shares a device/queue
while giving newly compiled graphs their own execution policy. Offline stages
also receive this policy; they retain their bounded stage lifetimes and host
handoffs for full-context files. The existing Engine/Runtime/GUI source-manifest
path forwards these settings without adding environment-variable overrides.

Mixed precision is unchanged: HuBERT Transformer FP16 with FP32 normalization,
RMVPE FP32, Synth decoder stages 0–2 FP16 and stage 3/final output FP32. The
default RMVPE input remains the full window, including for the 44960 preset.

## Ownership and resource lifetime

- `device_pool.rs` caches **weak** device references by stable DXGI ID and debug
  setting. It resolves enumeration indices to IDs, removes expired entries,
  checks device health before reuse, and never retains an unloaded graph.
- Each graph owns descriptors, compiled weights, scratch, persistent buffers,
  its recorded command list and mutable history. Device reuse does not share
  weights or state. Device access is serialized by the existing Rust mutex.
- Initializer weights share one temporary default-heap arena, with 256-byte
  binding offsets. A reusable bounded upload arena handles tensors that span
  multiple batches and pads small FP16 constants. Both arenas are released
  after initialization has completed its fence.
- `transfer.cpp` implements optional pooled GPU bindings: dynamic input,
  state input, host output and feedback output use separate arenas. Normal
  replay uploads only the dynamic prefix; feedback/reset never overwrite
  host inputs. Each direction uses one copy when pooling is selected.
- Upload, audio readback and timestamp resources stay mapped across replay.
  Rust reuses the ordered FFI binding vector; pipeline input positions are
  computed once and assembled on the stack each call.
- Fence completion is checked before arming the event. Inference still waits
  for completion before reusing or releasing resources.

`debug_layer` enables DirectML validation and deploys `DirectML.Debug.dll` in
release builds as well. The library no longer calls process-global
`ID3D12Debug::EnableDebugLayer`: enabling it after devices exist can invalidate
them, including GUI/WebView devices. Hosts needing D3D12 validation must enable
it before creating any D3D12 devices. The regression test creates both normal
and debug DirectML devices in one process and verifies weak ownership.

Canonical files continue to use the per-user DirectML source cache. Compiled
operators remain in memory; this change does not introduce a serialized DML
plan cache. Cache entries do not retain GPU resources after graph destruction.

## Operator experiments

`layout.cpp` owns strides, broadcasts, reinterpretation and GEMM construction.
Dense activation reshape views can skip Identity; noncontiguous views,
parameter materialization and transpose layout conversion retain their needed
boundaries. GRU parameter flags are handled explicitly.

Optional GEMM+bias fusion uses GEMM's C operand. It requires a sole consumer,
unchanged output geometry/dtype and no marked intermediate output. Marked or
branched results keep their original computation. The tested graphs contain
87 eligible sites. FP16 rounding changes after this fusion, so it remains off.

These are experiments, not proven speedups. DirectML already optimizes the
submitted graph. The paired measurements below do not justify changing the
default operator lowering or default I/O bindings.

## Measured results

RTX 3060, original default voice, Chinese input, existing mixed precision,
full-window F0, no retrieval, 4 warmups and 60 measured iterations per pair.
Only **two graphs** remain resident: the legacy graph and one candidate.
Execution order alternates; both receive identical stochastic inputs. Other
applications remain running. Compare each row against its own paired reference
because background load and clocks change between pairs.

| Window | Candidate | Paired legacy GPU ms | Candidate GPU ms | Candidate host ms |
| ---: | --- | ---: | ---: | ---: |
| 11200 | Default, pooled initializer weights | 24.626 | 24.690 | 25.440 |
| 11200 | Pooled I/O | 27.688 | 27.651 | 28.509 |
| 11200 | Pooled I/O + dense views | 28.347 | 28.639 | 29.413 |
| 11200 | Pooled I/O + GEMM bias | 29.171 | 29.113 | 29.876 |
| 44960 | Default, pooled initializer weights | 35.961 | 35.955 | 38.181 |
| 44960 | Pooled I/O | 39.943 | 40.223 | 42.324 |
| 44960 | Pooled I/O + dense views | 42.918 | 43.193 | 45.391 |
| 44960 | Pooled I/O + GEMM bias | 43.760 | 43.416 | 45.659 |

GPU time covers dispatch and state feedback. Host time includes noise creation,
validation, upload, queue/fence waits and audio copy. Production downloads only
audio. `GpuTimings` additionally reports native upload and readback timestamps:
for the default 44960 pair, 0.03867 and 0.00460 ms; for 11200, 0.01182 and
0.00671 ms. Transfers are a small fraction of this workload. These pipeline
times exclude audio-device callbacks, resampling, SOLA and playback buffering.

An earlier benchmark kept four/five complete graphs resident simultaneously.
Background applications alone used 6218 MiB of the 12288 MiB GPU. Later graphs
showed 70–80 ms dispatch times under that memory pressure. Those measurements
are excluded from operator conclusions; the large regression did not reproduce
with paired residency. The benchmark now bounds graph lifetimes explicitly.

| Initialization / binding metric | Legacy 44960 | Default 44960 | Pooled I/O 44960 |
| --- | ---: | ---: | ---: |
| Temporary weight resources | 1010 | 1 | 1 |
| Weight allocation bytes, with D3D12 alignment | 698941440 | 651558912 | 651558912 |
| I/O GPU resources | 14 | 14 | 4 |
| I/O allocation bytes | 1638400 | 1638400 | 1048576 |
| Replay upload copy commands | 9 | 9 | 1 |

Initializer weight pooling saves **45.19 MiB** of transient GPU allocations.
For 11200 it saves **45.56 MiB**, reducing 1009 resources to one. These figures
are not process/adapter memory measurements. Scratch and persistent allocation
sizes are unchanged. No stable dispatch acceleration was established.

## Accuracy

Default, pooled I/O and dense views are **bitwise identical to legacy** over all
64 calls of each pair at both lengths. Optional GEMM+bias fusion changes output:

| Window | GEMM bias cosine vs legacy, last call | RMSE | Max absolute error |
| ---: | ---: | ---: | ---: |
| 11200 | 0.999996891 | 0.000477483 | 0.005532205 |
| 44960 | 0.999997110 | 0.000361294 | 0.002668500 |

Independent parity uses original FP32 PyTorch HuBERT, Applio RMVPE and Synth,
with captured identical noise. Reference TF32 is disabled. Production and
diagnostic graph audio were also compared at identical seeds and were bitwise
equal at both lengths.

| Window | Default audio cosine vs FP32 Python | RMSE | Max absolute error |
| ---: | ---: | ---: | ---: |
| 11200, voiced Chinese window | 0.999956627 | 0.001780887 | 0.039332092 |
| 44960 | 0.999993114 | 0.000553739 | 0.004585385 |

RMVPE argmax mismatches: zero at both lengths. F0 RMSE: 0.00008294 and
0.00007241 Hz respectively. The native-only Synth comparison, using captured
native frontend inputs, has cosine 0.999993539 / 0.999993344. These reference
differences are the existing mixed-precision policy, not errors introduced by
weight or transfer pooling. This is one model/two windows, not a listening test
or a multi-model precision guarantee.

## Reproduction and reports

```powershell
rtk cargo build --release -p vocal-directml --examples
rtk proxy target/release/examples/benchmark_execution.exe target/dml-sources.json target/chinese-input-44960.f32 target/compare/dml-execution/paired-final-44960.json 60
rtk proxy target/release/examples/benchmark_execution.exe target/dml-sources-11200.json target/compare/dml-execution/voiced-11200.f32 target/compare/dml-execution/paired-final-11200.json 60
rtk proxy target/release/examples/validate.exe pipeline target/dml-sources.json target/chinese-input-44960.f32 target/compare/dml-execution/reference-native-44960.json manifest 1 debug
rtk .venv/Scripts/python.exe scripts/compare-directml-python.py pipeline target/compare/dml-execution/reference-native-44960.json target/chinese-input-44960.f32 target/compare/dml-execution/python-44960.json
```

The voiced 11200 input is `target/dml-chinese-stream.f32[166400:177600]`, RMS
0.077313. The former 11200 fixture had RMS 0.005102 and was unsuitable as the
only voiced accuracy check. Generated reports, diagnostic tensors and F32 audio
remain under `target/compare/dml-execution/` and are not tracked.

GPU regression coverage includes mixed GEMM, multiple aligned inputs/outputs,
two feedback states, invalid bindings, reset/replay, typed FP16 output, dense
views following transpose, marked GEMM branches, bounded 1/16 MiB weight upload,
small FP16 constant padding, retrieval and device lifetime management.

Verification: 20 DirectML tests passed with ignored GPU tests explicitly
included; Engine/Runtime with `directml` passed 83 tests (2 ignored). The real
Runtime bundle/worker test was also run explicitly: retrieval ratio 0.5,
433600 input/output samples, 160-sample callbacks, finite converted output and
no inference failures. Workspace tests passed 237 tests (19 ignored);
`cargo fmt --all -- --check` and workspace `--all-features` check passed.

API references: [DirectML bindings](https://learn.microsoft.com/en-us/windows/ai/directml/dml-binding),
[tensor strides](https://learn.microsoft.com/en-us/windows/ai/directml/dml-strides),
[GEMM C operand](https://learn.microsoft.com/en-us/windows/win32/api/directml/ns-directml-dml_gemm_operator_desc).
