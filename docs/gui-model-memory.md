# GUI model loading memory

For the subsequent TRT JIT memory fix and both backends' construction/cache
loading measurements, see [backend loading and caches](backend-loading-and-caches.md).

## Diagnosis

The running Windows GUI used DirectML and the 818.10 MiB
`default-complete-with-index.vfrm` bundle. Windows reported a 5.39 GiB peak
process commit; its later working set was approximately 598 MiB and its current
private commit approximately 1.93 GiB. Peak commit, current private commit,
working set, and adapter memory are different measurements.

A separate probe reproduced the larger post-load state using the same Runtime
control actor, Windows mimalloc allocator, real RTX 3060, and complete model.
The explicit DirectML manifest enabled retrieval and used the 44,960-sample
full F0 window. The 6-second input was the continuous Chinese male speech
fixture. The before and after runs used identical configuration, weights and
audio. No GUI configuration store was installed or modified.

The main allocations overlapped during graph construction:

1. Canonical HuBERT/RMVPE source packages and the parsed FAISS index.
2. Rust graph constants converted to execution precision: 682.69 MiB.
3. A second native `vector<vector<uint8_t>>` copy of those constants.
4. One committed upload resource per constant, retained until initialization
   ended. Small constants also incurred per-resource allocation alignment.
5. DEFAULT initialization weight buffers and DirectML persistent resources.
6. Freed Rust buffers whose mimalloc pages were retained by the long-lived
   control thread when it returned to blocking channel receive.

The shared model package cache uses weak references. Runtime intentionally
holds the voice-only package (approximately 112.66 MiB), rather than the full
818 MiB bundle and both acoustic packages for the entire loaded lifetime.
The measured behavior is allocation overlap and delayed reclamation; this
comparison does not establish an unbounded leak.

## Changes

- Native compilation borrows constants through a `ConstantView`; it no longer
  clones the entire Rust weight set. Source pointers are used only during the
  synchronous FFI call, and the initialization fence completes before return.
  Invalid constant extents are rejected before upload.
- Weight uploads share one arena capped at 16 MiB. Tensors can span batches,
  FP16 tensor padding is zeroed, and each batch waits for its fence before
  overwriting the arena or resetting its command allocator. The arena is
  released before allocating initialization scratch and persistent resources.
- Rust graph preparation and native compilation have separate ownership
  scopes. Source package references and the parsed index are dropped before
  compilation. A mimalloc collection at this construction boundary returns
  their freed pages before driver allocations begin.
- After initialization, owned DEFAULT weight buffers are released. Scratch
  larger than execution requirements is replaced with execution-sized scratch.
  Persistent resources remain alive with the compiled operator. This follows
  the documented [DirectML binding lifecycle](https://learn.microsoft.com/en-us/windows/ai/directml/dml-binding)
  and [GPU resource lifetime rules](https://learn.microsoft.com/en-us/windows/ai/directml/dml-resource-lifetime).
- Runtime marks heavy load, rebuild, pipeline reload and unload operations for
  memory collection, including failed candidates. Collection happens on the
  control actor after temporary values have been dropped and before replying
  to the controller. Parameter edits, snapshots, audio processing and polling
  do not collect. No process working-set trimming is used.
- `DirectMlGraph::memory()` exposes requested resource payload sizes, also
  recorded in tracing. These counters exclude driver overhead, heap alignment
  and shared adapter allocations; they are not a VRAM usage measurement.

## Measured comparison

Windows x64 debug builds, measurements via `GetProcessMemoryInfo`, GiB = 2^30
bytes. Values include the backend and driver resources in the probe process;
they exclude Tauri/WebView presentation processes.

| Measurement | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Loaded idle private commit | 4.270 GiB | 2.719 GiB | 36.3% |
| Loaded idle working set | 2.314 GiB | 0.588 GiB | 74.6% |
| Process peak commit across two loads | 5.986 GiB | 3.899 GiB | 34.9% |
| Private commit after second unload | 2.102 GiB | 0.600 GiB | 71.5% |

The first and second optimized unloaded states were 0.610 and 0.600 GiB.
Loaded idle commit differed by about 19 MiB between the first and second
loads. This is a two-cycle observation, not a long-duration soak test.

A further run using the GUI's saved configuration directly (48 kHz mono,
automatic complete-bundle sources, index ratio 0) measured 2.724 GiB loaded
idle commit, 0.589 GiB working set and 3.890 GiB peak commit across two loads.
Its private unloaded commit was 0.599 then 0.601 GiB. This probe also used a
private Runtime without changing the saved GUI configuration.

Output energy was exactly `402.146375413168` in the before and after probes.
The optimized 96,000-sample PCM16 output also matched the previous successful
idle reload regression WAV exactly: zero changed samples. Precision, window
length, retrieval and model controls were preserved.

The graph resource counters explain substantial necessary backend allocations:

| Resource | Requested payload |
| --- | ---: |
| Persistent operator resources | 1,416,634,368 bytes (1.319 GiB) |
| Execution scratch | 242,622,464 bytes (231.38 MiB) |
| Initialization scratch | 1,587,200 bytes (1.51 MiB) |
| Constant payload | 715,855,706 bytes (682.69 MiB), transient |
| Shared upload arena | 16,777,216 bytes (16 MiB), transient |
| Host input/output/timestamp resources | 896,784 bytes |

The initializer scratch was smaller than execution scratch for this model, so
the scratch replacement rule does not account for the measured reduction.
Further reducing persistent resources would require changing graph compilation,
precision or execution policy and validating the effect on performance/quality.

Generated reports remain outside Git:

- `target/compare/memory/before.json`
- `target/compare/memory/native-bounded.json`
- `target/compare/memory/idle-trim.json`
- `target/compare/memory/after.json` and `after.wav`
- `target/compare/memory/comparison.json`
- `target/compare/memory/gui-config.json` (the GUI's actual 48 kHz configuration)

## Reproduction and verification

```powershell
rtk cargo build -p vocal-fast --all-features --example model_memory_probe
rtk target/debug/examples/model_memory_probe.exe target/compare/idle-reload/dml.json target/test-assets/mandarin-continuous/male-01-liangzhu-16k.wav target/compare/memory/run.json
rtk cargo test -p vocal-directml --lib -- --include-ignored --test-threads=1
rtk cargo test -p vocal-engine -p vocal-runtime --all-features
rtk cargo check --workspace --all-features
rtk cargo test --workspace
rtk cargo test --manifest-path apps/tauri/src-tauri/Cargo.toml --lib --all-features
```

GPU tests run serially: enabling the D3D12 debug layer while another test in
the same process owns a device can remove that device. The added tests cover
uploads spanning the 16 MiB arena, FP16 padding, inference after source constants
are dropped, invalid extent rejection and reuse of the same device afterward.
GPU feedback replay/reset and existing graph operator tests also pass.

On Windows, wait for a probe to exit before rebuilding its executable, including
`cargo test --workspace` (which builds examples). The operating system locks an
executing EXE. The current GUI is left running; a separately named debug binary,
`apps/tauri/src-tauri/target/debug/vocal-fast-tauri-memory.exe`, contains these
changes. Close the existing GUI before opening it to avoid running two model
instances. No physical microphone/endpoint latency was measured in this memory
experiment.
