# Backend construction, memory and cache loading

## Implementation

TensorRT-RTX 1.6 has two compilation boundaries. A serialized plan does not
contain all device-specific runtime JIT kernels. The backend now persists
`IRuntimeCache`, following NVIDIA's
[runtime cache API](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/work-with-runtime-cache.html).
The SDK validates GPU, driver, SDK and CUDA-context compatibility. Rejected
blobs fall back to JIT; cache write errors are logged without failing a usable
engine.

After a first JIT and successful cache publication, the unused context is
replaced by one restored from compiled kernels. This releases the initial
context's compiler state before bindings or stream state are installed. Native
engines/weights remain shared. Restore failures retain the original context.

Other changes:

- Plans and JIT blobs use read-only mappings retained through synchronous SDK
  calls. Serialized builder output borrows `IHostMemory`, eliminating a full
  plan copy. Files are published atomically.
- Synchronous native builds borrow aligned F32 and F16/BF16 VFRM storage.
  Malformed byte extents are rejected before allocating conversion buffers.
- RMVPE weights stay alive across realtime/full builds, then are released
  before frontend/Synth construction.
- Five plans have independent keys containing effective stage options, shapes,
  source/weight content, linked builder-source fingerprint and SDK signature.
  Retrieval indices and embedded acoustics do not affect Synth keys. Acoustic
  content signatures are memoized by file identity, allowing identical files
  extracted to different voice directories to share plans.
- Stage hits verify size and content hash. Missing, interrupted or corrupted
  entries are misses; receipts are atomically published too. Source files follow
  the model loader's immutable-file/atomic-replacement contract.
- Per-file gates deduplicate concurrent package/native-engine loads without
  holding the global cache mutex across IO. Weak references allow unused
  packages and engines to be released; failures remain retryable.
- GUI preparation checks the complete pipeline fingerprint before decoding
  Synth/index payloads. Runtime graph settings do not invalidate plans. Its
  targeted TRT availability probe avoids unrelated DML device construction.
- GUI/CLI progress distinguishes compiled and reused stages. JIT persistence
  runs during construction and after the worker returns, outside audio callbacks
  and inference enqueue.
- Root/Tauri dev profiles optimize model decoding, DML and TRT code at level 2.
  Standalone TRT construction and CLI pipeline compilation selectively load
  voice tensors instead of embedded acoustic copies.

## Measurements

Windows, RTX 3060 12 GiB, driver 617.14, TensorRT-RTX 1.6.1.120; debug
executables using the GUI Runtime actor and mimalloc. Model:
`target/default-complete-with-index.vfrm`; 44,960-sample window, 16 kHz mono,
index ratio 0.5, protect 0.33. Before/after loading tests reused identical
existing plans and input. These are individual local observations, not percentile
benchmarks. Cold JIT means an empty application JIT cache, not flushed Windows
filesystem caches.

| Measurement | Before | After |
| --- | ---: | ---: |
| TRT loaded idle private commit, cold JIT | 8.061 GiB | 2.544 GiB |
| TRT peak commit across two loads, first JIT cold | 8.675 GiB | 4.460 GiB |
| TRT first load, cold JIT | 21.486 s | 20.202 s |
| TRT load with JIT reuse, fresh process | 21.486 s without reuse | about 2.4 s |
| TRT loaded idle commit with JIT reuse | 8.061 GiB without reuse | about 2.22 GiB |
| DML construction plus model load | 16.662 s | 6.238 s |

DML idle commit was 2.738 versus 2.719 GiB in the preceding memory fix. The
additional dev-profile change improves construction time, not its necessary
persistent GPU allocation budget.

Final plan measurements used a private C: cache after the workspace drive ran
short of space:

| Five-plan operation | Time | Reused |
| --- | ---: | ---: |
| Empty stage cache | 13.569 s | 0/5 |
| All stage entries present, with integrity checks | 0.885 s | 5/5 |
| Change Synth diagnostic outputs | 5.956 s | 4/5 |

GUI complete-pipeline hits took 83 ms on the first targeted probe, then
15.6 ms. Changing graph mode reused the same manifest in 17.3 ms. Before
removing unrelated DML discovery, complete-hit checks took 508–580 ms.
Preparation excludes subsequent engine loading, which has its own progress.

TRT's cold, JIT-hit, context replacement and damaged-cache outputs all matched
the baseline's 96,000 PCM16 samples exactly. DML matched its own baseline too.
The build probe uses requested GUI precision; those new plans differ from older
test artifacts' HuBERT frontend/Synth precision, so their audio is excluded
from the equality comparison against older artifacts.

The damaged-cache probe replaced the frontend `.jit` with invalid bytes. The
SDK rejected it, only that context repeated JIT, its file was repaired, and
conversion/unload/reload succeeded.

## Cache controls and limits

| Cache | Default |
| --- | --- |
| Runtime kernels | `%LOCALAPPDATA%/vocal-fast/cache/tensorrt/runtime/*.jit` |
| CLI stage plans | `%LOCALAPPDATA%/vocal-fast/cache/tensorrt/plans/` |
| GUI stage plans | `<tensorrt.cacheDirectory>/plans/` |
| GUI descriptors / extracted sources | Existing configured cache directory |

`VOCAL_TRT_RUNTIME_CACHE_DIR` and `VOCAL_TRT_PLAN_CACHE_DIR` override default
roots; an empty value disables the corresponding cache. The explicit
`compile_with_config_and_plan_cache` API accepts a private root or `None` for
uncached builds. Source signatures live in `plans/sources/`; plan receipts
record size/hash. JIT blobs are limited to 512 MiB per context.

If JIT cache publication fails, inference remains available, but compiler-state
memory cannot use the persisted-context handoff. New dynamic shapes can add
kernels during execution; these are saved after worker return. DirectML still
compiles graphs in process; no unsupported serialized DML graph format was
introduced. Persistent disk caches are not evicted by weak-cache memory cleanup.

## Reproduction and verification

Use private config/cache paths; the saved GUI config is never changed:

```powershell
rtk cargo build -p vocal-fast --all-features --example model_memory_probe --example plan_build_probe
rtk target/debug/examples/model_memory_probe.exe target/compare/idle-reload/trt.json target/test-assets/mandarin-continuous/male-01-liangzhu-16k.wav target/compare/backend-loading/repeat.json
rtk target/debug/examples/plan_build_probe.exe target/compare/idle-reload/trt.json $env:LOCALAPPDATA/Temp/vocal-fast-loading-check-20261008 target/compare/backend-loading/plan-repeat.json
rtk cargo test -p vocal-model -p vocal-tensorrt -p vocal-engine -p vocal-runtime --all-features -- --test-threads=1
rtk cargo test -p vocal-directml --lib -- --include-ignored --test-threads=1
rtk cargo check --workspace --all-features
rtk cargo test --workspace -- --test-threads=1
rtk cargo test --manifest-path apps/tauri/src-tauri/Cargo.toml --lib --all-features -- --test-threads=1
```

The ignored Tauri test `profile_tensor_rt_prepare_cache` checks real GUI
preparation and verifies runtime-only changes preserve the manifest. Set
`VOCAL_TEST_TRT_CONFIG`, `VOCAL_TEST_TRT_CACHE_DIR` and `VOCAL_TEST_TRT_REPORT`
to private paths, then run it with `--ignored --nocapture --test-threads=1`.
`trt_context_memory_probe` isolates context JIT memory.

Untracked reports/audio under `target/compare/backend-loading/` include
`trt-before.json`, `trt-cold-compact.json`, `trt-hit-final.json`,
`trt-corrupt-cache.json`, `dml-after.json`, `plan-build-final.json` and
`gui-prepare-targeted.json`. Earlier duplicate test plans on F: were removed
to recover test space; final plans live in the private C: cache above. No user
weights or user caches were removed.

GPU tests run serially. Running probes use copied EXEs because Windows locks
executing files. The updated separate GUI debug binary is
`apps/tauri/src-tauri/target/debug/vocal-fast-tauri-memory.exe`; the existing
user instance was not closed or replaced.

Verification passed: workspace all-features check; workspace all-features
tests (171 passed, 16 ignored); DML GPU-inclusive tests (13 passed); Tauri
all-features library tests (26 passed, 1 ignored); its real preparation-cache
test explicitly run (1 passed); root/Tauri formatting and `git diff --check`.
