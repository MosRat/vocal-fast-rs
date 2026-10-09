# Native DirectML backend

Current execution settings, memory lifetimes and paired GPU measurements are
documented in [DirectML execution tuning](directml-execution-tuning.md).

## Architecture

`vocal-directml` builds handwritten, fixed shape HuBERT → RMVPE → RVC frontend
→ Synth graphs from the existing canonical VFRM packages. Optional retrieval
is another GPU region in this graph. Inference uses native DirectML/D3D12;
Python and ONNX Runtime are used only by the independent parity tooling.

| Module | Responsibility |
| --- | --- |
| `config.rs` | Precision regions, adapter, static window and diagnostic settings |
| `graph.rs` | Typed operator IR, shapes, broadcasts, weight packing and parameter lookup |
| `models/` | HuBERT, RMVPE, Synth and retrieval architectures |
| `stages.rs` | Standalone model compilation for parity and profiling |
| `runtime.rs` | Rust ownership, serialized device access, bindings and typed errors |
| `device_pool.rs` | Stable adapter identity, weak device reuse and health checks |
| `native/device.cpp` | D3D12 device, queue, fence and allocations |
| `native/bindings.cpp` | Owned weight initialization, persistent/scratch storage, descriptors |
| `native/graph.cpp` | DirectMLX lowering and graph compilation |
| `native/runtime.cpp` | Recorded command list, uploads, dispatch, readback, GPU timestamps |
| `native/transfer.cpp` | Dedicated/pooled GPU I/O arenas and upload/readback commands |
| `native/layout.cpp` | Tensor strides, reshape views, broadcast and GEMM lowering |
| `native/state.cpp` | GPU feedback copies, zero initialization and reusable state reset command list |
| `sources.rs` | Source manifests, atomic publication and complete bundle extraction |
| `f0.rs` | RMVPE window selection and GPU rolling F0 assembly |
| `pipeline.rs` | Whole graph assembly, runtime controls and deterministic stochastic inputs |
| `vocal-engine/directml_backend.rs` | Callback staging, context, RMS envelope, SOLA/crossfade, lifecycle |

Unreachable nodes and input ports are removed before native compilation.
Weights are initialized once into DirectML owned resources; initialization
staging allocations are then released. Each graph keeps a closed D3D12 command
list, mapped upload/readback buffers, persistent storage and descriptors.
Replay uploads the new waveform/controls/noise and downloads final audio.
Production audio is bound directly to its Rust `Vec<f32>` destination; random
input workspaces are reused. Diagnostic tensor snapshots are optional.
Intermediate features remain on the GPU. A device mutex protects queue/fence
access when graphs share a device; Rust ownership keeps the device alive until
its graphs are dropped.

The implementation uses [DirectMLX graph compilation](https://learn.microsoft.com/en-us/windows/ai/directml/dml-directmlx).
The execution flag is `NONE`: FP16 regions are explicit; FP32 regions are not
silently lowered by an allow-half flag.

## Precision policy

The measured default is:

| Region | Default |
| --- | --- |
| HuBERT feature extractor | FP32 |
| HuBERT Transformer / positional convolution | FP16 weights and activations |
| HuBERT normalization moments and affine | FP32, cast back at region boundary |
| RMVPE STFT/mel, encoder, decoder, GRU, logits and F0 decode | FP32 |
| Synth encoder, reverse flow, NSF source, pre/conditioning | FP32 |
| Synth decoder stages 0–2, including noise convolutions | FP16 weights and activations |
| Synth decoder stage 3 and final convolution/tanh | FP32 |
| Retrieval distances and feature blend | FP32 |

`PrecisionConfig::fp32()` selects the FP32 reference configuration.
`rmvpe_encoder` and `rmvpe_decoder` independently allow FP16 while STFT,
head convolution, GRU and logits remain FP32. The Synth stage masks are
zero based: default `synth_fp32_stage_mask=8` keeps stage 3 FP32, and
`synth_noise_fp16_stage_mask=7` lowers the noise convolutions in stages 0–2.
Masks must fit the selected model's upsample stage count.

`synth_input_scale` can scale the decoder input before the half cast. Its
reciprocal is folded into the first upsample weight in FP32 before packing
to the execution dtype. The default is 1. Other policies require model/audio
parity validation; TensorRT measurements do not establish DirectML accuracy.

## Static shapes and streaming

| Preset | Waveform | HuBERT frames | RMVPE frames | Synth frames | Skip | Audio tail at 40 kHz | Playback chunk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11200 / voice-changer | 11200 | 34 | 71 | 70 | 8 | 24800 samples | 8192 samples at 16 kHz = 512 ms |
| 44960 / WebUI / Applio, full F0 | 44960 | 140 | 282 | 281 | 250 | 12400 samples | 4000 samples at 16 kHz = 250 ms |
| 44960 / WebUI / Applio, rolling F0 | 44960 | 140 | 32, from 4960 samples | 281 | 250 | 12400 samples | 4000 samples at 16 kHz = 250 ms |

The 44960 window includes 40000 context + 4000 chunk + 800 crossfade + 160
SOLA search. The neural backend's fixed input window differs from the audio
callback size. `RealtimeBufferState` stages small callbacks into the playback
chunk. Decoder tails cover chunk + crossfade + SOLA and are validated before
starting. Reference startup silence uses `ceil(window/chunk)+1` inference
chunks: 13 for 44960, 3 for 11200. An audio format change resets streaming
buffers and joins.

`DirectMlConfig::default()`, `DirectMlConfig::realtime_44960()` and
`prepare-dml 44960` now select full-window RMVPE. Continuous male speech
diagnostics found voiced/unvoiced and octave differences with the shorter
window; see [real-time GUI repair](realtime-gui-audio-fix.md). Explicit source
manifests can still select a 4960-sample RMVPE window with GPU rolling F0.
Its 32 frames are trimmed
with `[3:-1]`, leaving 28 interior frames. A 1024-frame GPU history advances by
25 frames per 4000-sample playback chunk; the newest 281 frames feed Synth.
This matches [Applio's realtime cache update](../ref_repos/Applio/rvc/realtime/pipeline.py).
History starts at zero and fills during the existing startup warmup. It is
stored as an internal graph output, copied back to a separate GPU input after
dispatch. Production replay neither uploads nor downloads the history.
`pipeline.reset()` clears it with a reusable command list. Format changes reset
the engine history. Rolling F0 requires an integral 160-sample hop advance and
an extractor window covering that advance; the 8192-sample 11200 preset uses
full F0 to avoid cumulative fractional-hop drift.

The source manifest's `config` object selects either policy explicitly:

```json
{ "f0": { "mode": "rolling", "advance_samples": 4000, "rmvpe_samples": 4960 } }
```

Use `"f0": { "mode": "full_window" }` for the current default realtime policy
and independent-window evaluation. Earlier rolling-policy benchmarks below
remain historical measurements of that explicitly selected policy.
The window/crop are fixed at graph
build time; speaker, pitch shift, index blend/protection and valid frame count
change through `DirectMlControls` without rebuilding. `process_exact` accepts
the same latent and NSF random tensors as the Python reference.
The compiled policy is exposed read-only via `pipeline.config()`; configure
the mutable source/build configuration before compiling a replacement graph.

## Build and run

Windows x64, MSVC C++ tools, Windows SDK and a DirectML hardware adapter are
required. This backend does not link CUDA or TensorRT. SDK/runtime setup is
pinned and SHA256 verified:

```powershell
rtk proxy pwsh -NoProfile -File scripts/setup-directml.ps1
rtk cargo build -p vocal-fast --features directml
rtk cargo run -p vocal-fast --features directml -- model prepare-dml target/default-canonical.vfr models/shared/hubert_base/hubert_features.vfr models/shared/rmvpe/rmvpe.vfr target/dml-sources.json 44960
rtk cargo run -p vocal-fast --features directml -- model run-pipeline directml target/default-canonical.vfr target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav target/offline-dml.wav target/dml-sources.json
```

`DIRECTML_ROOT` optionally selects another SDK directory. The pinned default
is DirectML 1.15.4 in `target/directml-sdk/package`; DirectMLX is pinned to
commit `8700779fe7a09ea7a007cf3d7ab4293c78e41017`. Build scripts copy
`DirectML.dll` next to binaries/tests/examples, skipping unchanged files.
Debug builds also copy `DirectML.Debug.dll`; set `config.debug_layer=true`
for native validation. The Tauri crate enables `directml` by default, including
plain Cargo builds and `tauri dev`. Run SDK setup first for those commands.
The Windows build helper provisions a missing default SDK automatically;
explicit `-DirectMlRoot`/`DIRECTML_ROOT` overrides must provide a complete SDK.
Setup repairs missing DLL/import library/license files in a partial install.

The default `scripts/build-windows.ps1` distribution enables both
`tensorrt-rtx` and `directml`. `-Backend DirectMl` builds DML without the
CUDA/TensorRT SDKs, while `-Backend Cpu` disables both GPU features. Release
installers and loose executables carry `DirectML.dll`; Debug also carries
`DirectML.Debug.dll`. Both profiles include the NuGet license and third-party
notices in `licenses/directml/`. Windows provides D3D12/DXGI and GPU drivers.
The resource map uses an explicit allowlist, and post-build deployment runs
even when Cargo reuses a cached build script. Unchanged DLLs are not overwritten.

The source manifest references the Synth, HuBERT and RMVPE VFRM files and
contains a serialized `DirectMlConfig`. Paths loaded from a manifest resolve
relative to that manifest; the preparation command writes absolute paths.
VFRM uses the shared model package cache. Compiled operators and GPU resources
live with the idle/worker engine, and are released on replacement/unload.
There is currently no serialized DML plan or global compiled graph cache;
graph compilation happens on model load, outside audio processing.

Complete VFRM voice bundles can be loaded directly from the GUI or the common
offline runner, with an empty source manifest path. `DirectMlSources::from_bundle`
and `prepare_bundle` use the shared atomic acoustic extraction cache and load
the Synth directory selectively. The default storage root is
`%LOCALAPPDATA%/vocal-fast/cache/directml`, with extracted files and the generated
`directml-sources.json` under `sources/<file-identity-key>/`. Identical manifests
are preserved; incomplete ones are repaired through atomic publication. Caller
supplied manifests keep priority. A bundle with an embedded retrieval index
builds its retrieval region so the index ratio can change without recompiling.

```powershell
rtk cargo run -p vocal-fast --features directml -- model run-pipeline directml target/default-complete-with-index.vfrm target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav target/offline-dml-rolling.wav
```

Rust callers can configure the graph directly:

```rust
use vocal_directml::{DirectMlConfig, DirectMlControls, DirectMlPipeline, DirectMlSources};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut sources = DirectMlSources::load("target/dml-sources.json")?;
    sources.config = DirectMlConfig::realtime_11200();
    sources.config.precision.synth_fp32_stage_mask = 8;
    let mut pipeline = DirectMlPipeline::from_sources(&sources)?;
    let mut controls = DirectMlControls::new(pipeline.frames);
    controls.pitch_semitones = 12.;
    let output = pipeline.process(&vec![0.; 11200], &controls)?;
    println!("{} audio samples", output.audio.len());
    Ok(())
}
```

Engine callers use `EngineConfig::with_directml(descriptor, sources_path)` or
`DirectMlEngine::from_sources` with an explicit `DirectMlStreamConfig`.
Runtime/GUI backend selection is `directMl`; `directmlSourcesPath` supplies
the manifest. Loading compiles the graph while audio is stopped. Runtime
source/speaker/GPU changes replace the idle engine atomically. Automatic backend selection
prefers usable TensorRT RTX, then DirectML, then passthrough. Device discovery
reports the real DXGI adapter name. Explicit saved backend choices are retained.

The GUI's `directmlDeviceId` is independent of backend selection. `null` chooses
a compatible hardware GPU in DXGI high-performance order, with virtual/remote
adapters used only if no physical GPU is compatible. An explicit ID is applied
to complete bundles and source manifests. The Runtime passes it through
`DirectMlEngineSpec.adapter_id` or `DirectMlConfig.adapter_id`; native creation
resolves the current adapter by identity, even if its enumeration index changes.
Library/CLI callers can still use `adapter_index` when `adapter_id` is absent.
Display interface paths identify adapters; headless adapters without such a
path use their current Windows boot's LUID. Missing saved devices return a
structured error and require reselection rather than switching to another GPU.

`list_processing_backends` includes DML device IDs, names, kind, vendor, dedicated
and shared memory limits, availability/reasons and the recommended device ID.
Discovery creates temporary D3D12/DML devices to validate the same feature levels
as inference. Software adapters remain visible but unavailable. Model preparation
and graph compilation happen on Load/Save while idle; worker ownership prevents
changing the device during audio processing.

## Validation on RTX 3060

Original models: prepared HuBERT Base, original RMVPE `.pt`, default-v2 RVC
`.pth`. The independent reference uses FP32 Hugging Face HuBERT and Applio
RMVPE/Synth, with TF32 disabled and identical latent/NSF noise/crops.
The following final audio metrics include all three neural stages:

| Window / policy | Final max abs | Final RMSE | Final cosine |
| --- | ---: | ---: | ---: |
| 44960 all FP32 | 0.000136584 | 0.0000335185 | 0.999999981 |
| 44960 default mixed | 0.00725053 | 0.000701013 | 0.999990140 |
| 11200 voiced, default mixed | 0.0368715 | 0.00118857 | 0.999974169 |

At 44960, FP32 HuBERT feature RMSE is `9.47e-7`. Fixed RMVPE F0 maximum
error is approximately `0.00022–0.00031 Hz`, with no argmax differences;
11200 voiced F0 maximum error is about `0.00025 Hz`.

Production 44960 default mixed graph, 2 warmup + 30 measured iterations:
**31.41 ms mean GPU dispatch, 40.76 ms mean pipeline call** in a Cargo debug
build. GPU timestamps surround the native dispatch; the pipeline call includes
Rust random/input preparation, uploads, fence wait, audio download/decoding.
Production profiling downloads audio only. The debug graph used for numerical
diagnostics has additional outputs and is not the production timing topology.
With identical inputs/random state, production and diagnostic audio matched
exactly across all 12400 samples (`max_abs=0`, `RMSE=0`), confirming that the
audio-only graph preserves the validated result.
The 11200 mixed diagnostic graph measured about 21.71 ms GPU; it decodes 62
frames, whereas 44960 decodes only 31. These are baseline measurements on one
device/model, without clock locking, and do not establish the fastest policy.

An optional 11200 voiced stage test lowered both RMVPE encoder and decoder to
FP16 while keeping STFT, head, GRU and logits FP32. Mean GPU dispatch was
4.21 ms versus 8.41 ms for the FP32 stage diagnostic. F0 maximum error was
0.02038 Hz, RMSE 0.006359 Hz, with no argmax mismatches in this input. This
policy has not yet been validated end to end or across voices; RMVPE FP32
remains the default. Set `rmvpe_encoder` and `rmvpe_decoder` to `"fp16"` in
the precision configuration to evaluate it.

The common offline WAV runner completed 109 playback chunks over 27.1 seconds
of augmented Chinese input in about 4.73 seconds of processing after graph
construction, and wrote `target/offline-dml.wav` at 40 kHz. This uses the
streaming join/envelope policy and is a separate test from the exact-input
single-window numerical comparison.

### Streaming and retrieval optimizations

Same Chinese stream, 44960 waveform, 2 initial chunks + 30 measured calls,
default mixed policy, production graphs returning audio only:

| F0 / retrieval | Mean GPU | Mean full call, Cargo debug |
| --- | ---: | ---: |
| Full-window F0, retrieval disabled | 34.93 ms | 43.66 ms |
| Rolling 4960 F0, retrieval disabled | 24.73 ms | 31.88 ms |
| Rolling 4960 F0, sparse IVF, ratio 0.5 | 27.78 ms | 35.38 ms |

The last-window exact-randomness audio versus an independent FP32 Python
implementation with the same rolling cache policy:

| Retrieval ratio | Audio max abs | Audio RMSE | Audio cosine |
| --- | ---: | ---: | ---: |
| 0 | 0.00612822 | 0.000771172 | 0.999993823 |
| 0.5 | 0.00476837 | 0.000584968 | 0.999996441 |

Across the retained 1024-frame history, maximum F0 error was `0.0002747 Hz`.
Production and diagnostic final audio matched exactly for both retrieval ratios.
This tests reference-compatible streaming semantics; full-window F0 computes
different boundary/context predictions and is a separate reference policy.

`retrieval_search` accepts `auto`, `dense`, or `sparse`. Sparse search gathers
only the nearest IVF lists into a padded candidate bank, masks invalid entries,
then computes distances and top-8. Per-vector norms are accumulated in F64
once on load and stored as FP32 constants, eliminating another full bank pass.
On the real IVF511 / 19949-vector / nprobe=1 index, the maximum list has 354
vectors. Auto uses 354 candidates rather than 19949 per feature frame. A
256 MiB candidate-bank bound keeps auto selection from materializing very
large padded banks; it falls back to dense search for Flat indexes, large
nprobe or poorly pruning layouts. Explicit sparse requests exceeding the
bound return a configuration error.

At 140 frames, retrieval dispatch dropped from **24.48 ms dense** to
**2.33 ms sparse with cached norms**. At blend 0.5, the latter matched Python
FAISS with max error `4.77e-7`, RMSE `1.71e-8`. GPU regressions cover exact
matches, empty lists, short lists and multiple probes. Source manifests enable
retrieval explicitly; automatic complete bundles enable it when an index is
embedded. A zero ratio preserves the original features, while the compiled
retrieval region still executes.

A streaming RMVPE encoder+decoder FP16 experiment gave last-window audio cosine
`0.999981284` and RMSE `0.00128412`, but maximum retained F0 cache error rose to
`1.5856 Hz` across 32 chunks. Its 22.68 ms GPU measurement used diagnostic
outputs. RMVPE FP32 remains the default; the earlier single-stage sample did
not capture this cumulative streaming sensitivity.

The complete bundle path processed 27.1 seconds of Chinese audio using 109
chunks in about **3.73 seconds** with retrieval disabled, producing
`target/offline-dml-rolling.wav`. With sparse retrieval at ratio 0.5, it took
**3.98 seconds** and wrote `target/offline-dml-rolling-retrieval.wav` at 40 kHz.
A real runtime integration test also used 160-sample callbacks, moved the
engine into the model worker and back, verified finite/nonzero converted audio,
and unloaded the model. Construction/compilation is excluded from processing
times. Hardware clocks are not locked; these results cover one GPU/model.

### Correctness fix discovered during validation

Torch `ConvTranspose2d(output_padding=1)` extends the computed convolution
region. DirectML's `OutputPadding` appended zeros in this test. Passing the
same value silently lost the last decoder row/column, causing F0 errors up to
3 Hz and a voiced 11200 audio cosine near 0.956. Lowering now reduces ending
crop and uses zero DML output padding. A small exact GPU regression test checks
the output against Torch; decoder/logit errors returned to FP32 rounding scale.
Reports with `fixed` in their names supersede the early `target/dml-*` runs.

### Reproduce

```powershell
rtk cargo test -p vocal-directml -- --ignored --nocapture --test-threads=1
rtk cargo run -p vocal-directml --example validate -- pipeline target/dml-sources.json target/chinese-input-44960.f32 target/dml-debug.json mixed 10
rtk .venv/Scripts/python.exe scripts/compare-directml-python.py pipeline target/dml-debug.json target/chinese-input-44960.f32 target/dml-parity.json
rtk cargo run -p vocal-directml --example validate -- pipeline target/dml-sources.json target/chinese-input-44960.f32 target/dml-production.json mixed 30 nodebug
rtk cargo run -p vocal-directml --example validate -- streaming target/dml-sources-rolling.json target/dml-chinese-stream.f32 target/dml-stream-debug.json mixed 30
rtk .venv/Scripts/python.exe scripts/compare-directml-python.py streaming target/dml-stream-debug.json target/dml-chinese-stream.f32 target/dml-stream-parity.json
rtk cargo run -p vocal-directml --example validate -- retrieval target/default-canonical-with-index.vfr target/compare/retrieval-fixed/zero-debug.json target/dml-retrieval-sparse.json mixed 30 sparse
rtk cargo test -p vocal-runtime --features directml bundle_loading_worker_handoff_and_small_callbacks -- --ignored --nocapture --test-threads=1
```

The validator also supports `hubert`, `rmvpe`, `synth` and `retrieval` stage
graphs. Policy selection accepts `fp32`, `mixed`, `manifest` (pipeline), or a
precision JSON path. Large snapshots are generated under ignored `target/`.
Pipeline/streaming validators accept a final index ratio argument after
`debug` or `nodebug`, e.g. `mixed 30 nodebug 0.5`. The streaming input is a
complete mono 16 kHz F32 file; its frames are staged into consecutive windows.

## Current scope

- Native Windows x64; the real GPU tests so far use RTX 3060.
- Whole pipeline assembly supports RVC v1 (layer 9 + learned 256-feature
  projection) and v2 (layer 12, 768 features), selected from Synth metadata.
  V1 requires the projection in the HuBERT package. Unsupported contracts fail
  explicitly. See [RVC v1 support and parity](rvc-v1-support.md).
- Fixed shapes and serialized calls per device; no concurrent pipeline submit
  or DML plan serialization yet.
- Optional low precision RMVPE policies and other GPUs/models need their own
  numerical/latency evaluation. Default RMVPE remains FP32.
