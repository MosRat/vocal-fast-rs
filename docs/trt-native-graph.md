# VFRM native TensorRT graph

The TensorRT backend consumes the canonical `rvc-synth-v1` tensor directory
directly. ONNX is only a build-time import format for the large HuBERT and
RMVPE source networks. The frontend and RVC Synth plans are expanded from VFRM
weights by the native C++ graph builder. Runtime deserialization and enqueue
need only the serialized plans; no ONNX file or intermediate tensor download is
required.

## Layer expansion

`native/graph/build.cpp` owns the graph entry points and optimization profile.
It declares the runtime inputs (`feats`, `p_len`, `sid`, `rnd`, and the F0
inputs), constructs the stages below, marks `audio`, and verifies that every
canonical weight was consumed.

| RVC stage | Native expansion |
| --- | --- |
| phone and pitch encoder | constant-backed matmul/gather, bias, activation, mask |
| relative attention | Q/K/V 1D convolutions, relative index gather, masked softmax, output projection |
| encoder FFN | padded Conv1d pair, activation, residual and channel normalization |
| posterior sampling | mean/log projection, exponentiation, noise injection |
| reverse flow | channel reverse, WaveNet gated blocks, speaker condition, affine reverse coupling |
| NSF source | frame phase accumulation, sine/noise mix, source projection |
| decoder | Conv1d, ConvTranspose1d upsampling, noise convolutions and residual blocks |
| waveform head | Conv1d and tanh |

`layers.h/.cpp` is the small operator vocabulary used by all stages. It owns
TensorRT layer creation, canonical weight lookup, F32 constant staging, shape
normalization and the common Conv1d/ConvTranspose1d layout conversion.

TensorRT `IAttention` is exposed as an opt-in experiment through
`NativeBuildOptions::fused_attention` (or `VOCAL_TRT_FUSED_ATTENTION=1`). It is
used only when the model has no relative key/value embedding
(`relative_window == 0 && relative_heads == 0`). RVC checkpoints from the
reference repositories use relative attention, so the default exact path is
kept for those models. TensorRT's fused operator cannot represent the dynamic
relative-key score and relative-value correction without changing the model
semantics. HuBERT has ordinary self-attention and now uses the same opt-in
operator. TensorRT-RTX may decompose it for dynamic short sequences when no
dedicated MHA tactic matches; that fallback is valid but is not necessarily
faster.

The native builders fuse the three HuBERT `q_proj`/`k_proj`/`v_proj` linear
layers into one `[768,2304]` GEMM and split the result at the attention input
boundary. Fused constants are assembled from the canonical HuBERT `[in,out]`
layout, so the runtime does not need to know whether a model came from PTH or
ONNX. HuBERT fusion is enabled by `VOCAL_TRT_FUSED_QKV` and is bit-identical
to the separate FP32 fixture in the current A/B test.

The RVC encoder has a separate optional fusion switch,
`VOCAL_TRT_FUSED_RVC_QKV`. It combines the three `Conv1d(hidden,hidden,1)`
projections into one `[3*hidden,...]` convolution before relative attention,
using the canonical `[out,in,k]` layout. Long Synth sequences amplify the
different CUDA accumulation order: the 44960 no-fusion Synth-only error was
`1.27e-4` RMSE, versus `1.36e-3` with both QKV paths fused. RVC fusion is
therefore disabled by default; enable it only after a voice/model parity gate.

## Module boundaries

The backend is split so additions do not require editing one monolithic shim.

| Module | Responsibility |
| --- | --- |
| `src/abi.rs` | checked conversion from VFRM/config tensors to the versioned `repr(C)` ABI |
| `src/ffi.rs` | raw C symbols and opaque handles only |
| `src/builder.rs` | ONNX and VFRM compilation, plan serialization, SDK checks |
| `src/runtime.rs` | RAII runtime/context, CUDA stream and device buffer wrappers |
| `src/graph.rs` | source-independent RVC graph contract and weight validation |
| `src/buffer.rs` | host weight table and device arena primitives |
| `native/common.cpp` | logger, native error allocation and exception guard |
| `native/runtime.cpp` | TensorRT runtime/context and enqueue entry points |
| `native/cuda.cpp` | stream, device memory, device-copy, and CUDA Graph entry points |
| `native/onnx.cpp` | ONNX parser and optimization profile |
| `native/serialization.cpp` | serialized engine ownership/copy/free |
| `native/plugin.cpp` | isolated future composite-stage plugin seam |
| `native/graph/*` | VFRM graph expansion and reusable layer vocabulary |

The C ABI remains versioned and pointer-based, while Rust owns all lifetime
relationships. `Context` borrows `Runtime`; native build inputs are borrowed
only for the duration of the build call; serialized plan bytes are copied before
the native build handle is released.

## Plugin boundary

The composite stages are represented as plugin-capable stages in the Rust graph
plan (`StageKind::Plugin`). TensorRT-RTX 1.6 does not expose a public
`INetworkDefinition::addPluginV2/addPluginV3` method, so the current builder
expands those stages into native layers instead of silently inserting an
identity placeholder. `RvcStagePlugin` remains an ABI-complete runtime seam in
`native/plugin.cpp`; it can be registered by a future SDK version or an
explicit plugin registry without changing the VFRM ABI.

The installed TensorRT-RTX 1.6 headers do not expose `BuilderFlag::kFP16`.
Native precision is therefore selected by explicit tensor types and constant
conversion in `hubert.cpp` and `rmvpe.cpp`. Synth keeps its encoder and flow
in FP32; selecting `synth_precision=fp16` or `bf16` enables an experimental
low-precision decoder while the public audio binding is cast back to FP32.
FP32 remains the production default because the decoder changes waveform
values more than the acoustic stages.

## ABI and ownership

`NativeConfig`, `NativeWeight` and `NativeOptions` are `repr(C)` Rust structs
mirrored by `native/bridge.h`. Rust validates the VFRM schema, decodes F16 and
BF16 tensors to owned F32 arrays, and keeps tensor names/data alive for the
duration of the native build call. TensorRT owns only the serialized plan after
the call returns.

The runtime path uses an opaque TensorRT runtime/context and a CUDA stream
wrapper. Device buffers expose bounded zero, host upload/download helpers for
smoke tests, and a coarse arena for future activation reuse.

## Smoke test

```powershell
$env:PATH = "C:\Program Files\NVIDIA Corporation\TensorRT-RTX\1.6.1.120\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;" + $env:PATH
cargo run -p vocal-fast --features tensorrt-rtx -- model compile-trt target/tsukuyomi-canonical.vfr target/tsukuyomi-native.plan
cargo run -p vocal-fast --features tensorrt-rtx -- model test-trt target/tsukuyomi-native.plan target/tsukuyomi-canonical.vfr
```

The second command sets all input addresses, uploads a valid frame length and
pitch fixture, enqueues the graph on a CUDA stream, synchronizes, downloads
the waveform buffer and rejects non-finite output values.

## Complete waveform pipeline

The source-independent Synth plan is now preceded by three TensorRT plans:

```text
waveform -> hubert.plan   -> hubert_features [1,T,768]
         -> rmvpe.plan    -> pitchf [1,P]
         -> frontend.plan -> feats [1,min(2*(T+1),P_rvc),768], pitch [1,F], pitchf [1,F]
         -> synth.plan    -> audio [1,1,F*upsample]
```

`RvcTrtPipelineArtifacts::compile` builds all four plans and writes a JSON
manifest. `RvcTrtPipeline::process` uploads the waveform once, binds all
intermediate buffers, and enqueues the four contexts on one CUDA stream. The
only synchronization is before the final audio download. The production path
accepts the converted `hubert_features.vfr` and `rmvpe.vfr` acoustic bundles;
the `.onnx` arguments remain an offline conversion/build fallback. The
waveform-input `rmvpe_20231006.onnx` is the source for the bundle; the older
`rmvpe.onnx` expects a precomputed mel tensor and is not used by this path.

## Realtime F0 alignment and CUDA Graph lifecycle

RVC WebUI/Applio append the last HuBERT frame before 2x interpolation and use
`p_len = waveform_samples / 160` as the Synth sequence length. The Rust
frontend follows that rule, so the measured windows now produce 70, 222, and
281 RVC frames for 11200, 35520, and 44960 samples. In streaming mode the
RMVPE binding is a recent 4960-sample view, producing 32 pitch frames. The
reference `pitchf[3:-1]` output is appended to a device-resident 1024-frame
cache; the frontend binds the cache tail directly and never downloads F0.
For full-window inference, RMVPE exposes one boundary frame beyond `p_len`
(71/282 versus 70/281). The frontend binds only the RVC-valid F0 prefix and
uses that execution shape for its crop, so no extra dynamic shape tensor or
out-of-bounds write is needed.
Call `process_realtime(waveform, sid, advance_frames)` for this mode and
`reset_streaming_state()` after a model/device reset.

The default graph path uses TensorRT-RTX's
`IRuntimeConfig::setCudaGraphStrategy(kWHOLE_GRAPH_CAPTURE)` on every stage.
TensorRT performs capture after shape-specialized kernels are ready and owns
replay inside `enqueueV3`; this avoids wrapping a non-capturable Myelin engine
in an external stream capture. The Rust manager tracks shape transitions as
`native_captures`, `native_replays`, and `native_invalidations` while keeping
all bindings and the realtime F0 cache at stable device addresses. A change in
waveform/profile shape, F0 advance, or streaming mode starts a new native
shape specialization. The SDK does not expose its internal graph cache
counters, so these are lifecycle counters around the TensorRT strategy.

Set `VOCAL_TRT_CUDA_GRAPH=0` to force eager execution. For an SDK/engine that
does not support the native strategy, set `VOCAL_TRT_NATIVE_CUDA_GRAPH=0`;
the Rust layer first calls `IExecutionContext::isStreamCapturable` on all four
contexts and only then enables its explicit CUDA stream graph fallback. The
CLI reports both modes under `cuda_graph`; `captures`, `replays`, and
`invalidations` are the external fallback counters, while the `native_*`
fields describe the TensorRT-managed path.

## Mixed precision and dispatch tuning

The builder carries independent precision fields for HuBERT, RMVPE, and
Synth. There are two supported operating points:

* The parity profile uses FP32 HuBERT, FP32 RMVPE, and FP32 Synth. This is the
  reference when a waveform or F0 track is being debugged.
* The throughput profile uses HuBERT FP16, FP32 RMVPE, and the Synth decoder
  FP16 path with residual stages left FP32 (`synth_resblock_stage_mask=0`).
  This avoids the large HuBERT feature drift from being mistaken for an RVC
  frontend bug while keeping the measured decoder speedup.

RMVPE U-Net/head FP16, Synth residual stage masks, and BF16 are diagnostic
options. The Python RMVPE matrix is encouraging for U-Net FP16, but the native
full-chain F0 gate on real speech is required before selecting it by default.
BF16 is slower on the test RTX 3060 and has substantially coarser F0
arithmetic. The component results and QKV A/B are recorded in
`docs/precision-profile-44960.md`.

The native RMVPE encoder and decoder fields are independent. The decoder entry
casts the FP32 bottleneck before zero-insertion/deconvolution, so an encoder
FP32/decoder FP16 diagnostic plan remains type-correct. The current native
measurement still rejects that combination for production: it saves about
`1.3 ms` at 44960 but changes three coarse F0 frames. Keep both U-Net regions
FP32 until a multi-voice gate passes; STFT, GRU, salience/logits, and decode
remain FP32 in all supported profiles.

`VOCAL_TRT_DISPATCH_TUNE=11200` or `44960` records the measured realtime
shape policy in the native ABI; the actual TensorRT tactic remains
SDK-selected, leaving a stable hook for timing-cache or dispatch additions.
The TensorRT-RTX tactic source bitset is now also recorded in the manifest;
see `docs/trt-tactic-stability.md` for the fixed 44960 A/B and the recommended
`VOCAL_TRT_TACTIC_SOURCES=0x1` reproducibility preset.

```powershell
rtk cargo run -p vocal-fast --features tensorrt-rtx -- model compile-trt-pipeline \
  target/tsukuyomi-canonical.vfr models/shared/hubert_base/hubert_features.vfr \
  models/shared/rmvpe/rmvpe.vfr target/trt-pipeline
rtk cargo run -p vocal-fast --features tensorrt-rtx -- model test-trt-pipeline \
  target/trt-pipeline/pipeline.json target/tsukuyomi-canonical.vfr target/input.f32
```

The profile is derived from the complete realtime window used by the host
application. `RvcRealtimeConfig` keeps the output chunk, extra context,
crossfade overlap, and SOLA search separate, then computes
`round_up(chunk + extra + crossfade + SOLA, 160)` for the waveform that enters
HuBERT and RMVPE. The reference defaults are:

| preset | chunk (16 kHz) | extra | crossfade | SOLA search | graph input | HuBERT / pitch / RVC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| voice-changer | 8192 | 1365 | 1365 | 192 | 11200 | 34 / 71 / 70 |
| RVC WebUI legacy config | 2080 | 32000 | 1280 | 160 | 35520 | 110 / 223 / 222 |
| RVC WebUI current GUI | 4000 | 40000 | 800 | 160 | 44960 | 140 / 282 / 281 |
| Applio realtime UI | 4000 | 40000 | 800 | 160 | 44960 | 140 / 282 / 281 |

The voice-changer row comes from its 48 kHz default callback (`192 * 128`
samples), 4096-sample overlap/context, and 12 ms SOLA search after the
16 kHz resample. WebUI and Applio expose 250 ms, 50 ms, 2.5 s, and 10 ms
defaults. The HuBERT count is the native convolution output; the Python
reference app appends one final feature frame before its 2x interpolation,
then crops to `input_samples / 160`. The native RMVPE graph exposes the
STFT-style `input_samples / 160 + 1` pitch frame count, and the native
frontend now performs the same append-and-crop operation, so its RVC count is
70, 222, and 281 at the three measured windows. This is a graph-contract
detail, not a profile rounding error.

`VOCAL_TRT_PROFILE=reference` (the default) builds the three measured points
as `min/opt/max = [11200, 35520, 44960]`, with the legacy WebUI point as the
middle tactic target. For a fixed deployment select
`voice_changer`, `rvc_webui_legacy`, `rvc_webui`, `applio`, or the explicit
`realtime_44960` alias; these use
`min=opt=max` for the selected window and give TensorRT the most useful tactic
and CUDA-graph shape. `VOCAL_TRT_PROFILE=legacy` remains available only for
old manifests (`[3200, 6400, 12800]`).

For example, compile a low-latency voice-changer plan or a WebUI/Applio plan:

```powershell
$env:VOCAL_TRT_PROFILE = "voice_changer"
rtk cargo run -p vocal-fast --features tensorrt-rtx -- model compile-trt-pipeline ...

$env:VOCAL_TRT_PROFILE = "applio"
rtk cargo run -p vocal-fast --features tensorrt-rtx -- model compile-trt-pipeline ...
```

This separation keeps the graph profile independent of the streaming joiner:

```text
inference_window = left_extra + output_chunk + right_context
graph_input      = inference_window
graph_output     = crop(graph_output, output_chunk)
stream_join      = crossfade/SOLA(previous_output, current_output)
```

The [voice-changer tutorial](https://github.com/w-okada/voice-changer/blob/master/tutorials/tutorial_rvc_en_1_5_2_5.md)
describes the same trade-off: larger context improves boundary continuity and
increases Transformer work, while a larger callback block raises latency.

For a future streaming wrapper, keep the concepts separate:

```text
inference_window = left_extra + output_chunk + right_context
graph_input      = inference_window
graph_output     = crop(graph_output, output_chunk)
stream_join      = crossfade/SOLA(previous_output, current_output)
```

The wrapper would choose `output_chunk`, `left_extra`/`right_context`,
`crossfade_samples`, and `sola_search_samples`; those values must not be
silently inferred from the TensorRT `opt_samples` point. The existing CPU
engine has a small 1,600-sample history buffer for continuity, but that is a
separate host-side mechanism and is not included in the graph input profile.

The checked native build used these self-contained bundles and produced
`target/trt-pipeline-final/{hubert,rmvpe,frontend,synth}.plan`:

```powershell
rtk cargo run -p vocal-fast --features tensorrt-rtx -- model compile-trt-pipeline `
  target/tsukuyomi-canonical.vfr models/shared/hubert_base/hubert_features.vfr `
  models/shared/rmvpe/rmvpe.vfr target/trt-pipeline-final
```

## Precision and profiling

The native acoustic builders use strongly typed TensorRT networks. `fp32` keeps
all internal HuBERT/RMVPE tensors in FP32; `fp16` and `bf16` convert constants
and internal tensors at the graph boundary while retaining FP32 public I/O.
Synth keeps encoder/flow in FP32 and can opt into a low-precision decoder for
experiments; its default remains FP32. Lengths, speaker ids, pitch class
indices, and the final audio binding stay FP32/INT32. INT8 should be
introduced selectively with calibration data; LayerNorm, attention softmax,
RMVPE salience decode, and the NSF source are the first accuracy-sensitive
regions.

The pipeline uses CUDA events around every enqueue, so reports distinguish GPU
graph time from host upload and final download time. Each enqueue is annotated
with an NVTX range (`rvc.hubert`, `rvc.rmvpe`, `rvc.frontend`, `rvc.synth`), and
host copies are marked `rvc.h2d`/`rvc.d2h`. The NVTX ranges surround the host
`enqueueV3` calls and therefore show enqueue overhead; use the CUDA-event stage
section of `scripts/summarize_nsys.py` for device duration of native graph
replay. Use the profile command for a warmup plus repeated, device-resident
run:

```powershell
rtk powershell -NoProfile -File scripts/profile-trt-nsight.ps1 `
  -Pipeline target/trt-pipeline/pipeline.json `
  -Model target/tsukuyomi-canonical.vfr -Audio target/input.f32
```

`VOCAL_TRT_WARMUP` and `VOCAL_TRT_PIPELINE_ITERATIONS` override the defaults
(10 and 100 for the profile command). Nsight Compute can then attach to the
same executable with `ncu --set full` to inspect the kernels selected by each
plan. `VOCAL_TRT_PARALLEL_ACOUSTIC=1` is an experimental native-graph mode
that launches HuBERT and RMVPE/cuFFT on separate streams before the frontend;
it is disabled by default because the two graphs can contend for memory
bandwidth on smaller GPUs.
plan. The TensorRT layer profiler remains an opt-in diagnostic
(`VOCAL_TRT_LAYER_PROFILE=1`) and is not the primary profile path.

Intermediate tensors are downloaded only when
`VOCAL_TRT_DEBUG_INTERMEDIATES=1` is set. That mode is intended for
`scripts/compare-trt-pipeline.py` numerical parity checks; it is disabled by
default and never used by the Nsight profile command.

The reference streaming implementations do not all feed the same audio span
to every stage. [RVC WebUI's realtime worker](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/RVCRealtimeVST/worker/rvc_worker.py)
and [Applio's realtime pipeline](https://github.com/IAHispano/Applio/blob/main/rvc/realtime/pipeline.py) extract
F0 from only the newest block plus roughly 800 samples (the 250 ms defaults
round this to about 4960 samples at 16 kHz), then update a full-window pitch
cache. [voice-changer RVCr2](https://github.com/w-okada/voice-changer/blob/master/server/voice_changer/RVC/RVCr2.py)
passes its complete converted window to its pitch extractor. The native
`process_realtime` path uses the recent 4960-sample RMVPE view and updates the
device-resident F0 cache; `process` remains the full-window parity path.

On the RTX 3060, the current fixed profiles were smoke-tested without debug
downloads. A 11200-sample FP16-acoustic/FP32-Synth run measured about 24 ms of
CUDA-event stage time after warmup (HuBERT 1.6 ms, RMVPE 5--7 ms, frontend
below 0.1 ms, Synth about 10--14 ms; clock state affects the last two values).
A 44960-sample run measured about 83 ms of stage time (HuBERT 3.4 ms, RMVPE
13.5 ms, frontend 0.09 ms, Synth 47 ms). Nsight Systems captures are stored as
`target/nsys-rvc-11200.nsys-rep` and `target/nsys-rvc-44960.nsys-rep`; their
kernel summaries put RMVPE convolution/GRU work and Synth convolution work at
the top, with HuBERT MatMul/attention groups next. The captures ran with
`VOCAL_TRT_DEBUG_INTERMEDIATES=0`, so the D2H traffic is only the final audio
and the diagnostic scalar bindings.

The fixed-shape 11200 A/B run used 20 measured iterations after five warmups:

| build | HuBERT | RMVPE | frontend | Synth | reported total |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 acoustic + FP32 Synth | 1.55 ms | 8.15 ms | 0.05 ms | 10.48 ms | 24.99 ms |
| FP16 acoustic + FP16 Synth decoder | 1.58 ms | 8.20 ms | 0.05 ms | 7.71 ms | 21.74 ms |
| BF16 acoustic + FP32 Synth | 1.94 ms | 8.50 ms | 0.10 ms | 13.50 ms | 33.13 ms |

The FP16 Synth row is an experimental speed path: on the same input its
waveform differs from the FP32 Synth plan by relative L2 `0.243`, so it is not
selected by default. BF16 is slower on this GPU and remains a diagnostic mode.

The refreshed 11200-sample plans are recorded in
`target/compare/h16-r32-synth-ab-11200.json`. They use HuBERT FP16, RMVPE FP32,
the same deterministic frontend outputs, and the same latent/NSF noise for both
Synth plans. Relative to the Applio Python Synthesizer in FP32 (the external
reference), the native outputs were:

| native plan | mean total | Synth stage | waveform max abs | waveform RMSE | relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 acoustic + FP32 Synth | 24.752 ms | 10.376 ms | 0.425591 | 0.176026 | 1.33757 |
| FP16 acoustic + FP16 Synth decoder | 18.511 ms | 4.785 ms | 0.423131 | 0.160478 | 1.21942 |

The native FP16 decoder versus native FP32 decoder difference is max abs
`0.360170`, RMSE `0.074108`, relative L2 `0.423428`. For comparison, the
separate CUDA PyTorch experiment with the Applio decoder alone in FP16 gave max
abs `8.40e-4`, RMSE `1.63e-4`, relative L2 `1.20e-3` against Applio FP32.
Therefore the large native-to-Applio gap is currently dominated by graph parity
(operator/layout/accumulation differences), rather than ordinary FP16 rounding;
the native Synth graph needs parity work before using FP16 error as a production
quality decision.

The runtime seeds the deterministic baseline latent and NSF-noise buffers once
at pipeline load. Each dynamic shape consumes a prefix of those stable device
allocations, so CUDA Graph replay no longer performs per-block host generation
or latent/noise uploads. Waveform upload and the final audio download remain
the recurring host/device transfers in the production path.

The first Nsight capture on the RTX 3060 with a 6400-sample window showed the
same ordering as the CUDA events: RMVPE and Synth dominate, followed by
HuBERT, while the frontend is negligible. The capture used ten warmups and one
hundred device-resident iterations with
`VOCAL_TRT_DEBUG_INTERMEDIATES=0`; no intermediate tensor was downloaded.
Across the last three iterations Nsight reported HuBERT 2.33 ms, RMVPE 17.31
ms, frontend 0.056 ms, and Synth 4.19 ms. The first iteration still includes
TensorRT-RTX `myelinGraphResolveShapes`/`myelinGraphLoadPersistent` work. The
pipeline caches the shape key; CUDA Graph capture after the first warmup is the
next setup reduction.

The earlier CUDA-event smoke profile on the old 6400-sample window (five
warmups, twenty measured iterations) is:

| plan precision | mean end-to-end | HuBERT | RMVPE | frontend | Synth |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 26.32 ms | 7.07 ms | 5.47 ms | 0.008 ms | 8.45 ms |
| FP16 | 20.29 ms | 1.58 ms | 3.49 ms | 0.027 ms | 10.70 ms |
| BF16 | 24.00 ms | 1.71 ms | 6.23 ms | 0.071 ms | 10.70 ms |

These are warm single-stream measurements, not a realtime guarantee. The
reported end-to-end number includes host-side enqueue overhead; the stage
columns are CUDA event timings. The layer report and Nsight kernel aggregation
identify RMVPE GRU unroll and U-Net convolutions, HuBERT MatMul/attention
groups, and Synth `dec.conv_pre` plus residual convolutions as the main
arithmetic hotspots. HuBERT QKV projection fusion is part of the default graph;
RVC encoder QKV fusion remains opt-in because of long-sequence accumulation
error. The next optimization order is CUDA Graph capture, testing dedicated
`IAttentionV2` tactics at fixed profiles, GRU unroll reduction, then selective
INT8 calibration.

With the old synthetic profile, the same pipeline was measured at
three power-of-two points (three warmups, ten measured iterations, RTX 3060,
FP32):

| input samples | HuBERT frames | pitch/RVC frames | mean total | stage sum (CUDA events) |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 12 | 26 / 24 | 14.89 ms | 13.33 ms |
| 8192 | 25 | 52 / 50 | 25.70 ms | 22.49 ms |
| 12288 | 38 | 77 / 76 | 32.92 ms | 31.96 ms |

Those numbers are retained as a historical baseline only. They do not describe
the application windows in the table above: the current fixed voice-changer
preset is 11200 samples, and the current WebUI/Applio preset is 44960 samples.
Re-run the Nsight command for each selected preset before making a tactic or
quantization decision; stage values above are CUDA-event summaries and include
no intermediate tensor downloads.

HuBERT `IAttentionV2` A/B on the same 6400-sample window (FP16, five warmups,
twenty iterations) measured 20.29 ms for the exact MatMul graph and 20.86 ms
with the opt-in attention path. The latter reduced the measured Synth stage in
that build, but HuBERT remained slightly slower and the dynamic short sequence
was decomposed instead of using a dedicated MHA kernel. Its HuBERT RMSE was
`0.424` versus `0.408` for the exact FP16 graph. Keep the option experimental;
fixed longer chunks or a TensorRT release with a matching dedicated MHA tactic
may change this result.

For the 6400-sample sine parity fixture, the acoustic F0 errors against the
Applio RMVPE predictor were:

| precision | max absolute error | RMSE | relative L2 |
| --- | ---: | ---: | ---: |
| FP32 | 0.100 Hz | 0.047 Hz | 1.08e-4 |
| FP16 | 2.70 Hz | 1.25 Hz | 2.85e-3 |
| BF16 | 20.95 Hz | 5.59 Hz | 1.27e-2 |

FP32 remains the default. FP16 is an optional speed mode whose F0 error is
visible but bounded on this fixture. BF16 is exposed for hardware experiments;
its coarse salience/decode quantization currently causes unacceptable F0 error
for production use.

The complete FP32 fixture comparison is saved in
`target/trt-pipeline-final/parity.json`; it reports HuBERT max/RMSE
`2.613/0.409`, RMVPE `0.098/0.047` Hz, and Synth waveform max/RMSE
`5.46e-4/1.19e-4` against the ONNX/PyTorch reference stages.

The fused-QKV voice-changer build is saved in
`target/trt-pipeline-qkv/parity.json`; the same fixture reports HuBERT
`2.613/0.409`, RMVPE `0.098/0.047` Hz, and Synth
`6.89e-4/1.37e-4` with relative L2 `8.94e-4`. The small Synth increase is
the expected convolution accumulation-order change, not a tensor-layout
conversion.

## Quantization experiment

TensorRT-RTX 1.6 exposes explicit `IQuantizeLayer`/`IDequantizeLayer`, but it
does not expose a global INT8 builder flag or calibration API in this SDK. A
static weight-only experiment is therefore available through
`VOCAL_TRT_WEIGHT_INT8=1`. It quantizes eligible Synth convolution/matrix
weights symmetrically to INT8 with per-output-channel scales and inserts
build-time DQ nodes. The acoustic plans stay on their selected FP32/FP16/BF16
path for this experiment. Synth biases, normalization affine parameters,
lookup tables, activations, softmax, flow state, and NSF/F0-sensitive math
remain floating point. The generated manifest records this as
`weight_int8: true`.

To reproduce the build on Windows:

```powershell
$env:VOCAL_TRT_WEIGHT_INT8 = "1"
rtk target/debug/vocal-fast.exe model compile-trt-pipeline `
  target/tsukuyomi-canonical.vfr models/shared/hubert_base/hubert_features.vfr `
  models/shared/rmvpe/rmvpe.vfr target/trt-pipeline-int8w
```

The experiment was built from the same canonical VFRM and tested on the
6400-sample sine fixture with five warmups and twenty measured iterations:

| build | mean end-to-end | HuBERT | RMVPE | frontend | Synth | Synth RMSE | relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 26.32 ms | 7.07 ms | 5.47 ms | 0.008 ms | 8.45 ms | 1.24e-4 | 8.06e-4 |
| INT8 weight Q/DQ | 26.48 ms | 4.72 ms | 7.14 ms | 0.046 ms | 9.43 ms | 2.01e-2 | 1.31e-1 |

The INT8 plan runs correctly, but is slower because this Q/DQ form is not
being selected as a native INT8 convolution tactic on the current RTX/SDK
combination. The weight-only plan does reduce the Synth plan from about
123 MB to 45 MB, but it does not improve latency. Its Synth error is also too
large for a voice conversion path. The result is kept as a diagnostic baseline
rather than the default. A useful production INT8 path needs activation
calibration and native INT8 convolution selection, followed by selective
exclusions for flow, NSF, normalization and F0-sensitive regions. The current
implementation deliberately does not claim an INT8 speedup.

The reproducible diagnostic artifacts are under `target/trt-pipeline-int8w/`,
including `pipeline.json`, plans, `trt-debug.json`, and `parity.json`.

The same comparison was also run after rebuilding the acoustic plans at lower
precision.  HuBERT's error is essentially unchanged; RMVPE salience decoding
is the precision-sensitive stage:

| precision | HuBERT RMSE | RMVPE max/RMSE (Hz) | Synth max/RMSE | Synth relative L2 |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 0.409 | 0.098 / 0.047 | 5.46e-4 / 1.19e-4 | 7.75e-4 |
| FP16 | 0.408 | 2.70 / 1.25 | 7.78e-4 / 1.90e-4 | 1.10e-3 |
| BF16 | 0.417 | 20.95 / 5.59 | 7.93e-4 / 1.39e-4 | 7.14e-4 |

The FP16 and BF16 reports are stored beside their plans as
`target/trt-pipeline-final-fp16/parity.json` and
`target/trt-pipeline-final-bf16/parity.json`.
