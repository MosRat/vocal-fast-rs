# Aligned Applio offline conversion

`model run-pipeline` and `model test-pipeline` now use native-rate offline
conversion for DirectML and TensorRT. The previous callback/history/SOLA
runner is retained as `model run-streaming-pipeline`. CPU and passthrough
continue to use the existing common engine runner.

## Signal and model contract

- Decode input to mono 16 kHz. The supplied Mandarin fixture is already 16 kHz.
- Limit input peaks to 0.95 only when necessary, then apply the reference
  48 Hz fifth-order Butterworth forward/backward filter in F64.
- Reflect one second at both ends. Extract full-segment HuBERT and RMVPE;
  repeat actual HuBERT features twice, without the realtime appended frame.
- Use RMVPE threshold 0.03, speaker 0, no pitch shift, full flow/decoder,
  eight-neighbor IVF retrieval and the reference inverse-square weights.
- Defaults: protect 0.5 and RMS mix 1.0. These disable unvoiced feature
  protection and preserve Synth loudness respectively. They do not multiply
  output by the realtime input-RMS gain.
- Trim one second of generated context at each end, retain the model's native
  output rate, and apply a common peak scale only above 0.99. No startup
  mute, SOLA, crossfade or output 40k -> 16k -> 40k conversion is used.

Native aligned adapters currently support one Applio segment: input duration
must be below the default 41-second split threshold. Longer files explicitly
fail because faithful multi-segment inference must extract global F0 before
slicing. The shared DSP supports split selection for backends which opt into
that global-F0 contract; it does not silently substitute per-segment F0.
Non-16-kHz input still uses the existing linear input resampler. Use 16-kHz
fixtures for parity with the supplied Python runner.

The 27.10-second fixture has 433600 input samples, 465600 samples with
reflection, 1454 HuBERT frames, 2911 raw RMVPE frames and 2908 Synth frames.
Native output is 1083200 samples / 27.08 seconds at 40 kHz. This shortening
is also present in Applio because the HuBERT feature extent limits Synth.

## Graph lifetime and storage

DirectML loads canonical VFRM weights and compiles HuBERT, RMVPE, retrieval,
then frontend/Synth sequentially. Each graph is released before the next
stage is built; intermediate stage tensors cross the host boundary in this
offline adapter. Retrieval uses batches of at most 128 independent queries,
without shortening acoustic or decoder context. This resolved the monolithic
long graph's `CompileGraph` E_INVALIDARG on the local adapter. HRESULT and
available D3D12 debug errors are now preserved at the FFI boundary.

TensorRT builds fixed full-context profiles directly from VFRM, with cuFFT
STFT and Auto dispatch (no realtime tail crop). The CLI's offline workspace
default is 2 GiB: 1 GiB failed on this long Synth, while 128 MiB also failed
on HuBERT. Typed callers can explicitly supply their own workspace budget.
Plan preparation is reported separately from DSP/inference time.

Caches reside in `%LOCALAPPDATA%\vocal-fast\cache\offline`. Acoustic sources
can be materialized from a complete VFRM. TensorRT plan keys include source
paths, sizes, modification times, build options, diagnostic graph outputs,
input shape and `NVIDIA_TF32_OVERRIDE`. `pipeline.json` is published only
after all plans compile. Failed partial plans are not accepted as a complete
pipeline. One active shape is retained by an adapter; persisted per-shape
plans remain until the cache is removed. DirectML graphs are prepared again
on the next invocation; there is no serialized DirectML plan cache.

### Strict FP32 on TensorRT-RTX 1.6

On this SDK/RTX 3060, clearing `BuilderFlag::kTF32` alone did not provide the
FP32 RMVPE reference result. Building **and running** with
`NVIDIA_TF32_OVERRIDE=0` removed most of the drift. A plan built with this
variable unset cannot be reused under 0: RTX rejected its execution context.
The CLI therefore launches its offline TRT worker with 0 before CUDA starts;
it does not mutate the parent process environment. Rust embedders must start
an equivalent worker with that setting before initializing CUDA. The aligned
adapter rejects an incompatible process policy.

NVIDIA documents the [TF32 builder flag](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/python-api/infer/Core/BuilderConfig.html)
and [process override](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html).
The result above is a measured SDK behavior, rather than a claim that the
builder flag always fails on other devices/releases.

## Numerical evidence (2026-10-07)

Same source weights and controls, with native latent/NSF random arrays shared
with the **original Applio offline pipeline**. Applio still independently
performs filtering, padding, feature/F0 extraction, retrieval, Synth and crop.
No native acoustic features or F0 are supplied to that end-to-end reference.

| Comparison with Applio FP32 | DML mixed decoder | TRT corrected FP32 policy |
| --- | ---: | ---: |
| Final WAV cosine | 0.99973201 | 0.99995217 |
| Final WAV RMSE | 0.00356601 | 0.00150399 |
| Final WAV maximum absolute error | 0.70244938 | 0.24670070 |
| Final WAV 99th percentile absolute error | 0.00488209 | 0.00266517 |
| Final WAV 99.9th percentile absolute error | 0.02682704 | 0.01963757 |
| Coarse pitch mismatching frames | 0 | 0 |
| Continuous F0 RMSE (Hz) | 0.00006922 | 0.00005968 |
| Same-input Synth cosine | 0.99999417 | 0.99999997 |

Input preprocessing is sample-exact in both traces. DML's retrieval against
FAISS with identical native HuBERT inputs has RMSE 1.70e-8 and maximum error
4.77e-7. End-to-end feature differences include the FP16 HuBERT output and
IVF list selection near a centroid boundary; low average error does not mean
every transient is sample-exact. These metrics do not rank subjective sound
quality. DML uses FP16 decoder/noise stages 0..2 and FP32 stage 3/post, with
FP32 encoder/flow. The TRT artifact used here has FP32 Synth with FP16
residual blocks in stages 0..2. Both use FP16 HuBERT transformer, FP32 HuBERT
feature extractor and FP32 RMVPE.

Before the process-policy repair, TRT's final WAV cosine was 0.74660335,
RMSE 0.10937081. F0 RMSE was 0.0302408 Hz, maximum 1.20016 Hz, with five
coarse-bin differences and up to 0.117 accumulated NSF phase cycles. Its
same-input Synth cosine was already 0.99999957, localizing the discrepancy
upstream. Feeding native mel into Python reduced F0 error to 0.000739 Hz;
the main difference was in the native neural graph, not the FFT alone.
After the override, native logits RMSE against FP32 Python fell from
0.00997319 to 0.00009891.

Reports, exact binary traces and listening WAVs are in
`target/compare/offline-aligned/` and remain untracked.

With debug downloads disabled, the final 27.08-second WAV took 0.893 seconds
of DML DSP/inference and 1.529 seconds of TRT DSP/inference in a single first
call. Graph preparation was separate: 30.339 seconds DML and 33.431 seconds
TRT on the first cache miss. TRT persists its plans; DML currently recompiles.
These are offline file timings, not realtime callback latency. The running
GUI also occupied GPU resources during measurement. Production WAVs were
byte-identical to the validated PCM16 diagnostic outputs.

Validation: workspace all-features check, 136 workspace all-features tests,
Python compilation checks, and two explicitly run CUDA regressions (dynamic
protection under graph replay and unaligned reflected cuFFT tail).

## Reproduce

```powershell
rtk cargo build -p vocal-fast --features directml,tensorrt-rtx
$env:VOCAL_INDEX_RATIO = '0.5'
rtk proxy target/debug/vocal-fast.exe model run-pipeline directml `
  target/default-complete-with-index.vfrm `
  target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav `
  target/compare/offline-aligned/dml.wav target/dml-sources-rolling-with-index.json
rtk proxy target/debug/vocal-fast.exe model run-pipeline tensorrt `
  target/default-complete-with-index.vfrm `
  target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav `
  target/compare/offline-aligned/trt.wav `
  target/compare/offline-dml-trt-python/trt-cufft/pipeline.json
```

The old TRT manifest supplies precision only; long shapes are rebuilt and
cached separately. A complete bundle needs no source manifest otherwise.
Optional controls are `VOCAL_OFFLINE_PROTECT`, `VOCAL_OFFLINE_RMS_MIX` and
`VOCAL_INDEX_RATIO`. Rust uses `AlignedOfflineConfig` and `ConversionConfig`.
`VOCAL_OFFLINE_TRACE_DIR` enables parity copies; leave it unset for timing.
`run-applio-offline.py --native-noise <segment-trace> --trace-dir <reference>`
creates the paired original reference. `compare-aligned-offline.py` reports
end-to-end and same-input Synth separately; `probe-offline-rmvpe.py` isolates
native mel error from neural graph error.
