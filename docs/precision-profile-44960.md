# Precision and fusion profile

These measurements use the local Applio/Transformers FP32 models as the
reference and a deterministic sine input. They are sensitivity experiments,
not a perceptual quality score. The TensorRT timings are from the RTX 3060
with fixed 11200 or 44960 profiles and CUDA Graph replay enabled.

## HuBERT component matrix

The Python experiment is reproducible with:

```powershell
rtk .venv\Scripts\python.exe scripts\hubert-precision-applio.py `
  --audio target\compare\sine-44960.f32 `
  --output target\compare\hubert-precision-applio-44960.json
```

The output compares each region with the same FP32 `HubertModel` invocation.

| region lowered to FP16 | 44960 hidden RMSE | hidden cosine | observation |
|---|---:|---:|---|
| feature extractor | `4.094e-2` | `0.981649` | dominant error; do not use for parity |
| feature projection | `9.244e-5` | `0.99999990` | safe boundary candidate |
| encoder | `4.997e-4` | `0.99999722` | safe boundary candidate |
| all regions | `4.092e-2` | `0.981673` | extractor error propagates |

The 11200 run gives the same ordering: extractor `5.402e-2` RMSE,
projection `2.471e-4`, and encoder `6.496e-4`. The native graph now exposes
the same boundary with `VOCAL_TRT_HUBERT_FEATURE_PRECISION`; the default
follows `VOCAL_TRT_HUBERT_PRECISION`, preserving old manifests. A recommended
mixed build is:

```powershell
$env:VOCAL_TRT_HUBERT_PRECISION = "fp16"
$env:VOCAL_TRT_HUBERT_FEATURE_PRECISION = "fp32"
```

On the real 44960 TensorRT plan this reduced HuBERT hidden RMSE against
Applio FP32 from `0.33670` (all native HuBERT FP16) to `0.004384`, while the
Synth-only cosine remained `0.999999864`. The complete waveform cosine was
`0.535811` because the independent RMVPE DFT frontend still contributes the
dominant phase error. This makes the split useful for throughput experiments,
but it does not by itself make the full chain parity-safe.

## RMVPE and Synth decisions

The RMVPE Python matrix keeps STFT/mel, GRU, logits, and F0 decode in FP32.
U-Net and head FP16 produce sub-`0.004 Hz` F0 RMSE on the deterministic
fixture and preserve all coarse bins, but the native graph must pass the
full-precision F0 gate on real speech before enabling that mode by default.
The safe native setting is FP32 U-Net/head with explicit FP32 GRU, logits, and
decode boundaries.

The Python reference was split further into U-Net encoder, intermediate, and
decoder regions. Encoder FP16 was the sensitive part (`1.14e-2 Hz` F0 RMSE,
two coarse-bin mismatches), while intermediate FP16 (`2.19e-4 Hz`) and decoder
FP16 (`1.22e-3 Hz`) preserved every coarse bin. A real 44960 TensorRT plan
with encoder FP32 and decoder FP16 built after adding an explicit cast at the
decoder zero-insertion boundary. It measured RMVPE `8.22 ms` versus `9.55 ms`
for the FP32 plan, but produced F0 RMSE `0.1074 Hz`, coarse exact `98.93%`,
and final waveform cosine `0.48987` (the FP32 plan is `0.53581` on this DFT
frontend fixture). Synth-only cosine stayed `0.999999864`. This tactic and
boundary combination therefore remains opt-in; production keeps RMVPE U-Net
FP32 until a real-speech multi-voice gate passes.

Synth sensitivity is different at 44960: encoder, flow, and source changes
remain below `1e-6` waveform RMSE, while FP16 residual blocks reach
`1.63e-6` and a complete FP16 decoder reaches `2.59e-6` RMSE (`0.999289`
cosine). The fixed graph therefore keeps encoder/flow/source and residual
blocks in FP32, while the optional decoder FP16 path is guarded by a stage
mask. A mask must be validated against at least two voices before shipping.

## QKV fusion A/B

The fixed 11200 plans were rebuilt from the same VFRM weights with
`VOCAL_TRT_FUSED_QKV=1/0` and the RVC fusion switch held at its safe default
`VOCAL_TRT_FUSED_RVC_QKV=0`, then profiled for 100 iterations after ten
warmups. The fused and unfused HuBERT output fixtures are bit-identical (max
error `0`, cosine `1.0`). Timing is tactic and clock sensitive on this RTX
3060; one pair of 100-iteration captures measured:

| plan | HuBERT event | RMVPE event | Synth event | event total | host mean |
|---|---:|---:|---:|---:|---:|
| fused HuBERT QKV | `4.652 ms` | `4.573 ms` | `10.115 ms` | `21.159 ms` | `21.108 ms` |
| separate HuBERT Q/K/V | `5.519 ms` | `4.597 ms` | `10.147 ms` | `22.434 ms` | `21.329 ms` |

The difference is under one millisecond and is not stable across repeated
captures. Keep HuBERT fusion enabled as the default graph shape, while RVC
fusion remains opt-in because its long-sequence accumulation error is visible.
Both switches remain available for future timing-cache tuning; no plugin is
needed.

For the 44960 FP32 parity check, the HuBERT-only fused plan and the safe
separate plan produced identical RVC features and pitch. Enabling the RVC
fusion switch as well changed the Synth-only waveform RMSE from about
`1.27e-4` to `1.36e-3`; this is why `fused_rvc_qkv` is a separate manifest
field and defaults to false.

On the fixed 11200 sine input, both safe plans produced the same complete-chain
Applio comparison: audio RMSE `6.147e-3`, relative L2 `2.625e-2`, and cosine
`0.999679`. The remaining error is the RMVPE coarse-bin/continuous-F0
difference (six coarse frames on this fixture), not HuBERT QKV fusion.

## Profile interpretation

Nsight Systems was rerun with node-level CUDA graph tracing and five warmups
(`target/compare/nsys-steady-44960.sqlite`). After skipping the first two
capture/JIT cycles, the host NVTX envelopes were HuBERT `0.432 ms`, RMVPE
`2.156 ms`, frontend `0.060 ms`, and Synth `0.617 ms` (33 steady ranges).
These are graph enqueue/projection envelopes; CUDA event timings from the same
pipeline remain the latency authority. The steady kernel table is dominated
by fused RMVPE convolutions (the largest kernel family), then Synth pointwise
and convolution kernels, followed by HuBERT matrix products. H2D activity was
about `0.975 ms` across the captured steady ranges. The profile script now
uses `--cuda-graph-trace=node` and the summarizer reports robust median/p95
values after rejecting capture outliers.

The next optimization should target an FP32 CUDA FFT plugin or compact FFT
operator for RMVPE, then RMVPE convolution tactic timing and Synth residual
fusion. Replacing the relative RVC attention with generic `IAttention` would
change model semantics.

## 44960 fresh FP32 baseline and FFT frontend experiment

The baseline was rebuilt from the current source with HuBERT QKV fusion on,
RVC QKV fusion off, and FP32 HuBERT/RMVPE/Synth.  It uses the real
`target/compare/sine-44960.f32` fixture and the Applio FP32 path as reference:

| boundary | result |
|---|---:|
| HuBERT hidden RMSE | `3.722e-4` |
| RMVPE F0 RMSE | `0.10238 Hz` |
| RMVPE coarse exact | `99.288%` (`2/281` mismatches) |
| Synth-only waveform cosine | `0.999999884` |
| complete waveform cosine | `0.52464` |

The Synth-only gate proves that the large complete-chain error is not caused
by the current Synth graph.  The native RMVPE mel basis is bit-identical to
Applio; the remaining error is the FP32 reduction order of the handwritten
1024-point DFT convolution.  High-frequency bins whose true magnitude is near
zero retain a larger cancellation residual than cuFFT, and the residual is
amplified by `log(clamp(mel, 1e-5))`.

An experimental radix-2 FFT frontend is available with
`VOCAL_TRT_RMVPE_FFT=1`.  A Python float32 model of the same ten-stage
butterfly graph changes mel RMSE from `0.05734` to `0.000418` and F0 RMSE from
`0.10238 Hz` to `0.000687 Hz` on this fixture.  The graph compiles, but its
large gather/butterfly network currently has excessive TensorRT-RTX first-use
JIT/load cost on the RTX 3060; it remains opt-in until the frame gather and
butterfly stages are fused into a CUDA plugin or a compact FFT operator.  The
production default therefore stays on the compact DFT path, with the F0 parity
gate explicitly recorded above.
