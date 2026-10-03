# Fixed 44960 realtime profile

The pipeline now accepts the explicit profile name `realtime_44960` (aliases
`applio`, `rvc_webui`, and `44960`). It builds all four plans with fixed
shapes:

| binding | shape point |
|---|---:|
| waveform | 44960 samples |
| HuBERT | 140 frames |
| RMVPE | 282 pitch frames |
| RVC/Synth | 281 frames |

Build it with:

```powershell
$env:VOCAL_TRT_PROFILE = "realtime_44960"
$env:VOCAL_TRT_DISPATCH_TUNE = "44960"
$env:VOCAL_TRT_HUBERT_PRECISION = "fp16"
$env:VOCAL_TRT_HUBERT_FEATURE_PRECISION = "fp32" # recommended mixed HuBERT path
$env:VOCAL_TRT_RMVPE_PRECISION = "fp32"
$env:VOCAL_TRT_SYNTH_PRECISION = "fp16"
rtk cargo run -p vocal-fast --features tensorrt-rtx -- model compile-trt-pipeline `
  target/tsukuyomi-canonical.vfr models/shared/hubert_base/hubert_features.vfr `
  models/shared/rmvpe/rmvpe.vfr target/trt-realtime-44960
```

For strict HuBERT parity use `VOCAL_TRT_HUBERT_PRECISION="fp32"` and leave
the feature override at `fp32`. The mixed setting above keeps the sensitive
feature extractor in FP32 while the projection and encoder use FP16.

On the RTX 3060, the fixed plan measured approximately 67.2 ms of CUDA event
stage time after warmup: HuBERT 3.50 ms, RMVPE 13.91 ms, frontend 0.01 ms and
Synth 44.65 ms. TensorRT-RTX native CUDA Graph capture/replay was active
(`native_captures=1`, `native_replays=24` in the 20-iteration run).

The 44960 boundary is handled explicitly in the frontend binding. RMVPE still
produces 282 frames, but the frontend input is bound to the first 281 frames
(`p_len = waveform_samples / 160`); both frontend outputs and Synth therefore
stay at 281 frames. This avoids a dynamic shape-control input and keeps the
TensorRT output allocation exact. A real 44960 sine input produced
`pitch_min=121`, `pitch_max=122`, and zero coarse-F0 values outside `[0,255]`;
`feats` has `281*768` elements and `pitchf`/`pitch` each have 281 elements.
The realtime-cache check also had zero out-of-range coarse-F0 values.

The hand-written HuBERT graph is now aligned with the exported graph. The fix
covered three details that are easy to miss when expanding the ONNX graph:

* `[B,C,T]` convolution output is transposed to `[B,T,C]` before feature
  projection and back to `[B,C,T]` around the positional convolution;
* the positional convolution consumes the effective `weight_norm(dim=2)`
  tensor (`g * v / ||v[:,:,k]||`), folded by the acoustic converter;
* each encoder layer adds the FFN output to the **post-LayerNorm** attention
  residual, matching `HubertEncoderLayer.forward`.

The CPU layer checker is `scripts/check-hubert-alignment.py`; every stage from
the feature extractor through encoder layer 11 is within about `6e-6` max
error against the local Applio/Transformers FP32 reference. The converter now
folds the positional weight norm over axes `(0,1)` (the previous `(1,2)` fold
was incorrect for HuBERT's `dim=2` parametrization). Regenerate the ignored
binary bundle after pulling this converter change with:

```powershell
rtk uv run python scripts/convert-acoustic-onnx-to-vfrm.py `
  --kind hubert --input models/shared/hubert_base/hubert_features.onnx `
  --output models/shared/hubert_base/hubert_features.vfr
```

## Final waveform parity

The complete-chain check uses the local Applio PyTorch HuBERT/RMVPE reference
and FP32 Applio Synth. Native `rnd` and NSF noise use the deterministic values
seeded by `RvcTrtPipeline`, so the comparison is repeatable. The check compares
the waveform before any host-side overlap, crossfade, or SOLA stage.

The previously measured `...11200-fix` and `...applio44960-fix3` results were
from stale Synth engine files. Their HuBERT, RMVPE, and frontend plan hashes
match the rebuilt artifacts, but their Synth plan differs and produces a large
Synth-only error. These rows are historical results from before the RMVPE
Intermediate expansion fix described below. Rebuilding the pipeline with the
current source gives:

| fixed profile | output samples | max abs | mean abs | RMSE | relative L2 | cosine |
|---|---:|---:|---:|---:|---:|---:|
| 11200 (`repro-11200-fixed`, HuBERT FP16) | 28000 | 0.161206 | 0.037741 | 0.049059 | 0.209497 | 0.978035 |
| 44960 (`repro-44960-fixed`, HuBERT FP16) | 112400 | 0.442166 | 0.110758 | 0.144525 | 1.023198 | 0.449208 |
| 44960 (`repro-44960-fp32hubert`, HuBERT FP32, RMVPE encoder/decoder FP16) | 112400 | 0.446610 | 0.089589 | 0.132964 | 0.941354 | 0.565874 |

The three rows above use the older `target/compare/sine-44960.f32` fixture
(peak about `0.25`) and are retained for regression history. They are not the
all-FP32 RMVPE result below. The current parity run uses
`target/sine-44960.f32` (peak `0.1`) and was rebuilt after the complete
Intermediate expansion; comparing the two fixtures or precision manifests
directly would mix separate experiments.

For those historical fixtures, the rebuilt Synth plans remain aligned
with FP32 Applio Synth: RMSE `0.000217` / cosine `0.99999969` at 11200 and
RMSE `0.000084` / cosine `0.99999986` at 44960. The corrected native HuBERT
features are within `0.000341` RMSE (11200, FP32) and `0.000372` RMSE (44960,
FP32) of Applio. The production FP16 HuBERT path is faster but accumulates a
larger feature error (`0.0597` / `0.0414` RMSE at 11200/44960); keep HuBERT
FP32 when waveform parity is the priority. Reports are stored under
`target/hubert-fp32-final`, `target/repro-11200-fixed`, and
`target/repro-44960-fixed`.

The 44960 end-to-end cosine should not be interpreted as HuBERT parity. The
isolation fields added to `compare-trt-end-to-end.py` report HuBERT-only cosine
`0.99999576`, while replacing only the RMVPE pitch produces cosine `0.56475`.
These are FP32 PyTorch Synth runs with identical random/noise inputs. Replacing
only continuous F0 gives cosine `0.570119`. The mean signed frequency offset is
`-0.083344 Hz`; NSF integrates it over 2.81 seconds into `-0.234196` cycles
(about -84.3 degrees). Four coarse-F0 frames also differ. This explains the
large sample-aligned waveform error without attributing it to HuBERT drift.

The `fp32hubert` manifest is **not an all-FP32 pipeline**: HuBERT is FP32,
RMVPE encoder/decoder are FP16 (STFT/GRU/logits FP32), and Synth uses its FP16
mixed path with FP32 residual blocks. That historical manifest was produced
before the precision parser fix, so its top-level `rmvpe_precision=fp32` did
not override the region fields. New builds with that environment setting now
select FP32 for both RMVPE U-Net regions unless a region override is present.
The pitch isolation identifies the responsible branch; it does not yet prove
how much of its F0 difference is caused by region precision versus other
implementation differences.

## RMVPE graph parity fix

The handwritten RMVPE graph originally expanded only the first of the four
Intermediate layers. Each layer contains four residual blocks, so twelve
trained blocks were missing. The native graph now expands all 16 blocks
(`138..216`), while retaining the explicit STFT, decoder BatchNorm, GRU, and
logit FP32 boundaries.

With RMVPE encoder/decoder, HuBERT, and decode tail in FP32 on the same
44960-sample sine fixture, the native RMVPE output is within `0.00527 Hz`
RMSE and `0.03262 Hz` maximum absolute error of Applio's RMVPE output; all
281 cropped coarse-F0 frames match. The resulting Applio end-to-end check has
continuous-F0 cosine `0.99999364` in the pitch-only isolation and final
waveform cosine `0.98139588` with the current FP16 Synth path. The remaining
waveform error is Synth precision/graph behavior, rather than the RMVPE F0
track.

The production (debug outputs disabled) fixed 44960 plan was also replayed
through TensorRT-RTX native CUDA Graphs: one capture and three replays, with
about `74.4 ms` total CUDA-event time after one warmup on the RTX 3060.

For stage diagnosis, compile with `VOCAL_TRT_RMVPE_DEBUG_OUTPUTS=1` and run
with `VOCAL_TRT_DEBUG_INTERMEDIATES=1`; the debug JSON then includes mel,
bottleneck, U-Net, GRU, and logits tensors. Production plans leave these
outputs out, so they do not add device writes to the realtime path.

## Current-source 11200 and 44960 measurements

The following runs were rebuilt from the current source with HuBERT extractor
FP32, HuBERT projection/encoder FP16, RMVPE FP32, Synth FP32, HuBERT QKV
fusion enabled, and RVC QKV fusion disabled. They use ten warmups and sixty
device-resident iterations with native CUDA Graph replay:

| fixed profile | HuBERT | RMVPE | frontend | Synth | event total | host mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11200 / 34 / 71 / 70 | 1.99 ms | 4.58 ms | 0.010 ms | 10.86 ms | 25.53 ms | 19.28 ms |
| 44960 / 140 / 282 / 281 | 4.64 ms | 9.55 ms | 0.013 ms | 37.77 ms | 57.29 ms | 57.86 ms |

The event total includes the device-side synchronization envelope and is the
latency value to use for GPU budgeting; the per-stage values are CUDA-event
measurements around each native graph. Both runs reported one native capture
and sixty-nine replays after warmup. The 11200 Applio comparison on the
deterministic sine fixture had final waveform RMSE `7.58e-3` and cosine
`0.999544`; the 44960 comparison had Synth-only cosine `0.999999864`, while
the complete cosine was `0.535811` because the compact handwritten DFT shifts
continuous F0 by about `0.10 Hz`.

An opt-in native decoder-FP16 RMVPE plan reduced the 44960 RMVPE stage to
`8.22 ms`, but its F0 coarse exact rate fell to `98.93%` and final cosine to
`0.48987`. It remains a diagnostic plan; production keeps RMVPE U-Net FP32.

## Hotspots and next optimizations

The steady Nsight Systems capture for the 44960 production plan measured
NVTX envelopes of approximately HuBERT `0.432 ms`, RMVPE `2.156 ms`, frontend
`0.060 ms`, and Synth `0.617 ms`; aggregate H2D activity was about `0.975 ms`.
The TensorRT layer profiler and Nsight kernel names agree on the order of
work: RMVPE fused convolutions and the 1024-point Fourier convolution first,
then Synth decoder pointwise/convolution kernels, then HuBERT positional
convolution and matrix products. Layer times are diagnostic sums and can
overlap, so CUDA events and Nsight ranges remain the latency authority.

The optimization order is therefore:

1. Replace the handwritten RMVPE DFT with one FP32 CUDA FFT/plugin operator.
   The radix-2 graph already reduces the Python float32 mel RMSE from `0.05734`
   to `0.000418`, but its expanded gather/butterfly graph has excessive first
   use JIT cost; a plugin keeps the numerical reduction order and removes that
   build/runtime overhead.
2. Time RMVPE convolution tactics at the fixed 11200 and 44960 shapes, then
   fuse Conv-BN-ReLU and residual add only when the FP32 parity gate remains
   unchanged.
3. Keep HuBERT feature extractor FP32 and the projection/encoder FP16 with
   fused QKV. For Synth, keep encoder/flow/source FP32 and test decoder or
   residual-stage FP16 per voice model; RVC QKV fusion remains opt-in because
   its long-sequence accumulation changes Synth output.
4. Preserve native CUDA Graph capture and move host audio into a pinned ring
   buffer. This removes the remaining H2D envelope without downloading any
   intermediate tensor.
