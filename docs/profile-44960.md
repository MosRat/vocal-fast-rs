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
$env:VOCAL_TRT_RMVPE_PRECISION = "fp32"
$env:VOCAL_TRT_SYNTH_PRECISION = "fp16"
rtk cargo run -p vocal-fast --features tensorrt-rtx -- model compile-trt-pipeline `
  target/tsukuyomi-canonical.vfr models/shared/hubert_base/hubert_features.vfr `
  models/shared/rmvpe/rmvpe.vfr target/trt-realtime-44960
```

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

The frontend fix does not resolve a separate HuBERT numerical alignment issue:
the current hand-built native HuBERT output differs from the ONNX/PyTorch
reference even in an FP32 A/B (max absolute error about `3.109`, RMSE about
`0.410`). RMVPE's raw F0 error is much smaller, but the final waveform is still
dominated by the HuBERT drift (and the current 44960 Synth plan has an
additional Synth-only mismatch); HuBERT needs a layer-level comparison before
feature parity can be used as an accuracy gate.

## Final waveform parity

The complete-chain check uses the local Applio PyTorch HuBERT/RMVPE reference
and FP32 Applio Synth. Native `rnd` and NSF noise use the deterministic values
seeded by `RvcTrtPipeline`, so the comparison is repeatable. The check compares
the waveform before any host-side overlap, crossfade, or SOLA stage.

| fixed profile | output samples | max abs | mean abs | RMSE | relative L2 | cosine |
|---|---:|---:|---:|---:|---:|---:|
| 11200 (`...11200-fix`) | 28000 | 0.754512 | 0.252605 | 0.299502 | 1.278981 | -0.151875 |
| 44960 (`...applio44960-fix3`) | 112400 | 0.643922 | 0.197209 | 0.246232 | 1.743264 | 0.122972 |

These are end-to-end values, not the earlier Synth-only values. The 11200
native-input Synth check is `RMSE=0.164391`, while the 44960 `fix3` plan is
currently `RMSE=0.278686`; therefore the 44960 plan also needs a Synth graph
parity pass after the HuBERT fix. Reports are stored at
`target/trt-h16-r32-s16-11200-fix/end-to-end-applio.json` and
`target/trt-h16-r32-s16-applio44960-fix3/end-to-end-applio.json`.
