# Synth FP16 policy

The Synth generator uses a fixed profile and keeps the numerical boundary
explicit.  The production candidate is mixed precision: the high cost
upsample/deconvolution path uses FP16, while the NSF source, noise convolutions,
residual accumulation, and final projection remain in FP32.  The aggressive
full decoder mode is available for experiments only.

## Weight storage

VFRM tensors now cross the native ABI with two views:

* an F32 mirror for host graph rewrites such as QKV packing and polyphase
  kernels;
* the original F16/BF16 bits for direct TensorRT low precision constants.

When the graph type matches the stored type, Synth `Layers::w()` passes the
16-bit pointer directly to TensorRT.  This removes the previous F16 -> F32 ->
F16 build-time conversion without changing the graph's arithmetic or public
I/O.

## Fixed 44960 result

The following measurements use `target/tsukuyomi-canonical.vfr`,
`target/compare/sine-44960.vin`, a static 281-frame profile, and the RTX 3060.
The reference is the FP32 Applio/PyTorch Synth using the same VIN tensors.

| graph policy | mean (3 runs) | relative L2 | cosine |
| --- | ---: | ---: | ---: |
| native FP32 | — | 1.5976% | 0.9998725 |
| mixed FP16 (raw F16 weights) | 39.99 ms | 3.0637% | 0.9995356 |
| full decoder FP16 (raw F16 weights) | 18.99 ms | 4.7733% | 0.9989155 |

The raw-weight change produced identical samples to the previous mixed and
full FP16 plans (the comparison script reported bit-for-bit matching metrics),
so it changes loading and build memory traffic rather than model semantics.
Full FP16 is substantially faster in this fixed-shape Synth-only benchmark,
but its error is outside the default parity gate and must remain opt-in.

The existing `VOCAL_TRT_SYNTH_FUSE_RESBLOCK_SCALE=1` switch was also tested
with the mixed policy at this shape.  It measured 41.26 ms, relative L2
3.2073%, and cosine 0.9994856 versus 39.99 ms, 3.0637%, and 0.9995356
without folding.  Because low-precision weight rounding makes this transform
slightly less accurate and slower in this build, it remains opt-in rather than
being used as the default FP16 boundary fix.

## Build

```powershell
$env:VOCAL_TRT_PROFILE = "realtime_44960"
$env:VOCAL_TRT_PRECISION = "fp16"
$env:VOCAL_TRT_SYNTH_PRECISION = "fp16"
$env:VOCAL_TRT_MIN_FRAMES = "281"
$env:VOCAL_TRT_OPT_FRAMES = "281"
$env:VOCAL_TRT_MAX_FRAMES = "281"

# Accuracy-safe candidate
$env:VOCAL_TRT_SYNTH_FULL_FP16 = "0"
rtk cargo run -p vocal-fast --features tensorrt-rtx -- model compile-trt `
  target/tsukuyomi-canonical.vfr target/compare/synth-mixed-fp16-44960.plan

# Experimental aggressive candidate
$env:VOCAL_TRT_SYNTH_FULL_FP16 = "1"
rtk cargo run -p vocal-fast --features tensorrt-rtx -- model compile-trt `
  target/tsukuyomi-canonical.vfr target/compare/synth-full-fp16-44960.plan
```

`VOCAL_TRT_SYNTH_FP16_RESIDUAL_ACCUM=1` and
`VOCAL_TRT_SYNTH_FP16_SOURCE=1` are additional experimental switches.  They
are rejected unless full FP16 is selected because they remove FP32 safety
boundaries in the residual sum and NSF phase arithmetic.
