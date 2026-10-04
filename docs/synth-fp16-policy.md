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

## FP32 to FP16 input scaling

`VOCAL_TRT_SYNTH_FP16_INPUT_SCALE` inserts a positive scale immediately before
the first decoder boundary and folds its reciprocal into `dec.ups.0.weight`.
This leaves the bias and later residual arithmetic untouched.  The first
`decoder_pre` activation reached about `85` on the static fixture, while the
first upsample output was below `4.5`, so this is the only boundary where such
a scale is potentially useful.

The sweep did not show a universal win:

| shape | scale | mean | relative L2 | cosine |
| ---: | ---: | ---: | ---: | ---: |
| 11200 | `1` | 11.73 ms | 0.14390% | 0.99999923 |
| 11200 | `1/256` | 11.93 ms | 0.14461% | 0.99999922 |
| 44960 | `1` | 37.55 ms | 3.06368% | 0.99953558 |
| 44960 | `1/256` | 36.79 ms | 3.05918% | 0.99953723 |

The scale is therefore an opt-in calibration hook rather than a default.  A
voice-specific calibration should check the largest activation and the scaled
kernel range before selecting it; the reciprocal folded weights must remain
well inside FP16 range.

## Selective residual blocks

The residual stage masks can remove part of the FP32 round-trip traffic, but
they are voice and shape dependent.  On the same fixed 44960 fixture:

| policy | mean | relative L2 | cosine |
| --- | ---: | ---: | ---: |
| mixed baseline, all residual FP32 | 39.99 ms | 3.0637% | 0.9995356 |
| residual mask `0x4`, low accumulation | 36.90 ms | 3.1598% | 0.9995055 |
| residual mask `0x8`, low accumulation | 37.18 ms | 4.5340% | 0.9990005 |
| Conv1/Conv2 mask `0x4`, FP32 accumulation | 36.04 ms | 3.1490% | 0.9995094 |

Bit `0x4` is the third decoder stage for this four-stage Synth; `0x8` is the
final, highest-rate stage.  The final stage is much more sensitive because its
error is applied directly at the waveform sample rate.  These options remain
experimental and should be gated against several real voice models before
deployment.

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

## Noise convolution boundary

`VOCAL_TRT_SYNTH_FP16_NOISE=1` is a narrower experiment.  The NSF phase and
source construction remain FP32, then one FP16 copy is kept for the learned
`dec.noise_convs.*` layers.  Their convolution and add therefore stay in the
decoder domain instead of doing an FP16 to FP32 to FP16 round trip at every
upsample stage.  Residual accumulation and `conv_post` remain on the normal
mixed-precision path.  The switch is disabled by default until it passes
several voice models, since the last noise convolution directly affects the
waveform-rate output.

For a safer partial experiment use
`VOCAL_TRT_SYNTH_FP16_NOISE_STAGE_MASK=0x1`, `0x2`, `0x4`, or `0x8`.  The mask
keeps only selected upsample stages in FP16 and leaves the other noise
convolutions on the FP32 path.  This is useful when the highest-rate stage is
the accuracy limiter; the boolean switch is equivalent to selecting all
stages.

On the deterministic 44960 VIN fixture, the stage sweep was:

| noise mask | Synth mean | relative L2 | cosine |
| ---: | ---: | ---: | ---: |
| `0` | 37.62 ms | 3.0643% | 0.9995350 |
| `0x2` | 39.34 ms | 3.0586% | 0.9995377 |

The earlier tactic build of the same mask measured about 37.15 ms, showing
that TensorRT-RTX tactic selection is larger than this small graph change on
the test GPU.  The mask is therefore useful as a profile candidate, not a
guaranteed speedup.  It remains opt-in and should be selected only after a
steady Nsight/CUDA-event replay on the target GPU.

## Hot residual convolution stage

The largest residual MAC block is upsample stage 1 (bit `0x2`, C=128).  The
existing per-operation precision controls can place only its `convs1` and
`convs2` in FP16 while keeping every residual add in FP32:

```powershell
$env:VOCAL_TRT_SYNTH_RESBLOCK_PRECISION = "fp32"
$env:VOCAL_TRT_SYNTH_RESBLOCK_STAGE_MASK = "0"
$env:VOCAL_TRT_SYNTH_RESBLOCK_CONV1_PRECISION = "fp16"
$env:VOCAL_TRT_SYNTH_RESBLOCK_CONV1_STAGE_MASK = "0x2"
$env:VOCAL_TRT_SYNTH_RESBLOCK_CONV2_PRECISION = "fp16"
$env:VOCAL_TRT_SYNTH_RESBLOCK_CONV2_STAGE_MASK = "0x2"
```

Measured Synth-only medians on the RTX 3060 fixture were:

| profile | mixed | hot residual FP16 | relative L2 | cosine |
| ---: | ---: | ---: | ---: | ---: |
| 11200 | 11.76 ms | 10.20 ms | 0.3017% | 0.9999958 |
| 44960 | 37.69 ms | 29.36 ms | 3.0559% | 0.9995358 |

This is a substantially larger opportunity than moving all residual adds to
FP16.  It is still an opt-in voice-model candidate: the 11200 error roughly
doubles in relative L2 even though cosine remains high, and TensorRT tactic
selection can move absolute timings between builds.  Combining hot residual
FP16 with noise mask `0x2` measured 28.89 ms at 44960 with cosine `0.9995346`,
so the noise mask should only be added when a steady replay proves a benefit.

For a shorter 11200 callback, lowering only one operation family is a more
conservative point.  The current fixture measured:

| policy | median | relative L2 | cosine |
| --- | ---: | ---: | ---: |
| Conv1 only, stage `0x2` | 11.11 ms | 0.2348% | 0.9999975 |
| Conv2 only, stage `0x2` | 10.14 ms | 0.2416% | 0.9999973 |
| Conv1 + Conv2, stage `0x2` | 10.20 ms | 0.3017% | 0.9999958 |

The Conv2-only setting is the current short-window candidate. These timings
come from separate TensorRT builds and should be confirmed with a steady
CUDA-event replay before selecting a deployment profile.

`VOCAL_TRT_SYNTH_FP16_POST=1` is available for isolating the final projection.
It casts the `conv_post` result to FP32 before tanh. On the native 44960 plan
it measured `36.60 ms`, relative L2 `3.1618%`, and cosine `0.9995051` versus
the mixed baseline `3.0643%` and `0.9995350`. The projection is too small to
offset its accuracy cost, so keep it disabled.
