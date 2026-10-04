# Synth fixed-profile optimization results

This report records the four opt-in Synth changes measured on an RTX 3060 with
TensorRT-RTX 1.6.1.120 and CUDA 13.4. The numerical reference is the FP32
Applio Synthesizer, and the end-to-end reference includes Applio HuBERT and
RMVPE. All runs use the same 44,960-sample input and the fixed
`realtime_44960` profile:

| Tensor | Shape |
| --- | ---: |
| waveform | 44,960 |
| HuBERT frames | 140 |
| RMVPE frames | 282 |
| RVC/Synth frames | 281 |

The steady latency numbers below are from 10 warmups and 100 replays. Nsight
Systems was run with `--cuda-graph-trace=node`; its NVTX ranges show graph
enqueue overhead, so CUDA event timings and kernel aggregates are the latency
authority.

## Results

| Plan | Synth setting | Mean pipeline event (ms) | Synth event (ms) | Final cosine | Synth-only cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | all FP32 | 71.05 | 42.41 | 0.999562 | 0.999868 |
| scale-fold | residual `1/3` folded into following kernels | 70.73 | 42.19 | 0.999348 | 0.999734 |
| op-mixed | Synth FP16; residual accumulation FP32; `convs1` FP16 on stage mask `0x2`; `convs2` FP32 | **51.65** | **32.27** | 0.999487 | 0.999796 |
| polyphase | phase-split Conv1d replacement for ConvTranspose1d | 71.31 | 42.77 | 0.999560 | 0.999868 |

The op-level mixed plan is the useful performance result: relative to the
existing HuBERT-FP16/RMVPE-FP32/Synth-FP16 plan, its Synth event fell from
36.20 ms to 32.27 ms and its pipeline mean fell from 55.65 ms to 51.65 ms.
The final cosine remained 0.999487, within the existing mixed-precision
baseline (0.999492). It remains opt-in because the best stage mask can depend
on the voice model.

Scale folding removes the explicit residual scale nodes, but TensorRT-RTX
selects slightly different convolution tactics. The result is numerically
close, but the measured gain is within run-to-run noise and the final cosine
is lower than the unfused baseline. It is therefore not enabled by default.

The polyphase graph is now numerically correct for the canonical `16/10` and
`4/2` upsamplers. It uses phase-specific asymmetric pre/post padding and
preserves the output length exactly. Nsight recorded 215,930 kernels and
7,370.91 ms of aggregate kernel time, compared with 213,730 and 7,303.58 ms
for the baseline. The extra phase convolutions do not provide a speedup, so
this path is a correctness/experimentation switch rather than the default
implementation.

## Nsight evidence

The SQLite exports are:

- `target/compare/nsys-opt-baseline-44960.sqlite`
- `target/compare/nsys-opt-scale-44960.sqlite`
- `target/compare/nsys-opt-op16-s2-44960.sqlite`
- `target/compare/nsys-opt-poly-44960.sqlite`

Kernel aggregates from those captures:

| Capture | Kernel launches | Aggregate kernel time (ms) |
| --- | ---: | ---: |
| baseline | 213,730 | 7,303.58 |
| scale-fold | 213,730 | 7,278.54 |
| op-mixed | 208,450 | 5,205.88 |
| polyphase | 215,930 | 7,370.91 |

The op-mixed capture shows Tensor Core convolution variants for the selected
stage and fewer pointwise/convert nodes. The scale-fold capture keeps the same
launch count; its residual elementwise operations are already fused into
TensorRT pointwise kernels, so removing the explicit scale is not a material
optimization on this SDK.

## Build controls

All controls default to the previous safe path:

```powershell
$env:VOCAL_TRT_SYNTH_RESBLOCK_CONV1_PRECISION = "fp16"
$env:VOCAL_TRT_SYNTH_RESBLOCK_CONV1_STAGE_MASK = "0x2"
$env:VOCAL_TRT_SYNTH_RESBLOCK_CONV2_PRECISION = "fp32"
$env:VOCAL_TRT_SYNTH_RESBLOCK_CONV2_STAGE_MASK = "0x2"
$env:VOCAL_TRT_SYNTH_FUSE_RESBLOCK_SCALE = "0"
$env:VOCAL_TRT_SYNTH_UPSAMPLE_IMPL = "default"
```

`VOCAL_TRT_SYNTH_UPSAMPLE_IMPL=polyphase` enables the phase-split graph.
The builder checks the stage masks against the number of upsample stages and
checks `kernel >= stride` plus the symmetric output-length condition
`(kernel - stride) % 2 == 0`.

## Timing-cache contract

`VOCAL_TRT_TIMING_CACHE=<path>` now loads a cache before building and writes it
back atomically after the build. The RTX 1.6.1.120 headers mark `ITimingCache`
deprecated and document its operations as no-ops. On this SDK every measured
cache file is the same 172-byte serialized header, and no tactic persistence or
build-time improvement was observed. The file contract is retained for
forward compatibility with a runtime that implements timing-cache data; it is
not reported as an optimization on the current SDK.

