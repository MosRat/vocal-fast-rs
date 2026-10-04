# TensorRT-RTX tactic stability

TensorRT-RTX 1.6 exposes two convolution tactic sources:

* bit `0x1`: edge-mask convolutions;
* bit `0x2`: source-code JIT convolutions.

`NativeBuildOptions::tactic_sources` and the `tactic_sources` field in
`pipeline.json` now record this bitset. The environment variable
`VOCAL_TRT_TACTIC_SOURCES` accepts decimal or hexadecimal values and defaults
to `0x3` for compatibility.

## Fixed 44960 A/B

The test used the Tsukuyomi VFRM, HuBERT FP16, RMVPE FP32 + cuFFT, Synth FP16
with residual stage mask `0x2`, and the Applio 44960 waveform. The native graph
was replayed with TensorRT's CUDA Graph strategy. The synthesized waveform was
also compared with the FP32 Applio Synth using the exact native frontend
outputs, so the Synth metric does not include HuBERT or RMVPE drift.

| tactic sources | Synth-only cosine | mean host (30 replays) |
| ---: | ---: | ---: |
| `0x1` edge-mask | `0.99999932` | `44.92 ms` |
| `0x2` JIT | `0.99999932` | `45.16 ms` (separate run) |

Both newly built plans passed the Synth gate. The edge-only build was slightly
faster in this run and is the recommended reproducibility preset for the RTX
3060 until a timing cache or a per-layer tactic selector is available. The
`0x3` combination remains useful for a fresh A/B build; an older `0x3` plan in
the workspace produced a large Synth-only error while another `0x3` plan with
the same graph options passed, which shows why serialized plans must be gated
by output parity rather than trusted from their option manifest alone.

The gate must still be run per graph. Rebuilding every graph with edge-only
tactics while leaving HuBERT at FP16 produced full-chain cosine `0.96919` on
the same waveform; the Synth-only cosine stayed `0.99999932`, while HuBERT
feature drift dominated the final error. Rebuilding the same edge-only plan
with HuBERT FP32 produced full-chain cosine `0.99572`, Synth-only cosine
`0.99999935`, and a 20-replay host mean of `56.25 ms`. This is the current
parity preset. HuBERT FP16 remains the throughput preset, but it must pass the
full acoustic gate for each TensorRT build. A future refinement should allow
separate tactic-source masks for HuBERT, RMVPE, and Synth instead of applying
one global mask.

Build the edge-only preset with:

```powershell
$env:VOCAL_TRT_TACTIC_SOURCES = "0x1"
$env:VOCAL_TRT_PROFILE = "realtime_44960"
$env:VOCAL_TRT_RMVPE_FFT = "cufft"
rtk cargo run -p vocal-fast --features tensorrt-rtx -- model compile-trt-pipeline `
  target/tsukuyomi-canonical.vfr models/shared/hubert_base/hubert_features.vfr `
  models/shared/rmvpe/rmvpe.vfr target/trt-tactic-edge
```

The manifest records `tactic_sources: 1`; loading the plan does not consult an
environment variable, so an artifact keeps the policy used to build it.

## Final audio transfer

The production path now queues the final audio D2H copy into a three-slot
pinned host ring before the single pipeline stream synchronization. Set
`VOCAL_TRT_ASYNC_AUDIO_D2H=0` to run the pageable synchronous copy for an A/B
comparison. On the same fixed edge-only plan, 30 replay runs measured `44.92`
ms with the pinned async copy and `45.16` ms with the synchronous copy. The
difference is small and GPU-clock dependent, but the async path removes a
second host-side synchronization point without changing the TensorRT graph or
the returned samples.
