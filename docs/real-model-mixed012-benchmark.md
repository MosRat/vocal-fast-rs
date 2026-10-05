# Default mixed012 benchmark

This report validates the default Synth precision policy on two independent
RVC voice models.  The plans were built from the local VFRM files and run on
an NVIDIA GeForce RTX 3060.

## Configuration

The fixed `realtime_44960` profile was used:

| quantity | value |
| --- | ---: |
| input VIN | `sine-44960-random.vin` |
| RVC frames | 281 |
| output samples | 112,400 |
| acoustic precision | FP32 |
| Synth precision | FP16 |
| residual FP16 stages | `0x7` (stages 0..2) |
| NSF noise FP16 | enabled for stages 0..2 |
| residual accumulation | FP32 |
| forced FP32 stage | `0x8` (stage 3) |
| warmups / iterations | 5 / 30 |

The FP32 comparison plan uses FP32 for both the acoustic graph and Synth.  The
numeric reference is the Applio/PyTorch FP32 path with the same VFRM weights and
VIN tensors.  Both native and Python paths execute the canonical encoder, flow
and Synth stages; HuBERT/RMVPE feature extraction is outside this VIN benchmark.

## Native latency and parity

| model | native default median / p95 (ms) | native FP32 median / p95 (ms) | median speedup | max abs | RMSE | relative L2 | cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Amitaro v2 40k | 21.962 / 30.542 | 39.992 / 54.337 | 1.82x | 2.246e-3 | 7.516e-5 | 0.5557% | 0.9999854 |
| Kikoto Kurage v2 40k | 22.959 / 33.197 | 40.126 / 51.500 | 1.75x | 2.418e-4 | 5.376e-6 | 0.9114% | 0.9999614 |

The default mixed plan reduces native median latency by 45.1% for Amitaro and
42.8% for Kikoto relative to their FP32 plans.  Both models remain above
0.99996 cosine similarity to the FP32 reference.  The relative-L2 metric is
more sensitive to low-energy waveform regions; max absolute error and cosine
are the primary parity gate for this run.

The Applio/PyTorch FP32 medians were 64.643 ms (Amitaro) and 64.211 ms
(Kikoto) on the same GPU and input.  Those timings also exclude HuBERT/RMVPE;
they are the Python canonical-input reference for the native graph comparison.

## Reproduction

```powershell
rtk cargo build -p vocal-fast --features tensorrt-rtx
rtk target\debug\vocal-fast.exe model compile-trt `
  target\compare\real-vfrm\amitaro_v2_40k.vfr `
  target\compare\real-bench\amitaro-default.plan
rtk target\debug\vocal-fast.exe model benchmark-trt `
  target\compare\real-bench\amitaro-default.plan `
  target\compare\sine-44960-random.vin `
  target\compare\real-bench\amitaro-default.f32 30
rtk .venv\Scripts\python.exe scripts\compare-synth-vin.py `
  --vfrm target\compare\real-vfrm\amitaro_v2_40k.vfr `
  --vin target\compare\sine-44960-random.vin `
  --native target\compare\real-bench\amitaro-default.f32
```

Set `VOCAL_TRT_SYNTH_PRECISION=fp32` and `VOCAL_TRT_PRECISION=fp32` before
building the comparison plan, and set `VOCAL_TRT_SYNTH_FP16_NOISE=0` to disable
the FP16-only default switch for that baseline.  Plans, VINs, downloaded weights and benchmark
outputs are generated files under `target/` and are intentionally not tracked.
