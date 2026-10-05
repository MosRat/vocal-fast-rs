# Real-model coherent Synth FP16 benchmark

This report compares the native VFRM Synth graph with the Applio/PyTorch FP32
Synth reference on two independent RVC v2 voice checkpoints.  The benchmark
starts at the canonical Synth inputs (`.vin`), so HuBERT and RMVPE are held
constant.  This isolates the decoder recipe being evaluated.

## Models

The checkpoints were downloaded from the public
[`wok000/vcclient_model`](https://huggingface.co/wok000/vcclient_model)
repository and converted from `.pth` to VFRM with the repository converter.
Both checkpoints have the RVC v2 F0/768-dimensional HuBERT layout, 40 kHz
upsampling, and 353 canonical tensors.

| model | checkpoint | SHA256 | VFRM |
|---|---|---|---|
| amitaro v2 40k | `amitaro_v2_40k.pth` | `b5d4fa9f3e32d4bddc54a213fc3b6aeb036f619bfb1904471199aa1ca8950bd6` | `target/compare/real-vfrm/amitaro_v2_40k.vfr` |
| kikoto kurage v2 40k | `kikoto_kurage_v2_40k.pth` | `c07b1bd31bce6503d234b384766e9acad6d692ae98815ade4a3a46c8208e1778` | `target/compare/real-vfrm/kikoto_kurage_v2_40k.vfr` |

All 353 tensors differ between these two models.  Their native mixed outputs
on the same VIN have cosine `-0.0909`, confirming that the plans are bound to
different voice weights rather than accidentally reusing one model.

The four earlier files from `fahmifauzi/rvc-malaysian-voices` have different
file hashes but their 353 canonical tensors are byte-identical; they are
therefore treated as duplicate packaging variants and are not counted as
independent voices here.

## Configuration and method

- GPU: NVIDIA GeForce RTX 3060
- Profile: fixed `realtime_44960` (`281` RVC frames, `112400` output samples)
- Synth precision: FP16
- Mixed: FP32 residual stages and FP16 decoder boundary
- Coherent: `VOCAL_TRT_SYNTH_FP16_COHERENT_STAGE_MASK=0x2`, keeping the second
  upsample/deconvolution, NSF noise convolution/add, residual convolutions,
  and residual accumulation in one FP16 region
- 30 CUDA event iterations after 10 warmups
- Reference: Applio/PyTorch FP32 Synth, same VFRM tensors and VIN

The reproducible commands are:

```powershell
rtk .venv\Scripts\python.exe scripts\convert-rvc-pth-to-vfrm.py <model-dir> <checkpoint.pth> <model.vfr>
rtk target\debug\vocal-fast.exe model compile-trt <model.vfr> <mixed.plan>
$env:VOCAL_TRT_PRECISION = "fp16"
$env:VOCAL_TRT_SYNTH_FP16_COHERENT_STAGE_MASK = "2"
rtk target\debug\vocal-fast.exe model compile-trt <model.vfr> <coherent.plan>
rtk target\debug\vocal-fast.exe model benchmark-trt <plan> target\compare\sine-44960.vin <output.f32> 30
rtk .venv\Scripts\python.exe scripts\compare-synth-vin.py --vfrm <model.vfr> --vin target\compare\sine-44960.vin --native <mixed.f32> <coherent.f32>
```

## Results

Latency is the native Synth plan enqueue plus synchronization.  The Applio
column is the same Synth operation measured with CUDA events; it excludes
audio capture and the HuBERT/RMVPE front end as well.

| model | path | median ms | p95 ms | vs Applio median |
|---|---|---:|---:|---:|
| amitaro | Applio FP32 | 70.601 | 79.867 | — |
| amitaro | native mixed | 40.845 | 54.127 | 42.2% faster |
| amitaro | native coherent `0x2` | 30.729 | 44.789 | 56.5% faster |
| kikoto kurage | Applio FP32 | 67.857 | 78.353 | — |
| kikoto kurage | native mixed | 38.430 | 48.104 | 43.4% faster |
| kikoto kurage | native coherent `0x2` | 31.375 | 37.977 | 53.8% faster |

The coherent region reduces Synth median latency by 24.8% for amitaro and
18.4% for kikoto relative to their native mixed plans.  The exact tactic
choice is profile and driver dependent, so the checked-in JSON benchmark files
under `target/compare/real-bench/` are the authoritative run for this report.

### Numerical parity

| model | path | max abs | RMSE | relative L2 | cosine |
|---|---|---:|---:|---:|---:|
| amitaro | native mixed | 2.738e-4 | 1.398e-5 | 0.1368% | 0.999999187 |
| amitaro | native coherent `0x2` | 1.074e-3 | 4.233e-5 | 0.4142% | 0.999991794 |
| kikoto kurage | native mixed | 3.185e-5 | 2.080e-6 | 0.5235% | 0.999987819 |
| kikoto kurage | native coherent `0x2` | 6.518e-5 | 2.351e-6 | 0.5918% | 0.999983326 |

The coherent recipe preserves high waveform similarity, but it increases the
Synth-only error relative to mixed FP16: roughly 3.0x relative-L2 for
amitaro and 13% for kikoto.  The mixed path is the safer default when parity
is prioritized; coherent `0x2` is the measured latency option for the fixed
44960 profile.  These are Synth-stage errors against the Applio reference,
not a claim about end-to-end waveform parity after HuBERT/RMVPE and overlap
add.

## Artifacts

- Weight and cross-model audit: `target/compare/real-model-diff.json`
- Native benchmark JSON and output: `target/compare/real-bench/`
- Applio parity JSON: `target/compare/real-parity/`
- Reproducible helpers: `scripts/prepare-real-model-manifests.py`,
  `scripts/analyze-real-models.py`, and `scripts/benchmark-applio-synth.py`

Downloaded weights, VFRM files, plans, and generated benchmark data remain
local and are ignored by Git.
