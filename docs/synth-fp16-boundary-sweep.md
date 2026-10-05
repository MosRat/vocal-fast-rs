# Python FP16 boundary sweep

`scripts/sweep-synth-fp16-boundaries.py` runs the Applio/PyTorch Synthesizer
with one decoder operation family at a time under CUDA FP16 autocast.  The
reference is the same VFRM checkpoint and VIN latent/pitch/noise input decoded
entirely in FP32.  This isolates numerical sensitivity before changing the
TensorRT graph.

The report now records the reference RMS, error-to-reference RMS, SNR, p99
absolute error, activation ranges, and the fraction of values that would
overflow or become FP16 subnormals.  Use `--modes` for a short candidate sweep
on another voice model:

```powershell
rtk .venv-cuda\Scripts\python.exe scripts\sweep-synth-fp16-boundaries.py `
  --vfrm target\compare\real-vfrm\kikoto_kurage_v2_40k.vfr `
  --vin target\compare\sine-44960.vin `
  --modes fp32,native_mixed,native_mixed_keep_last,native_mixed_keep_final `
  --output target\compare\kikoto-boundary-sweep.json
```

## What the sweep found

Measurements below are on an RTX 3060 with the canonical Tsukuyomi VFRM
checkpoint.  The isolated rows change only the named operation family; they do
not claim that all decoder stages can be enabled together.

| mode | 11200 relative L2 | 44960 relative L2 | 44960 SNR |
| --- | ---: | ---: | ---: |
| `source16` | 0.0503% | 0.1706% | 55.4 dB |
| `up1` | 0.0565% | 0.2156% | 53.3 dB |
| `up2` | 0.0472% | 0.2238% | 53.0 dB |
| `noise1` | 0.0318% | 0.1585% | 56.0 dB |
| `noise2` | 0.0135% | 0.1519% | 56.4 dB |
| `res1` | 0.0663% | 0.1979% | 54.1 dB |
| `up3` | 0.0190% | 1.1326% | 38.9 dB |
| `noise3` | 0.0020% | 1.2003% | 38.4 dB |
| `res3` | 0.0134% | 2.3444% | 32.6 dB |
| `mixed012` | — | 0.5866% | 44.6 dB |
| `all_mixed` | 0.1221% | 4.0586% | 27.8 dB |
| `all_coherent` | 0.1376% | 4.2398% | 27.5 dB |

`mixed012` is the useful “most of Synth in FP16” candidate: decoder stages 0,
1, and 2 run their upsample, NSF convolution, and residual convolutions in
FP16, while residual sums and the complete final stage remain FP32.  It reduced
the Python 44960 median from `55.3 ms` (FP32) to `42.2 ms` with `0.5866%`
relative L2.  On the normal-energy 440 Hz fixture it measured `0.0756%`
relative L2.  `coherent012`, which also accumulates those residual branches in
FP16, measured `0.7293%` at 44960, so FP32 residual accumulation is still the
safer boundary.  This is currently a Python calibration mode; the native graph
needs an explicit FP32-stage mask (stage bit `0x8`) before this recipe can be
selected in a TensorRT build.

The checked-in 44960 VIN was generated with `make-synth-vin.py
--native-deterministic`: its latent random term has RMS `0.007` and its NSF
noise has RMS `0.0007`.  Its FP32 waveform RMS is only `6.87e-5`.  A few
microvolts of absolute error therefore produce a large relative-L2 number even
though the maximum absolute error remains about `1e-5`.  The 440 Hz fixture
uses unit-scale random terms, has waveform RMS `0.148`, and the same
`all_mixed` path measures only `0.0838%` relative L2.  This is why cosine alone
is insufficient for this test: report VIN energy, RMS-normalized error, and SNR
as well.

Every canonical decoder activation stayed below `82.6` in the tested voices;
the FP16 overflow fraction was zero.  Per-tensor half quantization error was
roughly `2.0e-4` relative L2, with subnormal fractions below `0.1%`.  The large
44960 error is therefore accumulation and boundary placement, not overflow.
The final decoder stage operates at waveform sample rate and its residual
branches directly feed `conv_post`, so its rounding is not averaged by a
later upsample stage.

### Scaling experiment

The sweep also folds an inverse scale into the first deconvolution kernel and
tests larger activation scales on stages 1 and 3.  On the 44960 deterministic
fixture, stage 1 changed from `0.2614%` relative L2 at scale 1 to `0.2594%`
at scale 4 and `0.2633%` at scale 16.  The sensitive final stage changed from
`4.1912%` to `4.1905%` at scale 256.  These differences are below tactic and
run-to-run variation.  Scaling cannot add mantissa bits: for a normal FP16
value, both the value and its ULP grow by the same factor.  It only helps when
values are close to zero or subnormal, which is a very small fraction here.
Scaling a whole stage also requires rescaling both the learned noise branch
and residual sum; scaling only one branch changes the model.  Therefore no
global `scale > 1` is enabled.

## Boundary repair candidate

The script includes two Python-only repair variants that model a future native
stage mask:

* `native_mixed_keep_last`: stages 0..2 use the current mixed path and the
  complete final stage (upsample, NSF add, residual blocks, and accumulation)
  stays FP32.
* `native_mixed_keep_final`: keeps the final stage kernels in FP16 but avoids
  the final FP32-to-FP16 boundary cast before `conv_post`.

On the canonical 44960 input, `native_mixed` changed from `1.3235%` relative
L2 / `37.6 dB` SNR to `0.7340%` / `42.7 dB` with `native_mixed_keep_last`.
The same candidate changed the two independent real voices as follows:

| voice | current mixed | keep final stage FP32 | keep final boundary only |
| --- | ---: | ---: | ---: |
| Amitaro | 0.1028% | 0.1047% | 0.1020% |
| Kikoto Kurage | 0.2971% | 0.2515% | 0.2317% |

These are Synth-only comparisons against the FP32 Applio decoder.  The
candidate is not enabled by default because it adds a cast/tactic choice and
its benefit is voice dependent.  A native implementation should expose a
stage FP32 mask, validate it on several voices at both fixed profiles, and
select it only when the measured SNR/parity gate improves.

## Other model regions

The existing Python region reports provide the same guidance for the front
end:

* HuBERT feature extraction is sensitive: FP16 extractor hidden features are
  about `19%` relative L2 from the FP32 reference.  Projection (`0.032%`) and
  the Transformer encoder (`0.236%`) are suitable FP16 candidates.  Keep the
  convolutional feature extractor FP32 unless a downstream feature tolerance
  is explicitly accepted.
* RMVPE U-Net FP16 keeps coarse F0 exact for `99.29%` of frames and has F0
  relative L2 around `2.6e-5`; the head, GRU, and logits variants are below
  `4e-6` F0 relative L2.  The U-Net is the accuracy-sensitive boundary; the
  later regions are safer FP16 candidates, while STFT and F0 post-processing
  should remain FP32.

The sweep is a calibration tool, not a replacement for TensorRT-RTX timing:
the final decision still requires a fixed-profile native plan and CUDA-event
or Nsight measurements using the same voice checkpoints.
