# Synth INT8 screening

This experiment screens the generator convolution nodes before adding a real
TensorRT INT8 path.  It uses the Applio PyTorch FP32 Synth as the reference and
the actual VFRM model `target/tsukuyomi-canonical.vfr`.

The reproducible command is:

```powershell
$env:PYTHONPATH = "ref_repos/Applio"
rtk .venv\Scripts\python.exe scripts\synth-int8-sweep.py `
  --vfrm target\tsukuyomi-canonical.vfr `
  --vin target\compare\sine-11200.vin `
  --vin target\compare\sine-44960.vin `
  --calibration-vin target\compare\sine-11200.vin `
  --calibration-vin target\compare\sine-44960.vin `
  --max-candidates 24 --cumulative-counts 1,2,4,8 `
  --output target\compare\synth-int8-sweep-11200-44960.json
```

The JSON result is generated under `target/` and is intentionally not tracked.
The scan is limited to `dec.` because Nsight and the existing layer profile put
the Synth decoder residual and upsample convolutions on the critical path.

## Fake quantization definitions

- **W8A16**: per output channel INT8 weights, dequantized to FP16; the input
  activation is converted to FP16.
- **W8A8**: per output channel INT8 weights and a calibrated per tensor INT8
  activation scale.  The reference operation uses dequantized FP32 operands,
  so this is an accuracy screen for an INT32 accumulate and scale path, not a
  claim about a TensorRT tactic.
- **SmoothQuant**: calibrated per input channel activation/weight rescaling is
  applied before W8A8.  The tested alphas are `0.3`, `0.5`, and `0.7`.

The calibration set contains both 11200 and 44960 fixtures.  Every candidate
is tested at both lengths; recommendations use the worst relative L2 and the
lowest cosine across the two lengths.

## Results on RTX 3060

The single layer gate was cosine `>= 0.999` and waveform relative L2 `<= 1%`.
The highest MAC nodes are `dec.ups.1`, `dec.ups.0`, then the residual blocks
around `dec.resblocks.5`, `.4`, `.8`, and `.2`.

| candidate | 44960 W8A16 relative L2 | 44960 W8A8 relative L2 | decision |
|---|---:|---:|---|
| `dec.ups.1` | 3.38% | 6.11% | keep FP32 |
| `dec.ups.0` | 1.81% | 30.55% | keep FP32 |
| `dec.resblocks.5.convs2.2` | 0.52% | 7.53% | W8A16 candidate |
| `dec.resblocks.5.convs2.1` | 0.44% | 8.43% | W8A16 candidate |
| `dec.resblocks.4.convs1.0` | 0.25% | 0.98% | passes both individually |

The full per-layer metrics, shapes, MAC estimates and calibration scales are in
`target/compare/synth-int8-sweep-11200-44960.json`.

The cumulative W8A16 hot set is more useful than a single-layer result:

| individually passing layers selected by MAC | worst relative L2 | worst cosine |
|---:|---:|---:|
| 1 | 0.52% | 0.999987 |
| 2 | 0.67% | 0.999983 |
| 4 | 1.01% | 0.999960 |
| 8 | 1.52% | 0.999900 |

The initial opt-in set should therefore be the two `dec.resblocks.5.convs2`
nodes.  Expanding the set needs a real voice validation pass; the highest MAC
upsample layers do not meet the accuracy gate even though they offer the most
arithmetic to save.

W8A8 is not a default candidate.  Only `dec.resblocks.4.convs1.0` passed the
single-layer gate, and adding more layers was not evaluated as a useful speed
tradeoff.  W8A8 activation outliers explain much of this result.  For example,
the calibrated `dec.ups.0` input had channel max `80.24`, channel median
`2.15`, and max/median ratio `37.4`; a per tensor scale wastes most of the
INT8 range on a small number of channels.

## SmoothQuant result

SmoothQuant is not a general fix for this model.  On 44960, alpha `0.3`
improved `dec.ups.0` from `30.55%` to `7.26%` relative L2, but it still failed
the gate.  The same alpha changed `dec.resblocks.8.convs2.2` from `17.25%` to
`25.39%`, and several residual nodes became worse.  The best alpha is therefore
layer dependent and must be selected using speech calibration data rather than
enabled globally.

## What this does not establish

The Python wrapper executes floating point convolutions around fake quantized
values. Its latency is useful for checking that the experiment is bounded, but
it is not a TensorRT INT8 performance measurement. The native integration and
fixed-profile measurements are recorded below. A true W8A8 path should use
per-channel or group activation scales for the outlier-heavy upsample inputs
and must repeat this gate on real voices before being enabled.

## TensorRT-RTX W8A16 integration results

The first native path is now available through the opt-in environment variable
`VOCAL_TRT_SYNTH_W8A16_CONV2_MASK`. The mask uses
`resblock_index * 3 + conv2_suffix`; `0x30000` selects:

```text
dec.resblocks.5.convs2.1
dec.resblocks.5.convs2.2
```

Only those Conv2 kernels are represented as per-output-channel INT8 constants
followed by an FP16 dequantize node. The activation and residual accumulation
remain in the existing FP16/FP32 graph. The setting requires
`VOCAL_TRT_SYNTH_PRECISION=fp16` and cannot be combined with the global
`VOCAL_TRT_WEIGHT_INT8` path.

The plans were built and run on an RTX 3060 with TensorRT-RTX 1.6.1.120,
HuBERT FP16, RMVPE FP32 with cuFFT, Synth FP16, and native-subgraph CUDA Graph
replay. Measurements use fixed profiles and CUDA event timings:

| profile | Synth FP16 baseline | W8A16 selected | total baseline | total W8A16 |
|---|---:|---:|---:|---:|
| 11200 samples | 10.27 ms | 10.13 ms | 16.91 ms | 16.80 ms |
| 44960 samples | 36.73 ms | 36.13 ms | 50.67 ms | 49.03 ms |

These are close to tactic and clock noise; they do not establish an INT8
throughput gain. TensorRT-RTX may retain the explicit Q/DQ pair instead of
selecting a dedicated weight-only INT8 tactic for this graph.

The native audio difference between W8A16 and the corresponding FP16 baseline
was:

| profile | max absolute | RMSE | relative L2 | cosine |
|---|---:|---:|---:|---:|
| 11200 samples | 0.0024925 | 0.0007430 | 0.0032388 | 0.9999972 |
| 44960 samples | 0.0020577 | 0.0004851 | 0.0036909 | 0.9999964 |

Against the FP32 Applio Synth while feeding the native frontend tensors, the
W8A16 Synth-only cosine was `0.9999977` at 11200 and `0.9999966` at 44960.
The complete Applio end-to-end cosine was about `0.971` at 11200 and `0.969` at
44960; that larger gap is dominated by HuBERT/RMVPE implementation differences,
not by the selected Synth quantization.
