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
values.  Its latency is useful for checking that the experiment is bounded, but
it is not a TensorRT INT8 performance measurement.  The next implementation
step is to add explicit scales only for the selected W8A16 residual nodes,
build a fixed 11200/44960 TensorRT plan, and compare CUDA event latency and
Nsight kernels.  A true W8A8 path should use per-channel or group activation
scales for the outlier-heavy upsample inputs and must repeat this gate on real
voices before being enabled.
