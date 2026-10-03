# Synth residual stage tuning

Mixed Synth used to convert every upsample stage to FP32 for the residual
blocks and then convert back to FP16. The native graph now supports a stage
mask so this cast traffic can be measured without changing the stable default.

```powershell
$env:VOCAL_TRT_SYNTH_RESBLOCK_PRECISION = "fp16"
$env:VOCAL_TRT_SYNTH_RESBLOCK_STAGE_MASK = "0x8"
```

Bit 0 selects upsample stage 0. A zero mask keeps all residual blocks in FP32.
The mask is recorded in `pipeline.json`, and the builder rejects bits beyond
the model upsample count.

On an RTX 3060 with 11200 samples, HuBERT FP16, RMVPE FP32 and Synth FP16, the
single stage sweep was compared with Applio PyTorch FP32:

| mask | audio relative L2 | Synth event (ms) | interpretation |
|---:|---:|---:|---|
| `0` (all FP32 residual) | 0.00203 | 13.39 | stable baseline |
| `0x1` | 0.00454 | 9.23 | error too large |
| `0x2` | 0.00360 | 7.99 | speed gain, outside target |
| `0x4` | 0.00237 | 8.95 | close to baseline |
| `0x8` | 0.00189 | 9.46 | best observed trade-off |

These are tuning observations, not a universal default. Test a real voice model
over multiple speech blocks before enabling a mask. The next sweep should cover
`0x4`, `0x8`, and `0xc` over multiple windows, followed by Nsight Compute
verification that the changed stages use Tensor Core convolution tactics.

The Nsight Systems capture is
`target/compare/nsys-resmask8.nsys-rep` (SQLite export beside it). Across 50
iterations the stable pipeline ranges were dominated by RMVPE; the Synth NVTX
range was about 0.94 ms because it records enqueue/graph replay rather than
the complete device execution. The CUDA memory summary also showed host to
device copies dominating the recorded memcpy time. Those copies include
TensorRT shape bookkeeping and graph setup, so the next measurement should
separate the first capture from steady replay before changing host staging.

The machine-readable sweep is
`target/compare/synth-resblock-sweep-11200.json`.
