# Synth residual stage validation

The three low-precision stage masks were built and executed at 70 RVC frames
for two independent F0 voice packages:

* `target/default-canonical.vfr`
* `target/tsukuyomi-canonical.vfr`

The deterministic feature/latent/noise fixture was identical for each mask.
All outputs were finite. Relative L2 differences are measured against mask
`0x4` and are recorded in
`target/compare/synth-resblock-multi-voice.json`.

| voice | mask | max abs vs `0x4` | RMSE vs `0x4` | relative L2 |
| --- | ---: | ---: | ---: | ---: |
| default | `0x4` | 0 | 0 | 0 |
| default | `0x8` | 2.72e-4 | 4.07e-5 | 8.04e-4 |
| default | `0xc` | 2.23e-4 | 2.80e-5 | 5.52e-4 |
| tsukuyomi | `0x4` | 0 | 0 | 0 |
| tsukuyomi | `0x8` | 5.11e-4 | 1.10e-4 | 1.41e-3 |
| tsukuyomi | `0xc` | 2.24e-4 | 4.04e-5 | 5.21e-4 |

The direct test command accepts `VOCAL_TRT_TEST_FRAMES=70` and emits enqueue
latency plus finite/output statistics. The baseline defaults remain FP32
Synth; these masks are opt-in through the existing environment variables.
