# Retrieval audio regression: Python comparison and output bounds

Validated 2026-10-07 on RTX 3060 with the original `default-v2/default.pth`
and its `added_IVF511_Flat_nprobe_1_default_v2.index`. The native package is
`target/default-canonical-with-index.vfr`; the matching compiled pipeline is
`target/trt-default-retrieval-44960/pipeline.json`.

## Fault and repair

The fixed 44960 Synth plan encodes 281 frames but decodes only 31 frames after
`skip_head=250`. At 40 kHz it writes **12400 samples**. The non-realtime
`process()` download incorrectly requested 281 × 400 = **112400 samples**,
exposing **100000 unwritten samples** from a reusable CUDA allocation. Those
bytes could be zero, stale audio or other finite values; a finite-value check
could not establish correctness. Concatenating these buffers produced an
invalid listening file independently of retrieval accuracy.

The repair bounds the download to the compiled tail even for non-realtime F0
extraction. It additionally checks the actual TensorRT audio binding shape
when input geometry changes, so stale manifest metadata cannot cause another
oversized download. Full-context diagnostic plans expose the valid offline
prefix (280 frames for this input), matching Applio's non-appended feature
count. Realtime remains 281 frames with the appended HuBERT boundary frame.

`run-trt-pipeline` concatenates complete independent contexts and now rejects
static tail plans. Listening output uses the existing shared `run-pipeline`
runner with retained history, F0 cache, input RMS envelope and SOLA. For this
preset it advances 4000 samples per callback and publishes 250 ms of audio.
Do not pass 44960 as a playback chunk merely because it is the model context.

## Correct Python comparisons

Two older comparison assumptions were invalid:

- `hubert_features` is an in-place **post-retrieval** snapshot. Compare it to
  the FAISS blend, or run with ratio zero to obtain raw HuBERT features.
- The old parity scripts reconstructed small sine sequences for `rnd` and
  `nsf_noise`; production now draws fresh standard-normal values per call.
  Comparing independently sampled Synth waveforms does not isolate graph
  arithmetic. The earlier approximately 0.977 cosine is not a valid graph
  parity result.

Native debug output now includes the exact random tensors, valid input length
and output crop. `scripts/compare-retrieval-python.py` loads the **original
PTH** into the local Applio implementation with strict parameter validation.
It executes its real encoder, flow, NSF source and decoder using the recorded
inputs. For static tails it retains the WebUI 24-frame flow context and crops
before decoding, as the native graph does. No VFRM-derived weights are used
as the reference in this test.

The following measurements use index ratio 0.5. Native HuBERT is FP16 with
FP32 feature extraction; RMVPE is FP32. Synth encoder/flow and core operations
are FP32 with residual stages 0–2 in FP16 (`synth_resblock_stage_mask=7`).
This is a mixed precision plan, not an all-FP32 Synth.

| Comparison | Maximum absolute error | RMSE | Cosine |
| --- | ---: | ---: | ---: |
| GPU retrieval vs FAISS, identical raw queries | 4.77e-7 | 1.64e-8 | ≈1.0 |
| Frontend vs repeat/crop of FAISS blend | 4.77e-7 | 1.64e-8 | ≈1.0 |
| Tail Synth vs original PTH/Applio FP32, exact inputs | 8.82e-4 | 1.89e-4 | 0.999999833 |
| Realtime tail Synth with F0 cache and native graph replay | 9.89e-4 | 1.69e-4 | 0.999999827 |

The realtime test used one warmup and three measured calls; native replay
reported two captures, one shape invalidation and two replays, including the
transition from full-window F0 to the 4960-sample cached F0 path. Its output
remained 12400 samples. Diagnostic host copies were enabled, so these calls
are correctness tests, not production latency benchmarks.

One query searched a list with six candidates. The FAISS reference handles
the two `-1` IDs as zero-weight entries, matching the native implementation.

Reports are in `target/compare/retrieval-fixed/parity.json` and
`target/compare/retrieval-fixed/realtime-parity.json`. These are stage parity
results; they are **not** a claim that independent offline and realtime
waveforms have cosine 0.9999998.

## New listening files

All files use the augmented Chinese input and the matching default voice/index.

| File under `target/` | Duration | RMS | Peak |
| --- | ---: | ---: | ---: |
| `offline-retrieval-fixed.wav` (ratio 0.5) | 27.10 s | 0.005624 | 0.059631 |
| `offline-no-retrieval-fixed.wav` (ratio 0) | 27.10 s | 0.005615 | 0.061066 |
| `offline-retrieval-applio.wav` (original offline reference, ratio 0.5) | 27.08 s | 0.153906 | 0.810377 |

Native files are PCM16; the Python reference is float32. All are 40 kHz and
finite. The native output keeps the reference realtime startup warmup (13
silent blocks) and rolling input RMS scaling. Applio offline instead applies
a 48 Hz high-pass filter, reflected padding and a full offline crop; its
`volume_envelope=1.0` does not apply the realtime input RMS multiplier. Its
shorter valid feature sequence accounts for the 20 ms duration difference.
The Python file is consequently much louder. These are separate listening
references, not sample-aligned end-to-end parity fixtures. No listening
quality rating is inferred solely from the numerical checks.

## Reproduce

```powershell
rtk cargo build -p vocal-fast --features tensorrt-rtx
$env:VOCAL_INDEX_RATIO = '0.5'
rtk target/debug/vocal-fast.exe model run-pipeline tensorrt `
  target/default-canonical-with-index.vfr `
  target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav `
  target/offline-retrieval-fixed.wav target/trt-default-retrieval-44960/pipeline.json

rtk .venv/Scripts/python.exe scripts/run-applio-offline.py `
  --input target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav `
  --output target/offline-retrieval-applio.wav `
  --checkpoint models/voices/default-v2/default.pth `
  --index models/voices/default-v2/added_IVF511_Flat_nprobe_1_default_v2.index `
  --report target/compare/retrieval-fixed/applio-offline.json
```

For numerical parity, use fresh `test-trt-pipeline` processes on the same WAV
with `VOCAL_TRT_DEBUG_INTERMEDIATES=1`, `VOCAL_TRT_REALTIME_F0=0`, warmup zero
and one iteration. Save the generated `trt-debug.json` after each ratio-zero
and ratio-0.5 run as `zero-debug.json` and `half-debug.json`, respectively.

```powershell
rtk .venv/Scripts/python.exe scripts/compare-retrieval-python.py `
  --debug target/compare/retrieval-fixed/half-debug.json `
  --raw-debug target/compare/retrieval-fixed/zero-debug.json `
  --checkpoint models/voices/default-v2/default.pth `
  --index models/voices/default-v2/added_IVF511_Flat_nprobe_1_default_v2.index `
  --report target/compare/retrieval-fixed/parity.json `
  --audio-dir target/compare/retrieval-fixed
```

For a realtime snapshot use `VOCAL_TRT_REALTIME_F0=1`, warmup one and three
iterations; the same checker accepts it. `--raw-debug`/`--index` may be omitted
when only testing Synth.

## Verification

- `cargo fmt --all -- --check`
- `cargo test -p vocal-tensorrt --lib`: 30 passed, including the oversized
  static-tail download regression
- `cargo test -p vocal-fast --features tensorrt-rtx`
- `cargo check --workspace --all-features`
- `cargo test --workspace`: 109 passed
- Python `py_compile` for both new scripts
- Actual GPU pipeline, exact-input FAISS/Applio comparisons and generated WAVs
- Legacy full-context command rejects the tail artifact before writing output

Generated audio, snapshots, plans and weights remain outside Git under
`target/` or the existing model directories.
