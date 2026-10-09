# Full-file DML, TensorRT and original Python validation

Measured on 2026-10-07 with an RTX 3060 and Cargo debug builds. This check
converts actual audio through the shared engine file runner, retains every
output sample, and independently evaluates all 109 neural windows against
original FP32 PyTorch weights. It includes startup, rolling F0, retrieval,
tail decoding, RMS, SOLA, resampling and final padding/trimming.

## Audio and processing time

Input: `target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav`,
mono 16 kHz, 433600 samples / 27.10 s. Voice: `default-v2/default.pth` and
its IVF511 index. Index ratio 0.5, speaker 0, pitch shift 0. Native engines
and their matching Python streaming references use protect 0.33. The
original Applio offline listening file uses protect **0.5** (protection
disabled), volume_envelope 1.0 and a native-rate output. The earlier text
incorrectly described protect as common to all three files. They are not
equal-configuration audio conversions; see `applio-config-alignment.md`.
Native context is 44960 samples; playback advances 4000 samples / 250 ms.
Both backends retain a 1024-frame F0 history, extracting 28 interior RMVPE
frames from the recent 4960 samples and shifting history by 25 frames.
The shared CLI requests mono 16 kHz public engine callbacks and resamples
their outputs to the model's 40 kHz WAV rate. The matching Python DSP does
the same 40k → 16k → 40k conversion; the original Applio offline file keeps
its native 40 kHz decode. Frequency-content comparisons also include this
concrete difference in the file runner's output-rate contract.

Outputs are under `target/compare/offline-dml-trt-python/`:

| Output | Duration | Processing, excluding model load | Mean per native callback |
| --- | ---: | ---: | ---: |
| `dml.wav` | 27.10 s | 4224.57 ms | 38.76 ms |
| `trt-cufft.wav`, with repaired protect | 27.10 s | 2471.78 ms | 22.68 ms |
| `applio-offline.wav`, original FP32 offline pipeline | 27.08 s | 2711.96 ms | Not a streaming callback benchmark |

Native timings include all 109 calls, startup and DSP; they exclude loading,
compilation and WAV encoding. Python's original offline runner uses larger
offline segments with high-pass filtering and reflected padding. Its total
time is useful for file conversion, but does not measure realtime latency.
The separate diagnostic/reference runs copy intermediate tensors and perform
additional inference, and are excluded from this table.

The DML recipe uses FP16 HuBERT transformer layers and FP32 normalization and
feature extraction, FP32 RMVPE, and FP16 Synth decoder stages 0–2/noise
convolutions. Encoder, flow, NSF phase/source, pre, final stage and post remain
FP32. The TRT artifact in this comparison uses FP16 HuBERT with FP32 feature
extraction, FP32 RMVPE with cuFFT, and FP16 residual blocks at Synth stages
0–2; other Synth computations are FP32. This is a comparison of these actual
prepared artifacts, not a claim to benchmark every backend's fastest recipe.

Native files are quiet because the common realtime adapter scales by input
RMS. `listening/dml.wav`, `listening/trt.wav` and `listening/applio.wav` are
**separate float32 listening copies** with a target RMS of 0.1 and peak cap
0.9. These copies do not change inference. Original unnormalized files are
retained for metrics. Native originals have no clipped/non-finite samples,
zero RMS in the first/last 500 ms, and 40 kHz / 1084000 output samples.
The initial 13 muted callbacks reproduce the reference realtime warmup.

Native boundary jumps are smaller than the largest ordinary adjacent-sample
jumps: DML 0.00238 vs 0.01056; repaired TRT 0.00171 vs 0.01016. This checks
buffer bounds and discontinuities, and does not constitute a subjective
listening quality rating.

## Complete-stream numerical comparison

`scripts/run-python-streaming-reference.py` loads original HF HuBERT,
Applio RMVPE PT and Synth PTH files, with TF32 disabled in PyTorch. Exact
latent and NSF noise are recorded from each native stream in binary form.
Python independently computes all acoustic and Synth stages and reproduces
the shared realtime DSP. Its Synth reference deliberately uses WebUI's
24-frame flow context before the decoder tail; original Applio's realtime
`rate` crop removes the head before the flow without this context. Thus the
matching Python metrics validate original weights under our streaming
policy, not every DSP/crop policy in Applio's original realtime worker.
Each backend has its own matching random trace;
direct DML-versus-TRT production waveforms use different RNG sequences.

The following tail metrics aggregate **all 96 audible decode windows**, not
only the final window. The complete WAV comparisons contain all 27.10 s.

| Comparison against matching FP32 Python | DML cosine / RMSE | Repaired TRT + cuFFT cosine / RMSE |
| --- | ---: | ---: |
| Synth alone, identical native features/F0/noise | 0.999994857 / 0.00051190 | 0.999999754 / 0.00020246 |
| Complete neural tail, independently computed features/F0 | 0.999950802 / 0.00150111 | 0.999067652 / 0.00649660 |
| Complete WAV, identical SOLA offsets | 0.999863132 / 0.00009347 | 0.999113017 / 0.00023674 |
| Complete WAV, independent SOLA decisions | 0.288754 / 0.00673577 | 0.995808 / 0.00051463 |

SOLA performs an argmax over periodic correlation peaks. Small neural
differences can select another peak and alter the next overlap buffer,
propagating a phase offset. DML selected 70 different offsets, and repaired
TRT selected 5. Fixing offsets isolates graph error but is **a diagnostic
comparison**, not the autonomous production output. Both Python variants
are saved for listening. The low autonomous DML waveform cosine must not
be presented as either exact graph parity or proof of audible noise.

Reconstructing native diagnostic outputs through the Python DSP matches the
actual production WAV closely: cosine 0.999997695 for DML and 0.999997683
for TRT. This is also a check that the trace uses the actual file runner's
window/RNG/startup/joining contract. Diagnostic graph observability can
change numerical operation ordering; this check does not assert bit equality.

Production DML/TRT loudness envelopes have cosine 0.997382, no coarse time
offset, and 0.730 dB mean absolute log-magnitude difference after gain
matching on audible frames. These are descriptive signal comparisons.
Applio offline leads by approximately 60 ms and is much louder because its
DSP policy differs. Comparing its raw samples to streaming files would mix
DSP, noise, phase and graph differences.

Reports: `audio-analysis.json`, `python-streaming-parity.json`,
`python-trt-cufft-parity.json`. Audio: `python-streaming.wav`,
`python-streaming-fixed-sola.wav`, `python-trt-cufft-streaming.wav` and
`python-trt-cufft-streaming-fixed-sola.wav`.

## Correctness repair and remaining F0 difference

The TRT frontend previously consumed only post-retrieval HuBERT features.
Its advertised protect=0.33 policy did not preserve the original unvoiced
content. Raw features are now retained in a GPU buffer before retrieval.
After frontend interpolation, a CUDA kernel applies
`retrieved * 0.33 + raw * (1 - 0.33)` only when continuous F0 is zero and
index ratio is positive. The final appended HuBERT frame uses the same
repeat/crop mapping as Python. Voiced features and ratio-zero results remain
unchanged. Serial, parallel acoustic, whole and hybrid enqueue sequences all
include this operation; the ratio is device data compatible with replay.

The prior `trt.wav`/DFT run without protection had frontend cosine 0.979819,
neural-tail cosine 0.996062, and full-WAV cosine 0.995969 when fixing SOLA.
The repaired cuFFT run reaches 0.999770, 0.999068 and 0.999113 respectively.
The original reports/files are retained as the before-repair evidence.

There is still a measured TRT RMVPE arithmetic difference. Among 3024
interior frames appended after the initial call, five differ from DML/FP32
Python by more than 1 Hz. One frame at chunk 71 has reference maximum
salience **0.03007193**, versus native **0.02929005**; the common threshold
0.03 therefore produces approximately 162.47 Hz versus zero. The cache
retains this one discrepancy through subsequent overlapping windows; these
are not multiple independent failures. DML's whole F0 history max error
against FP32 Python is 0.000366 Hz.

Freshly rebuilding FP32 RMVPE and switching DFT to cuFFT did not remove this
event. Comparing PyTorch with the same native mel still shows U-Net/logit
differences, so this is not explained by STFT or cache indexing alone. At
chunks 15/43/71, native-logit RMSE is 0.01294/0.01062/0.00759. TF32 as the
cause has **not** been established. A runtime `NVIDIA_TF32_OVERRIDE=0` trial
failed to create a context and is not a supported workaround recorded here.
The threshold has not been adjusted to hide the discrepancy. Detailed
evidence is in `rmvpe-worst-stages.json` and `rmvpe-unique-events.json`.

A later complete build-and-run experiment under `NVIDIA_TF32_OVERRIDE=0`
corrected the long offline graph's F0 drift. The earlier runtime-only attempt
failed because RTX records this setting in the plan. See the
[aligned offline report](offline-applio-aligned.md) for the tested process
policy, separate cache namespace and newly generated WAVs. The files and
metrics above describe the previous callback runner.

## Reproduce with prepared artifacts

```powershell
rtk cargo build -p vocal-fast --features directml,tensorrt-rtx
$env:VOCAL_INDEX_RATIO = '0.5'
rtk proxy target/debug/vocal-fast.exe model run-streaming-pipeline directml `
  target/default-complete-with-index.vfrm `
  target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav `
  target/compare/offline-dml-trt-python/dml.wav target/dml-sources-rolling-with-index.json
rtk proxy target/debug/vocal-fast.exe model run-streaming-pipeline tensorrt `
  target/default-canonical-with-index.vfr `
  target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav `
  target/compare/offline-dml-trt-python/trt-cufft.wav `
  target/compare/offline-dml-trt-python/trt-cufft/pipeline.json
rtk .venv/Scripts/python.exe scripts/run-applio-offline.py `
  --input target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav `
  --output target/compare/offline-dml-trt-python/applio-offline.wav `
  --checkpoint models/voices/default-v2/default.pth `
  --index models/voices/default-v2/added_IVF511_Flat_nprobe_1_default_v2.index `
  --report target/compare/offline-dml-trt-python/applio-run.json
```

The cuFFT manifest retains the original voice-specific Synth and HuBERT
plans, and uses shared FP32 RMVPE full/realtime plans from
`target/opt-scale-final-44960/pipeline.json`. It records external STFT mode
2. No voice-specific weights are taken from that other pipeline.

For independent full-file parity, first decode the PCM16 source into raw
little-endian f32 (`/32768`), then record exact noise/stages:

```powershell
rtk cargo run -p vocal-directml --example validate -- streaming `
  target/dml-sources-rolling-with-index.json target/dml-chinese-stream.f32 `
  target/compare/offline-dml-trt-python/dml-trace-final.json `
  manifest 107 debug 0.5 target/compare/offline-dml-trt-python/dml-trace
rtk cargo run -p vocal-tensorrt --example trace_streaming -- `
  target/compare/offline-dml-trt-python/trt-cufft/pipeline.json `
  target/default-canonical-with-index.vfr target/dml-chinese-stream.f32 `
  target/compare/offline-dml-trt-python/trt-cufft-trace 0.5
rtk .venv/Scripts/python.exe scripts/run-python-streaming-reference.py `
  --input target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav `
  --trace target/compare/offline-dml-trt-python/trt-cufft-trace `
  --output target/compare/offline-dml-trt-python/python-trt-cufft-streaming.wav `
  --native-output target/compare/offline-dml-trt-python/trt-cufft.wav `
  --report target/compare/offline-dml-trt-python/python-trt-cufft-parity.json
```

For DML use its trace directory, `dml.wav` as native output and corresponding
report/output names. `compare-offline-wavs.py` accepts repeated
`--audio NAME=PATH`, optional `--run NAME=JSON`, `--report PATH` and
`--normalize-dir PATH` for backend-neutral file and listening comparisons.
`build_rmvpe` rebuilds an isolated FP32 graph from VFRM with a chosen fixed
sample count and optional cuFFT/debug outputs for future stage investigation.
WAVs, diagnostic traces, weights and plans remain untracked.

## Verification

- `cargo fmt --all -- --check`
- `cargo check --workspace --all-features`
- `cargo test --workspace`: 128 passed, 7 ignored
- Actual CUDA protection test: voiced/unvoiced frames, final appended frame,
  ratio zero, graph capture, and changing the device ratio during replay
- Python `py_compile` for the new/reference comparison scripts
- Actual shared-runner DML/TRT WAV generation and original Applio inference
- Complete 109-window binary traces and independent FP32 Python evaluations
