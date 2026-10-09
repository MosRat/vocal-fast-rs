# Realtime speech transitions

## Problems corrected

- The old SOLA joiner faded every boundary over 50 ms, including onsets where
  the saved tail was silence. Short consonants lost energy. It also faded away
  a previous decoded release when the next window revised it to near-silence.
- Correlation maximization on decoder residue/unvoiced noise selected random
  offsets, moving transients by up to 10 ms.
- RMS mixing used the newest source chunk after joining. For the 44,960/4,000
  preset, the decoder tail is 4,960 samples and playback begins 960 samples
  (60 ms) before the new chunk. New silence therefore attenuated an old tail.
- TensorRT multiplied the entire tail by callback-wide input RMS, even when
  RMS mix was 1, changing its level and attack/release relative to CPU/DML.
- Normal startup discarded 13 blocks (3.25 seconds) of captured speech.

## Shared policy

`vocal-engine/src/streaming_output.rs` owns the boundary DSP for all three
neural engines. It introduces no locks or extra playback buffering.

1. Voiced SOLA matches still align phase. Below -80 dBFS or correlation 0.35,
   retain the previous offset instead of moving noise/consonants randomly.
2. A 12 dB overlap energy imbalance selects onset/release handling. Hand over
   near the beginning/end of the overlap in 5 ms, keeping weak consonants and
   saved tails. No VAD gate clears context or restarts the stream during pauses.
3. Other overlaps use correlation-aware power compensation, bounded to 3 dB.
   Correlated voiced overlaps keep unity gain; breath/noise avoids a level dip.
   This adds no FFT or lookahead.
4. RMS mix below 1 uses centered 40 ms envelopes at 10 ms hops, interpolated per
   sample, on the source interval represented by the decoder tail. Apply it
   before SOLA so the saved tail retains its own aligned envelope. RMS mix 1
   preserves Synth volume on every backend.
5. Normal sessions immediately publish the valid tail with zero-padded missing
   history. CPU/DML keep explicit diagnostic startup muting as an opt-in stream
   setting. TensorRT now shares their stateful sinc input converter.

Local references checked: Applio `rvc/realtime/core.py` (rolling context,
scalar RMS, startup muting, onset fade, phase vocoder), Applio
`rvc/realtime/pipeline.py` (RMS source uses the decoder return interval), and
RVC WebUI `realtime_gui.py` (40 ms RMS/10 ms hops applied before SOLA).

The window/crop contract is retained. Startup discard and scalar output scaling
are replaced because they lose captured speech or its transition envelope.
Power-preserving overlap avoids adding the optional reference FFT phase vocoder.

## Verification, 2026-10-08

Generated audio/reports live outside Git in
`C:/Users/taizun/AppData/Local/vocal-fast/transition-verification/`.

- 72 identical real DirectML decoder tails, 18 seconds, default complete model:
  before/after RMS 0.066757/0.067193, neither clips. Selected weak transition
  overlaps retain 1–4 dB more energy; whole-stream level changes by 0.06 dB.
  This uses identical random inputs and is not a subjective listening score.
- Paced real inference through GUI batch adapter, SPSC queues and drainer:

  | backend / run | input drops | underflow | discontinuities | batch p50 / p95 / max |
  | --- | ---: | ---: | ---: | --- |
  | DirectML, repeated after | 0 | 0 | 0 | 104.1 / 124.3 / 229.2 ms |
  | TensorRT, after | 0 | 0 | 0 | 36.4 / 52.5 / 113.5 ms |
  | DirectML, immediate speech | 0 | 0 | 0 | 87.2 / 118.6 / 189.6 ms |
  | DirectML, current default pitch matching | 0 | 0 | 0 | 83.8 / 144.6 / 439.3 ms |

- Immediate speech: first 10 ms interval above RMS 0.005 moves from 3.80 s to
  1.14 s with matching disabled, and from 3.81 s to 1.11 s with current default
  pitch matching. First three seconds contain speech rather than discarded input.
  These threshold crossings include model/scheduling behavior, not hardware
  latency. FP32 PyTorch HuBERT/Synth were checked on five boundary windows:
  feature cosine >0.99998, decoded cosine >0.99983. The selected female model
  also produces weak output for some low male-register phonemes at pitch 0 in
  the reference; boundary DSP cannot reconstruct absent decoder output.
- CPU RMS mix 0: a real 17-second unpaced stream completes with finite output
  and unchanged sample count. Four-thread CPU inference was slower than realtime
  while another CPU benchmark was active; this is not a CPU latency guarantee.
- One concurrent-load DirectML attempt had seven input drops and one input
  discontinuity (`after-dml.json`). Its uninterrupted repeat remains separately
  recorded as `after-dml-repeat.json`; stressed attempts are not deleted.
- The additional RMS mix 0 DirectML run has no input loss/discontinuity but four
  underflow ticks at 7.55–7.58 s under concurrent load. Smooth envelope matching
  is not a guarantee against scheduler starvation; that report is retained as
  `after-dml-envelope.json`.
- Workspace all-feature checks/tests and Tauri tests pass. The embedded-frontend
  debug application is rebuilt with CPU, DirectML and TensorRT support. Test
  totals: workspace 236 passed/19 ignored, Tauri 26 passed/1 ignored; the real
  decoder-tail replay was also explicitly run successfully.

No physical capture/render device was opened by these probes. Hardware loopback
and subjective listening remain separate checks.

Reproduce the identical-tail DSP comparison (use a target directory with space):

```powershell
$env:VOCAL_DSP_TRACE_DIR = 'C:/Users/taizun/AppData/Local/vocal-fast/transition-verification/trace'
rtk cargo test -p vocal-engine --all-features --lib replay_real_decoder_tails -- --ignored
rtk .venv/Scripts/python.exe -X utf8 scripts/analyze-speech-transitions.py `
  $env:VOCAL_DSP_TRACE_DIR `
  C:/Users/taizun/AppData/Local/vocal-fast/transition-verification/trace-after.wav `
  C:/Users/taizun/AppData/Local/vocal-fast/transition-verification/dsp-report.json
```

Regression tests cover weak onsets, preserved releases, unreliable matches,
noise power, correlated unity gain, reset, disabled crossfade, four public
sample rates, aligned RMS tails and continuous envelope interpolation.
