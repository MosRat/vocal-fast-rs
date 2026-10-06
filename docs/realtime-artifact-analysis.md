# Realtime RVC boundary artifact analysis

`target/offline-augmented-rvc.wav` was generated before the realtime fixes. It
uses the legacy `run-trt-pipeline` loop: each 44,960 sample context is passed
to `RvcTrtPipeline::process()` and the returned waveforms are concatenated. That
path is useful for a full-context benchmark, but it is not the callback path
used by Applio, RVC WebUI, or voice-changer.

## Observed difference

The test input is 16 kHz, 27.1 seconds, RMS `0.03389`, peak `0.41345`.

| file | output rate | RMS | peak |
| --- | ---: | ---: | ---: |
| `offline-augmented-rvc.wav` (old) | 40 kHz | `0.15107` | `0.85181` |
| `offline-augmented-rvc-fixed.wav` (realtime runner) | 40 kHz | `0.00581` | `0.06644` |

The old file is roughly 4.5 times louder than the input RMS and contains a
stable, repeated residue at the start and end of context windows. The corrected
file uses 109 4,000-sample playback blocks with a 44,960-sample retained
context. Its first 13 blocks are intentionally silent while the context is
filled, matching the reference worker warmup policy.

## Root causes

### 1. Periodic Synth source instead of RVC noise

The old Rust path seeded `rnd` and `nsf_noise` once with low-amplitude sine
waves and reused the same prefix for every window. A deterministic periodic
source is audible after the flow and NSF decoder as a mechanical tone.

RVC and Applio sample `z_p` from a standard normal distribution on every
inference (`torch.randn_like(m_p) * 0.66666`). The NSF `SineGen` branch also
uses standard normal noise. The current Rust path regenerates both buffers for
each Synth call with a deterministic PRNG so runs remain reproducible without
reusing a periodic waveform:

- [pipeline.rs random source generation](../crates/vocal-tensorrt/src/pipeline.rs:2015)
- [pipeline.rs Box-Muller generator](../crates/vocal-tensorrt/src/pipeline.rs:2549)
- [Applio Synth](../ref_repos/Applio/rvc/lib/algorithm/synthesizers.py:228)
- [RVC SineGen noise](../ref_repos/Retrieval-based-Voice-Conversion-WebUI/infer/module/models.py:341)

### 2. Missing input RMS envelope

The old output was written directly after resampling. Applio keeps a rolling
`block + crossfade` input buffer, computes its RMS, and multiplies the decoded
audio by that RMS before output resampling. This suppresses decoder residue
when the input enters or leaves silence. The current engine follows that rule:

- [Applio rolling RMS and output scaling](../ref_repos/Applio/rvc/realtime/core.py:270)
- [Applio output envelope](../ref_repos/Applio/rvc/realtime/core.py:432)
- [Rust rolling RMS update](../crates/vocal-engine/src/tensorrt_backend.rs:349)
- [Rust output envelope](../crates/vocal-engine/src/tensorrt_backend.rs:399)

The old output's active-region RMS is `0.16381`; the corrected output is
`0.00630` after the same envelope is applied. The lower level is expected for
this low-level test input and removes the strong mechanical tail.

### 3. Full-context concatenation has no streaming boundary treatment

The callback implementations retain a context window, but publish only the
new block. They do not concatenate complete decoder contexts. They also keep a
crossfade/SOLA tail between callbacks:

- Applio returns zero blocks while `warmup_blocks` is nonzero, then uses a
  rolling SOLA buffer and onset-aware fade [core.py](../ref_repos/Applio/rvc/realtime/core.py:316)
  [core.py](../ref_repos/Applio/rvc/realtime/core.py:576).
- RVC WebUI keeps a zero SOLA buffer, applies RMS envelope mixing when enabled,
  then performs normalized SOLA and fade windows
  [realtime_gui.py](../ref_repos/Retrieval-based-Voice-Conversion-WebUI/realtime_gui.py:867)
  [realtime_gui.py](../ref_repos/Retrieval-based-Voice-Conversion-WebUI/realtime_gui.py:901).
- voice-changer initializes the SOLA buffer and returns a zero first block,
  then aligns the next block by normalized correlation
  [VoiceChanger.py](../ref_repos/voice-changer/server/voice_changer/VoiceChanger.py:222).

The shared Rust offline runner now exercises the same realtime engine adapter
and crops only the final padded block:

- [shared offline runner](../crates/vocal-engine/src/offline.rs:137)
- [realtime warmup and publish path](../crates/vocal-engine/src/tensorrt_backend.rs:346)

## Reproduction

Use the corrected shared runner for an audible offline result:

```powershell
.\target\debug\vocal-fast.exe model run-pipeline tensorrt `
  target\test-assets\mandarin-rvc\maobailing-rvc.vfr `
  target\test-assets\mandarin-rvc\chinese-input-offline-augmented.wav `
  target\offline-augmented-rvc-fixed.wav `
  target\maobailing-pipeline.json
```

The original file is intentionally retained for A/B comparison. Generated
audio and TensorRT plans remain under `target/` and are not repository assets.
