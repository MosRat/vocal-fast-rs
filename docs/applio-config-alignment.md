# Applio configuration and DSP alignment

The investigation below describes the previous streaming listening files.
The native file command has since been aligned to Applio offline processing:
see [the new implementation and numerical checks](offline-applio-aligned.md).
Use `model run-streaming-pipeline` to reproduce the historical callback path.

Checked on 2026-10-07 using the existing 27.10 s Mandarin recording and
`default-v2/default.pth`. This separates implementation/configuration
differences from floating-point graph errors. The production engines have
not been changed by this investigation.

## What the previous listening files actually compare

`model run-pipeline` feeds a file through the **streaming** engine. It is
not an implementation of Applio's long-segment offline policy. Both native
files use the 44960 profile, a 4000-sample playback advance, rolling F0 and
SOLA. `applio-offline.wav` uses the original Applio offline pipeline.

Common settings are the same original HuBERT/RMVPE/Synth weights, speaker
0, pitch shift 0, RMVPE threshold 0.03, matching IVF511 index, retrieval
ratio 0.5 and inverse-square weighting of eight neighbors. Autotune,
proposed pitch and the high-register correction are disabled in the
original offline runner. They do not explain this comparison.

| Setting / operation | Native DML / TRT listening files | Original Applio offline listening file |
| --- | --- | --- |
| Synth/output rate | Synth 40 kHz, callback 16 kHz, saved WAV 40 kHz | Native 40 kHz through to WAV |
| Output resampling | Linear 40k -> 16k -> 40k | No output-rate conversion in this test |
| Neural waveform window | 44960 samples / 2.81 s, advances 250 ms | One 29.1 s window: 27.1 s source plus 1 s reflected padding at each end |
| F0 extraction | Recent 4960 samples / 310 ms, `[3:-1]` interior, rolling history | Whole reflected input; no rolling cache |
| HuBERT final frame | Append final frame before 2x nearest repeat | No appended frame; crop to available feature length |
| Decoder/flow | Decode 31 frames after skip_head 250; WebUI flow context of 24 frames | Full flow/decoder followed by reflected-context waveform crop |
| Input filtering | GPU adapters retain raw waveform | 48 Hz fifth-order Butterworth `filtfilt` |
| Startup and padding | Zero-filled history, 13 muted callbacks, zero-pad final chunk | Reflect padding and crop; no streaming startup mute |
| RMS / loudness | Always multiply by 300 ms rolling input RMS; optional RMS mix is separate | volume_envelope=1 preserves Synth loudness; no realtime RMS multiplier |
| protect | 0.33 | **0.5**, protection disabled |
| Joining / excitation | New latent/NSF noise and source phase computation each callback; SOLA + sin² crossfade | Continuous long-segment decode; no streaming SOLA |

The earlier comparison report incorrectly described protect 0.33 as common
to all files. It is corrected in `offline-backend-comparison.md`. This is a
test-runner setting difference: Applio's realtime UI itself defaults to
protect 0.33. The implementation condition is `protect < 0.5`; at 0.5 no
protection blend is performed, regardless of the UI's explanatory wording.

Sources:

- [CLI output-rate contract](../crates/vocal-cli/src/main.rs),
  `run_offline_pipeline`.
- [Shared streaming adapter](../crates/vocal-engine/src/audio_adapter.rs),
  `StreamingOutputJoiner` and `apply_rms_mix`.
- [DML engine DSP](../crates/vocal-engine/src/directml_backend.rs), `infer`;
  [TRT engine DSP](../crates/vocal-engine/src/tensorrt_backend.rs),
  `infer_realtime_chunk`.
- Local reference `ref_repos/Applio/rvc/infer/pipeline.py`, `pipeline`,
  `get_f0` and `voice_conversion`; settings are recorded by
  [the Applio runner](../scripts/run-applio-offline.py).

### Why these differences can change naturalness

The output-rate round trip changes consonant/breath spectra and introduces
aliases because interpolation alone is not a proper antialiasing
downsampler. The input-rate requirement of HuBERT/RMVPE is 16 kHz; it does
not require reducing the generated voice to 16 kHz. GUI device callbacks
can request other public rates, so this exact round trip is a finding about
the current **file command**, not a claim that every GUI callback is 16 kHz.

The realtime RMS multiplier changes the amplitude envelope, not just the
overall gain. Quiet consonants/breaths can be suppressed a second time.
It is present in Applio's **realtime** `core.py`, which passes raw audio
without dividing it by RMS and multiplies the decoder result by `vol_t`.
Our multiplier follows that policy. It is nevertheless different from
Applio's **offline** volume_envelope=1 policy. Setting our rms_mix_rate=1
does not disable this independent multiplier.

Long offline extraction also gives bidirectional RMVPE, HuBERT and the Synth
encoder a different context. Streaming noise/source restarts and periodic
SOLA decisions add further boundary differences. These are plausible
contributors; their individual subjective impact has not been ranked by
the experiments below. Switching every graph to FP32 would not remove them.

## Controlled measurements without changing graph precision

[analyze-offline-dsp.py](../scripts/analyze-offline-dsp.py) reuses existing
Synth tail traces and validated Python streaming DSP helpers. The helpers
import the reference Python dependencies, but no models are loaded and no
neural inference is performed for the DSP experiments. Applio protect runs
are independent original FP32 conversions with the same seed and settings,
changing only protect.

| Controlled change | Result |
| --- | --- |
| Applio native 40 kHz vs the same audio through the current linear 40k -> 16k -> 40k policy | cosine 0.997305298; RMSE 0.01130862; max absolute error 0.19655755; SNR 22.68 dB |
| Applio protect 0.5 vs 0.33, same FP32 models/input/seed | cosine 0.996391713; RMSE 0.01308882; max absolute error 0.73626414 |
| Same recorded DML Synth tails at native 40 kHz, with vs without input-RMS multiplication; identical SOLA offsets | 20 ms envelope cosine 0.899437; envelope relative RMSE 0.43705 after fitting one constant gain |
| Reconstruct production DML DSP from the trace, including PCM16 output | cosine 0.999999997994; RMSE 3.58e-7; maximum difference one PCM16 unit |

These metrics do not establish which audio sounds better. They establish
that substantial differences exist without changing floating-point
precision. They must not be added together or treated as autonomous
end-to-end graph parity.

For the Applio round trip, the Welch spectrum's normalized 4-8 kHz power
share changes from 0.4562% to 0.2505%, and the 8-20 kHz share from 0.5458%
to 0.0952%. The latter includes upsampling images; it is not restored
original high-frequency content. A 12 kHz sine passed through the current
downsampler produces a peak at **4 kHz**, directly demonstrating aliasing.

For the native RMS ablation, the rolling gain's 5th/50th/95th percentiles are
0.001589 / 0.023043 / 0.058427. Native-rate audio RMS changes from 0.005681
to 0.143238 when the multiplier is removed. Matching listening loudness with
one constant gain does not undo this varying envelope. Removing that
multiplier globally from realtime use could expose quiet decoder residue;
an offline loudness policy and realtime silence suppression should be
explicit, independent choices.

The native-rate join experiment makes independent SOLA decisions on a
different sample grid. Its waveform difference also includes those
decisions. Only the Applio round trip above isolates resampling alone; the
native-rate RMS comparison locks offsets to isolate the gain operation.

Artifacts are in
`target/compare/offline-dml-trt-python/dsp-ablation/`. `dsp-analysis.json`
contains the measurements. `applio-protect050.json` and
`applio-protect033.json` now record actual conversion controls and DSP.
`listening/` contains separate float32 copies with constant gain toward
RMS 0.1, limited by peak 0.9:

- `applio-native.wav` / `applio-linear-roundtrip.wav`: resampling alone.
- `applio-native.wav` / `applio-protect-ablation.wav`: protect alone.
- `native-current-dsp.wav`: reconstruction of the current DML file.
- `native-rate-dsp.wav`: same neural tails joined at 40 kHz, RMS retained.
- `native-rate-without-rms.wav`: same 40 kHz tails and SOLA offsets, RMS
  multiplier removed for diagnosis.

The last two are diagnostic postprocessing outputs; production engines
still use their current DSP policy.

## Additional differences against actual Applio realtime

Do not confuse the original offline file above with Applio realtime.
Its realtime UI's 250 ms chunk, 50 ms crossfade, 2.5 s extra context and
10 ms SOLA search match our 44960 preset. Its recent RMVPE window and
`[3:-1]` interior update agree with our rolling policy. Other behavior differs:

1. `rvc/realtime/core.py` resamples generated audio directly to the 48 kHz
   device domain with `torchaudio.transforms.Resample`. It does not use a
   16 kHz output bottleneck. It also uses the rolling RMS multiplier and
   the same `ceil(window/chunk)+1` muted startup policy.
2. `VoiceChanger.process_audio` defaults `use_phase_vocoder=True`. We use
   normalized time-domain SOLA and sin² crossfading. Applio adds frequency
   phase compensation after SOLA; its plain crossfade alternative also
   handles silence-to-speech onset separately. These can affect periodic
   boundary modulation independently of model precision.
3. Applio Synth's realtime `rate` path crops encoder output at the requested
   head **before** flow, with no 24-frame warmup. Our graph uses WebUI's
   `max(skip_head-24,0)` flow head and then crops to the decoder tail. The
   matching Python reference explicitly uses our WebUI policy. Near-unity
   Synth parity therefore does not prove identical Applio realtime cropping.
4. Applio realtime's `IndexWrapper` normally searches **all** reconstructed
   vectors using a PyTorch GPU distance matrix; CPU fallback uses FAISS with
   nprobe 12. Our native GPU retrieval preserves the stored IVF policy
   (nprobe 1 for this index). Applio realtime also retrieves only features
   starting at `skip_head // 2`, leaving older encoder context raw. Our
   current graph retrieves the complete HuBERT window. Applio offline uses
   stored FAISS IVF and retrieves the whole segment, as does our matched
   numerical reference. These are different search/context policies, even
   when the blend ratio and number of neighbors agree.
5. The checked Applio realtime snapshot quantizes mel-F0 against 50/1100,
   whereas its offline pipeline uses mel(50)/mel(1100). Our frontend uses
   the latter RVC formula. This source inconsistency does not apply to the
   saved offline reference and should not be copied without validation.
6. Applio exposes alternative embedders, F0 estimators and optional cleanup
   or effects. The checked realtime UI initializes the F0 selector to FCPE.
   These are not enabled in the RMVPE/HuBERT listening reference. A separate
   live-GUI comparison must record the actual selected settings.

The remaining TRT RMVPE threshold-crossing differences documented in
`offline-backend-comparison.md` are a separate arithmetic issue. DML also
has the DSP/config differences above despite much closer FP32-reference F0.

## Priority for subsequent alignment work

1. Separate model input rate, generated output rate and device rate in the
   common offline/engine I/O boundary. Keep HuBERT/RMVPE input at 16 kHz,
   retain Synth output at native rate, and resample once to a device rate
   with a stateful bandlimited resampler when required.
2. Make offline loudness preservation, optional RMS envelope mixing and
   realtime silence suppression explicit settings. Keep the existing
   realtime suppression available while comparing offline native-rate audio.
3. Hold protect, index policy and all pitch controls constant for listening
   comparisons. The original runner now accepts `--protect` and
   `--volume-envelope`, and records their effective values.
4. Separately evaluate a longer F0 window and streaming boundary/source
   continuity; profile any latency increase. Then compare optional phase
   vocoder joining on the same generated tails.
5. Continue precision/tactic tuning with exact-input numerical checks and
   audio listening under the corrected common DSP policy.

## Reproduce

```powershell
rtk .venv/Scripts/python.exe scripts/run-applio-offline.py `
  --input target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav `
  --output target/compare/offline-dml-trt-python/dsp-ablation/applio-protect033.wav `
  --checkpoint models/voices/default-v2/default.pth `
  --index models/voices/default-v2/added_IVF511_Flat_nprobe_1_default_v2.index `
  --protect 0.33 `
  --report target/compare/offline-dml-trt-python/dsp-ablation/applio-protect033.json
```

Repeat with protect 0.5 and corresponding `050` output/report names, then:

```powershell
rtk .venv/Scripts/python.exe scripts/analyze-offline-dsp.py `
  --applio target/compare/offline-dml-trt-python/dsp-ablation/applio-protect050.wav `
  --protect-ablation target/compare/offline-dml-trt-python/dsp-ablation/applio-protect033.wav `
  --input target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav `
  --trace target/compare/offline-dml-trt-python/dml-trace `
  --native-output target/compare/offline-dml-trt-python/dml.wav `
  --output-dir target/compare/offline-dml-trt-python/dsp-ablation
```

Verification: Python `py_compile` passed for both changed scripts, both
original FP32 Applio protect runs completed, and all DSP ablations passed
the actual-production reconstruction check. No Rust/native code changed;
there was no new native benchmark or subjective listening verdict.
