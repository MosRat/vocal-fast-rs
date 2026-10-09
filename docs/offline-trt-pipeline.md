# Offline TensorRT pipeline test

`model run-pipeline` now uses native-rate Applio offline conversion for
TensorRT/DirectML. See [alignment, controls, cache policy and reference
measurements](offline-applio-aligned.md). A complete VFRM supplies acoustic
weights automatically; an optional old pipeline manifest supplies precision
for rebuilding full-context offline plans. The output remains at the model
rate. CPU/passthrough retain the existing shared engine runner.

## Realtime callback validation from a file

The previous runner is retained as `model run-streaming-pipeline`, including
its playback-rate adaptation, history, RMS/SOLA and startup warmup:

```powershell
rtk cargo run -p vocal-fast --features tensorrt-rtx -- model run-streaming-pipeline tensorrt `
  models/voice.vfr testdata/fixtures/sine-440hz-16k-mono.wav `
  target/offline-output.wav target/pipeline/pipeline.json
```

The command performs the same four native stages as the realtime backend:

1. Decode WAV PCM or IEEE-float samples (or raw little-endian `f32`).
2. Downmix to mono and linearly resample the input to the internal 16 kHz
   HuBERT/RMVPE rate.
3. Advance the input in the selected playback blocks (4000 samples for the
   WebUI/Applio preset) while retaining the profile-sized context, then run
   HuBERT, RMVPE, frontend, and the trailing Synth return window for every
   block.
4. Write the generated Synth waveform as mono PCM16 WAV at the model's native
   output sample rate.

The command prints JSON containing input/output lengths, window count, elapsed
time, and engine statistics. Short inputs are padded to the playback block
and the final output block is cropped back to the valid input duration. The
TensorRT context remains the profile maximum (44960 for WebUI/Applio), so
inputs longer than one context are processed as consecutive streaming windows.

The same runner supports `passthrough` and `cpu`; only TensorRT needs the
final pipeline manifest argument.

With the 44960 profile, each callback advances 4000 input samples. The retained
context produces 281 RVC frames; Synth decodes only the last 31 frames after
`skip_head=250`. At 40 kHz that is 12400 samples (310 ms), from which the
adapter performs RMS envelope/SOLA/crossfade and publishes a 250 ms block.
Startup follows the reference warmup policy (13 silent blocks). Use the
augmented speech fixture with leading/trailing silence for listening tests;
short files can be entirely consumed during warmup.

`test-trt-pipeline` operates on one model context and provides timing/parity
diagnostics. `run-trt-pipeline` is a legacy diagnostic that concatenates full
contexts; it now rejects `synth_tail_mode=true` artifacts. It is not the
streaming file runner and does not implement its boundary treatment.

See [the retrieval regression report](retrieval-audio-regression.md) for
matching-model audio, Python comparisons and reproduction commands.
