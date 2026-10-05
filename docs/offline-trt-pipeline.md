# Offline TensorRT pipeline test

Build the CLI with the TensorRT-RTX feature, then run a complete file-to-file
inference pass:

```powershell
rtk cargo run -p vocal-fast --features tensorrt-rtx -- \
  model run-trt-pipeline \
  target\pipeline\pipeline.json \
  models\voice.vfr \
  testdata\fixtures\sine-440hz-16k-mono.wav \
  target\offline-output.wav
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
time, and CUDA Graph statistics. Short inputs are padded to the playback block
and the final output block is cropped back to the valid input duration. The
TensorRT context remains the profile maximum (44960 for WebUI/Applio), so
inputs longer than one context are processed as consecutive streaming windows.

The existing `test-trt-pipeline` command remains useful for JSON timing and
parity diagnostics; `run-trt-pipeline` is the offline artifact-producing path.
