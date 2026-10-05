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
3. Pad and split the input into the selected TensorRT profile windows, then
   run HuBERT, RMVPE, frontend, and Synth for every window.
4. Write the generated Synth waveform as mono PCM16 WAV at the model's native
   output sample rate.

The command prints JSON containing input/output lengths, window count, elapsed
time, and CUDA Graph statistics. Short inputs are padded to the profile's
optimized shape and the padded tail is cropped from the saved result. Inputs
longer than the profile maximum are processed as consecutive windows.

The existing `test-trt-pipeline` command remains useful for JSON timing and
parity diagnostics; `run-trt-pipeline` is the offline artifact-producing path.
