# TensorRT-RTX engine backend

`vocal-engine` now exposes the native TensorRT-RTX pipeline through the same
`InferenceEngine` contract as the CPU backend.  The adapter is enabled with
`vocal-engine/tensorrt-rtx` (the CLI feature also enables the runtime feature).

## Ownership and data flow

```text
AudioChunk (interleaved, caller rate)
  -> validate + mono mix
  -> linear resample to 16 kHz
  -> fixed profile StreamingWindow (left zero pad, latest context)
  -> RvcTrtPipeline
       waveform H2D -> HuBERT/RMVPE/frontend (full context)
       HuBERT -> GPU IVF-Flat FAISS top-8 blend (optional)
       -> Synth (trailing skip_head/return_length frames) -> audio D2H
  -> model-rate resample to caller rate
  -> SOLA search + crossfade of the extended block
  -> exact callback crop + optional RMS mix
  -> channel interleave, preserving sequence
```

The adapter owns only the streaming window and public audio conversion. CUDA
streams, TensorRT contexts, device buffers, pinned rings, and CUDA Graph replay
remain owned by `RvcTrtPipeline`.  The complete engine is moved to the runtime
worker and is never accessed concurrently.

## Explicit construction

```rust
let config = EngineConfig::with_tensorrt(
    model_descriptor,
    "compiled/pipeline.json",
    "models/voice.vfrm",
);
let mut engine = vocal_engine::create_engine(config)?;
let output = engine.process(&input, ConversionConfig::default())?;
```

`pipeline.json` is the manifest emitted by `model compile-trt-pipeline`. The
VFRM package supplies the canonical Synth schema and is validated once at
startup; its tensor bytes are not copied into the realtime worker because the
compiled TensorRT plan owns the runtime weights.

For an already parsed package, use `TensorRtEngine::from_package` through
`TensorRtBackendConfig::for_package`. This is the path used by
`vocal-runtime`, so loading a model directory does not parse a second VFRM
copy.

## Runtime configuration

`RuntimeBackend::TensorRtRtx` requires:

- `tensorrtArtifactsPath`: the compiled pipeline manifest;
- a loaded VFRM model package;
- optional `tensorrtWindowSamples`, which must be inside the manifest profile
  and aligned to the 160-sample RVC hop;
- `speakerId` for multi-speaker Synth plans.

TensorRT runtime and diagnostic policy is parsed once through
`TensorRtConfig::from_env`, so graph mode, asynchronous audio D2H, and debug
switches use the same typed configuration as the CLI benchmark commands.

The VFRM package may contain `aux.retrieval.faiss`, a bounded RVC
`IndexIVFFlat,Flat,L2` payload. When present, TensorRT loads the index into a
CUDA-resident handle at startup. Each HuBERT frame searches the configured IVF
lists, takes the reference top-8 neighbors, applies inverse-square distance
weights, and blends in place before the native frontend. The ratio is held in
a one-element device buffer, so changing `ConversionConfig.index_ratio` does
not invalidate CUDA Graph replay. A non-zero ratio without a package index is
rejected. `pitch_semitones` is still rejected by this native frontend profile,
and the native protection value remains `0.33`. RMS mix is applied to the
returned block on the host, after the final D2H copy.
