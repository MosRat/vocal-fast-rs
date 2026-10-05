# Tauri TensorRT pipeline lifecycle

The GUI treats TensorRT plans as a prepared runtime artifact. It does not
write plans beside a model or into the repository. On startup the Tauri shell
sets the default cache root to:

```text
<app-cache>/vocal-fast/tensorrt/
```

Each model and build policy gets an isolated directory:

```text
<app-cache>/vocal-fast/tensorrt/<model-id>/<profile>-<fingerprint>/
  hubert.plan
  rmvpe-realtime.plan
  rmvpe-full.plan
  frontend.plan
  synth.plan
  pipeline.json
```

The fingerprint includes the model, HuBERT and RMVPE source metadata, profile,
precision policy, graph mode and FFT mode. A cache hit requires the manifest to
parse and every plan named by it to exist. Invalid or incomplete directories
are rebuilt into a temporary sibling and published by an atomic directory
rename.

The `prepare_tensorrt_pipeline` command is serialized with model and audio
commands. It rejects an active audio session, checks the optimistic config
revision, reports `building`, `ready`, or `failed` in the snapshot, and only
switches the runtime backend to TensorRT after all plans and the manifest are
ready. The persisted runtime config stores both the typed TensorRT policy and
the artifact path; the older top-level artifact fields remain as a migration
fallback for existing config files.

Build the Tauri application with the TensorRT feature to enable native plan
generation:

```powershell
rtk cargo tauri build --manifest-path apps/tauri/src-tauri/Cargo.toml --features tensorrt-rtx
```

Without that feature the controls remain part of the shared GUI protocol, but
the command returns a recoverable engine error rather than attempting a CPU or
ONNX fallback.
