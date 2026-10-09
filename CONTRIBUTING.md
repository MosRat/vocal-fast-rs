# Contributing

Thanks for helping improve vocal-fast-rs. Changes should keep the backend
boundary explicit and must not require downloaded model assets to build the
core workspace.

## Before opening a change

```powershell
cargo fmt --all -- --check
cargo check --workspace --exclude vocal-tensorrt
cargo test --workspace --exclude vocal-tensorrt
git diff --check
```

If you have the TensorRT-RTX SDK and CUDA installed, also run the relevant
TensorRT checks and record the SDK version and GPU in the pull request. DirectML
checks are Windows-only. Tauri changes should include:

```powershell
cd apps/tauri
npm ci
npm test
npm run build
```

## Change boundaries

Keep commits focused on one module: model/VFRM, graph, CPU, DirectML,
TensorRT, audio, runtime, CLI, GUI, or analysis tooling. Use imperative commit
subjects such as `feat(model): validate v1 projection metadata`.

Do not commit `target/`, downloaded checkpoints, VFRM packages, TensorRT plans,
reference checkouts, virtual environments, or generated audio. Small synthetic
fixtures and reproducible scripts belong in the repository; large assets belong
in the local model cache described by `models/README.md`.

## Model and numerical changes

Document the source model, RVC version, input shape, dtype, backend, and parity
reference. A performance claim should include the hardware and whether CUDA
graph capture, retrieval, and diagnostic readback were enabled. Keep v1 and v2
contracts in `vocal-model::RvcVersion`; do not add independent string-based
version tests in a backend.

## Reporting issues

Include the operating system, Rust version, backend, model manifest, window
profile, and a short reproduction command. Do not attach proprietary weights or
API keys. A redacted diagnostic report under `target/` is usually sufficient.
