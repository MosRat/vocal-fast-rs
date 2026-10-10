# vocal-fast-rs

Rust workspace for a real-time RVC voice converter. The repository contains a
shared model format, hand-built graphs, streaming audio boundaries, and three
execution backends:

- **CPU**: portable Rust graph execution with fused operators and optional
  calibrated quantization.
- **DirectML**: Windows GPU execution for supported DirectX 12 adapters.
- **TensorRT-RTX**: native CUDA/TensorRT-RTX graphs for RTX GPUs. This backend
  is optional and requires the TensorRT-RTX and CUDA SDKs on the build host.

The Tauri application and command-line tools use the same runtime and model
contracts. RVC v1 and v2 are selected from the model manifest; callers do not
choose a feature width independently.

## Repository layout

| Path | Responsibility |
| --- | --- |
| `crates/vocal-types` | Public audio, conversion, and runtime types |
| `crates/vocal-audio` | CPAL/WASAPI capture, rings, resampling, timing |
| `crates/vocal-model` | PTH/ONNX import, VFRM cache, manifests, retrieval |
| `crates/vocal-graph` | Source-independent graph IR and model builders |
| `crates/vocal-cpu` | CPU graph planner, fused kernels, and execution |
| `crates/vocal-directml` | DirectML graph compilation and execution |
| `crates/vocal-tensorrt` | CUDA/TensorRT-RTX FFI, native graphs, and replay |
| `crates/vocal-engine` | Backend-neutral engine boundary and streaming adapter |
| `crates/vocal-runtime` | Model lifecycle, backend selection, and diagnostics |
| `crates/vocal-cli` | `vocal-fast` command-line entry point |
| `apps/tauri` | Windows desktop application |
| `docs` | Architecture, model format, parity, and tuning notes |
| `scripts` | Reproducible conversion and analysis helpers |

## Quick start

The core workspace can be checked without a GPU SDK:

```powershell
cargo fmt --all -- --check
cargo check --workspace --exclude vocal-cpu --exclude vocal-tensorrt
cargo test --workspace --exclude vocal-cpu --exclude vocal-tensorrt
cargo run -p vocal-fast -- doctor
cargo run -p vocal-fast -- demo 8
```

The same SDK-independent checks can be run on Windows with PowerShell 7:

```powershell
pwsh -NoProfile -File scripts/verify.ps1
pwsh -NoProfile -File scripts/verify.ps1 -Tauri
```

Use `-Cpu` when the standalone oneDNN SDK is installed. Use `-AllFeatures`
only on a host that also has the optional TensorRT-RTX SDK. The normal check
intentionally excludes both native GPU and oneDNN builds so a clean checkout
can be verified on Linux, macOS, or Windows without vendor installations.

To use the Python conversion and parity helpers, install the analysis
environment with [uv](https://docs.astral.sh/uv/):

```powershell
uv sync
uv run python -m compileall scripts
```

The checked-in model manifests and the small sine fixture are only structural
fixtures. Downloaded checkpoints, HuBERT/RMVPE weights, FAISS indexes, VFRM
packages, and generated WAV files stay outside Git. See
[`models/README.md`](models/README.md) and [`testdata/README.md`](testdata/README.md).

## RVC model contract

VFRM is the source-independent intermediate format. Conversion from PTH or ONNX
normalizes tensor names, layouts, dtypes, and metadata before any backend sees
the model.

| Version | HuBERT layer | Synth feature width | Extra projection |
| --- | ---: | ---: | --- |
| RVC v1 | 9 | 256 | learned `hubert.model.final_proj` (`768 -> 256`) |
| RVC v2 | 12 | 768 | none |

The same `RvcVersion` contract is checked by CPU, DirectML, TensorRT, model
bundling, retrieval indexes, and the runtime. See
[`docs/rvc-version-and-fusion.md`](docs/rvc-version-and-fusion.md) and
[`docs/vfrm-format.md`](docs/vfrm-format.md).

## Optional GPU backends

### DirectML

DirectML is Windows-only and uses the system DirectX 12 runtime. Build the CLI
or Tauri app with the `directml` feature where needed:

```powershell
cargo check -p vocal-fast --features directml
```

### TensorRT-RTX

TensorRT-RTX is intentionally kept behind an optional feature because its SDK
and CUDA toolchain are not redistributable with this repository. Set
`TENSORRT_RTX_ROOT` (or `TRT_RTX_ROOT`) and `CUDA_PATH` before building:

```powershell
$env:TENSORRT_RTX_ROOT = 'C:\Program Files\NVIDIA Corporation\TensorRT-RTX\1.6.1.120'
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'
cargo check -p vocal-fast --features tensorrt-rtx
```

The native build links only the CUDA runtime, cuFFT, TensorRT-RTX, and the
TensorRT-RTX ONNX parser libraries. SDK DLLs and generated plans are local
build artifacts and must not be committed.

For a Windows Debug GUI instance or a packaged release, use the repository
build helper. It discovers the selected SDKs, stages only the required runtime
DLLs, writes a hash manifest under `target/windows-build/`, and keeps model
weights outside the repository:

```powershell
pwsh -NoProfile -File scripts/build-windows.ps1 -Profile Debug `
  -Backend TensorRt -NoBundle -Launch
pwsh -NoProfile -File scripts/build-windows.ps1 -Profile Release `
  -Backend Cpu
```

See [`docs/windows-runtime-packaging.md`](docs/windows-runtime-packaging.md)
for the backend matrix, SDK discovery rules, runtime staging, and installer
verification.

## Tauri application

The frontend is in `apps/tauri` and uses the same runtime configuration exposed
by the CLI. On Windows:

```powershell
cd apps/tauri
npm ci
npm run build
```

Native packaging additionally requires the Rust Windows toolchain and the
selected backend SDKs. See [`docs/windows-runtime-packaging.md`](docs/windows-runtime-packaging.md).

## Verification

The core CI checks formatting, tests, and the frontend type/build step. GPU
plans and downloaded weights are validated by local, hardware-specific probes;
their reports belong under `target/` and are ignored by Git.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).

The repository does not redistribute NVIDIA, DirectML, HuBERT, RMVPE, FAISS, or
voice-model weights. Their licenses remain applicable to the corresponding
external assets.
