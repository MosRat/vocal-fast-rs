# Windows runtime packaging

## Build matrix

Use PowerShell 7, MSVC C++ tools and the Windows SDK. Run the helper from the
repository root:

```powershell
# Default Release installer: TensorRT RTX + DirectML + CPU inference.
rtk proxy pwsh -NoProfile -File scripts/build-windows.ps1 -Profile Release

# Embedded frontend Debug instance, including both GPU backends.
rtk proxy pwsh -NoProfile -File scripts/build-windows.ps1 -Profile Debug -NoBundle -Launch -InspectPort 9223

# DML without the CUDA/TensorRT SDKs.
rtk proxy pwsh -NoProfile -File scripts/build-windows.ps1 -Backend DirectMl

# No GPU SDK requirement.
rtk proxy pwsh -NoProfile -File scripts/build-windows.ps1 -Backend Cpu

# Place build outputs, runtime staging and logs on a drive with enough space.
rtk proxy pwsh -NoProfile -File scripts/build-windows.ps1 -Profile Release `
  -TargetDirectory $env:LOCALAPPDATA/vocal-fast/build-release
```

| Backend argument | Explicit Cargo features | GPU redistributables |
| --- | --- | --- |
| `TensorRt` (default) | `tensorrt-rtx,directml,cpu-graph` | TensorRT RTX, ONNX parser, CUDA runtime, cuFFT, DirectML, oneDNN |
| `DirectMl` | `directml,cpu-graph` | DirectML, oneDNN |
| `Cpu` | `cpu-graph` | None; standalone CPU `dnnl.dll` |

The helper passes `--no-default-features` to **Cargo**, after Tauri's `--`
separator. Tauri's own build CLI does not accept that flag. Plain Cargo and
`tauri dev` builds of the GUI enable DirectML and CPU inference by default; run
`scripts/setup-directml.ps1` and `scripts/setup-cpu.ps1` first for those entry points. The CLI's GPU
features remain explicit.

## Native libraries

Every distribution includes standalone oneDNN 3.10.2 with THREADPOOL runtime and no GPU
primitives. `setup-cpu.ps1` builds its SDK in `%LOCALAPPDATA%/vocal-fast/cpu-operators/onednn`.
`-CpuRoot`, `VOCAL_CPU_ONEDNN_ROOT` or `ONEDNN_ROOT` selects another compatible SDK.
CPU deployment copies `bin/dnnl.dll` and `share/doc/dnnl/{LICENSE,THIRD-PARTY-PROGRAMS}`.
RustFFT and local SIMD kernels are statically linked. Torch/c10/OpenMP DLLs are
no longer CPU runtime dependencies. See [CPU migration](cpu-standalone-migration.md).

DirectML is pinned to the Microsoft.AI.DirectML 1.15.4 redistributable package.
The NuGet archive and DirectMLX header are checked against fixed SHA-256
values. Setup repairs an installation that retains its headers but is missing
DLLs, the import library or notices. The Windows helper provisions the default
SDK when necessary; `-DirectMlRoot` or `DIRECTML_ROOT` selects a supplied SDK.

| File | Release | Debug | Destination |
| --- | --- | --- | --- |
| `DirectML.dll` | Yes | Yes | Executable directory |
| `DirectML.Debug.dll` | No | Yes | Executable directory |
| `LICENSE.txt`, `LICENSE-CODE.txt`, `ThirdPartyNotices.txt` | Yes | Yes | `licenses/directml/` |
| `DirectML.lib`, headers, PDBs | No | No | Build SDK only |
| D3D12, DXGI, GPU driver libraries | System | System | Supplied by Windows/driver |

Debug deployment does not enable the native DML validation layer automatically;
that remains controlled by `DirectMlConfig.debug_layer`.

TensorRT-RTX packages are built from an authorized TensorRT-RTX archive and a
CUDA installation. In GitHub Actions the archive is downloaded and verified in
an Ubuntu container, then CUDA is installed on the hosted Windows runner. For
local builds, the helper resolves
`TENSORRT_RTX_ROOT`/`TRT_RTX_ROOT`/`TENSORRT_ROOT`/`TRT_ROOT` and `CUDA_PATH`
before copying the runtime DLLs. It also requires and stages the vendor notice
files so the installer carries the applicable terms:

| SDK | Runtime files | Notice destination |
| --- | --- | --- |
| TensorRT-RTX | `tensorrt_rtx_1_6.dll`, `tensorrt_onnxparser_rtx_1_6.dll` | `licenses/tensorrt-rtx/` |
| CUDA | `cudart64_*.dll`, `cufft64_*.dll` | `licenses/cuda/` |

TensorRT-RTX notices are `doc/Acknowledgements.txt` and `doc/README.txt`;
CUDA notices are `EULA.txt`, `LICENSE`, and `README`. The release manifest
records their sizes and SHA-256 hashes. Do not enable TensorRT distribution on
a runner unless the installed SDK licenses permit redistribution of these
runtime files and notices.

The Cargo build script places DML DLLs beside executables, tests and examples.
The Windows helper additionally refreshes the loose executable's runtime files
after building, even when Cargo reused a cached build script. Files whose hashes
already match are left in place, including libraries mapped by a running process.

Installers use an explicit resource allowlist, staged separately for each
profile/backend. Reusing a Cargo target directory can leave other builds' DLLs
beside a loose executable; those files are not added to the installer. Weights,
compiled plans and SDK directories are not bundled. The existing dynamic MSVC
CRT policy is retained and NSIS includes the VC runtime installer.

`target/windows-build/<Profile>-<Backend>/artifacts.json` records actual Cargo
features, resource destinations, runtime hashes and executable/installer hashes.
With `-TargetDirectory` (or `CARGO_TARGET_DIR`), the helper puts runtime staging
and build reports beneath that directory as well. The default path is unchanged.
For a target directory on another drive, the helper creates a uniquely named
junction under the ignored Tauri target directory. Resource paths stay relative
because Tauri's resource normalization removes Windows drive prefixes; the
runtime files themselves remain on the selected drive.

## Latest package: 2026-10-08

Rebuilt at 23:53 UTC+08:00 with `tensorrt-rtx,directml,cpu-graph`, including
the shared speech onset/release joining, aligned RMS envelope matching and
TensorRT streaming input resampling fixes. See
[real-time speech transitions](realtime-speech-transitions.md) for replay
results and their limits. Automatic pitch matching remains enabled by default.

- Installer: `$env:LOCALAPPDATA/vocal-fast/build-release/release/bundle/nsis/vocal-fast_0.1.0_x64-setup.exe`.
- Size: 204,044,065 bytes (194.59 MiB).
- SHA-256: `BD62D3A59481AD9A204747C3B378956E05E37E41AFA491FE18ADA7624514DDB2`.
- Root workspace: formatting and all-feature checking passed; all-feature
  library, binary and integration tests passed (241 tests, 19 ignored).
  Tauri all-feature library tests passed (26, one ignored). Frontend tests
  passed (30 tests); production frontend built. The new InnerProduct test
  initially loaded a stale oneDNN DLL from the test target directory; refreshing
  it from the updated SDK resolved the failure. This package includes that SDK's
  runtime (18,179,072 bytes).
- Extracted installer verification: all six runtime DLL hashes, three DirectML
  notices and two oneDNN notices match; the packaged CPU DLL loads with Windows
  DLL-directory/system search and reports oneDNN 3.10.2. No DML debug DLL,
  Torch/MKL/OpenMP DLL, weights or plans are bundled. The
  packaged executable matches the compiled Release executable except for the
  three-byte NSIS bundle marker. PE imports
  include DirectML, TensorRT RTX, ONNX parser, cuFFT and oneDNN. All six
  redistributed DLLs load using restricted package-directory/system search.
- Build and verification manifests:
  `$env:LOCALAPPDATA/vocal-fast/build-release/windows-build/Release-TensorRt/artifacts.json`
  and `package-verification.json`. Build outputs and extraction use C: because
  the repository drive has insufficient space. No GUI launch, installation or
  hardware loopback was performed for this packaging check.

## Validation on 2026-10-07

Test machine: Windows x64, NVIDIA GeForce RTX 3060.

| Check | Result |
| --- | --- |
| PowerShell parsing; incomplete SDK repair | Passed; restored DLLs/import library/notices match the pinned SDK |
| `rtk cargo fmt --all -- --check` | Passed for the root workspace |
| `rtk cargo check --workspace --all-features` | Passed |
| `rtk cargo test --workspace` | 141 passed, 9 ignored |
| Tauri library tests with default DML + `tensorrt-rtx` | 22 passed |
| TypeScript checking and frontend build | Passed in both builds |
| Frontend regressions | 15 passed; configuration drafts, concurrent edits, startup events and errors |
| Debug native PE import and module path | Imports `DirectML.dll`; loads it from the Debug executable directory |
| Debug backend discovery/model loading | DML available on RTX 3060; complete VFRM reaches Ready (5.482 s) |
| Release NSIS extraction | Five GPU DLL hashes and three notice hashes match; DML debug DLL absent |
| Extracted Release, with SDK variables removed and system-only PATH | Starts successfully; DML available; saved complete VFRM restored to Ready; user configuration preserved |
| Extracted Release module path | Loads `DirectML.dll` from the extracted application's directory |
| Latest Release UI | Single automatic-save status fixed in the sidebar footer; event-driven bootstrap; DirectML engine Ready; saved configuration unchanged |

The existing real DML worker integration test was also run explicitly:

```powershell
rtk cargo test -p vocal-runtime --features directml bundle_loading_worker_handoff_and_small_callbacks -- --ignored --nocapture --test-threads=1
```

It loaded the complete VFRM, transferred the engine to the worker, processed
433,600 samples in callbacks of 160 samples, and returned/unloaded the engine.
All output samples were finite, output length matched input and output energy
was nonzero (13.975760). Processing took 4.556 s excluding model preparation.
This validates GPU inference and ownership handoff; it does not measure live
microphone-to-speaker latency. Other vendors' GPUs were not tested on this machine.

The Release installer is
`apps/tauri/src-tauri/target/release/bundle/nsis/vocal-fast_0.1.0_x64-setup.exe`:

- Rebuilt: 2026-10-07 20:57:36 UTC+08:00, including event-driven startup,
  workspace model/engine preparation, typed audio startup recovery and the
  fixed sidebar automatic-save indicator.
- Size: 200,207,778 bytes (190.93 MiB).
- SHA-256: `561917D6319EC19F0DC832CA73B33081559FE005F02F2B19A0F6336B7BD78428`.
- DML runtime SHA-256: `9C9E6D822561C6C41B90E6994B3E8857CF1D66DBFB1E0C4C799C7C89B4E92DA1`.

The extracted executable differs from the loose executable only in Tauri's
three-byte bundle type marker (`UNK` -> `NSS`). All other bytes are identical;
installer/executable comparisons and all five GPU DLL hashes were verified.
The clean Release run reached shell readiness in 1,131 ms and restored the saved
DirectML model to Ready without changing its configuration. The smoke process
was stopped after verification; the existing Debug instance remained running.

The original model, selected DirectML backend and automatic GPU selection were
preserved. Detailed manifests, device
catalogs, model-load snapshots, extracted module paths and the SDK repair report
are retained under `target/windows-build/`; generated files remain untracked.
