# TensorRT-RTX build and runtime deployment

The `vocal-tensorrt` build script links the TensorRT-RTX and CUDA import
libraries and, on Windows, copies the runtime DLLs next to the Cargo profile
output. This keeps `cargo run`, CLI binaries, and packaged development builds
independent of the machine-wide `PATH`.

## Default build

```powershell
rtk cargo build -p vocal-fast --features tensorrt-rtx
```

The default deployment directory is the profile directory discovered from
`OUT_DIR`, normally `target\debug` or `target\release`. The script copies:

- `tensorrt_rtx_1_6.dll`
- `tensorrt_onnxparser_rtx_1_6.dll`
- matching x64 `cudart`, `cuFFT`, cuBLAS/cuBLASLt, and NVRTC/NVRTC-builtins
  DLLs

The current VFRM graph directly links `cudart` and `cuFFT`. TensorRT-RTX can
discover cuBLAS/cuBLASLt and NVRTC lazily: the former is an optional math
backend, while the latter is used by source-code JIT tactics and dynamic kernel
specialization. They are retained in the deployment set so a serialized engine
can use those tactics on a clean machine. A fixed plan built with only the
edge-mask tactic also runs in the repository's isolated test without cuBLAS or
NVRTC, but omitting them would make other TensorRT-RTX plans/configurations
machine-dependent.

`cufftw` is not used by the native cuFFT C API path. `nvJitLink` and `nvfatbin`
are CUDA toolkit tooling libraries and are not loaded by the TensorRT-RTX
runtime or by its NVRTC path, so they are deliberately not copied.

Copies are idempotent and are skipped when the destination file has the same
size. The source roots are resolved from `TENSORRT_RTX_ROOT`/`TRT_ROOT` and
`CUDA_PATH`, with the installed TensorRT-RTX/CUDA 13.4 paths as defaults.

## Deployment controls

Use these build environment variables when packaging or cross-compiling:

```powershell
$env:VOCAL_TRT_DLL_DIR = "dist\vocal-fast"
$env:VOCAL_TRT_COPY_DLLS = "1" # set to 0/false to disable copying
rtk cargo build -p vocal-fast --release --features tensorrt-rtx
```

The destination directory must be included beside the executable in the final
package. The native build still links against the SDK import libraries; the
copy step only controls runtime discovery.

The isolated Tauri workspace exposes the same feature:

```powershell
rtk cargo build --manifest-path apps/tauri/src-tauri/Cargo.toml --features tensorrt-rtx
```

This feature propagates to `vocal-engine` and `vocal-runtime`, which pulls in
the same `vocal-tensorrt` build script and DLL deployment policy.
