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
- matching x64 CUDA runtime, cuFFT, cuBLAS/cuBLASLt, and CUDA JIT DLLs

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
