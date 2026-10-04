# Windows allocator and linker policy

The executable targets use mimalloc at the binary boundary:

- `vocal-fast` enables `mimalloc` with `win_direct_tls` on Windows. The CLI is
  a controlled process with a small TLS footprint, so it can use the faster
  direct-TLS path.
- The Tauri executable uses the same allocator without `win_direct_tls`.
  WebView2 and shell integrations can consume many Windows TLS slots, so the
  fallback-aware mode is safer for the GUI.
- Library crates stay allocator-neutral so embedders can provide another
  allocator without conflicting global symbols.
- The real-time audio callbacks must still avoid allocation; the allocator is
  for control-plane, model, and UI work.

Rust and the native TensorRT/CUDA objects use the dynamic MSVC CRT (`/MD`).
The root `.cargo/config.toml` makes this explicit with
`target-feature=-crt-static`; the CUDA build already passes `/MD`. This keeps
ownership on one CRT family when buffers cross the Rust/C++ boundary.

The Windows target uses Rust's bundled `rust-lld` linker. Development builds
use line-table debug information, incremental compilation, and high codegen
parallelism. Release builds use thin LTO, 16 codegen units, aborting panics,
and symbol stripping. Thin LTO keeps cross-crate inlining while avoiding the
long link times of fat LTO.

## Local measurement

Run from the repository root after the first dependency build:

```powershell
rtk cargo build -p vocal-fast --no-default-features
rtk cargo build -p vocal-fast --no-default-features --release
rtk cargo build --manifest-path apps/tauri/src-tauri/Cargo.toml
rtk cargo build --manifest-path apps/tauri/src-tauri/Cargo.toml --release
```

Use PowerShell's `Measure-Command` around each command when comparing linker
or profile changes. Keep `target/` and Tauri's target directory out of Git.

On the development machine used for this change (`x86_64-pc-windows-msvc`,
Rust 1.98, warm dependency cache), the resulting builds measured about 0.4 s
for an incremental CLI build, 2.7 s for an incremental Tauri build, 15.5 s
for a clean CLI release build, and 92.5 s for a clean Tauri release build.
These are reference measurements rather than a cross-machine benchmark.
