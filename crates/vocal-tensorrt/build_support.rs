//! Build-time deployment helpers for the Windows TensorRT-RTX runtime.

use std::{
    env, fs,
    path::{Path, PathBuf},
};

const TRT_DLLS: &[&str] = &["tensorrt_rtx_1_6.dll", "tensorrt_onnxparser_rtx_1_6.dll"];

// TensorRT-RTX loads these CUDA libraries lazily while selecting tactics. We
// copy the small runtime plus the math/JIT libraries commonly used by RTX
// plans. Optional entries are skipped when a CUDA installation does not ship
// that component, which keeps the builder compatible with minor CUDA updates.
const CUDA_DLL_PATTERNS: &[&str] = &[
    "cudart64_*.dll",
    "cufft64_*.dll",
    "cufftw64_*.dll",
    "cublas64_*.dll",
    "cublasLt64_*.dll",
    "nvrtc64_*.dll",
    "nvrtc-builtins64_*.dll",
    "nvJitLink_*.dll",
    "nvfatbin_*.dll",
];

pub fn enabled() -> bool {
    env::var("VOCAL_TRT_COPY_DLLS")
        .ok()
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "0" | "false" | "off" | "no" => Some(false),
            "1" | "true" | "on" | "yes" => Some(true),
            _ => None,
        })
        .unwrap_or(true)
}

pub fn deploy(tensorrt_root: &Path, cuda_root: &Path, out_dir: &Path) {
    let destination = env::var_os("VOCAL_TRT_DLL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| profile_output_dir(out_dir));
    if let Err(error) = fs::create_dir_all(&destination) {
        panic!(
            "failed to create TensorRT DLL deployment directory {}: {error}",
            destination.display()
        );
    }

    let trt_bin = tensorrt_root.join("bin");
    for name in TRT_DLLS {
        copy_required(&trt_bin.join(name), &destination);
    }

    let cuda_bin = cuda_root.join("bin").join("x64");
    for pattern in CUDA_DLL_PATTERNS {
        for source in matching_dlls(&cuda_bin, pattern) {
            copy_optional(&source, &destination);
        }
    }
    println!(
        "cargo:warning=TensorRT runtime DLLs deployed to {}",
        destination.display()
    );
}

fn profile_output_dir(out_dir: &Path) -> PathBuf {
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".into());
    out_dir
        .ancestors()
        .find(|path| {
            path.file_name()
                .is_some_and(|name| name == profile.as_str())
        })
        .map(Path::to_path_buf)
        .unwrap_or_else(|| out_dir.to_path_buf())
}

fn matching_dlls(directory: &Path, pattern: &str) -> Vec<PathBuf> {
    let Some(prefix) = pattern.strip_suffix("*.dll") else {
        return directory
            .join(pattern)
            .is_file()
            .then(|| directory.join(pattern))
            .into_iter()
            .collect();
    };
    let Ok(entries) = fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut matches: Vec<_> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(prefix) && name.ends_with(".dll"))
        })
        .collect();
    matches.sort();
    matches
}

fn copy_required(source: &Path, destination: &Path) {
    if !source.is_file() {
        panic!(
            "required TensorRT runtime DLL is missing: {}",
            source.display()
        );
    }
    copy_optional(source, destination);
}

fn copy_optional(source: &Path, destination: &Path) {
    let target = destination.join(source.file_name().expect("DLL has a file name"));
    let source_size = fs::metadata(source)
        .unwrap_or_else(|error| panic!("failed to stat source DLL {}: {error}", source.display()))
        .len();
    let should_copy = fs::metadata(&target)
        .map(|metadata| metadata.len() != source_size)
        .unwrap_or(true);
    if !should_copy {
        return;
    }
    if let Err(error) = fs::copy(source, &target) {
        panic!(
            "failed to copy TensorRT runtime DLL {} -> {}: {error}",
            source.display(),
            target.display()
        );
    }
}
