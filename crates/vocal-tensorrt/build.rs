mod build_support;

use std::{env, path::PathBuf, process::Command};

fn main() {
    for variable in [
        "TENSORRT_RTX_ROOT",
        "TRT_RTX_ROOT",
        "TENSORRT_ROOT",
        "TRT_ROOT",
        "CUDA_PATH",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    let r = env::var_os("TENSORRT_RTX_ROOT")
        .or_else(|| env::var_os("TRT_RTX_ROOT"))
        .or_else(|| env::var_os("TENSORRT_ROOT"))
        .or_else(|| env::var_os("TRT_ROOT"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(r"C:\Program Files\NVIDIA Corporation\TensorRT-RTX\1.6.1.120")
        });
    let c = env::var_os("CUDA_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4")
        });
    for file in [
        "native/bridge.h",
        "native/nvtx.h",
        "native/common.cpp",
        "native/cuda.cpp",
        "native/runtime.cpp",
        "native/onnx.cpp",
        "native/hubert.cpp",
        "native/rmvpe.cpp",
        "native/frontend.cpp",
        "native/serialization.cpp",
        "native/plugin.cpp",
        "native/retrieval.cu",
        "native/cufft_frontend.cu",
        "native/graph/layers.h",
        "native/graph/layers.cpp",
        "native/graph/encoder.cpp",
        "native/graph/flow.cpp",
        "native/graph/decoder.cpp",
        "native/graph/build.cpp",
    ] {
        println!("cargo:rerun-if-changed={file}");
    }
    println!("cargo:rustc-link-search=native={}", r.join("lib").display());
    println!(
        "cargo:rustc-link-search=native={}",
        c.join("lib/x64").display()
    );
    println!("cargo:rustc-link-lib=dylib=tensorrt_rtx_1_6");
    println!("cargo:rustc-link-lib=dylib=tensorrt_onnxparser_rtx_1_6");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cufft");
    println!("cargo:rerun-if-env-changed=VOCAL_TRT_COPY_DLLS");
    println!("cargo:rerun-if-env-changed=VOCAL_TRT_DLL_DIR");
    if target_os_is_windows() && build_support::enabled() {
        let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
        build_support::deploy(&r, &c, &out_dir);
    }
    cc::Build::new()
        .cpp(true)
        .files([
            "native/common.cpp",
            "native/cuda.cpp",
            "native/runtime.cpp",
            "native/onnx.cpp",
            "native/hubert.cpp",
            "native/rmvpe.cpp",
            "native/frontend.cpp",
            "native/serialization.cpp",
            "native/plugin.cpp",
            "native/graph/layers.cpp",
            "native/graph/encoder.cpp",
            "native/graph/flow.cpp",
            "native/graph/decoder.cpp",
            "native/graph/build.cpp",
        ])
        .include(r.join("include"))
        .include(c.join("include"))
        .flag_if_supported("/std:c++17")
        .flag_if_supported("/EHsc")
        .compile("vocal_tensorrt_native");

    // cuFFT is a CUDA translation unit rather than a TensorRT plugin. Compile
    // it with nvcc into a small static library so the Rust crate only links a
    // stable C ABI. The plan and scratch allocations are created at runtime,
    // before any CUDA graph capture begins.
    let nvcc = c.join("bin/nvcc.exe");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let cufft_lib = out_dir.join("vocal_tensorrt_cufft.lib");
    let cl = find_msvc_cl().unwrap_or_else(|| {
        panic!("nvcc requires cl.exe; set CXX or run cargo from a VS developer shell")
    });
    let status = Command::new(&nvcc)
        .arg("-lib")
        .arg("-std=c++17")
        .arg("-ccbin")
        .arg(&cl)
        .arg("-Xcompiler=/EHsc")
        .arg("-Xcompiler=/MD")
        .arg("-I")
        .arg("native")
        .arg("-I")
        .arg(r.join("include"))
        .arg("-I")
        .arg(c.join("include"))
        .arg("native/cufft_frontend.cu")
        .arg("native/retrieval.cu")
        .arg("-o")
        .arg(&cufft_lib)
        .status()
        .unwrap_or_else(|error| panic!("failed to launch nvcc {}: {error}", nvcc.display()));
    if !status.success() {
        panic!("nvcc failed to build cuFFT frontend ({status})");
    }
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=vocal_tensorrt_cufft");
}

fn target_os_is_windows() -> bool {
    env::var("CARGO_CFG_TARGET_OS").is_ok_and(|value| value.eq_ignore_ascii_case("windows"))
}

fn find_msvc_cl() -> Option<PathBuf> {
    if let Some(value) = env::var_os("CXX") {
        let candidate = PathBuf::from(value);
        if candidate
            .file_stem()
            .is_some_and(|name| name.eq_ignore_ascii_case("cl"))
            && candidate.exists()
        {
            return Some(candidate);
        }
    }
    let roots = [
        PathBuf::from(r"C:\Program Files\Microsoft Visual Studio"),
        PathBuf::from(r"E:\VisualStudio"),
    ];
    for root in roots {
        if let Ok(versions) = std::fs::read_dir(root) {
            for version in versions.flatten() {
                let candidate = version
                    .path()
                    .join(r"VC\Tools\MSVC")
                    .join("14.51.36231")
                    .join(r"bin\Hostx64\x64\cl.exe");
                if candidate.exists() {
                    return Some(candidate);
                }
                if let Ok(tool_versions) = std::fs::read_dir(version.path().join(r"VC\Tools\MSVC"))
                {
                    for tool_version in tool_versions.flatten() {
                        let candidate = tool_version.path().join(r"bin\Hostx64\x64\cl.exe");
                        if candidate.exists() {
                            return Some(candidate);
                        }
                    }
                }
            }
        }
    }
    None
}
