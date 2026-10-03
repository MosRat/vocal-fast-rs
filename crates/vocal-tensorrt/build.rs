use std::{env, path::PathBuf};
fn main() {
    let r = env::var_os("TENSORRT_RTX_ROOT")
        .or_else(|| env::var_os("TRT_RTX_ROOT"))
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
}
