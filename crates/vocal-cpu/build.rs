use std::{env, path::PathBuf};

fn main() {
    for name in ["VOCAL_CPU_ONEDNN_ROOT", "ONEDNN_ROOT", "LOCALAPPDATA"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let repo = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    let root = env::var_os("VOCAL_CPU_ONEDNN_ROOT")
        .or_else(|| env::var_os("ONEDNN_ROOT"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            env::var_os("LOCALAPPDATA")
                .map(PathBuf::from)
                .map(|p| p.join("vocal-fast/cpu-operators/onednn"))
                .unwrap_or_else(|| repo.join("target/cpu-sdk/onednn"))
        });
    assert!(
        root.join("include/oneapi/dnnl/dnnl.hpp").is_file(),
        "standalone oneDNN SDK missing: run scripts/setup-cpu.ps1 or set VOCAL_CPU_ONEDNN_ROOT (THREADPOOL runtime, GPU NONE)"
    );
    // An SDK can be rebuilt in place. Refresh the ABI build and deployed DLL
    // when its configuration or library changes, even at the same SDK path.
    for relative in [
        "include/oneapi/dnnl/dnnl_config.h",
        "lib/dnnl.lib",
        "bin/dnnl.dll",
    ] {
        let path = root.join(relative);
        if path.is_file() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
    let modules = [
        "runtime",
        "memory",
        "affinity",
        "tensor",
        "threadpool",
        "elements",
        "dnnl_ops",
        "convolution_matmul",
        "linear",
        "quantization",
        "normalization",
        "tensor_io",
        "attention",
        "attention_online",
        "gru",
        "stft",
    ];
    let mut build = cc::Build::new();
    let avx2 = env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64");
    if avx2 {
        build.define("VOCAL_CPU_AVX2", None);
    }
    build
        .cpp(true)
        .std("c++20")
        .opt_level(3)
        .include(root.join("include"))
        .define("NOMINMAX", None)
        .flag_if_supported("/EHsc")
        .flag_if_supported("/Zc:__cplusplus")
        .flag_if_supported("/bigobj");
    for module in modules {
        let path = format!("native/{module}.cpp");
        build.file(&path);
        println!("cargo:rerun-if-changed={path}");
    }
    build.compile("vocal_cpu_native");
    if avx2 {
        cc::Build::new()
            .cpp(true)
            .std("c++20")
            .opt_level(3)
            .file("native/simd.cpp")
            .file("native/attention_simd.cpp")
            .file("native/normalization_simd.cpp")
            .flag_if_supported("/arch:AVX2")
            .flag_if_supported("-mavx2")
            .flag_if_supported("-mfma")
            .compile("vocal_cpu_simd");
        println!("cargo:rustc-cfg=cpu_avx2_kernel");
    }
    println!("cargo:rustc-check-cfg=cfg(cpu_avx2_kernel)");
    for file in [
        "bridge.h",
        "internal.h",
        "affinity.h",
        "tensor.h",
        "threadpool.h",
        "simd.cpp",
        "attention_simd.cpp",
        "attention_simd.h",
        "normalization_simd.cpp",
    ] {
        println!("cargo:rerun-if-changed=native/{file}");
    }
    println!(
        "cargo:rustc-link-search=native={}",
        root.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=dnnl");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let source = root.join("bin/dnnl.dll");
        assert!(
            source.is_file(),
            "oneDNN runtime DLL missing: {}",
            source.display()
        );
        let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        let profile = out.ancestors().nth(3).unwrap();
        for dir in [
            profile.to_owned(),
            profile.join("deps"),
            profile.join("examples"),
        ] {
            std::fs::create_dir_all(&dir).unwrap();
            let dest = dir.join("dnnl.dll");
            let source_meta = source.metadata().unwrap();
            if dest.metadata().ok().is_some_and(|m| {
                m.len() == source_meta.len() && m.modified().ok() == source_meta.modified().ok()
            }) {
                continue;
            }
            if dest.exists() {
                std::fs::remove_file(&dest).unwrap();
            }
            std::fs::hard_link(&source, &dest)
                .or_else(|_| std::fs::copy(&source, &dest).map(|_| ()))
                .unwrap();
            if let Ok(modified) = source_meta.modified() {
                std::fs::File::options()
                    .write(true)
                    .open(&dest)
                    .unwrap()
                    .set_times(std::fs::FileTimes::new().set_modified(modified))
                    .unwrap();
            }
        }
    } else {
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            root.join("lib").display()
        );
    }
}
