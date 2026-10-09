use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=DIRECTML_ROOT");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    assert_eq!(
        env::var("CARGO_CFG_TARGET_ARCH").as_deref(),
        Ok("x86_64"),
        "the packaged DirectML backend currently supports Windows x64"
    );
    let root = env::var_os("DIRECTML_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
                .join("../../target/directml-sdk/package")
        });
    assert!(
        root.join("include/DirectMLX.h").is_file(),
        "DirectML SDK missing: run scripts/setup-directml.ps1 or set DIRECTML_ROOT"
    );
    let files = [
        "native/device.cpp",
        "native/bindings.cpp",
        "native/runtime.cpp",
        "native/transfer.cpp",
        "native/state.cpp",
        "native/graph.cpp",
        "native/layout.cpp",
        "native/layout.h",
        "native/bridge.h",
    ];
    for file in files {
        println!("cargo:rerun-if-changed={file}");
    }
    cc::Build::new()
        .cpp(true)
        .files(files.iter().filter(|file| file.ends_with(".cpp")))
        .include(root.join("include"))
        .flag_if_supported("/std:c++17")
        .flag_if_supported("/Zc:__cplusplus")
        .flag_if_supported("/EHsc")
        .define("DML_TARGET_VERSION", "0x6400")
        .define("NOMINMAX", None)
        .compile("vocal_directml_native");
    println!(
        "cargo:rustc-link-search=native={}",
        root.join("bin/x64-win").display()
    );
    for lib in ["DirectML", "d3d12", "dxgi", "user32"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let profile = out.ancestors().nth(3).expect("cargo profile directory");
    for dest in [
        profile.to_owned(),
        profile.join("deps"),
        profile.join("examples"),
    ] {
        std::fs::create_dir_all(&dest).unwrap();
        deploy(
            &root.join("bin/x64-win/DirectML.dll"),
            &dest.join("DirectML.dll"),
        );
        // debug_layer is a runtime choice, including release validation runs.
        let debug = root.join("bin/x64-win/DirectML.Debug.dll");
        if debug.is_file() {
            deploy(&debug, &dest.join("DirectML.Debug.dll"));
        }
    }
}

fn deploy(source: &std::path::Path, destination: &std::path::Path) {
    let data = std::fs::read(source).expect("read DirectML runtime DLL");
    if std::fs::read(destination).ok().as_deref() != Some(data.as_slice()) {
        std::fs::write(destination, data).expect("deploy DirectML runtime DLL");
    }
}
