//! Rebuild an isolated fixed FP32 RMVPE graph for accuracy investigations.
//! build_rmvpe <rmvpe.vfr> <output.plan> <samples> [cufft|dft] [debug]
use std::error::Error;
use vocal_tensorrt::{NativeBuildOptions, NativeTensorRtBuilder};

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(3..=5).contains(&args.len()) {
        return Err("build_rmvpe <rmvpe.vfr> <output.plan> <samples> [cufft|dft] [debug]".into());
    }
    let samples: i32 = args[2].parse()?;
    if samples < 1024 {
        return Err("RMVPE requires at least 1024 waveform samples".into());
    }
    let cufft = match args.get(3).map(String::as_str).unwrap_or("cufft") {
        "cufft" => true,
        "dft" => false,
        _ => return Err("FFT mode must be cufft or dft".into()),
    };
    let package = vocal_model::load_cache_shared(&args[0])?;
    let options = NativeBuildOptions {
        output: args[1].clone().into(),
        min_samples: samples,
        opt_samples: samples,
        max_samples: samples,
        rmvpe_fft_cufft: cufft,
        rmvpe_debug_outputs: args.get(4).map(String::as_str) == Some("debug"),
        ..Default::default()
    };
    let output = NativeTensorRtBuilder::new(options).compile_rmvpe_vfrm(&package)?;
    println!("{}", output.display());
    Ok(())
}
