//! Real model benchmark. Separate processes allow different native thread pools.
use std::{path::PathBuf, time::Instant};
use vocal_cpu::{CpuConfig, CpuPipeline, CpuSources};
use vocal_types::{AutoPitchConfig, ConversionConfig};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 6 {
        return Err("usage: benchmark synth.vfrm hubert.vfrm rmvpe.vfrm waveform.f32 samples iterations [threads] [channels_last] [fuse] [skip] [spin_us] [pin_performance_cores]".into());
    }
    let samples: usize = args[4].parse()?;
    let iterations: usize = args[5].parse()?;
    if iterations == 0 {
        return Err("iterations must be positive".into());
    }
    let config = CpuConfig {
        window_samples: samples,
        threads: args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(6),
        spin_wait_us: args.get(10).map(|s| s.parse()).transpose()?.unwrap_or(0),
        pin_to_performance_cores: args
            .get(11)
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(false),
        channels_last: args.get(7).map(|s| s.parse()).transpose()?.unwrap_or(true),
        fuse: args.get(8).map(|s| s.parse()).transpose()?.unwrap_or(true),
        skip_head_frames: args
            .get(9)
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(if samples == 44960 { 250 } else { 8 }),
        ..CpuConfig::default()
    };
    let mut wave = std::fs::read(&args[3])?
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    wave.resize(samples, 0.);
    wave.truncate(samples);
    let build = Instant::now();
    let mut pipeline = CpuPipeline::from_sources(
        &CpuSources {
            synth: PathBuf::from(&args[0]),
            hubert: PathBuf::from(&args[1]),
            rmvpe: PathBuf::from(&args[2]),
        },
        config.clone(),
    )?;
    let build_ms = build.elapsed().as_secs_f64() * 1000.;
    let controls = ConversionConfig {
        auto_pitch: AutoPitchConfig::disabled(),
        ..ConversionConfig::default()
    };
    pipeline.process(&wave, controls, 0, false)?;
    let mut rows = Vec::new();
    let mut output = vec![];
    for _ in 0..iterations {
        let result = pipeline.process(&wave, controls, 0, false)?;
        rows.push([
            result.feature_ms,
            result.pitch_ms,
            result.retrieval_ms,
            result.synth_ms,
            result.total_ms,
        ]);
        output = result.audio;
    }
    if let Ok(path) = std::env::var("VOCAL_CPU_BENCH_OUTPUT") {
        std::fs::write(
            path,
            output
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
    }
    let mean = (0..5)
        .map(|i| rows.iter().map(|r| r[i]).sum::<f64>() / iterations as f64)
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"config":config,"memory":pipeline.memory_usage(),"noise":"seeded_normal","build_ms":build_ms,"iterations":iterations,"mean_ms":{"hubert":mean[0],"rmvpe":mean[1],"retrieval":mean[2],"synth":mean[3],"total":mean[4]},"output_samples":output.len(),"frames":pipeline.frames,"rows_ms":rows})
        )?
    );
    Ok(())
}
