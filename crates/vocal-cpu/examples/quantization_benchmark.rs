//! Shared Applio fixtures, native calibration and complete CPU W8A8 replay.
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Instant,
};
use vocal_cpu::{
    CpuConfig, CpuPipeline, CpuPipelineOutput, CpuQuantization, CpuQuantizationPreset, CpuSources,
};
use vocal_types::{AutoPitchConfig, ConversionConfig};

type Error = Box<dyn std::error::Error>;
#[derive(serde::Deserialize)]
struct Inputs {
    windows: Vec<Window>,
}
#[derive(serde::Deserialize)]
struct Window {
    split: String,
    samples: usize,
    offset: usize,
    directory: PathBuf,
}
struct Fixture {
    metadata: Window,
    wave: Vec<f32>,
    random: Vec<f32>,
    noise: Vec<f32>,
}
fn floats(path: &Path) -> Result<Vec<f32>, Error> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err("invalid FP32 fixture".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn execute(
    p: &mut CpuPipeline,
    f: &Fixture,
    c: ConversionConfig,
    debug: bool,
) -> Result<CpuPipelineOutput, Error> {
    Ok(p.process_exact(&f.wave, c, 0, &f.random, &f.noise, debug)?)
}
fn timing(o: CpuPipelineOutput) -> [f64; 5] {
    [
        o.feature_ms,
        o.pitch_ms,
        o.retrieval_ms,
        o.synth_ms,
        o.total_ms,
    ]
}
fn outputs(
    p: &mut CpuPipeline,
    fixtures: &[Fixture],
    c: ConversionConfig,
    directory: &Path,
) -> Result<Vec<serde_json::Value>, Error> {
    std::fs::create_dir_all(directory)?;
    let mut outputs = Vec::new();
    for f in fixtures {
        let o = execute(p, f, c, true)?;
        let stem = format!("{}-{}", f.metadata.split, f.metadata.offset);
        let audio = directory.join(format!("{stem}.f32"));
        std::fs::write(
            &audio,
            o.audio
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        std::fs::write(
            directory.join(format!("{stem}-pitch.i32")),
            &o.debug["pitch"],
        )?;
        std::fs::write(
            directory.join(format!("{stem}-pitchf.f32")),
            &o.debug["pitchf"],
        )?;
        outputs.push(serde_json::json!({"split":f.metadata.split,"offset":f.metadata.offset,"reference_directory":f.metadata.directory,"audio":audio}));
    }
    Ok(outputs)
}
fn summary(
    p: &CpuPipeline,
    build_ms: f64,
    rows: &[[f64; 5]],
    outputs: Vec<serde_json::Value>,
) -> serde_json::Value {
    let mean = (0..5)
        .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / rows.len() as f64)
        .collect::<Vec<_>>();
    serde_json::json!({"config":p.config,"build_ms":build_ms,"rows_ms":rows,
        "mean_ms":{"hubert":mean[0],"rmvpe":mean[1],"retrieval":mean[2],"synth":mean[3],"total":mean[4]},
        "memory":p.memory_usage(),"quantized":p.quantized_operators(),"outputs":outputs})
}
fn main() -> Result<(), Error> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(9..=11).contains(&args.len()) {
        return Err("usage: quantization_benchmark synth.vfrm hubert.vfrm rmvpe.vfrm inputs.json recipe.json calibration.json output_dir samples iterations [config.json] [baseline_config.json]; iterations=0 observes FP32 calibration, recipe=- runs FP32, recipe=preset:<serde_name> uses a typed preset".into());
    }
    let samples: usize = args[7].parse()?;
    let iterations: usize = args[8].parse()?;
    let directory = Path::new(&args[6]);
    std::fs::create_dir_all(directory)?;
    // Preserve the original layout as the experimental control even when the
    // production default adopts token-major output. This keeps old reports
    // comparable and makes the boundary optimization explicit in config.json.
    let default_config = CpuConfig {
        window_samples: samples,
        skip_head_frames: if samples == 44960 { 250 } else { 8 },
        threads: 6,
        attention: vocal_cpu::CpuAttentionConfig {
            token_major_output: false,
            ..Default::default()
        },
        ..CpuConfig::default()
    };
    let config: CpuConfig = if let Some(path) = args.get(9) {
        serde_json::from_slice(&std::fs::read(path)?)?
    } else {
        default_config.clone()
    };
    let baseline_config: CpuConfig = if let Some(path) = args.get(10) {
        serde_json::from_slice(&std::fs::read(path)?)?
    } else {
        default_config
    };
    if [&config, &baseline_config].iter().any(|c| {
        c.window_samples != samples
            || c.profile
            || c.skip_head_frames != config.skip_head_frames
            || c.append_hubert_frame != config.append_hubert_frame
    }) {
        return Err("benchmark config must match fixture geometry and disable node clocks".into());
    }
    let sources = CpuSources {
        synth: args[0].clone().into(),
        hubert: args[1].clone().into(),
        rmvpe: args[2].clone().into(),
    };
    let preset = args[4]
        .strip_prefix("preset:")
        .map(|name| {
            serde_json::from_value::<CpuQuantizationPreset>(serde_json::Value::String(name.into()))
        })
        .transpose()?;
    let mut recipe: CpuQuantization = if let Some(preset) = preset {
        preset.recipe()
    } else if args[4] == "-" {
        CpuQuantization::default()
    } else {
        serde_json::from_slice(&std::fs::read(&args[4])?)?
    };
    if iterations > 0 && args[4] != "-" {
        recipe.calibration = serde_json::from_slice(&std::fs::read(&args[5])?)?;
    }
    let input: Inputs = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    let fixtures = input
        .windows
        .into_iter()
        .filter(|w| w.samples == samples && (w.split == "calibration") == (iterations == 0))
        .map(|metadata| {
            Ok(Fixture {
                wave: floats(&metadata.directory.join("wave.f32"))?,
                random: floats(&metadata.directory.join("rnd.f32"))?,
                noise: floats(&metadata.directory.join("noise.f32"))?,
                metadata,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if fixtures.is_empty() {
        return Err("missing fixtures".into());
    }
    let start = Instant::now();
    let mut pipeline = if iterations == 0 {
        CpuPipeline::from_sources_calibration(&sources, config.clone(), &recipe)?
    } else if args[4] == "-" {
        CpuPipeline::from_sources(&sources, config.clone())?
    } else if let Some(preset) = preset {
        CpuPipeline::from_sources_preset(&sources, config.clone(), preset, recipe.calibration)?
    } else {
        CpuPipeline::from_sources_quantized(&sources, config.clone(), &recipe)?
    };
    let build_ms = start.elapsed().as_secs_f64() * 1000.;
    let conversion = ConversionConfig {
        index_ratio: 0.,
        auto_pitch: AutoPitchConfig::disabled(),
        ..ConversionConfig::default()
    };
    if iterations == 0 {
        for f in &fixtures {
            execute(&mut pipeline, f, conversion, false)?;
        }
        let calibration: BTreeMap<String, Vec<f32>> = pipeline.calibration()?;
        std::fs::write(&args[5], serde_json::to_vec(&calibration)?)?;
        println!(
            "{}",
            serde_json::json!({"calibrated_layers":calibration.len(),"samples":samples})
        );
        return Ok(());
    }
    let mut baseline = if args[4] != "-" || config != baseline_config || args.get(10).is_some() {
        let start = Instant::now();
        let p = CpuPipeline::from_sources(&sources, baseline_config)?;
        Some((p, start.elapsed().as_secs_f64() * 1000.))
    } else {
        None
    };
    let quant_outputs = outputs(&mut pipeline, &fixtures, conversion, directory)?;
    let base_outputs = if let Some((p, _)) = &mut baseline {
        outputs(p, &fixtures, conversion, &directory.join("fp32"))?
    } else {
        Vec::new()
    };
    for i in 0..8 {
        let f = &fixtures[i % fixtures.len()];
        execute(&mut pipeline, f, conversion, false)?;
        if let Some((p, _)) = &mut baseline {
            execute(p, f, conversion, false)?;
        }
    }
    // The timed path has no diagnostic outputs or per-node clocks. It includes
    // stage IO, static activation Q/layout reorder and final FP32 output copy.
    let mut rows = Vec::new();
    let mut base_rows = Vec::new();
    for i in 0..iterations {
        let fixture = &fixtures[i % fixtures.len()];
        if i % 2 == 0 {
            if let Some((p, _)) = &mut baseline {
                base_rows.push(timing(execute(p, fixture, conversion, false)?));
            }
            rows.push(timing(execute(&mut pipeline, fixture, conversion, false)?));
        } else {
            rows.push(timing(execute(&mut pipeline, fixture, conversion, false)?));
            if let Some((p, _)) = &mut baseline {
                base_rows.push(timing(execute(p, fixture, conversion, false)?));
            }
        }
    }
    let mut result = summary(&pipeline, build_ms, &rows, quant_outputs);
    if let Some((p, ms)) = &baseline {
        result["baseline"] = summary(p, *ms, &base_rows, base_outputs);
    }
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
