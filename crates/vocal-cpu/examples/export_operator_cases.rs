//! Capture real fixed-shape VFRM operator inputs. Capture timings are not benchmarks.
use std::{collections::BTreeMap, path::Path};
use vocal_cpu::{CpuConfig, CpuGraph, CpuPipeline, CpuSources};
use vocal_graph::{
    Precision, PrecisionConfig,
    graph::{DType, GraphBuilder, Op, Tensor},
    models,
};
use vocal_types::{AutoPitchConfig, ConversionConfig};

type Error = Box<dyn std::error::Error>;

#[derive(serde::Serialize)]
struct Case {
    id: String,
    stage: String,
    name: String,
    op: u32,
    attrs: Vec<i32>,
    input_shapes: Vec<Vec<usize>>,
    output_shape: Vec<usize>,
    occurrences: usize,
}

fn mark_cases(g: &GraphBuilder, stage: &str) -> Vec<Case> {
    let nodes = g.nodes.borrow();
    let mut groups: BTreeMap<String, (Case, Vec<Tensor>, Tensor)> = BTreeMap::new();
    for n in nodes.iter() {
        if !matches!(n.op, Op::Conv | Op::Deconv | Op::Matmul) {
            continue;
        }
        let mut inputs = n
            .inputs
            .iter()
            .map(|id| nodes[*id as usize].tensor.clone())
            .collect::<Vec<_>>();
        let mut output = n.tensor.clone();
        let mut op = n.op as u32;
        if matches!(n.op, Op::Matmul) {
            // Attention is fused in production. Benchmark static projection/FFN GEMMs.
            if inputs[1].shape.len() != 2 {
                continue;
            }
            if let Some(add) = nodes.iter().find(|a| {
                matches!(a.op, Op::Add)
                    && a.inputs[0] == n.tensor.id
                    && nodes[a.inputs[1] as usize].tensor.elements()
                        == *n.tensor.shape.last().unwrap()
            }) {
                inputs.push(nodes[add.inputs[1] as usize].tensor.clone());
                output = add.tensor.clone();
                op = Op::LinearBias as u32;
            }
        }
        if inputs.iter().any(|t| t.dtype != DType::F32) {
            continue;
        }
        let shapes = inputs.iter().map(|t| t.shape.clone()).collect::<Vec<_>>();
        let key = format!("{op}:{shapes:?}:{:?}:{:?}", n.attrs, output.shape);
        if let Some((case, _, _)) = groups.get_mut(&key) {
            case.occurrences += 1;
            continue;
        }
        let name = if n.name.is_empty() {
            nodes[n.inputs[1] as usize].name.clone()
        } else {
            n.name.clone()
        };
        groups.insert(
            key,
            (
                Case {
                    id: String::new(),
                    stage: stage.into(),
                    name,
                    op,
                    attrs: n.attrs.clone(),
                    input_shapes: shapes,
                    output_shape: output.shape.clone(),
                    occurrences: 1,
                },
                inputs,
                output,
            ),
        );
    }
    drop(nodes);
    groups
        .into_values()
        .enumerate()
        .map(|(i, (mut case, inputs, output))| {
            case.id = format!("{stage}_{i:03}");
            for (j, input) in inputs.iter().enumerate() {
                g.mark(&format!("{}_input_{j}", case.id), input);
            }
            g.mark(&format!("{}_output", case.id), &output);
            case
        })
        .collect()
}

fn capture(
    g: GraphBuilder,
    stage: &str,
    bindings: &[(&str, &[u8])],
    config: &CpuConfig,
    dir: &Path,
) -> Result<Vec<Case>, Error> {
    let cases = mark_cases(&g, stage);
    let mut capture_config = config.clone();
    capture_config.fuse = false;
    capture_config.profile = false;
    let mut graph = CpuGraph::compile(g, &capture_config)?;
    let bindings = bindings
        .iter()
        .copied()
        .filter(|(name, _)| graph.input_names().any(|n| n == *name))
        .collect::<Vec<_>>();
    let output = graph.run(&bindings)?;
    for (name, data) in output.tensors {
        std::fs::write(dir.join(format!("{name}.f32")), data)?;
    }
    eprintln!("captured {stage}: {} distinct operator shapes", cases.len());
    Ok(cases)
}

fn main() -> Result<(), Error> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 6 {
        return Err("usage: export_operator_cases synth.vfrm hubert.vfrm rmvpe.vfrm waveform.f32 samples directory [threads]".into());
    }
    let samples = args[4].parse()?;
    let dir = Path::new(&args[5]);
    std::fs::create_dir_all(dir)?;
    let sources = CpuSources {
        synth: args[0].clone().into(),
        hubert: args[1].clone().into(),
        rmvpe: args[2].clone().into(),
    };
    let config = CpuConfig {
        window_samples: samples,
        skip_head_frames: if samples == 44960 { 250 } else { 8 },
        threads: args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(6),
        profile: true,
        ..CpuConfig::default()
    };
    let mut wave = std::fs::read(&args[3])?;
    wave.resize(samples * 4, 0);
    let wave_f32 = wave
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let conversion = ConversionConfig {
        index_ratio: 0.,
        auto_pitch: AutoPitchConfig::disabled(),
        ..ConversionConfig::default()
    };
    let mut pipeline = CpuPipeline::from_sources(&sources, config.clone())?;
    for _ in 0..3 {
        pipeline.process(&wave_f32, conversion, 0, false)?;
    }
    let result = pipeline.process(&wave_f32, conversion, 0, true)?;
    let frames = pipeline.frames;
    let latent_channels = pipeline.latent_channels;
    let upsample = pipeline.upsample;
    drop(pipeline);
    std::fs::write(
        dir.join("pipeline-profile.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "samples": samples, "config": config, "nodes": result.timings,
            "stage_ms": {"hubert": result.feature_ms, "rmvpe": result.pitch_ms, "synth": result.synth_ms},
            "note": "one diagnostic run, per-node timing enabled; not a steady-state benchmark"
        }))?,
    )?;
    let precision = PrecisionConfig::fp32();
    let hubert = vocal_model::load_cache_shared(&sources.hubert)?;
    let g = GraphBuilder::default();
    let input = g.input("waveform", &[1, samples], DType::F32);
    let features = models::hubert(&g, &hubert, &input, &precision)?;
    g.mark("features", &features);
    let mut cases = capture(g, "hubert", &[("waveform", &wave)], &config, dir)?;
    drop(hubert);

    let rmvpe = vocal_model::load_cache_shared(&sources.rmvpe)?;
    let g = GraphBuilder::default();
    let input = g.input("waveform", &[1, samples], DType::F32);
    let threshold = g.input("threshold", &[1], DType::F32);
    let basis = g.parameter(&rmvpe, "rmvpe.mel_basis", Precision::Fp32)?;
    let mel = g.node(
        Op::StftMel,
        &[&input, &basis],
        vec![1, 128, samples / 160 + 1],
        DType::F32,
        vec![],
        [0.; 4],
        vec![],
        "rmvpe.stft_mel".into(),
    );
    let f0 = models::rmvpe_from_mel(&g, &rmvpe, &mel, &precision, &threshold, false)?;
    g.mark("pitchf", &f0);
    cases.extend(capture(
        g,
        "rmvpe",
        &[("waveform", &wave), ("threshold", &0.03f32.to_le_bytes())],
        &config,
        dir,
    )?);
    drop(rmvpe);

    let synth = vocal_model::load_voice_cache_shared(&sources.synth)?;
    let g = GraphBuilder::default();
    let feats = g.input("feats", &[1, frames, 768], DType::F32);
    let pitch = g.input("pitch", &[1, frames], DType::I32);
    let pitchf = g.input("pitchf", &[1, frames], DType::F32);
    let length = g.input("p_len", &[1], DType::I32);
    let sid = g.input("sid", &[1], DType::I32);
    let rnd = g.input("rnd", &[1, latent_channels, frames], DType::F32);
    let noise = g.input("nsf_noise", &[1, 1, frames * upsample], DType::F32);
    let audio = models::synth(
        &g,
        &synth,
        &feats,
        &pitch,
        &pitchf,
        &length,
        &sid,
        &rnd,
        &noise,
        config.skip_head_frames,
        &precision,
        false,
    )?;
    g.mark("audio", &audio);
    let d = &result.debug;
    cases.extend(capture(
        g,
        "synth",
        &[
            ("feats", &d["feats"]),
            ("pitch", &d["pitch"]),
            ("pitchf", &d["pitchf"]),
            ("p_len", &(frames as i32).to_le_bytes()),
            ("sid", &0i32.to_le_bytes()),
            ("rnd", &d["synth_random"]),
            ("nsf_noise", &d["synth_noise"]),
        ],
        &config,
        dir,
    )?);
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": 1, "sources": sources, "config": config, "frames": frames,
            "reference": "actual VFRM FP32 graph tensors, canonical contiguous order",
            "cases": cases,
        }))?,
    )?;
    println!("exported {} cases to {}", cases.len(), dir.display());
    Ok(())
}
