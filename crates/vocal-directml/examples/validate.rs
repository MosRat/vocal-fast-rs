//! Reproducible native GPU parity/profiling entry point.
//! validate <stage> <model-or-sources> <wave.f32|snapshot.json> <output.json>
//! [fp32|mixed|manifest] [iterations] [debug|nodebug] [index_ratio] [trace_dir]
//! The optional streaming trace records exact noise and stage outputs for
//! full-file Python parity. Its diagnostic copies are excluded from benchmarks.
use serde_json::{Value, json};
use std::{error::Error, time::Instant};
use vocal_directml::{
    DirectMlDevice, DirectMlPipeline, PrecisionConfig, pipeline::DirectMlSources,
};

fn floats(v: &Value, name: &str) -> Vec<f32> {
    v[name]
        .as_array()
        .unwrap_or_else(|| panic!("missing {name}"))
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 4 {
        return Err("validate <stage> <model> <input> <output> [fp32|mixed] [iterations]".into());
    }
    let mut precision = match args.get(4).map(String::as_str) {
        Some("fp32") => PrecisionConfig::fp32(),
        None | Some("mixed" | "manifest") => PrecisionConfig::default(),
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
    };
    let iterations = args
        .get(5)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(10usize)
        .max(1);
    let started = Instant::now();
    let mut gpu = Vec::new();
    let mut host = Vec::new();
    let result;
    let mut config = Value::Null;
    let mut controls_report = Value::Null;
    let mut final_audio = None;
    let mut stream = Value::Null;
    if args[0] == "pipeline" || args[0] == "streaming" {
        let mut sources = DirectMlSources::load(&args[1])?;
        if args.get(4).map(String::as_str) == Some("manifest") {
            precision = sources.config.precision.clone();
        } else {
            sources.config.precision = precision.clone();
        }
        sources.config.debug_outputs = args.get(6).map(String::as_str) != Some("nodebug");
        config = serde_json::to_value(&sources.config)?;
        let mut pipeline = DirectMlPipeline::from_sources(&sources)?;
        let mut controls = vocal_directml::DirectMlControls::new(pipeline.frames);
        controls.index_ratio = args.get(7).map(|s| s.parse()).transpose()?.unwrap_or(0.);
        controls_report = serde_json::to_value(&controls)?;
        eprintln!(
            "compiled pipeline in {:.2}s",
            started.elapsed().as_secs_f64()
        );
        let wave = std::fs::read(&args[2])?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        let mut last = None;
        let is_stream = args[0] == "streaming";
        let advance = match pipeline.config().f0 {
            vocal_directml::F0Config::Rolling {
                advance_samples, ..
            } => advance_samples,
            _ => 4000,
        };
        let count = if is_stream {
            wave.len().div_ceil(advance).min(iterations + 2)
        } else {
            iterations + 2
        };
        if count <= 2 {
            return Err("stream input must contain at least three playback chunks".into());
        }
        let mut window = vec![0.; pipeline.config().window_samples];
        let trace_dir = args.get(8).map(std::path::PathBuf::from);
        if let Some(dir) = &trace_dir {
            if !is_stream || !sources.config.debug_outputs {
                return Err("trace_dir requires streaming with debug outputs".into());
            }
            std::fs::create_dir_all(dir)?;
            std::fs::write(
                dir.join("trace.json"),
                serde_json::to_vec_pretty(&json!({
                    "config": config, "controls": controls_report,
                    "advance_samples": advance, "chunks": count,
                    "input_samples": wave.len(), "sample_rate": pipeline.sample_rate,
                    "frames": pipeline.frames, "latent_channels": pipeline.latent_channels,
                    "upsample": pipeline.upsample,
                    "encoding": "little-endian f32; pitch is little-endian i32"
                }))?,
            )?;
        }
        for i in 0..count {
            let input = if is_stream {
                window.copy_within(advance.., 0);
                let tail = window.len() - advance;
                window[tail..].fill(0.);
                let begin = i * advance;
                let end = (begin + advance).min(wave.len());
                window[tail..tail + end - begin].copy_from_slice(&wave[begin..end]);
                window.as_slice()
            } else {
                wave.as_slice()
            };
            let out = pipeline.process(input, &controls)?;
            if let Some(dir) = &trace_dir {
                let chunk_dir = dir.join(format!("{i:04}"));
                std::fs::create_dir_all(&chunk_dir)?;
                for name in [
                    "audio",
                    "synth_random",
                    "synth_noise",
                    "hubert_raw",
                    "hubert_features",
                    "feats",
                    "pitch",
                    "pitchf",
                    "f0_cache",
                ] {
                    if let Some((_, bytes)) = out.debug.tensors.iter().find(|(n, _)| n == name) {
                        std::fs::write(chunk_dir.join(format!("{name}.bin")), bytes)?;
                    }
                }
            }
            gpu.push(out.gpu_ms);
            host.push(out.total_ms);
            final_audio = Some(out.audio);
            last = Some(out.debug);
        }
        if is_stream {
            stream = json!({"advance_samples": advance, "chunks": count});
        }
        result = last.unwrap();
    } else {
        let model = vocal_model::load_cache_shared(&args[1])?;
        let device = DirectMlDevice::new(0)?;
        let mut bindings: Vec<(String, Vec<u8>)> = Vec::new();
        let mut graph = if args[0] == "retrieval" {
            let v: Value = serde_json::from_slice(&std::fs::read(&args[2])?)?;
            let values = floats(&v, "hubert_features");
            let index = vocal_model::RetrievalIndex::from_package(&model)?
                .ok_or("missing retrieval index")?;
            bindings.push(("features".into(), bytes(&values)));
            bindings.push(("index_ratio".into(), 0.5f32.to_le_bytes().to_vec()));
            let search = match args.get(6).map(String::as_str) {
                Some("dense") => vocal_directml::RetrievalSearch::Dense,
                Some("sparse") => vocal_directml::RetrievalSearch::Sparse,
                _ => vocal_directml::RetrievalSearch::Auto,
            };
            device.compile_retrieval_with_search(
                &index,
                values.len() / index.dimensions(),
                search,
            )?
        } else if args[0] == "synth" {
            let v: Value = serde_json::from_slice(&std::fs::read(&args[2])?)?;
            let frames = v["pitch"].as_array().ok_or("snapshot missing pitch")?.len();
            let skip = v["synth_skip_head"]
                .as_u64()
                .ok_or("snapshot missing crop")? as usize;
            for (input, key) in [
                ("feats", "rvc_features"),
                ("pitchf", "aligned_pitchf"),
                ("rnd", "synth_random"),
                ("nsf_noise", "synth_noise"),
            ] {
                bindings.push((input.into(), bytes(&floats(&v, key))));
            }
            bindings.push((
                "pitch".into(),
                v["pitch"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|x| (x.as_i64().unwrap() as i32).to_le_bytes())
                    .collect(),
            ));
            bindings.push((
                "p_len".into(),
                (v["valid_synth_frames"].as_u64().unwrap_or(frames as u64) as i32)
                    .to_le_bytes()
                    .to_vec(),
            ));
            bindings.push(("sid".into(), 0i32.to_le_bytes().to_vec()));
            device.compile_synth(&model, frames, skip, &precision, true)?
        } else {
            let wave = std::fs::read(&args[2])?;
            let samples = wave.len() / 4;
            bindings.push(("waveform".into(), wave));
            if args[0] == "hubert" {
                device.compile_hubert(&model, samples, &precision)?
            } else if args[0] == "rmvpe" {
                bindings.push(("threshold".into(), 0.03f32.to_le_bytes().to_vec()));
                device.compile_rmvpe(&model, samples, &precision, true)?
            } else {
                return Err("unknown stage".into());
            }
        };
        eprintln!(
            "compiled {} in {:.2}s",
            args[0],
            started.elapsed().as_secs_f64()
        );
        let bindings = bindings
            .iter()
            .map(|(n, b)| (n.as_str(), b.as_slice()))
            .collect::<Vec<_>>();
        let mut last = None;
        for _ in 0..iterations + 2 {
            let out = graph.run(&bindings)?;
            gpu.push(out.gpu_ms);
            host.push(out.total_ms);
            last = Some(out);
        }
        result = last.unwrap();
    }
    let mut tensors = serde_json::Map::new();
    for (name, _) in &result.tensors {
        if name != "pitch" {
            tensors.insert(name.clone(), json!(result.f32(name)?));
        }
    }
    if let Some(audio) = final_audio {
        tensors
            .entry("audio".to_string())
            .or_insert_with(|| json!(audio));
    }
    let mean = |a: &[f64]| a.iter().sum::<f64>() / a.len() as f64;
    eprintln!(
        "GPU {:.3} ms; host {:.3} ms",
        mean(&gpu[2..]),
        mean(&host[2..])
    );
    let pitch = result
        .tensors
        .iter()
        .find(|(n, _)| n == "pitch")
        .map(|(_, b)| {
            b.chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>()
        });
    let report = json!({"precision": precision, "config": config, "pitch": pitch, "stream": stream, "controls": controls_report,
        "gpu_ms": &gpu[2..], "host_ms": &host[2..], "tensors": tensors});
    std::fs::write(&args[3], serde_json::to_vec(&report)?)?;
    Ok(())
}
