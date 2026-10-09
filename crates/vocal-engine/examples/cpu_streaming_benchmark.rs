//! Paired CPU streaming experiments. Diagnostics, node profiles and actual
//! engine replay execute separately from paired production pipeline timing.
#[path = "cpu_streaming_support/mod.rs"]
mod support;
use serde_json::json;
use std::{error::Error, path::Path, time::Instant};
use vocal_audio::{read_wav, write_wav_pcm16};
use vocal_cpu::{CpuConfig, CpuF0Config, CpuPipeline, CpuPipelineOutput, CpuSources};
use vocal_engine::{CpuEngine, CpuStreamConfig, InferenceEngine};
use vocal_types::{AudioFormat, AutoPitchConfig, ConversionConfig};

const ADVANCE: usize = 4000;
const WARMUP: usize = 13;

fn window_push(window: &mut [f32], chunk: &[f32]) {
    window.copy_within(ADVANCE.., 0);
    let tail = window.len() - ADVANCE;
    window[tail..].fill(0.);
    window[tail..tail + chunk.len()].copy_from_slice(chunk);
}
fn row(output: CpuPipelineOutput) -> [f64; 5] {
    [
        output.feature_ms,
        output.pitch_ms,
        output.retrieval_ms,
        output.synth_ms,
        output.total_ms,
    ]
}
fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(3..=5).contains(&args.len()) {
        return Err(
            "usage: cpu_streaming_benchmark sources.json mono16k.wav report_directory [threads] [experiment.json]"
                .into(),
        );
    }
    let sources: CpuSources = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let input = read_wav(&args[1])?;
    if input.sample_rate != 16000 || input.channels != 1 || input.samples.len() / ADVANCE <= WARMUP
    {
        return Err("input must be mono 16 kHz and exceed startup warmup".into());
    }
    let directory = Path::new(&args[2]);
    std::fs::create_dir_all(directory)?;
    let options: support::Experiment = args
        .get(4)
        .map(|path| -> Result<_, Box<dyn Error>> {
            Ok(serde_json::from_slice(&std::fs::read(path)?)?)
        })
        .transpose()?
        .unwrap_or_default();
    let baseline_config = options.baseline.clone().unwrap_or(CpuConfig {
        threads: args
            .get(3)
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(6),
        ..CpuConfig::realtime_44960()
    });
    let config = options.candidate.clone().unwrap_or(CpuConfig {
        f0: CpuF0Config::rvc_realtime(),
        ..baseline_config.clone()
    });
    options.validate(&baseline_config, &config)?;
    let recipe = options.recipe()?;
    let conversion = ConversionConfig {
        auto_pitch: AutoPitchConfig::disabled(),
        index_ratio: 0.,
        rms_mix_rate: 1.,
        ..Default::default()
    };
    let started = Instant::now();
    if options.profile {
        return support::profile(
            &sources,
            &config,
            recipe.as_ref(),
            &input.samples,
            conversion,
            directory,
        );
    }
    let mut baseline = CpuPipeline::from_sources(&sources, baseline_config.clone())?;
    let mut candidate = support::pipeline(&sources, config.clone(), recipe.as_ref())?;
    support::validate_coverage(&candidate, recipe.as_ref())?;
    let quantized = candidate.quantized_operators();
    let build_ms = started.elapsed().as_secs_f64() * 1000.;
    let frames = candidate.frames;
    let sample_rate = candidate.sample_rate;
    let stream = CpuStreamConfig::default();
    let startup_mute_blocks = if stream.warmup { WARMUP } else { 0 };
    let memory =
        json!({"baseline": baseline.memory_usage(), "candidate": candidate.memory_usage()});
    let trace = directory.join("trace");
    let chunks = input.samples.len().div_ceil(ADVANCE);
    let mut window = vec![0.; config.window_samples];
    if options.trace {
        std::fs::create_dir_all(&trace)?;
        std::fs::write(
            trace.join("trace.json"),
            serde_json::to_vec_pretty(&json!({
                "backend": "cpu", "config": config,
                "controls": {"speaker_id": 0, "index_ratio": 0., "pitch_semitones": 0.,
                             "protect": conversion.protect, "rms_mix_rate": 1., "valid_frames": frames},
                "advance_samples": ADVANCE, "chunks": chunks, "input_samples": input.samples.len(),
                "sample_rate": sample_rate, "output_rate": 16000, "frames": frames,
            "boundary": {"mode": "speech_safe_v1", "overlap": stream.crossfade_samples_16k,
                         "search": stream.sola_search_samples_16k, "warmup_blocks": startup_mute_blocks,
                         "timing_warmup_blocks": WARMUP},
                "latent_channels": candidate.latent_channels, "upsample": candidate.upsample,
                "encoding": "little-endian f32; pitch is little-endian i32"
            }))?,
        )?;
        // Diagnostics execute before the timed pass. Two fresh, identically seeded
        // pipelines give exact noise provenance without collecting tensors in timing.
        for (i, chunk) in input.samples.chunks(ADVANCE).enumerate() {
            window_push(&mut window, chunk);
            let output = candidate.process(&window, conversion, 0, true)?;
            let chunk_dir = trace.join(format!("{i:04}"));
            std::fs::create_dir_all(&chunk_dir)?;
            for (name, data) in output.debug {
                std::fs::write(chunk_dir.join(format!("{name}.bin")), data)?;
            }
        }
        candidate.reset_stream();
        // Baseline must consume the same noise sequence before the timed pass.
        window.fill(0.);
        for chunk in input.samples.chunks(ADVANCE) {
            window_push(&mut window, chunk);
            baseline.process(&window, conversion, 0, false)?;
        }
        baseline.reset_stream();
    }
    let mut full = Vec::new();
    let mut rolling = Vec::new();
    window.fill(0.);
    for (i, chunk) in input.samples.chunks(ADVANCE).enumerate() {
        window_push(&mut window, chunk);
        let (left, right) = if i % 2 == 0 {
            let left = baseline.process(&window, conversion, 0, false)?;
            (left, candidate.process(&window, conversion, 0, false)?)
        } else {
            let right = candidate.process(&window, conversion, 0, false)?;
            (baseline.process(&window, conversion, 0, false)?, right)
        };
        if i >= WARMUP {
            full.push(row(left));
            rolling.push(row(right));
        }
    }
    // Release paired graphs before production engine replay; its timings are
    // reported separately and include the actual common audio boundary.
    drop(baseline);
    drop(candidate);
    let model = vocal_model::load_voice_cache_shared(&sources.synth)?
        .manifest
        .descriptor();
    let mut engine = match recipe.as_ref() {
        Some(recipe) => CpuEngine::from_sources_quantized(
            model,
            &sources,
            config.clone(),
            stream.clone(),
            0,
            recipe,
        )?,
        None => CpuEngine::from_sources(model, &sources, config.clone(), stream.clone(), 0)?,
    };
    let mut audio = Vec::with_capacity(chunks * ADVANCE);
    let mut engine_ms = Vec::new();
    for (i, chunk) in input.samples.chunks(ADVANCE).enumerate() {
        let mut padded = [0.; ADVANCE];
        padded[..chunk.len()].copy_from_slice(chunk);
        let started = Instant::now();
        let output =
            engine.process_samples(AudioFormat::MONO_16K, &padded, i as u64, conversion)?;
        if i >= WARMUP {
            engine_ms.push(started.elapsed().as_secs_f64() * 1000.);
        }
        audio.extend_from_slice(&output.samples[..chunk.len()]);
    }
    write_wav_pcm16(directory.join("cpu-rolling.wav"), 16000, &audio)?;
    let result = json!({
        "sources": sources, "input": args[1], "config": config, "baseline_config": baseline_config,
        "build_ms": build_ms, "warmup_blocks": WARMUP, "paired_chunks": full.len(),
        "stream": {"chunk_samples": stream.chunk_samples_16k, "warmup": stream.warmup},
        "startup_mute_blocks": startup_mute_blocks,
        "full_window_rows_ms": full, "rolling_rows_ms": rolling, "engine_ms": engine_ms,
        "baseline_rows_ms": full, "candidate_rows_ms": rolling, "quantized": quantized,
        "experiment": options,
        "memory": memory, "sample_rate": sample_rate, "output_rate": 16000,
        "chunk_budget_ms": 250, "audio": directory.join("cpu-rolling.wav"), "trace": trace,
        "timing": "unpaced file replay; paired process includes noise generation and stage IO; diagnostics excluded; engine includes resampling/SOLA/buffering, not audio devices"
    });
    std::fs::write(
        directory.join("benchmark.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!(
        "{}",
        json!({"paired_chunks": full.len(), "report": directory.join("benchmark.json")})
    );
    Ok(())
}
