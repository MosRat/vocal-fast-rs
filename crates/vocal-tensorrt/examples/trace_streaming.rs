//! Record an entire realtime stream for independent FP32 Python validation.
//! trace_streaming <pipeline.json> <model.vfr> <mono16k.f32> <trace_dir> [ratio]
//! Diagnostic downloads are intentional here; use the shared offline CLI for timing.
use std::{error::Error, path::PathBuf};
use vocal_tensorrt::{RvcTrtPipeline, RvcTrtPipelineArtifacts, TensorRtConfig};

fn save_f32(path: PathBuf, values: &[f32]) -> Result<(), std::io::Error> {
    std::fs::write(
        path,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(4..=5).contains(&args.len()) {
        return Err("trace_streaming <pipeline> <model> <mono16k.f32> <trace_dir> [ratio]".into());
    }
    let artifacts: RvcTrtPipelineArtifacts = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    if artifacts.profile.max_samples != 44960 || !artifacts.synth_tail_mode {
        return Err("full-file trace currently requires the 44960 static tail preset".into());
    }
    let package = vocal_model::load_voice_cache_shared(&args[1])?;
    let synth = package.validate_rvc_synth()?;
    let index = vocal_model::RetrievalIndex::from_package(&package)?;
    let mut config = TensorRtConfig::from_env()?;
    config.diagnostics.debug_intermediates = true;
    config.diagnostics.layer_profile = false;
    let mut pipeline = RvcTrtPipeline::load_with_runtime_config_and_retrieval(
        &artifacts,
        synth.clone(),
        config.runtime,
        config.diagnostics,
        index.as_ref(),
    )?;
    let ratio = args.get(4).map(|x| x.parse()).transpose()?.unwrap_or(0.5);
    pipeline.set_retrieval_ratio(ratio)?;
    let bytes = std::fs::read(&args[2])?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err("input must be a non-empty little-endian f32 waveform".into());
    }
    let wave = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let dir = PathBuf::from(&args[3]);
    std::fs::create_dir_all(&dir)?;
    let advance = 4000;
    let count = wave.len().div_ceil(advance);
    let mut window = vec![0.; 44960];
    for (i, chunk) in wave.chunks(advance).enumerate() {
        window.copy_within(advance.., 0);
        let tail = window.len() - advance;
        window[tail..].fill(0.);
        window[tail..tail + chunk.len()].copy_from_slice(chunk);
        let out = pipeline.process_realtime(&window, 0, advance / 160)?;
        if out.synth_skip_head != 250 || out.synth_output_frames != 31 {
            return Err("trace crop differs from the shared 44960 realtime preset".into());
        }
        let chunk_dir = dir.join(format!("{i:04}"));
        std::fs::create_dir_all(&chunk_dir)?;
        save_f32(chunk_dir.join("audio.bin"), &out.audio)?;
        for (name, values) in [
            ("synth_random", out.synth_random),
            ("synth_noise", out.synth_noise),
            ("hubert_features", out.hubert_features),
            ("feats", out.rvc_features),
            ("pitchf", out.aligned_pitchf),
        ] {
            save_f32(
                chunk_dir.join(format!("{name}.bin")),
                &values.ok_or("missing exact diagnostic output")?,
            )?;
        }
        for (name, values) in [
            ("rmvpe_raw_pitchf", out.pitchf),
            ("rmvpe_mel", out.rmvpe_mel),
            ("rmvpe_unet", out.rmvpe_unet),
            ("rmvpe_gru", out.rmvpe_gru),
            ("rmvpe_logits", out.rmvpe_logits),
        ] {
            if let Some(values) = values {
                save_f32(chunk_dir.join(format!("{name}.bin")), &values)?;
            }
        }
        std::fs::write(
            chunk_dir.join("pitch.bin"),
            out.pitch
                .ok_or("missing coarse pitch")?
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
    }
    std::fs::write(
        dir.join("trace.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "backend": "tensorrt",
            "config": {"window_samples": 44960, "skip_head_frames": 250,
                "f0": {"mode": "rolling", "advance_samples": advance, "rmvpe_samples": 4960}},
            "controls": {"speaker_id": 0, "pitch_semitones": 0., "index_ratio": ratio,
                "protect": 0.33, "valid_frames": 281},
            "advance_samples": advance, "chunks": count, "input_samples": wave.len(),
            "sample_rate": synth.sample_rate, "frames": 281,
            "encoding": "little-endian f32; pitch is little-endian i32"
        }))?,
    )?;
    println!("recorded {count} chunks in {}", dir.display());
    Ok(())
}
