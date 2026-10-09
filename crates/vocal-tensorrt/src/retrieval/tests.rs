//! Opt-in real CUDA retrieval benchmark and FAISS fixture comparison.
//! File generation and timing summaries live in scripts/benchmark-retrieval.py.
use super::*;
use crate::timing::CudaEvent;
use std::{env, fs, path::PathBuf};

#[test]
fn rejects_undersized_or_overflowing_bindings() {
    assert!(validate_blend_buffers(140, 768, 140 * 768 * 4, 4).is_ok());
    assert!(validate_blend_buffers(140, 768, 140 * 768 * 4 - 1, 4).is_err());
    assert!(validate_blend_buffers(140, 768, 140 * 768 * 4, 3).is_err());
    assert!(validate_blend_buffers(usize::MAX, 768, usize::MAX, 4).is_err());
}

#[test]
#[ignore = "requires CUDA and VOCAL_RETRIEVAL_BENCH fixture directory"]
fn benchmark_fixture() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env::var("VOCAL_RETRIEVAL_BENCH")?);
    let index = RetrievalIndex::from_faiss(&fs::read(root.join("index.faiss"))?)?;
    let input = fs::read(root.join("query.f32"))?;
    let values = input.len() / 4;
    assert_eq!(input.len() % (index.dimensions() * 4), 0);
    let frames = values / index.dimensions();
    let gpu = GpuRetrievalIndex::new(&index)?;
    let stream = CudaStream::new()?;
    let source = CudaBuffer::new(input.len())?;
    source.upload_bytes(&input)?;
    let features = CudaBuffer::new(input.len())?;
    let rate = CudaBuffer::new(4)?;
    rate.upload_f32(&[0.5])?;
    let start = CudaEvent::new()?;
    let end = CudaEvent::new()?;
    features.device_copy_from(&source, 0, 0, input.len(), &stream)?;
    gpu.blend(&features, frames, &rate, &stream)?;
    stream.synchronize()?;
    let capture = stream.begin_capture()?;
    gpu.blend(&features, frames, &rate, &stream)?;
    let graph = capture.finish()?;
    let iterations: usize = env::var("VOCAL_RETRIEVAL_ITERATIONS")
        .unwrap_or_else(|_| "100".into())
        .parse()?;
    assert!(iterations > 0);
    let mut results = Vec::new();
    for replay in [false, true] {
        let mut times = Vec::new();
        for i in 0..iterations + 20 {
            // Reset on device so every iteration searches identical features.
            // Record events outside capture: default captured events express
            // dependencies and do not refresh elapsed-time timestamps.
            features.device_copy_from(&source, 0, 0, input.len(), &stream)?;
            start.record(&stream)?;
            if replay {
                graph.launch(&stream)?;
            } else {
                gpu.blend(&features, frames, &rate, &stream)?;
            }
            end.record(&stream)?;
            stream.synchronize()?;
            if i >= 20 {
                times.push(start.elapsed(&end)?);
            }
        }
        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        results.push(serde_json::json!({
            "graph_replay": replay, "iterations": iterations,
            "mean_ms": times.iter().sum::<f64>() / iterations as f64,
            "median_ms": sorted[iterations / 2],
            "p95_ms": sorted[(iterations * 95 / 100).min(iterations - 1)],
            "samples_ms": times,
        }));
    }
    // Accuracy downloads happen once, after every timed iteration.
    let mut output = vec![0; input.len()];
    features.download_bytes(&mut output)?;
    fs::write(root.join("actual.f32"), &output)?;
    // A captured graph must observe rate changes without recapture.
    rate.upload_f32(&[0.0])?;
    features.device_copy_from(&source, 0, 0, input.len(), &stream)?;
    graph.launch(&stream)?;
    stream.synchronize()?;
    features.download_bytes(&mut output)?;
    assert_eq!(output, input, "ratio=0 replay must preserve query bits");
    let report = serde_json::json!({
        "frames": frames, "dimensions": index.dimensions(),
        "nlist": index.nlist(), "nprobe": index.nprobe(), "vectors": index.len(),
        "results": results,
    });
    fs::write(
        root.join("timings.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
