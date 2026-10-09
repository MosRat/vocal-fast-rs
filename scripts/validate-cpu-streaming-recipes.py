"""Compare streaming recipes to a separately verified original FP32 reference.

First run run-python-streaming-reference.py for an FP32 diagnostic directory.
This tool verifies identical input, sources, controls and exact latent/NSF noise
before comparing other engine WAVs with that original Python WAV. Stage errors
are explicitly relative to the native FP32 trace, whose Python parity is retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def metrics(a, b):
    a, b = [np.asarray(v, np.float64).reshape(-1) for v in [a, b]]
    if not a.size or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("non-finite or mismatched audio")
    d = b - a
    energy = float(np.dot(a, a))
    error = float(np.dot(d, d))
    return {"cosine": float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
            "rmse": float(np.sqrt(error / len(a))), "max_abs": float(np.abs(d).max()),
            "snr_db": float(10 * np.log10(max(energy, 1e-30) / max(error, 1e-30))),
            "reference_rms": float(np.sqrt(energy / len(a)))}


def wave(path):
    rate, audio = wavfile.read(path)
    if audio.ndim != 1:
        raise ValueError("mono WAV required")
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768
    elif audio.dtype != np.float32:
        raise ValueError("PCM16 or FP32 WAV required")
    return rate, audio


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-directory", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, nargs="+", required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    baseline = read(args.reference_directory / "benchmark.json")
    parity = read(args.reference_directory / "python-parity.json")
    rate, python = wave(parity["output"])
    native_rate, fp32 = wave(baseline["audio"])
    if rate != native_rate:
        raise ValueError("reference rates differ")
    reference_trace = Path(baseline["trace"])
    reference_meta = read(reference_trace / "trace.json")
    if reference_meta["backend"] != "cpu" or baseline["quantized"]:
        raise ValueError("reference must be an unquantized CPU trace")
    if parity["sola_offset_mismatches"] or parity["waveform_metrics"]["python_vs_native_full_wav"]["cosine"] < 0.99999:
        raise ValueError("FP32 reference has not established whole-stream parity")
    report = {"reference": parity["reference"], "reference_parity": parity,
              "reference_python_sha256": digest(parity["output"]),
              "reference_input_sha256": digest(baseline["input"]), "results": []}
    for directory in args.candidates:
        raw = read(directory / "benchmark.json")
        trace = Path(raw["trace"])
        meta = read(trace / "trace.json")
        if raw["sources"] != baseline["sources"] or digest(raw["input"]) != report["reference_input_sha256"]:
            raise ValueError("unpaired input or model sources")
        for key in ["controls", "advance_samples", "chunks", "input_samples", "sample_rate", "output_rate", "frames", "boundary"]:
            if meta[key] != reference_meta[key]:
                raise ValueError(f"different streaming contract: {key}")
        for key in ["f0", "window_samples", "skip_head_frames", "append_hubert_frame"]:
            if meta["config"][key] != reference_meta["config"][key]:
                raise ValueError(f"different graph geometry: {key}")
        candidate_rate, audio = wave(raw["audio"])
        if candidate_rate != rate:
            raise ValueError("candidate rate differs")
        chunks, mismatches = [], 0
        stages = {name: 0 for name in ["hubert_raw", "rmvpe_pitchf", "pitchf", "f0_cache", "audio"]}
        for i in range(meta["chunks"]):
            old, new = reference_trace / f"{i:04}", trace / f"{i:04}"
            for name in ["synth_random", "synth_noise"]:
                if (old / f"{name}.bin").read_bytes() != (new / f"{name}.bin").read_bytes():
                    raise ValueError(f"unpaired random input at chunk {i}")
            for name in stages:
                stages[name] += (old / f"{name}.bin").read_bytes() != (new / f"{name}.bin").read_bytes()
            a = np.fromfile(old / "pitch.bin", dtype="<i4")
            b = np.fromfile(new / "pitch.bin", dtype="<i4")
            if a.shape != b.shape:
                raise ValueError("pitch shape differs")
            mismatches += int(np.count_nonzero(a != b))
            if i >= raw["warmup_blocks"]:
                a = np.fromfile(old / "audio.bin", dtype="<f4")
                b = np.fromfile(new / "audio.bin", dtype="<f4")
                chunks.append({"chunk": i, **metrics(a, b)})
        result = {"directory": str(directory), "quantized_layers": len(raw["quantized"]),
                  "python_full_wav": metrics(python, audio), "native_fp32_full_wav": metrics(fp32, audio),
                  "native_fp32_byte_different_chunks": stages, "coarse_mismatches_vs_native_fp32": mismatches,
                  "native_fp32_tail_chunk_metrics": chunks,
                  "native_fp32_quiet_tail_chunks": [x for x in chunks if x["reference_rms"] < 0.01],
                  "wav_byte_identical_to_native_fp32": digest(raw["audio"]) == digest(baseline["audio"])}
        report["results"].append(result)
        print(json.dumps({k: v for k, v in result.items() if "chunk_metrics" not in k and "quiet_tail" not in k}), flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
