"""Inspect full inference WAVs, timings, envelopes and boundary discontinuities.

Independent Synth noise, DSP and SOLA choices make raw waveform metrics
descriptive, rather than numerical graph parity. Exact-input validation is
handled separately by run-python-streaming-reference.py.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile


def named_path(text):
    name, separator, path = text.partition("=")
    if not name or not separator or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def load_wav(path):
    rate, samples = wavfile.read(path)
    if samples.ndim != 1:
        raise ValueError(f"expected mono WAV: {path}")
    if samples.dtype.kind == "i":
        samples = samples.astype(np.float64) / -float(np.iinfo(samples.dtype).min)
    elif samples.dtype == np.uint8:
        samples = (samples.astype(np.float64) - 128) / 128
    else:
        samples = samples.astype(np.float64)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError(f"empty or non-finite WAV: {path}")
    return int(rate), samples


def rms(samples):
    return float(np.sqrt(np.mean(samples ** 2)))


def envelope(samples, block):
    valid = len(samples) // block * block
    return np.sqrt(np.mean(samples[:valid].reshape(-1, block) ** 2, axis=1))


def describe(rate, samples, chunk_ms):
    block = round(rate * chunk_ms / 1000)
    difference = np.abs(np.diff(samples))
    edges = np.arange(block, len(samples), block)
    jumps = difference[edges - 1]
    active = np.flatnonzero(np.abs(samples) > 1e-5)
    return {"sample_rate": rate, "samples": len(samples), "duration_s": len(samples) / rate,
            "rms": rms(samples), "peak": float(np.max(np.abs(samples))),
            "clipped_samples": int(np.sum(np.abs(samples) >= 0.999)),
            "first_above_1e_minus5_s": float(active[0] / rate) if active.size else None,
            "last_above_1e_minus5_s": float(active[-1] / rate) if active.size else None,
            "first_500ms_rms": rms(samples[:rate // 2]), "last_500ms_rms": rms(samples[-rate // 2:]),
            "boundary_jump_max": float(np.max(jumps)) if jumps.size else 0.0,
            "boundary_jump_p99": float(np.percentile(jumps, 99)) if jumps.size else 0.0,
            "all_adjacent_jump_max": float(np.max(difference)),
            "all_adjacent_jump_p99": float(np.percentile(difference, 99))}


def compare(rate, left, right):
    # Align coarse loudness envelopes only, at 20 ms resolution. Independent
    # waveform phase remains untouched; no alignment is called graph parity.
    block = round(rate * 0.020)
    a, b = envelope(left, block), envelope(right, block)
    correlations = signal.correlate(b, a, mode="full", method="fft")
    lags = signal.correlation_lags(len(b), len(a))
    limit = round(0.5 * rate / block)
    selected = np.abs(lags) <= limit
    lag = int(lags[selected][np.argmax(correlations[selected])])
    if lag > 0:
        right, b = right[lag * block:], b[lag:]
    elif lag < 0:
        left, a = left[-lag * block:], a[-lag:]
    length = min(len(a), len(b))
    a, b = a[:length], b[:length]
    amount = min(len(left), len(right))
    left, right = left[:amount], right[:amount]
    gain = np.sum(a * b) / max(np.sum(b ** 2), 1e-30)
    normalized = right * gain
    # Restrict spectral comparison to audible reference frames; low amplitude
    # silence would otherwise dominate log-magnitude differences.
    _, _, lspec = signal.stft(left, rate, nperseg=1024, noverlap=768, boundary=None)
    _, _, rspec = signal.stft(normalized, rate, nperseg=1024, noverlap=768, boundary=None)
    magnitude = np.abs(lspec)
    active = np.sqrt(np.mean(magnitude ** 2, axis=0)) > np.max(magnitude) * 1e-3
    spectral_delta = (20 * np.log10(np.maximum(np.abs(rspec), 1e-5))
                      - 20 * np.log10(np.maximum(magnitude, 1e-5)))[:, active]
    return {"right_envelope_lag_ms": lag * block / rate * 1000,
            "right_gain_to_match_left": float(gain),
            "envelope_cosine": float(np.sum(a * b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
            "envelope_relative_rmse_after_gain": rms(b * gain - a) / max(rms(a), 1e-30),
            "waveform_cosine_after_envelope_alignment": float(np.sum(left * right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-30)),
            "log_magnitude_mae_db_after_gain": float(np.mean(np.abs(spectral_delta))) if spectral_delta.size else None}


def read_json(path):
    data = path.read_bytes()
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    return json.loads(data.decode(encoding))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", type=named_path, action="append", required=True)
    ap.add_argument("--run", type=named_path, action="append", default=[])
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--normalize-dir", type=Path, help="optional separate listening WAVs at RMS 0.1, peak <= 0.9")
    ap.add_argument("--chunk-ms", type=float, default=250)
    args = ap.parse_args()
    if args.chunk_ms <= 0:
        ap.error("--chunk-ms must be positive")
    if len(dict(args.audio)) != len(args.audio):
        ap.error("audio names must be unique")
    audio = {name: load_wav(path) for name, path in args.audio}
    report = {"comparison_policy": "independent noise/DSP/SOLA; descriptive listening metrics, not graph parity",
              "audio": {}, "pairs": {}, "timings": {}}
    for name, path in args.audio:
        rate, samples = audio[name]
        report["audio"][name] = {"path": str(path), **describe(rate, samples, args.chunk_ms)}
        if args.normalize_dir:
            gain = min(0.1 / max(rms(samples), 1e-30), 0.9 / max(np.max(np.abs(samples)), 1e-30))
            args.normalize_dir.mkdir(parents=True, exist_ok=True)
            destination = args.normalize_dir / f"{name}.wav"
            wavfile.write(destination, rate, (samples * gain).astype(np.float32))
            report["audio"][name]["listening_copy"] = str(destination)
            report["audio"][name]["listening_gain_db"] = 20 * math.log10(gain)
    for left, right in itertools.combinations(audio, 2):
        if audio[left][0] != audio[right][0]:
            raise ValueError("comparison requires matching sample rates")
        report["pairs"][f"{left}_vs_{right}"] = compare(audio[left][0], audio[left][1], audio[right][1])
    for name, path in args.run:
        run = read_json(path)
        duration = audio[name][1].size / audio[name][0]
        elapsed = run["elapsed_ms"]
        report["timings"][name] = {"processing_ms_excluding_model_load": elapsed,
                                   "real_time_factor": elapsed / (duration * 1000)}
        if run.get("windows"):
            report["timings"][name]["mean_ms_per_window_including_startup"] = elapsed / run["windows"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
