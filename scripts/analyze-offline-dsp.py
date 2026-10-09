"""Isolate output resampling and RMS policies using existing WAVs/tail traces.

No neural inference is performed. The native-rate diagnostic joins the same
recorded Synth tails at their original rate. The RMS ablation locks its SOLA
offsets to that diagnostic so only the gain policy changes. These files are
DSP experiments, not newly validated production backend outputs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]


def script_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cli_resample(audio, input_rate, output_rate):
    """Match vocal_audio::resample_linear's length and f64 position arithmetic."""
    audio = np.asarray(audio, np.float32)
    if input_rate == output_rate:
        return audio.copy()
    size = max(len(audio) * output_rate // input_rate, 1)
    position = np.arange(size, dtype=np.float64) * (input_rate / output_rate)
    left = np.floor(position).astype(np.int64)
    right = np.minimum(left + 1, len(audio) - 1)
    weight = (position - left).astype(np.float32)
    return audio[np.minimum(left, len(audio) - 1)] * (1 - weight) + audio[right] * weight


def metrics(reference, actual, dsp):
    stat = dsp.Errors()
    stat.add(reference, actual, 0)
    return stat.report()


def band_power(audio, rate):
    frequencies, psd = signal.welch(audio.astype(np.float64), rate, nperseg=4096)
    total = max(float(psd.sum()), 1e-30)
    return {f"{low}-{high}Hz": float(psd[(frequencies >= low) & (frequencies < high)].sum() / total)
            for low, high in [(0, 4000), (4000, 8000), (8000, rate // 2)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--applio", type=Path, required=True)
    ap.add_argument("--protect-ablation", type=Path,
                    help="Applio output with identical settings/seed except protect")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--input", type=Path)
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--native-output", type=Path)
    args = ap.parse_args()
    trace_options = (args.input, args.trace, args.native_output)
    if any(trace_options) and not all(trace_options):
        ap.error("--input, --trace and --native-output must be provided together")
    inspection = script_module("wav_inspection", "compare-offline-wavs.py")
    # Reuse already validated streaming DSP helpers, without loading any models.
    dsp = script_module("streaming_dsp", "run-python-streaming-reference.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    listening = args.output_dir / "listening"
    listening.mkdir(exist_ok=True)
    report = {"policy": "DSP-only ablations; identical recorded neural tails, no graph inference",
              "files": {}, "resampling": {}}

    def save(name, rate, audio):
        audio = np.asarray(audio, np.float32)
        if not audio.size or not np.isfinite(audio).all():
            raise ValueError(f"invalid diagnostic audio: {name}")
        destination = args.output_dir / f"{name}.wav"
        wavfile.write(destination, rate, audio)
        gain = min(0.1 / max(inspection.rms(audio), 1e-30),
                   0.9 / max(float(np.abs(audio).max()), 1e-30))
        normalized = listening / destination.name
        wavfile.write(normalized, rate, audio * gain)
        report["files"][name] = {"path": str(destination), "listening_copy": str(normalized),
                                 "constant_listening_gain": gain,
                                 "sample_rate": rate, "samples": len(audio),
                                 "rms": inspection.rms(audio), "bands": band_power(audio, rate)}

    rate, original = inspection.load_wav(args.applio)
    if rate <= 16000:
        raise ValueError("resampling ablation needs a Synth WAV above 16 kHz")
    save("applio-native", rate, original)
    roundtrip = cli_resample(dsp.resample_linear(original.astype(np.float32), rate, 16000), 16000, rate)
    if len(roundtrip) != len(original):
        raise ValueError("roundtrip length differs; use a whole 16 kHz-aligned source duration")
    save("applio-linear-roundtrip", rate, roundtrip)
    report["resampling"]["applio_native_vs_linear_roundtrip"] = metrics(original, roundtrip, dsp)
    if args.protect_ablation:
        protected_rate, protected = inspection.load_wav(args.protect_ablation)
        if protected_rate != rate or protected.shape != original.shape:
            raise ValueError("protect ablation must have the same rate and length")
        save("applio-protect-ablation", rate, protected)
        report["protect_ablation"] = {"path": str(args.protect_ablation),
                                      "metrics": metrics(original, protected, dsp)}
    report["resampling"]["band_note"] = "Upsampling images above 8 kHz are not restored original high-frequency content."
    # Demonstrate that interpolation-only downsampling does not reject aliases.
    tone_hz = 12000
    tone = np.sin(2 * np.pi * tone_hz * np.arange(rate) / rate).astype(np.float32)
    down = dsp.resample_linear(tone, rate, 16000)
    frequency, spectrum = signal.periodogram(down, 16000)
    report["resampling"]["alias_probe"] = {"input_tone_hz": tone_hz,
                                            "downsampled_peak_hz": float(frequency[np.argmax(spectrum)])}

    if args.trace:
        metadata = json.loads((args.trace / "trace.json").read_text(encoding="utf-8"))
        source = dsp.read_wave(args.input)
        config, advance = metadata["config"], metadata["advance_samples"]
        if (config["window_samples"], advance) != (44960, 4000):
            raise ValueError("tail DSP reconstruction currently supports the 44960/4000 preset")
        if metadata["input_samples"] != len(source) or metadata["chunks"] != math.ceil(len(source) / advance):
            raise ValueError("trace must cover the complete input")
        synth_rate = metadata["sample_rate"]
        if synth_rate != rate:
            raise ValueError("Applio and native trace must have the same Synth sample rate")
        warmup = math.ceil(config["window_samples"] / advance) + 1
        chunk_out, overlap, search = (n * rate // 16000 for n in (advance, 800, 160))
        current_join = dsp.Joiner()
        rate_join = dsp.Joiner(overlap, search)
        no_rms_join = dsp.Joiner(overlap, search)
        level = np.zeros(advance + 800, np.float32)
        current_blocks, rate_blocks, no_rms_blocks, gains, offset_changes = [], [], [], [], []
        for i in range(metadata["chunks"]):
            chunk = source[i * advance:(i + 1) * advance]
            valid = len(chunk)
            chunk = np.pad(chunk, (0, advance - valid))
            level[:-advance] = level[advance:].copy()
            level[-advance:] = chunk
            gain = np.float32(np.sqrt(np.mean(level ** 2)))
            tail = np.fromfile(args.trace / f"{i:04d}" / "audio.bin", dtype="<f4")
            required = chunk_out + overlap + search
            if tail.size < required or not np.isfinite(tail).all():
                raise ValueError(f"invalid extended Synth tail at chunk {i}")
            if i < warmup:
                current = np.zeros(advance, np.float32)
                native_rate = no_rms = np.zeros(chunk_out, np.float32)
            else:
                gains.append(float(gain))
                current, offset16 = current_join.process(dsp.resample_linear(tail, rate, 16000) * gain, advance)
                native_rate, offset_native = rate_join.process(tail * gain, chunk_out)
                no_rms, _ = no_rms_join.process(tail, chunk_out, offset_native)
                if offset_native != round(offset16 * rate / 16000):
                    offset_changes.append(i)
            current_blocks.append(current[:valid])
            rate_blocks.append(native_rate[:valid * rate // 16000])
            no_rms_blocks.append(no_rms[:valid * rate // 16000])
        reconstructed = cli_resample(np.concatenate(current_blocks), 16000, rate)
        rate_audio, no_rms_audio = np.concatenate(rate_blocks), np.concatenate(no_rms_blocks)
        production_rate, production = inspection.load_wav(args.native_output)
        if production_rate != rate:
            raise ValueError("production WAV must have the native Synth rate")
        reconstruction_metrics = metrics(production, dsp.pcm16(reconstructed).astype(np.float32) / 32768, dsp)
        if reconstruction_metrics["cosine"] < 0.9999:
            raise ValueError(f"DSP reconstruction does not match production: {reconstruction_metrics}")
        save("native-current-dsp", rate, reconstructed)
        save("native-rate-dsp", rate, rate_audio)
        save("native-rate-without-rms", rate, no_rms_audio)
        report["trace"] = {"path": str(args.trace), "backend": metadata.get("backend", "directml"),
                           "warmup_blocks": warmup, "production_reconstruction": reconstruction_metrics,
                           "native_rate_changed_sola_chunks": offset_changes,
                           "native_rate_note": "Independent native-rate SOLA choices; not a pure resampling parity comparison.",
                           "rms_ablation_note": "Same native-rate SOLA offsets; only the rolling input-RMS multiplier is removed.",
                           "rolling_rms_percentiles_5_50_95": np.percentile(gains, [5, 50, 95]).tolist(),
                           "rms_ablation_envelope": inspection.compare(rate, rate_audio, no_rms_audio)}
    destination = args.output_dir / "dsp-analysis.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
