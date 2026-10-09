"""Compare boundary DSP on identical recorded decoder tails.

Record tails with vocal-directml's streaming validator, then replay them with
vocal-engine's ignored replay_real_decoder_tails test. This measures signal
preservation; it does not substitute for listening or hardware latency tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def legacy_join(audio, previous, block, overlap, search, initialized):
    energy = np.sum(previous.astype(np.float64) ** 2)
    offset = 0
    if initialized and energy > 1e-12:
        candidates = np.lib.stride_tricks.sliding_window_view(
            audio[:overlap + search].astype(np.float64), overlap
        )
        scores = (candidates @ previous) / np.sqrt(
            energy * np.sum(candidates ** 2, axis=1) + 1e-8
        )
        offset = int(np.argmax(scores))
    current = audio[offset:]
    output = current[:block].copy()
    fade = np.sin(np.linspace(0, np.pi / 2, overlap)) ** 2
    output[:overlap] = previous * (1 - fade) + output[:overlap] * fade
    return output, current[block:block + overlap].copy(), offset


def rms(audio):
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.trace / "trace.json").read_text())
    rate = metadata["sample_rate"]
    block = round(metadata["advance_samples"] * rate / 16000)
    overlap, search = round(rate * 0.05), round(rate * 0.01)
    warmup = 13  # Historical DSP replay; startup removal is tested separately.
    previous = np.zeros(overlap, np.float32)
    before_blocks, offsets = [], []
    for i in range(metadata["chunks"]):
        audio = np.fromfile(args.trace / f"{i:04}/audio.bin", dtype="<f4")
        if not np.isfinite(audio).all() or len(audio) < block + overlap + search:
            raise ValueError(f"invalid decoder tail at {i}")
        if i < warmup:
            before_blocks.append(np.zeros(block, np.float32))
            offsets.append(0)
        else:
            output, previous, offset = legacy_join(
                audio, previous, block, overlap, search, i > warmup
            )
            before_blocks.append(output)
            offsets.append(offset)
    before = np.concatenate(before_blocks)
    # Use matching PCM16 quantization for the waveform comparison.
    scaled = np.clip(before, -1, 1) * np.float32(32767)
    before = (np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)).astype(np.int16)
    before_path = args.report.with_name("trace-before.wav")
    wavfile.write(before_path, rate, before)
    after_rate, after = wavfile.read(args.after)
    if after_rate != rate or after.dtype != np.int16 or after.shape != before.shape:
        raise ValueError("before/after must have matching sample count, rate and PCM16 format")
    before, after = before.astype(np.float64) / 32768, after.astype(np.float64) / 32768
    rows = []
    for i in range(warmup, metadata["chunks"]):
        start = i * block
        old_rms, new_rms = rms(before[start:start + overlap]), rms(after[start:start + overlap])
        rows.append({
            "chunk": i, "time_s": start / rate,
            "legacy_offset_ms": offsets[i] * 1000 / rate,
            "before_overlap_rms": old_rms, "after_overlap_rms": new_rms,
            "overlap_energy_change_db": 20 * np.log10(max(new_rms, 1e-12) / max(old_rms, 1e-12)),
        })
    report = {
        "mode": "identical_decoder_tails_dsp_only", "physical_devices": False,
        "before": str(before_path), "after": str(args.after), "rate": rate,
        "chunks": metadata["chunks"], "samples": len(before),
        "before_rms": rms(before), "after_rms": rms(after),
        "before_clip_fraction": float(np.mean(np.abs(before) >= 32767 / 32768)),
        "after_clip_fraction": float(np.mean(np.abs(after) >= 32767 / 32768)),
        "note": "An energy increase alone does not prove better perceptual quality.",
        "boundaries": rows,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "boundaries"}, indent=2))


if __name__ == "__main__":
    main()
