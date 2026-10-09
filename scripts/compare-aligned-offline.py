"""Compare full-context native traces with original Applio and exact-input Synth.

The original Applio runner must share native latent/NSF noise. Stage comparisons
include native acoustic/retrieval error; exact Synth comparisons hold those
inputs fixed and isolate the handwritten generator graph.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("retrieval_reference", ROOT / "scripts/compare-retrieval-python.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def load(path, name):
    return np.fromfile(path / f"{name}.bin", dtype="<i4" if name == "pitch" else "<f4")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--native-wav", type=Path, required=True)
    ap.add_argument("--reference-wav", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models/voices/default-v2/default.pth")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = ap.parse_args()
    meta = json.loads((args.native / "metadata.json").read_text())
    if not meta.get("offline") or meta.get("skip_head") != 0:
        ap.error("requires full-context offline traces")
    metrics = {}
    for name in ("waveform", "hubert_raw", "feats", "pitch", "pitchf", "audio"):
        if (args.native / f"{name}.bin").exists() and (args.reference / f"{name}.bin").exists():
            metrics[name] = reference.metric(load(args.reference, name), load(args.native, name))
    for key, path in [("reference", args.reference_wav), ("native", args.native_wav)]:
        rate, audio = wavfile.read(path)
        if rate != meta["sample_rate"]:
            raise ValueError("WAV must retain the model sample rate")
        if audio.dtype.kind == "i":
            audio = audio.astype(np.float32) / -float(np.iinfo(audio.dtype).min)
        else:
            audio = audio.astype(np.float32)
        if key == "reference":
            ref_audio = audio
        else:
            native_audio = audio
    metrics["final_wav"] = reference.metric(ref_audio, native_audio)
    metrics["final_wav"]["abs_error_percentiles_99_999"] = np.percentile(np.abs(ref_audio - native_audio), [99, 99.9]).tolist()
    f0_error = load(args.native, "pitchf") - load(args.reference, "pitchf")
    pitch_error = load(args.native, "pitch") != load(args.reference, "pitch")
    phase = np.cumsum(f0_error.astype(np.float64)) * 0.01
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    net, width, _ = reference.load_synth(args.checkpoint, args.device)
    snapshot = {
        "synth_frames": meta["frames"], "valid_synth_frames": meta["frames"],
        "synth_skip_head": 0, "synth_output_frames": meta["frames"],
    }
    for key, name in [("synth_random", "synth_random"), ("synth_noise", "synth_noise"),
                      ("rvc_features", "feats"), ("pitch", "pitch"), ("aligned_pitchf", "pitchf")]:
        snapshot[key] = load(args.native, name)
    exact = reference.synth_reference(net, snapshot, width, args.device)
    metrics["exact_input_synth"] = reference.metric(exact, load(args.native, "audio"))
    report = {
        "reference": "Applio FP32 Synth/offline pipeline, standard RVC acoustic contract, shared native noise",
        "native": str(args.native), "reference_trace": str(args.reference), "metadata": meta,
        "metrics": metrics,
        "f0_diagnostics": {"coarse_mismatch_frames": int(pitch_error.sum()),
                           "cumulative_phase_max_cycles": float(np.abs(phase).max()),
                           "cumulative_phase_final_cycles": float(phase[-1])},
        "listening_outputs": {"native": str(args.native_wav), "reference": str(args.reference_wav)},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
