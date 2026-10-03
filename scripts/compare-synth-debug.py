"""Compare native pipeline debug audio with the Applio FP32 Synth path.

The debug file contains the exact native frontend outputs.  The script uses
the same deterministic latent and NSF noise sequence as
``RvcTrtPipeline::seed_static_inputs`` so the comparison isolates the Synth
graph rather than HuBERT/RMVPE.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PRECISION = ROOT / "scripts" / "synth-precision-applio.py"
spec = importlib.util.spec_from_file_location("synth_precision", PRECISION)
mod = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(mod)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", type=Path, required=True)
    ap.add_argument("--vfrm", type=Path, default=ROOT / "target/tsukuyomi-canonical.vfr")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--direct-native", type=Path, help="optional direct Synth output (.f32) for pipeline binding parity")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()
    debug = json.loads(args.debug.read_text(encoding="utf-8"))
    config, tensors = mod.read_vfrm(args.vfrm)
    c = config["metadata"]["rvc_config"]
    frames = int(debug["rvc_frames"])
    latent = int(c["latent_channels"])
    up = int(np.prod(c["upsample_rates"], dtype=np.int64))
    feats = np.asarray(debug["rvc_features"], dtype=np.float32).reshape(1, frames, 768)
    pitch = np.asarray(debug["pitch"], dtype=np.int64).reshape(1, frames)
    pitchf = np.asarray(debug["aligned_pitchf"], dtype=np.float32).reshape(1, frames)
    rnd = np.asarray([0.01 * np.sin(i * 0.017) for i in range(latent * frames)], dtype=np.float32).reshape(1, latent, frames)
    noise = np.asarray([0.001 * np.sin(i * 0.013) for i in range(frames * up)], dtype=np.float32).reshape(1, 1, frames * up)
    inp = {"feats": feats, "pitch": pitch.astype(np.int32), "pitchf": pitchf, "rnd": rnd,
           "noise": noise, "frames": frames, "p_len": frames, "sid": 0, "upsample": up}
    net, _ = mod.build_reference(config, tensors, noise)
    device = torch.device(args.device)
    net.to(device).eval()
    with torch.inference_mode():
        result = mod.run_variant(net, inp, device, "fp32")
    native = np.asarray(json.loads(args.debug.read_text(encoding="utf-8"))["audio"], dtype=np.float32)
    reference = result["audio"].detach().cpu().numpy().astype(np.float32).reshape(-1)
    if native.shape != reference.shape:
        raise ValueError(f"shape mismatch native={native.shape}, reference={reference.shape}")
    metrics = mod.metric(torch.from_numpy(reference), torch.from_numpy(native))
    direct_metrics = None
    if args.direct_native:
        direct = np.fromfile(args.direct_native, dtype="<f4")
        if direct.shape != native.shape:
            raise ValueError(f"shape mismatch direct={direct.shape}, pipeline={native.shape}")
        direct_metrics = mod.metric(torch.from_numpy(direct), torch.from_numpy(native))
    report = {"reference": "Applio Synthesizer PyTorch FP32", "native_debug": str(args.debug),
              "device": str(device), "frames": frames, "audio_samples": int(native.size),
              "metrics": metrics, "native_stats": {"min": float(native.min()), "max": float(native.max()),
              "mean": float(native.mean()), "std": float(native.std())},
              "reference_stats": {"min": float(reference.min()), "max": float(reference.max()),
              "mean": float(reference.mean()), "std": float(reference.std())}}
    report["pipeline_vs_direct_native"] = direct_metrics
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.save(args.output.with_suffix(".reference.npy"), reference)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
