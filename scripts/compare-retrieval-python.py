"""Validate a native retrieval snapshot against FAISS and Applio FP32.

Use debug snapshots from fresh native runs of the same input, one with
VOCAL_INDEX_RATIO=0 and one with the requested ratio. The native debug file
must include the exact latent/NSF random inputs and Synth crop metadata.
This isolates retrieval and Synth; it does not claim offline/realtime
acoustic contexts or stochastic, independently seeded waveforms are equal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
import torch
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref_repos" / "Applio"))
from rvc.lib.algorithm.synthesizers import Synthesizer  # noqa: E402


def metric(reference, actual) -> dict:
    a, b = (np.asarray(value, dtype=np.float64).reshape(-1) for value in (reference, actual))
    if a.shape != b.shape or not a.size:
        raise ValueError(f"incompatible shapes: {a.shape}, {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("non-finite comparison input")
    delta = b - a
    return {
        "elements": int(a.size), "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "cosine": float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
        "reference_rms": float(np.sqrt(np.mean(a ** 2))),
        "native_rms": float(np.sqrt(np.mean(b ** 2))),
    }


def retrieval_reference(query: np.ndarray, index_path: Path, ratio: float) -> np.ndarray:
    # FAISS's Windows fopen path cannot open non-ASCII voice model filenames.
    index = faiss.deserialize_index(np.frombuffer(index_path.read_bytes(), dtype=np.uint8))
    scores, ids = index.search(np.ascontiguousarray(query, dtype=np.float32), 8)
    if (ids < 0).all(axis=1).any() or not np.isfinite(scores).all():
        raise ValueError("index has no candidates or non-finite distances for a query")
    vectors = index.reconstruct_n(0, index.ntotal)
    # The ordinary path is the exact RVC/Applio inverse-square weighting.
    # At zero distance select the first exact match, as the native guard does.
    weights = np.zeros_like(scores)
    exact = scores[:, 0] <= 1e-12
    weights[exact, 0] = 1
    weights[~exact] = np.square(1 / scores[~exact])
    weights[ids < 0] = 0
    weights[~exact] /= weights[~exact].sum(axis=1, keepdims=True)
    retrieved = np.sum(vectors[ids.clip(min=0)] * weights[..., None], axis=1)
    return ratio * retrieved + (1 - ratio) * query


def load_synth(checkpoint: Path, device: str):
    cpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = list(cpt["config"])
    config[-3] = cpt["weight"]["emb_g.weight"].shape[0]
    width = int(cpt["weight"]["enc_p.emb_phone.weight"].shape[1])
    if width not in (256, 768):
        raise ValueError(f"unsupported RVC feature width {width}")
    net = Synthesizer(*config, use_f0=bool(cpt.get("f0", 1)),
                      text_enc_hidden_dim=width, vocoder=cpt.get("vocoder", "HiFi-GAN"))
    if hasattr(net, "enc_q"):
        del net.enc_q
    weights = {k: v for k, v in cpt["weight"].items() if not k.startswith("enc_q.")}
    net.load_state_dict(weights, strict=True)
    return net.to(device).float().eval(), width, int(config[-1])


@torch.inference_mode()
def synth_reference(net, snapshot: dict, width: int, device: str) -> np.ndarray:
    required = ("synth_random", "synth_noise", "valid_synth_frames", "synth_skip_head", "synth_output_frames")
    if any(snapshot.get(key) is None for key in required):
        raise ValueError("rerun native diagnostics with exact Synth inputs/crop metadata")
    frames = int(snapshot["synth_frames"])
    skip = int(snapshot["synth_skip_head"])
    output_frames = int(snapshot["synth_output_frames"])
    valid = int(snapshot["valid_synth_frames"])
    if not 0 < valid <= frames or not 0 <= skip < frames or output_frames > frames - skip:
        raise ValueError("invalid Synth frame/crop metadata")
    tensor = lambda key: torch.tensor(snapshot[key], dtype=torch.float32, device=device)
    feats = tensor("rvc_features").reshape(1, frames, width)
    pitch = tensor("pitch").long().reshape(1, frames)
    f0 = tensor("aligned_pitchf").reshape(1, frames)
    rnd = tensor("synth_random").reshape(1, -1, frames)
    noise = tensor("synth_noise").reshape(1, 1, -1)
    upsample = noise.shape[-1] // frames
    if noise.shape[-1] != frames * upsample:
        raise ValueError("NSF noise length is not frame aligned")
    g = net.emb_g(torch.tensor([0], device=device)).unsqueeze(-1)
    mean, logs, mask = net.enc_p(feats, pitch, torch.tensor([valid], device=device))
    z = (mean + logs.exp() * rnd * 0.66666) * mask
    # WebUI realtime keeps 24 frames of flow context before skip_head;
    # this is the crop used by the fixed native tail plan as well.
    flow_head = max(skip - 24, 0)
    z = net.flow(z[:, :, flow_head:], mask[:, :, flow_head:], g=g, reverse=True)
    z = (z * mask[:, :, flow_head:])[:, :, skip - flow_head:]
    source_noise = noise[:, :, skip * upsample:].transpose(1, 2)

    def exact_noise(value, *args, **kwargs):
        if value.shape != source_noise.shape:
            raise ValueError(f"unexpected Applio noise shape {value.shape}, expected {source_noise.shape}")
        return source_noise.to(value)

    # Run Applio's real NSF source/decoder, replacing only random sampling.
    with patch.object(torch, "randn_like", side_effect=exact_noise):
        audio = net.dec(z, f0[:, skip:], g=g).reshape(-1)
    return audio[:output_frames * upsample].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--debug", type=Path, required=True)
    ap.add_argument("--raw-debug", type=Path, help="same input, fresh native run with index ratio 0")
    ap.add_argument("--index", type=Path)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--audio-dir", type=Path)
    args = ap.parse_args()
    if not 0 <= args.ratio <= 1 or bool(args.raw_debug) != bool(args.index):
        ap.error("ratio must be in [0,1]; supply both --raw-debug and --index")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    j = json.loads(args.debug.read_text(encoding="utf-8"))
    net, width, sample_rate = load_synth(args.checkpoint, args.device)
    reference = synth_reference(net, j, width, args.device)
    report = {"reference": "original PTH + Applio FP32 Synth, exact native inputs and crop",
              "checkpoint": str(args.checkpoint), "debug": str(args.debug),
              "synth": metric(reference, j["audio"]),
              "frames": {key: j[key] for key in ("synth_frames", "valid_synth_frames", "synth_skip_head", "synth_output_frames")}}
    if args.raw_debug:
        raw = json.loads(args.raw_debug.read_text(encoding="utf-8"))
        for key in ("synth_frames", "aligned_pitchf", "synth_random", "synth_noise"):
            if raw[key] != j[key]:
                raise ValueError(f"snapshots have different {key}; use fresh runs with the same input")
        query = np.asarray(raw["hubert_features"], dtype=np.float32).reshape(-1, width)
        expected = retrieval_reference(query, args.index, args.ratio)
        report["retrieval"] = metric(expected, j["hubert_features"])
        upsampled = np.repeat(np.concatenate([expected, expected[-1:]]), 2, axis=0)[:j["rvc_frames"]]
        report["frontend_features"] = metric(upsampled, j["rvc_features"])
    if args.audio_dir:
        args.audio_dir.mkdir(parents=True, exist_ok=True)
        wavfile.write(str(args.audio_dir / "python-synth.wav"), sample_rate, reference)
        wavfile.write(str(args.audio_dir / "native-synth.wav"), sample_rate, np.asarray(j["audio"], np.float32))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
