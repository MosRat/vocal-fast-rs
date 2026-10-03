"""Measure HuBERT component precision against the local FP32 reference.

The experiment keeps the Transformers/Applio model and the input waveform
fixed, then lowers one graph region at a time.  FP32 casts at region
boundaries make the result useful for mapping to the native TensorRT builder:
feature extractor, feature projection, encoder, and the complete model.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import HubertModel

ROOT = Path(__file__).resolve().parents[1]


def stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    value = np.asarray(candidate, dtype=np.float64)
    delta = value - ref
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "relative_l2": float(
            np.linalg.norm(delta) / max(float(np.linalg.norm(ref)), 1.0e-12)
        ),
        "cosine": float(
            np.dot(ref.reshape(-1), value.reshape(-1))
            / max(
                float(np.linalg.norm(ref)) * float(np.linalg.norm(value)),
                1.0e-12,
            )
        ),
    }


def run_regions(
    model: HubertModel, waveform: torch.Tensor, region: str
) -> dict[str, torch.Tensor]:
    """Run the same stages as HubertModel.forward with explicit FP32 joins."""
    lower_extractor = region in {"extractor", "all_fp16"}
    lower_projection = region in {"projection", "all_fp16"}
    lower_encoder = region in {"encoder", "all_fp16"}

    if lower_extractor:
        features = model.feature_extractor(waveform.half()).float()
    else:
        features = model.feature_extractor(waveform).float()
    if lower_projection:
        projected = model.feature_projection(features.transpose(1, 2).half()).float()
    else:
        projected = model.feature_projection(features.transpose(1, 2)).float()
    if lower_encoder:
        hidden = model.encoder(projected.half()).last_hidden_state.float()
    else:
        hidden = model.encoder(projected).last_hidden_state.float()
    return {"features": features, "projected": projected, "hidden": hidden}


def timed(fn, device: torch.device, warmups: int, iterations: int) -> list[float]:
    for _ in range(warmups):
        fn()
    values: list[float] = []
    for _ in range(iterations):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            values.append(float(start.elapsed_time(end)))
        else:
            begin = time.perf_counter()
            fn()
            values.append((time.perf_counter() - begin) * 1000.0)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models/shared/hubert_base"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    audio = np.fromfile(args.audio, dtype="<f4").astype(np.float32, copy=False)
    waveform = torch.from_numpy(audio).to(device).unsqueeze(0)
    base = HubertModel.from_pretrained(
        str(args.model), local_files_only=True
    ).float().eval().to(device)

    with torch.inference_mode():
        reference = run_regions(base, waveform, "fp32")
    reference_np = {
        name: value.detach().float().cpu().numpy()
        for name, value in reference.items()
    }

    report = {
        "reference": "local Transformers/Applio HubertModel FP32",
        "audio": str(args.audio),
        "model": str(args.model),
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "shape": {
            "waveform_samples": int(audio.size),
            "features": list(reference_np["features"].shape),
            "projected": list(reference_np["projected"].shape),
            "hidden": list(reference_np["hidden"].shape),
        },
        "regions": {},
    }
    regions = ["fp32", "extractor", "projection", "encoder", "all_fp16"]
    for region in regions:
        if region == "fp32":
            model = base
        else:
            model = copy.deepcopy(base).eval()
            if region in {"extractor", "all_fp16"}:
                model.feature_extractor.half()
            if region in {"projection", "all_fp16"}:
                model.feature_projection.half()
            if region in {"encoder", "all_fp16"}:
                model.encoder.half()
        result: dict[str, torch.Tensor] | None = None

        def invoke() -> None:
            nonlocal result
            with torch.inference_mode():
                result = run_regions(model, waveform, region)

        timings = timed(invoke, device, args.warmups, args.iterations)
        assert result is not None
        result_np = {
            name: value.detach().float().cpu().numpy()
            for name, value in result.items()
        }
        report["regions"][region] = {
            "latency_ms": {
                "mean": float(np.mean(timings)),
                "median": float(np.median(timings)),
                "p95": float(np.percentile(timings, 95)),
            },
            "stages": {
                name: stats(reference_np[name], result_np[name])
                for name in ("features", "projected", "hidden")
            },
            "finite": bool(all(np.isfinite(value).all() for value in result_np.values())),
        }
        if region != "fp32":
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
