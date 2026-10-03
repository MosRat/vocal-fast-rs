"""Measure RMVPE region precision against the Applio FP32 reference.

The experiment keeps STFT/mel extraction in FP32 and changes one native graph
region at a time.  It reports hidden/logit/F0 errors and CUDA event latency so
the result can be mapped directly to the TensorRT RMVPE precision fields.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
APPLIO = ROOT / "ref_repos" / "Applio"
sys.path.insert(0, str(APPLIO))

from rvc.lib.predictors.RMVPE import RMVPE0Predictor  # noqa: E402


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


def quantize_f0(f0: np.ndarray) -> np.ndarray:
    f0 = np.asarray(f0, dtype=np.float32)
    mel = 1127.0 * np.log1p(np.maximum(f0, 0.0) / 700.0)
    mel_min = 1127.0 * np.log1p(50.0 / 700.0)
    mel_max = 1127.0 * np.log1p(1100.0 / 700.0)
    mapped = np.where(
        f0 > 0, (mel - mel_min) * 254.0 / (mel_max - mel_min) + 1.0, 1.0
    )
    return np.clip(np.rint(mapped), 1, 255).astype(np.int32)


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


def run_regions(net, mel: torch.Tensor, region: str) -> dict[str, torch.Tensor]:
    """Run E2E while lowering one explicit graph region at a time.

    ``DeepUnet`` has three trained regions (encoder, intermediate and
    decoder).  Calling ``net.unet.half()`` as one unit hides which boundary is
    responsible for a coarse-F0 change, so the reference path is expanded here
    with explicit FP32 joins.  Decoder skip tensors are cast together with the
    decoder input to match the native TensorRT region contract.
    """
    use_unet = region in {"unet", "unet_head", "unet_gru", "all_fp16"}
    use_encoder = use_unet or region in {"encoder", "encoder_head"}
    use_intermediate = use_unet or region in {"intermediate", "intermediate_head"}
    use_decoder = use_unet or region in {"decoder", "decoder_head"}
    use_head = region in {"head", "unet_head", "encoder_head", "intermediate_head", "decoder_head", "all_fp16"}
    use_gru = region in {"gru", "unet_gru", "all_fp16"}
    use_logits = region in {"logits", "all_fp16"}

    x = mel.transpose(-1, -2).unsqueeze(1)
    encoder_input = x.half() if use_encoder else x
    encoded, skips = net.unet.encoder(encoder_input)
    encoded = encoded.float()
    skips = [value.float() for value in skips]

    intermediate_input = encoded.half() if use_intermediate else encoded
    intermediate = net.unet.intermediate(intermediate_input).float()
    decoder_input = intermediate.half() if use_decoder else intermediate
    decoder_skips = [value.half() if use_decoder else value for value in skips]
    unet = net.unet.decoder(decoder_input, decoder_skips).float()

    head_input = unet.half() if use_head else unet
    head = net.cnn(head_input).float()
    sequence = head.transpose(1, 2).flatten(-2)

    gru_input = sequence.half() if use_gru else sequence
    gru = net.fc[0](gru_input).float()
    logits_input = gru.half() if use_logits else gru
    logits = net.fc[1](logits_input).float()
    salience = torch.sigmoid(logits)
    return {
        "unet": unet,
        "head": head,
        "gru": gru,
        "logits": logits,
        "salience": salience,
    }


def run_variant(
    base,
    mel: torch.Tensor,
    region: str,
    device: torch.device,
    warmups: int,
    iterations: int,
) -> tuple[dict[str, torch.Tensor], list[float]]:
    net = copy.deepcopy(base).eval()
    if region in {"unet", "unet_head", "unet_gru", "all_fp16", "encoder", "encoder_head"}:
        net.unet.encoder.half()
    if region in {"unet", "unet_head", "unet_gru", "all_fp16", "intermediate", "intermediate_head"}:
        net.unet.intermediate.half()
    if region in {"unet", "unet_head", "unet_gru", "all_fp16", "decoder", "decoder_head"}:
        net.unet.decoder.half()
    if region in {
        "head",
        "unet_head",
        "encoder_head",
        "intermediate_head",
        "decoder_head",
        "all_fp16",
    }:
        net.cnn.half()
    if region in {"gru", "unet_gru", "all_fp16"}:
        net.fc[0].half()
    if region in {"logits", "all_fp16"}:
        net.fc[1].half()

    result: dict[str, torch.Tensor] | None = None

    def invoke():
        nonlocal result
        with torch.inference_mode():
            result = run_regions(net, mel, region)

    timings = timed(invoke, device, warmups, iterations)
    assert result is not None
    output = {name: value.detach().float() for name, value in result.items()}
    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output, timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/shared/rmvpe/rmvpe.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    audio = np.fromfile(args.audio, dtype="<f4").astype(np.float32, copy=False)
    predictor = RMVPE0Predictor(str(args.model), device=str(device))
    base = predictor.model.eval()
    waveform = torch.from_numpy(audio).to(device).unsqueeze(0)
    with torch.inference_mode():
        mel = predictor.mel_extractor(waveform, center=True)
        original_frames = int(mel.shape[-1])
        padded_frames = 32 * ((original_frames - 1) // 32 + 1)
        mel = F.pad(mel, (0, padded_frames - original_frames), mode="reflect")

    regions = [
        "fp32",
        "encoder",
        "intermediate",
        "decoder",
        "unet",
        "head",
        "gru",
        "logits",
        "unet_head",
        "unet_gru",
        "all_fp16",
    ]
    with torch.inference_mode():
        reference = run_regions(base, mel, "fp32")
    reference_np = {
        name: value[..., :original_frames, :].detach().cpu().numpy()
        if name in {"unet", "head"}
        else value[:, :original_frames].detach().cpu().numpy()
        for name, value in reference.items()
    }
    reference_f0 = predictor.decode(reference_np["salience"][0])
    reference_coarse = quantize_f0(reference_f0)

    report = {
        "reference": "Applio RMVPE E2E FP32",
        "audio": str(args.audio),
        "model": str(args.model),
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "shape": {
            "waveform_samples": int(audio.size),
            "mel_frames": original_frames,
            "padded_frames": padded_frames,
        },
        "regions": {},
    }
    for region in regions:
        if region == "fp32":
            result = reference
            timings = timed(
                lambda: run_regions(base, mel, "fp32"),
                device,
                args.warmups,
                args.iterations,
            )
        else:
            result, timings = run_variant(
                base, mel, region, device, args.warmups, args.iterations
            )
        result_np = {
            name: value[..., :original_frames, :].detach().cpu().numpy()
            if name in {"unet", "head"}
            else value[:, :original_frames].detach().cpu().numpy()
            for name, value in result.items()
        }
        f0 = predictor.decode(result_np["salience"][0])
        coarse = quantize_f0(f0)
        report["regions"][region] = {
            "latency_ms": {
                "mean": float(np.mean(timings)),
                "median": float(np.median(timings)),
                "p95": float(np.percentile(timings, 95)),
            },
            "stages": {
                name: stats(reference_np[name], result_np[name])
                for name in ("unet", "head", "gru", "logits", "salience")
            },
            "f0": stats(reference_f0, f0),
            "coarse_exact_fraction": float(np.mean(reference_coarse == coarse)),
            "coarse_mismatch_frames": np.flatnonzero(reference_coarse != coarse).tolist(),
            "finite": bool(np.isfinite(f0).all()),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
