"""Compare RMVPE debug tensors with the local Applio FP32 stages.

The native debug outputs retain the padded U-Net/GRU time axis.  This checker
normalizes the layouts and reports each boundary, making it possible to tell a
mel/layout error from a TensorRT tactic or F0 decode difference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref_repos" / "Applio"))
from rvc.lib.predictors.RMVPE import RMVPE0Predictor  # noqa: E402


def stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | list[int]]:
    ref = np.asarray(reference, dtype=np.float64)
    value = np.asarray(candidate, dtype=np.float64)
    if ref.shape != value.shape:
        raise ValueError(f"shape mismatch: reference={ref.shape}, native={value.shape}")
    delta = value - ref
    return {
        "shape": list(ref.shape),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "relative_l2": float(
            np.linalg.norm(delta) / max(float(np.linalg.norm(ref)), 1.0e-12)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--debug", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models/shared/rmvpe/rmvpe.pt"
    )
    args = parser.parse_args()

    audio = np.fromfile(args.audio, dtype="<f4").astype(np.float32, copy=False)
    debug = json.loads(args.debug.read_text(encoding="utf-8"))
    predictor = RMVPE0Predictor(str(args.model), device="cpu")
    waveform = torch.from_numpy(audio).unsqueeze(0)
    with torch.inference_mode():
        mel = predictor.mel_extractor(waveform, center=True)
        original_frames = int(mel.shape[-1])
        padded_frames = 32 * ((original_frames - 1) // 32 + 1)
        padded_mel = F.pad(mel, (0, padded_frames - original_frames), mode="reflect")
        x = padded_mel.transpose(-1, -2).unsqueeze(1)
        unet = predictor.model.unet(x)
        head = predictor.model.cnn(unet)
        sequence = head.transpose(1, 2).flatten(-2)
        gru = predictor.model.fc[0](sequence)
        logits = predictor.model.fc[1](gru)

    native_mel = np.asarray(debug["rmvpe_mel"], dtype=np.float32).reshape(
        1, 128, original_frames
    )
    native_bottleneck = np.asarray(debug["rmvpe_bottleneck"], dtype=np.float32).reshape(
        1, 512, padded_frames // 32, 4
    )
    native_unet = np.asarray(debug["rmvpe_unet"], dtype=np.float32).reshape(
        1, 16, padded_frames, 128
    )
    native_gru = np.asarray(debug["rmvpe_gru"], dtype=np.float32).reshape(
        1, padded_frames, 512
    )
    native_logits = np.asarray(debug["rmvpe_logits"], dtype=np.float32).reshape(
        1, padded_frames, 360
    )
    result = {
        "audio": str(args.audio),
        "debug": str(args.debug),
        "frames": {"original": original_frames, "padded": padded_frames},
        "stages": {
            "mel": stats(mel.numpy(), native_mel),
            "unet": stats(unet.numpy(), native_unet),
            "gru": stats(gru.numpy(), native_gru),
            "logits": stats(logits.numpy(), native_logits),
        },
        "native_bottleneck": {
            "shape": list(native_bottleneck.shape),
            "min": float(native_bottleneck.min()),
            "max": float(native_bottleneck.max()),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
