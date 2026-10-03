"""Numerically compare the native RMVPE graph against the Applio predictor."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref_repos" / "Applio"))
from rvc.lib.predictors.RMVPE import RMVPE0Predictor  # noqa: E402


def main() -> None:
    debug = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "target/trt-pipeline-final/trt-debug.json")
    audio_path = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "target/compare/sine-6400.f32")
    native = np.asarray(json.loads(debug.read_text())["pitchf"], dtype=np.float32)
    audio = np.fromfile(audio_path, dtype="<f4")
    predictor = RMVPE0Predictor(str(ROOT / "models/shared/rmvpe/rmvpe.pt"), device="cpu")
    reference = np.asarray(predictor.infer_from_audio(audio), dtype=np.float32)[: native.size]
    delta = native.astype(np.float64) - reference.astype(np.float64)
    print(json.dumps({
        "frames": int(native.size),
        "native": native.tolist(),
        "reference": reference.tolist(),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-12)),
    }, indent=2))


if __name__ == "__main__":
    main()
