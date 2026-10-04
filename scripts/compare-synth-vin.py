"""Compare TensorRT Synth benchmark outputs with the Applio FP32 reference.

The native benchmark writes raw little-endian FP32 audio for one fixed VIN
shape. This tool reconstructs the same canonical VFRM model and VIN tensors,
runs the PyTorch reference once, and reports metrics for one or more native
outputs. It is intended for fixed-profile FP16 boundary experiments.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_precision_module():
    path = ROOT / "scripts" / "synth-precision-applio.py"
    spec = importlib.util.spec_from_file_location("synth_precision_applio", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vfrm", type=Path, default=ROOT / "target/tsukuyomi-canonical.vfr")
    parser.add_argument("--vin", type=Path, required=True)
    parser.add_argument("--native", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    precision = load_precision_module()
    config, tensors = precision.read_vfrm(args.vfrm)
    inp = precision.read_vin(args.vin)
    net, _ = precision.build_reference(config, tensors, inp["noise"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).eval()
    with torch.inference_mode():
        reference = precision.run_variant(net, inp, device, "fp32")["audio"]
    reference_np = reference.detach().float().cpu().numpy().reshape(-1)

    report: dict[str, object] = {
        "reference": "Applio/PyTorch FP32 Synth",
        "device": str(device),
        "vin": str(args.vin),
        "frames": int(inp["frames"]),
        "samples": int(reference_np.size),
        "native": {},
    }
    for path in args.native:
        value = np.fromfile(path, dtype="<f4")
        if value.size != reference_np.size:
            raise ValueError(f"{path}: {value.size} samples, expected {reference_np.size}")
        metrics = precision.metric(torch.from_numpy(reference_np), torch.from_numpy(value))
        report["native"][str(path)] = metrics

    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
