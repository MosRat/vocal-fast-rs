"""Measure the Applio/PyTorch FP32 Synth reference for one fixed VIN."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

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
    parser.add_argument("--vfrm", type=Path, required=True)
    parser.add_argument("--vin", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmups < 0:
        raise ValueError("iterations must be positive and warmups cannot be negative")

    precision = load_precision_module()
    config, tensors = precision.read_vfrm(args.vfrm)
    inp = precision.read_vin(args.vin)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ = precision.build_reference(config, tensors, inp["noise"])
    net.to(device).eval()
    fn = lambda: precision.run_variant(net, inp, device, "fp32")
    result, values = precision.timed(fn, device, args.warmups, args.iterations)
    summary = precision.summary(values)
    report = {
        "reference": "Applio/PyTorch FP32 Synth",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "vfrm": str(args.vfrm),
        "vin": str(args.vin),
        "frames": int(inp["frames"]),
        "samples": int(result["audio"].numel()),
        "warmups": args.warmups,
        "iterations": args.iterations,
        "latency": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
