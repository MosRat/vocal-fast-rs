"""Sweep fake INT8 Synth convolution quantization against the Applio FP32 path.

This is deliberately a Python screening tool.  It does not claim that the
TensorRT graph is using INT8 tactics: each selected Conv1d/ConvTranspose1d is
replaced by a numerically equivalent fake-quant module.  The report is used to
choose safe candidates before adding real TensorRT W8A8/W8A16 scales.

W8A16 keeps activations in FP16 and uses per-output (per-input-filter for a
transposed convolution) INT8 weights.  W8A8 uses a calibrated per-tensor INT8
activation scale and per-output INT8 weights.  SmoothQuant optionally moves a
calibrated per-input-channel scale from the activation into the weight before
weight quantization.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PRECISION = ROOT / "scripts" / "synth-precision-applio.py"
spec = importlib.util.spec_from_file_location("synth_precision", PRECISION)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {PRECISION}")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


@dataclass
class Calibration:
    absmax: float = 0.0
    channel_absmax: np.ndarray | None = None
    calls: int = 0
    shape: tuple[int, ...] | None = None


def metric(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    a = reference.detach().float().cpu().numpy().astype(np.float64).reshape(-1)
    b = actual.detach().float().cpu().numpy().astype(np.float64).reshape(-1)
    d = b - a
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    return {
        "max_abs": float(np.max(np.abs(d))),
        "mean_abs": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "relative_l2": float(np.linalg.norm(d) / max(an, 1e-12)),
        "cosine": float(np.dot(a, b) / max(an * bn, 1e-12)),
    }


def iter_modules(net: nn.Module, scope: str) -> Iterable[tuple[str, nn.Module]]:
    for name, layer in net.named_modules():
        if scope and not name.startswith(scope):
            continue
        if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            yield name, layer


def _input_tensor(args: tuple[object, ...]) -> torch.Tensor | None:
    if not args or not isinstance(args[0], torch.Tensor):
        return None
    return args[0]


def set_input_noise(net: nn.Module, item: dict[str, object], device: torch.device) -> None:
    """Keep the deterministic NSF noise aligned when several VIN lengths are scanned."""
    noise = torch.from_numpy(np.asarray(item["noise"], dtype=np.float32)).to(device)
    net.dec.m_source.noise = noise


def collect_calibration(
    net: nn.Module,
    inputs: list[dict[str, object]],
    device: torch.device,
    scope: str,
) -> tuple[dict[str, Calibration], dict[str, dict[str, object]]]:
    stats: dict[str, Calibration] = {name: Calibration() for name, _ in iter_modules(net, scope)}
    shapes: dict[str, dict[str, object]] = {}
    handles = []

    def hook(name: str, layer: nn.Module):
        def capture(_module: nn.Module, args: tuple[object, ...], output: object):
            x = _input_tensor(args)
            if x is None:
                return
            value = x.detach().float()
            entry = stats[name]
            entry.calls += 1
            entry.shape = tuple(int(v) for v in value.shape)
            entry.absmax = max(entry.absmax, float(value.abs().max().item()))
            if isinstance(layer, nn.Conv1d):
                channel = value.abs().amax(dim=(0, 2)).cpu().numpy()
            elif isinstance(layer, nn.ConvTranspose1d):
                channel = value.abs().amax(dim=(0, 2)).cpu().numpy()
            else:
                channel = value.abs().amax(dim=tuple(i for i in range(value.ndim) if i != value.ndim - 1)).cpu().numpy()
            if entry.channel_absmax is None:
                entry.channel_absmax = channel
            else:
                entry.channel_absmax = np.maximum(entry.channel_absmax, channel)
            out_shape = tuple(int(v) for v in output.shape) if isinstance(output, torch.Tensor) else None
            shapes[name] = {
                "class": type(layer).__name__,
                "input_shape": list(value.shape),
                "output_shape": list(out_shape) if out_shape else None,
            }

        return capture

    for name, layer in iter_modules(net, scope):
        handles.append(layer.register_forward_hook(hook(name, layer)))
    try:
        with torch.inference_mode():
            for item in inputs:
                set_input_noise(net, item, device)
                mod.run_variant(net, item, device, "fp32")
    finally:
        for handle in handles:
            handle.remove()
    return stats, shapes


def _weight_max_by_input(layer: nn.Module) -> np.ndarray:
    weight = layer.weight.detach().float()
    if isinstance(layer, nn.Conv1d):
        return weight.abs().amax(dim=(0, 2)).cpu().numpy()
    if isinstance(layer, nn.ConvTranspose1d):
        return weight.abs().amax(dim=(1, 2)).cpu().numpy()
    return weight.abs().amax(dim=0).cpu().numpy()


def smooth_scale(layer: nn.Module, calibration: Calibration, alpha: float) -> torch.Tensor | None:
    if alpha <= 0.0 or calibration.channel_absmax is None:
        return None
    activation = np.maximum(calibration.channel_absmax, 1e-8)
    weight = np.maximum(_weight_max_by_input(layer), 1e-8)
    # Normalize the scale to avoid introducing a large global gain.  The
    # input/weight pair remains exactly equivalent before quantization.
    raw = np.power(activation, alpha) / np.power(weight, 1.0 - alpha)
    raw *= math.exp(-float(np.log(raw).mean()))
    return torch.from_numpy(raw.astype(np.float32))


def quantize_weight(layer: nn.Module, smooth: torch.Tensor | None) -> torch.Tensor:
    weight = layer.weight.detach().float()
    if smooth is not None:
        s = smooth.to(weight.device, dtype=weight.dtype)
        if isinstance(layer, nn.Conv1d):
            weight = weight * s.reshape(1, -1, 1)
        elif isinstance(layer, nn.ConvTranspose1d):
            weight = weight * s.reshape(-1, 1, 1)
        else:
            weight = weight * s.reshape(1, -1)
    if isinstance(layer, nn.ConvTranspose1d):
        reduce_dims = (1, 2)
    else:
        reduce_dims = tuple(range(1, weight.ndim))
    scale = weight.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8) / 127.0
    quantized = torch.round((weight / scale).clamp(-127.0, 127.0)) * scale
    return quantized


class FakeInt8Layer(nn.Module):
    """One Conv/Linear with a calibrated fake W8A16 or W8A8 path."""

    def __init__(
        self,
        original: nn.Module,
        mode: str,
        calibration: Calibration,
        smooth: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.kind = type(original).__name__
        self.weight = nn.Parameter(quantize_weight(original, smooth), requires_grad=False)
        self.bias = None if original.bias is None else nn.Parameter(original.bias.detach().float(), requires_grad=False)
        self.smooth = None if smooth is None else smooth.detach().float()
        self.activation_scale = max(calibration.absmax / 127.0, 1e-8)
        if isinstance(original, (nn.Conv1d, nn.ConvTranspose1d)):
            self.stride = original.stride
            self.padding = original.padding
            self.dilation = original.dilation
            self.groups = original.groups
            self.output_padding = original.output_padding if isinstance(original, nn.ConvTranspose1d) else None

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.smooth is not None:
            s = self.smooth.to(x.device, dtype=x.dtype)
            if x.ndim == 3:
                x = x / s.reshape(1, -1, 1)
            else:
                x = x / s.reshape(*([1] * (x.ndim - 1)), -1)
        if self.mode == "w8a16":
            return x.to(torch.float16)
        # W8A8: calibrated per-tensor activation quantization.  Keeping the
        # dequantized value in FP32 makes this a conservative accuracy model
        # for an eventual INT32 accumulate + scale implementation.
        scale = torch.as_tensor(self.activation_scale, device=x.device, dtype=x.dtype)
        return torch.round((x / scale).clamp(-127.0, 127.0)) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare(x)
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        if isinstance(self.kind, str) and self.kind == "Conv1d":
            y = F.conv1d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.kind == "ConvTranspose1d":
            y = F.conv_transpose1d(
                x, weight, bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation
            )
        elif self.kind == "Linear":
            y = F.linear(x, weight, bias)
        else:
            raise TypeError(self.kind)
        return y.float()


def replace_module(net: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = net.get_submodule(parent_name) if parent_name else net
    setattr(parent, leaf, replacement)


def profile_cost(layer: nn.Module, shape: dict[str, object]) -> int:
    out = shape.get("output_shape") or []
    if isinstance(layer, nn.Conv1d):
        kernel = int(layer.kernel_size[0])
        return int(np.prod(out, dtype=np.int64) * (layer.in_channels // layer.groups) * kernel)
    if isinstance(layer, nn.ConvTranspose1d):
        kernel = int(layer.kernel_size[0])
        return int(np.prod(out, dtype=np.int64) * (layer.in_channels // layer.groups) * kernel)
    if isinstance(layer, nn.Linear):
        return int(np.prod(out, dtype=np.int64) * layer.in_features)
    return 0


def timed_run(net: nn.Module, item: dict[str, object], device: torch.device, iterations: int) -> tuple[dict[str, torch.Tensor], float]:
    with torch.inference_mode():
        for _ in range(1):
            set_input_noise(net, item, device)
            mod.run_variant(net, item, device, "fp32")
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = None
            for _ in range(iterations):
                set_input_noise(net, item, device)
                result = mod.run_variant(net, item, device, "fp32")
            end.record()
            end.synchronize()
            return result, float(start.elapsed_time(end) / iterations)
        t0 = time.perf_counter()
        result = None
        for _ in range(iterations):
            set_input_noise(net, item, device)
            result = mod.run_variant(net, item, device, "fp32")
        return result, (time.perf_counter() - t0) * 1000.0 / iterations


def load_inputs(paths: list[Path]) -> list[dict[str, object]]:
    if not paths:
        paths = [ROOT / "target/compare/sine-11200.vin"]
    return [mod.read_vin(path) for path in paths]


def parse_alphas(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vfrm", type=Path, default=ROOT / "target/tsukuyomi-canonical.vfr")
    parser.add_argument("--vin", type=Path, action="append", default=[])
    parser.add_argument("--calibration-vin", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "target/compare/synth-int8-sweep.json")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--scope", default="dec.", help="module prefix to scan (default: decoder)")
    parser.add_argument("--max-candidates", type=int, default=36)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--smooth-alphas", default="0.3,0.5,0.7")
    parser.add_argument("--include-smooth", action="store_true")
    parser.add_argument("--cumulative-counts", default="1,2,4,8", help="hot-set sizes to evaluate after the single-layer sweep")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    inputs = load_inputs(args.vin)
    calibration_inputs = load_inputs(args.calibration_vin or args.vin)
    config, tensors = mod.read_vfrm(args.vfrm)
    longest_noise = max(inputs, key=lambda item: int(np.asarray(item["noise"]).size))["noise"]
    base_cpu, _ = mod.build_reference(config, tensors, longest_noise)
    base_cpu.eval()
    base = base_cpu.to(device)
    with torch.inference_mode():
        references = []
        for item in inputs:
            set_input_noise(base, item, device)
            references.append(mod.run_variant(base, item, device, "fp32"))
    stats, shapes = collect_calibration(base, calibration_inputs, device, args.scope)
    modules = dict(iter_modules(base, args.scope))
    candidates = []
    for name, layer in modules.items():
        if name not in shapes or stats[name].calls == 0:
            continue
        candidates.append((profile_cost(layer, shapes[name]), name, layer))
    candidates.sort(reverse=True)
    candidates = candidates[: max(1, args.max_candidates)]
    rank = {name: index + 1 for index, (_cost, name, _layer) in enumerate(candidates)}

    report: dict[str, object] = {
        "reference": "Applio Synthesizer PyTorch FP32",
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "vfrm": str(args.vfrm),
        "inputs": [str(path) for path in (args.vin or [ROOT / "target/compare/sine-11200.vin"])],
        "calibration_inputs": [str(path) for path in (args.calibration_vin or args.vin)],
        "quantization": {
            "weight": "per-output-channel int8 fake quant",
            "w8a16": "FP16 activation, FP16 dequantized weights",
            "w8a8": "calibrated per-tensor INT8 activation, FP32 dequantized reference op",
            "smoothquant": "per-input-channel activation/weight rescaling before W8A8",
        },
        "candidates": [],
        "inputs_results": {},
    }
    for _cost, name, layer in candidates:
        c = stats[name]
        candidate = {
            "name": name,
            "kind": type(layer).__name__,
            "rank_by_macs": rank[name],
            "estimated_macs": _cost,
            "input_shape": shapes[name].get("input_shape"),
            "output_shape": shapes[name].get("output_shape"),
            "calibration_absmax": c.absmax,
            "activation_scale": max(c.absmax / 127.0, 1e-8),
            "calibration_calls": c.calls,
            "results": {},
        }
        if c.channel_absmax is not None:
            channel = np.asarray(c.channel_absmax, dtype=np.float64)
            candidate["activation_channel_distribution"] = {
                "channels": int(channel.size),
                "max_abs": float(channel.max(initial=0.0)),
                "p99_abs": float(np.percentile(channel, 99.0)),
                "median_abs": float(np.median(channel)) if channel.size else 0.0,
                "outlier_ratio_max_over_median": float(channel.max() / max(np.median(channel), 1e-8)) if channel.size else 0.0,
            }
        for mode in ("w8a16", "w8a8"):
            variants = [("none", None)]
            if args.include_smooth and mode == "w8a8":
                variants += [(f"alpha_{alpha:g}", alpha) for alpha in parse_alphas(args.smooth_alphas)]
            for smooth_name, alpha in variants:
                smooth = smooth_scale(layer, c, alpha) if alpha is not None else None
                variant_net = copy.deepcopy(base_cpu).to(device).eval()
                target = variant_net.get_submodule(name)
                replacement = FakeInt8Layer(target, mode, c, smooth).to(device)
                replace_module(variant_net, name, replacement)
                per_input = []
                for index, item in enumerate(inputs):
                    actual, latency = timed_run(variant_net, item, device, args.iterations)
                    per_input.append({
                        "input": str((args.vin or [ROOT / "target/compare/sine-11200.vin"])[index]),
                        "latency_ms": latency,
                        "audio": metric(references[index]["audio"], actual["audio"]),
                        "z": metric(references[index]["z"], actual["z"]),
                        "finite": bool(torch.isfinite(actual["audio"]).all().item()),
                    })
                key = f"{mode}:{smooth_name}"
                candidate["results"][key] = per_input
                del variant_net
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        smooth_results = {
            key: value
            for key, value in candidate["results"].items()
            if key.startswith("w8a8:alpha_")
        }
        if smooth_results:
            def smooth_score(value: list[dict[str, object]]) -> float:
                return max((entry["audio"]["relative_l2"] for entry in value), default=1.0)

            best_key = min(smooth_results, key=lambda key: smooth_score(smooth_results[key]))
            candidate["smoothquant_best"] = {
                "variant": best_key,
                "worst_relative_l2": smooth_score(smooth_results[best_key]),
                "worst_cosine": min((entry["audio"]["cosine"] for entry in smooth_results[best_key]), default=0.0),
            }
        report["candidates"].append(candidate)

    # Build cumulative hot sets from individually passing candidates.  A
    # single layer can pass while the accumulated error of a whole stage does
    # not, so these measurements are the gate used for deciding whether a
    # TensorRT path is worth implementing.
    single_pass: dict[str, list[str]] = {"w8a16": [], "w8a8": []}
    for candidate in report["candidates"]:
        for mode in single_pass:
            items = candidate["results"].get(f"{mode}:none", [])
            worst_relative_l2 = max((item.get("audio", {}).get("relative_l2", 1.0) for item in items), default=1.0)
            worst_cosine = min((item.get("audio", {}).get("cosine", 0.0) for item in items), default=0.0)
            candidate.setdefault("worst", {})[mode] = {"relative_l2": worst_relative_l2, "cosine": worst_cosine}
            if worst_cosine >= 0.999 and worst_relative_l2 <= 0.01:
                single_pass[mode].append(candidate["name"])
    cumulative: dict[str, list[dict[str, object]]] = {"w8a16": [], "w8a8": []}
    counts = sorted({max(1, int(part)) for part in args.cumulative_counts.split(",") if part.strip()})
    for mode in cumulative:
        for count in counts:
            pool = [candidate for candidate in candidates if candidate[1] in single_pass[mode]]
            selected = pool[: min(count, len(pool))]
            variant_net = copy.deepcopy(base_cpu).to(device).eval()
            for _cost, name, layer in selected:
                target = variant_net.get_submodule(name)
                replacement = FakeInt8Layer(target, mode, stats[name], None).to(device)
                replace_module(variant_net, name, replacement)
            per_input = []
            for index, item in enumerate(inputs):
                actual, latency = timed_run(variant_net, item, device, args.iterations)
                per_input.append({
                    "input": str((args.vin or [ROOT / "target/compare/sine-11200.vin"])[index]),
                    "latency_ms": latency,
                    "audio": metric(references[index]["audio"], actual["audio"]),
                    "z": metric(references[index]["z"], actual["z"]),
                    "finite": bool(torch.isfinite(actual["audio"]).all().item()),
                })
            cumulative[mode].append({"count": len(selected), "layers": [name for _cost, name, _layer in selected], "inputs": per_input})
            del variant_net
            if device.type == "cuda":
                torch.cuda.empty_cache()
    report["cumulative_hotsets"] = cumulative

    # Build per-layer recommendations using the worst input length, rather
    # than silently gating only the first (usually 11200) fixture.
    gates = single_pass
    report["recommendation"] = {
        "individual_gate": {"cosine": 0.999, "relative_l2": 0.01},
        "w8a16_individual_pass": gates["w8a16"],
        "w8a8_individual_pass": gates["w8a8"],
        "cumulative": {
            mode: [
                {
                    "count": item["count"],
                    "worst_relative_l2": max((v["audio"]["relative_l2"] for v in item["inputs"]), default=1.0),
                    "worst_cosine": min((v["audio"]["cosine"] for v in item["inputs"]), default=0.0),
                }
                for item in cumulative[mode]
            ]
            for mode in cumulative
        },
        "note": "Validate cumulative hot sets and real voices before changing TensorRT graph defaults.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "device": str(device), "candidates": len(candidates), "recommendation": report["recommendation"]}, indent=2))


if __name__ == "__main__":
    main()
