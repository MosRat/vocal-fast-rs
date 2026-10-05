"""Audit real RVC model diversity and summarize the fixed-profile benchmark.

This is intentionally independent from TensorRT plan generation.  It checks
that each VFRM contains different Synth weights, that the native benchmark
actually produces model-dependent samples, and combines the per-model
PyTorch/Applio parity reports with the measured mixed/coherent timings.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_precision_module():
    path = ROOT / "scripts" / "synth-precision-applio.py"
    spec = importlib.util.spec_from_file_location("synth_precision_applio", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def array_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    a = left.astype(np.float64, copy=False).reshape(-1)
    b = right.astype(np.float64, copy=False).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    denom = max(float(np.linalg.norm(a)), 1e-12)
    cosine_denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-24)
    return {
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        "relative_l2": float(np.linalg.norm(diff) / denom),
        "cosine": float(np.dot(a, b) / cosine_denom),
    }


def scalar_stats(value: np.ndarray) -> dict[str, object]:
    data = value.astype(np.float64, copy=False)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "rms": float(np.sqrt(np.mean(data * data))),
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vfrm-dir", type=Path, default=ROOT / "target/compare/malaysian-vfrm"
    )
    parser.add_argument(
        "--bench-dir", type=Path, default=ROOT / "target/compare/malaysian-bench"
    )
    parser.add_argument(
        "--parity-dir", type=Path, default=ROOT / "target/compare/malaysian-parity"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "target/compare/malaysian-weight-diff.json"
    )
    args = parser.parse_args()

    precision = load_precision_module()
    vfrms = sorted(args.vfrm_dir.glob("*.vfr"))
    if len(vfrms) < 2:
        raise SystemExit(f"need at least two VFRM files in {args.vfrm_dir}")

    models: dict[str, dict[str, np.ndarray]] = {}
    model_info: dict[str, object] = {}
    for path in vfrms:
        config, tensors = precision.read_vfrm(path)
        name = path.stem
        models[name] = tensors
        model_info[name] = {
            "path": str(path),
            "config": config,
            "tensor_count": len(tensors),
            "tensor_bytes": int(sum(value.nbytes for value in tensors.values())),
        }

    common_names = sorted(set.intersection(*(set(tensors) for tensors in models.values())))
    if not common_names:
        raise SystemExit("VFRM files have no common tensor names")
    representative_names = [
        name
        for name in (
            "emb_g.weight",
            "dec.conv_pre.weight",
            "dec.cond.weight",
            "dec.resblocks.0.convs1.0.weight",
            "dec.resblocks.0.convs2.0.weight",
        )
        if name in common_names
    ]

    weight_pairs: dict[str, object] = {}
    names = list(models)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            differing = 0
            max_abs = 0.0
            relative_l2 = 0.0
            for tensor_name in common_names:
                left = models[left_name][tensor_name]
                right = models[right_name][tensor_name]
                if left.shape != right.shape:
                    raise ValueError(f"{tensor_name}: shape differs between {left_name} and {right_name}")
                metrics = array_metrics(left, right)
                max_abs = max(max_abs, metrics["max_abs"])
                relative_l2 = max(relative_l2, metrics["relative_l2"])
                if not np.array_equal(left, right):
                    differing += 1
            key = f"{left_name}__vs__{right_name}"
            weight_pairs[key] = {
                "tensor_count": len(common_names),
                "differing_tensor_count": differing,
                "max_tensor_max_abs": max_abs,
                "max_tensor_relative_l2": relative_l2,
                "representative": {
                    tensor_name: array_metrics(
                        models[left_name][tensor_name], models[right_name][tensor_name]
                    )
                    for tensor_name in representative_names
                },
            }

    representative_stats = {
        model_name: {
            tensor_name: scalar_stats(tensors[tensor_name])
            for tensor_name in representative_names
        }
        for model_name, tensors in models.items()
    }

    native_outputs: dict[str, dict[str, np.ndarray]] = {}
    benchmark: dict[str, object] = {}
    parity: dict[str, object] = {}
    for model_name in models:
        native_outputs[model_name] = {}
        benchmark[model_name] = {}
        for variant in ("mixed", "coherent"):
            output_path = args.bench_dir / f"{model_name}-{variant}.f32"
            native_outputs[model_name][variant] = np.fromfile(output_path, dtype="<f4")
            report_path = args.bench_dir / f"{model_name}-{variant}.json"
            if report_path.exists():
                benchmark[model_name][variant] = load_json(report_path)
        parity_path = args.parity_dir / f"{model_name}.json"
        if parity_path.exists():
            parity[model_name] = load_json(parity_path)

    output_pairs: dict[str, object] = {}
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            key = f"{left_name}__vs__{right_name}"
            output_pairs[key] = {
                variant: array_metrics(
                    native_outputs[left_name][variant], native_outputs[right_name][variant]
                )
                for variant in ("mixed", "coherent")
            }

    result = {
        "models": model_info,
        "common_tensor_count": len(common_names),
        "representative_tensor_stats": representative_stats,
        "weight_pairs": weight_pairs,
        "native_output_pairs": output_pairs,
        "benchmark": benchmark,
        "parity": parity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"models: {', '.join(names)}")
    print(f"common tensors: {len(common_names)}")
    for key, value in weight_pairs.items():
        print(
            f"weights {key}: differing={value['differing_tensor_count']}/{value['tensor_count']} "
            f"max_abs={value['max_tensor_max_abs']:.6g} "
            f"max_rel_l2={value['max_tensor_relative_l2']:.6g}"
        )
    for key, variants in output_pairs.items():
        for variant, metrics in variants.items():
            print(
                f"output {key} {variant}: cosine={metrics['cosine']:.9f} "
                f"relative_l2={metrics['relative_l2']:.6g} "
                f"max_abs={metrics['max_abs']:.6g}"
            )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
