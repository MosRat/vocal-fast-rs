"""Serial FP32 convolution experiments against the current CPU production policy.

Isolated NHWC operators, paired streaming pipelines and actual Engine replay
have separate measurements. Per-node clocks never enter timed production runs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def convolution(name):
    return {"mode": "matmul1d" if name.startswith("matmul1d") else
            "pointwise1d" if name.startswith("pointwise1d") else "direct",
            "fuse_gelu": name.endswith("gelu") and "linear" not in name,
            "fuse_residual": "residual" in name}


def variant_config(base, name):
    return dict(base, convolution=convolution(name),
                fuse_linear_gelu="linear_gelu" in name,
                linear_mode="inner_product" if "inner_product" in name else
                            "pointwise1d" if "pointwise_linear" in name else
                            "hybrid" if "hybrid_linear" in name else "matmul")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", type=Path, required=True)
    ap.add_argument("--binary-dir", type=Path, required=True)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--baseline-report", type=Path, required=True)
    ap.add_argument("--control", choices=["production", "residual", "residual_hybrid_linear"],
                    default="production", help="isolate a further change against an already optimized control")
    ap.add_argument("--fixtures", type=Path, default=ROOT / "target/compare/cpu-operators")
    ap.add_argument("--samples", nargs="+", type=int, default=[11200, 44960])
    ap.add_argument("--iterations", type=int, default=32)
    ap.add_argument("--variants", nargs="*", choices=["conv_gelu", "matmul1d", "matmul1d_gelu", "pointwise1d", "pointwise1d_residual", "residual", "inner_product", "residual_inner_product", "pointwise_linear", "residual_pointwise_linear", "hybrid_linear", "residual_hybrid_linear", "residual_hybrid_linear_gelu"],
                    default=["conv_gelu", "matmul1d"])
    ap.add_argument("--skip-operators", action="store_true")
    ap.add_argument("--linear-operators", action="store_true", help="also compare static FP32 projection implementations")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.iterations < 16:
        ap.error("at least 16 paired iterations required")
    args.directory = args.directory.resolve()
    args.directory.mkdir(parents=True, exist_ok=True)
    affinity = module("cpu_affinity", ROOT / "scripts/benchmark-cpu-operators.py").performance_cpu_affinity()
    streaming = module("cpu_streaming", ROOT / "scripts/benchmark-cpu-streaming.py")
    recipes = module("cpu_recipes", ROOT / "scripts/benchmark-cpu-streaming-recipes.py")
    sources = read(args.sources)
    base = read(args.baseline_report)["config"]
    # Explicit control: current FP32 norm + IO + rolling raw-Hz history. Old
    # recipe scripts intentionally disable some defaults to measure past gains.
    if not (base.get("fuse_channel_norm") and base.get("fast_tensor_io") and
            base["f0"]["mode"] == "rolling" and not base["profile"]):
        raise ValueError("control must use fused channel norm, fast IO and rolling F0")
    base["convolution"] = {"mode": "direct", "fuse_gelu": False, "fuse_residual": False}
    base["linear_mode"] = "matmul"
    if args.control != "production":
        base = variant_config(base, args.control)
    binary = args.binary_dir.resolve() / "cpu_streaming_benchmark.exe"
    operator_binary = args.binary_dir.resolve() / "convolution_benchmark.exe"
    linear_binary = args.binary_dir.resolve() / "linear_benchmark.exe"
    library = args.binary_dir.resolve() / "dnnl.dll"
    contract = {"baseline": base, "sources": {k: {"path": v, "sha256": digest(v)} for k, v in sources.items()},
                "input_sha256": digest(args.input), "binary_sha256": digest(binary),
                "operator_binary_sha256": digest(operator_binary), "affinity": affinity,
                "onednn_sha256": digest(library),
                "linear_binary_sha256": digest(linear_binary) if args.linear_operators else None,
                "fixture_manifests": {str(n): digest(args.fixtures / str(n) / "manifest.json") for n in args.samples},
                "iterations": args.iterations, "trace": args.trace, "profile": args.profile}
    path = args.directory / "report.json"
    report = {"contract": contract, "operators": {}, "linear_operators": {}, "results": []}
    if args.resume:
        report = read(path)
        if report["contract"] != contract:
            raise ValueError("resume contract changed; use a new directory")
    write(path, report)
    if not args.skip_operators:
        for samples in args.samples:
            if str(samples) in report["operators"]:
                continue
            output = args.directory / f"operators-{samples}.json"
            subprocess.run([str(operator_binary), str(args.fixtures / str(samples)), str(output), str(args.iterations)], check=True)
            raw = read(output)
            rows = []
            for row in raw["rows"]:
                a, b = np.array(row["baseline_ms"]), np.array(row["candidate_ms"])
                rows.append({"name": row["case"]["name"], "shape": row["case"]["input_shapes"],
                             "attrs": row["case"]["attrs"], "variant": row["variant"],
                             "baseline_ms": float(a.mean()), "candidate_ms": float(b.mean()),
                             "speedup": float(a.mean() / b.mean()), "error": row["fp32_error"],
                             "implementations": [t["implementation"] for t in row["diagnostic"]]})
            report["operators"][str(samples)] = rows
            write(path, report)
            print(json.dumps({"samples": samples, "top": sorted(rows, key=lambda x: x["baseline_ms"], reverse=True)[:12]}), flush=True)
    if args.linear_operators:
        for samples in args.samples:
            if str(samples) in report["linear_operators"]:
                continue
            output = args.directory / f"linear-{samples}.json"
            subprocess.run([str(linear_binary), str(args.fixtures / str(samples)), str(output), str(args.iterations)], check=True)
            raw = read(output)
            rows = []
            for row in raw["rows"]:
                a, b = np.array(row["baseline_ms"]), np.array(row["candidate_ms"])
                rows.append({"name": row["case"]["name"], "variant": row.get("variant", "inner_product"), "shape": row["case"]["input_shapes"],
                             "occurrences": row["case"]["occurrences"], "baseline_ms": float(a.mean()),
                             "candidate_ms": float(b.mean()), "speedup": float(a.mean() / b.mean()),
                             "error": row["fp32_error"], "implementations": [t["implementation"] for t in row["diagnostic"]]})
            report["linear_operators"][str(samples)] = rows
            write(path, report)
            print(json.dumps({"samples": samples, "linears": rows}), flush=True)
    for name in args.variants:
        if any(r["name"] == name for r in report["results"]):
            continue
        directory = args.directory / name
        directory.mkdir(exist_ok=True)
        candidate = variant_config(base, name)
        options = {"baseline": base, "candidate": candidate, "recipe_path": None,
                   "trace": args.trace, "profile": False}
        option_path = directory / "experiment.json"
        write(option_path, options)
        subprocess.run([str(binary), str(args.sources.resolve()), str(args.input.resolve()), str(directory),
                        str(base["threads"]), str(option_path)], check=True)
        raw = read(directory / "benchmark.json")
        result = streaming.summarize(raw)
        result["baseline"] = result.pop("full_window")
        result["candidate"] = result.pop("rolling")
        result["stage_mean_ms"] = {"baseline": result["stage_mean_ms"]["full_window"],
                                   "candidate": result["stage_mean_ms"]["rolling"]}
        result["config"] = raw["config"]
        write(directory / "summary.json", result)
        report["results"].append(dict(name=name, **result))
        write(path, report)
        print(json.dumps({"name": name, "baseline": result["baseline"], "candidate": result["candidate"],
                          "speedup": result["speedup"], "ci": result["speedup_ci95"], "engine": result["engine"]}), flush=True)
        if args.profile:
            profile_dir = directory / "profile"
            profile_dir.mkdir(exist_ok=True)
            options["profile"], options["trace"] = True, False
            write(option_path, options)
            subprocess.run([str(binary), str(args.sources.resolve()), str(args.input.resolve()), str(profile_dir),
                            str(base["threads"]), str(option_path)], check=True)
            write(profile_dir / "profile-summary.json", recipes.profile_summary(read(profile_dir / "profile.json")))
            # Preserve the timed experiment's contract in its original file.
            options["profile"], options["trace"] = False, args.trace
            write(option_path, options)


if __name__ == "__main__":
    main()
