"""Compare CPU recipes with the current dense/token-major FP32 graph.

Uses the held-out Applio fixtures from the previous quantization experiment.
Each job alternates two complete native pipelines in one process; no diagnostic
tensor collection or per-node clocks occur inside the timed path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def script_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def recipes(broad, hubert):
    def combine(*plans):
        rules = {r["prefix"]: r for plan in plans for r in plan["rules"]}
        return {"rules": list(rules.values()), "calibration": {},
                "fold_deconvolution_batchnorm": True}

    rmvpe = broad["rmvpe_intermediate"]
    projection = {"rules": [{"prefix": "hubert.model.feature_projection.projection",
                             "smooth_alpha": 0.5}]}
    return {
        "linear_gelu": combine(),
        "channel_norm": combine(),
        "tensor_io": combine(),
        "rmvpe_intermediate": combine(rmvpe),
        "memory_attention": combine(rmvpe, hubert["qk-alpha-1"],
                                    hubert["vout-alpha0.5"], projection),
        "balanced": combine(rmvpe, hubert["mixed"]),
        "balanced_stage2": combine(broad["stage2"], hubert["mixed"]),
    }


def interval(baseline, candidate, cycle):
    a, b = np.asarray(baseline)[:, -1], np.asarray(candidate)[:, -1]
    if a.shape != b.shape or len(a) % cycle:
        raise ValueError("paired complete fixture cycles required")
    blocks = np.random.default_rng(369).integers(0, len(a) // cycle,
                                               size=(2000, len(a) // cycle))
    ids = (blocks[:, :, None] * cycle + np.arange(cycle)[None, None, :]).reshape(2000, -1)
    return np.percentile(a[ids].mean(1) / b[ids].mean(1), [2.5, 97.5]).tolist()


def compare_outputs(raw, directory, metrics, sample_rate):
    comparisons = []
    for actual, baseline in zip(raw["outputs"], raw["baseline"]["outputs"], strict=True):
        if (actual["split"], actual["offset"]) != (baseline["split"], baseline["offset"]):
            raise ValueError("unpaired diagnostic fixtures")
        fixture = Path(actual["reference_directory"])
        stem = f"{actual['split']}-{actual['offset']}"
        floats = lambda p: np.fromfile(p, dtype="<f4")
        reference = floats(fixture / "reference.f32")
        audio = floats(actual["audio"])
        fp32 = floats(baseline["audio"])
        pitch = np.fromfile(directory / f"{stem}-pitch.i32", dtype="<i4")
        pitch_ref = np.fromfile(fixture / "pitch.i32", dtype="<i4")
        f0 = floats(directory / f"{stem}-pitchf.f32")
        f0_ref = floats(fixture / "pitchf.f32")
        comparisons.append({"split": actual["split"], "offset": actual["offset"],
                            "applio": metrics.metric(reference, audio),
                            "native_fp32": metrics.metric(fp32, audio),
                            "fp32_applio": metrics.metric(reference, fp32),
                            "coarse_mismatch": int(np.count_nonzero(pitch != pitch_ref)),
                            "voicing_mismatch": int(np.count_nonzero((f0 > 0) != (f0_ref > 0))),
                            "f0": metrics.metric(f0_ref, f0)})
        wavfile.write(str(directory / f"{stem}.wav"), sample_rate, audio)
    return metrics.summarize(comparisons, sample_rate)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, default=ROOT / "target/debug/examples/quantization_benchmark.exe")
    ap.add_argument("--directory", type=Path, default=ROOT / "target/cpu-recipes")
    ap.add_argument("--iterations", type=int, default=32)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--fuse-linear-gelu", action="store_true", help="enable GELU fusion in paired FP32 and quantized graphs")
    ap.add_argument("--recipes", nargs="+", default=["linear_gelu", "rmvpe_intermediate", "memory_attention", "balanced"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.iterations < 16 or args.iterations % 4 or not 1 <= args.threads <= 256:
        ap.error("at least 16 paired iterations, divisible by 4; threads must be 1..256")
    args.directory = args.directory.resolve()
    args.directory.mkdir(parents=True, exist_ok=True)
    plans = recipes(read_json(ROOT / "target/cpu-w8a8/broad/recipes.json"),
                    read_json(ROOT / "target/hubert-cpu/recipes.json"))
    if set(args.recipes) - plans.keys():
        ap.error("unknown recipe")
    write_json(args.directory / "recipes.json", plans)
    inputs_path = ROOT / "target/cpu-w8a8/broad/inputs.json"
    inputs = read_json(inputs_path)
    sources = [ROOT / "target/default-canonical-with-index.vfr",
               ROOT / "models/shared/hubert_base/hubert_features.vfr",
               ROOT / "models/shared/rmvpe/rmvpe.vfr"]
    metrics = script_module("quant_metrics", ROOT / "scripts/benchmark-cpu-w8a8.py")
    ops = script_module("cpu_affinity", ROOT / "scripts/benchmark-cpu-operators.py")
    affinity = ops.performance_cpu_affinity()
    contract = {"binary": digest(args.binary), "sources": {str(p): digest(p) for p in sources},
                "inputs": digest(inputs_path), "iterations": args.iterations,
                "threads": args.threads, "fuse_linear_gelu": args.fuse_linear_gelu,
                "affinity": affinity, "recipes": plans,
                "fixtures": {str(Path(w["directory"]) / name): digest(Path(w["directory"]) / name)
                             for w in inputs["windows"] for name in ["wave.f32", "rnd.f32", "noise.f32", "reference.f32", "pitch.i32", "pitchf.f32"]}}
    report_path = args.directory / "report.json"
    report = {"contract": contract, "reference": "original Applio FP32 + HF HuBERT FP32, fixed noise, index=0",
              "timing": "paired complete native pipelines; 8 warmup pairs then alternating execution; includes Q/layout/Smooth, frontend and output copy; excludes graph build, fixture IO, SOLA and audio devices",
              "results": []}
    if args.resume and report_path.is_file():
        saved = read_json(report_path)
        if saved["contract"] != contract:
            raise ValueError("resume requires unchanged binary, inputs, sources, recipes and scheduling")
        report = saved
    env = dict(os.environ, DNNL_VERBOSE="0", ONEDNN_VERBOSE="0")

    def config(samples):
        return {"window_samples": samples, "skip_head_frames": 250 if samples == 44960 else 8,
                "threads": args.threads, "spin_wait_us": 0, "pin_to_performance_cores": False,
                "fuse": True, "fuse_linear_gelu": args.fuse_linear_gelu,
                "fuse_channel_norm": False,
                "fast_tensor_io": False,
                "channels_last": True, "profile": False,
                "attention": {"mode": "dense", "token_major_output": True}}

    def run(samples, name, iterations, calibration, settings):
        directory = args.directory / f"{samples}-{name}"
        directory.mkdir(exist_ok=True)
        recipe_path = directory / "recipe.json"
        write_json(recipe_path, plans[name])
        candidate = directory / "config.json"
        control = directory / "baseline-config.json"
        write_json(candidate, settings)
        write_json(control, config(samples))
        command = [str(args.binary.resolve()), *map(str, sources), str(inputs_path),
                   str(recipe_path) if plans[name]["rules"] else "-", str(calibration),
                   str(directory), str(samples), str(iterations), str(candidate), str(control)]
        proc = subprocess.run(command, capture_output=True, text=True, env=env)
        if proc.returncode:
            raise RuntimeError(f"{name}/{samples}: {proc.stderr or proc.stdout}")
        result = json.loads(proc.stdout)
        write_json(directory / "raw.json", result)
        return result, directory

    calibration = args.directory / "calibration.json"
    # Calibrate every available recipe when any INT8 job is requested, so a
    # resumed run can add a candidate without borrowing held-out activations.
    union = {r["prefix"]: r for plan in plans.values() for r in plan["rules"]} if any(plans[name]["rules"] for name in args.recipes) else {}
    if union:
        if args.resume and "calibration_sha256" in report:
            if not calibration.is_file() or digest(calibration) != report["calibration_sha256"]:
                raise ValueError("calibration changed during resume")
        else:
            plans["calibration_union"] = {"rules": list(union.values()), "calibration": {}}
            merged = {}
            for samples in [11200, 44960]:
                part = args.directory / f"calibration-{samples}.json"
                info, _ = run(samples, "calibration_union", 0, part, config(samples))
                for name, maxima in read_json(part).items():
                    merged[name] = np.maximum(merged.get(name, np.zeros(len(maxima))), maxima).tolist()
                print(json.dumps({"calibration": info}), flush=True)
            plans.pop("calibration_union")
            write_json(calibration, merged)
            report["calibration_sha256"] = digest(calibration)
            write_json(report_path, report)

    for samples in [11200, 44960]:
        for name in args.recipes:
            if any(r["samples"] == samples and r["name"] == name for r in report["results"]):
                continue
            settings = config(samples)
            if name == "linear_gelu":
                settings["fuse_linear_gelu"] = True
                if args.fuse_linear_gelu:
                    continue
            if name == "channel_norm":
                settings["fuse_channel_norm"] = True
            if name == "tensor_io":
                settings.update(fuse_channel_norm=True, fast_tensor_io=True)
            raw, directory = run(samples, name, args.iterations, calibration, settings)
            base = raw["baseline"]
            expected = {rule["prefix"] for rule in plans[name]["rules"]}
            actual = [operator["name"] for operator in raw["quantized"]]
            if set(actual) != expected or len(actual) != len(expected):
                raise ValueError("unexpected quantized layer coverage")
            if any("vnni" not in n["implementation"] and "amx" not in n["implementation"] for n in raw["quantized"]):
                raise ValueError("native VNNI/AMX integer kernels required")
            audio = compare_outputs(raw, directory, metrics, inputs["sample_rate"])
            baseline_times = np.asarray(base["rows_ms"])[:, -1]
            candidate_times = np.asarray(raw["rows_ms"])[:, -1]
            budget = 512 if samples == 11200 else 250
            packed = lambda m: sum(stage["packed_bytes"] for stage in m.values())
            row = {"samples": samples, "name": name, "config": raw["config"],
                   "baseline_config": base["config"], "latency": {"fp32": base["mean_ms"], "candidate": raw["mean_ms"],
                    "speedup": baseline_times.mean() / candidate_times.mean(),
                    "speedup_ci95": interval(base["rows_ms"], raw["rows_ms"], len(audio["windows"])),
                    "fp32_p95_ms": float(np.percentile(baseline_times, 95)),
                    "candidate_p95_ms": float(np.percentile(candidate_times, 95)),
                    "candidate_max_ms": float(candidate_times.max()), "chunk_budget_ms": budget,
                    "inference_rtf": float(candidate_times.mean() / budget)},
                   "audio": audio, "quiet": [w for w in audio["windows"] if w["applio"]["reference_rms"] < 0.01],
                   "packed_weight_saving_mib": (packed(base["memory"]) - packed(raw["memory"])) / 2**20,
                   "baseline_memory": base["memory"], "candidate_memory": raw["memory"],
                   "quantized_layers": len(raw["quantized"]), "quantized": raw["quantized"]}
            report["results"].append(row)
            write_json(report_path, report)
            print(json.dumps({"samples": samples, "name": name, "latency": row["latency"],
                              "cosine_min": audio["cosine_min"], "rmse": audio["rmse_mean"],
                              "max_abs": audio["max_abs"], "packed_saving_mib": row["packed_weight_saving_mib"]}), flush=True)


if __name__ == "__main__":
    main()
