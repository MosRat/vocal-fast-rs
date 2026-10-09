"""Profile and compare calibrated CPU recipes using rolling F0 and real engine IO.

Calibration uses the existing disjoint calibration windows. The default test WAV
contains only the held-out source tail starting at sample 208000. Profiles and
diagnostic traces execute separately from alternating paired pipeline timing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.io import wavfile

from benchmark_cpusupport import load_module

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def plans():
    hubert = read(ROOT / "target/hubert-cpu/recipes.json")["mixed"]
    rmvpe = read(ROOT / "target/cpu-w8a8/broad/recipes.json")["rmvpe_intermediate"]
    def combine(*items):
        return {"rules": [r for p in items for r in p["rules"]], "calibration": {}}
    attention = {"rules": [r for r in hubert["rules"] if ".feed_forward." not in r["prefix"]]}
    return {"fp32_gelu": combine(), "hubert_mixed": combine(hubert),
            "hubert_mixed_gelu": combine(hubert), "balanced": combine(hubert, rmvpe),
            "balanced_gelu": combine(hubert, rmvpe), "fp32_channel_norm": combine(),
            "hubert_mixed_channel_norm": combine(hubert), "balanced_channel_norm": combine(hubert, rmvpe),
            "fp32_io": combine(), "hubert_mixed_io": combine(hubert),
            "hubert_attention_io": combine(attention)}


def profile_summary(raw):
    # Graph-local IDs restart for each stage, so operator/name form the grouping.
    by_node = {}
    for row in raw["rows"]:
        for node in row["timings"]:
            key = (node.get("stage", ""), node["operator"], node["name"], node["id"], node.get("implementation", ""))
            by_node.setdefault(key, []).append(node["ms"])
    if not by_node:
        raise ValueError("input too short for representative profile windows")
    nodes = [{"stage": stage, "operator": op, "name": name, "id": node_id, "implementation": impl, "mean_ms": float(np.mean(values))}
             for (stage, op, name, node_id, impl), values in by_node.items()]
    nodes.sort(key=lambda n: n["mean_ms"], reverse=True)
    operators = {}
    for node in nodes:
        operators[node["operator"]] = operators.get(node["operator"], 0) + node["mean_ms"]
    return {"nodes": nodes, "operator_mean_ms": operators,
            "note": raw["note"], "sampled_chunks": [r["chunk"] for r in raw["rows"]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", type=Path, required=True)
    ap.add_argument("--binary-dir", type=Path, required=True)
    ap.add_argument("--recipes", nargs="+", choices=list(plans()), default=["hubert_mixed", "hubert_mixed_gelu", "balanced"])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--trace", action="store_true", help="collect exact reference inputs before timing")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--baseline-channel-norm", action="store_true", help="isolate tensor IO changes against already fused FP32")
    args = ap.parse_args()
    args.directory = args.directory.resolve()
    args.directory.mkdir(parents=True, exist_ok=True)
    recipes_module = load_module("recipe_tools", ROOT / "scripts/benchmark-cpu-recipes.py")
    streaming = load_module("streaming_tools", ROOT / "scripts/benchmark-cpu-streaming.py")
    affinity = load_module("affinity_tools", ROOT / "scripts/benchmark-cpu-operators.py").performance_cpu_affinity()
    # Timed trials inherit this affinity. No other inference process runs here.
    config = read(ROOT / "target/cpu-f0-rolling/benchmark.json")["config"]
    config["threads"] = args.threads
    config["fuse_channel_norm"] = args.baseline_channel_norm
    config["fast_tensor_io"] = False
    if not 1 <= args.threads <= 256:
        ap.error("threads must be 1..256")
    sources = read(ROOT / "target/cpu-f0-rolling/sources.json")
    fixture_path = ROOT / "target/cpu-w8a8/broad/inputs.json"
    fixtures = read(fixture_path)
    source_wave = ROOT / "target/compare/cpu-final2-trace/0000/waveform.bin"
    # Calibration intervals end at 156960; no sample in this WAV overlaps them.
    input_path = args.directory / "heldout.wav"
    if not args.resume:
        waveform = np.fromfile(source_wave, dtype="<f4")
        pcm = np.clip(waveform[208000:], -1, 1) * 32767
        wavfile.write(input_path, 16000, np.rint(pcm).astype(np.int16))
    binary = args.binary_dir.resolve() / "cpu_streaming_benchmark.exe"
    calibrator = args.binary_dir.resolve() / "quantization_benchmark.exe"
    digest = recipes_module.digest
    contract = {"binary": digest(binary), "calibrator": digest(calibrator),
                "sources": {k: {"path": v, "sha256": digest(v)} for k, v in sources.items()},
                "input": digest(input_path), "source_wave": digest(source_wave),
                "recipes": plans(), "fixture_manifest": digest(fixture_path),
                "test_interval_samples": [208000, source_wave.stat().st_size // 4],
                "calibration_wave_sha256": {w["directory"]: digest(Path(w["directory"]) / "wave.f32")
                                             for w in fixtures["windows"] if w["split"] == "calibration" and w["samples"] == 44960},
                "calibration_noise_sha256": {str(Path(w["directory"]) / name): digest(Path(w["directory"]) / name)
                                             for w in fixtures["windows"] if w["split"] == "calibration" and w["samples"] == 44960
                                             for name in ["rnd.f32", "noise.f32"]},
                "config": config, "affinity": affinity}
    report_path = args.directory / "report.json"
    report = {"contract": contract, "results": []}
    if args.resume:
        report = read(report_path)
        if report["contract"] != contract:
            raise ValueError("resume contract changed: use a new directory")
        for result in report["results"]:
            if result["recipe_sha256"] and digest(args.directory / result["name"] / "recipe.json") != result["recipe_sha256"]:
                raise ValueError("calibrated recipe changed during resume")
    write(report_path, report)
    source_path = args.directory / "sources.json"
    write(source_path, sources)

    def run(directory, settings, recipe=None, profile=False):
        directory.mkdir(exist_ok=True)
        options = {"baseline": config, "candidate": settings, "trace": args.trace and not profile,
                   "profile": profile, "recipe_path": str(recipe) if recipe else None}
        option_path = directory / "experiment.json"
        write(option_path, options)
        subprocess.run([str(binary), str(source_path), str(input_path), str(directory),
                        str(args.threads), str(option_path)], check=True)
        if profile:
            result = profile_summary(read(directory / "profile.json"))
            write(directory / "profile-summary.json", result)
            print(json.dumps({"profile": str(directory), "top": result["nodes"][:20]}), flush=True)
            return result
        raw = read(directory / "benchmark.json")
        result = streaming.summarize(raw)
        result["baseline"] = result.pop("full_window")
        result["candidate"] = result.pop("rolling")
        result["stage_mean_ms"] = {"baseline": result["stage_mean_ms"]["full_window"],
                                   "candidate": result["stage_mean_ms"]["rolling"]}
        result["quantized"] = raw["quantized"]
        result["config"] = raw["config"]
        write(directory / "summary.json", result)
        return result

    if args.profile:
        report["fp32_profile"] = run(args.directory / "fp32-profile", config, profile=True)
        write(report_path, report)
    for name in args.recipes:
        if any(r["name"] == name for r in report["results"]):
            continue
        directory = args.directory / name
        directory.mkdir(exist_ok=True)
        settings = dict(config, fuse_linear_gelu=name.endswith("gelu"),
                        fuse_channel_norm=name.endswith(("channel_norm", "io")), fast_tensor_io=name.endswith("io"))
        recipe = plans()[name]
        recipe_path = None
        if recipe["rules"]:
            recipe_path = directory / "recipe.json"
            calibration_path = directory / "calibration.json"
            candidate_path = directory / "config.json"
            write(recipe_path, recipe)
            write(candidate_path, settings)
            subprocess.run([str(calibrator), sources["synth"], sources["hubert"], sources["rmvpe"],
                            str(fixture_path), str(recipe_path), str(calibration_path), str(directory),
                            "44960", "0", str(candidate_path), str(candidate_path)], check=True)
            recipe["calibration"] = read(calibration_path)
            write(recipe_path, recipe)
        result = run(directory, settings, recipe_path)
        if any("vnni" not in op["implementation"] and "amx" not in op["implementation"] for op in result["quantized"]):
            raise ValueError("native integer VNNI/AMX kernels required")
        result["recipe_sha256"] = digest(recipe_path) if recipe_path else None
        report["results"].append(dict(name=name, **result))
        write(report_path, report)
        print(json.dumps({"name": name, "baseline": result["baseline"], "candidate": result["candidate"],
                          "speedup": result["speedup"], "ci": result["speedup_ci95"], "engine": result["engine"]}), flush=True)
        if args.profile:
            run(directory / "profile", settings, recipe_path, profile=True)


if __name__ == "__main__":
    main()
