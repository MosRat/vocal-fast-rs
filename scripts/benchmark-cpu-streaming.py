"""Measure CPU rolling F0 against full-window FP32, including real engine IO.

Build cpu_streaming_benchmark first. Large diagnostics should use --directory
on a drive with sufficient space. Python reference validation runs separately.
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


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def latency(rows):
    totals = np.asarray(rows, dtype=np.float64)[:, -1]
    return {"mean_ms": float(totals.mean()), "p95_ms": float(np.percentile(totals, 95)),
            "max_ms": float(totals.max()), "rtf": float(totals.mean() / 250),
            "over_budget_chunks": int(np.count_nonzero(totals > 250))}


def summarize(raw):
    full = np.asarray(raw["full_window_rows_ms"], dtype=np.float64)
    rolling = np.asarray(raw["rolling_rows_ms"], dtype=np.float64)
    if full.shape != rolling.shape or full.ndim != 2 or full.shape[1] != 5 or len(full) < 16:
        raise ValueError("at least 16 complete paired chunks required")
    # Adjacent streaming chunks are correlated. Resample complete one-second
    # blocks instead of treating every callback as an independent trial.
    complete = len(full) // 4
    blocks = np.random.default_rng(369).integers(0, complete, size=(2000, complete))
    ids = (blocks[:, :, None] * 4 + np.arange(4)[None, None, :]).reshape(2000, -1)
    ratios = full[ids, -1].mean(1) / rolling[ids, -1].mean(1)
    return {"full_window": latency(full), "rolling": latency(rolling),
            "speedup": float(full[:, -1].mean() / rolling[:, -1].mean()),
            "speedup_ci95": np.percentile(ratios, [2.5, 97.5]).tolist(),
            "bootstrap_complete_chunks": complete * 4,
            "stage_mean_ms": {"full_window": full.mean(0).tolist(), "rolling": rolling.mean(0).tolist()},
            "engine": latency(np.asarray(raw["engine_ms"], dtype=np.float64)[:, None]),
            "paired_chunks": len(full), "chunk_budget_ms": 250, "memory": raw["memory"],
            "audio": raw["audio"], "trace": raw["trace"], "timing": raw["timing"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, default=ROOT / "target/debug/examples/cpu_streaming_benchmark.exe")
    ap.add_argument("--directory", type=Path, default=ROOT / "target/cpu-f0")
    ap.add_argument("--input", type=Path, default=ROOT / "target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.threads <= 256:
        ap.error("threads must be 1..256")
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    sources = {"synth": str(ROOT / "target/default-canonical-with-index.vfr"),
               "hubert": str(ROOT / "models/shared/hubert_base/hubert_features.vfr"),
               "rmvpe": str(ROOT / "models/shared/rmvpe/rmvpe.vfr")}
    if not args.summarize_only:
        spec = importlib.util.spec_from_file_location("cpu_affinity", ROOT / "scripts/benchmark-cpu-operators.py")
        affinity_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(affinity_module)
        affinity = affinity_module.performance_cpu_affinity()
        (directory / "affinity.json").write_text(json.dumps(affinity, indent=2), encoding="utf-8")
        source_path = directory / "sources.json"
        source_path.write_text(json.dumps(sources, indent=2), encoding="utf-8")
        subprocess.run([str(args.binary.resolve()), str(source_path), str(args.input.resolve()),
                        str(directory), str(args.threads)], check=True)
    raw = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
    # Preserve the exact run configuration even when summarizing existing data.
    if raw["sources"] != sources or Path(raw["input"]).resolve() != args.input.resolve():
        raise ValueError("stored sources/input differ from this command")
    if raw["config"]["threads"] != args.threads:
        raise ValueError("stored thread policy differs from this command")
    report = summarize(raw)
    report["provenance_at_summary"] = {
        "binary": digest(args.binary), "input": digest(args.input),
        "models": {path: digest(path) for path in sources.values()},
        "affinity": json.loads((directory / "affinity.json").read_text(encoding="utf-8")),
        "config": raw["config"], "baseline_config": raw["baseline_config"],
    }
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"memory", "provenance_at_summary"}}, indent=2))


if __name__ == "__main__":
    main()
