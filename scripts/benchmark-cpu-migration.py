"""Serial complete-pipeline ATen/oneDNN comparison on identical real VFRM inputs.

Build the new benchmark and retain the old executable plus its CPU Torch DLLs.
Affinity uses Windows CPU-set efficiency classes. Output/noise is deterministic.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, default=ROOT / "target/cpu-migration/baseline/benchmark.exe")
    ap.add_argument("--native", type=Path, default=ROOT / "target/debug/examples/benchmark.exe")
    ap.add_argument("--baseline-name", default="aten")
    ap.add_argument("--native-name", default="onednn")
    ap.add_argument("--synth", type=Path, default=ROOT / "target/default-canonical-with-index.vfr")
    ap.add_argument("--hubert", type=Path, default=ROOT / "models/shared/hubert_base/hubert_features.vfr")
    ap.add_argument("--rmvpe", type=Path, default=ROOT / "models/shared/rmvpe/rmvpe.vfr")
    ap.add_argument("--waveform", type=Path, default=ROOT / "target/compare/cpu-final2-trace/0000/waveform.bin")
    ap.add_argument("--threads", type=int, nargs="+", default=[6])
    ap.add_argument("--samples", type=int, nargs="+", default=[11200, 44960])
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--output", type=Path, default=ROOT / "target/cpu-migration/benchmark.json")
    args = ap.parse_args()
    if args.iterations < 1 or any(t < 1 for t in args.threads): ap.error("positive iterations/threads required")
    spec = importlib.util.spec_from_file_location("operators", ROOT / "scripts/benchmark-cpu-operators.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    affinity = module.performance_cpu_affinity()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for threads in args.threads:
        for samples in args.samples:
            pair = {}
            for provider, binary in [(args.baseline_name, args.baseline), (args.native_name, args.native)]:
                audio = args.output.parent / f"{provider}-{samples}-{threads}.f32"
                env = dict(os.environ, VOCAL_CPU_BENCH_OUTPUT=str(audio))
                env["PATH"] = str(ROOT / "target/cpu-sdk/torch/lib") + os.pathsep + env["PATH"]
                command = [str(binary), str(args.synth), str(args.hubert), str(args.rmvpe), str(args.waveform),
                           str(samples), str(args.iterations), str(threads), "true", "true"]
                proc = subprocess.run(command, env=env, text=True, capture_output=True)
                if proc.returncode: raise RuntimeError(f"{provider}: {proc.stderr or proc.stdout} (exit {proc.returncode})")
                pair[provider] = json.loads(proc.stdout)
                latency = np.asarray(pair[provider]["rows_ms"])[:, -1]
                pair[provider]["total_percentiles_ms"] = dict(zip(["p50", "p95", "max"], [*np.percentile(latency,[50,95]).tolist(),float(latency.max())]))
                print(json.dumps({"provider":provider,"samples":samples,"threads":threads,"mean_ms":pair[provider]["mean_ms"]}),flush=True)
            a = np.fromfile(args.output.parent / f"{args.baseline_name}-{samples}-{threads}.f32", dtype="<f4").astype(np.float64)
            b = np.fromfile(args.output.parent / f"{args.native_name}-{samples}-{threads}.f32", dtype="<f4").astype(np.float64)
            if a.shape != b.shape: raise ValueError("output shape mismatch")
            norm = np.linalg.norm(a)*np.linalg.norm(b)
            pair["parity"] = {"cosine":float(a@b/norm) if norm else None,"rmse":float(np.sqrt(np.mean((a-b)**2))),"max_abs":float(np.max(np.abs(a-b)))}
            pair["speedup"] = pair[args.baseline_name]["mean_ms"]["total"]/pair[args.native_name]["mean_ms"]["total"]
            rows.append(pair)
    args.output.write_text(json.dumps({"affinity":affinity,"note":"complete pipeline, index disabled, deterministic noise, serial processes, no tensor trace","results":rows},indent=2)+"\n",encoding="utf-8")

if __name__ == "__main__": main()
