"""Serial pipeline benchmark with Windows private/working-set sampling.

Run once before a change and once after it. Process counters include the DLL,
primitive cache and mapped model; native graph accounting is reported separately.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class Counters(ct.Structure):
    _fields_ = [("cb", ct.c_ulong), ("PageFaultCount", ct.c_ulong)] + [
        (name, ct.c_size_t) for name in
        ["PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
         "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
         "PagefileUsage", "PeakPagefileUsage", "PrivateUsage"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, default=ROOT / "target/debug/examples/benchmark.exe")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threads", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--samples", type=int, nargs="+", default=[11200, 44960])
    ap.add_argument("--spin-us", type=int, nargs="+", default=[0])
    ap.add_argument("--pin-performance-cores", action="store_true")
    ap.add_argument("--iterations", type=int, default=20)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("operators", ROOT / "scripts/benchmark-cpu-operators.py")
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    affinity = ops.performance_cpu_affinity()
    psapi = ct.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ct.c_void_p, ct.POINTER(Counters), ct.c_ulong]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for spin in args.spin_us:
        for threads in args.threads:
            for samples in args.samples:
                audio = args.output.with_name(f"{args.output.stem}-{samples}-{threads}-{spin}.f32")
                command = [str(args.binary.resolve()), str(ROOT / "target/default-canonical-with-index.vfr"),
                           str(ROOT / "models/shared/hubert_base/hubert_features.vfr"),
                           str(ROOT / "models/shared/rmvpe/rmvpe.vfr"),
                           str(ROOT / "target/cpu-migration/speech.f32"), str(samples),
                           str(args.iterations), str(threads), "true", "true",
                           str(250 if samples == 44960 else 8), str(spin),
                           str(args.pin_performance_cores).lower()]
                proc = subprocess.Popen(command, env=dict(os.environ, VOCAL_CPU_BENCH_OUTPUT=str(audio)),
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                # Drain pipes while sampling: large iteration reports must not
                # block the child at stdout and prevent process completion.
                reader = ThreadPoolExecutor(max_workers=1)
                completed = reader.submit(proc.communicate)
                peak_private = peak_rss = last_private = last_rss = 0
                observations = []
                launched = time.perf_counter()
                while proc.poll() is None:
                    counter = Counters(cb=ct.sizeof(Counters))
                    if psapi.GetProcessMemoryInfo(int(proc._handle), ct.byref(counter), ct.sizeof(counter)):
                        last_private, last_rss = counter.PrivateUsage, counter.WorkingSetSize
                        peak_private = max(peak_private, last_private)
                        peak_rss = max(peak_rss, counter.PeakWorkingSetSize)
                        observations.append((time.perf_counter()-launched, last_private, last_rss))
                    time.sleep(0.02)
                stdout, stderr = completed.result()
                reader.shutdown()
                if proc.returncode:
                    raise RuntimeError(f"exit {proc.returncode}: {stderr or stdout}")
                row = json.loads(stdout)
                steady = [sample[1:] for sample in observations
                          if row["build_ms"]/1000+0.4 < sample[0] < observations[-1][0]-0.15]
                if not steady:
                    raise RuntimeError("not enough inference-time memory samples; increase iterations")
                private, rss = np.median(steady, axis=0).astype(np.int64).tolist()
                latency = np.asarray(row["rows_ms"])[:, -1]
                row["process_bytes"] = dict(peak_private=peak_private, peak_rss=peak_rss,
                                            steady_private=private, steady_rss=rss)
                row["total_percentiles_ms"] = dict(zip(["p50", "p95", "max"],
                    [*np.percentile(latency, [50, 95]).tolist(), float(latency.max())]))
                rows.append(row)
                print(json.dumps({"samples":samples, "threads":threads, "spin_us":spin,
                                  "mean_ms":row["mean_ms"], "memory_mib":private/2**20}), flush=True)
                args.output.write_text(json.dumps({"affinity":affinity, "results":rows}, indent=2)+"\n")


if __name__ == "__main__":
    main()
