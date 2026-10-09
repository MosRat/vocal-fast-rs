"""Benchmark fixed VFRM CPU graphs in independent processes, with warmup.

Build `rtk cargo build -p vocal-cpu --example benchmark` first. The waveform
is little-endian mono 16 kHz float32; an offline stage trace can supply it.
Each process owns a fresh native thread pool. Timings exclude construction,
and run without per-node instrumentation or intermediate tensor export.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/debug/examples/benchmark.exe"))
    parser.add_argument("--synth", type=Path, required=True)
    parser.add_argument("--hubert", type=Path, required=True)
    parser.add_argument("--rmvpe", type=Path, required=True)
    parser.add_argument("--waveform", type=Path, required=True)
    parser.add_argument("--samples", type=int, nargs="+", default=[11200, 44960])
    parser.add_argument("--threads", type=int, nargs="+", default=[6])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or any(t < 1 for t in args.threads):
        parser.error("iterations and threads must be positive")
    results = []
    for threads in args.threads:
        for samples in args.samples:
            for channels_last, fuse in [(False, True), (True, True), (True, False)]:
                command = [str(args.binary), str(args.synth), str(args.hubert), str(args.rmvpe),
                           str(args.waveform), str(samples), str(args.iterations), str(threads),
                           str(channels_last).lower(), str(fuse).lower()]
                process = subprocess.run(command, capture_output=True, text=True, check=True)
                result = json.loads(process.stdout)
                results.append(result)
                print(json.dumps({"samples": samples, "threads": threads, "packed": channels_last,
                                  "fuse": fuse, "mean_ms": result["mean_ms"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
