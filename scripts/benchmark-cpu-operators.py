"""Compare standalone oneDNN/XNNPACK with ATen on captured real VFRM tensors.

Providers run in separate, serial processes. Native timing excludes Python,
construction, tensor downloads and error calculation. Both prepared-layout and
canonical-layout boundary costs are measured. Weighted sums are operator-only
estimates, never complete pipeline latency.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("LOCALAPPDATA", ROOT / "target")) / "vocal-fast/cpu-operators"
FLOAT_P = ct.POINTER(ct.c_float)


class Spec(ct.Structure):
    _fields_ = [("op", ct.c_uint32), ("x_rank", ct.c_uint32), ("w_rank", ct.c_uint32),
                ("y_rank", ct.c_uint32), ("x", ct.c_int64 * 8), ("w", ct.c_int64 * 8),
                ("y", ct.c_int64 * 8), ("attrs", ct.c_int32 * 12), ("bias", ct.c_uint32)]


def performance_cpu_affinity():
    """Use Windows CPU-set efficiency classes; no assumptions about CPU numbering."""
    if os.name != "nt":
        return None
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ct.c_void_p
    kernel.GetSystemCpuSetInformation.argtypes = [ct.c_void_p, ct.c_uint32, ct.POINTER(ct.c_uint32), ct.c_void_p, ct.c_uint32]
    required = ct.c_uint32()
    kernel.GetSystemCpuSetInformation(None, 0, ct.byref(required), None, 0)
    buffer = ct.create_string_buffer(required.value)
    if not kernel.GetSystemCpuSetInformation(buffer, required, ct.byref(required), None, 0):
        raise ct.WinError(ct.get_last_error())
    data, rows, offset = buffer.raw, [], 0
    while offset < required.value:
        size = int.from_bytes(data[offset:offset+4], "little")
        if size < 20 or offset + size > required.value:
            raise RuntimeError("invalid Windows CPU-set topology")
        kind = int.from_bytes(data[offset+4:offset+8], "little")
        group = int.from_bytes(data[offset+12:offset+14], "little")
        if kind == 0 and group == 0:
            rows.append({"logical": data[offset+14], "core": data[offset+15], "efficiency": data[offset+18]})
        offset += size
    if not rows:
        raise RuntimeError("no CPUs in processor group zero")
    best = max(row["efficiency"] for row in rows)
    selected = [row["logical"] for row in rows if row["efficiency"] == best]
    kernel.SetProcessAffinityMask.argtypes = [ct.c_void_p, ct.c_size_t]
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), sum(1 << cpu for cpu in selected)):
        raise ct.WinError(ct.get_last_error())
    return {"selected": selected, "topology": rows}


def load_library(backend, build, torch):
    # Keep handles alive: Python closes AddDllDirectory registrations otherwise.
    directories = [build, build.parent / "onednn/bin", torch / "lib"]
    handles = [os.add_dll_directory(str(path.resolve())) for path in directories if path.is_dir()]
    library = ct.CDLL(str((build / f"bench_{backend}.dll").resolve()))
    library.bench_create.argtypes = [ct.POINTER(Spec), FLOAT_P, FLOAT_P, FLOAT_P, ct.c_int, ct.c_int]
    library.bench_create.restype = ct.c_void_p
    library.bench_error.restype = ct.c_char_p
    library.bench_measure.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.POINTER(ct.c_double), FLOAT_P]
    library.bench_measure.restype = ct.c_int
    library.bench_implementation.argtypes = [ct.c_void_p]
    library.bench_implementation.restype = ct.c_char_p
    library.bench_destroy.argtypes = [ct.c_void_p]
    return library, handles


def metric(reference, actual):
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    if not np.isfinite(b).all():
        raise RuntimeError("non-finite provider output")
    delta = b - a
    rms = float(np.sqrt(np.mean(a*a)))
    rmse = float(np.sqrt(np.mean(delta*delta)))
    return {"max_abs": float(np.max(np.abs(delta))), "rmse": rmse,
            "relative_rmse": rmse / max(rms, 1e-30),
            "cosine": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
            "reference_rms": rms}


def worker(args):
    affinity = performance_cpu_affinity() if not args.all_cores else None
    # Enforce the local AVX2 target even when the library contains other JIT code.
    os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX2"
    os.environ["DNNL_MAX_CPU_ISA"] = "AVX2"
    lib, dll_directories = load_library(args.worker, args.build, args.torch)
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    cases = list(manifest["cases"])
    random.Random(117).shuffle(cases)
    rows = []
    for case in cases:
        spec = Spec()
        spec.op = case["op"]
        for field, dims in [("x", case["input_shapes"][0]), ("w", case["input_shapes"][1]), ("y", case["output_shape"])]:
            setattr(spec, field + "_rank", len(dims))
            for i, dim in enumerate(dims):
                getattr(spec, field)[i] = dim
        for i, value in enumerate(case["attrs"]):
            spec.attrs[i] = value
        spec.bias = len(case["input_shapes"]) == 3
        inputs = [np.fromfile(args.fixtures / f"{case['id']}_input_{i}.f32", dtype="<f4")
                  for i in range(len(case["input_shapes"]))]
        for array, shape in zip(inputs, case["input_shapes"]):
            if array.size != np.prod(shape):
                raise ValueError(f"fixture input size mismatch: {case['id']}")
        reference = np.fromfile(args.fixtures / f"{case['id']}_output.f32", dtype="<f4")
        if reference.size != np.prod(case["output_shape"]):
            raise ValueError(f"fixture output size mismatch: {case['id']}")
        result = np.empty_like(reference)
        pointers = [array.ctypes.data_as(FLOAT_P) for array in inputs]
        started = time.perf_counter()
        handle = lib.bench_create(ct.byref(spec), pointers[0], pointers[1], pointers[2] if spec.bias else None, args.threads[0], args.layout)
        create_ms = (time.perf_counter() - started) * 1000
        row = {**case, "create_ms": create_ms}
        if not handle:
            row["error"] = lib.bench_error().decode()
            rows.append(row)
            continue
        try:
            row["implementation"] = lib.bench_implementation(handle).decode()
            for boundary in [False, True]:
                pilot = (ct.c_double * 1)()
                if lib.bench_measure(handle, boundary, 5, 1, 1, pilot, result.ctypes.data_as(FLOAT_P)):
                    raise RuntimeError(lib.bench_error().decode())
                loops = max(1, min(64, int(args.batch_ms / max(pilot[0], 0.001))))
                times = (ct.c_double * args.repeats)()
                if lib.bench_measure(handle, boundary, 3, args.repeats, loops, times, result.ctypes.data_as(FLOAT_P)):
                    raise RuntimeError(lib.bench_error().decode())
                values = list(times)
                row["boundary" if boundary else "prepared"] = {
                    "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
                    "loops": loops, "batches_ms": values, "error": metric(reference, result),
                }
                if row["boundary" if boundary else "prepared"]["error"]["relative_rmse"] > 1e-4:
                    raise RuntimeError(f"numerical mismatch: {row}")
        finally:
            lib.bench_destroy(handle)
        rows.append(row)
    output = {"provider": args.worker, "layout": args.layout, "threads": args.threads[0],
              "samples": manifest["config"]["window_samples"], "affinity": affinity,
              "manifest_sha256": hashlib.sha256((args.fixtures / "manifest.json").read_bytes()).hexdigest(),
              "cases": rows, "python": platform.python_version(),
              "library_sha256": hashlib.sha256((args.build / f"bench_{args.worker}.dll").read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ["provider", "layout", "threads", "samples"]}
                     | {"cases": len(rows), "unsupported": sum("error" in row for row in rows)}), flush=True)


def summary(reports):
    rows = []
    for report in reports:
        for stage in ["hubert", "rmvpe", "synth"]:
            cases = [row for row in report["cases"] if row["stage"] == stage and "error" not in row]
            rows.append({key: report[key] for key in ["provider", "layout", "threads", "samples"]}
                        | {"stage": stage, "shapes": len(cases), "occurrences": sum(row["occurrences"] for row in cases),
                           "weighted_prepared_ms": sum(row["occurrences"] * row["prepared"]["median_ms"] for row in cases),
                           "weighted_boundary_ms": sum(row["occurrences"] * row["boundary"]["median_ms"] for row in cases),
                           "worst_max_abs": max((row["prepared"]["error"]["max_abs"] for row in cases), default=0),
                           "worst_relative_rmse": max((row["prepared"]["error"]["relative_rmse"] for row in cases), default=0)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True, help="root containing 11200/ and 44960/, or one manifest directory in worker mode")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", type=Path, default=CACHE / "benchmark-build")
    parser.add_argument("--torch", type=Path, default=ROOT / "target/cpu-sdk/torch")
    parser.add_argument("--samples", type=int, nargs="+", default=[11200, 44960])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 6])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch-ms", type=float, default=3)
    parser.add_argument("--all-cores", action="store_true", help="allow Windows scheduling on efficiency cores too")
    parser.add_argument("--worker", choices=["aten", "onednn", "xnnpack"])
    parser.add_argument("--layout", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--providers", nargs="+", choices=["aten", "onednn", "xnnpack"], default=["aten", "onednn", "xnnpack"])
    parser.add_argument("--pipeline-baseline", type=Path, help="also time the existing complete ATen CPU pipeline with the captured model/input")
    args = parser.parse_args()
    if args.repeats < 3 or args.batch_ms <= 0 or any(n < 1 for n in args.threads):
        parser.error("repeats >= 3, batch-ms > 0 and positive threads required")
    if args.worker:
        worker(args)
        return
    reports, pipeline_reports = [], []
    for samples in args.samples:
        for threads in args.threads:
            for provider, layout in [("aten", 1), ("onednn", 0), ("onednn", 1), ("onednn", 2), ("xnnpack", 1)]:
                if provider not in args.providers:
                    continue
                output = args.output.parent / f"operators-{samples}-{threads}-{provider}-{layout}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", provider, "--layout", str(layout),
                           "--fixtures", str(args.fixtures / str(samples)), "--output", str(output),
                           "--build", str(args.build), "--torch", str(args.torch), "--threads", str(threads),
                           "--repeats", str(args.repeats), "--batch-ms", str(args.batch_ms)]
                if args.all_cores:
                    command.append("--all-cores")
                subprocess.run(command, check=True)
                reports.append(json.loads(output.read_text()))
                args.output.write_text(json.dumps({"note": "weighted isolated operator times; not full pipeline latency",
                                                   "summary": summary(reports), "reports": reports}, indent=2) + "\n")
            if args.pipeline_baseline:
                affinity = performance_cpu_affinity() if not args.all_cores else None
                manifest = json.loads((args.fixtures / str(samples) / "manifest.json").read_text())
                sources = manifest["sources"]
                waveform = args.fixtures / str(samples) / "waveform.f32"
                # HuBERT's first Conv input is the original waveform.
                first = next(case for case in manifest["cases"] if case["stage"] == "hubert" and case["input_shapes"][0] == [1, 1, 1, samples])
                waveform.write_bytes((args.fixtures / str(samples) / f"{first['id']}_input_0.f32").read_bytes())
                command = [str(args.pipeline_baseline.resolve()), sources["synth"], sources["hubert"], sources["rmvpe"], str(waveform),
                           str(samples), "10", str(threads), "true", "true", str(manifest["config"]["skip_head_frames"])]
                process = subprocess.run(command, capture_output=True, text=True, check=True)
                result = json.loads(process.stdout)
                result["affinity"] = affinity
                pipeline_reports.append(result)
                (args.output.parent / f"pipeline-{samples}-{threads}.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps({"pipeline": "existing ATen", "samples": samples, "threads": threads, "mean_ms": result["mean_ms"]}), flush=True)
    if pipeline_reports:
        result = json.loads(args.output.read_text())
        result["complete_aten_pipeline"] = pipeline_reports
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
