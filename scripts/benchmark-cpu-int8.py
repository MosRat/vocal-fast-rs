"""Real oneDNN W8A8 Conv/Deconv/GEMM versus FP32 on captured RVC shapes.

Times include the scaled FP32 output epilogue. Boundary times also include
activation quantization and layout conversion, but not model loading/packing.
Quantization calibration here uses each operator case, not the held-out audio
calibration in experiment-cpu-w8a8.py; these are operator performance probes.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import importlib.util
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("operators", ROOT / "scripts/benchmark-cpu-operators.py")
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


def descriptor(case):
    s = ops.Spec()
    s.op = case["op"]
    for field, dims in [("x",case["input_shapes"][0]),("w",case["input_shapes"][1]),("y",case["output_shape"])]:
        setattr(s, field+"_rank", len(dims))
        for i, dim in enumerate(dims): getattr(s, field)[i] = dim
    for i, value in enumerate(case["attrs"]): s.attrs[i] = value
    s.bias = len(case["input_shapes"]) == 3
    return s


def quantized_samples(case, inputs, range_, even_grid=False):
    """Independent sampled integer dot products; catches scales, bias and axes."""
    x, w = [v.reshape(shape) for v,shape in zip(inputs[:2],case["input_shapes"][:2])]
    sx = max(float(np.max(np.abs(x))),1e-8)/range_
    weight_range = 126 if even_grid else range_
    step = 2 if even_grid else 1
    def qw(value,scale):
        return (np.rint(value/scale/step)*step).clip(-weight_range,weight_range).astype(np.int32)
    if case["op"] in [23,53]:
        w = w.T if case["attrs"][0] else w
        sw = np.maximum(np.max(np.abs(w),axis=0),1e-8)/weight_range
        weight_q = qw(w,sw)
    elif case["op"] == 24:
        sw = np.maximum(np.max(np.abs(w),axis=(1,2,3)),1e-8)/weight_range
        weight_q = qw(w,sw[:,None,None,None])
    else:
        groups = case["attrs"][8]
        grouped = w.reshape(groups,w.shape[0]//groups,*w.shape[1:])
        sw = np.maximum(np.max(np.abs(grouped),axis=(1,3,4)),1e-8)/weight_range
        weight_q = qw(grouped,sw[:,None,:,None,None]).reshape(w.shape)
        sw = sw.flatten()
    qx = np.rint(x/sx).clip(-range_,range_).astype(np.int32)
    shape = case["output_shape"]
    positions = np.unique(np.linspace(0,int(np.prod(shape))-1,32,dtype=np.int64))
    expected = []
    for position in positions:
        if case["op"] in [23,53]:
            matrix = qx.reshape(-1,qx.shape[-1]); row, channel = divmod(position,shape[-1])
            dot = np.dot(matrix[row].astype(np.int64),weight_q[:,channel].astype(np.int64))
        else:
            batch, channel, y0, y1 = np.unravel_index(position,shape)
            a = case["attrs"]; groups = a[8]; co = shape[1]//groups; ci = x.shape[1]//groups
            group = channel//co; dot = 0
            for k0 in range(w.shape[2]):
                for k1 in range(w.shape[3]):
                    if case["op"] == 24:
                        p0,p1 = y0*a[0]-a[4]+k0*a[2],y1*a[1]-a[5]+k1*a[3]
                    else:
                        t0,t1 = y0+a[4]-k0*a[2],y1+a[5]-k1*a[3]
                        if t0%a[0] or t1%a[1]: continue
                        p0,p1 = t0//a[0],t1//a[1]
                    if not 0<=p0<x.shape[2] or not 0<=p1<x.shape[3]: continue
                    xv = qx[batch,group*ci:(group+1)*ci,p0,p1].astype(np.int64)
                    wv = (weight_q[channel,:,k0,k1] if case["op"] == 24 else
                          weight_q[group*ci:(group+1)*ci,channel%co,k0,k1]).astype(np.int64)
                    dot += xv@wv
        bias = inputs[2][channel] if len(inputs) == 3 else 0
        expected.append(float(dot)*sx*float(sw[channel])+float(bias))
    return positions,np.asarray(expected)


def stress(libraries, threads):
    case = {"op":24,"input_shapes":[[1,64,1,4],[16,64,1,1],[16]],
            "output_shape":[1,16,1,4], "attrs":[1,1,1,1,0,0,0,0,1]}
    s = descriptor(case)
    inputs = [np.ones(256,dtype="f4"),np.full(1024,0.25,dtype="f4"),np.full(16,0.5,dtype="f4")]
    row = {}
    for provider,(lib,_) in libraries.items():
        pointers = [v.ctypes.data_as(ops.FLOAT_P) for v in inputs]
        handle = lib.bench_create(ct.byref(s),*pointers,threads,1)
        if not handle: raise RuntimeError(lib.bench_error().decode())
        try:
            output = np.empty(64,dtype="f4")
            times = (ct.c_double*1)()
            if lib.bench_measure(handle,True,1,1,1,times,output.ctypes.data_as(ops.FLOAT_P)):
                raise RuntimeError(lib.bench_error().decode())
            row[provider] = {"implementation":lib.bench_implementation(handle).decode(),
                "max_abs":float(abs(output-16.5).max()),"expected":16.5,"actual_first":float(output[0])}
        finally: lib.bench_destroy(handle)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=Path, default=ROOT / "target/compare/cpu-operators")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--samples", type=int, nargs="+", default=[11200,44960])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--isa", choices=["default","AVX2","AVX2_VNNI"], default="default")
    ap.add_argument("--reduced-range", action="store_true")
    ap.add_argument("--even-weight-grid", action="store_true", help="AVX2 signed-conv probe: even codes in [-126,126], effectively 6-bit magnitudes")
    ap.add_argument("--cases", nargs="+")
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()
    if args.isa != "default": os.environ["ONEDNN_MAX_CPU_ISA"] = args.isa
    if args.reduced_range: os.environ["VOCAL_INT8_REDUCED_RANGE"] = "1"
    if args.even_weight_grid: os.environ["VOCAL_INT8_EVEN_WEIGHT_GRID"] = "1"
    affinity = ops.performance_cpu_affinity()
    libraries = {provider:ops.load_library(provider,ops.CACHE/"benchmark-build",ROOT/"target/cpu-sdk/torch")
                 for provider in ["onednn","onednn_int8"]}
    report = {"note":__doc__,"affinity":affinity,"isa":args.isa,"threads":args.threads,
              "range":63 if args.reduced_range else 127,"cases":[]}
    report["even_weight_grid"] = args.even_weight_grid
    report["full_positive_dot_stress"] = stress(libraries,args.threads)
    print(json.dumps(report["full_positive_dot_stress"]),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for samples in args.samples:
        fixtures = args.fixtures/str(samples)
        for case in json.loads((fixtures/"manifest.json").read_text())["cases"]:
            if args.cases and case["id"] not in args.cases: continue
            s = descriptor(case)
            inputs = [np.fromfile(fixtures/f"{case['id']}_input_{i}.f32",dtype="<f4") for i in range(len(case["input_shapes"]))]
            reference = np.fromfile(fixtures/f"{case['id']}_output.f32",dtype="<f4")
            row = dict(case,samples=samples)
            positions,expected = quantized_samples(case,inputs,report["range"],args.even_weight_grid)
            for provider,(lib,_) in libraries.items():
                pointers = [v.ctypes.data_as(ops.FLOAT_P) for v in inputs]
                handle = lib.bench_create(ct.byref(s),pointers[0],pointers[1],pointers[2] if s.bias else None,args.threads,1)
                if not handle:
                    row[provider] = {"unsupported":lib.bench_error().decode()}; continue
                try:
                    result = np.empty_like(reference)
                    result_row = {"implementation":lib.bench_implementation(handle).decode()}
                    for boundary in [False,True]:
                        pilot = (ct.c_double*1)()
                        if lib.bench_measure(handle,boundary,3,1,1,pilot,result.ctypes.data_as(ops.FLOAT_P)):
                            raise RuntimeError(lib.bench_error().decode())
                        loops = min(64,max(1,int(3/max(pilot[0],0.001))))
                        times = (ct.c_double*args.repeats)()
                        if lib.bench_measure(handle,boundary,3,args.repeats,loops,times,result.ctypes.data_as(ops.FLOAT_P)):
                            raise RuntimeError(lib.bench_error().decode())
                        result_row["boundary" if boundary else "prepared"] = {"median_ms":float(np.median(times)),
                            "p95_ms":float(np.percentile(times,95)), "batches_ms":list(times)}
                    result_row["error_vs_fp32"] = ops.metric(reference,result)
                    if provider.endswith("int8"):
                        delta = result[positions].astype(np.float64)-expected
                        result_row["integer_dot_check"] = {"max_abs":float(np.max(np.abs(delta))),
                            "relative_rmse":float(np.linalg.norm(delta)/max(np.linalg.norm(expected),1e-20))}
                        # FP32 reciprocal in the oneDNN input reorder can move
                        # rare half-grid ties by one integer. This tolerance
                        # permits that rounding effect, but catches saturation.
                        if result_row["integer_dot_check"]["relative_rmse"] > 1e-4:
                            result_row["numerical_failure"] = "integer-dot mismatch (scales/layout or AVX2 saturation)"
                    elif result_row["error_vs_fp32"]["relative_rmse"] > 1e-4:
                        raise RuntimeError(f"FP32 operator regression: {case['id']}")
                    row[provider] = result_row
                finally:
                    lib.bench_destroy(handle)
            report["cases"].append(row)
            args.output.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps({"samples":samples,"case":case["id"],"providers":{
                p:row[p].get("boundary",row[p]).get("median_ms",row[p]) for p in libraries}}),flush=True)


if __name__ == "__main__":main()
