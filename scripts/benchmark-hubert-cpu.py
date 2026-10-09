"""Serial native CPU HuBERT quantization and cache-attention measurements.

Paired complete pipelines use the same held-out inputs/noise and six P-core
threads. Microbenchmarks include graph binding, output layout and copy costs.
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

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory",type=Path,default=ROOT/"target/hubert-cpu")
    ap.add_argument("--iterations",type=int,default=32)
    ap.add_argument("--only",choices=["micro","pipeline","all"],default="all")
    ap.add_argument("--recipes",nargs="+",default=["qk-alpha-1","vout-alpha0.5","ffn_in-alpha0.7","mixed","linear-alpha0.7"])
    ap.add_argument("--configs",nargs="+",choices=["dense_token","online_4_32","online_4_64"],default=["dense_token","online_4_32","online_4_64"])
    ap.add_argument("--resume",action="store_true",help="append missing jobs with unchanged binary, fixtures and calibration")
    ap.add_argument("--jobs",nargs="+",help="optional subset of generated job names")
    args=ap.parse_args()
    if args.iterations<16:ap.error("at least 16 pairs required")
    ops=module("operators",ROOT/"scripts/benchmark-cpu-operators.py")
    metrics=module("quant_bench",ROOT/"scripts/benchmark-cpu-w8a8.py")
    affinity=ops.performance_cpu_affinity()
    native=ROOT/"target/debug/examples"
    if args.only in ("micro","all"):
        output=args.directory/"native-attention.json"
        subprocess.run([str(native/"attention_benchmark.exe"),str(args.directory.resolve()),str(output.resolve()),str(args.iterations*4)],check=True)
        data=json.loads(output.read_text())
        summaries=[]
        for samples in [11200,44960]:
            for variant in ["dense_token","online_4_32","online_4_64","online_8_64"]:
                rows=[r for r in data["rows"] if r["case"]["samples"]==samples and r["variant"]==variant]
                base=sum(np.median(r["baseline_ms"]) for r in rows)
                actual=sum(np.median(r["candidate_ms"]) for r in rows)
                summaries.append(dict(samples=samples,variant=variant,baseline_12_layers_ms=base,
                    candidate_12_layers_ms=actual,speedup=base/actual,
                    max_abs=max(r["reference_error"]["max_abs"] for r in rows),
                    reference_max_rmse=max(r["reference_error"]["rmse"] for r in rows),
                    fp32_max_abs=max(r["fp32_error"]["max_abs"] for r in rows),
                    score_workspace_bytes=rows[0]["candidate_memory"]["workspace_bytes"],
                    baseline_workspace_bytes=rows[0]["baseline_memory"]["workspace_bytes"]))
        data["summary"]=summaries;data["affinity"]=affinity
        output.write_text(json.dumps(data,indent=2)+"\n")
        for row in summaries:print(json.dumps(row),flush=True)
    if args.only=="micro":return
    plans=json.loads((args.directory/"recipes.json").read_text())
    fixtures=ROOT/"target/cpu-w8a8/broad/inputs.json"
    meta=json.loads(fixtures.read_text())
    calibration=ROOT/"target/cpu-w8a8/broad/native/calibration.json"
    report=dict(affinity=affinity,iterations=args.iterations,
        note="native oneDNN static W8A8, 8 warmup pairs then alternating FP32/candidate; only HuBERT changes; index=0, fixed noise; process_exact includes stage IO, quantization/layout/online-softmax and output copy, excludes model construction, WAV IO, SOLA",
        inputs_sha256=hashlib.sha256(fixtures.read_bytes()).hexdigest(),
        calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256((native/"quantization_benchmark.exe").read_bytes()).hexdigest(),results=[])
    jobs=[(n,"-",n) for n in args.configs]+[(n,n,"default") for n in args.recipes]
    if "mixed" in args.recipes:
        jobs.extend([("mixed_online","mixed","online_4_32"),("mixed_dense","mixed","dense_token")])
    if args.jobs:
        if set(args.jobs)-{j[0] for j in jobs}:ap.error("unknown job")
        jobs=[j for j in jobs if j[0] in args.jobs]
    if args.resume and (args.directory/"native-pipeline.json").is_file():
        saved=json.loads((args.directory/"native-pipeline.json").read_text())
        if any(saved[k]!=report[k] for k in ("binary_sha256","calibration_sha256","inputs_sha256","iterations","affinity")):
            raise ValueError("resume requires identical benchmark artifacts and affinity")
        report=saved
    for samples in [11200,44960]:
        for label,recipe,config in jobs:
            if any(r["samples"]==samples and r["name"]==label for r in report["results"]):continue
            directory=args.directory/"native"/f"{samples}-{label}";directory.mkdir(parents=True,exist_ok=True)
            command=[str(native/"quantization_benchmark.exe"),str(ROOT/"target/default-canonical-with-index.vfr"),
                     str(ROOT/"models/shared/hubert_base/hubert_features.vfr"),str(ROOT/"models/shared/rmvpe/rmvpe.vfr"),
                     str(fixtures),"-",str(calibration),str(directory),str(samples),str(args.iterations)]
            if recipe!="-":
                path=directory/"recipe.json";path.write_text(json.dumps(plans[recipe]));command[5]=str(path)
            if config!="default":
                settings=dict(window_samples=samples,skip_head_frames=250 if samples==44960 else 8,threads=6,
                    attention=dict(mode="online" if config.startswith("online") else "dense",query_tile=4,
                                   key_tile=64 if config.endswith("64") else 32,token_major_output=True))
                path=directory/"config.json";path.write_text(json.dumps(settings));command.append(str(path))
            proc=subprocess.run(command,capture_output=True,text=True,env=os.environ)
            if proc.returncode:raise RuntimeError(proc.stderr or proc.stdout)
            candidate=json.loads(proc.stdout);base=candidate["baseline"]
            if recipe!="-" and len(candidate["quantized"])!=len(plans[recipe]["rules"]):
                raise ValueError("native quantized layer coverage differs from the Python recipe")
            (directory/"raw.json").write_text(json.dumps(candidate,indent=2)+"\n")
            comparisons=[]
            for q,b in zip(candidate["outputs"],base["outputs"]):
                reference=np.fromfile(Path(q["reference_directory"])/"reference.f32",dtype="<f4")
                actual=np.fromfile(q["audio"],dtype="<f4");fp32=np.fromfile(b["audio"],dtype="<f4")
                comparisons.append(dict(split=q["split"],offset=q["offset"],applio=metrics.metric(reference,actual),
                    native_fp32=metrics.metric(fp32,actual),fp32_applio=metrics.metric(reference,fp32)))
            row=dict(samples=samples,name=label,recipe=recipe,config=candidate["config"],
                     fp32_ms=base["mean_ms"],candidate_ms=candidate["mean_ms"],
                     speedup=base["mean_ms"]["total"]/candidate["mean_ms"]["total"],
                     hubert_speedup=base["mean_ms"]["hubert"]/candidate["mean_ms"]["hubert"],
                     audio=metrics.summarize(comparisons,meta["sample_rate"]),
                     baseline_memory=base["memory"],candidate_memory=candidate["memory"],
                     quantized=candidate["quantized"],rows_ms=dict(fp32=base["rows_ms"],candidate=candidate["rows_ms"]))
            if any("vnni" not in n["implementation"] and "amx" not in n["implementation"] for n in candidate["quantized"]):
                raise ValueError("native integer hardware path not selected")
            report["results"].append(row)
            (args.directory/"native-pipeline.json").write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps({k:row[k] for k in ["samples","name","speedup","hubert_speedup"]}|
                dict(fp32_ms=row["fp32_ms"]["total"],candidate_ms=row["candidate_ms"]["total"],
                     speech_cosine_min=row["audio"]["speech_cosine_min"],rmse=row["audio"]["rmse_mean"])),flush=True)


if __name__=="__main__":main()
