"""Run complete CPU WAV conversion and independent Applio FP32 parity.

Only random latent/NSF tensors are shared. Features, F0 and index are extracted
independently. Traces are diagnostic; use benchmark-cpu-migration.py for timing.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--index-ratio",type=float,default=0.)
    ap.add_argument("--model",type=Path,default=ROOT/"target/default-complete-with-index.vfrm")
    ap.add_argument("--checkpoint",type=Path,default=ROOT/"models/voices/default-v2/default.pth")
    ap.add_argument("--index",type=Path,default=ROOT/"models/voices/default-v2/added_IVF511_Flat_nprobe_1_default_v2.index")
    ap.add_argument("--binary",type=Path,default=ROOT/"target/debug/vocal-fast.exe")
    ap.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args=ap.parse_args()
    if not 0<=args.index_ratio<=1: ap.error("index ratio must be in [0,1]")
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    def run(command,name,env=None):
        result=subprocess.run([str(v) for v in command],text=True,capture_output=True,env=env)
        (out/f"{name}.log").write_text(result.stdout+result.stderr,encoding="utf-8")
        if result.returncode: raise RuntimeError(f"{name} failed ({result.returncode}); see {out/name}.log")
        return result.stdout
    env=dict(os.environ,VOCAL_CPU_THREADS="6",VOCAL_AUTO_PITCH="false",VOCAL_INDEX_RATIO=str(args.index_ratio),VOCAL_OFFLINE_TRACE_DIR=str(out/"native-trace"))
    native=run([args.binary,"model","run-pipeline","cpu",args.model,args.input,out/"native.wav"],"native",env)
    if json.loads(native).get("backend_config",{}).get("operator_library",{}).get("name")!="oneDNN":
        raise RuntimeError("stale CLI binary: rebuild vocal-fast with cpu-graph (without --examples)")
    (out/"native-report.json").write_text(native,encoding="utf-8")
    trace=out/"native-trace/0000"
    run([sys.executable,ROOT/"scripts/run-applio-offline.py","--input",args.input,"--output",out/"applio.wav","--checkpoint",args.checkpoint,"--index",args.index,"--index-ratio",args.index_ratio,"--native-noise",trace,"--trace-dir",out/"applio-trace","--device",args.device,"--report",out/"applio-report.json"],"applio")
    run([sys.executable,ROOT/"scripts/compare-aligned-offline.py","--native",trace,"--reference",out/"applio-trace","--native-wav",out/"native.wav","--reference-wav",out/"applio.wav","--checkpoint",args.checkpoint,"--device",args.device,"--report",out/"parity.json"],"compare")
    report=json.loads((out/"parity.json").read_text())
    print(json.dumps({"wav":str(out/"native.wav"),"final_wav":report["metrics"]["final_wav"],"f0_diagnostics":report["f0_diagnostics"]},indent=2))
    if report["metrics"]["final_wav"]["cosine"]<0.99999 or report["f0_diagnostics"]["coarse_mismatch_frames"]:
        raise RuntimeError("Applio parity regression; inspect parity.json")

if __name__=="__main__":main()
