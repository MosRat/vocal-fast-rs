"""Compact HuBERT accuracy/cache report with paired within-run speed intervals."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def interval(rows,column):
    a=np.asarray(rows["fp32"])[:,column];b=np.asarray(rows["candidate"])[:,column]
    if len(a)!=len(b) or len(a)%4:raise ValueError("paired complete four-fixture cycles required")
    rng=np.random.default_rng(369)
    n=len(a)//4
    blocks=rng.integers(0,n,size=(2000,n))
    indexes=(blocks[:,:,None]*4+np.arange(4)[None,None,:]).reshape(2000,-1)
    return np.percentile(a[indexes].mean(1)/b[indexes].mean(1),[2.5,97.5]).tolist()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory",type=Path,default=ROOT/"target/hubert-cpu")
    args=ap.parse_args()
    accuracy=json.loads((args.directory/"accuracy.json").read_text())
    native=json.loads((args.directory/"native-pipeline.json").read_text())
    attention=json.loads((args.directory/"native-attention.json").read_text())
    rows=[]
    for r in native["results"]:
        windows=r["audio"]["windows"]
        packed=lambda m:m["hubert"]["packed_bytes"]
        row={k:r[k] for k in ("samples","name","fp32_ms","candidate_ms","speedup","hubert_speedup")}
        row.update(speedup_ci95=interval(r["rows_ms"],-1),hubert_speedup_ci95=interval(r["rows_ms"],0),
            applio_cosine_min=r["audio"]["cosine_min"],speech_cosine_min=r["audio"]["speech_cosine_min"],
            rmse=r["audio"]["rmse_mean"],max_abs=r["audio"]["max_abs"],
            native_fp32_max_abs=max(w["native_fp32"]["max_abs"] for w in windows),
            packed_weight_saving_mib=(packed(r["baseline_memory"])-packed(r["candidate_memory"]))/2**20,
            quantized_layers=len(r["quantized"]),
            quiet=[w for w in windows if w["applio"]["reference_rms"]<.01])
        rows.append(row)
        print(json.dumps(row),flush=True)
    report=dict(reference=accuracy["reference"],
                note="latency from native CPU; FP32 Applio/HF audio reference, quiet windows retained; cycle block-bootstrap intervals measure within-run variance only; paired controls vary between jobs, do not rank close absolute times from separate runs",
                pipeline=rows,attention=attention["summary"],
                python=[{k:r[k] for k in ("name","alpha","layers","weight_elements","clipped_fraction","audio","hidden")} for r in accuracy["quantization"]],
                rescue=accuracy.get("rescue",[]))
    (args.directory/"summary.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
