"""Merge native PTQ phases without losing per-window or low-volume errors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory",type=Path,default=ROOT/"target/cpu-w8a8/broad")
    args = ap.parse_args()
    sources = [args.directory/"native-results.json",args.directory/"folded-bn/native-results.json"]
    merged, provenance = {}, []
    rng = np.random.default_rng(4826)
    for path in sources:
        report = json.loads(path.read_text())
        provenance.append(dict(path=str(path),contract=report["contract"],note=report["note"]))
        for row in report["results"]:
            row["report"] = str(path)
            mode = report.get("deconv_batchnorm", "fold" if path.parent.name == "folded-bn" else "preserve")
            row["deconv_batchnorm_folded"] = mode == "fold"
            directory = path.parent/"native"/f"{row['samples']}-{row['name']}-int8"
            for window in row["audio"]["windows"]:
                actual = np.fromfile(directory/f"{window['split']}-{window['offset']}.f32",dtype="<f4").astype(np.float64)
                window["applio"]["actual_rms"] = float(np.sqrt(np.mean(actual*actual)))
            pairs = np.asarray(row["rows_ms"]["fp32"])[:,-1],np.asarray(row["rows_ms"]["int8"])[:,-1]
            # Each block contains one complete cycle through four fixtures.
            # This interval measures within-run variability, not cross-machine
            # or multi-voice generalization.
            n = len(pairs[0])//4
            blocks = rng.integers(0,n,size=(2000,n))
            indexes = (blocks[:,:,None]*4+np.arange(4)[None,None,:]).reshape(2000,-1)
            ratios = pairs[0][indexes].mean(1)/pairs[1][indexes].mean(1)
            row["latency"]["within_run_speedup_ci95"] = np.percentile(ratios,[2.5,97.5]).tolist()
            row["packed_weight_saving_mib"] = (sum(m["packed_bytes"] for m in row["fp32_memory"].values())-
                sum(m["packed_bytes"] for m in row["int8_memory"].values()))/2**20
            row["weight_fraction"] = sum(c["weight_elements"] for c in row["coverage"].values())/sum(c["model_parameters"] for c in report["eligible"].values())
            row["coarse_mismatch_max"] = max(w["coarse_mismatch"] for w in row["audio"]["windows"])
            row["voicing_mismatch_max"] = max(w["voicing_mismatch"] for w in row["audio"]["windows"])
            row["f0_max_abs"] = max(w["f0"]["max_abs"] for w in row["audio"]["windows"])
            merged[row["name"],row["samples"]] = row
    rows = list(merged.values())
    summary = dict(source_reports=provenance,note="updated deconv-BN phase replaces only its four affected recipes; every numerical error is versus original Applio FP32; silence retained",results=rows)
    (args.directory/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("name samples fp32_ms int8_ms speedup speech_cos all_cos rmse max_abs f0_max coarse max_voicing")
    for r in rows:
        l,a = r["latency"],r["audio"]
        print(r["name"],r["samples"],f"{l['fp32']['total']:.2f}",f"{l['int8']['total']:.2f}",f"{l['speedup']:.3f}",
              f"{a['speech_cosine_min']:.9f}",f"{a['cosine_min']:.9f}",f"{a['rmse_mean']:.6g}",f"{a['max_abs']:.6g}",
              f"{r['f0_max_abs']:.5g}",r["coarse_mismatch_max"],r["voicing_mismatch_max"])
    for r in rows:
        if r["name"] in ["rmvpe_intermediate","stage2","res_all","generator","broad"]:
            quiet = [w for w in r["audio"]["windows"] if w["applio"]["reference_rms"]<.01]
            print("detail",r["name"],r["samples"],"packed_saved_mib",r["packed_weight_saving_mib"],
                  "ci",r["latency"]["within_run_speedup_ci95"],"quiet",[(w["applio"]["reference_rms"],w["applio"]["actual_rms"],w["applio"]["rmse"]) for w in quiet])


if __name__ == "__main__": main()
