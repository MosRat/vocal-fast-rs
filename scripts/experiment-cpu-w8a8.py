"""Calibrated W8A8 region sweep on held-out Mandarin with Applio FP32 reference.

GPU FP32 fake quantization is used only to screen accuracy, never CPU speed.
Calibration and test windows are disjoint; latent/NSF noise is identical.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.io import wavfile
from transformers import HubertModel
from cpu_quantization import calibrate, quantized, fold_batchnorm, KINDS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref_repos/Applio"))
from rvc.lib.predictors.RMVPE import RMVPE0Predictor


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ref = load_script("retrieval_reference", "compare-retrieval-python.py")
f0mod = load_script("rmvpe_precision", "rmvpe-precision-applio.py")


def aggregate(metrics):
    return {"cosine_min":min(m["cosine"] for m in metrics),
            "rmse_mean":float(np.mean([m["rmse"] for m in metrics])),
            "max_abs":max(m["max_abs"] for m in metrics), "per_window":metrics}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "target/cpu-w8a8/accuracy.json")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models/voices/default-v2/default.pth")
    ap.add_argument("--components", nargs="+", choices=["synth", "hubert", "rmvpe"], default=["synth", "hubert", "rmvpe"])
    ap.add_argument("--alpha", type=float, nargs="+", default=[-1, 0.5, 0.7])
    ap.add_argument("--percentile", type=float, default=1.)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--regions", nargs="+", help="optional region names to screen")
    ap.add_argument("--dynamic-activation", action="store_true", help="scalar max per invocation, includes a runtime reduction")
    args = ap.parse_args()
    if not 0 < args.percentile <= 1 or any(a != -1 and not 0 <= a <= 1 for a in args.alpha):
        ap.error("invalid clipping/SmoothQuant parameters")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = "cuda"
    waveform = np.fromfile(ROOT / "target/compare/cpu-final2-trace/0000/waveform.bin", dtype="<f4")
    windows = []
    for split, offsets in [("calibration", [32000, 112000]), ("test", [208000, 352000])]:
        for samples in [11200, 44960]:
            for offset in offsets:
                windows.append({"split":split, "samples":samples, "offset":offset,
                    "wave":torch.tensor(waveform[offset:offset+samples], device=device).unsqueeze(0)})
    calibration = [w for w in windows if w["split"] == "calibration"]
    tests = [w for w in windows if w["split"] == "test"]
    hubert = HubertModel.from_pretrained(str(ROOT / "models/shared/hubert_base"), local_files_only=True,
                                       attn_implementation="eager").to(device).float().eval()
    predictor = RMVPE0Predictor(str(ROOT / "models/shared/rmvpe/rmvpe.pt"), device=device)
    rmvpe = predictor.model.float().eval()
    synth, width, sample_rate = ref.load_synth(args.checkpoint, device)
    for layer in synth.modules():
        if hasattr(layer, "weight_g"):
            nn.utils.remove_weight_norm(layer)
    rng = np.random.default_rng(24601)

    def run_hubert(item):
        return hubert(item["wave"]).last_hidden_state.float().cpu().numpy()[0]

    def run_rmvpe(item):
        salience = rmvpe(item["mel"])[:, :item["mel_frames"]].float().cpu().numpy()[0]
        return salience, predictor.decode(salience, thred=0.03)

    for item in windows:
        mel = predictor.mel_extractor(item["wave"], center=True)
        frames = mel.shape[-1]
        item["mel_frames"] = frames
        item["mel"] = F.pad(mel, (0, ((frames+31)//32)*32-frames), mode="reflect")
        hidden = run_hubert(item)
        salience, pitchf = run_rmvpe(item)
        frames = item["samples"]//160
        skip = 250 if item["samples"] == 44960 else 8
        features = np.repeat(np.concatenate([hidden, hidden[-1:]]), 2, axis=0)[:frames]
        pitchf = pitchf[:frames].astype(np.float32)
        snapshot = {"synth_frames":frames, "valid_synth_frames":frames,
            "synth_skip_head":skip, "synth_output_frames":frames-skip,
            "rvc_features":features, "pitch":f0mod.quantize_f0(pitchf), "aligned_pitchf":pitchf,
            "synth_random":rng.standard_normal((1, synth.enc_p.out_channels, frames), dtype=np.float32),
            "synth_noise":rng.standard_normal((1, 1, frames*(sample_rate//100)), dtype=np.float32)}
        item.update(hidden=hidden, salience=salience, pitchf=pitchf, snapshot=snapshot)
        item["audio"] = ref.synth_reference(synth, snapshot, width, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"reference":"original Applio Synth/RMVPE + local HF HuBERT FP32, TF32 disabled",
        "experiment_contract":2,
        "torch":torch.__version__, "device":torch.cuda.get_device_name(),
        "mode":f"fake W8A8, symmetric -127..127, scalar {'dynamic' if args.dynamic_activation else 'static'} activation, output-channel weights, FP32 bias/output",
        "note":"calibration/test disjoint, no fake-quant latency claims, index disabled, common latent/NSF noise",
        "checkpoint":str(args.checkpoint), "windows":[{k:w[k] for k in ["split", "samples", "offset"]} for w in windows],
        "regions":[], "calibration":{}}
    if args.resume and args.output.is_file():
        saved = json.loads(args.output.read_text())
        if any(saved.get(k) != report[k] for k in ["reference", "experiment_contract", "mode", "checkpoint", "windows"]):
            raise ValueError("resume requires the same reference, model and input windows")
        report = saved
    # Reference F0/audio above use the original Conv->BN implementation. Fold
    # only the quantized candidate, recording its FP32 baseline independently.
    fold_batchnorm(rmvpe)
    fold_metrics = []
    for item in tests:
        salience, f0 = run_rmvpe(item)
        f0 = f0[:item["snapshot"]["synth_frames"]].astype(np.float32)
        pitch = f0mod.quantize_f0(f0)
        audio = ref.synth_reference(synth,dict(item["snapshot"],pitch=pitch,aligned_pitchf=f0),width,device)
        fold_metrics.append({"salience":ref.metric(item["salience"],salience),
            "f0":ref.metric(item["pitchf"],f0), "audio":ref.metric(item["audio"],audio),
            "coarse_mismatch_frames":int(np.count_nonzero(pitch != item["snapshot"]["pitch"]))})
    report["rmvpe_batchnorm_fold"] = fold_metrics
    policies = {
        "synth": {
            "resblocks": lambda n,l:n.startswith("dec.resblocks."),
            **{f"res_stage_{i}":lambda n,l,i=i:n.startswith("dec.resblocks.") and int(n.split(".")[2])//3==i for i in range(len(synth.dec.ups))},
            "ups":lambda n,l:n.startswith("dec.ups."),
            "pre_post":lambda n,l:n in ["dec.conv_pre", "dec.conv_post"],
            "noise":lambda n,l:n.startswith("dec.noise_convs."),
            "flow":lambda n,l:n.startswith("flow."),
            "enc_p":lambda n,l:n.startswith("enc_p."),
            "decoder":lambda n,l:n.startswith("dec.") and not n.startswith("dec.m_source."),
        },
        "hubert": {
            "ffn":lambda n,l:"feed_forward." in n,
            "attention_linear":lambda n,l:"attention." in n and isinstance(l,nn.Linear),
            "extractor":lambda n,l:n.startswith("feature_extractor."),
            "all_linear":lambda n,l:isinstance(l,nn.Linear),
        },
        "rmvpe": {
            "encoder":lambda n,l:n.startswith("unet.encoder."),
            "intermediate":lambda n,l:n.startswith("unet.intermediate."),
            "decoder":lambda n,l:n.startswith("unet.decoder."),
            "unet":lambda n,l:n.startswith("unet."),
            "head":lambda n,l:n=="cnn",
            "logits":lambda n,l:n=="fc.1",
        }
    }
    invocations = {"hubert":run_hubert, "rmvpe":run_rmvpe,
                   "synth":lambda item:ref.synth_reference(synth,item["snapshot"],width,device)}
    for component in args.components:
        net = {"hubert":hubert,"rmvpe":rmvpe,"synth":synth}[component]
        invoke = invocations[component]
        stats = calibrate(net, calibration, invoke)
        report["calibration"][component] = {name:{"calls":s.calls,"max":float(s.channel_max.max()),
            "p999_sample_abs":float(torch.quantile(torch.cat(s.samples,dim=1).flatten(),0.999)),
            "weight_elements":net.get_submodule(name).weight.numel()} for name,s in stats.items()}
        for region, select in policies[component].items():
            if args.regions and region not in args.regions:
                continue
            for alpha in args.alpha:
                if any(r["component"] == component and r["region"] == region and r["alpha"] == alpha
                       and r["percentile"] == args.percentile
                       and r.get("dynamic_activation", False) == args.dynamic_activation for r in report["regions"]):
                    continue
                metrics, stage_metrics, pitch_diagnostics = [], [], []
                with quantized(net, stats, select, None if alpha == -1 else alpha, args.percentile, args.dynamic_activation) as wrappers:
                    if not wrappers:
                        continue
                    for i, item in enumerate(tests):
                        result = invoke(item)
                        if component == "synth":
                            audio = result
                        elif component == "hubert":
                            stage_metrics.append(ref.metric(item["hidden"],result))
                            snapshot = dict(item["snapshot"], rvc_features=np.repeat(
                                np.concatenate([result,result[-1:]]),2,axis=0)[:item["snapshot"]["synth_frames"]])
                            audio = ref.synth_reference(synth,snapshot,width,device)
                        else:
                            salience, f0 = result
                            stage_metrics.append(ref.metric(item["salience"],salience))
                            f0 = f0[:item["snapshot"]["synth_frames"]].astype(np.float32)
                            pitch = f0mod.quantize_f0(f0)
                            pitch_diagnostics.append({"coarse_mismatch_frames":int(np.count_nonzero(pitch != item["snapshot"]["pitch"])),
                                "voicing_mismatch_frames":int(np.count_nonzero((f0>0)!=(item["pitchf"]>0))),
                                "f0":ref.metric(item["pitchf"],f0)})
                            audio = ref.synth_reference(synth,dict(item["snapshot"],pitch=pitch,aligned_pitchf=f0),width,device)
                        metrics.append(ref.metric(item["audio"],audio))
                        if i == 0 and component == "synth" and region == "res_stage_0":
                            wavfile.write(str(args.output.parent/f"{component}-{region}-alpha{alpha}.wav"),sample_rate,audio)
                            wavfile.write(str(args.output.parent/"reference.wav"),sample_rate,item["audio"])
                    row = {"component":component,"region":region,"alpha":alpha,"percentile":args.percentile,
                           "dynamic_activation":args.dynamic_activation,
                           "layers":len(wrappers),"weight_elements":sum(w.weight.numel() for w in wrappers),
                           "heldout_clipped_fraction":sum(w.clipped for w in wrappers)/max(1,sum(w.elements for w in wrappers)),
                           "audio":aggregate(metrics)}
                    if stage_metrics: row["stage"] = aggregate(stage_metrics)
                    if pitch_diagnostics: row["pitch"] = pitch_diagnostics
                report["regions"].append(row)
                args.output.write_text(json.dumps(report,indent=2)+"\n")
                print(json.dumps({k:row[k] for k in ["component","region","alpha","layers","heldout_clipped_fraction"]}|{
                    "cos_min":row["audio"]["cosine_min"],"rmse":row["audio"]["rmse_mean"]}),flush=True)


if __name__ == "__main__":
    main()
