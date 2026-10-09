"""Cross-stage static W8A8 screening and shared inputs for real CPU benchmarks.

Original Applio FP32 Synth/RMVPE and local HF HuBERT are the numerical reference.
Python fake quantization screens accuracy only; CPU latency is measured separately.
"""
from __future__ import annotations
from contextlib import ExitStack
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.io import wavfile
from transformers import HubertModel
from cpu_quantization import calibrate, quantized, fold_batchnorm
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("w8a8_regions", ROOT / "scripts/experiment-cpu-w8a8.py")
region = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = region
spec.loader.exec_module(region)
ref, f0mod = region.ref, region.f0mod


def rmvpe_names():
    """Explicit architecture mapping, including shortcuts and deconvolutions."""
    result = {"cnn": "rmvpe.node.344", "fc.1": "rmvpe.node.360"}
    for i in range(5):
        for j in range(4):
            base = 28 + 22*i if j == 0 else 29 + 22*i + 5*j
            prefix = f"unet.encoder.layers.{i}.conv.{j}"
            result[prefix+".conv.0"] = f"rmvpe.node.{base}"
            result[prefix+".conv.3"] = f"rmvpe.node.{base+2}"
            if j == 0: result[prefix+".shortcut"] = f"rmvpe.node.{base+4}"
    for k in range(16):
        base = 138 if k == 0 else 139 + 5*k
        prefix = f"unet.intermediate.layers.{k//4}.conv.{k%4}"
        result[prefix+".conv.0"] = f"rmvpe.node.{base}"
        result[prefix+".conv.3"] = f"rmvpe.node.{base+2}"
        if k == 0: result[prefix+".shortcut"] = "rmvpe.node.142"
    for i in range(5):
        prefix = f"unet.decoder.layers.{i}"
        result[prefix+".conv1.0"] = f"rmvpe.node.{219+25*i}"
        for j in range(4):
            base = 223+25*i if j == 0 else 224+25*i+5*j
            result[f"{prefix}.conv2.{j}.conv.0"] = f"rmvpe.node.{base}"
            result[f"{prefix}.conv2.{j}.conv.3"] = f"rmvpe.node.{base+2}"
            if j == 0: result[f"{prefix}.conv2.{j}.shortcut"] = f"rmvpe.node.{base+4}"
    return result


def recipes():
    inter = {"rmvpe": [("intermediate", None)]}
    safe = {**inter, "synth": [("noise", None)]}
    flow = {**inter, "synth": [("noise", None), ("flow", .7)]}
    return {
        "rmvpe_intermediate": inter,
        "safe": safe,
        "flow": flow,
        "stage2": {**flow, "synth": flow["synth"]+[("stage2", .7)]},
        "res_convs2": {**flow, "synth": flow["synth"]+[("res_convs2", .7)]},
        "res_convs1": {**flow, "synth": flow["synth"]+[("res_convs1", .7)]},
        "res_late": {**flow, "synth": flow["synth"]+[("res_late", .7)]},
        "res_all": {**flow, "synth": flow["synth"]+[("res_all", .7)]},
        "generator": {**inter, "synth": [("decoder", .7), ("flow", .7)]},
        "attention": {**flow, "hubert": [("attention", .5)]},
        "attention_stage2": {**flow, "hubert": [("attention", .5)], "synth": flow["synth"]+[("stage2", .7)]},
        "late_ffn": {**flow, "hubert": [("late_ffn", .7)]},
        "hubert_linear": {**flow, "hubert": [("linear", .5)]},
        "rmvpe_decoder_late": {**flow, "rmvpe": inter["rmvpe"]+[("decoder_late", .7)]},
        "rmvpe_decoder": {**flow, "rmvpe": inter["rmvpe"]+[("decoder", .7)]},
        "rmvpe_unet": {**flow, "rmvpe": [("unet", None)]},
        "balanced": {"rmvpe": inter["rmvpe"], "hubert": [("attention", .5)],
                     "synth": [("decoder", .7), ("flow", .7)]},
        "broad": {"rmvpe": [("unet", None), ("head", .5), ("logits", None)],
                  "hubert": [("linear", .5)], "synth": [("all", .7)]},
    }


def selected(component, name, layer, region_name):
    if component == "synth":
        res = name.startswith("dec.resblocks.")
        stage = int(name.split(".")[2])//3 if res else -1
        return {"noise": name.startswith("dec.noise_convs."), "flow": name.startswith("flow."),
                "stage2": res and stage == 2, "res_all": res,
                "res_late": res and stage >= 2,
                "res_convs1": res and ".convs1." in name,
                "res_convs2": res and ".convs2." in name,
                "decoder": name.startswith("dec.") and not name.startswith("dec.m_source."),
                "all": not name.startswith("dec.m_source.")}[region_name]
    if component == "hubert":
        return {"attention": "attention." in name and isinstance(layer, nn.Linear),
                "linear": isinstance(layer, nn.Linear),
                "late_ffn": "feed_forward." in name and int(name.split(".")[2]) >= 8
                    if name.startswith("encoder.layers.") else False}[region_name]
    return {"intermediate": name.startswith("unet.intermediate."),
            "decoder": name.startswith("unet.decoder."),
            "decoder_late": name.startswith("unet.decoder.layers.") and int(name.split(".")[3]) >= 3,
            "unet": name.startswith("unet."), "head": name == "cnn", "logits": name == "fc.1"}[region_name]


def mapping(component, name):
    if component == "hubert": return "hubert.model."+name
    if component == "synth": return name
    return rmvpe_names()[name]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=ROOT/"target/cpu-w8a8/broad")
    ap.add_argument("--recipes", nargs="+", choices=list(recipes()))
    ap.add_argument("--checkpoint", type=Path, default=ROOT/"models/voices/default-v2/default.pth")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    waveform = np.fromfile(ROOT/"target/compare/cpu-final2-trace/0000/waveform.bin", dtype="<f4")
    hubert = HubertModel.from_pretrained(str(ROOT/"models/shared/hubert_base"), local_files_only=True,
                                       attn_implementation="eager").to(device).float().eval()
    predictor = region.RMVPE0Predictor(str(ROOT/"models/shared/rmvpe/rmvpe.pt"), device=device)
    rmvpe = predictor.model.float().eval()
    synth, width, rate = ref.load_synth(args.checkpoint, device)
    for layer in synth.modules():
        if hasattr(layer, "weight_g"): nn.utils.remove_weight_norm(layer)
    nets = dict(hubert=hubert, rmvpe=rmvpe, synth=synth)
    rng = np.random.default_rng(24601)
    windows, manifest = [], []
    def run_hubert(item):
        return hubert(item["wave"]).last_hidden_state.float().cpu().numpy()[0]
    def run_rmvpe(item):
        salience = rmvpe(item["mel"])[:, :item["mel_frames"]].float().cpu().numpy()[0]
        return salience, predictor.decode(salience, thred=.03)
    for split, offsets in [("calibration", [32000,112000]), ("test", [208000,352000]), ("validation", [272000,416000])]:
        for samples in [11200,44960]:
            for offset in offsets:
                wave = waveform[offset:offset+samples]
                if wave.size != samples: raise ValueError("window exceeds source")
                item = dict(split=split,samples=samples,offset=offset,wave=torch.tensor(wave,device=device)[None])
                mel = predictor.mel_extractor(item["wave"], center=True)
                item.update(mel_frames=mel.shape[-1], mel=F.pad(mel,(0,(-mel.shape[-1])%32),mode="reflect"))
                hidden = run_hubert(item)
                salience, f0 = run_rmvpe(item)
                frames, skip = samples//160, 250 if samples == 44960 else 8
                snapshot = dict(synth_frames=frames,valid_synth_frames=frames,synth_skip_head=skip,
                    synth_output_frames=frames-skip,
                    rvc_features=np.repeat(np.concatenate([hidden,hidden[-1:]]),2,axis=0)[:frames],
                    pitch=f0mod.quantize_f0(f0[:frames]),aligned_pitchf=f0[:frames].astype(np.float32),
                    synth_random=rng.standard_normal((1,synth.enc_p.out_channels,frames),dtype=np.float32),
                    synth_noise=rng.standard_normal((1,1,frames*(rate//100)),dtype=np.float32))
                audio = ref.synth_reference(synth,snapshot,width,device)
                item.update(hidden=hidden,salience=salience,snapshot=snapshot,audio=audio)
                directory = args.output_dir/f"{split}-{samples}-{offset}"
                directory.mkdir(exist_ok=True)
                for name, data in [("wave",wave),("rnd",snapshot["synth_random"]),("noise",snapshot["synth_noise"]),
                                   ("reference",audio),("pitchf",snapshot["aligned_pitchf"])]:
                    np.asarray(data,dtype="<f4").tofile(directory/(name+".f32"))
                np.asarray(snapshot["pitch"],dtype="<i4").tofile(directory/"pitch.i32")
                manifest.append(dict(split=split,samples=samples,offset=offset,directory=str(directory.resolve())))
                if split != "calibration": wavfile.write(str(directory/"reference.wav"),rate,audio)
                windows.append(item)
    cal = [w for w in windows if w["split"] == "calibration"]
    tests = [w for w in windows if w["split"] != "calibration"]
    fold_batchnorm(rmvpe)
    invoke = dict(hubert=run_hubert,rmvpe=run_rmvpe,
                  synth=lambda w:ref.synth_reference(synth,w["snapshot"],width,device))
    stats = {c:calibrate(nets[c],cal,invoke[c]) for c in nets}
    report = dict(reference="original Applio FP32 Synth/RMVPE + HF HuBERT, TF32 disabled",
                  note="static fake W8A8 accuracy only, no CPU speed claim, index=0, identical noise; weights [-127,127], activation [-128,127] matches oneDNN S8 reorder",
                  checkpoint=str(args.checkpoint.resolve()),windows=manifest,sample_rate=rate,recipes=[])
    report["eligible"] = {c:dict(layers=len(stats[c]),
        weight_elements=sum(nets[c].get_submodule(name).weight.numel() for name in stats[c]),
        model_parameters=sum(p.numel() for p in nets[c].parameters())) for c in nets}
    (args.output_dir/"inputs.json").write_text(json.dumps(dict(windows=manifest,sample_rate=rate),indent=2)+"\n")
    plans = {}
    for label, policy in recipes().items():
        if args.recipes and label not in args.recipes: continue
        metrics, diagnostics, rules, coverage = [], [], [], {}
        with ExitStack() as stack:
            for component, regions in policy.items():
                net, wrappers = nets[component], []
                # Regions within a component are intentionally disjoint. Select
                # before replacement so names/axis mappings remain canonical.
                chosen = set()
                for area, alpha in regions:
                    names = [name for name in stats[component] if selected(component,name,net.get_submodule(name),area)]
                    if chosen.intersection(names): raise ValueError("overlapping recipe regions")
                    chosen.update(names)
                    for name in names: rules.append(dict(prefix=mapping(component,name),smooth_alpha=alpha))
                    wrappers.extend(stack.enter_context(quantized(net,stats[component],lambda n,l,names=set(names):n in names,alpha,activation_min=-128)))
                coverage[component] = dict(layers=len(wrappers),weight_elements=sum(w.weight.numel() for w in wrappers))
            for item in tests:
                hidden = run_hubert(item)
                salience, f0 = run_rmvpe(item)
                f0 = f0[:item["snapshot"]["synth_frames"]].astype(np.float32)
                pitch = f0mod.quantize_f0(f0)
                features = np.repeat(np.concatenate([hidden,hidden[-1:]]),2,axis=0)[:f0.size]
                audio = ref.synth_reference(synth,dict(item["snapshot"],rvc_features=features,pitch=pitch,aligned_pitchf=f0),width,device)
                metrics.append(ref.metric(item["audio"],audio))
                diagnostics.append(dict(split=item["split"],samples=item["samples"],offset=item["offset"],
                    coarse_mismatch=int(np.count_nonzero(pitch != item["snapshot"]["pitch"])),
                    voicing_mismatch=int(np.count_nonzero((f0>0)!=(item["snapshot"]["aligned_pitchf"]>0))),
                    hubert=ref.metric(item["hidden"],hidden),f0=ref.metric(item["snapshot"]["aligned_pitchf"],f0)))
                if item["samples"] == 44960 and item["split"] == "validation":
                    wavfile.write(str(Path(next(w["directory"] for w in manifest if w["split"]==item["split"] and w["samples"]==item["samples"] and w["offset"]==item["offset"]))/f"{label}-fake.wav"),rate,audio)
        row = dict(name=label,policy=policy,coverage=coverage,audio=region.aggregate(metrics),diagnostics=diagnostics)
        report["recipes"].append(row)
        plans[label] = dict(rules=rules,calibration={})
        (args.output_dir/"accuracy.json").write_text(json.dumps(report,indent=2)+"\n")
        (args.output_dir/"recipes.json").write_text(json.dumps(plans,indent=2)+"\n")
        print(json.dumps(dict(name=label,cosine_min=row["audio"]["cosine_min"],rmse=row["audio"]["rmse_mean"],coverage=coverage)),flush=True)


if __name__ == "__main__": main()
