"""Isolate HuBERT static W8A8 and online attention against Applio FP32 audio.

Existing disjoint calibration/test/validation fixtures keep RMVPE, Synth and
noise unchanged. Fake quantization is an accuracy experiment, not a speed test.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from scipy.io import wavfile
from transformers import HubertModel
from cpu_quantization import calibrate, quantized
from hubert_attention import online_hubert, online_attention
from hubert_quantization import outlier_quantized

ROOT = Path(__file__).resolve().parents[1]


def load_reference():
    spec = importlib.util.spec_from_file_location("retrieval_reference", ROOT/"scripts/compare-retrieval-python.py")
    ref = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ref
    spec.loader.exec_module(ref)
    return ref


def selectors():
    def ffn(name): return "feed_forward." in name
    def number(name): return int(name.split(".")[2]) if name.startswith("encoder.layers.") else -1
    result = {
        "qk": lambda n, l: ".attention." in n and n.endswith(("q_proj", "k_proj")),
        "vout": lambda n, l: ".attention." in n and n.endswith(("v_proj", "out_proj")),
        "attention": lambda n, l: ".attention." in n,
        "ffn_in": lambda n, l: ffn(n) and n.endswith("intermediate_dense"),
        "ffn_out": lambda n, l: ffn(n) and n.endswith("output_dense"),
        "early_ffn": lambda n, l: ffn(n) and number(n) < 4,
        "late_ffn": lambda n, l: ffn(n) and number(n) >= 8,
        "ffn": lambda n, l: ffn(n),
        "projection": lambda n, l: n == "feature_projection.projection",
        "linear": lambda n, l: isinstance(l, nn.Linear),
        "extractor": lambda n, l: n.startswith("feature_extractor."),
        "extractor_hot": lambda n, l: n in {
            "feature_extractor.conv_layers.1.conv", "feature_extractor.conv_layers.2.conv"},
        "extractor_late": lambda n, l: n.startswith("feature_extractor.conv_layers.")
            and int(n.split(".")[2]) > 0,
    }
    for i in range(12):
        result[f"block_{i:02}"] = lambda n, l, i=i: n.startswith(f"encoder.layers.{i}.")
    return result


def metric(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape or not a.size or not np.isfinite(b).all():
        raise ValueError("invalid comparison")
    d = b-a
    return dict(cosine=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30)),
                rmse=float(np.sqrt(np.mean(d*d))),max_abs=float(np.abs(d).max()),
                reference_rms=float(np.sqrt(np.mean(a*a))), actual_rms=float(np.sqrt(np.mean(b*b))))


def aggregate(rows, name):
    values = [r[name] for r in rows]
    speech = [r[name] for r in rows if r["audio"]["reference_rms"] >= .01]
    return dict(cosine_min=min(m["cosine"] for m in values),
                speech_cosine_min=min((m["cosine"] for m in speech),default=None),
                rmse_mean=float(np.mean([m["rmse"] for m in values])),max_abs=max(m["max_abs"] for m in values))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=Path, default=ROOT/"target/cpu-w8a8/broad/inputs.json")
    ap.add_argument("--output-dir", type=Path, default=ROOT/"target/hubert-cpu")
    ap.add_argument("--regions", nargs="+", choices=list(selectors()), default=list(selectors())[:11])
    ap.add_argument("--alpha", type=float, nargs="+", default=[-1,.5,.7])
    ap.add_argument("--checkpoint", type=Path, default=ROOT/"models/voices/default-v2/default.pth")
    ap.add_argument("--attention-only", action="store_true")
    ap.add_argument("--extend", action="store_true", help="append to an existing accuracy report")
    ap.add_argument("--rescue", action="store_true", help="screen mixed alpha and research-only outlier residuals")
    args = ap.parse_args()
    if any(a != -1 and not 0 <= a <= 1 for a in args.alpha): ap.error("invalid SmoothQuant alpha")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    ref = load_reference()
    hubert = HubertModel.from_pretrained(str(ROOT/"models/shared/hubert_base"), local_files_only=True,
                                       attn_implementation="eager").to(device).float().eval()
    synth, width, rate = ref.load_synth(args.checkpoint, device)
    for layer in synth.modules():
        if hasattr(layer,"weight_g"): nn.utils.remove_weight_norm(layer)
    windows = []
    manifest = json.loads(args.fixtures.read_text())
    for meta in manifest["windows"]:
        p = Path(meta["directory"])
        frames = meta["samples"]//160
        snapshot = dict(synth_frames=frames, valid_synth_frames=frames,
                        synth_skip_head=250 if meta["samples"]==44960 else 8,
                        synth_output_frames=frames-(250 if meta["samples"]==44960 else 8),
                        pitch=np.fromfile(p/"pitch.i32",dtype="<i4"),
                        aligned_pitchf=np.fromfile(p/"pitchf.f32",dtype="<f4"),
                        synth_random=np.fromfile(p/"rnd.f32",dtype="<f4").reshape(1,synth.enc_p.out_channels,frames),
                        synth_noise=np.fromfile(p/"noise.f32",dtype="<f4").reshape(1,1,-1))
        item = dict(meta,wave=torch.tensor(np.fromfile(p/"wave.f32",dtype="<f4"),device=device)[None],snapshot=snapshot,
                    audio=np.fromfile(p/"reference.f32",dtype="<f4"))
        item["hidden"] = hubert(item["wave"]).last_hidden_state.cpu().numpy()[0]
        windows.append(item)
    calibration = [w for w in windows if w["split"]=="calibration"]
    tests = [w for w in windows if w["split"]!="calibration"]
    invoke = lambda w: hubert(w["wave"]).last_hidden_state.cpu().numpy()[0]
    def evaluate(label):
        rows=[]
        for item in tests:
            hidden=invoke(item)
            frames=item["snapshot"]["synth_frames"]
            features=np.repeat(np.concatenate([hidden,hidden[-1:]]),2,axis=0)[:frames]
            audio=ref.synth_reference(synth,dict(item["snapshot"],rvc_features=features),width,device)
            rows.append(dict(split=item["split"],samples=item["samples"],offset=item["offset"],
                             hidden=metric(item["hidden"],hidden),audio=metric(item["audio"],audio)))
            if item["split"]=="validation" and item["samples"]==44960:
                wavfile.write(str(args.output_dir/f"{label}-{item['offset']}.wav"),rate,audio)
        return dict(hidden=aggregate(rows,"hidden"),audio=aggregate(rows,"audio"),windows=rows)
    report=dict(reference="HF HuBERT eager FP32 + original Applio FP32 Synth; fixed Applio F0/noise, index=0, TF32 disabled",
                torch=torch.__version__,checkpoint=str(args.checkpoint.resolve()),fixtures=str(args.fixtures.resolve()),
                note="static per-layer activation S8 [-128,127], per-output weight S8 [-127,127]; Python dots remain FP32; no speed claim",
                quantization=[],attention=[])
    if args.extend:
        previous=json.loads((args.output_dir/"accuracy.json").read_text())
        for key in ("reference","torch","checkpoint","fixtures","note"):
            if previous[key]!=report[key]: raise ValueError("extend requires the same experiment contract")
        report=previous
    def save(): (args.output_dir/"accuracy.json").write_text(json.dumps(report,indent=2)+"\n")
    report["baseline"]=evaluate("fp32")
    if not args.attention_only:
        stats=calibrate(hubert,calibration,invoke)
        distributions={}
        for name,st in stats.items():
            values=torch.cat(st.samples,dim=1).flatten()
            p999=float(torch.quantile(values,.999))
            distributions[name]=dict(max=float(st.channel_max.max()),p999=p999,
                max_p999_ratio=float(st.channel_max.max())/max(p999,1e-8),channels=st.channel_max.numel(),calls=st.calls)
        report["distributions"]=distributions
        plans=json.loads((args.output_dir/"recipes.json").read_text()) if args.extend else {}
        for name in args.regions:
            for alpha in args.alpha:
                label=f"{name}-alpha{alpha}"
                if any(r["name"]==name and r["alpha"]==alpha for r in report["quantization"]): continue
                with quantized(hubert,stats,selectors()[name],None if alpha==-1 else alpha,activation_min=-128) as wrappers:
                    result=evaluate(label)
                    result.update(name=name,alpha=alpha,layers=len(wrappers),weight_elements=sum(w.weight.numel() for w in wrappers),
                                  clipped_fraction=sum(w.clipped for w in wrappers)/max(1,sum(w.elements for w in wrappers)))
                plans[label]=dict(rules=[dict(prefix="hubert.model."+n,smooth_alpha=None if alpha==-1 else alpha)
                                        for n in stats if selectors()[name](n,hubert.get_submodule(n))],calibration={})
                report["quantization"].append(result)
                (args.output_dir/"recipes.json").write_text(json.dumps(plans,indent=2)+"\n")
                save()
                print(json.dumps(dict(name=label,audio=result["audio"],hidden=result["hidden"],clipped=result["clipped_fraction"])),flush=True)
        if args.rescue:
            for fraction in [0,.01,.02,.04]:
                label=f"mixed-outliers-{fraction}"
                rules=[]
                with ExitStack() as stack:
                    wrappers=[]
                    for area,alpha in [("qk",None),("vout",.5),("projection",.5),("ffn_in",.7)]:
                        select=selectors()[area]
                        wrappers+=stack.enter_context(quantized(hubert,stats,select,alpha,activation_min=-128))
                        rules.extend(dict(prefix="hubert.model."+n,smooth_alpha=alpha)
                                     for n in stats if select(n,hubert.get_submodule(n)))
                    if fraction:
                        wrappers+=stack.enter_context(outlier_quantized(hubert,stats,selectors()["ffn_out"],.5,fraction))
                    result=evaluate(label)
                    result.update(name=label,layers=len(wrappers),weight_elements=sum(w.weight.numel() for w in wrappers),
                                  outlier_fraction=fraction,native_supported=fraction==0,
                                  clipped_fraction=sum(w.clipped for w in wrappers)/max(1,sum(w.elements for w in wrappers)))
                report.setdefault("rescue",[]).append(result)
                if not fraction:
                    plans["mixed"]=dict(rules=rules,calibration={})
                    (args.output_dir/"recipes.json").write_text(json.dumps(plans,indent=2)+"\n")
                save()
                print(json.dumps(dict(rescue=label,audio=result["audio"],hidden=result["hidden"],native=result["native_supported"])),flush=True)
    # Real projections from every layer, with the same token-major storage as
    # the Rust graph views. Native benchmarks reconstruct these strides.
    cases=[]
    for samples in [11200,44960]:
        item=next(w for w in tests if w["samples"]==samples)
        handles=[]
        def capture(module, arguments, layer):
            x=arguments[0]; b,t,_=x.shape
            q,k,v=[proj(x).view(b,t,module.num_heads,module.head_dim).transpose(1,2)
                   for proj in (module.q_proj,module.k_proj,module.v_proj)]
            reference=torch.softmax((q@k.transpose(-2,-1))*module.scaling,-1)@v
            output=online_attention(q,k,v,module.scaling,4,32)
            stem=f"attention-{samples}-{layer:02}"
            for suffix,data in [("q",q),("k",k),("v",v),("reference",reference)]:
                # Serialize canonical BHTD; the benchmark recreates BTHD views.
                data.contiguous().cpu().numpy().astype("<f4").tofile(args.output_dir/f"{stem}-{suffix}.f32")
            cases.append(dict(stem=stem,samples=samples,layer=layer,shape=list(q.shape),scale=module.scaling,
                              online=metric(reference.cpu().numpy(),output.cpu().numpy())))
        for i,block in enumerate(hubert.encoder.layers):
            handles.append(block.attention.register_forward_pre_hook(lambda m,a,i=i:capture(m,a,i)))
        try: invoke(item)
        finally:
            for handle in handles: handle.remove()
    (args.output_dir/"attention-cases.json").write_text(json.dumps(dict(cases=cases),indent=2)+"\n")
    for qt,kt in [(4,32),(16,64)]:
        if args.extend and any(r["query_tile"]==qt and r["key_tile"]==kt for r in report["attention"]): continue
        with online_hubert(hubert,qt,kt): result=evaluate(f"online-{qt}-{kt}")
        report["attention"].append(dict(query_tile=qt,key_tile=kt,**result));save()
        print(json.dumps(dict(online=[qt,kt],audio=result["audio"],hidden=result["hidden"])),flush=True)
    save()


if __name__=="__main__": main()
