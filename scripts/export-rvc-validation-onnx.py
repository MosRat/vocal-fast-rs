"""Export a real v1/v2 checkpoint using voice-changer's Synth ONNX classes.

Offline validation only. Runtime engines consume canonical VFRM tensors.
"""
from pathlib import Path
import argparse
import json
import sys
import torch
import onnx
from onnxsim import simplify

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref_repos/voice-changer/server"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(6)
    cpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    width = int(cpt["weight"]["enc_p.emb_phone.weight"].shape[1])
    if width not in (256, 768) or not cpt.get("f0", 1):
        ap.error("validation export supports standard F0 RVC v1/v2 checkpoints")
    if width == 256:
        from voice_changer.RVC.onnxExporter.SynthesizerTrnMs256NSFsid_ONNX import SynthesizerTrnMs256NSFsid_ONNX as Synth
    else:
        from voice_changer.RVC.onnxExporter.SynthesizerTrnMs768NSFsid_ONNX import SynthesizerTrnMs768NSFsid_ONNX as Synth
    config = list(cpt["config"])
    config[-3] = int(cpt["weight"]["emb_g.weight"].shape[0])
    net = Synth(*config, is_half=False).eval().float()
    del net.enc_q
    net.load_state_dict({k: v for k, v in cpt["weight"].items() if not k.startswith("enc_q.")}, strict=True)
    frames = 8
    inputs = (torch.zeros(1, frames, width), torch.tensor([frames]),
              torch.ones(1, frames, dtype=torch.int64), torch.ones(1, frames) * 200, torch.tensor([0]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(net, inputs, str(args.output), dynamo=False, opset_version=17,
                          input_names=["feats", "p_len", "pitch", "pitchf", "sid"], output_names=["audio"],
                          dynamic_axes={"feats": {1: "frames"}, "pitch": {1: "frames"}, "pitchf": {1: "frames"}, "audio": {0: "samples"}},
                          do_constant_folding=False)
    model, valid = simplify(onnx.load(args.output))
    if not valid:
        raise ValueError("voice-changer ONNX simplification did not validate")
    meta = model.metadata_props.add()
    meta.key = "metadata"
    meta.value = json.dumps({"application": "VC_CLIENT", "version": "2.1", "samplingRate": int(config[-1]),
                            "f0": True, "embChannels": width, "embOutputLayer": 9 if width == 256 else 12,
                            "useFinalProj": width == 256, "embedder": "hubert_base"})
    onnx.save(model, args.output)
    print(json.dumps({"output": str(args.output), "width": width, "nodes": len(model.graph.node)}))


if __name__ == "__main__":
    main()
