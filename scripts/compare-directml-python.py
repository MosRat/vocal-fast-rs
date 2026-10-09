"""Compare handwritten DirectML graphs with original FP32 Python models.

Diagnostics download declared stage tensors; production timing uses audio only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref_repos" / "Applio"))
spec = importlib.util.spec_from_file_location("retrieval_reference", ROOT / "scripts/compare-retrieval-python.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["hubert", "rmvpe", "synth", "pipeline", "streaming", "retrieval"])
    ap.add_argument("native", type=Path)
    ap.add_argument("input", type=Path, help="waveform.f32 or exact Synth snapshot.json")
    ap.add_argument("report", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models/voices/default-v2/default.pth")
    ap.add_argument("--index", type=Path, default=ROOT / "models/voices/default-v2/added_IVF511_Flat_nprobe_1_default_v2.index")
    ap.add_argument("--rvc-version", choices=["v1", "v2"], help="Standalone HuBERT feature selection; pipeline derives it from checkpoint")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    native = json.loads(args.native.read_text())
    tensors = native["tensors"]
    result = {"reference": "original FP32 PyTorch models", "stage": args.stage, "metrics": {}}
    if args.stage == "retrieval":
        snapshot = json.loads(args.input.read_text())
        import faiss
        index = faiss.deserialize_index(np.frombuffer(args.index.read_bytes(), dtype=np.uint8))
        query = np.asarray(snapshot["hubert_features"], np.float32).reshape(-1, index.d)
        result["metrics"]["features"] = ref.metric(ref.retrieval_reference(query, args.index, 0.5), tensors["features"])
    elif args.stage == "synth":
        snapshot = json.loads(args.input.read_text())
        net, width, _ = ref.load_synth(args.checkpoint, "cuda")
        audio = ref.synth_reference(net, snapshot, width, "cuda")
        result["metrics"]["audio"] = ref.metric(audio, tensors["audio"])
    else:
        wave = np.fromfile(args.input, dtype="<f4")
        windows = []
        if args.stage == "streaming":
            advance = native["stream"]["advance_samples"]
            window = np.zeros(native["config"]["window_samples"], np.float32)
            for i in range(native["stream"]["chunks"]):
                window[:-advance] = window[advance:].copy()
                window[-advance:] = 0
                chunk = wave[i*advance:(i+1)*advance]
                window[-advance:-advance+len(chunk) if len(chunk) < advance else None] = chunk
                windows.append(window.copy())
            wave = windows[-1]
            args.stage = "pipeline"
        if args.stage in ["hubert", "pipeline"]:
            from transformers import HubertModel
            hubert = HubertModel.from_pretrained(str(ROOT / "models/shared/hubert_base"), local_files_only=True)
            hubert = hubert.to("cuda").float().eval()
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True) if args.stage == "pipeline" else None
            version = args.rvc_version or ("v1" if checkpoint is not None and checkpoint["weight"]["enc_p.emb_phone.weight"].shape[1] == 256 else "v2")
            with torch.inference_mode():
                outputs = hubert(torch.from_numpy(wave[None]).to("cuda"), output_hidden_states=version == "v1")
                features = outputs.last_hidden_state
                if version == "v1":
                    weights = torch.load(ROOT / "models/shared/hubert_base/pytorch_model.bin", map_location="cuda", weights_only=True)
                    features = torch.nn.functional.linear(outputs.hidden_states[9], weights["final_proj.weight"].float(), weights["final_proj.bias"].float())
                h = features.cpu().numpy()
            result["rvc_version"] = version
            key = "features" if args.stage == "hubert" else "hubert_raw"
            if key in tensors:
                result["metrics"]["hubert"] = ref.metric(h, tensors[key])
            del hubert
        if args.stage in ["rmvpe", "pipeline"]:
            from rvc.lib.predictors.RMVPE import RMVPE0Predictor
            rmvpe = RMVPE0Predictor(str(ROOT / "models/shared/rmvpe/rmvpe.pt"), device="cuda")
            captured = {}
            handles = []
            for name, module in [("rmvpe_unet", rmvpe.model.unet), ("rmvpe_gru", rmvpe.model.fc[0]),
                                 ("rmvpe_logits", rmvpe.model.fc[1]), ("rmvpe_bottleneck", rmvpe.model.unet.intermediate),
                                 *[(f"rmvpe_encoder_{i}", m.conv[-1]) for i, m in enumerate(rmvpe.model.unet.encoder.layers)],
                                 *[(f"rmvpe_decoder_{i}", m) for i, m in enumerate(rmvpe.model.unet.decoder.layers)]]:
                handles.append(module.register_forward_hook(lambda m, i, out, name=name: captured.update({name: out.detach().cpu().numpy()})))
            policy = native.get("config", {}).get("f0", {"mode": "full_window"})
            history = np.zeros(1024, np.float32)
            with torch.inference_mode():
                for acoustic_window in windows or [wave]:
                    if policy["mode"] == "rolling":
                        acoustic_window = acoustic_window[-policy["rmvpe_samples"]:]
                    mel = rmvpe.mel_extractor(torch.from_numpy(acoustic_window[None]).to("cuda"), center=True)
                    hidden = rmvpe.mel2hidden(mel)
                    f0 = rmvpe.decode(hidden[0].cpu().numpy(), thred=0.03)
                    if policy["mode"] == "rolling":
                        shift = policy["advance_samples"] // 160
                        history[:-shift] = history[shift:].copy()
                        history[-len(f0[3:-1]):] = f0[3:-1]
            if policy["mode"] == "rolling":
                result["metrics"]["f0_cache"] = ref.metric(history, tensors["f0_cache"])
                cached_f0 = history[-len(wave)//160:]
            for handle in handles:
                handle.remove()
            for name, value in captured.items():
                if name in tensors:
                    result["metrics"][name] = ref.metric(value, tensors[name])
            if "rmvpe_logits" in tensors:
                logits = np.asarray(tensors["rmvpe_logits"], np.float32).reshape(captured["rmvpe_logits"].shape)
                native_hidden = torch.sigmoid(torch.from_numpy(logits))[0, :len(f0)].numpy()
                result["rmvpe_argmax_mismatch"] = np.flatnonzero(np.argmax(hidden[0].cpu().numpy(), -1) != np.argmax(native_hidden, -1)).tolist()
            if "rmvpe_mel" in tensors:
                result["metrics"]["mel"] = ref.metric(mel.cpu().numpy(), tensors["rmvpe_mel"])
            key = "pitchf" if args.stage == "rmvpe" else "rmvpe_pitchf"
            observed_frames = len(tensors[key])
            if observed_frames > len(f0):
                raise ValueError("native F0 is longer than the reference extractor output")
            # TRT diagnostic I/O exposes the valid frontend prefix; CPU/DML
            # also retain the RMVPE boundary frame. Compare the declared prefix.
            result["metrics"]["pitchf"] = ref.metric(f0[:observed_frames], tensors[key])
        if args.stage == "pipeline":
            config = native["config"]
            skip = config["skip_head_frames"]
            net, width, _ = ref.load_synth(args.checkpoint, "cuda")
            frames = len(tensors["feats"]) // width
            controls = native.get("controls") or {}
            ratio = controls.get("index_ratio", controls.get("indexRatio", 0))
            content = ref.retrieval_reference(h[0], args.index, ratio).reshape(h[0].shape) if ratio else h[0]
            if "hubert_features" in tensors:
                result["metrics"]["hubert_after_retrieval"] = ref.metric(content, tensors["hubert_features"])
            augmented = np.concatenate([content, content[-1:]]) if config.get("append_hubert_frame", True) else content
            features = np.repeat(augmented, 2, axis=0)[:frames]
            pitchf = (cached_f0 if policy["mode"] == "rolling" else f0[:frames]).astype(np.float32)
            pitchf *= 2 ** (controls.get("pitch_semitones", controls.get("pitchSemitones", 0)) / 12)
            if ratio and controls.get("protect", 0.33) < 0.5:
                raw = np.repeat(np.concatenate([h[0], h[0, -1:]]), 2, axis=0)[:frames]
                amount = np.where(pitchf > 0, 1, controls.get("protect", 0.33)).astype(np.float32)[:, None]
                features = features * amount + raw * (1 - amount)
            mel = 1127 * np.log1p(pitchf / 700)
            minimum, maximum = 1127 * np.log1p(np.array([50, 1100]) / 700)
            pitch = np.rint(np.clip((mel - minimum) * 254 / (maximum - minimum) + 1, 1, 255)).astype(int)
            result["metrics"]["frontend_features"] = ref.metric(features, tensors["feats"])
            result["coarse_pitch_mismatches"] = int(np.count_nonzero(pitch.reshape(-1) != native["pitch"]))
            snapshot = {"synth_frames": frames, "synth_skip_head": skip,
                        "synth_output_frames": frames-skip, "valid_synth_frames": frames,
                        "synth_random": tensors["synth_random"], "synth_noise": tensors["synth_noise"],
                        "rvc_features": features.reshape(-1).tolist(), "pitch": pitch.tolist(),
                        "aligned_pitchf": pitchf.tolist()}
            audio = ref.synth_reference(net, snapshot, width, "cuda")
            result["metrics"]["audio"] = ref.metric(audio, tensors["audio"])
            snapshot["rvc_features"] = tensors["feats"]
            snapshot["pitch"] = native["pitch"]
            snapshot["aligned_pitchf"] = tensors["pitchf"]
            result["metrics"]["synth_only"] = ref.metric(ref.synth_reference(net, snapshot, width, "cuda"), tensors["audio"])
    result["gpu_ms"] = native["gpu_ms"]
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
