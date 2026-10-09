"""Run the local Applio offline pipeline using original PTH/RMVPE weights.

This includes Applio's high-pass filter, reflected padding, FAISS search and
offline crop. Optional native noise sharing isolates end-to-end graph error
from stochastic sampling; native traces never supply acoustic features/F0.
For v1 the default adapter selects HuBERT layer 9, as RVC WebUI/voice-changer
do. --v1-hubert-layer 12 reproduces this Applio checkout's last-layer behavior.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import types
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]


def unavailable(*args, **kwargs):
    raise RuntimeError("this reference runner supports only local RMVPE weights")


def import_reference():
    # Applio eagerly imports optional pitch/download backends. Keep the
    # original RMVPE path usable without installing those unused packages;
    # invoking any stub fails rather than silently supplying fake inference.
    optional = {
        "torchcrepe": {"predict": unavailable},
        "swift_f0": {"FRAME_PERIOD": 0.016, "SwiftF0": unavailable},
        "torchfcpe": {"spawn_infer_model_from_pt": unavailable},
        "wget": {"download": unavailable},
    }
    for name, attributes in optional.items():
        if importlib.util.find_spec(name) is None:
            module = types.ModuleType(name)
            module.__dict__.update(attributes)
            sys.modules[name] = module
    sys.path.insert(0, str(ROOT / "ref_repos" / "Applio"))
    from rvc.infer import pipeline
    from rvc.lib.algorithm.synthesizers import Synthesizer
    from rvc.lib.predictors.RMVPE import RMVPE0Predictor
    from rvc.lib.utils import HubertModelWithFinalProj
    return pipeline, Synthesizer, RMVPE0Predictor, HubertModelWithFinalProj


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--index", type=Path)
    ap.add_argument("--index-ratio", type=float, default=0.5)
    ap.add_argument("--pitch", type=float, default=0.0, help="F0 shift in semitones; does not change duration")
    ap.add_argument("--proposed-pitch-target", type=float,
                    help="enable Applio's original Proposed Pitch with a target in Hz; adds to --pitch")
    ap.add_argument("--protect", type=float, default=0.5,
                    help="unvoiced feature blend; 0.5 disables protection")
    ap.add_argument("--volume-envelope", type=float, default=1.0,
                    help="1 keeps Synth loudness; 0 follows the input envelope")
    ap.add_argument("--hubert", type=Path, default=ROOT / "models/shared/hubert_base")
    ap.add_argument("--rmvpe", type=Path, default=ROOT / "models/shared/rmvpe/rmvpe.pt")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--v1-hubert-layer", type=int, choices=[9, 12], default=9,
                    help="9: standard RVC v1; 12: this Applio checkout's last-layer v1 behavior")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--trace-dir", type=Path, help="save exact reference stage inputs/outputs")
    ap.add_argument("--native-noise", type=Path,
                    help="single-segment native trace directory; share only latent/NSF noise")
    args = ap.parse_args()
    if not 0 <= args.index_ratio <= 1:
        ap.error("--index-ratio must be in [0,1]")
    if args.index_ratio > 0 and (args.index is None or not args.index.is_file()):
        ap.error("nonzero --index-ratio requires an existing --index file")
    if not math.isfinite(args.pitch) or not -96 <= args.pitch <= 96:
        ap.error("--pitch must be finite and in [-96,96]")
    if args.proposed_pitch_target is not None and (
            not math.isfinite(args.proposed_pitch_target) or not 50 <= args.proposed_pitch_target <= 1200):
        ap.error("--proposed-pitch-target must be finite and in [50,1200] Hz")
    if not 0 <= args.protect <= 0.5:
        ap.error("--protect must be in [0,0.5]")
    if not 0 <= args.volume_envelope <= 1:
        ap.error("--volume-envelope must be in [0,1]")
    sr, audio = wavfile.read(args.input)
    if sr != 16000:
        ap.error("reference input must be a 16 kHz WAV")
    if audio.dtype.kind == "i":
        audio = audio.astype(np.float32) / -float(np.iinfo(audio.dtype).min)
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if not audio.size or not np.isfinite(audio).all():
        ap.error("input must contain finite audio samples")
    if (args.trace_dir or args.native_noise) and audio.size + 160 > 41 * 16000:
        ap.error("stage tracing/noise sharing currently requires a single Applio segment")
    peak = float(np.abs(audio).max())
    if peak > 0.95:
        audio *= 0.95 / peak
    pipeline, Synthesizer, RMVPE0Predictor, Hubert = import_reference()
    rmvpe_path = args.rmvpe.resolve()
    f0_observations = {}

    class LocalRMVPE(pipeline.RMVPE):
        def __init__(self, device, **kwargs):
            self.device, self.sample_rate, self.hop_size = device, 16000, 160
            self.high_register = {"enabled": False, "mode": "true_pitch", "f0_ceil": 1250.0}
            self.model = RMVPE0Predictor(str(rmvpe_path), device=device)
            if args.trace_dir:
                args.trace_dir.mkdir(parents=True, exist_ok=True)

                def stage_hook(name):
                    def capture(module, inputs, output):
                        output.detach().cpu().numpy().astype("<f4").tofile(args.trace_dir / f"{name}.bin")
                    return capture

                for name, module in [("rmvpe_mel", self.model.mel_extractor),
                                     ("rmvpe_bottleneck", self.model.model.unet.intermediate),
                                     ("rmvpe_unet", self.model.model.unet),
                                     ("rmvpe_gru", self.model.model.fc[0]),
                                     ("rmvpe_logits", self.model.model.fc[1])]:
                    module.register_forward_hook(stage_hook(name))

        def get_f0(self, *values, **kwargs):
            result = super().get_f0(*values, **kwargs)
            # Observe the original extractor before Pipeline.get_f0 adjusts
            # it in place. Never supply or alter the reference's F0 decisions.
            f0_observations["raw"] = result.copy()
            if args.trace_dir:
                result.astype("<f4").tofile(args.trace_dir / "rmvpe_pitchf.bin")
            return result

    pipeline.RMVPE = LocalRMVPE
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    width = int(cpt["weight"]["enc_p.emb_phone.weight"].shape[1])
    if width not in (256, 768):
        raise ValueError(f"unsupported RVC feature width {width}")
    version = "v1" if width == 256 else "v2"
    config = list(cpt["config"])
    config[-3] = int(cpt["weight"]["emb_g.weight"].shape[0])
    net = Synthesizer(*config, use_f0=bool(cpt.get("f0", 1)),
                      text_enc_hidden_dim=width,
                      vocoder=cpt.get("vocoder", "HiFi-GAN"))
    del net.enc_q
    net.load_state_dict({k: v for k, v in cpt["weight"].items() if not k.startswith("enc_q.")}, strict=True)
    net = net.to(args.device).float().eval()
    class VersionedHubert(Hubert):
        def forward(self, *values, **kwargs):
            select_v1 = version == "v1" and args.v1_hubert_layer == 9
            if select_v1:
                kwargs["output_hidden_states"] = True
            outputs = super().forward(*values, **kwargs)
            if select_v1:
                outputs["last_hidden_state"] = outputs.hidden_states[9]
            return outputs

    hubert = VersionedHubert.from_pretrained(str(args.hubert), local_files_only=True).to(args.device).float().eval()
    cfg = types.SimpleNamespace(device=args.device, x_pad=1, x_query=6, x_center=38, x_max=41)
    vc = pipeline.Pipeline(int(config[-1]), cfg)
    original_get_f0 = vc.get_f0

    def observed_get_f0(*values, **kwargs):
        coarse, shifted = original_get_f0(*values, **kwargs)
        raw = f0_observations.pop("raw")
        valid = np.flatnonzero(raw > 0)
        median = float(np.median(np.interp(np.arange(len(raw)), valid, raw[valid]))) if valid.size else None
        effective_shift = float(12 * np.log2(np.median(shifted[valid] / raw[valid]))) if valid.size else None
        f0_observations["diagnostics"] = {
            "raw_interpolated_median_hz": median,
            "effective_pitch_semitones": effective_shift,
            "proposed_offset_semitones": effective_shift - args.pitch if effective_shift is not None else None,
        }
        return coarse, shifted

    vc.get_f0 = observed_get_f0
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)

        def save(name, value):
            value = value.detach().cpu().numpy()
            value.astype("<i4" if value.dtype.kind in "iu" else "<f4").tofile(args.trace_dir / f"{name}.bin")

        def hubert_hook(module, inputs, output):
            save("waveform", inputs[0])
            features = output["last_hidden_state"]
            if version == "v1":
                features = module.final_proj(features)
            save("hubert_raw", features)

        def encoder_hook(module, inputs):
            save("feats", inputs[0])
            save("pitch", inputs[1])

        def decoder_hook(module, inputs, output):
            save("pitchf", inputs[1])
            save("audio", output)

        hubert.register_forward_hook(hubert_hook)
        net.enc_p.register_forward_pre_hook(encoder_hook)
        net.dec.register_forward_hook(decoder_hook)
    native_noise = None
    if args.native_noise:
        meta = json.loads((args.native_noise / "metadata.json").read_text(encoding="utf-8"))
        if not meta.get("offline") or meta.get("skip_head") != 0:
            ap.error("--native-noise requires a full-context offline trace")
        frames = meta["frames"]
        latent = np.fromfile(args.native_noise / "synth_random.bin", dtype="<f4").reshape(1, meta["latent_channels"], frames)
        noise = np.fromfile(args.native_noise / "synth_noise.bin", dtype="<f4").reshape(1, -1, 1)
        consumed = {"latent": 0, "nsf": 0}

        def exact_noise(value, *unused, **kwargs):
            if tuple(value.shape) == latent.shape:
                consumed["latent"] += 1
                source = latent
            elif tuple(value.shape) == noise.shape:
                consumed["nsf"] += 1
                source = noise
            else:
                raise ValueError(f"unexpected random shape {tuple(value.shape)}")
            return torch.as_tensor(source, device=value.device, dtype=value.dtype)

        native_noise = {"path": str(args.native_noise), "shared": "latent and NSF random tensors only"}
    started = time.perf_counter()
    with ExitStack() as stack, torch.inference_mode():
        if args.index_ratio > 0:
            # Adapt only Windows filename handling. Search/reconstruction and
            # blending still run through the original Applio FAISS pipeline.
            index = pipeline.faiss.deserialize_index(
                np.frombuffer(args.index.read_bytes(), dtype=np.uint8))
            if index.d != width:
                raise ValueError(f"retrieval width {index.d} disagrees with RVC {version}/{width}")
            stack.enter_context(patch.object(pipeline.faiss, "read_index", return_value=index))
        if args.native_noise:
            stack.enter_context(patch.object(torch, "randn_like", side_effect=exact_noise))
        result = vc.pipeline(
            model=hubert, net_g=net, sid=0, audio=audio,
            pitch=args.pitch, f0_method="rmvpe", file_index=str(args.index.resolve()) if args.index else "",
            index_rate=args.index_ratio, pitch_guidance=True,
            volume_envelope=args.volume_envelope,
            version=version, protect=args.protect,
            f0_autotune=False, f0_autotune_strength=1.0,
            proposed_pitch=args.proposed_pitch_target is not None,
            proposed_pitch_threshold=args.proposed_pitch_target if args.proposed_pitch_target is not None else 155.0,
        )
    if args.native_noise and consumed != {"latent": 1, "nsf": 1}:
        raise ValueError(f"single-segment noise was consumed incorrectly: {consumed}")
    result = np.asarray(result, np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Applio produced non-finite audio")
    report = {"reference": "Applio original offline pipeline, FP32",
              "rvc_version": version,
              "hubert_output_layer": args.v1_hubert_layer if version == "v1" else 12,
              "reference_adaptations": (["standard RVC v1 HuBERT layer 9"]
                                        if version == "v1" and args.v1_hubert_layer == 9 else []),
              "input": str(args.input), "checkpoint": str(args.checkpoint), "native_noise": native_noise,
              "index": str(args.index) if args.index else None, "index_ratio": args.index_ratio,
              "controls": {"protect": args.protect,
                           "volume_envelope": args.volume_envelope,
                           "pitch_semitones": args.pitch, "speaker_id": 0,
                           "f0_method": "rmvpe", "f0_threshold": 0.03,
                           "autotune": False, "proposed_pitch": args.proposed_pitch_target is not None,
                           "proposed_pitch_target_hz": args.proposed_pitch_target,
                           "f0_diagnostics": f0_observations.get("diagnostics"),
                           "high_register": False, "seed": 1234},
              "dsp": {"input_rate": sr, "output_rate": int(config[-1]),
                      "highpass_hz": pipeline.CUTOFF_FREQUENCY,
                      "highpass_order": pipeline.FILTER_ORDER,
                      "highpass_zero_phase": True, "padding": "reflect",
                      "padding_seconds_per_side": cfg.x_pad,
                      "max_segment_seconds": cfg.x_max,
                      "rolling_f0": False, "streaming_sola": False,
                      "input_rms_multiplier": False},
              "output": str(args.output), "sample_rate": int(config[-1]),
              "input_samples": int(audio.size), "output_samples": int(result.size),
              "elapsed_ms": (time.perf_counter() - started) * 1000,
              "rms": float(np.sqrt(np.mean(result ** 2))), "peak": float(np.max(np.abs(result)))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(args.output), int(config[-1]), np.clip(result, -1, 1))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
