"""Compare the complete native TensorRT RVC path with a Python reference.

The native debug snapshot contains the result of HuBERT -> RMVPE -> frontend
-> Synth.  Earlier parity reports compared only the Synth stage while feeding
the same native intermediate tensors to both implementations.  This script
uses the independent ONNX HuBERT/RMVPE outputs, applies the RVC append/repeat
and F0 crop rules, runs the FP32 PyTorch Synthesizer, and compares the final
waveform that a user would hear.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "compare_native_python", ROOT / "scripts/compare-native-python.py"
)
_ref = importlib.util.module_from_spec(_spec)
assert _spec.loader
_spec.loader.exec_module(_ref)
build_reference, synthesize, read_vfrm = (
    _ref.build_reference,
    _ref.synthesize,
    _ref.read_vfrm,
)


def stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    """Return audio parity metrics with stable float64 accumulation."""
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"audio shape mismatch: reference={reference.shape}, native={candidate.shape}"
        )
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("reference or native audio contains a non-finite value")
    delta = candidate - reference
    ref_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    cosine = float(
        np.dot(reference, candidate)
        / max(ref_norm * candidate_norm, np.finfo(np.float64).tiny)
    )
    return {
        "samples": int(reference.size),
        "reference_peak": float(np.max(np.abs(reference))),
        "native_peak": float(np.max(np.abs(candidate))),
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "relative_l2": float(np.linalg.norm(delta) / max(ref_norm, 1.0e-12)),
        "cosine_similarity": cosine,
    }


def feature_stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    delta = candidate - reference
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "relative_l2": float(
            np.linalg.norm(delta) / max(float(np.linalg.norm(reference)), 1.0e-12)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio", type=Path, required=True, help="16 kHz float32 waveform used by the native run"
    )
    parser.add_argument("--debug", type=Path, required=True, help="native trt-debug.json")
    parser.add_argument(
        "--vfrm", type=Path, default=ROOT / "target/tsukuyomi-canonical.vfr"
    )
    parser.add_argument(
        "--hubert",
        type=Path,
        default=ROOT / "models/shared/hubert_base/hubert_features.onnx",
    )
    parser.add_argument(
        "--rmvpe",
        type=Path,
        default=ROOT / "models/shared/rmvpe/rmvpe_20231006.onnx",
    )
    parser.add_argument(
        "--reference",
        choices=("onnx", "applio"),
        default="onnx",
        help="acoustic reference: exported ONNX or the local Applio PyTorch path",
    )
    parser.add_argument(
        "--hubert-model",
        type=Path,
        default=ROOT / "models/shared/hubert_base",
        help="local Hugging Face HuBERT directory for --reference applio",
    )
    parser.add_argument(
        "--rmvpe-model",
        type=Path,
        default=ROOT / "models/shared/rmvpe/rmvpe.pt",
        help="local Applio RMVPE checkpoint for --reference applio",
    )
    parser.add_argument("--report", type=Path, help="optional JSON report path")
    args = parser.parse_args()

    debug = json.loads(args.debug.read_text(encoding="utf-8"))
    audio = np.fromfile(args.audio, dtype="<f4").astype(np.float32, copy=False)
    waveform = audio[None, :]

    if args.reference == "onnx":
        hubert_session = ort.InferenceSession(
            str(args.hubert), providers=["CPUExecutionProvider"]
        )
        rmvpe_session = ort.InferenceSession(
            str(args.rmvpe), providers=["CPUExecutionProvider"]
        )
        hubert_reference = hubert_session.run(None, {"waveform": waveform})[0].astype(
            np.float32, copy=False
        )
        rmvpe_reference = rmvpe_session.run(
            None,
            {
                "waveform": waveform,
                "threshold": np.asarray([0.03], dtype=np.float32),
            },
        )[0].astype(np.float32, copy=False)
        reference_name = "ONNX HuBERT + ONNX RMVPE + FP32 Applio PyTorch Synth"
    else:
        # This follows Applio's Python worker directly. The native graph's
        # frontend still owns the append/repeat/crop geometry below, so the
        # acoustic tensors are compared at precisely the same RVC frame count.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hubert_model = _ref.HubertModel.from_pretrained(
            str(args.hubert_model), local_files_only=True
        ).to(device).float().eval()
        with torch.inference_mode():
            hubert_reference = (
                hubert_model(torch.from_numpy(waveform).to(device)).last_hidden_state
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
        rmvpe_model = _ref.RMVPE0Predictor(str(args.rmvpe_model), device="cpu")
        rmvpe_reference = np.asarray(
            rmvpe_model.infer_from_audio(audio), dtype=np.float32
        )[None, :]
        reference_name = "Applio PyTorch HuBERT + Applio RMVPE + FP32 Applio PyTorch Synth"

    hubert_native = np.asarray(debug["hubert_features"], dtype=np.float32).reshape(
        hubert_reference.shape
    )
    native_frames = int(debug["rvc_frames"])
    pitch_native = np.asarray(debug["pitch"], dtype=np.int32).reshape(-1)[:native_frames]
    pitchf_native = np.asarray(debug["aligned_pitchf"], dtype=np.float32).reshape(-1)[
        :native_frames
    ]
    raw_pitchf_native = np.asarray(debug["pitchf"], dtype=np.float32).reshape(-1)[
        :native_frames
    ]

    # RVC/Applio append the final HuBERT feature, repeat each feature twice,
    # and crop to p_len.  TensorRT frontend.cpp uses nearest/asymmetric resize,
    # which is exactly repeat(..., 2) for this integer scale factor.
    hubert_augmented = np.concatenate(
        [hubert_reference[0], hubert_reference[:, -1:, :][0]], axis=0
    )
    reference_features = np.repeat(hubert_augmented, 2, axis=0)[:native_frames]
    reference_pitchf = rmvpe_reference[0, :native_frames]
    reference_pitch, _ = _ref.quantize_f0(reference_pitchf)

    config, tensors = read_vfrm(args.vfrm)
    latent = int(config["metadata"]["rvc_config"]["latent_channels"])
    upsample = int(np.prod(config["metadata"]["rvc_config"]["upsample_rates"]))
    frames = native_frames
    rnd = np.asarray(
        [0.01 * np.sin(i * 0.017) for i in range(latent * frames)], dtype=np.float32
    ).reshape(1, latent, frames)
    noise = np.asarray(
        [0.001 * np.sin(i * 0.013) for i in range(frames * upsample)], dtype=np.float32
    ).reshape(1, 1, frames * upsample)
    net, _ = build_reference(config, tensors, noise)
    with torch.inference_mode():
        reference_audio = synthesize(
            net,
            torch.from_numpy(reference_features[None, :]),
            torch.from_numpy(reference_pitch[None, :]),
            torch.from_numpy(reference_pitchf[None, :]),
            torch.from_numpy(rnd),
            torch.tensor([frames], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        ).numpy()
    native_audio = np.asarray(debug["audio"], dtype=np.float32)

    # This second metric isolates TensorRT Synth precision from acoustic feature
    # drift: it feeds the native frontend tensors to the same FP32 PyTorch Synth.
    with torch.inference_mode():
        stage_reference = synthesize(
            net,
            torch.from_numpy(
                np.asarray(debug["rvc_features"], dtype=np.float32).reshape(
                    1, frames, 768
                )
            ),
            torch.from_numpy(pitch_native[None, :]),
            torch.from_numpy(pitchf_native[None, :]),
            torch.from_numpy(rnd),
            torch.tensor([frames], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        ).numpy()

    result = {
        "scope": "complete waveform HuBERT/RMVPE/frontend/Synth parity",
        "reference": reference_name,
        "audio": str(args.audio),
        "debug": str(args.debug),
        "shapes": {
            "waveform_samples": int(audio.size),
            "hubert": list(hubert_reference.shape),
            "rmvpe": list(rmvpe_reference.shape),
            "rvc_frames": frames,
            "audio_samples": int(native_audio.size),
        },
        "intermediate": {
            "hubert": feature_stats(hubert_reference, hubert_native),
            "rmvpe_pitchf": feature_stats(
                rmvpe_reference[0, :native_frames], raw_pitchf_native
            ),
            "frontend_pitchf": feature_stats(
                rmvpe_reference[0, :native_frames], pitchf_native
            ),
            "coarse_pitch_exact_fraction": float(
                np.mean(reference_pitch == pitch_native)
            ),
        },
        "final_audio": stats(reference_audio, native_audio),
        "synth_only_with_native_inputs": stats(stage_reference, native_audio),
    }
    report = args.report or args.debug.with_name("end-to-end-parity.json")
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
