"""Numerical parity check for the complete HuBERT/RMVPE/frontend/Synth chain."""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref_repos" / "Applio"))
_spec = importlib.util.spec_from_file_location("compare_native_python", ROOT / "scripts/compare-native-python.py")
_ref = importlib.util.module_from_spec(_spec); assert _spec.loader; _spec.loader.exec_module(_ref)
build_reference, synthesize, read_vfrm = _ref.build_reference, _ref.synthesize, _ref.read_vfrm

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, default=ROOT / "target/compare/sine-6400.f32")
    ap.add_argument("--debug", type=Path, default=ROOT / "target/trt-pipeline-final/trt-debug.json")
    ap.add_argument("--vfrm", type=Path, default=ROOT / "target/tsukuyomi-canonical.vfr")
    ap.add_argument("--hubert", type=Path, default=ROOT / "models/shared/hubert_base/hubert_features.onnx")
    ap.add_argument("--rmvpe", type=Path, default=ROOT / "models/shared/rmvpe/rmvpe_20231006.onnx")
    args = ap.parse_args()
    audio = np.fromfile(args.audio, dtype="<f4").astype(np.float32)
    debug = json.loads(args.debug.read_text())
    x = audio[None, :]
    h_sess = ort.InferenceSession(str(args.hubert), providers=["CPUExecutionProvider"])
    r_sess = ort.InferenceSession(str(args.rmvpe), providers=["CPUExecutionProvider"])
    t0 = time.perf_counter(); h_ref = h_sess.run(None, {"waveform": x})[0]; h_ms = (time.perf_counter()-t0)*1000
    t0 = time.perf_counter(); r_ref = r_sess.run(None, {"waveform": x, "threshold": np.array([.03], np.float32)})[0]; r_ms = (time.perf_counter()-t0)*1000
    h_trt = np.asarray(debug["hubert_features"], np.float32).reshape(h_ref.shape)
    r_trt = np.asarray(debug["pitchf"], np.float32)
    h_delta = h_trt - h_ref[0]
    r_delta = r_trt - r_ref[0, :r_trt.size]
    pitch = np.asarray(debug["pitch"], np.int64)[None, :]
    aligned_pitchf_debug = debug.get("aligned_pitchf")
    pitchf = np.asarray(
        aligned_pitchf_debug if aligned_pitchf_debug is not None else r_trt,
        np.float32,
    )[None, :]
    # RVC WebUI/Applio append the final HuBERT frame before the 2x temporal
    # interpolation and crop the result to p_len. The native frontend does
    # the same; applying the alignment here keeps the Python Synth reference
    # comparable for the real 11200/44960 application windows (where
    # repeat(hubert_frames, 2) is shorter than p_len).
    rvc_frames = pitch.shape[1]
    rvc_features_debug = debug.get("rvc_features")
    if rvc_features_debug is not None:
        feats = np.asarray(rvc_features_debug, np.float32).reshape(1, rvc_frames, 768)
    else:
        hubert_for_rvc = np.concatenate([h_trt, h_trt[:, -1:, :]], axis=1)
        feats = np.repeat(hubert_for_rvc, 2, axis=1)[:, :rvc_frames, :]
    config, tensors = read_vfrm(args.vfrm)
    net, c = build_reference(config, tensors, np.zeros((1, 1, pitchf.shape[1] * 400), np.float32))
    latent = int(c["latent_channels"]); frames = pitchf.shape[1]; up = int(np.prod(c["upsample_rates"]))
    rnd = np.array([0.01 * np.sin(i * .017) for i in range(latent * frames)], np.float32).reshape(1, latent, frames)
    noise = np.array([0.001 * np.sin(i * .013) for i in range(frames * up)], np.float32).reshape(1, 1, frames * up)
    net, _ = build_reference(config, tensors, noise)
    with torch.inference_mode():
        t0 = time.perf_counter()
        ref = synthesize(net, torch.from_numpy(feats), torch.from_numpy(pitch), torch.from_numpy(pitchf),
                         torch.from_numpy(rnd), torch.tensor([frames]), torch.tensor([0])).numpy()
        synth_ms = (time.perf_counter()-t0)*1000
    native = np.asarray(debug["audio"], np.float32)
    delta = native.astype(np.float64) - ref.astype(np.float64)
    result = {
        "shapes": {"hubert": list(h_ref.shape), "pitchf": list(r_ref.shape), "rvc_frames": frames, "audio": int(native.size)},
        "hubert_ort_ms": h_ms, "rmvpe_ort_ms": r_ms, "python_synth_ms": synth_ms,
        "hubert_max_abs": float(np.max(np.abs(h_delta))), "hubert_rmse": float(np.sqrt(np.mean(h_delta*h_delta))),
        "rmvpe_max_abs": float(np.max(np.abs(r_delta))), "rmvpe_rmse": float(np.sqrt(np.mean(r_delta*r_delta))),
        "synth_max_abs": float(np.max(np.abs(delta))), "synth_rmse": float(np.sqrt(np.mean(delta*delta))),
        "synth_relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(ref), 1e-12)),
    }
    print(json.dumps(result, indent=2))
    (args.debug.with_name("parity.json")).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
