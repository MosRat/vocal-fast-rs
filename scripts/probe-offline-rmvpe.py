"""Isolate cuFFT window/mel error from native RMVPE graph error on an offline trace."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("offline_reference", ROOT / "scripts/run-applio-offline.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
_, _, Predictor, _ = reference.import_reference()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("report", type=Path)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    predictor = Predictor(str(ROOT / "models/shared/rmvpe/rmvpe.pt"), device="cuda")
    wave = torch.from_numpy(np.fromfile(args.trace / "waveform.bin", dtype="<f4")[None]).cuda()
    native_mel = torch.from_numpy(np.fromfile(args.trace / "rmvpe_mel.bin", dtype="<f4").reshape(1, 128, -1)).cuda()
    native_f0 = np.fromfile(args.trace / "pitchf.bin", dtype="<f4")
    captured = {}
    for name, module in [("rmvpe_bottleneck", predictor.model.unet.intermediate),
                         ("rmvpe_unet", predictor.model.unet), ("rmvpe_gru", predictor.model.fc[0]),
                         ("rmvpe_logits", predictor.model.fc[1])]:
        module.register_forward_hook(lambda m, i, out, name=name: captured.update({name: out.detach().cpu().numpy().reshape(-1)}))

    def metric(a, b):
        delta = a.astype(np.float64) - b
        return {"rmse": float(np.sqrt(np.mean(delta ** 2))), "max_abs": float(np.abs(delta).max())}

    with torch.inference_mode():
        original = predictor.mel_extractor(wave, center=True)
        expected = predictor.decode(predictor.mel2hidden(original)[0].cpu().numpy(), thred=.03)[:len(native_f0)]
        result = {"original_vs_native_f0": metric(expected, native_f0)}
        for name, mel_input in [("native_mel", native_mel)]:
            f0 = predictor.decode(predictor.mel2hidden(mel_input)[0].cpu().numpy(), thred=.03)[:len(native_f0)]
            result[name] = {"f0_vs_original": metric(expected, f0), "f0_vs_native": metric(native_f0, f0)}
            result[name]["graph_stages"] = {key: metric(value, np.fromfile(args.trace / f"{key}.bin", dtype="<f4")) for key, value in captured.items()}
        # The native cuFFT path used double-angle cosine; PyTorch constructs
        # its CPU Hann window with a float angle before moving it to CUDA.
        hann = (.5 - .5 * np.cos(2 * np.pi * np.arange(1024) / 1024).astype(np.float32)).astype(np.float32)
        fft = torch.stft(wave, 1024, 160, 1024, torch.from_numpy(hann).cuda(), center=True, return_complex=True)
        mel = torch.log((predictor.mel_extractor.mel_basis @ torch.sqrt(fft.real.square() + fft.imag.square())).clamp(min=1e-5))
        f0 = predictor.decode(predictor.mel2hidden(mel)[0].cpu().numpy(), thred=.03)[:len(native_f0)]
        result["double_angle_hann"] = {"mel_vs_native": metric(mel.cpu().numpy().reshape(-1), native_mel.cpu().numpy().reshape(-1)), "f0_vs_original": metric(expected, f0)}
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
