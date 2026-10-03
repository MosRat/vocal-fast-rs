"""Compare native Synth debug outputs with Applio FP32 stage outputs."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PRECISION = ROOT / "scripts" / "synth-precision-applio.py"
spec = importlib.util.spec_from_file_location("synth_precision", PRECISION)
mod = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(mod)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vfrm", type=Path, default=ROOT / "target/tsukuyomi-canonical.vfr")
    ap.add_argument("--vin", type=Path, required=True)
    ap.add_argument("--native-prefix", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()
    config, tensors = mod.read_vfrm(args.vfrm)
    inp = mod.read_vin(args.vin)
    net, _ = mod.build_reference(config, tensors, inp["noise"])
    device = torch.device(args.device)
    net.to(device).eval()
    with torch.inference_mode():
        ref = mod.run_variant(net, inp, device, "fp32")
        pitchf = torch.from_numpy(inp["pitchf"]).to(device)
        source = net.dec.m_source(pitchf, int(inp["upsample"]))[0].transpose(1, 2).float()
        sid = torch.tensor([int(inp["sid"])], device=device, dtype=torch.long)
        g = net.emb_g(sid).unsqueeze(-1)
        dec = net.dec
        x = (dec.conv_pre(ref["z"]) + dec.cond(g)).float()
        decoder_ref = {"decoder_pre": x.detach().cpu()}
        harmonic = source
        for i, (ups, noise_conv) in enumerate(zip(dec.ups, dec.noise_convs)):
            x = torch.nn.functional.leaky_relu(x, dec.lrelu_slope)
            x = ups(x)
            decoder_ref[f"decoder_up_{i}"] = x.detach().cpu()
            x = x + noise_conv(harmonic)
            decoder_ref[f"decoder_noise_{i}"] = x.detach().cpu()
            blocks = [dec.resblocks[i * dec.num_kernels + j](x) for j in range(dec.num_kernels)]
            x = sum(blocks) / dec.num_kernels
            decoder_ref[f"decoder_res_{i}"] = x.detach().cpu()
        # HiFi-GAN's final activation intentionally uses torch's default
        # F.leaky_relu slope (0.01); the intermediate upsample/residual
        # activations use dec.lrelu_slope (0.1).
        pre = dec.conv_post(torch.nn.functional.leaky_relu(x))
        decoder_ref["audio_pre_tanh"] = pre.detach().cpu()

    def native(name: str, shape: tuple[int, ...]) -> torch.Tensor:
        values = np.fromfile(args.native_prefix.with_suffix(name), dtype="<f4")
        return torch.from_numpy(values.reshape(shape))

    frames = int(inp["frames"])
    latent = int(inp["rnd"].shape[1])
    up = int(inp["upsample"])
    values = {
        "encoder_z": (ref["z_p"], native(".encoder_z.f32", (1, latent, frames))),
        "flow_z": (ref["z"], native(".flow_z.f32", (1, latent, frames))),
        "nsf_source": (source.cpu(), native(".nsf_source.f32", (1, 1, frames * up))),
        "audio": (ref["audio"].cpu(), native(".f32", (frames * up,))),
    }
    native_flow = values["flow_z"][1].to(device)
    sid = torch.tensor([int(inp["sid"])], device=device, dtype=torch.long)
    g = net.emb_g(sid).unsqueeze(-1)
    native_flow_audio = mod.decode(
        net, native_flow, torch.from_numpy(inp["pitchf"]).to(device), g, "fp32", device
    ).reshape(-1).cpu()
    values["audio_with_native_flow_z"] = (native_flow_audio, values["audio"][1])
    native_pre = native(".audio_pre_tanh.f32", tuple(decoder_ref["audio_pre_tanh"].shape))
    values["tanh_native_pre"] = (torch.tanh(native_pre).reshape(-1), values["audio"][1])
    values["reference_decode_vs_manual"] = (
        ref["audio"].reshape(-1).cpu(), torch.tanh(decoder_ref["audio_pre_tanh"]).reshape(-1)
    )
    for name, expected in decoder_ref.items():
        shape = tuple(expected.shape)
        values[name] = (expected, native(f".{name}.f32", shape))
    # Isolate the final conv_post from accumulated decoder error.  Feeding the
    # native resblock-3 activation through the Python FP32 post convolution
    # tells us whether the remaining waveform gap is caused by conv_post
    # itself or by error propagated from earlier decoder blocks.
    native_res3 = native(".decoder_res_3.f32", tuple(decoder_ref["decoder_res_3"].shape))
    native_post_input = native(".decoder_post_input.f32", tuple(decoder_ref["decoder_res_3"].shape))
    values["decoder_post_input"] = (
        torch.nn.functional.leaky_relu(native_res3), native_post_input
    )
    python_post_from_native = dec.conv_post(
        torch.nn.functional.leaky_relu(native_res3.to(device))
    ).detach().cpu()
    report = {"reference": "Applio Synthesizer PyTorch FP32", "device": str(device),
              "vin": str(args.vin), "native_prefix": str(args.native_prefix), "stages": {}}
    for name, (expected, actual) in values.items():
        report["stages"][name] = mod.metric(expected, actual)
    report["stages"]["python_conv_post_from_native_res3"] = mod.metric(
        decoder_ref["audio_pre_tanh"], python_post_from_native
    )
    report["stages"]["native_conv_post_from_native_res3"] = mod.metric(
        python_post_from_native, native(".audio_pre_tanh.f32", tuple(decoder_ref["audio_pre_tanh"].shape))
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
