"""Search Synth FP16 boundaries and explain the resulting waveform error.

The native graph lowers complete decoder stages, so this experiment separates
the three operations in a stage (upsample, NSF noise convolution, residual
blocks) and also compares FP16 versus FP32 residual accumulation.  It uses the
Applio/PyTorch FP32 graph as the reference and records activation ranges and
half-precision quantization statistics for every decoder stage.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def load_precision_module():
    path = ROOT / "scripts" / "synth-precision-applio.py"
    spec = importlib.util.spec_from_file_location("synth_precision_applio", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def autocast(enabled: bool, device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled and device.type == "cuda",
    )


def metric(reference: torch.Tensor, value: torch.Tensor) -> dict[str, float]:
    a = reference.detach().float().cpu().numpy().astype(np.float64).reshape(-1)
    b = value.detach().float().cpu().numpy().astype(np.float64).reshape(-1)
    delta = b - a
    denom = max(float(np.linalg.norm(a)), 1e-12)
    cosine_denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-24)
    reference_rms = float(np.sqrt(np.mean(a * a))) if a.size else 0.0
    value_rms = float(np.sqrt(np.mean(b * b))) if b.size else 0.0
    error_rms = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    return {
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_abs": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "relative_l2": float(np.linalg.norm(delta) / denom),
        "cosine": float(np.dot(a, b) / cosine_denom),
        "reference_rms": reference_rms,
        "value_rms": value_rms,
        "error_to_reference_rms": float(error_rms / max(reference_rms, 1e-12)),
        "snr_db": float(20.0 * np.log10(max(reference_rms, 1e-12) / max(error_rms, 1e-12))),
        "p99_abs": float(np.quantile(np.abs(delta), 0.99)) if delta.size else 0.0,
    }


def tensor_stats(value: torch.Tensor) -> dict[str, float]:
    data = value.detach().float()
    absolute = data.abs()
    half = data.half().float()
    quant_error = half - data
    tiny = torch.finfo(torch.float16).tiny
    return {
        "min": float(data.min().item()),
        "max": float(data.max().item()),
        "rms": float(torch.sqrt(torch.mean(data * data)).item()),
        "p99_abs": float(torch.quantile(absolute.reshape(-1), 0.99).item()),
        "subnormal_fraction": float((absolute < tiny).float().mean().item()),
        "half_overflow_fraction": float((absolute > torch.finfo(torch.float16).max).float().mean().item()),
        "half_quant_rmse": float(torch.sqrt(torch.mean(quant_error * quant_error)).item()),
        "half_quant_relative_l2": float(
            (torch.linalg.vector_norm(quant_error) / torch.linalg.vector_norm(data).clamp_min(1e-12)).item()
        ),
    }


def build_latents(precision, net, inp, device):
    feats = torch.from_numpy(inp["feats"]).to(device)
    pitch = torch.from_numpy(inp["pitch"]).to(device=device, dtype=torch.long)
    pitchf = torch.from_numpy(inp["pitchf"]).to(device)
    rnd = torch.from_numpy(inp["rnd"]).to(device)
    p_len = torch.tensor([int(inp["p_len"])], device=device, dtype=torch.long)
    sid = torch.tensor([int(inp["sid"])], device=device, dtype=torch.long)
    with torch.inference_mode():
        g = net.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = net.enc_p(feats, pitch, p_len)
        z_p = (m_p + torch.exp(logs_p) * rnd * 0.66666) * x_mask
        z = net.flow(z_p, x_mask, g=g, reverse=True)
    return z * x_mask, pitchf, g.float()


def decode_trace(dec, latent, pitchf, g, mode: str, device: torch.device, trace: bool):
    """Decode with independently selectable FP16 operation families.

    Names use ``stageN`` (0..3), ``coherentN`` (all three operations and half
    accumulation), ``mixedN`` (all three operations but FP32 residual sum),
    ``upN``, ``noiseN`` or ``resN``.  ``all_mixed`` lowers all decoder stages
    while preserving FP32 residual sums.
    """
    dec = dec
    source16 = mode == "source16"
    pre16 = mode == "pre16"
    post16 = mode == "post16"
    native_mixed = mode == "native_mixed"
    native_coherent = mode.startswith("native_coherent")
    native_keep_last = mode == "native_mixed_keep_last"
    native_keep_final = mode == "native_mixed_keep_final"
    native_mixed_like = native_mixed or native_keep_last or native_keep_final
    native_stage = int(mode[-1]) if native_coherent else -1
    all_mixed = mode == "all_mixed"
    all_coherent = mode == "all_coherent"
    selected: dict[str, int] = {}
    if mode == "coherent012":
        selected["coherent_mask"] = 0x7
    elif mode == "mixed012":
        selected["mixed_mask"] = 0x7
    elif mode.startswith("coherent"):
        selected["coherent"] = int(mode[-1])
    elif mode.startswith("mixed"):
        selected["mixed"] = int(mode[-1])
    elif mode.startswith("up"):
        selected["up"] = int(mode[-1])
    elif mode.startswith("noise"):
        selected["noise"] = int(mode[-1])
    elif mode.startswith("res"):
        selected["res"] = int(mode[-1])
    scale = 1.0
    for suffix, factor in (
        ("_scale_up2", 2.0),
        ("_scale_up4", 4.0),
        ("_scale_up16", 16.0),
        ("_scale_up256", 256.0),
        # Retain the original down-scale experiment for compatibility with
        # the native input-scale option and older reports.
        ("_scale256", 1.0 / 256.0),
    ):
        if mode.endswith(suffix):
            scale = factor
            mode = mode[: -len(suffix)]
            selected = {"coherent": int(mode[-1])}
            break

    with autocast(source16, device):
        har_source, _, _ = dec.m_source(pitchf, dec.upp)
    har_source = har_source.transpose(1, 2).float()
    with autocast(pre16, device):
        x = dec.conv_pre(latent)
        if g is not None:
            x = x + dec.cond(g)
    x = x.float()
    if native_mixed_like or native_coherent:
        x = x.half() if device.type == "cuda" else x
    stage_trace: list[dict[str, torch.Tensor]] = []
    for i, (ups, noise_conv) in enumerate(zip(dec.ups, dec.noise_convs)):
        coherent = all_coherent or (selected.get("coherent_mask", 0) & (1 << i)) != 0 or selected.get("coherent") == i or (
            native_coherent and native_stage == i
        )
        mixed = all_mixed or (selected.get("mixed_mask", 0) & (1 << i)) != 0 or selected.get("mixed") == i
        keep_last_fp32 = native_keep_last and i == len(dec.ups) - 1
        up16 = (
            all_mixed
            or all_coherent
            or native_mixed_like and not keep_last_fp32
            or native_coherent
            or coherent
            or mixed
            or selected.get("up") == i
        )
        noise16 = (
            all_mixed
            or all_coherent
            or native_coherent and native_stage == i
            or native_keep_last and not keep_last_fp32
            or coherent
            or mixed
            or selected.get("noise") == i
        )
        res16 = all_mixed or all_coherent or coherent or mixed or selected.get("res") == i
        if coherent and device.type == "cuda":
            x = x.half()
        elif x.dtype != torch.float32:
            x = x.float()
        x = F.leaky_relu(x, dec.lrelu_slope)
        x_before = x.float()
        with autocast(up16, device):
            if coherent and scale != 1.0:
                xu = F.conv_transpose1d(
                    x * scale,
                    ups.weight / scale,
                    ups.bias,
                    stride=ups.stride,
                    padding=ups.padding,
                    output_padding=ups.output_padding,
                    groups=ups.groups,
                    dilation=ups.dilation,
                )
            else:
                xu = ups(x)
        with autocast(noise16, device):
            xn = noise_conv(har_source)
        if native_mixed_like or (native_coherent and native_stage != i):
            # Native mixed Synth keeps the learned NSF convolution and add in
            # FP32, then casts the stage result back to the decoder domain.
            mixed_value = xu.float() + xn.float()
        else:
            mixed_value = xu + xn
        blocks = []
        with autocast(res16, device):
            for j in range(dec.num_kernels):
                blocks.append(dec.resblocks[i * dec.num_kernels + j](mixed_value))
        if coherent:
            x = sum(blocks) / dec.num_kernels
        else:
            x = sum(block.float() for block in blocks) / dec.num_kernels
            x = x.float()
        if native_mixed_like or native_coherent:
            # The two repair candidates differ only at the final boundary:
            # keep_last_fp32 executes the complete last stage in FP32, while
            # keep_final retains the low-precision kernels but avoids the
            # final FP32 -> FP16 round trip before conv_post.
            keep_final_fp32 = native_keep_final and i == len(dec.ups) - 1
            x = x.float() if keep_final_fp32 else (x.half() if device.type == "cuda" else x)
        if trace:
            stage_trace.append(
                {
                    "input": x_before.detach().float(),
                    "upsample": xu.detach().float(),
                    "noise": xn.detach().float(),
                    "pre_residual": mixed_value.detach().float(),
                    "output": x.detach().float(),
                }
            )
    if not post16:
        x = x.float()
    x = F.leaky_relu(x)
    with autocast(post16, device):
        audio = torch.tanh(dec.conv_post(x))
    result = audio[:, 0].float()
    return result, stage_trace


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vfrm", type=Path, default=ROOT / "target/tsukuyomi-canonical.vfr")
    parser.add_argument("--vin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument(
        "--modes",
        default="all",
        help=(
            "comma-separated decoder modes to run; default is the complete "
            "boundary sweep. Use this to run a small repair sweep on a real voice."
        ),
    )
    args = parser.parse_args()
    precision = load_precision_module()
    config, tensors = precision.read_vfrm(args.vfrm)
    inp = precision.read_vin(args.vin)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ = precision.build_reference(config, tensors, inp["noise"])
    net.to(device).eval()
    latent, pitchf, g = build_latents(precision, net, inp, device)
    all_modes = [
        "fp32",
        "native_mixed",
        "native_mixed_keep_last",
        "native_mixed_keep_final",
        "source16",
        "pre16",
        "post16",
        *[f"up{i}" for i in range(4)],
        *[f"noise{i}" for i in range(4)],
        *[f"res{i}" for i in range(4)],
        *[f"mixed{i}" for i in range(4)],
        *[f"coherent{i}" for i in range(4)],
        *[f"native_coherent{i}" for i in range(4)],
        "coherent1_scale_up2",
        "coherent1_scale_up4",
        "coherent1_scale_up16",
        "coherent1_scale256",
        "coherent3_scale_up2",
        "coherent3_scale_up4",
        "coherent3_scale_up16",
        "coherent3_scale_up256",
        "all_mixed",
        "all_coherent",
        "mixed012",
        "coherent012",
    ]
    # Preserve one row per mode while keeping the command line sweep stable.
    all_modes = list(dict.fromkeys(all_modes))
    if args.modes.strip().lower() == "all":
        modes = all_modes
    else:
        modes = list(dict.fromkeys(m.strip() for m in args.modes.split(",") if m.strip()))
        unknown = [mode for mode in modes if mode not in all_modes]
        if unknown:
            raise ValueError(f"unknown mode(s): {', '.join(unknown)}")
    reference, reference_trace = decode_trace(net.dec, latent, pitchf, g, "fp32", device, True)
    report = {
        "reference": "Applio/PyTorch FP32 Synth decoder",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "vfrm": str(args.vfrm),
        "vin": str(args.vin),
        "frames": int(inp["frames"]),
        "audio_samples": int(reference.numel()),
        "input_stats": {
            "features_rms": float(np.sqrt(np.mean(np.asarray(inp["feats"], dtype=np.float64) ** 2))),
            "latent_noise_rms": float(np.sqrt(np.mean(np.asarray(inp["rnd"], dtype=np.float64) ** 2))),
            "nsf_noise_rms": float(np.sqrt(np.mean(np.asarray(inp["noise"], dtype=np.float64) ** 2))),
            "pitchf_min": float(np.min(inp["pitchf"])),
            "pitchf_max": float(np.max(inp["pitchf"])),
        },
        "reference_stage_stats": {
            f"stage{i}": {
                name: tensor_stats(value)
                for name, value in stage.items()
            }
            for i, stage in enumerate(reference_trace)
        },
        "variants": {},
    }
    for mode in modes:
        for _ in range(args.warmups):
            decode_trace(net.dec, latent, pitchf, g, mode, device, False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(args.iterations):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                value, _ = decode_trace(net.dec, latent, pitchf, g, mode, device, False)
                end.record()
                end.synchronize()
                times.append(float(start.elapsed_time(end)))
            else:
                value, _ = decode_trace(net.dec, latent, pitchf, g, mode, device, False)
                times.append(0.0)
        value, trace = decode_trace(net.dec, latent, pitchf, g, mode, device, True)
        stage_metrics = {}
        for i, stage in enumerate(trace):
            stage_metrics[f"stage{i}"] = {
                name: metric(reference_trace[i][name], stage[name])
                for name in ("upsample", "noise", "pre_residual", "output")
            }
        report["variants"][mode] = {
            "median_ms": float(np.median(times)),
            "mean_ms": float(np.mean(times)),
            "audio": metric(reference, value),
            "stage_metrics": stage_metrics,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
