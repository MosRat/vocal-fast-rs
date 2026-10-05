"""Component precision study for the Applio/RVC Synthesizer.

The FP32 PyTorch Applio model is the numerical reference.  The variants use
CUDA autocast only around a selected Synth component and convert the result
back to FP32 at the component boundary.  This keeps the experiment
repeatable while making it possible to identify which block is responsible
for a waveform error.  Inputs are read from the deterministic ``.vin`` file
used by the native benchmark; no TensorRT output is used as a reference.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

ROOT = Path(__file__).resolve().parents[1]
APPLIO = ROOT / "ref_repos" / "Applio"
sys.path.insert(0, str(APPLIO))
from rvc.lib.algorithm.synthesizers import Synthesizer  # noqa: E402

VFRM_MAGIC = b"VFRM\0\0\0\1"
VIN_MAGIC = b"VFRTEST1"


def read_vfrm(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    raw = path.read_bytes()
    if raw[:8] != VFRM_MAGIC:
        raise ValueError(f"{path} is not VFRM v1")
    metadata_len, count, directory_len, data_offset = struct.unpack_from("<QQQQ", raw, 16)
    config_end = 64 + metadata_len
    config = json.loads(raw[64:config_end])
    cursor = config_end
    dtype_map = {1: np.uint8, 2: np.int32, 3: np.int64, 4: np.float16, 5: np.float32}
    tensors: dict[str, np.ndarray] = {}
    for _ in range(count):
        name_len, dtype, rank, _reserved = struct.unpack_from("<IBBH", raw, cursor)
        cursor += 8
        name = raw[cursor : cursor + name_len].decode("utf-8")
        cursor += name_len
        shape = struct.unpack_from("<" + "Q" * rank, raw, cursor)
        cursor += 8 * rank
        offset, length = struct.unpack_from("<QQ", raw, cursor)
        cursor += 16
        if dtype not in dtype_map or offset < data_offset or offset + length > len(raw):
            raise ValueError(f"invalid VFRM tensor {name}")
        np_dtype = dtype_map[dtype]
        itemsize = np.dtype(np_dtype).itemsize
        if length != int(np.prod(shape, dtype=np.uint64)) * itemsize:
            raise ValueError(f"invalid byte length for {name}")
        tensors[name] = np.frombuffer(raw, dtype=np_dtype, count=length // itemsize, offset=offset).copy().reshape(shape)
    if cursor > config_end + directory_len:
        raise ValueError("VFRM directory exceeds its declared size")
    return config, tensors


def read_vin(path: Path) -> dict[str, np.ndarray | int]:
    raw = path.read_bytes()
    if raw[:8] != VIN_MAGIC:
        raise ValueError(f"{path} is not a VFR test input")
    frames, feature_width, latent, upsample, p_len, sid = struct.unpack_from("<6I", raw, 8)
    cursor = 32

    def take(dtype: str, count: int) -> np.ndarray:
        nonlocal cursor
        size = np.dtype(dtype).itemsize * count
        value = np.frombuffer(raw, dtype=dtype, count=count, offset=cursor).copy()
        cursor += size
        return value

    feats = take("<f4", frames * feature_width).reshape(1, frames, feature_width)
    p_len_file = int(take("<i4", 1)[0])
    sid_file = int(take("<i4", 1)[0])
    if p_len_file != p_len or sid_file != sid:
        raise ValueError("duplicated VIN header fields do not agree")
    rnd = take("<f4", latent * frames).reshape(1, latent, frames)
    pitch = take("<i4", frames).reshape(1, frames)
    pitchf = take("<f4", frames).reshape(1, frames)
    noise = take("<f4", frames * upsample).reshape(1, 1, frames * upsample)
    if cursor != len(raw):
        raise ValueError(f"unexpected trailing bytes in {path}")
    return {"feats": feats, "rnd": rnd, "pitch": pitch, "pitchf": pitchf, "noise": noise,
            "frames": frames, "p_len": p_len, "sid": sid, "upsample": upsample}


class DeterministicSource(nn.Module):
    """NSF source with the deterministic noise from a VIN fixture."""

    def __init__(self, linear: nn.Linear, noise: torch.Tensor, sample_rate: int):
        super().__init__()
        self.l_linear = linear
        self.register_buffer("noise", noise)
        self.sample_rate = sample_rate

    def forward(self, f0: torch.Tensor, upsample_factor: int = 1):
        f0 = f0.unsqueeze(-1)
        batch, length, _ = f0.shape
        grid = torch.arange(1, upsample_factor + 1, device=f0.device, dtype=f0.dtype).reshape(1, 1, -1)
        increments = (f0 / self.sample_rate) * grid
        remainder = torch.fmod(increments[:, :-1, -1:] + 0.5, 1.0) - 0.5
        cumulative = torch.fmod(remainder.cumsum(dim=1), 1.0)
        cumulative = F.pad(cumulative, (0, 0, 1, 0))
        increments = (increments + cumulative).reshape(batch, -1, 1)
        sine = torch.sin(2.0 * np.pi * increments) * 0.1
        uv = (f0 > 0).to(f0.dtype)
        uv = F.interpolate(uv.transpose(1, 2), scale_factor=float(upsample_factor), mode="nearest").transpose(1, 2)
        amplitude = uv * 0.003 + (1.0 - uv) * (0.1 / 3.0)
        noise = self.noise.to(device=f0.device, dtype=f0.dtype).transpose(1, 2)
        source = sine * uv + noise * amplitude
        return torch.tanh(self.l_linear(source)), None, None


def build_reference(config: dict, tensors: dict[str, np.ndarray], noise: np.ndarray):
    c = config["metadata"]["rvc_config"]
    cfg = [1025, 32, c["latent_channels"], c["hidden_channels"], c["filter_channels"],
           c["attention_heads"], c["encoder_layers"], c["encoder_kernel"], 0,
           str(c["resblock_type"]), c["resblock_kernels"], c["resblock_dilations"],
           c["upsample_rates"], c["decoder_channels"], c["upsample_kernels"],
           c["speakers"], c["speaker_channels"], c["sample_rate"]]
    net = Synthesizer(*cfg, use_f0=bool(c["uses_f0"]), text_enc_hidden_dim=c["feature_width"],
                      randomized=False).float().eval()
    for module in net.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    state = net.state_dict()
    for name, value in tensors.items():
        if name not in state:
            raise ValueError(f"canonical tensor {name} is not an Applio parameter")
        if name == "enc_p.emb_phone.weight":
            value = value.T
        if tuple(value.shape) != tuple(state[name].shape):
            raise ValueError(f"shape mismatch for {name}: VFRM={value.shape}, Applio={state[name].shape}")
        state[name] = torch.from_numpy(np.asarray(value, dtype=np.float32).copy())
    missing, unexpected = net.load_state_dict(state, strict=False)
    missing = [n for n in missing if not n.startswith("enc_q.")]
    unexpected = [n for n in unexpected if not n.startswith("enc_q.")]
    if missing or unexpected:
        raise ValueError(f"VFRM state mapping failed: missing={missing}, unexpected={unexpected}")
    net.dec.m_source = DeterministicSource(net.dec.m_source.l_linear,
                                           torch.from_numpy(noise.astype(np.float32, copy=False)),
                                           int(c["sample_rate"]))
    return net, c


def autocast(enabled: bool, device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def decode(net, x, pitchf, g, mode: str, device: torch.device):
    """HiFi-GAN NSF decoder with selectable FP16 component boundaries."""
    dec = net.dec
    # Keep the component switches explicit.  ``decoder_core_fp16`` models the
    # deployment candidate that moves the expensive upsample/noise/post Conv
    # operators to FP16 while retaining FP32 residual-block accumulation.
    # ``decoder_full_no_source`` additionally moves residual Conv arithmetic to
    # FP16 but leaves the NSF phase/source in FP32.  The source is tested
    # separately because phase arithmetic is more sensitive than convolution.
    full = mode in ("decoder_full", "decoder_full_no_source", "decoder_full_accum", "all_fp16")
    coherent_masks = {
        "decoder_coherent_stage0": 0x1,
        "decoder_coherent_stage1": 0x2,
        "decoder_coherent_stage2": 0x4,
        "decoder_coherent_stage3": 0x8,
        "decoder_coherent_stage12": 0x6,
    }
    coherent_mask = coherent_masks.get(mode, 0)
    # The suffix is a build-time boundary scale experiment.  It mirrors the
    # native input-scale transform: the first low-precision upsample receives
    # s*x and its kernel is folded by 1/s, so the affine result is unchanged
    # apart from the FP16 rounding point.
    input_scale = 1.0
    if mode.endswith("_scale256"):
        input_scale = 1.0 / 256.0
        mode = mode[:-len("_scale256")]
        coherent_mask = coherent_masks.get(mode, 0)
    core = mode in ("decoder_core_fp16", "decoder_full", "decoder_full_no_source", "decoder_full_accum", "all_fp16") or mode.startswith("decoder_core_stage")
    source16 = mode in ("decoder_source", "decoder_full", "all_fp16")
    pre16 = mode in ("decoder_pre", "decoder_full", "decoder_full_no_source", "decoder_full_accum", "all_fp16")
    stage_mask = {f"decoder_core_stage{i}": 1 << i for i in range(len(dec.ups))}
    post16 = core or mode == "decoder_post"
    upsample16 = core or mode == "decoder_upsample" or coherent_mask != 0
    noise16 = core or mode == "decoder_noise" or coherent_mask != 0

    with autocast(source16, device):
        har_source, _, _ = dec.m_source(pitchf, dec.upp)
    har_source = har_source.transpose(1, 2).float()
    with autocast(pre16, device):
        x = dec.conv_pre(x)
        if g is not None:
            x = x + dec.cond(g)
    x = x.float()
    if input_scale != 1.0:
        x = x * input_scale
    for i, (ups, noise_conv) in enumerate(zip(dec.ups, dec.noise_convs)):
        coherent = bool(coherent_mask & (1 << i))
        if coherent and x.dtype != torch.float16 and device.type == "cuda":
            x = x.half()
        elif not coherent and x.dtype != torch.float32:
            x = x.float()
        x = F.leaky_relu(x, dec.lrelu_slope)
        with autocast(upsample16, device):
            if i == 0 and input_scale != 1.0:
                # Match native weight folding exactly:
                # W/s * (s*x) + b == W*x + b.  Dividing the complete output
                # would incorrectly scale the deconvolution bias.
                xu = F.conv_transpose1d(
                    x, ups.weight / input_scale, ups.bias,
                    stride=ups.stride, padding=ups.padding,
                    output_padding=ups.output_padding, groups=ups.groups,
                    dilation=ups.dilation,
                )
            else:
                xu = ups(x)
        with autocast(noise16, device):
            xn = noise_conv(har_source)
        if coherent and device.type == "cuda":
            x = xu + xn
        else:
            x = (xu + xn).float()
        blocks = []
        for j in range(dec.num_kernels):
            stage_enabled = full or mode == "decoder_resblocks" or (stage_mask.get(mode, 0) & (1 << i)) != 0
            with autocast(stage_enabled or coherent, device):
                blocks.append(dec.resblocks[i * dec.num_kernels + j](x))
        # A half accumulation is deliberately a separate experiment.  The
        # default full-decoder candidate still sums branches in FP32 even when
        # each residual Conv runs in FP16.
        if mode == "decoder_full_accum" or coherent:
            x = sum(blocks) / dec.num_kernels
            if not coherent:
                x = x.float()
        else:
            x = (sum(v.float() for v in blocks) / dec.num_kernels).float()
    # Match Applio HiFiGANNSFGenerator.forward: the final activation calls
    # F.leaky_relu without an explicit slope (PyTorch default 0.01), while
    # the upsample/residual loop uses LRELU_SLOPE=0.1.
    if not post16:
        x = x.float()
    x = F.leaky_relu(x)
    with autocast(post16, device):
        x = torch.tanh(dec.conv_post(x))
    return x[:, 0].float()


@torch.inference_mode()
def run_variant(net, inp: dict, device: torch.device, mode: str) -> dict[str, torch.Tensor]:
    feats = torch.from_numpy(inp["feats"]).to(device)
    pitch = torch.from_numpy(inp["pitch"]).to(device=device, dtype=torch.long)
    pitchf = torch.from_numpy(inp["pitchf"]).to(device)
    rnd = torch.from_numpy(inp["rnd"]).to(device)
    p_len = torch.tensor([int(inp["p_len"])], device=device, dtype=torch.long)
    sid = torch.tensor([int(inp["sid"])], device=device, dtype=torch.long)
    fp16_encoder = mode in ("encoder", "all_fp16")
    fp16_flow = mode in ("flow", "all_fp16")
    with autocast(fp16_encoder, device):
        g = net.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = net.enc_p(feats, pitch, p_len)
    m_p, logs_p, x_mask = m_p.float(), logs_p.float(), x_mask.float()
    z_p = (m_p + torch.exp(logs_p) * rnd * 0.66666) * x_mask
    with autocast(fp16_flow, device):
        z = net.flow(z_p, x_mask, g=g, reverse=True)
    z = z.float()
    audio = decode(net, z * x_mask, pitchf, g.float(), mode, device)
    return {"m_p": m_p, "logs_p": logs_p, "z_p": z_p, "z": z, "audio": audio}


def metric(ref: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    a = ref.detach().float().cpu().numpy().astype(np.float64)
    b = test.detach().float().cpu().numpy().astype(np.float64)
    d = b - a
    return {"max_abs": float(np.max(np.abs(d))), "mean_abs": float(np.mean(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d * d))),
            "relative_l2": float(np.linalg.norm(d) / max(np.linalg.norm(a), 1e-12)),
            "cosine": float(np.dot(a.reshape(-1), b.reshape(-1)) /
                              (max(np.linalg.norm(a), 1e-12) * max(np.linalg.norm(b), 1e-12)))}


def timed(fn: Callable[[], dict[str, torch.Tensor]], device: torch.device, warmups: int, iterations: int):
    for _ in range(warmups):
        fn()
    if device.type != "cuda":
        values = []
        result = None
        for _ in range(iterations):
            t0 = time.perf_counter(); result = fn(); values.append((time.perf_counter() - t0) * 1000)
        return result, values
    values = []
    result = None
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); result = fn(); end.record(); end.synchronize(); values.append(float(start.elapsed_time(end)))
    return result, values


def summary(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()), "max_ms": float(a.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vfrm", type=Path, default=ROOT / "target/tsukuyomi-canonical.vfr")
    ap.add_argument("--vin", type=Path, default=ROOT / "target/compare/sine-440hz.vin")
    ap.add_argument("--output", type=Path, default=ROOT / "target/compare/synth-precision-applio-11200.json")
    ap.add_argument("--warmups", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch.set_grad_enabled(False)
    config, tensors = read_vfrm(args.vfrm)
    inp = read_vin(args.vin)
    net, c = build_reference(config, tensors, inp["noise"])
    net.to(device).eval()
    modes = ["fp32", "encoder", "flow", "decoder_source", "decoder_pre", "decoder_upsample",
             "decoder_noise", "decoder_resblocks", "decoder_post", "decoder_core_fp16",
             "decoder_core_stage0", "decoder_core_stage1", "decoder_core_stage2", "decoder_core_stage3",
             "decoder_coherent_stage0", "decoder_coherent_stage1", "decoder_coherent_stage2",
             "decoder_coherent_stage3", "decoder_coherent_stage12",
             "decoder_coherent_stage1_scale256", "decoder_coherent_stage12_scale256",
             "decoder_full_no_source", "decoder_full", "decoder_full_accum", "all_fp16"]
    # The reference is explicitly Applio/PyTorch FP32, independent of native TRT.
    ref = run_variant(net, inp, device, "fp32")
    report = {"reference": "Applio Synthesizer PyTorch FP32 eval/inference_mode",
              "device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "input": {"vin": str(args.vin), "frames": int(inp["frames"]), "p_len": int(inp["p_len"]),
                        "audio_samples": int(ref["audio"].numel())}, "variants": {}}
    for mode in modes:
        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        result, times = timed(lambda m=mode: run_variant(net, inp, device, m), device, args.warmups, args.iterations)
        stages = {name: metric(ref[name], result[name]) for name in ("m_p", "logs_p", "z_p", "z", "audio")}
        report["variants"][mode] = {"latency": summary(times), "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                                    "finite": bool(all(torch.isfinite(v).all().item() for v in result.values())),
                                    "stages": stages}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = args.output.with_suffix(".md")
    lines = ["# Synth precision sensitivity (Applio FP32 reference)", "", f"Reference: `{report['reference']}`", f"Device: `{device}` ({report['gpu']})", "",
             "| variant | mean ms | audio max abs | audio RMSE | audio relative L2 | peak memory |", "|---|---:|---:|---:|---:|---:|"]
    for mode, value in report["variants"].items():
        audio = value["stages"]["audio"]
        lines.append(f"| {mode} | {value['latency']['mean_ms']:.3f} | {audio['max_abs']:.6g} | {audio['rmse']:.6g} | {audio['relative_l2']:.6g} | {value['peak_memory_bytes'] or '-'} |")
    lines += ["", "Stage metrics are also recorded for `m_p`, `logs_p`, `z_p`, and `z`; all are compared with the same Applio FP32 run.", ""]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
