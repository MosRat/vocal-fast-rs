"""Compare the native TensorRT RVC Synthesizer with the Applio reference.

The comparison intentionally uses one set of HuBERT/RMVPE outputs and one set
of deterministic random tensors for both backends.  The native graph currently
starts at the Synthesizer inputs, so this makes the numerical comparison
stage-aligned while reporting HuBERT and RMVPE timings separately.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from transformers import HubertModel


ROOT = Path(__file__).resolve().parents[1]
APPLIO = ROOT / "ref_repos" / "Applio"
sys.path.insert(0, str(APPLIO))

from rvc.lib.algorithm.synthesizers import Synthesizer  # noqa: E402
from rvc.lib.predictors.RMVPE import RMVPE0Predictor  # noqa: E402


VFRM_MAGIC = b"VFRM\0\0\0\1"
VIN_MAGIC = b"VFRTEST1"


def read_vfrm(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    raw = path.read_bytes()
    if raw[:8] != VFRM_MAGIC:
        raise ValueError(f"{path} is not VFRM v1")
    metadata_len, count, directory_len, data_offset = struct.unpack_from(
        "<QQQQ", raw, 16
    )
    config_start = 64
    config_end = config_start + metadata_len
    config = json.loads(raw[config_start:config_end])
    cursor = config_end
    descriptors = []
    tensors: dict[str, np.ndarray] = {}
    dtype_map = {1: np.uint8, 2: np.int32, 3: np.int64, 4: np.float16, 5: np.float32}
    for _ in range(count):
        name_len, dtype, rank, _reserved = struct.unpack_from("<IBBH", raw, cursor)
        cursor += 8
        name = raw[cursor : cursor + name_len].decode("utf-8")
        cursor += name_len
        shape = struct.unpack_from("<" + "Q" * rank, raw, cursor)
        cursor += 8 * rank
        offset, length = struct.unpack_from("<QQ", raw, cursor)
        cursor += 16
        if dtype not in dtype_map:
            raise ValueError(f"unsupported VFRM dtype {dtype} in {name}")
        if offset < data_offset or offset + length > len(raw):
            raise ValueError(f"VFRM tensor range is invalid for {name}")
        itemsize = np.dtype(dtype_map[dtype]).itemsize
        if length != int(np.prod(shape, dtype=np.uint64)) * itemsize:
            raise ValueError(f"VFRM tensor byte length is invalid for {name}")
        values = np.frombuffer(raw, dtype=dtype_map[dtype], count=length // itemsize, offset=offset)
        tensors[name] = values.copy().reshape(shape)
    if cursor > config_end + directory_len:
        raise ValueError("VFRM directory exceeds its declared length")
    return config, tensors


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if channels != 1 or sample_width != 2 or sample_rate != 16000:
        raise ValueError("the fixture must be mono, signed 16-bit PCM at 16 kHz")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return audio, sample_rate


def quantize_f0(f0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Applio's 50..1100 Hz coarse F0 mapping."""
    f0 = np.asarray(f0, dtype=np.float32)
    mel = 1127.0 * np.log1p(np.maximum(f0, 0.0) / 700.0)
    mel_min = 1127.0 * np.log1p(50.0 / 700.0)
    mel_max = 1127.0 * np.log1p(1100.0 / 700.0)
    mapped = np.where(f0 > 0, (mel - mel_min) * 254.0 / (mel_max - mel_min) + 1.0, 1.0)
    coarse = np.clip(np.rint(mapped), 1, 255).astype(np.int32)
    return coarse, f0.astype(np.float32)


class DeterministicSource(nn.Module):
    """Applio NSF source with the random noise supplied by the .vin input."""

    def __init__(self, linear: nn.Linear, noise: torch.Tensor, sample_rate: int):
        super().__init__()
        self.l_linear = linear
        self.register_buffer("noise", noise)
        self.sample_rate = sample_rate

    def forward(self, f0: torch.Tensor, upsample_factor: int = 1):
        f0 = f0.unsqueeze(-1)
        batch, length, _ = f0.shape
        grid = torch.arange(1, upsample_factor + 1, device=f0.device, dtype=f0.dtype)
        grid = grid.reshape(1, 1, -1)
        increments = (f0 / self.sample_rate) * grid
        remainder = torch.fmod(increments[:, :-1, -1:] + 0.5, 1.0) - 0.5
        cumulative = torch.fmod(remainder.cumsum(dim=1), 1.0)
        cumulative = F.pad(cumulative, (0, 0, 1, 0))
        increments = (increments + cumulative).reshape(batch, -1, 1)
        sine = torch.sin(2.0 * np.pi * increments) * 0.1

        uv = (f0 > 0).to(f0.dtype)
        uv = F.interpolate(
            uv.transpose(1, 2), scale_factor=float(upsample_factor), mode="nearest"
        ).transpose(1, 2)
        amplitude = uv * 0.003 + (1.0 - uv) * (0.1 / 3.0)
        noise = self.noise.to(device=f0.device, dtype=f0.dtype).transpose(1, 2)
        source = sine * uv + noise * amplitude
        return torch.tanh(self.l_linear(source)), None, None


def build_reference(config: dict, tensors: dict[str, np.ndarray], noise: np.ndarray):
    c = config["metadata"]["rvc_config"]
    cfg = [
        1025,
        32,
        c["latent_channels"],
        c["hidden_channels"],
        c["filter_channels"],
        c["attention_heads"],
        c["encoder_layers"],
        c["encoder_kernel"],
        0,
        str(c["resblock_type"]),
        c["resblock_kernels"],
        c["resblock_dilations"],
        c["upsample_rates"],
        c["decoder_channels"],
        c["upsample_kernels"],
        c["speakers"],
        c["speaker_channels"],
        c["sample_rate"],
    ]
    net = Synthesizer(
        *cfg,
        use_f0=bool(c["uses_f0"]),
        text_enc_hidden_dim=c["feature_width"],
        randomized=False,
    ).float().eval()
    for module in net.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)

    state = net.state_dict()
    for name, value in tensors.items():
        if name not in state:
            raise ValueError(f"canonical tensor {name} is not an Applio Synthesizer parameter")
        if name == "enc_p.emb_phone.weight":
            value = value.T
        expected = tuple(state[name].shape)
        if tuple(value.shape) != expected:
            raise ValueError(f"shape mismatch for {name}: VFRM={value.shape}, Applio={expected}")
        state[name] = torch.from_numpy(np.asarray(value, dtype=np.float32).copy())
    missing, unexpected = net.load_state_dict(state, strict=False)
    unexpected = [name for name in unexpected if not name.startswith("enc_q.")]
    missing = [name for name in missing if not name.startswith("enc_q.")]
    if missing or unexpected:
        raise ValueError(f"VFRM state mapping failed: missing={missing}, unexpected={unexpected}")

    if c["uses_f0"]:
        old_source = net.dec.m_source
        net.dec.m_source = DeterministicSource(
            old_source.l_linear,
            torch.from_numpy(noise.astype(np.float32, copy=False)),
            int(c["sample_rate"]),
        )
    return net, c


def extract_hubert(model: HubertModel, audio: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        feats = model(audio).last_hidden_state
        return F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)


def synthesize(net, feats, pitch, pitchf, rnd, p_len, sid):
    with torch.inference_mode():
        g = net.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = net.enc_p(feats, pitch, p_len)
        z_p = (m_p + torch.exp(logs_p) * rnd * 0.66666) * x_mask
        z = net.flow(z_p, x_mask, g=g, reverse=True)
        return net.dec(z * x_mask, pitchf, g=g)[0, 0].float()


def timings(fn, warmups: int, iterations: int) -> tuple[object, list[float]]:
    result = None
    for _ in range(warmups):
        result = fn()
    values = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn()
        values.append((time.perf_counter() - start) * 1000.0)
    return result, values


def stats(values: list[float]) -> dict[str, float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return {
        "mean_ms": float(np.mean(ordered)),
        "median_ms": float(np.median(ordered)),
        "p95_ms": float(np.percentile(ordered, 95)),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def write_vin(path: Path, feats, pitch, pitchf, rnd, noise, p_len: int, sid: int, upsample: int):
    feats = np.ascontiguousarray(feats, dtype=np.float32)
    rnd = np.ascontiguousarray(rnd, dtype=np.float32)
    pitch = np.ascontiguousarray(pitch, dtype=np.int32)
    pitchf = np.ascontiguousarray(pitchf, dtype=np.float32)
    noise = np.ascontiguousarray(noise, dtype=np.float32)
    frames, feature_width = feats.shape[1:]
    latent = rnd.shape[1]
    with path.open("wb") as out:
        out.write(VIN_MAGIC)
        out.write(struct.pack("<6I", frames, feature_width, latent, upsample, p_len, sid))
        out.write(feats.tobytes())
        out.write(np.asarray([p_len], dtype="<i4").tobytes())
        out.write(np.asarray([sid], dtype="<i4").tobytes())
        out.write(rnd.tobytes())
        out.write(pitch.tobytes())
        out.write(pitchf.tobytes())
        out.write(noise.tobytes())


def run_native(plan: Path, vin: Path, output: Path, iterations: int) -> dict:
    native_path = ROOT / "target" / "native-benchmark.json"
    env = os.environ.copy()
    trt_bin = r"C:\Program Files\NVIDIA Corporation\TensorRT-RTX\1.6.1.120\bin"
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64"
    env["PATH"] = ";".join([trt_bin, cuda_bin, env.get("PATH", "")])
    command = [
        "rtk",
        "cargo",
        "run",
        "-p",
        "vocal-fast",
        "--features",
        "tensorrt-rtx",
        "--quiet",
        "--",
        "model",
        "benchmark-trt",
        str(plan),
        str(vin),
        str(output),
        str(iterations),
    ]
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            "native benchmark failed\n"
            + completed.stdout[-4000:]
            + "\n"
            + completed.stderr[-4000:]
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("{"):
            result = json.loads(line)
            native_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            return result
    raise RuntimeError(f"native benchmark did not emit JSON:\n{completed.stdout}")


def compare(reference: np.ndarray, native: np.ndarray) -> dict[str, float | int]:
    if reference.shape != native.shape:
        raise ValueError(f"output shape mismatch: python={reference.shape}, native={native.shape}")
    if not np.isfinite(native).all() or not np.isfinite(reference).all():
        raise ValueError("one backend produced a non-finite waveform")
    delta = native.astype(np.float64) - reference.astype(np.float64)
    ref64 = reference.astype(np.float64)
    native64 = native.astype(np.float64)
    denom = np.linalg.norm(ref64)
    cosine = float(np.dot(ref64, native64) / (denom * np.linalg.norm(native64)))
    return {
        "output_samples": int(reference.size),
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "relative_l2": float(np.linalg.norm(delta) / max(denom, 1e-12)),
        "cosine_similarity": cosine,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=ROOT / "testdata/fixtures/sine-440hz-16k-mono.wav")
    parser.add_argument("--vfrm", type=Path, default=ROOT / "target/tsukuyomi-canonical.vfr")
    parser.add_argument("--plan", type=Path, default=ROOT / "target/tsukuyomi-native.plan")
    parser.add_argument("--hubert", type=Path, default=ROOT / "models/shared/hubert_base")
    parser.add_argument("--rmvpe", type=Path, default=ROOT / "models/shared/rmvpe/rmvpe.pt")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "target/compare")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_np, sample_rate = load_wav(args.audio)
    audio = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    config, tensors = read_vfrm(args.vfrm)
    net, c = build_reference(config, tensors, np.zeros((1, 1, 1), dtype=np.float32))
    upsample = int(np.prod(c["upsample_rates"], dtype=np.int64))

    # HuBERT is the same local model used by the Applio pipeline.
    hubert_load_start = time.perf_counter()
    hubert = HubertModel.from_pretrained(args.hubert, local_files_only=True).to(device).float().eval()
    hubert_load_ms = (time.perf_counter() - hubert_load_start) * 1000.0
    feats, hubert_times = timings(lambda: extract_hubert(hubert, audio), 2, args.iterations)
    frames = int(feats.shape[1])
    p_len = min(audio_np.shape[0] // 160, frames)

    # Run the actual local RMVPE model and quantize with Applio's mapping.
    rmvpe_load_start = time.perf_counter()
    rmvpe = RMVPE0Predictor(str(args.rmvpe), device="cpu")
    rmvpe_load_ms = (time.perf_counter() - rmvpe_load_start) * 1000.0
    f0, rmvpe_times = timings(lambda: rmvpe.infer_from_audio(audio_np), 1, args.iterations)
    f0 = np.asarray(f0, dtype=np.float32)[:p_len]
    pitch_short, pitchf_short = quantize_f0(f0)
    # The graph has dynamic frame tensors; the tail is masked by p_len but
    # still needs a valid frame-sized pitch input for TensorRT shape binding.
    pitch_np = np.zeros(frames, dtype=np.int32)
    pitchf_np = np.zeros(frames, dtype=np.float32)
    pitch_np[:p_len] = pitch_short
    pitchf_np[:p_len] = pitchf_short

    rng = np.random.default_rng(args.seed)
    rnd_np = rng.standard_normal((1, int(c["latent_channels"]), frames), dtype=np.float32)
    noise_np = rng.standard_normal((1, 1, frames * upsample), dtype=np.float32)
    net, _ = build_reference(config, tensors, noise_np)
    feats_np = feats.detach().cpu().numpy().astype(np.float32, copy=False)
    pitch = torch.from_numpy(pitch_np).unsqueeze(0).to(device=device, dtype=torch.long)
    pitchf = torch.from_numpy(pitchf_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    rnd = torch.from_numpy(rnd_np).to(device=device, dtype=torch.float32)
    p_len_t = torch.tensor([p_len], device=device, dtype=torch.long)
    sid = torch.tensor([0], device=device, dtype=torch.long)
    reference_t, synth_times = timings(
        lambda: synthesize(net, feats, pitch, pitchf, rnd, p_len_t, sid), 2, args.iterations
    )
    reference = reference_t.detach().cpu().numpy().astype(np.float32)

    vin = args.work_dir / "sine-440hz.vin"
    native_output_path = args.work_dir / "native-output.f32"
    write_vin(vin, feats_np, pitch_np, pitchf_np, rnd_np, noise_np, p_len, 0, upsample)
    native_timing = run_native(args.plan, vin, native_output_path, args.iterations)
    native = np.fromfile(native_output_path, dtype="<f4")
    numerical = compare(reference, native)
    np.save(args.work_dir / "python-output.npy", reference)

    report = {
        "scope": "RVC Synthesizer only; shared HuBERT/RMVPE-derived inputs",
        "audio": str(args.audio),
        "vfrm": str(args.vfrm),
        "plan": str(args.plan),
        "device_python": str(device),
        "sample_rate": sample_rate,
        "frames": frames,
        "p_len": p_len,
        "upsample": upsample,
        "hubert_load_ms": hubert_load_ms,
        "hubert": stats(hubert_times),
        "rmvpe_load_ms": rmvpe_load_ms,
        "rmvpe": stats(rmvpe_times),
        "python_synthesizer": stats(synth_times),
        "native_tensorrt": native_timing,
        "python_cpu_synth_speedup_over_native_mean": stats(synth_times)["mean_ms"]
        / native_timing["mean_ms"],
        "python_cpu_frontend_plus_synth_mean_ms": stats(hubert_times)["mean_ms"]
        + stats(rmvpe_times)["mean_ms"]
        + stats(synth_times)["mean_ms"],
        "numerical": numerical,
        "artifacts": {
            "vin": str(vin),
            "python_output": str(args.work_dir / "python-output.npy"),
            "native_output": str(native_output_path),
        },
    }
    report_path = args.work_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
