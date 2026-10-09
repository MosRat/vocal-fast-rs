"""Validate a complete CPU/DML/TRT file against FP32 models and realtime DSP.

The native validation example records exact random inputs as binary files.
All acoustic features, rolling F0 and Synth outputs are independently computed
from original PyTorch weights here. The streaming window, RMS, startup silence,
linear resampling and SOLA match the shared Rust file runner. Diagnostic trace
timings are deliberately not reported as production performance.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from scipy.io import wavfile
from streaming_reference_dsp import SpeechSafeJoiner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref_repos/Applio"))
spec = importlib.util.spec_from_file_location("reference", ROOT / "scripts/compare-retrieval-python.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


class Errors:
    """Accumulate whole-stream errors without retaining stage activations."""

    def __init__(self):
        self.n = 0
        self.error2 = self.a2 = self.b2 = self.ab = 0.0
        self.max_abs = 0.0
        self.worst_chunk = None

    def add(self, reference, native, chunk):
        a, b = (np.asarray(x, np.float64).reshape(-1) for x in (reference, native))
        if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"invalid comparison at chunk {chunk}: {a.shape}, {b.shape}")
        d = b - a
        maximum = float(np.max(np.abs(d)))
        if self.worst_chunk is None or maximum > self.max_abs:
            self.max_abs, self.worst_chunk = maximum, chunk
        self.n += a.size
        self.error2 += float(np.sum(d * d))
        self.a2 += float(np.sum(a * a))
        self.b2 += float(np.sum(b * b))
        self.ab += float(np.sum(a * b))

    def report(self):
        return {"elements": self.n, "max_abs": self.max_abs,
                "rmse": math.sqrt(self.error2 / self.n),
                "cosine": self.ab / max(math.sqrt(self.a2 * self.b2), 1e-30),
                "snr_db": 10 * math.log10(max(self.a2, 1e-30) / max(self.error2, 1e-30)),
                "worst_chunk": self.worst_chunk}


def read_wave(path):
    rate, audio = wavfile.read(path)
    if rate != 16000 or audio.ndim != 1:
        raise ValueError("this reference expects a mono 16 kHz WAV")
    if audio.dtype.kind == "i":
        audio = audio.astype(np.float32) / -float(np.iinfo(audio.dtype).min)
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128
    else:
        audio = audio.astype(np.float32)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("empty or non-finite input")
    return audio


def resample_linear(audio, input_rate, output_rate):
    if input_rate == output_rate:
        return audio.copy()
    size = (len(audio) * output_rate + input_rate // 2) // input_rate
    position = np.arange(size, dtype=np.float32) * np.float32(input_rate / output_rate)
    left = position.astype(np.int64)
    right = np.minimum(left + 1, len(audio) - 1)
    weight = position - left.astype(np.float32)
    return audio[np.minimum(left, len(audio) - 1)] * (1 - weight) + audio[right] * weight


class Joiner:
    def __init__(self, overlap=800, search=160):
        self.overlap, self.search = overlap, search
        self.previous = np.zeros(overlap, np.float32)
        self.initialized = False
        phase = np.arange(overlap, dtype=np.float32) / np.float32(overlap - 1)
        self.fade = np.sin(np.float32(0.5) * np.float32(np.pi) * phase) ** 2

    def process(self, audio, block, fixed_offset=None):
        offset = 0
        previous = self.previous.astype(np.float64)
        energy = np.sum(previous ** 2)
        if self.initialized and energy > 1e-12:
            candidates = np.lib.stride_tricks.sliding_window_view(
                audio[:self.overlap + self.search].astype(np.float64), self.overlap)
            numerator = np.sum(candidates * previous, axis=1)
            scores = numerator / np.sqrt(energy * np.sum(candidates ** 2, axis=1) + 1e-8)
            offset = int(np.argmax(scores))
        if fixed_offset is not None:
            offset = fixed_offset
        self.initialized = True
        current = audio[offset:]
        output = current[:block].copy()
        output[:self.overlap] = self.previous * (1 - self.fade) + output[:self.overlap] * self.fade
        self.previous = current[block:block + self.overlap].copy()
        return output, offset


def pcm16(audio):
    scaled = np.clip(audio, -1, 1) * np.float32(32767)
    return (np.sign(scaled) * np.floor(np.abs(scaled) + np.float32(0.5))).astype(np.int16)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--native-output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models/voices/default-v2/default.pth")
    ap.add_argument("--index", type=Path, default=ROOT / "models/voices/default-v2/added_IVF511_Flat_nprobe_1_default_v2.index")
    args = ap.parse_args()
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = json.loads((args.trace / "trace.json").read_text())
    config, controls = metadata["config"], metadata["controls"]
    backend = metadata.get("backend", "directml")
    if backend not in {"cpu", "directml", "tensorrt", "trt"}:
        raise ValueError(f"unsupported trace backend: {backend}")
    policy = config["f0"]
    advance, frames = metadata["advance_samples"], metadata["frames"]
    if advance != 4000 or config["window_samples"] != 44960 or policy["mode"] != "rolling":
        raise ValueError("reference DSP currently implements the 44960/4000 rolling preset")
    if backend == "cpu" and controls.get("rms_mix_rate") != 1:
        raise ValueError("CPU reference currently requires RMS mix=1")
    audio = read_wave(args.input)
    if metadata["input_samples"] != len(audio) or metadata["chunks"] != math.ceil(len(audio) / advance):
        raise ValueError("trace must cover the entire input WAV")
    from transformers import HubertModel
    from rvc.lib.predictors.RMVPE import RMVPE0Predictor
    hubert = HubertModel.from_pretrained(str(ROOT / "models/shared/hubert_base"), local_files_only=True)
    hubert = hubert.to("cuda").float().eval()
    rmvpe = RMVPE0Predictor(str(ROOT / "models/shared/rmvpe/rmvpe.pt"), device="cuda")
    synth, width, rate = ref.load_synth(args.checkpoint, "cuda")
    if rate != metadata["sample_rate"] or width not in (256, 768) or controls["speaker_id"] != 0:
        raise ValueError("unsupported model geometry or speaker")
    index = faiss.deserialize_index(np.frombuffer(args.index.read_bytes(), dtype=np.uint8))
    if index.d != width:
        raise ValueError("retrieval index dimensions disagree with RVC model")
    projection = torch.load(ROOT / "models/shared/hubert_base/pytorch_model.bin", map_location="cuda", weights_only=True) if width == 256 else None
    vectors = index.reconstruct_n(0, index.ntotal)
    ratio = controls["index_ratio"]

    def retrieve(query):
        distances, ids = index.search(np.ascontiguousarray(query), 8)
        weights = np.zeros_like(distances)
        exact = distances[:, 0] <= 1e-12
        if (ids < 0).all(axis=1).any():
            raise ValueError("empty retrieval candidates")
        weights[exact, 0] = 1
        weights[~exact] = (1 / distances[~exact]) ** 2
        weights[ids < 0] = 0
        weights[~exact] /= weights[~exact].sum(axis=1, keepdims=True)
        return ratio * (vectors[ids.clip(min=0)] * weights[..., None]).sum(axis=1) + (1 - ratio) * query

    acoustic_keys = ["hubert", "f0_cache"] if backend in {"cpu", "directml"} else ["hubert_after_retrieval", "aligned_pitchf"]
    if backend == "cpu":
        acoustic_keys.append("rmvpe_pitchf")
    errors = {k: Errors() for k in [*acoustic_keys, "frontend_features", "audio_tail", "synth_only"]}
    window = np.zeros(config["window_samples"], np.float32)
    history = np.zeros(1024, np.float32)
    level = np.zeros(4800, np.float32)
    boundary_mode = metadata["boundary"].get("mode", "legacy_v1")
    if boundary_mode not in {"legacy_v1", "speech_safe_v1"}:
        raise ValueError(f"unsupported streaming boundary: {boundary_mode}")
    joiner = SpeechSafeJoiner if boundary_mode == "speech_safe_v1" else Joiner
    native_join, python_join, fixed_join = joiner(), joiner(), joiner()
    native_blocks, python_blocks, fixed_blocks = [], [], []
    offsets = []
    chunk_metrics = []
    coarse_mismatches = 0
    audible_coarse_mismatches = 0
    warmup = metadata["boundary"].get("warmup_blocks", math.ceil(len(window) / advance) + 1)
    started = time.perf_counter()
    with torch.inference_mode():
        for i in range(metadata["chunks"]):
            chunk_dir = args.trace / f"{i:04}"
            read = lambda name, dtype="<f4": np.fromfile(chunk_dir / f"{name}.bin", dtype=dtype)
            valid = min(advance, len(audio) - i * advance)
            chunk = np.zeros(advance, np.float32)
            chunk[:valid] = audio[i * advance:i * advance + valid]
            window[:-advance] = window[advance:].copy()
            window[-advance:] = chunk
            level[:-advance] = level[advance:].copy()
            level[-advance:] = chunk
            rms = np.sqrt(np.cumsum(level * level, dtype=np.float32)[-1] / np.float32(len(level)))
            hidden = hubert(torch.from_numpy(window[None]).to("cuda"), output_hidden_states=width == 256)
            features = hidden.last_hidden_state
            if width == 256:
                features = torch.nn.functional.linear(hidden.hidden_states[9], projection["final_proj.weight"].float(), projection["final_proj.bias"].float())
            h = features[0].cpu().numpy()
            mel = rmvpe.mel_extractor(torch.from_numpy(window[None, -policy["rmvpe_samples"]:]).to("cuda"), center=True)
            f0 = rmvpe.decode(rmvpe.mel2hidden(mel)[0].cpu().numpy(), thred=0.03)
            shift = policy["advance_samples"] // 160
            history[:-shift] = history[shift:].copy()
            history[-len(f0[3:-1]):] = f0[3:-1]
            pitchf = history[-frames:].copy() * np.float32(2 ** (controls["pitch_semitones"] / 12))
            content = retrieve(h) if ratio else h
            features = np.repeat(np.concatenate([content, content[-1:]]), 2, axis=0)[:frames]
            if ratio and controls["protect"] < 0.5:
                raw = np.repeat(np.concatenate([h, h[-1:]]), 2, axis=0)[:frames]
                amount = np.where(pitchf > 0, 1, controls["protect"]).astype(np.float32)[:, None]
                features = features * amount + raw * (1 - amount)
            f0_mel = 1127 * np.log1p(pitchf / 700)
            low, high = 1127 * np.log1p(np.array([50, 1100], np.float32) / 700)
            pitch = np.rint(np.clip((f0_mel - low) * 254 / (high - low) + 1, 1, 255)).astype(np.int64)
            skip = config["skip_head_frames"]
            snapshot = {"synth_frames": frames, "synth_skip_head": skip,
                        "synth_output_frames": frames - skip, "valid_synth_frames": controls["valid_frames"],
                        "synth_random": read("synth_random"), "synth_noise": read("synth_noise"),
                        "rvc_features": features, "pitch": pitch, "aligned_pitchf": pitchf}
            expected = ref.synth_reference(synth, snapshot, width, "cuda")
            native = read("audio")
            if backend in {"cpu", "directml"}:
                errors["hubert"].add(h, read("hubert_raw"), i)
                errors["f0_cache"].add(history, read("f0_cache"), i)
                if backend == "cpu":
                    errors["rmvpe_pitchf"].add(f0, read("rmvpe_pitchf"), i)
            else:
                errors["hubert_after_retrieval"].add(content, read("hubert_features"), i)
                # TRT initially primes F0 with its full-window graph; startup is muted.
                if i >= warmup:
                    errors["aligned_pitchf"].add(pitchf, read("pitchf"), i)
            errors["frontend_features"].add(features, read("feats"), i)
            mismatch = int(np.count_nonzero(pitch != read("pitch", "<i4")))
            coarse_mismatches += mismatch
            # Ignore startup outputs for audible-tail metrics, retaining their F0 history.
            if i >= warmup:
                audible_coarse_mismatches += mismatch
                errors["audio_tail"].add(expected, native, i)
                chunk_error = Errors()
                chunk_error.add(expected, native, i)
                chunk_metrics.append({"chunk": i, **chunk_error.report(),
                                      "reference_rms": float(np.sqrt(np.mean(expected.astype(np.float64) ** 2))),
                                      "native_rms": float(np.sqrt(np.mean(native.astype(np.float64) ** 2))),
                                      "coarse_mismatch": mismatch})
                snapshot.update(rvc_features=read("feats"), pitch=read("pitch", "<i4"), aligned_pitchf=read("pitchf"))
                errors["synth_only"].add(ref.synth_reference(synth, snapshot, width, "cuda"), native, i)
                gain = np.float32(1) if backend == "cpu" else rms
                native_pcm, native_offset = native_join.process(resample_linear(native, rate, 16000) * gain, advance)
                python_pcm, python_offset = python_join.process(resample_linear(expected, rate, 16000) * gain, advance)
                fixed_pcm, _ = fixed_join.process(resample_linear(expected, rate, 16000) * gain, advance, native_offset)
                offsets.append({"chunk": i, "native": native_offset, "python": python_offset})
            else:
                native_pcm = python_pcm = fixed_pcm = np.zeros(advance, np.float32)
            native_blocks.append(native_pcm[:valid])
            python_blocks.append(python_pcm[:valid])
            fixed_blocks.append(fixed_pcm[:valid])
            if (i + 1) % 20 == 0:
                print(f"checked {i + 1}/{metadata['chunks']} chunks", flush=True)
    elapsed = (time.perf_counter() - started) * 1000
    final_rate = metadata.get("output_rate", rate)
    native_audio = resample_linear(np.concatenate(native_blocks), 16000, final_rate)
    python_audio = resample_linear(np.concatenate(python_blocks), 16000, final_rate)
    fixed_audio = resample_linear(np.concatenate(fixed_blocks), 16000, final_rate)
    output_rate, production_audio = wavfile.read(args.native_output)
    if output_rate != final_rate or production_audio.dtype != np.int16:
        raise ValueError("native output must be a matching PCM16 WAV")
    production_audio = production_audio.astype(np.float32) / 32768
    matches = {}
    for name, left, right in [
        ("diagnostic_vs_production", pcm16(native_audio).astype(np.float32) / 32768, production_audio),
        ("python_vs_native_full_wav", python_audio, production_audio),
        ("python_vs_native_fixed_sola_offsets", fixed_audio, production_audio),
    ]:
        stat = Errors()
        stat.add(left, right, 0)
        matches[name] = stat.report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(args.output, final_rate, python_audio)
    wavfile.write(args.output.with_stem(args.output.stem + "-fixed-sola"), final_rate, fixed_audio)
    wavfile.write(args.output.with_stem(backend + "-diagnostic"), final_rate, pcm16(native_audio))
    report = {"reference": "original FP32 HF HuBERT and Applio RMVPE/Synth; exact native random inputs",
              "backend": backend,
              "input": str(args.input), "native_output": str(args.native_output),
              "output": str(args.output), "sample_rate": rate, "output_rate": final_rate,
              "input_samples": len(audio), "output_samples": len(python_audio), "chunks": metadata["chunks"],
              "warmup_blocks": warmup, "controls": controls,
              "boundary": metadata["boundary"],
              "validation_elapsed_ms_not_production_latency": elapsed,
              "stage_metrics": {k: v.report() for k, v in errors.items()}, "waveform_metrics": matches,
              "coarse_mismatches": coarse_mismatches, "audible_coarse_mismatches": audible_coarse_mismatches,
              "audible_chunk_metrics": chunk_metrics,
              "sola_offset_mismatches": [x for x in offsets if x["native"] != x["python"]]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"sola_offset_mismatches", "audible_chunk_metrics"}}
                     | {"sola_offset_mismatch_count": len(report["sola_offset_mismatches"])}, indent=2))


if __name__ == "__main__":
    main()
