"""Generate reproducible offline voice/pitch/retrieval listening comparisons.

Uses one continuous excerpt, serial GPU subprocesses, model-native sample rates,
and separate constant-gain listening copies. Does not change saved GUI settings.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]


def listening_page(root, report):
    labels = {"elysia-cn": "爱莉希雅", "luotianyi-speech": "洛天依 · 说话",
              "luotianyi-song": "洛天依 · 歌唱", "silver-wolf-cn": "银狼"}
    rows = []
    for item in report["results"]:
        label = next((name for model, name in labels.items() if item["name"].startswith(model)), item["name"])
        link = html.escape(Path(item["listening"]).relative_to(root).as_posix(), quote=True)
        rows.append(f'<tr><td>{label}</td><td>{item["pitch_semitones"]:+g}</td>'
                    f'<td>{item["index_ratio"]:g}</td><td>{item["sample_rate"] // 1000} kHz</td>'
                    f'<td><audio controls preload="none" src="{link}"></audio></td></tr>')
    for path in sorted(root.glob("*-applio-*.wav")):
        rate, samples = wavfile.read(path)
        samples = samples.astype(np.float64) / 32768 if samples.dtype == np.int16 else samples.astype(np.float64)
        gain = min(0.1 / max(float(np.sqrt(np.mean(samples ** 2))), 1e-30),
                   0.9 / max(float(np.abs(samples).max()), 1e-30))
        destination = root / "listening" / path.name
        wavfile.write(destination, rate, (samples * gain).astype(np.float32))
        link = html.escape(destination.relative_to(root).as_posix(), quote=True)
        rows.append(f'<tr><td colspan="4">原始 Applio · {html.escape(path.stem)}</td>'
                    f'<td><audio controls preload="none" src="{link}"></audio></td></tr>')
    source = html.escape(Path(report["input"]).relative_to(root).as_posix(), quote=True)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>中文特色模型离线试听</title>
<style>body{background:#10151d;color:#e5eaf3;font:15px "Segoe UI",sans-serif;max-width:1040px;margin:40px auto;padding:0 24px}
h1{font-size:28px}p{color:#acbbcf;line-height:1.6}table{border-collapse:collapse;width:100%;background:#192230;border-radius:12px;overflow:hidden}
th,td{padding:14px 18px;text-align:left;border-bottom:1px solid #2b3749}th{color:#90cfff;font-weight:600}audio{height:36px;width:320px;max-width:100%;color-scheme:dark}
@media(max-width:800px){th,td{padding:10px 6px}audio{width:230px}}</style>
<h1>中文特色模型 · 离线试听</h1>
<p>同一段连续男声，无实时拼接、SOLA 或启动静音。试听副本只调整整段恒定音量，不改语速、不加 EQ。<br>
先比较爱莉希雅的 0 / +12 半音，再比较是否启用 index。+12 是本次实验参数，不是所有模型的固定建议。</p>
<p>原始输入：</p><audio controls preload="none" src="SOURCE"></audio>
<p>音高列单位为半音；检索列为 index ratio。银狼模型不包含 index。</p>
<table><thead><tr><th>模型</th><th>音高</th><th>检索</th><th>原生采样率</th><th>播放</th></tr></thead><tbody>ROWS</tbody></table>
<script>document.addEventListener('play',e=>{if(e.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true)</script></html>'''
    (root / "audition.html").write_text(page.replace("SOURCE", source).replace("ROWS", "\n".join(rows)), encoding="utf-8")


def json_result(text):
    decoder = json.JSONDecoder()
    for offset, value in enumerate(text):
        if value != "{":
            continue
        try:
            result, _ = decoder.raw_decode(text[offset:])
        except ValueError:
            continue
        if isinstance(result, dict) and "report" in result and "backend" in result:
            return result
    raise ValueError("offline command did not emit its JSON report")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--executable", type=Path, default=ROOT / "target/debug/vocal-fast.exe")
    ap.add_argument("--offset", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--models", nargs="+", default=["elysia-cn", "luotianyi-speech", "luotianyi-song", "silver-wolf-cn"])
    ap.add_argument("--pitch", type=float, default=12.0)
    ap.add_argument("--trace-model", default="elysia-cn")
    ap.add_argument("--resume", action="store_true", help="reuse completed WAV/report pairs")
    ap.add_argument("--render-only", action="store_true", help="refresh listening page from the existing report and Applio WAVs")
    args = ap.parse_args()
    args.backend = "directml"
    if args.render_only:
        root = args.output_dir.resolve()
        listening_page(root, json.loads((root / "audition-report.json").read_text(encoding="utf-8")))
        return
    if not math.isfinite(args.offset) or args.offset < 0 or not math.isfinite(args.duration) or not 0 < args.duration < 38:
        ap.error("offset must be nonnegative; duration must be in (0,38) seconds")
    if not math.isfinite(args.pitch) or not -96 <= args.pitch <= 96:
        ap.error("pitch must be finite and in [-96,96]")
    rate, samples = wavfile.read(args.input)
    if rate != 16000 or samples.ndim != 1:
        ap.error("input must be mono 16 kHz")
    start = round(rate * args.offset)
    stop = start + round(rate * args.duration)
    if stop > len(samples):
        ap.error("excerpt exceeds source WAV")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    clip = root / f"input-male-{args.duration:g}s.wav"
    wavfile.write(clip, rate, samples[start:stop])
    cases = []
    for model in args.models:
        ratio = 0.0 if model == "silver-wolf-cn" else 0.5
        cases.extend([(model, 0.0, 0.0), (model, args.pitch, ratio)])
        if model == args.trace_model and ratio != 0:
            # Isolate pitch from retrieval instead of changing both at once.
            cases.insert(len(cases) - 1, (model, args.pitch, 0.0))
    report = {
        "source": str(args.input.resolve()), "excerpt_offset_seconds": args.offset,
        "excerpt_duration_seconds": args.duration, "input": str(clip),
        "input_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
        "backend": args.backend, "protect": 0.33, "rms_mix_rate": 1.0,
        "normalization": "separate listening copies: constant gain, RMS 0.1, peak <=0.9",
        "results": [],
    }
    executable = args.executable.resolve()
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    for model, pitch, ratio in cases:
        name = f"{model}-{args.backend}-p{pitch:g}-i{ratio:g}"
        output, metadata = root / f"{name}.wav", root / f"{name}.json"
        trace = root / "traces" / name
        model_path = args.library.resolve() / f"{model}-complete.vfrm"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        model_stat = model_path.stat()
        identity = {"input_sha256": report["input_sha256"], "executable_sha256": executable_sha256,
                    "model_path": str(model_path), "model_bytes": model_stat.st_size,
                    "model_mtime_ns": model_stat.st_mtime_ns, "pitch": pitch, "index_ratio": ratio,
                    "protect": 0.33, "rms_mix_rate": 1.0, "trace": model == args.trace_model}
        cached = None
        if args.resume and output.is_file() and metadata.is_file():
            try:
                cached = json.loads(metadata.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass
        if cached is not None and cached.get("audition_identity") != identity:
            cached = None
        env = os.environ.copy()
        env["VOCAL_OFFLINE_PITCH"] = str(pitch)
        env["VOCAL_AUTO_PITCH"] = "0"  # Manual-only single-factor comparisons.
        env["VOCAL_INDEX_RATIO"] = str(ratio)
        env["VOCAL_OFFLINE_PROTECT"] = "0.33"
        env["VOCAL_OFFLINE_RMS_MIX"] = "1"
        env.pop("VOCAL_OFFLINE_TRACE_DIR", None)
        if model == args.trace_model:
            env["VOCAL_OFFLINE_TRACE_DIR"] = str(trace)
        if cached is None:
            print(f"START {name}", flush=True)
            started = time.perf_counter()
            command = [str(executable), "model", "run-pipeline", args.backend,
                       str(model_path), str(clip), str(output)]
            with (root / f"{name}.stdout.log").open("w", encoding="utf-8") as stdout, \
                    (root / f"{name}.stderr.log").open("w", encoding="utf-8") as stderr:
                result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
            if result.returncode:
                raise RuntimeError(f"{name} failed with {result.returncode}; see stderr/stdout logs")
            data = json_result((root / f"{name}.stdout.log").read_text(encoding="utf-8"))
            data["wall_seconds_including_loading"] = time.perf_counter() - started
            data["audition_identity"] = identity
            metadata.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            data = cached
        output_rate, audio = wavfile.read(output)
        audio = audio.astype(np.float64) / 32768 if audio.dtype == np.int16 else audio.astype(np.float64)
        if not np.isfinite(audio).all() or not len(audio):
            raise ValueError(f"invalid output: {output}")
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.abs(audio).max())
        gain = min(0.1 / max(rms, 1e-30), 0.9 / max(peak, 1e-30))
        listening = root / "listening" / f"{name}.wav"
        listening.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(listening, output_rate, (audio * gain).astype(np.float32))
        item = {"name": name, "model": str(model_path), "pitch_semitones": pitch,
                "index_ratio": ratio, "sample_rate": int(output_rate),
                "duration_seconds": len(audio) / output_rate, "rms": rms, "peak": peak,
                "clipped_samples": int(np.sum(np.abs(audio) >= 0.999)),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "wall_seconds_including_loading": data["wall_seconds_including_loading"],
                "wav": str(output), "listening": str(listening), "report": str(metadata)}
        if (trace / "0000/pitchf.bin").is_file():
            f0 = np.fromfile(trace / "0000/pitchf.bin", dtype="<f4")
            voiced = f0[f0 > 0]
            item["aligned_f0_hz_p5_p50_p95"] = np.percentile(voiced, [5, 50, 95]).tolist() if voiced.size else []
            item["voiced_fraction"] = float(np.mean(f0 > 0))
        report["results"].append(item)
        (root / "audition-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"DONE {name}: {item['duration_seconds']:.2f}s, wall {item['wall_seconds_including_loading']:.1f}s, RMS {rms:.4f}", flush=True)
    listening_page(root, report)


if __name__ == "__main__":
    main()
