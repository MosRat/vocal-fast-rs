"""Download verified male AISHELL-3 utterances and make offline-test WAVs.

Requires requests, numpy, scipy and soundfile (available in the project .venv).
Downloaded recordings and generated audio remain under the ignored target/.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
import requests
from scipy.signal import resample_poly
import soundfile as sf


REPO = "shenyunhang/AISHELL-3"
REVISION = "e1af0fa23933585f1844b3c07b71006f86589bab"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
# SHA-256 comes from the pinned Hugging Face LFS tree. Gender is separately
# checked against spk-info.txt before any recordings are downloaded.
SOURCES = {
    "SSB0261": [
        ("SSB02610348", "157600d87502edd3e672c08089bdd242ac14f5d9196bc39ec54f72435884a11d"),
        ("SSB02610339", "3aa65f85c07b63f78948cc742df3b0d4234f8a2e0ce928f0a74a12ef873aabb3"),
        ("SSB02610275", "1349b43ecc77733d3db00636c2440befa012e9b58c571fffc6944e782d58cd58"),
    ],
    "SSB0710": [
        ("SSB07100494", "f2c965f624a1a88139f9870ebc7affbc76cdc853a72f5ea1a6738842e33b7e8b"),
        ("SSB07100023", "99c3cc7a6613225204351f2cb2e2cc8b6c2399c3380318bb450dd5983eaf8320"),
        ("SSB07100288", "2f1d357f162cd0795c1cb6a16a3c1a59d3f244c929c42eb1006c6c81caeedd8e"),
    ],
    "SSB0434": [
        ("SSB04340069", "f21d80712235ed36a238c48a91bb200cb736aad121173f3b7486b46fc1c72623"),
        ("SSB04340165", "4050c2daa6119700498b6cf61f74cf851dcc7170663ed3bbed5286b49faa5f32"),
        ("SSB04340137", "78590109fb62bf2e59419c31fafcd64c3c08f3797502de2dc0baa51ab808a0a5"),
    ],
}
AGE_GROUPS = {"B": "14–25", "C": "26–40", "D": "over 41"}


def fetch(url: str) -> bytes:
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(15, 60))
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def recording_info(audio: np.ndarray, rate: int) -> dict:
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("expected nonempty, finite mono audio")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio * audio)))
    clipping = float(np.mean(np.abs(audio) >= 32767 / 32768))
    if rms < 0.003 or peak <= 0 or clipping > 0.0001:
        raise ValueError(f"unsuitable recording: RMS={rms}, clipped fraction={clipping}")
    return {
        "sample_rate": rate,
        "channels": 1,
        "frames": int(audio.size),
        "duration_seconds": round(audio.size / rate, 3),
        "peak_dbfs": round(20 * math.log10(peak), 2),
        "rms_dbfs": round(20 * math.log10(rms), 2),
        "clipped_samples": int(np.count_nonzero(np.abs(audio) >= 32767 / 32768)),
    }


def write_wav(path: Path, audio: np.ndarray, rate: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.wav")
    sf.write(temporary, audio, rate, subtype="PCM_16")
    decoded, actual_rate = sf.read(temporary, dtype="float64")
    info = recording_info(decoded, actual_rate)
    temporary.replace(path)
    return {"file": path.name, "sha256": digest(path.read_bytes()), **info}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("target/test-assets/mandarin-male"))
    args = parser.parse_args()
    output = args.output.resolve()
    originals = output / "originals"
    originals.mkdir(parents=True, exist_ok=True)

    print("Reading pinned speaker gender metadata and transcripts…", flush=True)
    speaker_data = fetch(f"{BASE}/spk-info.txt")
    speakers = {}
    for line in speaker_data.decode("utf-8-sig").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        speaker, age, gender, accent = line.split()
        speakers[speaker] = {"age_group": age, "age_years": AGE_GROUPS.get(age, age),
                             "gender": gender, "accent_region": accent}
    for speaker in SOURCES:
        if speakers.get(speaker, {}).get("gender") != "male":
            raise ValueError(f"{speaker} is not marked male in source metadata")
    (output / "spk-info.txt").write_bytes(speaker_data)

    transcript_file = output / "source-content.txt"
    if not transcript_file.exists():
        transcript_file.write_bytes(fetch(f"{BASE}/train/content.txt"))
    required = {utterance for sources in SOURCES.values() for utterance, _ in sources}
    transcripts = {}
    for line in transcript_file.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        utterance = parts[0].removesuffix(".wav") if parts else ""
        if utterance in required:
            transcripts[utterance] = "".join(parts[1::2])
    if set(transcripts) != required:
        raise ValueError(f"missing transcripts: {required - set(transcripts)}")

    def download(speaker_and_sources: tuple) -> dict:
        speaker, sources = speaker_and_sources
        records = []
        chunks = []
        sample_rate = None
        for utterance, expected_hash in sources:
            path = originals / f"{utterance}.wav"
            url = f"{BASE}/train/wav/{speaker}/{utterance}.wav"
            print(f"Downloading {utterance} · verified male {speaker}", flush=True)
            data = path.read_bytes() if path.exists() else fetch(url)
            if digest(data) != expected_hash:
                raise ValueError(f"source checksum mismatch: {utterance}")
            audio, rate = sf.read(io.BytesIO(data), dtype="float64")
            info = recording_info(audio, rate)
            if rate != 44100 or not 4 <= info["duration_seconds"] <= 12:
                raise ValueError(f"unexpected source format/duration for {utterance}")
            path.write_bytes(data)
            sample_rate = rate
            chunks.append(audio)
            records.append({"file": str(path.relative_to(output)), "url": url,
                            "sha256": expected_hash, "transcript": transcripts[utterance], **info})
        gap = np.zeros(round(sample_rate * 0.3), dtype=np.float64)
        joined = np.concatenate([part for i, chunk in enumerate(chunks)
                                 for part in ([gap, chunk] if i else [chunk])])
        index = list(SOURCES).index(speaker) + 1
        native = write_wav(output / f"male-{index:02d}-{speaker}-44100.wav", joined, sample_rate)
        # Anti-alias resampling only: preserve speaker pitch and speaking rate.
        divisor = math.gcd(sample_rate, 16000)
        resampled = resample_poly(joined, 16000 // divisor, sample_rate // divisor,
                                  window=("kaiser", 8.6))
        gain = min(4.0, 10 ** (-1 / 20) / float(np.max(np.abs(resampled))))
        offline = write_wav(output / f"male-{index:02d}-{speaker}-16k.wav", resampled * gain, 16000)
        print(f"Ready: {offline['file']} · {offline['duration_seconds']} s", flush=True)
        return {"speaker": speaker, **speakers[speaker], "originals": records,
                "native": native, "offline_16k": offline, "offline_gain_db": round(20 * math.log10(gain), 3)}

    with ThreadPoolExecutor(max_workers=3) as pool:
        selections = list(pool.map(download, SOURCES.items()))
    manifest = {
        "dataset": "AISHELL-3", "license": "Apache-2.0", "repository": REPO,
        "revision": REVISION, "dataset_source": "https://www.openslr.org/93/",
        "gender_source": f"{BASE}/spk-info.txt", "speaker_metadata_sha256": digest(speaker_data),
        "processing": "Three utterances per speaker, joined with 300 ms silence; native samples unchanged. "
                      "16 kHz versions use anti-alias resampling and peak gain to -1 dBFS (maximum +12.04 dB). "
                      "No pitch shift, denoising, synthesis or voice conversion.",
        "speakers": selections,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = ["# 中文男声输入测试音频", "", "三位真实说话人均经 AISHELL-3 `spk-info.txt` 的 `male` 标签核对。",
            "原始录音在 `originals/`；未对说话人变调或做语音合成。", "",
            "| 说话人 | 来源年龄组 | 16 kHz 测试文件 | 时长 |", "| --- | --- | --- | --- |"]
    for entry in selections:
        rows.append(f"| {entry['speaker']} | {entry['age_years']} | {entry['offline_16k']['file']} | {entry['offline_16k']['duration_seconds']} 秒 |")
    rows += ["", "同名 `44100.wav` 文件保留原生采样率、原始音量，仅拼接句子。",
             "`16k.wav` 使用抗混叠重采样和音量调整，适合当前 CLI 离线测试。",
             "缺失采样、非有限数据、过低 RMS 和明显数字削波会使下载验证失败。",
             "来源、逐句中文文本、格式、声道数、峰值、RMS、SHA-256 见 `manifest.json`。", "",
             "来源：https://www.openslr.org/93/ · https://huggingface.co/datasets/shenyunhang/AISHELL-3", "许可：Apache-2.0", ""]
    (output / "README.md").write_text("\n".join(rows), encoding="utf-8")
    print(f"Downloaded 9 original male recordings; created 3 native and 3 offline-test WAVs in {output}", flush=True)


if __name__ == "__main__":
    main()
