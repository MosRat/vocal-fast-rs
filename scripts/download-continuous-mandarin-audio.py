"""Download complete male Mandarin narratives and extract one interval per file.

Requires ffmpeg on PATH and requests, numpy, scipy, soundfile in the Python env.
No utterance concatenation, silence removal, gain changes, or voice conversion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import requests
from scipy.signal import resample_poly
import soundfile as sf

REVISION = "v2211"
BASE_URL = f"https://raw.githubusercontent.com/Multi-CAST/mcmandarin/{REVISION}/"
CITATION = (
    "Vollmer, Maria. 2022. Multi-CAST Mandarin. In Haig, Geoffrey & Schnell, "
    "Stefan (eds.), Multi-CAST: Multilingual corpus of annotated spoken texts. "
    "Version 2211. Bamberg: University of Bamberg."
)
METADATA = {
    "cldf/texts.csv": "18c6d55c17b36d8d1a949dd573d9c8795592a66a7700b7c5a41878d78f7403b1",
    "cldf/utterances.csv": "ef851db86bb5ab37b063693af50a09ec462b5dd3968cf6815240097663c77772",
    "LICENSE": "945fbcb37deca05bccd7c3ca0e261679f1f5373515504fdbad68f6f3cf6f49b9",
    "README.md": "205b9864bf8087e53783b1e13ce32c9b196e02c144f2f712d95de8739afebc54",
}
RECORDINGS = (
    {
        "id": "lzh", "name": "male-01-liangzhu", "title": "Liang Shanbo and Zhu Yingtai",
        "first_utterance": "lzh_0002", "last_utterance": "lzh_0015",
        "sha256": "f2c1ac740f301ba19e9acb837529d9aed432568aed956a7edc8ed514e950b11c",
    },
    {
        "id": "hml", "name": "male-02-mulan", "title": "Hua Mulan",
        "first_utterance": "hml_0025", "last_utterance": "hml_0041",
        "sha256": "649ed615fb2a9c3d06eabc9c98a2a69aba4dbba06c32cfc3031783ad98da2c96",
    },
)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download(session: requests.Session, relative_url: str, path: Path, expected: str) -> None:
    if path.exists() and sha256(path) == expected:
        print(f"[cache] {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    print(f"[download] {relative_url}", flush=True)
    try:
        with session.get(BASE_URL + relative_url, stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        if sha256(temporary) != expected:
            raise ValueError(f"SHA-256 mismatch for {relative_url}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def decode(ffmpeg: str, mp3: Path, wav: Path) -> None:
    temporary = wav.with_name(wav.name + ".part")
    try:
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp3),
             "-map", "0:a:0", "-c:a", "pcm_s16le", "-f", "wav", str(temporary)],
            check=True,
        )
        temporary.replace(wav)
    finally:
        temporary.unlink(missing_ok=True)


def write_wav(path: Path, audio: np.ndarray, rate: int) -> dict:
    temporary = path.with_name(path.name + ".part")
    try:
        sf.write(temporary, audio, rate, subtype="PCM_16", format="WAV")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    info = sf.info(path)
    return {
        "file": path.name, "sample_rate": info.samplerate, "channels": info.channels,
        "frames": info.frames, "seconds": info.duration, "subtype": info.subtype,
        "sha256": sha256(path),
    }


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def prepare(root: Path, ffmpeg: str) -> None:
    metadata_dir = root / "metadata"
    originals = root / "originals"
    originals.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "source": {
            "repository": "https://github.com/Multi-CAST/mcmandarin/tree/" + REVISION,
            "archive": "https://fd-repo.uni-bamberg.de/records/bxcvm-m9e27",
            "doi": "10.5281/zenodo.14859162", "license": "CC-BY-4.0", "citation": CITATION,
        },
        "processing": {
            "native_wav": "PCM16 decode followed by a single continuous sample slice",
            "test_wav": "same slice; arithmetic mean of channels; polyphase resample to 16000 Hz",
            "concatenation": False, "pause_removal": False, "gain_adjustment": False,
        },
        "clips": [],
    }
    with requests.Session() as session:
        for relative_url, expected in METADATA.items():
            download(session, relative_url, metadata_dir / Path(relative_url).name, expected)
        texts = {row["ID"]: row for row in csv_rows(metadata_dir / "texts.csv")}
        utterances = {row["ID"]: row for row in csv_rows(metadata_dir / "utterances.csv")}
        for recording in RECORDINGS:
            text = texts[recording["id"]]
            if text["Speaker_Gender"] != "male":
                raise ValueError(f"Source metadata does not identify {recording['id']} as male")
            relative_url = f"cldf/media/mc_mandarin_{recording['id']}.mp3"
            mp3 = originals / Path(relative_url).name
            download(session, relative_url, mp3, recording["sha256"])
            full_wav = originals / f"mc_mandarin_{recording['id']}-full.wav"
            print(f"[decode] {mp3.name}", flush=True)
            decode(ffmpeg, mp3, full_wav)
            audio, rate = sf.read(full_wav, dtype="int16", always_2d=True)
            first = utterances[recording["first_utterance"]]
            last = utterances[recording["last_utterance"]]
            if first["Text_ID"] != recording["id"] or last["Text_ID"] != recording["id"]:
                raise ValueError("Utterances and recording do not match")
            start_ms, end_ms = float(first["Audio_Start"]), float(last["Audio_End"])
            start, end = round(start_ms * rate / 1000), round(end_ms * rate / 1000)
            if not 0 <= start < end <= len(audio):
                raise ValueError(f"Invalid excerpt bounds for {recording['id']}")
            clip = audio[start:end]
            native_path = root / f"{recording['name']}-{rate}.wav"
            native_info = write_wav(native_path, clip, rate)
            actual, actual_rate = sf.read(native_path, dtype="int16", always_2d=True)
            if actual_rate != rate or not np.array_equal(actual, clip):
                raise ValueError("Native excerpt differs from the original PCM interval")
            mono = clip.astype(np.float64).mean(axis=1) / 32768
            divisor = math.gcd(rate, 16000)
            converted = resample_poly(mono, 16000 // divisor, rate // divisor)
            if not np.isfinite(converted).all() or np.max(np.abs(converted)) >= 32767 / 32768:
                raise ValueError("Resampled audio would clip; source level must be reviewed explicitly")
            test_info = write_wav(root / f"{recording['name']}-16k.wav", converted, 16000)
            if abs(test_info["seconds"] - native_info["seconds"]) > 1 / 16000:
                raise ValueError("Resampling unexpectedly changed the excerpt duration")
            manifest["clips"].append({
                "title": recording["title"], "speaker": text["Speaker"],
                "gender": text["Speaker_Gender"], "age": int(text["Speaker_Age"]),
                "original_mp3": str(mp3.relative_to(root)), "download_url": BASE_URL + relative_url,
                "original_sha256": sha256(mp3), "original_seconds": len(audio) / rate,
                "original_wav": str(full_wav.relative_to(root)), "original_wav_sha256": sha256(full_wav),
                "start_ms": start_ms, "end_ms": end_ms, "start_frame": start, "end_frame": end,
                "first_utterance": first["ID"], "last_utterance": last["ID"],
                "native_matches_original_pcm": True,
                "native_source_clipped_samples": int(np.count_nonzero((clip == -32768) | (clip == 32767))),
                "resampled_peak": float(np.max(np.abs(converted))),
                "native": native_info, "test_16k": test_info,
            })
            print(f"[verified] {native_path.name}: {native_info['seconds']:.3f} seconds", flush=True)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Continuous male Mandarin input samples", "", CITATION, "",
        "Source: https://github.com/Multi-CAST/mcmandarin/tree/v2211", "License: CC-BY-4.0", "",
        "Each WAV excerpt is one consecutive interval in a complete recording. Natural pauses,",
        "hesitation, breaths, and background sounds are preserved. No audition is claimed.", "",
        "Use the native 44100 Hz stereo WAV for direct playback, and 16k mono for pipeline tests.", "",
        "| Native WAV | Speaker | Original interval (s) | Duration (s) |",
        "| --- | --- | --- | --- |",
    ]
    for clip in manifest["clips"]:
        lines.append(f"| {clip['native']['file']} | {clip['speaker']} (male) | "
                     f"{clip['start_ms']/1000:.3f}–{clip['end_ms']/1000:.3f} | {clip['native']['seconds']:.3f} |")
    lines += ["", "Complete MP3s and decoded WAVs are in `originals/`.",
              "Speaker metadata, utterance boundaries, and license are in `metadata/`.",
              "Checksums and the sample equality check are recorded in `manifest.json`.", ""]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("target/test-assets/mandarin-continuous"))
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg is required on PATH")
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args.output, ffmpeg)


if __name__ == "__main__":
    main()
