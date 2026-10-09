"""Download pinned Chinese RVC checkpoints, verify them and make complete VFRM bundles.

Run with .venv/Scripts/python.exe. Requires torch for weights_only metadata inspection
and the vocal-model prepare-rvc example for restricted Rust checkpoint conversion.
Weights, provenance, intermediates and reports stay under target/chinese-rvc/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

MODELS = {
    "maobailing": {
        "repo": "maotianchen/maobailing_rvc",
        "revision": "e56357e07e6788617318733dce1e10eb92889633",
        "name": "猫百灵 · 中文合成女声",
        "pth": "maobailing_rvc.pth",
        "index": "maobailing_rvc.index",
        "documents": ["README.md", "LICENSE", "PARAMS.md"],
    },
    "qingnian": {
        "repo": "chaye741/RVC-Voice-Models",
        "revision": "acfd9f836ca1fc862a594bc303cfccee8d565e94",
        "name": "青年 · 中文男声",
        "pth": "weights/男声-青年.pth",
        "index": "indices/男声-青年.index",
        "documents": [],
    },
    "wanxin": {
        "repo": "chaye741/RVC-Voice-Models",
        "revision": "acfd9f836ca1fc862a594bc303cfccee8d565e94",
        "name": "婉心 · 中文女声",
        "pth": "weights/女声-婉心.pth",
        "index": "indices/女声-婉心.index",
        "documents": [],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output: Path, size: int, expected_hash: str | None) -> dict:
    if output.is_file() and output.stat().st_size == size:
        digest = sha256(output)
        if expected_hash is None or digest == expected_hash:
            print(f"Verified cached download: {output}", flush=True)
            return {"url": url, "bytes": size, "sha256": digest}
    temporary = output.with_suffix(output.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "vocal-fast-model-downloader"})
            with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as stream:
                written, last_report = 0, 0.0
                digest = hashlib.sha256()
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 2 or written == size:
                        print(f"{output.name}: {written / 1048576:.1f}/{size / 1048576:.1f} MiB ({written * 100 / max(size, 1):.0f}%)", flush=True)
                        last_report = now
            if written != size or (expected_hash is not None and digest.hexdigest() != expected_hash):
                raise ValueError(f"Download size/hash mismatch: {output.name}")
            temporary.replace(output)
            return {"url": url, "bytes": written, "sha256": digest.hexdigest()}
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["maobailing", "wanxin"])
    parser.add_argument("--output", type=Path, default=Path("target/chinese-rvc"))
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--converter", type=Path, default=Path("target/debug/examples/prepare-rvc.exe"))
    parser.add_argument("--hubert", type=Path, default=Path("models/shared/hubert_base/hubert_features.vfr"))
    parser.add_argument("--rmvpe", type=Path, default=Path("models/shared/rmvpe/rmvpe.vfr"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.download_only and not args.converter.is_file():
        parser.error("Build the converter first: rtk cargo build -p vocal-model --example prepare-rvc")
    for model_id in args.models:
        spec = MODELS[model_id]
        directory = args.output / "downloads" / model_id
        directory.mkdir(parents=True, exist_ok=True)
        api = f"https://huggingface.co/api/models/{spec['repo']}/revision/{spec['revision']}?blobs=true"
        with urllib.request.urlopen(api, timeout=30) as response:
            metadata = json.load(response)
        if metadata["sha"] != spec["revision"]:
            raise ValueError("Model revision mismatch")
        files = {item["rfilename"]: item for item in metadata["siblings"]}
        needed = [spec["pth"], spec["index"], *spec["documents"]]
        # Reusing verified files or an existing bundle must not require their space again.
        output = args.output / f"{model_id}-complete.vfrm"
        required = sum(files[name]["size"] for name in needed if not (directory / Path(name).name).is_file())
        if not args.download_only and not output.is_file():
            required += args.hubert.stat().st_size + args.rmvpe.stat().st_size + 2 * (files[spec["pth"]]["size"] + files[spec["index"]]["size"])
        required += 64 * 1024 * 1024
        if shutil.disk_usage(args.output).free < required:
            raise OSError("Insufficient disk space for downloads and complete bundle")
        records = {}
        for name in needed:
            item = files[name]
            url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{urllib.parse.quote(name, safe='/')}"
            records[Path(name).name] = download(url, directory / Path(name).name, item["size"], item.get("lfs", {}).get("sha256"))
        provenance = {"id": model_id, "source": f"https://huggingface.co/{spec['repo']}", "revision": spec["revision"], "license": metadata.get("cardData", {}).get("license", "not specified by publisher"), "files": records}
        (directory / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.download_only:
            continue
        import torch
        checkpoint = torch.load(directory / Path(spec["pth"]).name, map_location="cpu", weights_only=True)
        version = checkpoint.get("version", "v1")
        if version != "v2" or not checkpoint.get("f0", 1):
            raise ValueError(f"{model_id}: expected an RVC v2 F0 checkpoint for the bundled HuBERT")
        config = checkpoint["config"]
        sample_rate = config[-1]
        if isinstance(sample_rate, str):
            sample_rate = {"32k": 32000, "40k": 40000, "48k": 48000}[sample_rate]
        manifest = {"id": f"chinese-{model_id}", "display_name": spec["name"], "architecture": "rvc-v2", "version": "v2", "sample_rate": sample_rate, "model_file": Path(spec["pth"]).name, "index_file": Path(spec["index"]).name, "source": provenance["source"], "notes": f"Source revision {spec['revision']}; see adjacent provenance.json for hashes and license."}
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        del checkpoint
        if output.exists():
            print(f"Bundle already exists, retained: {output}", flush=True)
            continue
        report = subprocess.run(["rtk", "proxy", str(args.converter.resolve()), str(directory.resolve()), str(args.hubert.resolve()), str(args.rmvpe.resolve()), str(output.resolve())], check=True, capture_output=True, text=True, encoding="utf-8")
        (args.output / f"{model_id}-inspection.json").write_text(report.stdout, encoding="utf-8")
        provenance["bundle"] = {"path": str(output), "bytes": output.stat().st_size, "sha256": sha256(output)}
        provenance["acoustic_sources"] = [{"path": str(path), "sha256": sha256(path)} for path in [args.hubert, args.rmvpe]]
        (args.output / f"{model_id}-provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Complete bundle ready: {output} ({output.stat().st_size / 1048576:.1f} MiB)", flush=True)


if __name__ == "__main__":
    main()
