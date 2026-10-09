"""Audit local acoustic weights against Applio's advertised ContentVec source.

Download to an independent cache, pin its revision and verify SHA256. Compare
parameters before and after casting to the local checkpoint dtype; differing
serialization hashes alone cannot establish different model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests
import torch


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local", type=Path, default=Path("models/shared/hubert_base/pytorch_model.bin"))
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    repo, folder = "IAHispano/Applio", "Resources/embedders/contentvec"
    info = requests.get(f"https://huggingface.co/api/models/{repo}", timeout=30)
    info.raise_for_status()
    revision = info.json()["sha"]
    tree = requests.get(f"https://huggingface.co/api/models/{repo}/tree/{revision}/{folder}", timeout=30)
    tree.raise_for_status()
    asset = next(x for x in tree.json() if x["path"].endswith("/pytorch_model.bin"))
    expected = asset["lfs"]["oid"]
    args.cache.mkdir(parents=True, exist_ok=True)
    path = args.cache / f"{expected}.bin"
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{asset['path']}"
    if not path.is_file() or file_hash(path) != expected:
        temporary = path.with_suffix(".part")
        started = time.perf_counter()
        print(f"Downloading pinned Applio ContentVec: {asset['size']} bytes", flush=True)
        with requests.get(url, stream=True, timeout=(30, 30)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                downloaded, next_notice = 0, 64 << 20
                for chunk in response.iter_content(chunk_size=1 << 20):
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_notice:
                        print(f"Downloaded {downloaded >> 20} MiB in {time.perf_counter() - started:.1f}s", flush=True)
                        next_notice += 64 << 20
        if file_hash(temporary) != expected:
            raise ValueError("downloaded ContentVec SHA256 mismatch")
        temporary.replace(path)
    native = torch.load(args.local, map_location="cpu", weights_only=True)
    reference = torch.load(path, map_location="cpu", weights_only=True)
    # PyTorch's legacy weight_norm and its parametrization replacement name
    # the same g/v tensors differently. Compare the parameters themselves.
    aliases = {"encoder.pos_conv_embed.conv.weight_g": "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
               "encoder.pos_conv_embed.conv.weight_v": "encoder.pos_conv_embed.conv.parametrizations.weight.original1"}
    used_aliases = {name: target for name, target in aliases.items() if name in reference}
    reference = {aliases.get(name, name): value for name, value in reference.items()}
    comparisons = []
    for name in sorted(native.keys() & reference.keys()):
        a, b = native[name], reference[name]
        if a.shape != b.shape:
            comparisons.append({"name": name, "local_shape": list(a.shape), "reference_shape": list(b.shape)})
            continue
        comparisons.append({"name": name, "elements": a.numel(),
                            "local_dtype": str(a.dtype), "reference_dtype": str(b.dtype),
                            "same_fp32": torch.equal(a.float(), b.float()),
                            "same_after_local_cast": torch.equal(a, b.to(a.dtype)),
                            "max_abs_fp32": float((a.float() - b.float()).abs().max())})
    report = {"source": url, "revision": revision, "reference_sha256": expected,
              "local": str(args.local.resolve()), "local_sha256": file_hash(args.local),
              "local_tensor_count": len(native), "reference_tensor_count": len(reference),
              "reference_key_aliases": used_aliases,
              "local_only": sorted(native.keys() - reference.keys()),
              "reference_only": sorted(reference.keys() - native.keys()),
              "same_after_local_cast_count": sum(x.get("same_after_local_cast", False) for x in comparisons),
              "same_fp32_count": sum(x.get("same_fp32", False) for x in comparisons),
              "comparisons": comparisons}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
