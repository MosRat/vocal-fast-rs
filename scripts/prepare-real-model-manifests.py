"""Write local manifests for downloaded RVC checkpoints used in parity runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("model_id")
    parser.add_argument("--source", default="https://huggingface.co/wok000/vcclient_model")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), list):
        raise ValueError(f"{args.checkpoint} is not an RVC checkpoint")
    config = checkpoint["config"]
    if len(config) < 18:
        raise ValueError(f"{args.checkpoint}: expected an 18-entry RVC config")
    if not isinstance(checkpoint.get("weight"), dict):
        raise ValueError(f"{args.checkpoint}: no weight state_dict")
    raw = args.checkpoint.read_bytes()
    manifest = {
        "id": args.model_id,
        "display_name": args.model_id,
        "architecture": "rvc-v2",
        "version": "v2",
        "sample_rate": int(config[17]),
        "model_file": args.checkpoint.name,
        "index_file": None,
        "source": args.source,
        "source_repository": args.source,
        "source_revision": "main",
        "license": "upstream repository terms apply",
        "uses_f0": bool(checkpoint.get("f0", 1)),
        "feature_model": "hubert_base",
        "pitch_model": "rmvpe",
        "notes": "Public RVC v2 model downloaded for local parity benchmarking; weights are not committed.",
        "files": {
            args.checkpoint.name: {
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        },
    }
    args.model_dir.mkdir(parents=True, exist_ok=True)
    (args.model_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"model_dir": str(args.model_dir), **manifest}, indent=2))


if __name__ == "__main__":
    main()
