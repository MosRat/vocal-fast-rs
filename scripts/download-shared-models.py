#!/usr/bin/env python3
"""Download and verify the shared HuBERT/RMVPE assets listed in a manifest.

Only entries with a URL are downloaded. Generated ONNX/VFRM artifacts remain
local and are reported as skipped. Downloads are written to a temporary file,
verified, and then renamed so an interrupted transfer cannot look complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry: dict[str, object], root: Path, force: bool) -> str:
    relative = Path(str(entry["path"]))
    destination = root / relative
    expected_bytes = entry.get("bytes")
    expected_hash = entry.get("sha256")
    if destination.exists() and not force:
        if expected_bytes is not None and destination.stat().st_size != int(expected_bytes):
            raise RuntimeError(f"existing file has wrong size: {destination}")
        if expected_hash and sha256(destination) != str(expected_hash):
            raise RuntimeError(f"existing file has wrong SHA-256: {destination}")
        return f"verified {relative}"

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(
            str(entry["url"]), headers={"User-Agent": "vocal-fast-rs/0.1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if expected_bytes is not None and temporary.stat().st_size != int(expected_bytes):
            raise RuntimeError(
                f"downloaded {relative} has {temporary.stat().st_size} bytes, "
                f"expected {expected_bytes}"
            )
        if expected_hash and sha256(temporary) != str(expected_hash):
            raise RuntimeError(f"downloaded {relative} has an unexpected SHA-256")
        temporary.replace(destination)
        return f"downloaded {relative}"
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("models/shared/manifest.json"),
        help="shared asset manifest",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("models/shared"),
        help="destination directory for manifest paths",
    )
    parser.add_argument("--force", action="store_true", help="replace existing files")
    parser.add_argument(
        "--verify-only", action="store_true", help="verify existing downloadable files"
    )
    args = parser.parse_args()
    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = 0
    for entry in document.get("files", []):
        if "url" not in entry:
            print(f"skip generated {entry['path']}")
            continue
        try:
            if args.verify_only:
                path = args.root / str(entry["path"])
                if not path.exists():
                    raise RuntimeError(f"missing {path}")
                if entry.get("bytes") is not None and path.stat().st_size != int(entry["bytes"]):
                    raise RuntimeError(f"wrong size: {path}")
                if entry.get("sha256") and sha256(path) != str(entry["sha256"]):
                    raise RuntimeError(f"wrong SHA-256: {path}")
                print(f"verified {entry['path']}")
            else:
                print(download(entry, args.root, args.force))
        except (OSError, RuntimeError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            failures += 1
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
