"""Publish an exact verified allow-list, then check remote sizes and digests."""
from __future__ import annotations

import json
from pathlib import Path

from .pipeline import command, verify
from .sources import ms_api, sha256


def publish(output: Path, repo: str, cli: Path) -> None:
    inventory = verify(output, cli)
    lock = json.loads((output / "sources.lock.json").read_text(encoding="utf-8"))
    if repo != lock["repository"]:
        raise ValueError("publish repository differs from the locked destination")
    allowed = sorted(set(inventory) | {"inventory.json", "SHA256SUMS"})
    for name in allowed:
        if not (name in {"README.md", "catalog.json", "sources.lock.json", "configuration.json",
                         "inventory.json", "SHA256SUMS", ".gitattributes"}
                or name.startswith("provenance/") and Path(name).suffix.lower() in (".json", ".md", ".txt")
                or name.startswith("provenance/") and Path(name).name.lower() in ("license", "copying", "notice")
                or name.startswith(("voices/", "shared/", "bundles/")) and name.endswith(".vfrm")):
            raise ValueError(f"file is not a collection distribution artifact: {name}")
    api = ms_api()
    from modelscope_hub.errors import NotExistError
    try:
        api.get_repo(repo, "model")
    except NotExistError:
        api.create_repo(repo, "model", visibility="public", license="other",
                        description="Canonical VFRM RVC voices with pinned upstream provenance and per-model licenses")
    api.upload_folder(repo, "model", output, allow_patterns=allowed, max_workers=2,
                      commit_message="Publish verified VFRM collection", disable_tqdm=True,
                      sync_remote_repo=False)
    remote = {f.path: f for f in api.list_repo_files(repo, "model", recursive=True) if f.type == "blob"}
    for name in allowed:
        local = output / name
        item = remote.get(name)
        if item is None or item.size != local.stat().st_size or item.sha256 != sha256(local):
            raise ValueError(f"remote verification failed: {name}")
    # Round-trip one actual voice through the public download API and Rust parser.
    catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    voice = catalog["voices"][0]
    downloaded = api.download_file(repo, "model", voice["file"], force=True,
                                   expected_sha256=voice["sha256"])
    command(cli, "verify", downloaded)
    print(f"published and verified https://modelscope.cn/models/{repo}", flush=True)
