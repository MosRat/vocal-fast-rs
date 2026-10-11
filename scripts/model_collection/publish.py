"""Publish an exact verified allow-list and produce an auditable remote diff."""
from __future__ import annotations

import json
from pathlib import Path

from .pipeline import COLLECTION_SCHEMA, command, verify
from .sources import ms_api, sha256


STATIC_FILES = {"README.md", "catalog.json", "sources.lock.json", "configuration.json",
                "inventory.json", "SHA256SUMS", ".gitattributes"}


def allowed_file(name: str) -> bool:
    return (name in STATIC_FILES
            or name.startswith("provenance/") and Path(name).suffix.lower() in (".json", ".md", ".txt")
            or name.startswith("provenance/") and Path(name).name.lower() in ("license", "copying", "notice")
            or name.startswith(("voices/", "shared/", "bundles/")) and name.endswith(".vfrm"))


def remote_diff(output: Path, inventory: dict, remote: dict[str, object]) -> dict:
    local_names = set(inventory) | {"inventory.json", "SHA256SUMS"}
    remote_names = set(remote)
    local_only = sorted(local_names - remote_names)
    remote_only = sorted(remote_names - local_names)
    changed = []
    for name in sorted(local_names & remote_names):
        local = output / name
        item = remote[name]
        size = getattr(item, "size", None)
        digest = getattr(item, "sha256", None)
        local_digest = inventory.get(name, {}).get("sha256") or sha256(local)
        if size != local.stat().st_size or (digest and digest.lower() != local_digest):
            changed.append({"path": name, "localSize": local.stat().st_size,
                            "remoteSize": size, "localSha256": local_digest,
                            "remoteSha256": digest})
    return {"localOnly": local_only, "remoteOnly": remote_only, "changed": changed}


def write_report(output: Path, report: dict, path: Path | None) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_catalog(output: Path, catalog: dict, inventory: dict) -> None:
    if catalog.get("schema") != COLLECTION_SCHEMA or catalog.get("catalogVersion") != 2:
        raise ValueError("catalog.json is not collection schema v2")
    artifacts = catalog.get("voices", []) + catalog.get("shared", []) + catalog.get("bundles", [])
    for artifact in artifacts:
        name = artifact.get("file")
        if not isinstance(name, str) or name not in inventory:
            raise ValueError(f"catalog artifact is missing from inventory: {name}")
        if artifact.get("sha256") != inventory[name]["sha256"] or artifact.get("size") != inventory[name]["size"]:
            raise ValueError(f"catalog digest does not match inventory: {name}")


def publish(output: Path, repo: str, cli: Path, prune: bool = False,
            report_path: Path | None = None, roundtrip: str = "sample") -> dict:
    inventory = verify(output, cli)
    lock = json.loads((output / "sources.lock.json").read_text(encoding="utf-8"))
    if repo != lock["repository"]:
        raise ValueError("publish repository differs from the locked destination")
    catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    validate_catalog(output, catalog, inventory)
    allowed = sorted(set(inventory) | {"inventory.json", "SHA256SUMS"})
    for name in allowed:
        if not allowed_file(name):
            raise ValueError(f"file is not a collection distribution artifact: {name}")
    api = ms_api()
    from modelscope_hub.errors import NotExistError
    try:
        api.get_repo(repo, "model")
    except NotExistError:
        api.create_repo(repo, "model", visibility="public", license="other",
                        description="Canonical VFRM RVC voices with pinned upstream provenance and per-model licenses")
    remote_before = {f.path: f for f in api.list_repo_files(repo, "model", recursive=True) if f.type == "blob"}
    before_diff = remote_diff(output, inventory, remote_before)
    # Keep the resumable tracker outside the distribution tree. A tracker created
    # by an older SDK can otherwise mark changed metadata as already committed.
    tracker = output.parent / ".modelscope-upload-cache" / f"{repo.replace('/', '--')}.json"
    api.upload_folder(repo, "model", output, allow_patterns=allowed, max_workers=2,
                      commit_message=f"Publish VFRM catalog {catalog['sourceLockSha256'][:12]}", disable_tqdm=True,
                      sync_remote_repo=False, tracker_path=tracker)
    remote = {f.path: f for f in api.list_repo_files(repo, "model", recursive=True) if f.type == "blob"}
    for name in allowed:
        local = output / name
        item = remote.get(name)
        if item is None or item.size != local.stat().st_size or item.sha256 != sha256(local):
            raise ValueError(f"remote verification failed: {name}")
    after_diff = remote_diff(output, inventory, remote)
    stale = after_diff["remoteOnly"]
    if stale and prune:
        api.delete_files(repo, "model", file_paths=stale,
                         commit_message=f"Prune stale VFRM files {catalog['sourceLockSha256'][:12]}")
        remote = {f.path: f for f in api.list_repo_files(repo, "model", recursive=True) if f.type == "blob"}
        after_diff = remote_diff(output, inventory, remote)
        if after_diff["remoteOnly"]:
            raise ValueError(f"remote prune left stale files: {after_diff['remoteOnly']}")
    roundtrip_paths = []
    if roundtrip not in {"none", "sample", "all"}:
        raise ValueError("roundtrip must be none, sample or all")
    if roundtrip != "none":
        roundtrip_paths.extend(entry["file"] for entry in catalog.get("shared", []))
        roundtrip_paths.extend(entry["file"] for entry in catalog.get("bundles", []))
        voices = catalog.get("voices", []) if roundtrip == "all" else catalog.get("voices", [])[:1]
        roundtrip_paths.extend(entry["file"] for entry in voices)
        for name in roundtrip_paths:
            expected = inventory[name]["sha256"]
            downloaded = api.download_file(repo, "model", name, force=True, expected_sha256=expected)
            if name.endswith(".vfrm"):
                command(cli, "verify", downloaded)
    report = {"repository": repo, "catalogSchema": catalog["schema"],
              "sourceLockSha256": catalog["sourceLockSha256"], "before": before_diff,
              "after": after_diff, "pruned": stale if prune else [],
              "roundtrip": roundtrip_paths}
    write_report(output, report, report_path)
    print(f"published and verified https://modelscope.cn/models/{repo} "
          f"({len(allowed)} files, {len(after_diff['remoteOnly'])} stale)", flush=True)
    return report
