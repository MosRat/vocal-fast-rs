"""Hub adapters. Downloads are limited to explicitly selected, hashed files."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path, PurePosixPath


def safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or str(path) == "." or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"unsafe repository path: {value!r}")
    return str(path)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def matches(path: Path, record: dict) -> bool:
    return path.is_file() and path.stat().st_size == record["size"] and sha256(path) == record["sha256"]


def ms_api():
    from modelscope_hub.api import HubApi
    return HubApi(token=os.environ.get("MODELSCOPE_API_TOKEN"))


def repository(provider: str, repo: str, revision: str | None = None) -> dict:
    if provider == "hf":
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        files = []
        for item in info.siblings:
            files.append({"path": item.rfilename, "size": item.size,
                          "sha256": item.lfs.sha256 if item.lfs else None})
        card = info.card_data.to_dict() if info.card_data else {}
        return {"provider": provider, "repo": repo, "revision": info.sha,
                "license": card.get("license", "unknown"), "files": files}
    if provider == "ms":
        api = ms_api()
        info = api.get_repo(repo, "model", revision=revision).to_dict()
        files = [{"path": f.path, "size": f.size, "sha256": f.sha256}
                 for f in api.list_repo_files(repo, "model", recursive=True, revision=revision)
                 if f.type == "blob"]
        return {"provider": provider, "repo": repo, "revision": revision or "master",
                "license": info.get("license") or info.get("License") or "unknown", "files": files}
    raise ValueError("provider must be hf or ms")


def search(provider: str, query: str, limit: int) -> list[dict]:
    """Return normalized, inspectable candidates from one or both hubs.

    Search endpoints only return repository summaries. We inspect each small
    result set so callers can distinguish a repository containing exported
    Synth checkpoints from one containing only training assets or documentation.
    A failed repository inspection is retained as a warning instead of making
    a broad search fail.
    """
    if provider not in {"hf", "ms", "all"}:
        raise ValueError("provider must be hf, ms or all")
    providers = ["hf", "ms"] if provider == "all" else [provider]
    names: list[dict] = []
    for current in providers:
        if current == "hf":
            from huggingface_hub import HfApi
            names.extend({"repo": item.id, "provider": current}
                         for item in HfApi().list_models(search=query, limit=limit))
        else:
            page = ms_api().list_repos("model", search=query, page_size=limit, page_number=1)
            names.extend({"repo": item.repo_id, "provider": current} for item in page.items)
    result = []
    seen: set[tuple[str, str]] = set()
    for candidate in names:
        key = (candidate["provider"], candidate["repo"].lower())
        if key in seen:
            continue
        seen.add(key)
        try:
            info = repository(candidate["provider"], candidate["repo"])
            files = eligible_files(info["files"])
            result.append({**candidate, "revision": info["revision"], "license": info["license"],
                           "files": files, "checkpointCount": sum(f["kind"] == "checkpoint" for f in files),
                           "indexCount": sum(f["kind"] == "index" for f in files),
                           "warnings": candidate_warnings(info["license"], files)})
        except Exception as error:
            result.append({**candidate, "files": [], "warnings": [f"inspection failed: {type(error).__name__}"]})
    return sorted(result, key=lambda item: (-item.get("checkpointCount", 0), item["provider"], item["repo"]))


def eligible_files(files: list[dict]) -> list[dict]:
    """Reduce hub listings to runtime-relevant RVC artifacts."""
    result = []
    for file in files:
        path = file["path"]
        suffix = Path(path).suffix.lower()
        name = Path(path).name.lower()
        if suffix == ".pth":
            kind = "checkpoint"
        elif suffix == ".index":
            kind = "index"
        elif suffix == ".onnx":
            kind = "onnx"
        elif name in {"readme.md", "license", "license.txt", "copying", "notice"}:
            kind = "notice"
        else:
            continue
        result.append({**file, "kind": kind})
    return result


def candidate_warnings(license_name: str, files: list[dict]) -> list[str]:
    warnings = []
    if not license_name or str(license_name).lower() in {"unknown", "none"}:
        warnings.append("license is not declared in hub metadata")
    if not any(file["kind"] == "checkpoint" for file in files):
        warnings.append("no exported .pth Synth checkpoint found")
    if any(file["kind"] == "onnx" for file in files):
        warnings.append("ONNX requires an explicit RVC manifest and shape validation")
    return warnings


def download(source: dict, record: dict, cache: Path) -> Path:
    safe_path(record["path"])
    if source["provider"] == "hf":
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(source["repo"], record["path"], revision=source["revision"],
                                    cache_dir=str(cache / "hf")))
    return ms_api().download_file(source["repo"], "model", record["path"],
                                  revision=source["revision"], cache_dir=cache / "ms",
                                  expected_sha256=record.get("sha256"))


def lock_source(selection: dict, cache: Path) -> dict:
    info = repository(selection["provider"], selection["repo"], selection.get("revision"))
    wanted = selection["files"]
    available = {f["path"]: f for f in info.pop("files")}
    locked = []
    for path in wanted:
        record = dict(available[safe_path(path)])
        if not record["sha256"]:
            local = download(info, record, cache)
            record.update(sha256=sha256(local), size=local.stat().st_size)
        if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ValueError(f"missing SHA-256 for {path}")
        locked.append(record)
    return {**selection, **info, "files": locked}


def fetch(source: dict, record: dict, cache: Path, seeds: list[Path]) -> Path:
    destination = cache / "verified" / record["sha256"] / Path(record["path"]).name
    if matches(destination, record):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Local files are merely seeds: accept them only after full size/hash checks.
    for seed in seeds:
        if matches(seed, record):
            shutil.copyfile(seed, destination)
            return destination
    downloaded = download(source, record, cache)
    if not matches(downloaded, record):
        raise ValueError(f"source changed or download corrupted: {source['repo']}/{record['path']}")
    shutil.copyfile(downloaded, destination)
    return destination
