"""Assemble only verified VFRM artifacts and source notices into a publish tree."""
from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .sources import fetch, matches, safe_path, sha256

ROOT = Path(__file__).resolve().parents[2]
COLLECTION_SCHEMA = "vocal-vfrm-collection-v2"
GENERATOR_VERSION = "vocal-fast-model-collection/2"
LOCAL_UPLOAD_CACHE = ".ms_upload_cache"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_fingerprint(source: dict) -> str:
    """Identify the reviewed source inputs that affect conversion output."""
    return stable_digest({key: source.get(key) for key in (
        "provider", "repo", "revision", "license", "files", "voices", "components")})


def reusable_entry(previous: dict, entries_key: str, identity: str, source_id: str,
                   fingerprint: str, path: Path) -> dict | None:
    for entry in previous.get(entries_key, []):
        if (entry.get("id", entry.get("kind")) == identity
                and entry.get("source") == source_id
                and entry.get("sourceFingerprint") == fingerprint
                and matches(path, {"size": entry.get("size"), "sha256": entry.get("sha256")})):
            return dict(entry)
    return None


def command(cli: Path, *args: str) -> dict:
    result = subprocess.run([str(cli), "model", *map(str, args)], capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"model {args[0]} failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def public_inspection(info: dict) -> dict:
    info.pop("path", None)
    for component in info["components"].values():
        if isinstance(component, dict):
            component.pop("source", None)
    return info


def prepare(lock: dict, output: Path, cache: Path, cli: Path, seed_root: Path | None,
            shared: dict[str, Path], bundle: str | None, update: bool = False,
            prune: bool = False) -> None:
    if output.exists() and not update:
        raise ValueError("output directory already exists; choose a new collection directory")
    output.mkdir(parents=True, exist_ok=update)
    previous = {}
    if update and (output / "catalog.json").is_file():
        previous = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    elif update and any(output.iterdir()):
        raise ValueError("--update requires an existing catalog.json or an empty directory")
    seed_files = list(seed_root.rglob("*")) if seed_root else []
    by_name: dict[str, list[Path]] = {}
    for file in seed_files:
        if file.is_file():
            by_name.setdefault(file.name, []).append(file)
    lock_digest = stable_digest(lock)
    catalog = {
        "schema": COLLECTION_SCHEMA,
        "catalogVersion": 2,
        "repository": lock["repository"],
        "generatorVersion": GENERATOR_VERSION,
        "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "sourceLockSha256": lock_digest,
        "defaultSharedDependencies": ["shared/hubert.vfrm", "shared/rmvpe.vfrm"],
        "voices": [],
        "shared": [],
        "bundles": [],
    }
    for source in lock["sources"]:
        source_id = safe_path(source["id"])
        if "/" in source_id:
            raise ValueError("source id must be one path segment")
        fingerprint = source_fingerprint(source)
        local = {}
        for record in source["files"]:
            verified = fetch(source, record, cache, by_name.get(Path(record["path"]).name, []))
            # Keep repository-relative layout for ONNX external tensor files.
            staged = cache / "staged" / source_id / safe_path(record["path"])
            staged.parent.mkdir(parents=True, exist_ok=True)
            if not matches(staged, record):
                shutil.copyfile(verified, staged)
            local[record["path"]] = staged
            record_name = Path(record["path"]).name.lower()
            if (Path(record["path"]).suffix.lower() in (".md", ".txt")
                    or record_name in {"license", "copying", "notice"}
                    or "license" in record_name):
                notice = output / "provenance" / source_id / safe_path(record["path"])
                notice.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local[record["path"]], notice)
        provenance = {key: source[key] for key in ("provider", "repo", "revision", "license", "files")}
        write_json(output / "provenance" / source_id / "source.json", provenance)
        for voice in source.get("voices", []):
            voice_id = safe_path(voice["id"])
            if "/" in voice_id or any(v["id"] == voice_id for v in catalog["voices"]):
                raise ValueError("voice ids must be unique path segments")
            target = output / "voices" / f"{voice_id}.vfrm"
            reused = reusable_entry(previous, "voices", voice_id, source_id, fingerprint, target)
            if reused:
                reused["sourceRevision"] = source.get("revision")
                catalog["voices"].append(reused)
                print(f"reused {voice_id}: unchanged locked source", flush=True)
                continue
            info_path = cache / "metadata" / f"{voice_id}.json"
            write_json(info_path, {"author": source["repo"].split("/")[0],
                                  "language": voice.get("language", ""), "license": source["license"],
                                  "tags": ["rvc", "vfrm", source["provider"]]})
            args = ["convert", str(local[voice["checkpoint"]]), str(target),
                    "--id", voice_id, "--name", voice["name"], "--source", source_url(source),
                    "--provenance", str(output / "provenance" / source_id / "source.json"),
                    "--library-info", str(info_path)]
            if voice.get("index"):
                args += ["--index", str(local[voice["index"]])]
            if voice.get("manifest"):
                args += ["--manifest", str(local[voice["manifest"]])]
            info = public_inspection(command(cli, *args))
            command(cli, "verify", target)
            catalog["voices"].append({"id": voice_id, "file": target.relative_to(output).as_posix(),
                                       "source": source_id, "sourceRevision": source.get("revision"),
                                       "sourceFingerprint": fingerprint, "license": source["license"], **info})
            print(f"converted {voice_id}: {info['manifest']['version']} {info['manifest']['sample_rate']} Hz", flush=True)
        for component in source.get("components", []):
            kind = component["kind"]
            target = output / "shared" / f"{kind}.vfrm"
            target.parent.mkdir(parents=True, exist_ok=True)
            reused = reusable_entry(previous, "shared", kind, source_id, fingerprint, target)
            if reused:
                reused["sourceRevision"] = source.get("revision")
                catalog["shared"].append(reused)
                print(f"reused shared {kind}: unchanged locked source", flush=True)
                continue
            if kind in shared:
                # Seed VFRMs must be accompanied by a locked digest of their actual contents.
                expected = component.get("seed_vfrm_sha256")
                if not expected or sha256(shared[kind]) != expected:
                    raise ValueError(f"{kind} VFRM seed does not match its locked digest")
                shutil.copyfile(shared[kind], target)
            else:
                if component["recipe"] == "hubert-hf-export":
                    work = cache / "acoustic-export" / kind
                    work.mkdir(parents=True, exist_ok=True)
                    for name in component["inputs"]:
                        shutil.copyfile(local[name], work / Path(name).name)
                    onnx = work / "hubert_features.onnx"
                    subprocess.run([sys.executable, str(ROOT / "scripts/export-hubert-onnx.py"),
                                    "--model", str(work), "--output", str(onnx), "--opset", "18"], check=True)
                    checkpoint = work / "pytorch_model.bin"
                else:
                    onnx = local[component["input"]]
                    checkpoint = None
                args = [sys.executable, str(ROOT / "scripts/convert-acoustic-onnx-to-vfrm.py"),
                        "--kind", kind, "--input", str(onnx), "--output", str(target)]
                if checkpoint:
                    args += ["--hubert-checkpoint", str(checkpoint)]
                subprocess.run(args, check=True)
            info = public_inspection(command(cli, "verify", target))
            if not info["components"][kind]["present"]:
                raise ValueError(f"wrong acoustic component: {kind}")
            catalog["shared"].append({"kind": kind, "file": target.relative_to(output).as_posix().replace("\\", "/"),
                                       "source": source_id, "sourceRevision": source.get("revision"),
                                       "sourceFingerprint": fingerprint, "license": source["license"], **info})
            print(f"verified shared {kind}", flush=True)
    if bundle:
        voice = next(v for v in catalog["voices"] if v["id"] == bundle)
        target = output / "bundles" / f"{bundle}-complete.vfrm"
        target.parent.mkdir(parents=True, exist_ok=True)
        voice_path = output / voice["file"]
        shared_paths = [output / "shared/hubert.vfrm", output / "shared/rmvpe.vfrm"]
        dependency_hashes = [sha256(voice_path), *(sha256(path) for path in shared_paths)]
        old_bundle = next((entry for entry in previous.get("bundles", [])
                           if entry.get("id") == bundle), None)
        if (old_bundle and old_bundle.get("dependencySha256") == dependency_hashes
                and matches(target, {"size": old_bundle.get("size"), "sha256": old_bundle.get("sha256")})):
            catalog["bundles"].append(dict(old_bundle))
            print(f"reused bundle {bundle}: dependencies unchanged", flush=True)
        else:
            info = public_inspection(command(cli, "bundle", voice_path, *shared_paths, target))
            command(cli, "verify", target)
            catalog["bundles"].append({"id": bundle, "file": target.relative_to(output).as_posix(),
                                        "componentIds": [bundle, "hubert", "rmvpe"],
                                        "dependencySha256": dependency_hashes,
                                        "licenses": [voice["license"], *[c["license"] for c in catalog["shared"]]], **info})
    if prune and update:
        desired = {entry["file"] for entry in catalog["voices"] + catalog["shared"] + catalog["bundles"]}
        for root in (output / "voices", output / "shared", output / "bundles"):
            if not root.is_dir():
                continue
            for path in root.glob("*.vfrm"):
                if path.relative_to(output).as_posix() not in desired:
                    path.unlink()
    catalog["voices"].sort(key=lambda entry: entry["id"])
    catalog["shared"].sort(key=lambda entry: entry["kind"])
    catalog["bundles"].sort(key=lambda entry: entry["id"])
    write_json(output / "sources.lock.json", lock)
    write_json(output / "catalog.json", catalog)
    write_json(output / "configuration.json", {"framework": "other", "task": "voice-conversion", "format": "VFRM"})
    (output / ".gitattributes").write_text("*.vfrm filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    lines = ["---", "license: other", "tags:", "- rvc", "- vfrm", "- voice-conversion", "---", "",
             "# VFRM model collection", "", "RVC weights converted with [vocal-fast-rs](https://github.com/MosRat/vocal-fast-rs).",
             "", "Each artifact retains its upstream license; conversion does not relicense weights.",
             "Original model cards and license notices are preserved under `provenance/`.",
             "Catalog v2 records the source lock digest, immutable source revisions, dependency hashes and per-file SHA-256 values.",
             "", "| Voice | RVC | Rate | Retrieval | License | Source |", "|---|---|---:|---|---|---|"]
    for voice in catalog["voices"]:
        m = voice["manifest"]
        lines.append(f"| [{m['display_name']}]({voice['file']}) | {m['version']} | {m['sample_rate']} | "
                     f"{voice['components']['retrieval']['present']} | {voice['license']} | {m['source']} |")
    lines += ["", "`voices/` contains Synth plus optional retrieval, without duplicating acoustic weights.",
              "`shared/hubert.vfrm` and `shared/rmvpe.vfrm` are loaded once per engine.",
              "`bundles/` contains self-contained GUI-ready voices. A bundle carries the licenses of all its components.",
              "", "## GUI usage", "",
              "Recommended: download a file from `bundles/` into the GUI model library.",
              "Select it in Studio, choose Automatic, CPU, DirectML or TensorRT-RTX, and click Load.",
              "A complete bundle automatically extracts HuBERT/RMVPE into the local cache and prepares the backend.",
              "No ONNX file, PTH file, pipeline JSON or manual acoustic path is needed.",
              "",
              "For a smaller voice-only download, copy one file from `voices/` into the model library.",
              "The GUI offers the shared dependencies on first load, downloads them once, verifies",
              "their catalog hashes, and creates a cached bound model. Manual acoustic paths remain",
              "available in Advanced settings for offline or custom registries.",
              "The optional embedded retrieval index is used automatically when present.",
              "",
              "The component badges in the GUI show `Synth`, `HuBERT`, `RMVPE` and `Index`.",
              "Only a complete bundle shows all required components as embedded and enables",
              "automatic extraction and plan preparation.",
              "", "## CLI usage", "", "```text", "vocal-fast model verify model.vfrm",
              "vocal-fast model bundle voice.vfrm hubert.vfrm rmvpe.vfrm complete.vfrm", "```", "",
              "CPU, DirectML and TensorRT-RTX use the same canonical tensor names and layouts.",
              "ONNX/PTH are conversion inputs only; runtime inference uses VFRM.",
              "", "Reproduce using `scripts/model-collection.py build` in vocal-fast-rs and this `sources.lock.json`.",
              "For maintenance, rerun with `--update`; unchanged locked artifacts are reused. Add `--prune`",
              "only when obsolete generated voice/bundle files should be removed from the local tree.",
              "See the project's `docs/model-collection.md` for commands and dependencies.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    files = [p for p in output.rglob("*") if p.is_file()
             and p.relative_to(output).as_posix() not in {
                 LOCAL_UPLOAD_CACHE, "inventory.json", "SHA256SUMS"
             }]
    for path in files:
        relative = path.relative_to(output).as_posix()
        for entry in catalog["voices"] + catalog["shared"] + catalog["bundles"]:
            if entry["file"] == relative:
                entry.update(sha256=sha256(path), size=path.stat().st_size)
    write_json(output / "catalog.json", catalog)
    inventory = {p.relative_to(output).as_posix(): {"sha256": sha256(p), "size": p.stat().st_size}
                 for p in sorted(files)}
    write_json(output / "inventory.json", inventory)
    (output / "SHA256SUMS").write_text("".join(f"{r['sha256']}  {name}\n" for name, r in inventory.items()), encoding="utf-8")
    verify(output, cli)


def migrate_catalog(lock: dict, output: Path) -> None:
    """Upgrade an existing v1 distribution without touching VFRM payloads."""
    if not output.is_dir() or not (output / "catalog.json").is_file():
        raise ValueError("migration requires an existing distribution with catalog.json")
    previous = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    if previous.get("repository") != lock.get("repository"):
        raise ValueError("migration lock repository differs from catalog repository")
    inventory = verify(output)
    sources = {source["id"]: source for source in lock.get("sources", [])}
    catalog = {
        "schema": COLLECTION_SCHEMA,
        "catalogVersion": 2,
        "repository": lock["repository"],
        "generatorVersion": GENERATOR_VERSION,
        "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "sourceLockSha256": stable_digest(lock),
        "defaultSharedDependencies": ["shared/hubert.vfrm", "shared/rmvpe.vfrm"],
        "voices": [], "shared": [], "bundles": [],
    }
    for entry in previous.get("voices", []):
        source = sources.get(entry.get("source"))
        if source is None:
            raise ValueError(f"catalog source is not present in lock: {entry.get('source')}")
        item = dict(entry)
        item["sourceRevision"] = source.get("revision")
        item["sourceFingerprint"] = source_fingerprint(source)
        catalog["voices"].append(item)
    for entry in previous.get("shared", []):
        source = sources.get(entry.get("source"))
        if source is None:
            raise ValueError(f"catalog source is not present in lock: {entry.get('source')}")
        item = dict(entry)
        item["sourceRevision"] = source.get("revision")
        item["sourceFingerprint"] = source_fingerprint(source)
        catalog["shared"].append(item)
    for entry in previous.get("bundles", []):
        item = dict(entry)
        voice_id = item.get("id")
        voice = next((v for v in catalog["voices"] if v.get("id") == voice_id), None)
        shared = {v.get("kind"): v for v in catalog["shared"]}
        dependencies = [voice.get("file")] if voice else []
        dependencies.extend(shared[k].get("file") for k in ("hubert", "rmvpe") if k in shared)
        item["dependencySha256"] = [inventory[path]["sha256"] for path in dependencies if path in inventory]
        catalog["bundles"].append(item)
    catalog["voices"].sort(key=lambda entry: entry["id"])
    catalog["shared"].sort(key=lambda entry: entry["kind"])
    catalog["bundles"].sort(key=lambda entry: entry["id"])
    write_json(output / "catalog.json", catalog)
    readme = output / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        text = text.replace("Catalog, pinned source revisions and SHA-256 digests are provided for reproducible verification.",
                            "Catalog v2 records the source lock digest, immutable source revisions, dependency hashes and per-file SHA-256 values.")
        text = text.replace("For a smaller voice-only download, copy one file from `voices/` plus\n`shared/hubert.vfrm` and `shared/rmvpe.vfrm` into the model workspace.\nSelect the voice in Studio, then choose the shared files in the backend's\nAdvanced acoustic/source settings before loading. A voice-only file without\nthese acoustic sources cannot run the full conversion pipeline by itself.",
                            "For a smaller voice-only download, copy one file from `voices/` into the model library.\nThe GUI offers the shared dependencies on first load, downloads them once, verifies\ntheir catalog hashes, and creates a cached bound model. Manual acoustic paths remain\navailable in Advanced settings for offline or custom registries.")
        if "--update" not in text:
            text += "\n## Maintenance\n\nThe directory uses catalog schema v2. Run `scripts/model-collection.py build --update` to reuse unchanged locked artifacts.\n"
        readme.write_text(text, encoding="utf-8")
    files = [path for path in output.rglob("*") if path.is_file()
             and path.name not in {"inventory.json", "SHA256SUMS", LOCAL_UPLOAD_CACHE}]
    inventory = {path.relative_to(output).as_posix(): {"sha256": sha256(path), "size": path.stat().st_size}
                 for path in sorted(files)}
    write_json(output / "inventory.json", inventory)
    (output / "SHA256SUMS").write_text("".join(f"{record['sha256']}  {name}\n"
                                               for name, record in inventory.items()), encoding="utf-8")
    verify(output)


def source_url(source: dict) -> str:
    domain = "huggingface.co" if source["provider"] == "hf" else "modelscope.cn/models"
    return f"https://{domain}/{source['repo']}"


def verify(output: Path, cli: Path | None = None) -> dict:
    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    expected = set(inventory) | {"inventory.json", "SHA256SUMS"}
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
              and p.relative_to(output).as_posix() != LOCAL_UPLOAD_CACHE}
    if actual != expected or any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("collection has unexpected/missing files or symlinks")
    sums = []
    for name, record in inventory.items():
        path = output / safe_path(name)
        if not matches(path, record):
            raise ValueError(f"collection checksum mismatch: {name}")
        sums.append(f"{record['sha256']}  {name}\n")
        if name.endswith(".vfrm"):
            if cli:
                command(cli, "verify", path)
        elif path.stat().st_size < 4 * 1024 * 1024:
            text = path.read_text(encoding="utf-8", errors="replace")
            import re
            if re.search(r"ms-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|hf_[A-Za-z0-9]{20,}", text):
                raise ValueError(f"credential pattern found: {name}")
    if (output / "SHA256SUMS").read_text(encoding="utf-8") != "".join(sums):
        raise ValueError("SHA256SUMS does not match inventory")
    return inventory
