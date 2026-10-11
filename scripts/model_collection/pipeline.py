"""Assemble only verified VFRM artifacts and source notices into a publish tree."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from .sources import fetch, matches, safe_path, sha256

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
            shared: dict[str, Path], bundle: str | None) -> None:
    if output.exists():
        raise ValueError("output directory already exists; choose a new collection directory")
    output.mkdir(parents=True)
    seed_files = list(seed_root.rglob("*")) if seed_root else []
    by_name: dict[str, list[Path]] = {}
    for file in seed_files:
        if file.is_file():
            by_name.setdefault(file.name, []).append(file)
    catalog = {"schema": "vocal-vfrm-collection-v1", "repository": lock["repository"],
               "voices": [], "shared": [], "bundles": []}
    for source in lock["sources"]:
        source_id = safe_path(source["id"])
        if "/" in source_id:
            raise ValueError("source id must be one path segment")
        local = {}
        for record in source["files"]:
            verified = fetch(source, record, cache, by_name.get(Path(record["path"]).name, []))
            # Keep repository-relative layout for ONNX external tensor files.
            staged = cache / "staged" / source_id / safe_path(record["path"])
            staged.parent.mkdir(parents=True, exist_ok=True)
            if not matches(staged, record):
                shutil.copyfile(verified, staged)
            local[record["path"]] = staged
            if Path(record["path"]).suffix.lower() in (".md", ".txt") or "license" in Path(record["path"]).name.lower():
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
                                       "source": source_id, "license": source["license"], **info})
            print(f"converted {voice_id}: {info['manifest']['version']} {info['manifest']['sample_rate']} Hz", flush=True)
        for component in source.get("components", []):
            kind = component["kind"]
            target = output / "shared" / f"{kind}.vfrm"
            target.parent.mkdir(parents=True, exist_ok=True)
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
            catalog["shared"].append({"kind": kind, "file": target.relative_to(output).as_posix(),
                                       "source": source_id, "license": source["license"], **info})
            print(f"verified shared {kind}", flush=True)
    if bundle:
        voice = next(v for v in catalog["voices"] if v["id"] == bundle)
        target = output / "bundles" / f"{bundle}-complete.vfrm"
        target.parent.mkdir(parents=True, exist_ok=True)
        info = public_inspection(command(cli, "bundle", output / voice["file"], output / "shared/hubert.vfrm",
                                         output / "shared/rmvpe.vfrm", target))
        command(cli, "verify", target)
        catalog["bundles"].append({"id": bundle, "file": target.relative_to(output).as_posix(),
                                    "componentIds": [bundle, "hubert", "rmvpe"],
                                    "licenses": [voice["license"], *[c["license"] for c in catalog["shared"]]], **info})
    write_json(output / "sources.lock.json", lock)
    write_json(output / "catalog.json", catalog)
    write_json(output / "configuration.json", {"framework": "other", "task": "voice-conversion", "format": "VFRM"})
    (output / ".gitattributes").write_text("*.vfrm filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    lines = ["---", "license: other", "tags:", "- rvc", "- vfrm", "- voice-conversion", "---", "",
             "# VFRM model collection", "", "RVC weights converted with [vocal-fast-rs](https://github.com/MosRat/vocal-fast-rs).",
             "", "Each artifact retains its upstream license; conversion does not relicense weights.",
             "Original model cards and license notices are preserved under `provenance/`.",
             "Catalog, pinned source revisions and SHA-256 digests are provided for reproducible verification.",
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
              "For a smaller voice-only download, copy one file from `voices/` plus",
              "`shared/hubert.vfrm` and `shared/rmvpe.vfrm` into the model workspace.",
              "Select the voice in Studio, then choose the shared files in the backend's",
              "Advanced acoustic/source settings before loading. A voice-only file without",
              "these acoustic sources cannot run the full conversion pipeline by itself.",
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
              "See the project's `docs/model-collection.md` for commands and dependencies.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    files = [p for p in output.rglob("*") if p.is_file()]
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


def source_url(source: dict) -> str:
    domain = "huggingface.co" if source["provider"] == "hf" else "modelscope.cn/models"
    return f"https://{domain}/{source['repo']}"


def verify(output: Path, cli: Path | None = None) -> dict:
    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    expected = set(inventory) | {"inventory.json", "SHA256SUMS"}
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
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
