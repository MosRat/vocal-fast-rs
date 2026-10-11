"""Discover, lock, convert, verify and publish an RVC VFRM collection."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_collection.pipeline import migrate_catalog, prepare, verify, write_json
from model_collection.sources import lock_source, repository, safe_path, search


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    find = sub.add_parser("search", help="search public model repository names")
    find.add_argument("provider", choices=["hf", "ms", "all"])
    find.add_argument("query")
    find.add_argument("--limit", type=int, choices=range(1, 101), default=20)
    find.add_argument("--output", type=Path, required=True)
    discover = sub.add_parser("discover", help="list eligible source files and reported license")
    discover.add_argument("provider", choices=["hf", "ms"])
    discover.add_argument("repo")
    discover.add_argument("--revision")
    discover.add_argument("--output", type=Path, required=True)
    merge = sub.add_parser("merge-search", help="merge normalized search result files")
    merge.add_argument("inputs", nargs="+", type=Path)
    merge.add_argument("--output", type=Path, required=True)
    lock = sub.add_parser("lock", help="pin source revisions and hashes from a reviewed selection")
    lock.add_argument("--selection", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    lock.add_argument("--cache", type=Path, default=Path("target/model-collection/cache"))
    build = sub.add_parser("build", help="fetch and convert only locked sources")
    build.add_argument("--lock", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--cache", type=Path, default=Path("target/model-collection/cache"))
    build.add_argument("--cli", type=Path, required=True)
    build.add_argument("--seed-root", type=Path, help="reuse raw files only if their full hash matches")
    build.add_argument("--hubert-seed", type=Path)
    build.add_argument("--rmvpe-seed", type=Path)
    build.add_argument("--bundle", help="voice id to produce a complete GUI-ready bundle")
    build.add_argument("--update", action="store_true",
                       help="incrementally maintain an existing distribution directory")
    build.add_argument("--prune", action="store_true",
                       help="with --update, remove obsolete generated voice/bundle files")
    migrate = sub.add_parser("migrate", help="upgrade an existing v1 distribution manifest without rebuilding weights")
    migrate.add_argument("--lock", type=Path, required=True)
    migrate.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--cli", type=Path, required=True)
    publish = sub.add_parser("publish", help="publish a verified collection using cached or env credentials")
    publish.add_argument("--output", type=Path, required=True)
    publish.add_argument("--repo", required=True)
    publish.add_argument("--cli", type=Path, required=True)
    publish.add_argument("--prune", action="store_true",
                         help="delete remote files absent from the verified local inventory")
    publish.add_argument("--report", type=Path,
                         help="write a JSON local/remote diff report")
    publish.add_argument("--roundtrip", choices=["none", "sample", "all"], default="sample",
                         help="download and parse no, representative, or all VFRM artifacts")
    args = parser.parse_args()
    if args.command == "search":
        write_json(args.output, search(args.provider, args.query, args.limit))
    elif args.command == "merge-search":
        merged = {}
        for path in args.inputs:
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise ValueError(f"search result must be a JSON array: {path}")
            for record in records:
                key = (record.get("provider"), str(record.get("repo", "")).lower())
                if key[0] not in {"hf", "ms"} or not key[1]:
                    raise ValueError(f"invalid search candidate in {path}")
                merged[key] = record
        write_json(args.output, sorted(merged.values(), key=lambda item: (
            -int(item.get("checkpointCount", 0)), item["provider"], item["repo"])))
    elif args.command == "discover":
        result = repository(args.provider, args.repo, args.revision)
        result["files"] = [f for f in result["files"] if f["path"].lower().endswith((".pth", ".onnx", ".index"))
                           or Path(f["path"]).name.lower() in ("readme.md", "license", "license.txt", "notice")]
        write_json(args.output, result)
    elif args.command == "lock":
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        ids = [s["id"] for s in selection["sources"]]
        if len(set(ids)) != len(ids) or any("/" in safe_path(i) for i in ids):
            raise ValueError("source ids must be unique path segments")
        sources = [lock_source(s, args.cache.resolve()) for s in selection["sources"]]
        write_json(args.output, {"schema": "vocal-vfrm-sources-v1", "repository": selection["repository"], "sources": sources})
    elif args.command == "build":
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        shared = {k: v.resolve() for k, v in {"hubert": args.hubert_seed, "rmvpe": args.rmvpe_seed}.items() if v}
        prepare(lock, args.output.resolve(), args.cache.resolve(), args.cli.resolve(),
                args.seed_root.resolve() if args.seed_root else None, shared, args.bundle,
                update=args.update, prune=args.prune)
    elif args.command == "verify":
        inventory = verify(args.output.resolve(), args.cli.resolve())
        print(f"verified {len(inventory)} distribution files")
    elif args.command == "migrate":
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        migrate_catalog(lock, args.output.resolve())
        print(f"migrated {args.output} to catalog schema v2")
    else:
        from model_collection.publish import publish
        publish(args.output.resolve(), args.repo, args.cli.resolve(), prune=args.prune,
                report_path=args.report.resolve() if args.report else None,
                roundtrip=args.roundtrip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
