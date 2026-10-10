#!/usr/bin/env python3
"""Read, validate, bump, and synchronize the public application version."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


VERSION_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def normalize(raw: str) -> str:
    value = raw.strip()
    if value.startswith("v"):
        value = value[1:]
    if not VERSION_RE.fullmatch(value):
        raise ValueError(f"invalid semantic version: {raw!r}")
    return value


def files(root: Path) -> dict[str, Path]:
    return {
        "workspace": root / "Cargo.toml",
        "tauri_package": root / "apps/tauri/package.json",
        "tauri_package_lock": root / "apps/tauri/package-lock.json",
        "tauri_cargo": root / "apps/tauri/src-tauri/Cargo.toml",
        "tauri_config": root / "apps/tauri/src-tauri/tauri.conf.json",
    }


def read_versions(root: Path) -> dict[str, str]:
    paths = files(root)
    workspace = paths["workspace"].read_text(encoding="utf-8")
    workspace_match = re.search(
        r"(?ms)^\[workspace\.package\]\s.*?^version\s*=\s*\"([^\"]+)\"",
        workspace,
    )
    if workspace_match is None:
        raise ValueError("[workspace.package] version is missing")

    tauri_package = paths["tauri_package"].read_text(encoding="utf-8")
    package_match = re.search(r'(?m)^\s*"version"\s*:\s*"([^"]+)"', tauri_package)
    if package_match is None:
        raise ValueError("apps/tauri/package.json version is missing")

    package_lock = paths["tauri_package_lock"].read_text(encoding="utf-8")
    package_lock_matches = re.findall(r'(?m)^\s*"version"\s*:\s*"([^"]+)"', package_lock)
    if len(package_lock_matches) < 2:
        raise ValueError("apps/tauri/package-lock.json versions are missing")
    package_lock_versions = {normalize(value) for value in package_lock_matches[:2]}
    if len(package_lock_versions) != 1:
        raise ValueError("apps/tauri/package-lock.json versions disagree")

    tauri_cargo = paths["tauri_cargo"].read_text(encoding="utf-8")
    package_section = re.search(r"(?ms)^\[package\]\s.*?(?=^\[|\Z)", tauri_cargo)
    if package_section is None:
        raise ValueError("Tauri Cargo [package] section is missing")
    cargo_match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', package_section.group(0))
    if cargo_match is None:
        raise ValueError("apps/tauri/src-tauri/Cargo.toml package version is missing")

    tauri_config = paths["tauri_config"].read_text(encoding="utf-8")
    config_match = re.search(r'(?m)^\s*"version"\s*:\s*"([^"]+)"', tauri_config)
    if config_match is None:
        raise ValueError("tauri.conf.json version is missing")

    result = {
        "workspace": normalize(workspace_match.group(1)),
        "tauri_package": normalize(package_match.group(1)),
        "tauri_package_lock": next(iter(package_lock_versions)),
        "tauri_cargo": normalize(cargo_match.group(1)),
        "tauri_config": normalize(config_match.group(1)),
    }
    if len(set(result.values())) != 1:
        details = ", ".join(f"{key}={value}" for key, value in result.items())
        raise ValueError(f"version files disagree: {details}")
    return result


def replace_once(text: str, pattern: str, version: str, description: str) -> str:
    updated, count = re.subn(pattern, lambda match: f'{match.group(1)}"{version}"', text, count=1)
    if count != 1:
        raise ValueError(f"could not update {description}")
    return updated


def set_version(root: Path, version: str) -> None:
    version = normalize(version)
    paths = files(root)

    workspace = paths["workspace"].read_text(encoding="utf-8")
    workspace = replace_once(
        workspace,
        r'(?ms)(^\[workspace\.package\]\s.*?^version\s*=\s*)"[^"]+"',
        version,
        "Cargo workspace version",
    )
    paths["workspace"].write_text(workspace, encoding="utf-8", newline="\n")

    package = paths["tauri_package"].read_text(encoding="utf-8")
    package = replace_once(
        package,
        r'(?m)^(\s*"version"\s*:\s*)"[^"]+"',
        version,
        "Tauri npm version",
    )
    paths["tauri_package"].write_text(package, encoding="utf-8", newline="\n")

    package_lock = paths["tauri_package_lock"].read_text(encoding="utf-8")
    package_lock, count = re.subn(
        r'(?m)^(\s*"version"\s*:\s*)"[^"]+"',
        lambda match: f'{match.group(1)}"{version}"',
        package_lock,
        count=2,
    )
    if count != 2:
        raise ValueError("could not update npm lockfile versions")
    paths["tauri_package_lock"].write_text(package_lock, encoding="utf-8", newline="\n")

    cargo = paths["tauri_cargo"].read_text(encoding="utf-8")
    package_section = re.search(r"(?ms)^\[package\]\s.*?(?=^\[|\Z)", cargo)
    if package_section is None:
        raise ValueError("Tauri Cargo [package] section is missing")
    section = replace_once(
        package_section.group(0),
        r'(?m)^(version\s*=\s*)"[^"]+"',
        version,
        "Tauri Cargo package version",
    )
    cargo = cargo[: package_section.start()] + section + cargo[package_section.end() :]
    paths["tauri_cargo"].write_text(cargo, encoding="utf-8", newline="\n")

    config = paths["tauri_config"].read_text(encoding="utf-8")
    config = replace_once(
        config,
        r'(?m)^(\s*"version"\s*:\s*)"[^"]+"',
        version,
        "Tauri configuration version",
    )
    paths["tauri_config"].write_text(config, encoding="utf-8", newline="\n")


def bump(base: str, kind: str) -> str:
    normalized = normalize(base)
    major, minor, patch = (int(part) for part in normalized.split("-", 1)[0].split("+", 1)[0].split("."))
    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    if kind == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unsupported bump kind: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("read", help="print the synchronized version")
    normalize_parser = subparsers.add_parser("normalize", help="validate and print a version")
    normalize_parser.add_argument("version")
    verify_parser = subparsers.add_parser("verify", help="verify all version files")
    verify_parser.add_argument("--expected")
    set_parser = subparsers.add_parser("set", help="set every version file")
    set_parser.add_argument("version")
    bump_parser = subparsers.add_parser("bump", help="print a bumped version")
    bump_parser.add_argument("version")
    bump_parser.add_argument("kind", choices=("major", "minor", "patch"))

    args = parser.parse_args()
    try:
        if args.command == "bump":
            print(bump(args.version, args.kind))
            return 0
        if args.command == "normalize":
            print(normalize(args.version))
            return 0
        if args.command == "set":
            set_version(args.root, normalize(args.version))
            print(normalize(args.version))
            return 0

        current = next(iter(read_versions(args.root).values()))
        if args.command == "verify" and args.expected is not None:
            expected = normalize(args.expected)
            if current != expected:
                raise ValueError(f"expected {expected}, found {current}")
        print(current)
        return 0
    except (OSError, ValueError) as error:
        print(f"release-version: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
