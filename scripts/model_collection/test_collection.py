"""Fast tests for integrity and output boundaries, without hub/network access."""
import tempfile
import unittest
import json
from pathlib import Path

from .pipeline import COLLECTION_SCHEMA, migrate_catalog, stable_digest, verify, write_json
from .publish import remote_diff, validate_catalog
from .sources import candidate_warnings, eligible_files, matches, safe_path, search, sha256


class IntegrityTests(unittest.TestCase):
    def test_stable_digest_is_order_independent(self):
        self.assertEqual(stable_digest({"b": 2, "a": 1}), stable_digest({"a": 1, "b": 2}))

    def test_repository_paths_cannot_escape(self):
        for path in ("../secret", "/secret", "C:/secret", "a\\secret", "", "."):
            with self.assertRaises(ValueError):
                safe_path(path)

    def test_digest_rejects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "weight"
            path.write_bytes(b"abcd")
            record = {"size": 4, "sha256": sha256(path)}
            self.assertTrue(matches(path, record))
            path.write_bytes(b"abce")
            self.assertFalse(matches(path, record))

    def test_eligible_files_and_warnings(self):
        files = eligible_files([
            {"path": "voice.pth", "size": 4, "sha256": "a" * 64},
            {"path": "voice.index", "size": 4, "sha256": "b" * 64},
            {"path": "notes.bin", "size": 4, "sha256": "c" * 64},
        ])
        self.assertEqual([item["kind"] for item in files], ["checkpoint", "index"])
        self.assertEqual(candidate_warnings("unknown", files), ["license is not declared in hub metadata"])
        self.assertIn("no exported .pth Synth checkpoint found", candidate_warnings("MIT", []))

    def test_search_rejects_unknown_provider_before_network_access(self):
        with self.assertRaises(ValueError):
            search("unknown", "rvc", 1)

    def test_verify_rejects_extra_files_and_credentials(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            path = output / "README.md"
            path.write_text("safe", encoding="utf-8")
            digest = sha256(path)
            write_json(output / "inventory.json", {"README.md": {"size": 4, "sha256": digest}})
            (output / "SHA256SUMS").write_text(f"{digest}  README.md\n", encoding="utf-8")
            verify(output)
            (output / "private.env").write_text("secret", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify(output)
            (output / "private.env").unlink()
            path.write_text("hf_" + "a" * 32, encoding="utf-8")
            digest = sha256(path)
            write_json(output / "inventory.json", {"README.md": {"size": path.stat().st_size, "sha256": digest}})
            (output / "SHA256SUMS").write_text(f"{digest}  README.md\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "credential"):
                verify(output)

    def test_catalog_v2_matches_inventory(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            voice = output / "voices" / "voice.vfrm"
            voice.parent.mkdir()
            voice.write_bytes(b"vfrm")
            digest = sha256(voice)
            inventory = {"voices/voice.vfrm": {"size": 4, "sha256": digest}}
            catalog = {"schema": COLLECTION_SCHEMA, "catalogVersion": 2,
                       "voices": [{"id": "voice", "file": "voices/voice.vfrm",
                                   "size": 4, "sha256": digest}], "shared": [], "bundles": []}
            validate_catalog(output, catalog, inventory)
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_catalog(output, {**catalog, "voices": [{"file": "voices/missing.vfrm"}]}, inventory)

    def test_remote_diff_classifies_local_remote_and_changed(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            local = output / "README.md"
            local.write_bytes(b"local")
            inventory = {"README.md": {"size": 5, "sha256": sha256(local)}}

            class Remote:
                def __init__(self, size, digest):
                    self.size = size
                    self.sha256 = digest

            diff = remote_diff(output, inventory, {"README.md": Remote(4, "0" * 64),
                                                   "old.txt": Remote(1, "1" * 64)})
            self.assertEqual(diff["localOnly"], ["SHA256SUMS", "inventory.json"])
            self.assertEqual(diff["remoteOnly"], ["old.txt"])
            self.assertEqual([item["path"] for item in diff["changed"]], ["README.md"])

    def test_migrate_catalog_preserves_payloads(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            (output / "voices").mkdir()
            (output / "shared").mkdir()
            (output / "voices" / "voice.vfrm").write_bytes(b"voice")
            (output / "shared" / "hubert.vfrm").write_bytes(b"hubert")
            (output / "shared" / "rmvpe.vfrm").write_bytes(b"rmvpe")
            (output / "README.md").write_text("Catalog, pinned source revisions and SHA-256 digests are provided for reproducible verification.", encoding="utf-8")
            (output / "configuration.json").write_text("{}", encoding="utf-8")
            files = [path for path in output.rglob("*") if path.is_file()]
            inventory = {path.relative_to(output).as_posix(): {"sha256": sha256(path), "size": path.stat().st_size}
                         for path in files}
            write_json(output / "inventory.json", inventory)
            (output / "SHA256SUMS").write_text("".join(f"{v['sha256']}  {k}\n" for k, v in inventory.items()), encoding="utf-8")
            old = {"schema": "vocal-vfrm-collection-v1", "repository": "owner/models",
                   "voices": [{"id": "voice", "file": "voices/voice.vfrm", "source": "source",
                               "license": "MIT", "sha256": sha256(output / "voices/voice.vfrm"), "size": 5}],
                   "shared": [{"kind": "hubert", "file": "shared/hubert.vfrm", "source": "source",
                               "license": "MIT", "sha256": sha256(output / "shared/hubert.vfrm"), "size": 6},
                              {"kind": "rmvpe", "file": "shared/rmvpe.vfrm", "source": "source",
                               "license": "MIT", "sha256": sha256(output / "shared/rmvpe.vfrm"), "size": 5}],
                   "bundles": []}
            write_json(output / "catalog.json", old)
            # catalog.json was added after the original inventory.
            files = [path for path in output.rglob("*") if path.is_file() and path.name not in {"inventory.json", "SHA256SUMS"}]
            inventory = {path.relative_to(output).as_posix(): {"sha256": sha256(path), "size": path.stat().st_size}
                         for path in files}
            write_json(output / "inventory.json", inventory)
            (output / "SHA256SUMS").write_text("".join(f"{v['sha256']}  {k}\n" for k, v in inventory.items()), encoding="utf-8")
            lock = {"schema": "vocal-vfrm-sources-v1", "repository": "owner/models", "sources": [
                {"id": "source", "provider": "hf", "repo": "owner/model", "revision": "a" * 40,
                 "license": "MIT", "files": [], "voices": [], "components": []}]}
            migrate_catalog(lock, output)
            migrated = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
            self.assertEqual(migrated["schema"], COLLECTION_SCHEMA)
            self.assertEqual((output / "voices" / "voice.vfrm").read_bytes(), b"voice")
            self.assertEqual(verify(output)["voices/voice.vfrm"]["size"], 5)


if __name__ == "__main__":
    unittest.main()
