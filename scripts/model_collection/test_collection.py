"""Fast tests for integrity and output boundaries, without hub/network access."""
import tempfile
import unittest
from pathlib import Path

from .pipeline import verify, write_json
from .sources import matches, safe_path, sha256


class IntegrityTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
