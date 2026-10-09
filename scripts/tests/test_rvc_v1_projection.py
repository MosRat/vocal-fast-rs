"""The v1 head must retain its learned values and match the exported embedder."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("acoustic_converter", Path(__file__).parents[1] / "convert-acoustic-onnx-to-vfrm.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class V1ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.shared = torch.ones(768, 512, dtype=torch.float32)
        self.head = torch.arange(256 * 768, dtype=torch.float32).reshape(256, 768)
        self.state = {"feature_projection.projection.weight": self.shared,
                      "final_proj.weight": self.head, "final_proj.bias": torch.ones(256)}
        exported = np.ascontiguousarray(self.shared.numpy().T)
        self.tensors = [("hubert.model.feature_projection.projection.weight", 5, [512, 768], exported.tobytes())]

    def convert(self):
        with patch("torch.load", return_value=self.state):
            return converter.hubert_projection(Path("checkpoint.bin"), self.tensors)

    def test_canonical_head_is_transposed_and_named(self):
        weight, bias = self.convert()
        self.assertEqual(weight[:3], ("hubert.model.final_proj.weight", 5, [768, 256]))
        self.assertEqual(bias[:3], ("hubert.model.final_proj.bias", 5, [256]))
        np.testing.assert_array_equal(np.frombuffer(weight[3], np.float32).reshape(768, 256), self.head.numpy().T)

    def test_mixed_checkpoint_or_missing_head_is_rejected(self):
        self.state["feature_projection.projection.weight"] = self.shared * 2
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.convert()
        self.state["feature_projection.projection.weight"] = self.shared
        del self.state["final_proj.bias"]
        with self.assertRaisesRegex(ValueError, "final_proj"):
            self.convert()


if __name__ == "__main__":
    unittest.main()
