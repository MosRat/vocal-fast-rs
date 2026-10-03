"""Compare native RMVPE stage max values with the full ONNX export."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
ONNX = ROOT / "models/shared/rmvpe/rmvpe_20231006.onnx"
AUDIO = ROOT / "target/compare/sine-6400.f32"
NATIVE = Path(os.environ.get("RMVPE_NATIVE_DEBUG", str(ROOT / "target/trt-pipeline-stage0/trt-debug.json")))


def stats(x: np.ndarray) -> dict[str, float | list[int]]:
    x = np.asarray(x, dtype=np.float32)
    return {
        "shape": list(x.shape),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "min": float(np.min(x)),
        "rms": float(np.sqrt(np.mean(x * x))),
    }


def main() -> None:
    model = onnx.load(str(ONNX), load_external_data=True)
    selected = [
        (i, n.output[0])
        for i, n in enumerate(model.graph.node)
        if n.op_type == "AveragePool" and "/unet/encoder/layers." in n.name
    ]
    selected += [(i, n.output[0]) for i, n in enumerate(model.graph.node) if n.name.endswith("encoder/layers.0/conv.0/Add")]
    selected += [(i, n.output[0]) for i, n in enumerate(model.graph.node) if n.name == "/rmvpe/mel_extractor/Log"]
    selected += [(i, n.output[0]) for i, n in enumerate(model.graph.node) if n.name.endswith("encoder/bn/BatchNormalization")]
    selected += [(i, n.output[0]) for i, n in enumerate(model.graph.node) if "/unet/intermediate/layers.3" in n.name and n.op_type == "Add"][-1:]
    values = {x: helper.make_tensor_value_info(x, TensorProto.FLOAT, None) for _, x in selected}
    del model.graph.output[:]
    model.graph.output.extend(values.values())
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    audio = np.fromfile(AUDIO, dtype="<f4").reshape(1, -1)
    outputs = session.run(list(values), {"waveform": audio, "threshold": np.array([0.03], np.float32)})
    def time_max(value: np.ndarray) -> list[float]:
        axes = (1, 3) if value.ndim == 4 else tuple(range(1, value.ndim - 1))
        return np.max(value, axis=axes).reshape(-1).tolist()
    ort_stats = {f"node_{i}_{name}": {**stats(value), "time_max": time_max(value)} for (i, name), value in zip(selected, outputs)}
    native = json.loads(NATIVE.read_text())
    print(json.dumps({"ort": ort_stats, "native_stage0": stats(np.asarray(native["pitchf"], np.float32))}, indent=2))


if __name__ == "__main__":
    main()
