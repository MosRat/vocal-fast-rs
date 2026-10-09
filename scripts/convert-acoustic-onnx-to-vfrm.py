"""Convert the waveform HuBERT/RMVPE ONNX exports to the acoustic VFRM ABI.

The resulting file is self contained: it stores the normalized tensor table and
an operator IR in the metadata section.  TensorRT never needs to open the ONNX
file after conversion.  ONNX is deliberately used only by this offline tool to
read external data and to validate the exported signature.

Examples::

    rtk .venv/Scripts/python.exe scripts/convert-acoustic-onnx-to-vfrm.py \
      --kind hubert --input models/shared/hubert_base/hubert_features.onnx \
      --output models/shared/hubert_base/hubert_features.vfr
    rtk .venv/Scripts/python.exe scripts/convert-acoustic-onnx-to-vfrm.py \
      --kind rmvpe --input models/shared/rmvpe/rmvpe_20231006.onnx \
      --output models/shared/rmvpe/rmvpe.vfr
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

MAGIC = b"VFRM\0\0\0\1"
HEADER = 64
ALIGNMENT = 64


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def dtype_code(value: np.ndarray) -> int:
    value = np.asarray(value)
    if value.dtype == np.uint8:
        return 1
    if value.dtype == np.int32:
        return 2
    if value.dtype == np.int64:
        return 3
    if value.dtype == np.float16:
        return 4
    if value.dtype == np.float32:
        return 5
    if value.dtype == np.dtype("bfloat16"):
        return 6
    raise ValueError(f"unsupported initializer dtype {value.dtype}")


def canonical_initializer(kind: str, source: str, value: np.ndarray, semantic: str | None = None) -> str:
    # Exported torch parameters have stable semantic names.  Graph constants
    # (val_*, onnx::*, and exporter paths) are addressed by content hash so a
    # different ONNX node numbering cannot silently change the ABI.
    if semantic:
        return f"{kind}.{semantic}.weight"
    if source.startswith("rmvpe."):
        return source
    if source.startswith("model."):
        return f"{kind}.{source}"
    digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()[:16]
    return f"{kind}.const.{digest}"


def shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    tensor = value_info.type.tensor_type
    result: list[int | str | None] = []
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def attribute(value: onnx.AttributeProto) -> object:
    if value.type == onnx.AttributeProto.INT:
        return int(value.i)
    if value.type == onnx.AttributeProto.INTS:
        return [int(x) for x in value.ints]
    if value.type == onnx.AttributeProto.FLOAT:
        return float(value.f)
    if value.type == onnx.AttributeProto.FLOATS:
        return [float(x) for x in value.floats]
    if value.type == onnx.AttributeProto.STRING:
        return value.s.decode("utf-8", errors="replace")
    if value.type == onnx.AttributeProto.STRINGS:
        return [x.decode("utf-8", errors="replace") for x in value.strings]
    # Tensor attributes are uncommon in the two shipped exports.  Preserve a
    # digest and shape instead of embedding another tensor outside the table.
    if value.type == onnx.AttributeProto.TENSOR:
        array = numpy_helper.to_array(value.t)
        return {"tensor_shape": list(array.shape), "tensor_sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
    return None


def validate(kind: str, graph: onnx.GraphProto) -> tuple[list[dict], list[dict]]:
    init_names = {x.name for x in graph.initializer}
    produced = {output for node in graph.node for output in node.output}
    inputs = [
        {"name": x.name, "shape": shape(x), "dtype": int(x.type.tensor_type.elem_type)}
        for x in graph.input
        if x.name not in init_names and x.name not in produced
    ]
    outputs = [{"name": x.name, "shape": shape(x), "dtype": int(x.type.tensor_type.elem_type)} for x in graph.output]
    ops = {node.op_type for node in graph.node}
    if kind == "hubert":
        if [x["name"] for x in inputs] != ["waveform"] or not any(x["name"] == "features" for x in outputs):
            raise ValueError("HuBERT must expose waveform -> features")
        required = {"Conv", "MatMul", "Softmax", "LayerNormalization"}
        if not required.issubset(ops):
            raise ValueError(f"HuBERT graph misses operators {sorted(required - ops)}")
        feature = next(x for x in outputs if x["name"] == "features")
        if len(inputs[0]["shape"]) != 2 or len(feature["shape"]) != 3 or feature["shape"][-1] != 768:
            raise ValueError("acoustic conversion requires the complete layer-12/768 HuBERT export; attach v1 final_proj with --hubert-checkpoint")
    elif kind == "rmvpe":
        names = {x["name"] for x in inputs}
        if not {"waveform", "threshold"}.issubset(names) or not any(x["name"] == "pitchf" for x in outputs):
            raise ValueError("RMVPE must expose waveform, threshold -> pitchf")
        required = {"STFT", "Conv", "GRU", "Sigmoid"}
        if not required.issubset(ops):
            raise ValueError(f"RMVPE graph misses operators {sorted(required - ops)}")
    else:
        raise ValueError(f"unsupported acoustic kind {kind!r}")
    return inputs, outputs


def hubert_projection(checkpoint: Path, tensors: list) -> list:
    """Recover the trained RVC v1 head omitted by v2 ONNX feature exports.

    Accept HF (including RVC's custom HF model) or fairseq checkpoints. Check
    a shared learned parameter against the export to prevent mixing embedders.
    """
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    hf = "feature_projection.projection.weight" in state
    projection_key = "feature_projection.projection.weight" if hf else "post_extract_proj.weight"
    if projection_key not in state or "final_proj.weight" not in state or "final_proj.bias" not in state:
        raise ValueError("HuBERT checkpoint must contain the learned final_proj weight and bias")
    exported = next((t for t in tensors if t[0] == "hubert.model.feature_projection.projection.weight"), None)
    expected = state[projection_key].float().numpy().T
    if exported is None or exported[2] != list(expected.shape) or not np.array_equal(
        np.frombuffer(exported[3], dtype="<f4").reshape(expected.shape), expected
    ):
        raise ValueError("HuBERT checkpoint feature projection does not match ONNX export; use the same checkpoint")
    weight = state["final_proj.weight"].float().numpy()
    bias = state["final_proj.bias"].float().numpy()
    if weight.shape != (256, 768) or bias.shape != (256,) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("unsupported or non-finite HuBERT v1 final_proj layout")
    return [("hubert.model.final_proj.weight", 5, [768, 256], np.ascontiguousarray(weight.T, dtype="<f4").tobytes()),
            ("hubert.model.final_proj.bias", 5, [256], np.ascontiguousarray(bias, dtype="<f4").tobytes())]


def convert(kind: str, source: Path, output: Path, hubert_checkpoint: Path | None = None) -> None:
    model = onnx.load(str(source), load_external_data=True)
    if model.opset_import and any(op.domain and op.domain != "ai.onnx" for op in model.opset_import):
        raise ValueError("custom ONNX domains are not accepted by acoustic VFRM")
    graph = model.graph
    inputs, outputs = validate(kind, graph)

    aliases: dict[str, str] = {}
    # Linear weights in the HuBERT export are anonymous val_* initializers.
    # Their following bias Add has the stable Transformers parameter name.
    semantic_by_initializer: dict[str, str] = {}
    for index, node in enumerate(graph.node):
        if node.op_type != "MatMul" or len(node.output) != 1:
            continue
        node_output = node.output[0]
        for consumer in graph.node[index + 1 : index + 4]:
            if node_output not in consumer.input or consumer.op_type != "Add":
                continue
            for bias in consumer.input[1:]:
                if bias.startswith("model.") and bias.endswith(".bias"):
                    semantic_by_initializer[node.input[1]] = bias[:-5]
                    break
            break
    tensors: list[tuple[str, int, list[int], bytes]] = []
    tensor_map = []
    raw_values = {
        item.name: np.ascontiguousarray(numpy_helper.to_array(item)).astype(np.float32, copy=False)
        for item in graph.initializer
    }
    for initializer in graph.initializer:
        value = np.asarray(numpy_helper.to_array(initializer))
        if value.dtype == np.float64:
            value = value.astype(np.float32)
        if kind == "hubert" and initializer.name.endswith(".parametrizations.weight.original1"):
            base = initializer.name.removesuffix(".parametrizations.weight.original1")
            gain = raw_values.get(base + ".parametrizations.weight.original0")
            if gain is not None:
                # Hubert applies weight_norm(dim=2): the norm is over the
                # output and input channel axes for each kernel position.
                # Folding over (1,2) would produce a per-output-channel norm
                # and silently change the positional embedding scale.
                norm = np.linalg.vector_norm(value.astype(np.float32), axis=(0, 1), keepdims=True)
                value = value.astype(np.float32) * gain / np.maximum(norm, np.finfo(np.float32).tiny)
                semantic_by_initializer[initializer.name] = base
        value = np.ascontiguousarray(value)
        name = canonical_initializer(kind, initializer.name, value, semantic_by_initializer.get(initializer.name))
        aliases[initializer.name] = name
        if any(name == x[0] for x in tensors):
            # Exporters often materialize the same scalar constant more than
            # once.  One canonical tensor is enough; all source names retain
            # an alias to it in the graph IR.
            continue
        tensors.append((name, dtype_code(value), list(value.shape), value.tobytes(order="C")))
        tensor_map.append({"canonical": name, "source": initializer.name, "sha256": hashlib.sha256(value.tobytes()).hexdigest()})

    if kind == "hubert":
        layers = {int(t[0].split(".encoder.layers.")[1].split(".")[0])
                  for t in tensors if ".encoder.layers." in t[0]}
        if layers != set(range(12)):
            raise ValueError("HuBERT template requires exactly twelve canonical encoder blocks")
        # The v2 export contains all 12 encoder layers, but never executes or
        # stores the v1 head. Attach its canonical parameters for handwritten
        # builders; retain the original exported operator IR as provenance.
        checkpoint = hubert_checkpoint or source.parent / "pytorch_model.bin"
        if checkpoint.is_file():
            for name, code, dims, data in hubert_projection(checkpoint, tensors):
                existing = next((t for t in tensors if t[0] == name), None)
                if existing is not None:
                    if existing != (name, code, dims, data):
                        raise ValueError(f"export and checkpoint disagree on {name}")
                    continue
                tensors.append((name, code, dims, data))
                tensor_map.append({"canonical": name, "source": f"{checkpoint.name}#final_proj.{name.rsplit('.', 1)[-1]}",
                                   "sha256": hashlib.sha256(data).hexdigest()})
        elif hubert_checkpoint is not None:
            raise FileNotFoundError(hubert_checkpoint)

    nodes = []
    for index, node in enumerate(graph.node):
        nodes.append({
            "id": index,
            "name": node.name or f"{node.op_type}_{index}",
            "op": node.op_type,
            "inputs": [aliases.get(x, x) for x in node.input],
            "outputs": list(node.output),
            "attrs": {x.name: attribute(x) for x in node.attribute},
        })

    manifest = {
        "id": f"{kind}-base",
        "display_name": kind.upper(),
        "architecture": f"{kind}-native-v1",
        "version": "v1",
        "sample_rate": 16000,
        "model_file": output.name,
        "index_file": None,
        "source": "offline-onnx-conversion",
        "notes": "Self-contained acoustic VFRM bundle; ONNX is not needed at runtime.",
    }
    metadata = {
        "source_format": "onnx-acoustic",
        "canonical_schema": "acoustic-v1",
        "kind": kind,
        "model_kind": kind,
        "graph_schema": f"{kind}-native-v1",
        "canonical_layout": "contiguous-row-major-little-endian",
        "inputs": inputs,
        "outputs": outputs,
        "opset": [{"domain": x.domain, "version": int(x.version)} for x in model.opset_import],
        "node_count": len(nodes),
        "nodes": nodes,
        "tensor_map": tensor_map,
        "conversion": {"source": str(source), "external_data_loaded": True},
    }
    if kind == "hubert":
        metadata["rvc_feature_modes"] = [
            {"version": version, "output_layer": layer, "feature_width": width,
             "projection": "hubert.model.final_proj" if version == "v1" else None}
            for version, layer, width in ([("v1", 9, 256), ("v2", 12, 768)]
                if any(t[0] == "hubert.model.final_proj.weight" for t in tensors) else [("v2", 12, 768)])
        ]
    config = json.dumps({"manifest": manifest, "metadata": metadata}, separators=(",", ":"), ensure_ascii=False).encode()
    directory_len = sum(8 + len(name.encode()) + 8 * len(shape_) + 16 for name, _, shape_, _ in tensors)
    data_offset = align(HEADER + len(config) + directory_len)
    offsets: list[int] = []
    cursor = data_offset
    for _, _, _, data in tensors:
        cursor = align(cursor)
        offsets.append(cursor)
        cursor += len(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = bytearray(struct.pack("<8sIIQQQQQQ", MAGIC, 0, HEADER, len(config), len(tensors), directory_len, data_offset, 0, 0))
    raw += config
    for (name, code, shape_, data), offset in zip(tensors, offsets):
        encoded = name.encode()
        raw += struct.pack("<IBBH", len(encoded), code, len(shape_), 0) + encoded
        if shape_:
            raw += struct.pack(f"<{len(shape_)}Q", *shape_)
        raw += struct.pack("<QQ", offset, len(data))
    raw += b"\0" * (data_offset - len(raw))
    for (_, _, _, data), offset in zip(tensors, offsets):
        raw += b"\0" * (offset - len(raw)) + data
    # A failed conversion or an in-use file must not truncate a runtime asset.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".vfrm.tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"converted {kind}: {len(nodes)} nodes, {len(tensors)} tensors, {len(raw)} bytes -> {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["hubert", "rmvpe"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hubert-checkpoint", type=Path,
                        help="HF/fairseq checkpoint with trained v1 final_proj; defaults to adjacent pytorch_model.bin")
    args = parser.parse_args()
    if args.hubert_checkpoint is not None and args.kind != "hubert":
        parser.error("--hubert-checkpoint is only valid for HuBERT")
    convert(args.kind, args.input, args.output, args.hubert_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
