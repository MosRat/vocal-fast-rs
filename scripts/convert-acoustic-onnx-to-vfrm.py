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
import struct
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


def convert(kind: str, source: Path, output: Path) -> None:
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
                norm = np.linalg.vector_norm(value.astype(np.float32), axis=(1, 2), keepdims=True)
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
    output.write_bytes(raw)
    print(f"converted {kind}: {len(nodes)} nodes, {len(tensors)} tensors, {len(raw)} bytes -> {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["hubert", "rmvpe"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.kind, args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
