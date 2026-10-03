"""Convert a trusted RVC checkpoint into VFRM v1.

Usage:
  uv run --with torch --with numpy python scripts/convert-rvc-pth-to-vfrm.py \
    models/voices/default-v2 models/voices/default-v2/default.pth \
    models/voices/default-v2-converted.vfr
"""
from __future__ import annotations
import json
import struct
import sys
from pathlib import Path
import torch

MAGIC = b"VFRM\0\0\0\1"
HEADER = 64
ALIGNMENT = 64

def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

def dtype_code(tensor: torch.Tensor) -> int:
    return {torch.uint8: 1, torch.int32: 2, torch.int64: 3, torch.float16: 4, torch.float32: 5, torch.bfloat16: 6}.get(tensor.dtype, 0)

def tensor_bytes(tensor: torch.Tensor) -> bytes:
    # A byte view also handles bfloat16 without a lossy NumPy conversion.
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes(order="C")

def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    model_dir, checkpoint_path, output_path = map(Path, sys.argv[1:])
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weights = checkpoint.get("weight") if isinstance(checkpoint, dict) else None
    if not isinstance(weights, dict):
        raise ValueError("checkpoint does not contain a weight state_dict")
    tensors = []
    canonical = {}
    tensor_map = []
    consumed = set()
    feature_width = 768 if "v2" in str(manifest.get("version", "")).lower() or "v2" in str(manifest.get("architecture", "")).lower() else 256
    for source_name in sorted(weights):
        if source_name in consumed:
            continue
        value = weights[source_name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"weight {source_name!r} is not a tensor")
        name = source_name
        transform = "identity"
        source_names = [source_name]
        if source_name.endswith(".weight_g"):
            base = source_name[:-9]
            v_name = base + ".weight_v"
            if v_name not in weights:
                raise ValueError(f"{source_name} has no {v_name}")
            v = weights[v_name]
            dims = tuple(range(1, v.ndim))
            value = v * weights[source_name] / torch.linalg.vector_norm(v, dim=dims, keepdim=True).clamp_min(torch.finfo(v.dtype).tiny)
            name = base + ".weight"
            transform = "weight_norm(g,v)->weight"
            source_names = [source_name, v_name]
            consumed.update(source_names)
        elif source_name.endswith(".weight_v"):
            continue
        else:
            consumed.add(source_name)
        if name == "enc_p.emb_phone.weight":
            if list(value.shape) == [192, feature_width]:
                value = value.transpose(0, 1).contiguous()
                transform = transform + "+transpose2d" if transform != "identity" else "transpose2d"
            elif list(value.shape) != [feature_width, 192]:
                raise ValueError(f"unsupported phone projection shape: {tuple(value.shape)}")
        if not (name.startswith(("enc_p.", "dec.", "flow.")) or name == "emb_g.weight"):
            raise ValueError(f"unsupported RVC tensor: {name}")
        if name in canonical:
            raise ValueError(f"duplicate canonical tensor: {name}")
        canonical[name] = value.contiguous()
        tensor_map.append({"canonical": name, "source": source_names, "transform": transform})
    for name in sorted(canonical):
        value = canonical[name]
        code = dtype_code(value)
        if code == 0:
            raise TypeError(f"unsupported dtype for {name!r}: {value.dtype}")
        tensors.append((name, code, list(value.shape), tensor_bytes(value)))
    tensor_map.sort(key=lambda x: x["canonical"])
    cfg = checkpoint.get("config")
    if not isinstance(cfg, list) or len(cfg) != 18:
        raise ValueError("checkpoint config must contain 18 entries")
    rvc_config = {
        "feature_width": feature_width, "uses_f0": bool(checkpoint.get("f0", 1)),
        "sample_rate": int(manifest["sample_rate"]), "hidden_channels": int(cfg[3]),
        "latent_channels": int(cfg[2]), "filter_channels": int(cfg[4]),
        "attention_heads": int(cfg[5]), "encoder_layers": int(cfg[6]),
        "encoder_kernel": int(cfg[7]), "relative_window": 10, "relative_heads": 1,
        "norm_epsilon": 1e-5, "speakers": int(cfg[15]), "speaker_channels": int(cfg[16]),
        "decoder_channels": int(cfg[13]), "resblock_type": int(cfg[9]),
        "resblock_kernels": [int(x) for x in cfg[10]],
        "resblock_dilations": [[int(x) for x in row] for row in cfg[11]],
        "upsample_rates": [int(x) for x in cfg[12]],
        "upsample_kernels": [int(x) for x in cfg[14]],
        "flow_blocks": 4, "flow_layers": 3, "flow_kernel": 5,
        "flow_dilations": [1, 1, 1],
    }
    metadata = {"checkpoint": {"config": checkpoint.get("config"), "sr": checkpoint.get("sr"), "f0": checkpoint.get("f0"), "version": checkpoint.get("version"), "info": checkpoint.get("info")}, "source_format": "rvc-pytorch-checkpoint", "tensor_count": len(tensors), "canonical_schema": "rvc-synth-v1", "canonical_layout": "contiguous-row-major-little-endian", "tensor_name_namespace": "rvc.synth.v1", "graph_schema": "rvc-v2-f0-feats" if feature_width == 768 and checkpoint.get("f0", 1) else "rvc-v2-nof0-feats" if feature_width == 768 else "rvc-v1-f0-feats" if checkpoint.get("f0", 1) else "rvc-v1-nof0-feats", "canonical_tensor_count": len(tensors), "tensor_map": tensor_map, "rvc_config": rvc_config}
    config = json.dumps({"manifest": manifest, "metadata": metadata}, separators=(",", ":"), ensure_ascii=False).encode()
    directory_len = sum(8 + len(name.encode()) + 8 * len(shape) + 16 for name, _, shape, _ in tensors)
    data_offset = align(HEADER + len(config) + directory_len)
    offsets, cursor = [], data_offset
    for _, _, _, data in tensors:
        cursor = align(cursor); offsets.append(cursor); cursor += len(data)
    output = bytearray(struct.pack("<8sIIQQQQQQ", MAGIC, 0, HEADER, len(config), len(tensors), directory_len, data_offset, 0, 0))
    output += config
    for (name, code, shape, data), offset in zip(tensors, offsets):
        name_bytes = name.encode()
        output += struct.pack("<IBBH", len(name_bytes), code, len(shape), 0) + name_bytes
        output += struct.pack(f"<{len(shape)}Q", *shape) if shape else b""
        output += struct.pack("<QQ", offset, len(data))
    output += b"\0" * (data_offset - len(output))
    for (_, _, _, data), offset in zip(tensors, offsets):
        output += b"\0" * (offset - len(output)) + data
    output_path.write_bytes(output)
    print(f"converted {len(tensors)} tensors ({len(output)} bytes) to {output_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
