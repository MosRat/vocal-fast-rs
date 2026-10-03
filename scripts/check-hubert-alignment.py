"""Compare the handwritten HuBERT equations with the local HF reference.

This is intentionally a small CPU diagnostic.  It follows the tensor layouts
used by ``native/hubert.cpp`` and reports the first stage where the VFRM
weights diverge from ``HubertModel``.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import HubertModel


ROOT = Path(__file__).resolve().parents[1]


def read_vfrm(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    metadata_len, count, _directory_len, data_offset = struct.unpack_from("<QQQQ", raw, 16)
    cursor = 64 + metadata_len
    dtypes = {1: np.uint8, 2: np.int32, 3: np.int64, 4: np.float16, 5: np.float32}
    out: dict[str, np.ndarray] = {}
    for _ in range(count):
        name_len, dtype, rank, _reserved = struct.unpack_from("<IBBH", raw, cursor)
        cursor += 8
        name = raw[cursor : cursor + name_len].decode()
        cursor += name_len
        shape = struct.unpack_from("<" + "Q" * rank, raw, cursor)
        cursor += 8 * rank
        offset, length = struct.unpack_from("<QQ", raw, cursor)
        cursor += 16
        kind = dtypes[dtype]
        value = np.frombuffer(raw, dtype=kind, count=length // np.dtype(kind).itemsize, offset=offset)
        out[name] = value.reshape(shape).copy()
    return out


def stat(name: str, value: torch.Tensor, ref: torch.Tensor | None = None) -> None:
    value = value.detach().float().cpu()
    text = f"{name:24s} shape={tuple(value.shape)!s:16s} range=[{value.min(): .5f},{value.max(): .5f}] mean={value.mean(): .5f}"
    if ref is not None:
        ref = ref.detach().float().cpu()
        delta = value - ref
        text += f"  ref=[{ref.min(): .5f},{ref.max(): .5f}] err(max/rmse)={delta.abs().max():.6g}/{delta.square().mean().sqrt():.6g}"
    print(text)


def linear(x: torch.Tensor, weights: dict[str, np.ndarray], prefix: str) -> torch.Tensor:
    w = torch.from_numpy(weights[prefix + ".weight"]).float()
    b = torch.from_numpy(weights[prefix + ".bias"]).float()
    return x @ w + b


def main() -> None:
    weights = read_vfrm(ROOT / "models/shared/hubert_base/hubert_features.vfr")
    model = HubertModel.from_pretrained(ROOT / "models/shared/hubert_base", local_files_only=True).float().eval()
    audio = np.fromfile(ROOT / "target/compare/sine-11200.f32", dtype="<f4")
    waveform = torch.from_numpy(audio)[None]
    with torch.no_grad():
        ref_feat = model.feature_extractor(waveform)
        ref_proj_input = ref_feat.transpose(1, 2)
        ref_proj = model.feature_projection(ref_proj_input)
        ref_pos = model.encoder.pos_conv_embed(ref_proj)
        ref_x = model.encoder.layer_norm(ref_proj + ref_pos)

    x = waveform[:, None]
    for i, (kernel, stride) in enumerate(zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2))):
        w = torch.from_numpy(weights[f"hubert.model.feature_extractor.conv_layers.{i}.conv.weight"])
        x = F.conv1d(x, w, stride=stride)
        if i == 0:
            x = F.group_norm(x, 512, torch.ones(512), torch.zeros(512), 1.0e-5)
            scale = torch.from_numpy(weights["hubert.const.d8b9ba9bce6ed02c"]).reshape(1, 512, 1)
            bias = torch.from_numpy(weights["hubert.const.685ea624c050bb77"]).reshape(1, 512, 1)
            x = x * scale + bias
        x = F.gelu(x, approximate="none")
    stat("feature_extractor", x, ref_feat)

    x = x.transpose(1, 2)
    lnw = torch.from_numpy(weights["hubert.model.feature_projection.layer_norm.weight"])
    lnb = torch.from_numpy(weights["hubert.model.feature_projection.layer_norm.bias"])
    x = F.layer_norm(x, (512,), lnw, lnb, 1.0e-5)
    stat("feature_projection.ln", x, model.feature_projection.layer_norm(ref_proj_input))
    x = linear(x, weights, "hubert.model.feature_projection.projection")
    stat("feature_projection", x, ref_proj)

    pos_v = torch.from_numpy(weights["hubert.model.encoder.pos_conv_embed.conv.weight"])
    pos_g = torch.from_numpy(weights["hubert.model.encoder.pos_conv_embed.conv.parametrizations.weight.original0"])
    pos_norm = torch.sqrt((pos_v * pos_v).sum(dim=(0, 1), keepdim=True))
    pos_w = pos_v * pos_g / pos_norm
    pos_b = torch.from_numpy(weights["hubert.model.encoder.pos_conv_embed.conv.bias"])
    with torch.no_grad():
        hf_pos_conv = model.encoder.pos_conv_embed.conv(ref_proj.transpose(1, 2))
        manual_pos_conv = F.conv1d(ref_proj.transpose(1, 2), pos_w, pos_b, padding=64, groups=16)
    stat("positional.conv", manual_pos_conv, hf_pos_conv)
    pos = F.conv1d(x.transpose(1, 2), pos_w, pos_b, padding=64, groups=16)[:, :, :-1]
    pos = F.gelu(pos, approximate="none").transpose(1, 2)
    stat("positional", pos, ref_pos)
    x = F.layer_norm(x + pos, (768,), torch.from_numpy(weights["hubert.model.encoder.layer_norm.weight"]), torch.from_numpy(weights["hubert.model.encoder.layer_norm.bias"]), 1.0e-5)
    stat("encoder.input", x, ref_x)

    ref_hidden = ref_x
    for i in range(12):
        layer = model.encoder.layers[i]
        p = f"hubert.model.encoder.layers.{i}"
        ref_q = layer.attention.q_proj(ref_hidden).view(1, -1, 12, 64).transpose(1, 2)
        ref_k = layer.attention.k_proj(ref_hidden).view(1, -1, 12, 64).transpose(1, 2)
        ref_v = layer.attention.v_proj(ref_hidden).view(1, -1, 12, 64).transpose(1, 2)
        q = linear(x, weights, p + ".attention.q_proj").view(1, -1, 12, 64).transpose(1, 2)
        k = linear(x, weights, p + ".attention.k_proj").view(1, -1, 12, 64).transpose(1, 2)
        v = linear(x, weights, p + ".attention.v_proj").view(1, -1, 12, 64).transpose(1, 2)
        if i == 0:
            stat("layer0.q", q, ref_q)
            stat("layer0.k", k, ref_k)
            stat("layer0.v", v, ref_v)
        attn = (q @ k.transpose(2, 3) * (64.0 ** -0.5)).softmax(-1) @ v
        ref_attn = (ref_q @ ref_k.transpose(2, 3) * (64.0 ** -0.5)).softmax(-1) @ ref_v
        if i == 0:
            stat("layer0.context", attn, ref_attn)
        attn = attn.transpose(1, 2).reshape(1, -1, 768)
        attn = linear(attn, weights, p + ".attention.out_proj")
        if i == 0:
            stat("layer0.attention", attn, layer.attention.out_proj(ref_attn.transpose(1, 2).reshape(1, -1, 768)))
        residual = x + attn
        ffn_residual = F.layer_norm(residual, (768,), torch.from_numpy(weights[p + ".layer_norm.weight"]), torch.from_numpy(weights[p + ".layer_norm.bias"]), 1.0e-5)
        y = ffn_residual
        ref_ffn_ln = layer.layer_norm(ref_hidden + layer.attention(ref_hidden)[0])
        if i == 0:
            stat("layer0.ffn_ln", y, ref_ffn_ln)
        y = linear(y, weights, p + ".feed_forward.intermediate_dense")
        ref_intermediate = layer.feed_forward.intermediate_dense(ref_ffn_ln)
        if i == 0:
            stat("layer0.ffn_linear", y, ref_intermediate)
        y = F.gelu(y, approximate="none")
        ref_intermediate = F.gelu(ref_intermediate, approximate="none")
        if i == 0:
            stat("layer0.ffn_gelu", y, ref_intermediate)
        y = linear(y, weights, p + ".feed_forward.output_dense")
        ref_ffn = layer.feed_forward.output_dense(ref_intermediate)
        if i == 0:
            stat("layer0.ffn_out", y, ref_ffn)
            ref_residual = ref_hidden + layer.attention(ref_hidden)[0]
            stat("layer0.residual2", residual, ref_residual)
            stat("layer0.pre_final", ffn_residual + y, layer.layer_norm(ref_residual) + ref_ffn)
        x = F.layer_norm(ffn_residual + y, (768,), torch.from_numpy(weights[p + ".final_layer_norm.weight"]), torch.from_numpy(weights[p + ".final_layer_norm.bias"]), 1.0e-5)
        with torch.no_grad():
            ref_hidden = layer(ref_hidden)[0]
        stat(f"encoder.layer.{i}", x, ref_hidden)


if __name__ == "__main__":
    main()
