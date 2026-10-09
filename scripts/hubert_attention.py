"""FP32 bidirectional tiled online attention, used for correctness screening.

This is a PyTorch prototype, not a CPU latency implementation. It visits every
key for every query; no causal mask or cross-window KV cache is introduced.
"""
from contextlib import contextmanager
from types import MethodType
import torch


def online_attention(q, k, v, scale=None, query_tile=16, key_tile=32):
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4
            or q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3]
            or q.shape[-1] != k.shape[-1] or min(query_tile, key_tile) <= 0
            or min(q.shape[-2], k.shape[-2]) == 0
            or any(x.dtype != torch.float32 for x in (q, k, v))):
        raise ValueError("expected FP32 [batch, heads, time, channels] Q/K/V")
    scale = q.shape[-1] ** -.5 if scale is None else scale
    out = torch.empty((*q.shape[:-1], v.shape[-1]), device=q.device, dtype=q.dtype)
    for start in range(0, q.shape[-2], query_tile):
        qi = q[..., start:start+query_tile, :]
        maximum = torch.full(qi.shape[:-1], -torch.inf, device=q.device)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros((*qi.shape[:-1], v.shape[-1]), device=q.device)
        for key in range(0, k.shape[-2], key_tile):
            logits = (qi @ k[..., key:key+key_tile, :].transpose(-2, -1)) * scale
            new_max = torch.maximum(maximum, logits.amax(-1))
            old_scale = (maximum-new_max).exp()
            probabilities = (logits-new_max[..., None]).exp()
            numerator = numerator * old_scale[..., None] + probabilities @ v[..., key:key+key_tile, :]
            denominator = denominator * old_scale + probabilities.sum(-1)
            maximum = new_max
        out[..., start:start+query_tile, :] = numerator / denominator[..., None]
    return out


@contextmanager
def online_hubert(model, query_tile=16, key_tile=32):
    """Replace only HF HuBERT eager attention math; keep all projections FP32."""
    saved = []
    def forward(self, hidden_states, key_value_states=None, attention_mask=None,
                layer_head_mask=None, output_attentions=False, **kwargs):
        if (self.training or key_value_states is not None or attention_mask is not None
                or layer_head_mask is not None or output_attentions):
            raise ValueError("prototype supports unmasked evaluation self-attention only")
        batch, frames, _ = hidden_states.shape
        q, k, v = [proj(hidden_states).view(batch, frames, self.num_heads, self.head_dim).transpose(1, 2)
                   for proj in (self.q_proj, self.k_proj, self.v_proj)]
        context = online_attention(q, k, v, self.scaling, query_tile, key_tile)
        return self.out_proj(context.transpose(1, 2).reshape(batch, frames, -1)), None, None
    try:
        for block in model.encoder.layers:
            saved.append((block.attention, block.attention.forward))
            block.attention.forward = MethodType(forward, block.attention)
        yield
    finally:
        for module, method in saved:
            module.forward = method
