"""CPU-compatible W8A8 accuracy simulation: scalar activation/output-channel weight.

All dot products remain FP32 in this screening module. SmoothQuant uses a real
input-channel multiply; group/deconvolution weight axes follow PyTorch layouts.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
import torch
from torch import nn
from torch.nn import functional as F

KINDS = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)
TRANSPOSE = (nn.ConvTranspose1d, nn.ConvTranspose2d)


@dataclass
class Calibration:
    channel_max: torch.Tensor | None = None
    samples: list[torch.Tensor] = field(default_factory=list)
    calls: int = 0

    def capture(self, layer, x):
        axis = x.ndim - 1 if isinstance(layer, nn.Linear) else 1
        values = x.detach().abs().movedim(axis, 0).reshape(x.shape[axis], -1)
        maxima = values.amax(1).cpu()
        self.channel_max = maxima if self.channel_max is None else torch.maximum(self.channel_max, maxima)
        # Bounded, deterministic samples for outlier diagnosis and optional clip.
        step = max(1, values.shape[1] // max(8, 32768 // values.shape[0]))
        self.samples.append(values[:, ::step].cpu())
        self.calls += 1


def grouped_weight(layer):
    w = layer.weight.detach()
    if isinstance(layer, nn.Linear):
        return w.unsqueeze(0)
    g, ci, co = layer.groups, layer.in_channels // layer.groups, layer.out_channels // layer.groups
    if isinstance(layer, TRANSPOSE):
        return w.reshape(g, ci, co, *w.shape[2:]).transpose(1, 2)
    return w.reshape(g, co, ci, *w.shape[2:])


class FakeW8A8(nn.Module):
    def __init__(self, layer, calibration, alpha=None, percentile=1.0, dynamic=False, activation_min=-127):
        super().__init__()
        self.layer = layer
        self.dynamic = dynamic
        if activation_min not in (-128, -127):
            raise ValueError("activation_min must be -128 or -127")
        self.activation_min = activation_min
        for name in ["kernel_size", "stride", "padding", "dilation", "groups",
                     "in_channels", "out_channels", "in_features", "out_features"]:
            if hasattr(layer, name):
                setattr(self, name, getattr(layer, name))
        w = grouped_weight(layer)
        input_max = w.abs().amax(dim=(1, *range(3, w.ndim))).reshape(-1).clamp_min(1e-8)
        amax = calibration.channel_max.to(w.device).clamp_min(1e-8)
        smooth = torch.ones_like(amax) if alpha is None else amax.pow(alpha) / input_max.pow(1-alpha)
        smooth /= smooth.log().mean().exp()
        self.register_buffer("smooth", smooth)
        scaled = w * smooth.reshape(w.shape[0], 1, w.shape[2], *([1]*(w.ndim-3)))
        scale = scaled.abs().amax(dim=tuple(range(2, w.ndim)), keepdim=True).clamp_min(1e-8) / 127
        quantized = (scaled / scale).round().clamp(-127, 127) * scale
        if isinstance(layer, nn.Linear):
            quantized = quantized.squeeze(0)
        elif isinstance(layer, TRANSPOSE):
            quantized = quantized.transpose(1, 2).reshape_as(layer.weight)
        else:
            quantized = quantized.reshape_as(layer.weight)
        self.register_buffer("weight", quantized.contiguous())
        if percentile == 1:
            bound = (amax / smooth).max()
        else:
            values = torch.cat(calibration.samples, dim=1).to(w.device) / smooth[:, None]
            bound = torch.quantile(values.flatten(), percentile)
        self.register_buffer("activation_scale", bound.clamp_min(1e-8) / 127)
        self.clipped, self.elements = 0, 0

    def forward(self, x):
        layer = self.layer
        shape = [1] * x.ndim
        shape[-1 if isinstance(layer, nn.Linear) else 1] = -1
        value = x / self.smooth.reshape(shape)
        scale = value.abs().max().clamp_min(1e-8)/127 if self.dynamic else self.activation_scale
        self.clipped += int(((value < self.activation_min*scale) | (value > 127*scale)).sum())
        self.elements += value.numel()
        value = (value / scale).round().clamp(self.activation_min, 127) * scale
        if isinstance(layer, nn.Linear):
            return F.linear(value, self.weight, layer.bias)
        if isinstance(layer, TRANSPOSE):
            fn = F.conv_transpose1d if isinstance(layer, nn.ConvTranspose1d) else F.conv_transpose2d
            return fn(value, self.weight, layer.bias, layer.stride, layer.padding,
                      layer.output_padding, layer.groups, layer.dilation)
        if layer.padding_mode != "zeros":
            value = F.pad(value, layer._reversed_padding_repeated_twice, mode=layer.padding_mode)
            padding = 0
        else:
            padding = layer.padding
        fn = F.conv1d if isinstance(layer, nn.Conv1d) else F.conv2d
        return fn(value, self.weight, layer.bias, layer.stride, padding, layer.dilation, layer.groups)


def calibrate(model, inputs, invoke):
    stats, handles = {}, []
    for name, layer in model.named_modules():
        if isinstance(layer, KINDS):
            entry = stats[name] = Calibration()
            handles.append(layer.register_forward_pre_hook(
                lambda module, args, entry=entry: entry.capture(module, args[0])))
    try:
        for item in inputs:
            invoke(item)
    finally:
        for handle in handles:
            handle.remove()
    return {name: entry for name, entry in stats.items() if entry.calls}


@contextmanager
def quantized(model, stats, select, alpha=None, percentile=1.0, dynamic=False, activation_min=-127):
    replaced = []
    try:
        for name, layer in list(model.named_modules()):
            if name in stats and select(name, layer):
                parent_name, _, child = name.rpartition(".")
                parent = model.get_submodule(parent_name)
                fake = FakeW8A8(layer, stats[name], alpha, percentile, dynamic, activation_min)
                setattr(parent, child, fake)
                replaced.append((parent, child, layer, fake))
        yield [fake for _, _, _, fake in replaced]
    finally:
        for parent, child, layer, _ in replaced:
            setattr(parent, child, layer)


def fold_batchnorm(model):
    for child in model.children():
        fold_batchnorm(child)
    if isinstance(model, nn.Sequential):
        for i in range(len(model)-1):
            conv, bn = model[i], model[i+1]
            if isinstance(conv, (nn.Conv1d, nn.Conv2d, *TRANSPOSE)) and isinstance(bn, nn.modules.batchnorm._BatchNorm):
                model[i] = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn, transpose=isinstance(conv, TRANSPOSE))
                model[i+1] = nn.Identity()
