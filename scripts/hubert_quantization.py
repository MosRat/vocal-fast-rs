"""Research-only FP32 outlier residual + static W8A8 HuBERT Linear screening.

This needs an additional small FP32 GEMM and is not the native oneDNN recipe.
"""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F
from cpu_quantization import FakeW8A8


class OutlierLinear(FakeW8A8):
    def __init__(self, layer, calibration, alpha, fraction):
        if not isinstance(layer, nn.Linear) or not 0 < fraction < 1:
            raise ValueError("outlier residual requires Linear and fraction in (0,1)")
        super().__init__(layer, calibration, alpha=alpha, activation_min=-128)
        amax=calibration.channel_max.to(self.weight.device)
        scaled=amax/self.smooth
        ids=scaled.topk(max(1,math.ceil(layer.in_features*fraction))).indices
        self.register_buffer("protected",ids)
        self.register_buffer("residual_weight",layer.weight.detach()[:,ids].contiguous())
        w=layer.weight.detach()*self.smooth[None]
        w[:,ids]=0
        scales=w.abs().amax(1,keepdim=True).clamp_min(1e-8)/127
        self.weight=((w/scales).round().clamp(-127,127)*scales).contiguous()
        scaled[ids]=0
        self.activation_scale=scaled.max().clamp_min(1e-8)/127

    def forward(self,x):
        residual=F.linear(x[...,self.protected],self.residual_weight)
        value=x/self.smooth
        value[...,self.protected]=0
        scale=self.activation_scale
        self.clipped+=int(((value < -128*scale) | (value > 127*scale)).sum())
        self.elements+=value.numel()
        value=(value/scale).round().clamp(-128,127)*scale
        return F.linear(value,self.weight,self.layer.bias)+residual


@contextmanager
def outlier_quantized(model,stats,select,alpha=.5,fraction=.01):
    saved=[]
    try:
        for name,layer in list(model.named_modules()):
            if name in stats and select(name,layer):
                parent,_,child=name.rpartition(".")
                module=model.get_submodule(parent)
                wrapper=OutlierLinear(layer,stats[name],alpha,fraction)
                setattr(module,child,wrapper)
                saved.append((module,child,layer,wrapper))
        yield [w for _,_,_,w in saved]
    finally:
        for module,child,layer,_ in saved: setattr(module,child,layer)
