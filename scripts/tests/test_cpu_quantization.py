"""Independent group/output-channel and smoothing contracts for CPU W8A8."""
import sys
from pathlib import Path
import unittest
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_quantization import Calibration, FakeW8A8, fold_batchnorm


class QuantizationTests(unittest.TestCase):
    def test_grouped_conv_and_deconv_output_channel_scales(self):
        torch.manual_seed(37)
        for transposed in [False, True]:
            layer = (nn.ConvTranspose1d if transposed else nn.Conv1d)(4, 6, 3, groups=2).eval()
            x = torch.randn(2, 4, 17)
            stats = Calibration(); stats.capture(layer, x)
            fake = FakeW8A8(layer, stats, alpha=0.7)
            smooth = fake.smooth
            scaled_x = x / smooth.reshape(1, 4, 1)
            quant_x = (scaled_x/fake.activation_scale).round().clamp(-127,127)*fake.activation_scale
            outputs = []
            for group in range(2):
                w = layer.weight[group*2:(group+1)*2] if transposed else layer.weight[group*3:(group+1)*3]
                w = w*smooth[group*2:(group+1)*2].reshape(2,1,1) if transposed else w*smooth[group*2:(group+1)*2].reshape(1,2,1)
                scales = w.abs().amax(dim=(0,2) if transposed else (1,2),keepdim=True).clamp_min(1e-8)/127
                wq = (w/scales).round().clamp(-127,127)*scales
                fn = F.conv_transpose1d if transposed else F.conv1d
                outputs.append(fn(quant_x[:,group*2:(group+1)*2],wq,layer.bias[group*3:(group+1)*3]))
            torch.testing.assert_close(fake(x),torch.cat(outputs,dim=1),atol=2e-6,rtol=2e-6)

    def test_linear_static_and_dynamic_scales_zero_input(self):
        layer = nn.Linear(4,3).eval()
        x = torch.arange(8,dtype=torch.float32).reshape(2,4)*0.01
        stats = Calibration(); stats.capture(layer,x)
        for dynamic in [False,True]:
            fake = FakeW8A8(layer,stats,alpha=0.5,dynamic=dynamic)
            zeros = torch.zeros_like(x)
            torch.testing.assert_close(fake(zeros),layer.bias.expand(2,3))
            self.assertTrue(torch.isfinite(fake(x)).all())

    def test_batchnorm_fold_preserves_effective_deconv_weight(self):
        model = nn.Sequential(nn.ConvTranspose2d(3,5,3),nn.BatchNorm2d(5),nn.ReLU()).eval()
        model[1].running_mean = torch.arange(5,dtype=torch.float32)*0.1
        model[1].running_var = torch.arange(5,dtype=torch.float32)+0.5
        x = torch.randn(1,3,4,4)
        expected = model(x)
        fold_batchnorm(model)
        torch.testing.assert_close(model(x),expected,atol=2e-6,rtol=2e-6)

    def test_onednn_activation_negative_saturation_contract(self):
        layer = nn.Linear(1,1,bias=False).eval()
        layer.weight.data.fill_(1)
        stats = Calibration(); stats.capture(layer,torch.ones(1,1))
        negative = torch.tensor([[-2.]])
        torch.testing.assert_close(FakeW8A8(layer,stats)(negative),torch.tensor([[-1.]]))
        torch.testing.assert_close(FakeW8A8(layer,stats,activation_min=-128)(negative),torch.tensor([[-128/127]]))


if __name__ == "__main__":unittest.main()
