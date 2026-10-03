# Synth 精度敏感性分析

`scripts/synth-precision-applio.py` 使用 Applio 的 `Synthesizer` 作为外部
FP32 标准。它加载 VFRM canonical 权重和确定性的 `.vin` 输入，在 CUDA 上
逐次将一个组件包在 FP16 autocast 中，然后把组件边界恢复为 FP32。这样得到
的误差是相对于 RVC WebUI/Applio Python 实现，而不是相对于 TensorRT 实现。

## 环境

CUDA 环境使用独立的 `.venv-cuda`，PyTorch wheel 来自南大镜像：

```powershell
rtk uv venv .venv-cuda --python 3.12
rtk uv pip install --python .venv-cuda\Scripts\python.exe `
  --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu130 `
  torch torchvision torchaudio numpy
rtk .venv-cuda\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

`pyproject.toml` 中的 `tool.uv.sources` 也固定了相同的 `cu130` index。

## 运行

先从 11200-sample 的真实 pipeline debug 输出生成 Synth-only fixture：

```powershell
rtk .venv\Scripts\python.exe scripts\make-synth-vin.py `
  --debug target\trt-h16-r32-s32-11200\trt-debug.json `
  --output target\compare\sine-11200.vin
```

然后运行敏感性矩阵：

```powershell
rtk .venv-cuda\Scripts\python.exe scripts\synth-precision-applio.py `
  --vin target\compare\sine-11200.vin `
  --output target\compare\synth-precision-applio-11200.json `
  --warmups 10 --iterations 30
```

同时生成同名 `.md` 汇总。矩阵包括 TextEncoder、flow、NSF source、decoder
输入/上采样/noise/resblocks/post，以及 decoder 全 FP16 和全链路 FP16。

## 当前 RTX 3060 结果

当前结果保存在
`target/compare/synth-precision-applio-11200.json`。Applio FP32 的单次
Synth 平均约 30.29 ms。单组件相对误差的排序以 waveform RMSE/relative-L2
为准：`decoder_post` 最小，`flow` 与 `decoder_source/noise` 较小；
`encoder`、`decoder_resblocks`、`decoder_full` 和全链路 FP16 误差更大。

全链路 FP16 的当前结果为：max abs `9.588e-4`、RMSE `2.034e-4`、
relative L2 `1.506e-3`。所有变体均为 finite。这个测试只评价 Synth；
RMVPE 保持 FP32，HuBERT 的 FP16 结论由单独的 frontend 测试记录。
