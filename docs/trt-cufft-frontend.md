# RMVPE cuFFT 前端

TensorRT-RTX 1.6 的插件 ABI 在 SDK 中存在，但插件库的加载、序列化和
CUDA Graph 兼容性需要逐版本验证。RMVPE 的 STFT 不依赖插件：可以把
`VOCAL_TRT_RMVPE_FFT=cufft` 作为 VFRM pipeline 的实验开关。

## 图边界

```text
waveform [1, samples]
  -> CUDA reflect-pad + periodic Hann kernel
  -> cuFFT batched R2C (1024 point, hop 160)
  -> CUDA magnitude kernel
  -> stft_magnitude [1, 513, frames]
  -> TensorRT mel basis + clamp/log
  -> RMVPE U-Net + FP32 GRU + FP32 logits/decode
```

cuFFT 只产生频谱幅度，mel basis 仍然来自 RMVPE VFRM 权重并由 TensorRT
执行，因此不会出现第二份模型权重或 ONNX 中间图。CUDA magnitude kernel
直接按 mel GEMM 所需的 `[bin, frame]` 布局输出，省去一次 TensorRT transpose。
反射边界与 PyTorch
`center=True, pad_mode="reflect"` 相同，`frames = samples / 160 + 1`。

## 使用

该模式只适用于 RMVPE VFRM 包，ONNX RMVPE 会在 pipeline 编译阶段直接报错：

```powershell
$env:VOCAL_TRT_RMVPE_FFT = "cufft"
```

构建时 nvcc 将 `native/cufft_frontend.cu` 编译成 crate 私有静态库；运行时
每个固定 sample shape 创建一个 cuFFT plan。当前 profile 会预建
`4960` 以及完整窗口的 `min/opt/max` 计划，调用期间不重建 plan，也不进行
D2H/H2D 中间拷贝。cuFFT workspace 在 plan 创建时由应用显式分配并绑定，
避免第一次 `cufftExecR2C` 把 lazy allocation 混入首帧或 CUDA Graph 捕获。

TensorRT 原生 CUDA Graph 模式下，cuFFT 前端在 TensorRT engine graph 之外、
同一 CUDA stream 上执行；Rust 显式 Graph 模式会先检查所有 TensorRT
execution context 的 `isStreamCapturable`，只有检查通过才把 cuFFT kernel、
cuFFT enqueue 和四个 TensorRT enqueue 一起捕获。cuFFT plan、scratch 和
device 输出缓冲区都在捕获之前创建。RTX 1.6.1.120 的当前实测中，外部
capture 检查因 TensorRT stage 返回不可捕获而回退到 eager；生产默认使用
TensorRT 原生 whole-graph，cuFFT 仍在同一 stream 上按序执行。

## 精度和调优边界

- STFT、mel、GRU、logits/decode 保持 FP32；RMVPE U-Net 的 FP16 仍由已有
  `VOCAL_TRT_RMVPE_ENCODER_PRECISION` / `...DECODER_PRECISION` 控制。
- magnitude kernel 直接生成 `[1,513,T]`，mel GEMM 直接消费该布局；因此
  不再为外部 FFT 路径执行 `[T,513] -> [513,T]` 的 TensorRT transpose。
- 先比较 `rmvpe_debug_mel` 和 Python/PyTorch mel 的 RMSE，再比较 F0；这
  可以把手写 DFT 的消减误差与 U-Net 精度误差分开。
- Nsight 中应分别观察 `cufftExecR2C`、`gather_window`、`magnitude_kernel`
  和 TensorRT RMVPE 区间。若 cuFFT 占用已经低于 U-Net，继续融合 STFT
  的收益有限；下一步应优化 U-Net convolution tactic 或减少 H2D。

旧的 compact DFT（默认）和 radix-2 TensorRT graph 仍保留，便于 A/B 和
没有 CUDA/cuFFT 部署时的回退。

## RTX 3060 实测

测试输入是仓库中的正弦波，RMVPE FP32、HuBERT FP16/FP32 extractor、Synth
FP32；数值参考为同一输入的 RMVPE ONNX 或 Applio Python 路径：

| profile | 前端 | mel RMSE | F0 RMSE | RMVPE CUDA event |
|---|---|---:|---:|---:|
| 11200 | compact DFT | `5.525e-2` | `4.750e-2 Hz` | `4.59 ms` |
| 11200 | cuFFT | `7.603e-4` | `2.448e-2 Hz` | `4.20 ms` |
| 44960 | cuFFT | `1.603e-4` | `4.803e-3 Hz` | `8.04 ms` |

44960 cuFFT 的粗粒度 F0 bin 与 ONNX 参考 `100%` 一致。对 Applio FP32
参考的完整 waveform 结果为 RMSE `3.721e-6`、cosine `0.999579`；Synth 只
使用 native feature/pitch 输入时 cosine 为 `0.999867`，所以剩余差异主要
来自 HuBERT 与 RVC Synth 的实现/精度，而不是 STFT 消减误差。

布局优化后的固定 44960 计划在同一 RTX 3060 上、5 次 warmup 后 20 次测量为：
RMVPE CUDA event `8.02 ms`，其中外部前端 `0.014 ms`；完整阶段总和为
`62.78 ms`（HuBERT `14.02 ms`、Synth `36.29 ms`，GPU 时钟状态会影响
HuBERT/Synth）。它与旧 `[T,513] + transpose` 计划的音频输出差异为
max abs `1.10e-5`、RMSE `1.71e-6`；差异来自 TensorRT tactic/时钟重选，
而不是布局语义变化。Applio FP32 对比的 Synth RMSE 为 `1.896e-6`。

## 不采用插件的替代策略

| 部分 | 当前实现 | 保留原因 |
| --- | --- | --- |
| 反射 padding + Hann | 独立 CUDA kernel | 无插件 ABI，边界语义与 PyTorch reflect/periodic Hann 固定一致 |
| 1024 点 STFT | batched cuFFT R2C | 由 CUDA 运行库维护 tactic，FP32 数值稳定，plan/workspace 可预建 |
| magnitude | 独立 CUDA kernel，直接转置布局 | 避免 TensorRT transpose，仍输出 FP32 |
| mel 投影 | TensorRT MatrixMultiply | 权重继续来自 VFRM，避免第二份 mel 权重；GEMM 可由 TensorRT 选择 tactic |
| U-Net/GRU/decode | 手写 TensorRT 图 | 插件不参与序列化；敏感的 STFT、GRU、logits 保持 FP32 |

cuFFT FP16/cufftXt、FFT load callback 和把 mel basis 搬进自定义 CUDA kernel
均保留为后续实验项。它们会改变输入/累加精度或引入第二份权重，当前不作为
默认路径；只有在真实语音多声线的 F0 coarse-bin 和最终音频门限都通过后才应
开启。

Nsight Systems 测试可直接选择前端，不需要手动修改环境变量：

```powershell
rtk powershell -NoProfile -File scripts/profile-trt-nsight.ps1 `
  -Pipeline target/repro-cufft-44960-layout/pipeline.json `
  -Model target/default-canonical.vfr -Audio target/sine-44960.f32 `
  -Output target/nsys-cufft-44960 -RmvpeFft cufft
```

该脚本仍使用 TensorRT-RTX 原生 graph 和 node-level CUDA graph tracing；
`cufftExecR2C`、`gather_window`、`magnitude_kernel` 会和 RMVPE 的 TensorRT
kernel 一起出现在同一条 CUDA stream 上。

### Nsight Systems 结果（RTX 3060，44960）

对该固定计划执行 5 次 warmup、20 次测量并丢弃捕获阶段后，NVTX 的稳态
stage envelope 为 HuBERT `0.352 ms`、RMVPE `1.626 ms`、frontend
`0.063 ms`、Synth `0.573 ms`。CUDA kernel 聚合中，25 次记录的三个外部
前端 kernel 总时间分别为：`gather_window 0.251 ms`、cuFFT R2C
`0.248 ms`、`magnitude_kernel 0.240 ms`，即每次约 `10 us` 量级。它们远低于
RMVPE U-Net 与 Synth convolution；继续优化 FFT 不会显著降低端到端延迟。

同一 steady replay 的内存复制总时间约 `0.640 ms`，主要是 TensorRT/输入的
H2D；因此后续优先级是固定 profile 的 tactic/timing cache、RMVPE U-Net
convolution 融合以及 Synth decoder/resblock，而不是把 FFT 改成插件或再写
一套 mel 权重。
