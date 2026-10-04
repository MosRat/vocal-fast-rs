# Synth 瓶颈与优化顺序

## 当前基线

测试为 RTX 3060、44960 固定 profile、RVC v2 default 模型：

```text
RVC frames      281
upsample rates  [10, 10, 2, 2]
channels        [256, 128, 64, 32]
resblock type   1
resblock kernels [3, 7, 11]
```

cuFFT + 安全混合精度路径的 CUDA event 延迟为：

| 阶段 | 延迟 |
| --- | ---: |
| HuBERT | `4.613 ms` |
| RMVPE + cuFFT | `8.306 ms` |
| cuFFT 外部前端 | `0.014 ms` |
| Synth | `39.700 ms` |
| event 总和 | `60.239 ms` |

Synth 占 event 总和约 66%。Nsight 中 cuFFT 的三个 kernel 只有每次约 10 us
量级，Synth 的主要 kernel 是 TensorRT-RTX 选择的 cuDNN Conv1d/pointwise
族，以及大量 TensorRT fused elementwise kernel。

## 算子分解

Synth decoder 每个上采样阶段有 3 个 ResBlock。`resblock_type=1` 的一个
ResBlock 包含 3 个 `convs1` 和 3 个 `convs2`，因此 4 个阶段共 72 个卷积，
还包括 4 个 ConvTranspose、4 个 noise convolution 和最终 projection。

ResBlock 的 MAC 量与 `C² * L * sum(kernel)` 成正比。以 44960 的实际长度估算：

| stage | C | L | residual MAC 占比 |
| ---: | ---: | ---: | ---: |
| 0 | 256 | 2810 | 18.6% |
| 1 | 128 | 28100 | 46.5% |
| 2 | 64 | 56200 | 23.3% |
| 3 | 32 | 112400 | 11.6% |

因此 stage 1 是第一热点。stage 3 的时间维最长，但通道数小，更多表现为
带宽和 kernel launch 成本；它不是最高 MAC 阶段。

Encoder 和 affine flow 仍然需要执行完整 RVC 语义，但其卷积规模远小于
decoder residual path。仅把 flow 改成低精度不能解决 44960 的主要问题。

## 已验证的精度边界

Applio FP32 Synth 组件实验显示：

| 变体 | waveform relative L2 | 观察 |
| --- | ---: | --- |
| encoder FP16 | `8.79e-3` | 可见误差，收益有限 |
| flow FP16 | `3.66e-3` | 仍会累积误差 |
| decoder source FP16 | `1.71e-3` | NSF 相位敏感 |
| residual blocks FP16 | `2.38e-2` | 误差最大之一 |
| decoder 全 FP16 | `3.77e-2` | 只适合实验 |

因此当前图保留 encoder/flow/source 和 residual blocks 为 FP32，只把 decoder
边界降到 FP16。该策略在 44960 的最终 waveform cosine 为 `0.999492`；全
HuBERT FP16 的 cosine 只有约 `0.9578`，不能作为默认组合。

11200 的 residual stage sweep 也说明了速度/精度取舍：

| mask | 含义 | Synth event | relative L2 |
| ---: | --- | ---: | ---: |
| `0` | 所有 residual FP32 | `13.39 ms` | `0.00203` |
| `0x2` | stage 1 FP16 | `7.99 ms` | `0.00360` |
| `0x4` | stage 2 FP16 | `8.95 ms` | `0.00237` |
| `0x8` | stage 3 FP16 | `9.46 ms` | `0.00189` |

stage 1 是最值得尝试的低精度候选，但它同时也是误差风险最高的主计算阶段。
`0x8` 是目前更保守的速度/精度点；所有 mask 仍需在目标声音模型和真实语音
上验证后才能启用。

## 优化优先级

### 1. 固定 shape 的 tactic/timing cache

44960 和 11200 必须分别建立固定 plan。当前层 profiler 曾出现单个
`dec.resblocks.4.convs1.0` tactic 明显偏慢，说明 TensorRT tactic 选择和
首次 JIT 会影响结果。应为 `(model hash, profile, precision, mask)` 建立
持久 timing cache，并在相同 RTX 架构上复用；比较时使用 CUDA event steady
replay，不使用 NVTX enqueue envelope。

### 2. 减少 FP16/FP32 边界

当前 mask=0 的混合 decoder 在每个上采样阶段会产生 FP16→FP32→FP16 边界。
优先测试只把 stage 1 或 stage 3 的 `convs1` 降为 FP16、而 residual add 和
`convs2` 保持 FP32。这样可以减少 Tensor Core 卷积成本，同时保留累加精度；
需要增加独立的 stage/op 精度字段，不能把整个 ResBlock 粗略切换。

### 3. 融合 ResBlock 的激活、加法和平均

当前模式是：

```text
leaky_relu -> conv -> leaky_relu -> conv -> add
```

TensorRT 已经能把部分 activation 融入 cuDNN convolution，但多分支的 residual
add 和三个分支的最终平均仍可能产生独立 elementwise kernel。可行的低风险
改动是保持数学顺序，显式构造单个 `sum -> scale` 子图并检查 tactic 是否融合；
不能改变三个分支的求和顺序，否则会引入可见的 FP16 累加差异。

更激进的方案是为同一 stage 的三个分支构造 grouped/block-diagonal convolution
或专用 CUDA fused kernel。它可减少 18 次卷积之间的 launch 和中间写回，但会
复制输入或增加寄存器压力，必须与 TensorRT Conv1d tactic 做 A/B，且 RTX SDK
当前不把自定义插件作为默认依赖。

### 4. 测试 ConvTranspose 的 polyphase 改写

当前 4 个上采样使用 ConvTranspose1d。由于 `[10,10,2,2]` 的 kernel 分别为
`[16,16,4,4]`，理论上可以改写为零插入/phase-shuffle + Conv1d，或将每个
phase 合并为一次普通卷积。该形式可能获得更好的 Tensor Core tactic，但必须
先在 FP32 对齐 PyTorch 的边界、padding 和 phase 顺序，再比较固定 profile 的
速度；它不是可直接替换的 nearest resize + conv。

### 5. 真实 INT8 校准

当前 `weight_int8` 只是权重 Dequantize 图，不能假定会得到 INT8 Tensor Core
收益。要获得真正收益，需要用多声音、多音高语料校准 activation scale，并
让 stage 1/2 的卷积使用 INT8 kernel；`m_source`、noise convolution、conv_post、
residual sum 和 tanh 保持 FP32/FP16。任何 INT8 方案都必须同时检查 F0、清音、
爆破音和尾部噪声，不能只看正弦输入 cosine。

### 6. 缩短实时图输入

44960 是 WebUI/Applio 的完整上下文，Synth 计算量随输出长度近似线性增长。
如果实时场景允许更短上下文，11200 profile 的 Synth 约 10--13 ms，远低于
44960 的约 40 ms。应通过 host 侧 overlap/crossfade/SOLA 保持边界质量，不能
把 44960 动态裁成短输入后继续使用同一个 tactic profile。

## 暂不建议

- 把 flow/encoder 整体改 FP16：收益小，误差会累积。
- 把 NSF source 或 pitch 量化：会直接放大相位漂移。
- 用 generic attention 替换 RVC relative flow：改变模型语义。
- 继续优化 cuFFT：它已经远低于 Synth residual/convolution 成本。

## Fresh Nsight Systems replay and host binding cache

A new 100-replay capture of the stage-2 plan (`nsys-op16-s2-c2-cached`) used
CUDA Graph node tracing with no intermediate downloads. NVTX ranges measure the
CPU enqueue envelope; the CUDA-event section added to `summarize_nsys.py` reads
device durations from the pipeline's reusable start/end event pairs:

| stage | device mean (ms) | device p95 (ms) |
| --- | ---: | ---: |
| HuBERT | 4.90 | 5.83 |
| RMVPE + cuFFT | 9.06 | 10.06 |
| frontend | 0.016 | 0.016 |
| Synth | 29.50 | 30.14 |

Synth therefore uses about two thirds of steady device time, with RMVPE second.
The largest repeated kernels are TensorRT-RTX cuDNN pointwise convolution
tactics for Synth residual/upsample blocks, followed by grouped matmul and
Myelin fused add/multiply/max kernels. cuFFT R2C and magnitude kernels remain
below the stage noise floor.

The runtime now caches TensorRT binding addresses and resolved HuBERT/RMVPE
frame counts. A shape transition clears the address cache; fixed-profile
replays keep the same device allocations and skip those FFI calls. A debug
replay before and after this change matched audio, features, pitch and coarse
pitch bit-for-bit. This is a host-side reduction and does not change CUDA-event
stage time or numerical output.

The same capture with stage-1 FP16 residual accumulation measured HuBERT
`4.76 ms`, RMVPE `8.78 ms`, frontend `0.016 ms`, and Synth `29.13 ms` over the
last ten CUDA-event samples. The application profile reported `45.94 ms` event
total and `47.59 ms` host mean over 100 replays. Its deterministic sine gate
against the FP32 Applio reference remained final cosine `0.99888755`, Synth-only
cosine `0.99999917`, F0 RMSE `0.00681 Hz`, and coarse F0 agreement `99.644%`.
This is the current throughput candidate; the FP32-accumulation stage-2 plan
remains the safer default until real voices pass the same gate.

### Combined RMVPE decoder FP16 experiment

The combined 44960 plan used RMVPE decoder FP16 together with stage-1 FP16
residual accumulation. Against the FP32 Applio reference it produced F0 RMSE
`0.002846 Hz`, 100% coarse-F0 agreement, final cosine `0.99984818`, and
Synth-only cosine `0.99999919`. The RMVPE CUDA-event median in the fresh
Nsight capture was `6.91 ms` (about 1--1.5 ms below the FP32 decoder), but Synth
was `30.47 ms` in that capture. Repeated application profiles ranged from
`48.8` to `54.5 ms` event total, overlapping the FP32-decoder stage-1 plan once
GPU clock and tactic variance are included. Keep decoder FP16 opt-in; it is a
useful RMVPE-only saving, but it is not yet a proven end-to-end speed win.
