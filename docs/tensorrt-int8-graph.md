# TensorRT-RTX INT8 建图与实测结论

本文记录当前 TensorRT-RTX `1.6.1.120` 在 RTX 3060 上的 W8A8 结果。文中把
“图中存在 INT8 量化节点”和“卷积实际执行了 INT8 Tensor Core 指令”分开
统计；前者不能推出后者。

## 结论

当前 W8A8 **没有实测证据证明命中了 INT8 Tensor Core**。

修复前，选中的 `dec.resblocks.4.convs1.0` 进入的是：

```text
FP32 activation -> Quantize(INT8) -> Dequantize(FP32) -> FP32 convolution
```

这条路径把 Q/DQ 放在 FP32 residual 域中，卷积没有低精度输入。

修复后，图形态变为：

```text
FP16 activation -> Quantize(INT8) -> Dequantize(FP16) -> FP16 convolution
INT8 per-output-channel weight -> Dequantize(FP16) ------------^
```

这是 TensorRT explicit Q/DQ 可以识别的低精度图形态，但仍由 builder 自己
决定是否折叠 Q/DQ 并选择 INT8 tactic。当前固定 44960 plan 的 engine inspector
显示选中卷积的 activation、weight 和 output 都是 `Half`，tactic 为
`__myelin_CASK_STUB`；INT8 常量出现在相邻的 `kgen CastMulCast` 层，而不是
卷积的实际输入/输出类型中。

Nsight Systems 的固定 profile 也只出现：

```text
cudnn_generated_fort_native_sm80_convFwd_pointwise_...
cudnn_generated_convBwdDataStrided_pointwise_cutlass_sm80_...
```

没有看到能指向 INT8 的 `imma`、`dp4a` 或 `s8` kernel 名称。W8A8 Synth
延迟约为 44960 的 `40.13 ms`，FP16 baseline 约为 `36.73 ms`；增加的 Q/DQ
和 cast 开销大于可能的权重节省。因此当前应称为“explicit W8A8 candidate”，
不能称为“已启用 INT8 Tensor Core”。

这仍不是硬件指令级证明。Nsight Compute 在本机返回 `ERR_NVGPUCTRPERM`，
无法读取 IMMA 计数器。允许硬件计数器后，应同时检查：

```text
sm__inst_executed_pipe_tensor_op_imma.sum
sm__inst_executed_pipe_tensor_op_hmma.sum
```

并把计数、engine inspector 和 CUDA event 延迟一起作为判定依据。

## TensorRT-RTX 1.6 的量化粒度

以下结论来自本机 SDK 的
`include/NvInfer.h` 中 `IQuantizeLayer`/`IDequantizeLayer` 注释（约
5284--5334、5434--5488 行）：

| 对象 | 支持的粒度 | 当前结论 |
| --- | --- | --- |
| INT8 activation | per-tensor，scalar scale | 支持，当前 W8A8 使用此形式 |
| INT8 weight | per-channel，1-D scale + axis | 支持，当前按输出通道量化 |
| INT8 activation per-channel | 不支持 | 头文件明确写明 activation 不能 per-channel |
| INT8 activation block/group | 没有可依赖的 Q/DQ API | 不应把 `setBlockShape` 当作 INT8 activation group 量化 |
| FP4/FP8/INT4 | block quantization | `setBlockShape` 明确用于这些低精度类型 |
| zero point | 只支持对称量化 | zero point 必须为零 |

`IQuantizeLayer::setAxis()` 只对 per-axis 量化有意义；权重路径的 scale
长度必须等于 axis 的大小。`IDequantizeLayer` 还明确限制：per-channel
dequantization 只允许输入根在 `IConstantLayer`，也就是权重，不能用于动态
activation。

`INetworkDefinition::createNetworkV2()` 的本地文档写明 TensorRT-RTX 网络
始终 strongly typed，`kSTRONGLY_TYPED` 仅为兼容而保留并被忽略。因此不能
依赖旧的 `setPrecision(kINT8)` 或隐式类型转换来强制卷积使用 INT8。
本 SDK 的 `BuilderFlag` 也没有可强制 INT8 的 flag；`setTacticSources()`
只限制 tactic 来源，不能指定某个 INT8 tactic。

`addDynamicQuantizeV2()` 的声明（约 9323--9341 行）把 `kINT8` 列为
`outputType` 的允许值，但 `IDynamicQuantizeLayer::setToType()` 的类文档
（约 5621--5629 行）只承诺 FP4/NVFP4 与 FP8/MXFP8。旧的
`addDynamicQuantize()` 也只接受 FP4/FP8。这个文档矛盾不能作为 INT8
block activation 的依据；若要尝试，必须单独构建最小图并检查 build、
deserialize、enqueue 和 inspector 结果。当前生产图不使用它。

## 当前图的正确构造方式

### 1. 离线校准

对每一个候选 Conv 保存：

```text
activation_scale: scalar > 0
weight_scale: [out_channels] > 0
```

activation scale 必须来自多声音、多音高、清音和爆破音片段，而不是单个
正弦输入。Synth 的上采样输入存在明显 outlier，per-tensor scale 会浪费
INT8 动态范围。

### 2. 权重

对 Conv 的 canonical `[out, in, kernel]` 权重按输出通道计算：

```text
s_w[k] = max(abs(W[k, ...])) / 127
W_q[k, ...] = round(W[k, ...] / s_w[k])
```

图中放置 `INT8 Constant`，然后接 `IDequantizeLayer(axis=0,
outputType=kHALF)`。当前实现对 ConvTranspose 使用对应的输出通道 axis。

### 3. activation

对选中的 decoder Conv：

```text
x_fp16
  -> IQuantizeLayer(scale=s_a, toType=kINT8)
  -> IDequantizeLayer(scale=s_a, toType=kHALF)
  -> IConvolutionLayer
```

这是一种 explicit Q/DQ 标注图。Q/DQ 仍可能被 builder 折叠为 FP16
correlation tactic；因此必须检查生成 engine，而不是只检查网络构造代码。

当前实现的额外约束是：

* W8A8 只能在 Synth decoder FP16 域启用；
* 选中的 Conv 进入时必须是 `Half`；
* 选中的 Conv 输出必须推导为 `Half`；
* residual sum、NSF source、pitch 和最终 audio 仍按现有 FP32/FP16
  精度策略处理；
* 若 inspector 显示选中 Conv 为 `Float`，建图应直接失败，而不是继续生成
  一个容易被误解的 plan。

这些类型断言位于 `native/graph/layers.cpp`，并且 Q/DQ 层现在带有稳定名称，
便于 inspector 与回归脚本定位。

### 4. SmoothQuant 的位置

TensorRT-RTX 没有 SmoothQuant 专用 API。可在单个 Conv 前后手写等价的
ElementWise rescale：

```text
x' = x / s_channel
W' = W * s_channel
x' -> per-tensor INT8 Q/DQ -> Conv(W')
```

`s_channel` 应作为 VFRM calibration/aux tensor 保存，按模型和声音校准。
这里的 channel-wise rescale 是浮点 ElementWise 操作，不是伪造 activation
per-channel INT8；真正送入 Q/DQ 的 activation 仍只有一个 scalar scale。
应先验证一个 decoder 节点，再做累计节点和真实人声回归。

## 如何确认一个 plan 真正使用 INT8

按以下顺序检查，单项通过都不够：

1. **构造检查**：Q/DQ 存在，selected Conv 不在 FP32 域，权重为 per-channel
   INT8 constant。
2. **engine inspector**：selected Conv 的输入/权重/输出、tactic 和 format；
   如果输入和权重已经是 Half，只能说明 builder 选择了浮点执行路径。
3. **Nsight Systems**：固定 shape、warmup 后比较 W8A8 与 FP16 的 kernel
   名称、实例数和 device event 时间。
4. **Nsight Compute**：在有权限时检查 HMMA/IMMA 指令计数；没有权限时，
   只能报告“未证实”，不能把 kernel 名称当作硬件指令证明。
5. **数值门禁**：对 11200、35520、44960 及多声音模型比较 max abs、RMSE、
   relative L2、cosine 和最终音频主观/频谱质量。

可复核的当前产物：

```text
target/compare/w8a8-fixed-layer-info.json
target/compare/nsys-w8a8-fixed.nsys-rep
target/compare/nsys-fp16-synth.nsys-rep
```

## 后续可行路线

1. 先保留 FP16 Synth 作为默认最快路径，W8A8 只作 opt-in accuracy gate。
2. 加入 SmoothQuant calibration tensor，优先测试 `dec.resblocks.*` 单节点；
   不量化 upsample、NSF source、pitch 和 residual accumulation。
3. 若 SDK 后续明确支持 INT8 activation block/group，增加最小独立图验证，
   不直接把 `addDynamicQuantizeV2(kINT8)` 合入主图。
4. 在允许硬件计数器的机器上重新构建固定 11200/44960 profile，确认 IMMA
   计数和端到端收益后再扩大节点集合。

