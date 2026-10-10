# RVC v1/v2 统一版本合同与算子融合

## 版本合同

`vocal-model::RvcVersion` 是三个执行后端共同使用的唯一版本解析入口：

| 版本 | HuBERT 输出层 | 特征宽度 | 投影 |
| --- | ---: | ---: | --- |
| v1 | 9 | 256 | `hubert.model.final_proj`，`[768, 256]` + `[256]` |
| v2 | 12 | 768 | 无 |

HuBERT Transformer 的内部隐藏宽度仍为 768；版本差异只发生在送入检索和
Synth 的输出合同。`RvcVersion::from_labels(version, architecture)` 同时
解析 manifest 和模型描述中的标签，并拒绝已知标签互相矛盾的组合。未知的
RVC 版本不会降级成 v1。

创建 CPU、DirectML 或 TensorRT 引擎时，会比较三份合同：

1. GUI/runtime 选中的 `ModelDescriptor`；
2. VFRM Synth manifest 和 `feature_width`；
3. TensorRT 计划 manifest（仅 TensorRT）。

任意一份不一致都会在建图或载入阶段报错。检索索引的维度也必须等于版本
特征宽度，因此 v1 索引不能被 v2 图复用。

## CPU 图中的融合算子

CPU planner 在分配 native buffer 前执行局部模式融合，融合只改变执行表示，
不改变 VFRM 的固定 tensor 名称和 v1/v2 ABI。

### `LinearBias`

匹配 `Add(MatMul, bias)` 或 `Add(bias, MatMul)`。planner 会把输入槽位固定为
`[matmul_a, matmul_b, bias]`，并检查：

- MatMul 输出形状等于 Add 输出形状；
- MatMul、Add、bias 的 dtype 一致；
- bias 是末维可广播的向量（允许前导 1）；
- 权重输入是二维矩阵。

### `Attention`

匹配
`MatMul(Softmax(Mul(MatMul(q, k^T), scale)), v)`，并接受 scale 在 Mul 的任一
输入侧。融合前会检查 q/k/v 的 rank、batch/head 前缀、内积宽度、softmax
形状和输出末维，且 scale 必须是有限 FP32 标量。检查失败时保留原始算子，
由通用执行路径处理。

这些守卫使 v1/v2 的宽度变化不会影响融合正确性，也避免外部导入图中错误
的广播形状被静默接受。

## 验证

```powershell
rtk cargo fmt --all -- --check
rtk cargo check --workspace --all-features --target-dir $env:LOCALAPPDATA/vocal-fast/validation/cpu-gui-speaker-cargo
rtk cargo test --workspace --all-features --target-dir $env:LOCALAPPDATA/vocal-fast/validation/cpu-gui-speaker-cargo
```

当前 workspace 回归结果：255 passed，24 ignored。
