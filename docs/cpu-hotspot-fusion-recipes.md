# CPU 热点融合与滚动窗口量化

后续 FP32 residual 融合、真实 BRGEMM 内核选择与投影形状分派的实测见
[CPU FP32 卷积融合与投影分派](cpu-fp32-projection-fusion.md)。

## 默认与实现

默认仍为完整 FP32，新增两项默认优化：

- `CpuConfig.fuse_channel_norm = true`：HuBERT 首层按连续 64 通道分块，融合
  temporal normalization + affine。FP64 归约，保留 FP32 center、square、
  variance、epsilon、sqrt、division、affine 的运算边界。
- `CpuConfig.fast_tensor_io = true`：非连续 reshape 在编译时准备 oneDNN
  reorder；RVC relative attention 的连续末轴 GatherElements 按行读取 I32
  索引。支持负索引、动态越界检查，其他布局/类型走原路径。

`fuse_linear_gelu` 保持 false，attention 保持 dense/token-major。量化须显式选择。
44960 Engine 继续使用 4960 samples RMVPE + raw-Hz F0 cache。

模块：`native/normalization.cpp`、`normalization_simd.cpp`、`tensor_io.cpp`。
reorder、scratch 和绑定复用到 replay。profile 增加图阶段和规范化 Linear
名称，避免不同图中相同 node ID 被合并。

## 实验合同

i5-13600K，6 threads，spin=0，进程限制在 P 核逻辑处理器 0–11（包括 SMT）。
单个 default-v2 / 40 kHz 声音，index=0，pitch=0，RMS mix=1。
测试源区间 `[208000,465600)`，16.1 秒单声道 16 kHz；校准仅用 offset
32000/112000 的 44960 窗口，最晚在 156960 结束，不与测试输入重叠。
这是同一原始语料的不同区间，尚未验证多声音/跨语料泛化。

44960 / 4000 advance：HuBERT 140、RMVPE 32、Synth 281 frames，skip 250。
65 blocks 中前 13 块不计时，52 块计时，包含低音量尾段。每块播放 250 ms。
当前 Engine 默认不做启动静音；早期实验使用 13 块静音。trace 将实际启动静音
和计时预热分别记录，参考按 trace 选择，整 WAV 验证不丢弃当前入口的启动输出。
同进程交替执行控制与候选；诊断、node profile 另行执行。`process` 包含
随机输入生成与图间 IO，Engine 另含重采样、SOLA、缓冲，不含音频设备。
按连续一秒块 bootstrap，未删除离群值。不同作业的均值有漂移，不能跨行相减。

## 隔离验证结果

最新默认配方（`speech_safe_v1`，启动静音关闭）：

| 配方 | 同进程原 FP32 → 候选 ms | 加速率 [95% 区间] | 实际 Engine mean/P95/max ms |
|---|---:|---:|---:|
| 默认 FP32 norm + IO | 227.41 → 196.73 | 1.156× [1.136,1.176] | 191.87 / 214.94 / 230.09 |
| 实验 HuBERT mixed W8A8 + norm + IO | 256.93 → 192.06 | 1.338× [1.312,1.366] | 178.71 / 203.53 / 224.12 |

两项实际 Engine 均为 0/52 超过 250 ms。此轮各自的控制漂移明显，量化相对
新默认 FP32 的独立增益尚不能用这两行均值相减得出。

当前边界下的 mixed W8A8：整 WAV cosine **0.916445**、RMSE **0.0563036**；
固定同一 SOLA 偏移为 **0.999290** / **0.00519240**，62/65 块偏移不同。
模型尾部 cosine **0.999292** / RMSE **0.00512685**，coarse mismatch 0，
诊断重建与实际 Engine PCM16 完全一致。当前边界减轻了旧版的衔接差异，但
量化仍未通过整 WAV 严格对齐，继续保留实验状态。

默认完整 65 块对原始 FP32 模型 + 当前边界参考：整 WAV cosine
**0.999999978699**、RMSE **3.60412e-5**、max abs **0.000895962**；coarse
F0 与独立 SOLA 偏移 mismatch 均为 **0**。诊断重建与实际 Engine PCM16
逐字节一致。参考模型始终是原始 HF HuBERT/Applio 权重，边界是当前 Rust
SOLA 的独立 NumPy 实现，并非声称 Applio 整个实时 DSP 与本项目相同。

`cpu-recipes-final` 曾把 13 块计时预热误标为启动静音；已根据该运行编译的
Engine 默认 `warmup=false` 修正 trace 元数据并重做参考。原 WAV、计时及
binary 指纹没有改变，修正原因保留在 `metadata_repair`。

下表是分别隔离优化的早期测量：

| 实验 | 控制 → 候选 ms | 加速率及 95% 区间 |
|---|---:|---:|
| 滚动 FP32 → channel norm 融合 | 216.64 → 199.45 | 1.086× [1.068,1.104] |
| 已融合 FP32 → 再启用 tensor IO | 197.07 → 185.82 | 1.061× [1.050,1.070] |
| 已融合 FP32 → IO + HuBERT attention W8A8 | 200.24 → 177.05 | 1.131× [1.121,1.141] |

第二行实际 Engine mean/P95/max = 193.06/215.66/232.28 ms，52 块未超预算。
这些运行使用旧版公共 SOLA；工作区随后更新为 `speech_safe_v1`，最新边界的
验证单独记录，避免混用 WAV 对齐结论。

固定窗口补充实验是 `process_exact`、完整窗口 F0，32 对/四个 held-out 窗口；
含图间 IO，不含 RNG/SOLA/音频设备，不能与上表混用：

| samples | 原 FP32 → norm + IO ms | 加速率及区间 | 原生 FP32 音频 max abs |
|---:|---:|---:|---:|
| 11200 | 198.31 → 191.17 | 1.037× [1.018,1.056] | 0 |
| 44960 | 273.98 → 231.55 | 1.183× [1.164,1.202] | 0 |

独立 node profile：首层两个 temporal Mean 合计约 16–18 ms，融合 norm
约 1.9–2.3 ms；Synth reshape 6.33→0.16 ms，GatherElements 6.86→0.53 ms。
下一步主要是 Synth residual Conv 与 HuBERT extractor Conv 1/2；140 帧
HuBERT attention 约 3 ms。arena 几乎不变，生命周期复用已覆盖这些临时张量。

## 量化与衔接敏感性

新增 Rust presets `HubertAttention`（49 个 projection、Q/K/V/out Linear）
与 `HubertMixed`（再加 12 个 FFN intermediate_dense，共 61 个）。
Q/K 无 Smooth，V/out、feature projection α=0.5，FFN input α=0.7。
激活每层静态 S8、权重每输出通道 S8、S32 累加、输出/bias FP32；实测 VNNI
INT8 内核。RMVPE intermediate 在 4960 窗口下主要节省 packed 权重。

61 Linear + norm 的滚动配对延迟 215.73→174.43 ms，但原始 FP32 Python
参考（HF HuBERT + Applio RMVPE/Synth、TF32 关闭、相同噪声）发现：

| 指标 | mixed W8A8 |
|---|---:|
| 模型音频尾部 cosine / RMSE | 0.999265945 / 0.00532833 |
| Synth 独立接相同输入 cosine | 0.999999981 |
| coarse F0 mismatch | 0 |
| 12 个低音量尾部最小 cosine | 0.974167 |
| 旧边界独立 SOLA 整 WAV cosine | 0.133613 |
| 固定相同 SOLA 偏移整 WAV cosine | 0.999299 |
| SOLA 偏移不同的 chunk | 48/52 |

量化误差会改变周期信号的 SOLA 偏移和后续缓冲。固定偏移只隔离模型误差，
不能作为实际 Engine 整 WAV parity。49 Linear 的旧边界整 WAV cosine
也只有 0.516211，因此两种量化均保留实验选项。

Python 对 extractor Conv 1/2 的 W8A8 快筛：无 Smooth、α=0.5、α=0.7 均失败。
α=0.5 的最小音频 cosine 0.0445、speech 最小 0.93585、平均 RMSE 0.02777；
α=0.7 为 0.74767/0.93940/0.02500。α=0.5 activation clipped fraction 仅约
1.16e-7，不能只凭无饱和接受。保留 extractor FP32，不扩大 Synth INT8。
Linear/GELU + INT8 尚未证明稳定的额外收益，保持独立开关。

## 接入与复现

`CpuEngine::from_spec` 自动采用默认 FP32 优化。量化用
`CpuPipeline::from_sources_preset`，或完整校准 recipe 调用新增
`CpuEngine::from_sources_quantized`，复用来源验证、音频边界与生命周期。
调用者绑定模型/图设置/校准语料；本轮用滚动图重新观察校准，未借用旧完整窗口地图。

```powershell
rtk cargo build -p vocal-engine --features cpu-graph --example cpu_streaming_benchmark -p vocal-cpu --example quantization_benchmark
rtk proxy .venv/Scripts/python.exe scripts/benchmark-cpu-streaming-recipes.py --directory target/cpu-streaming-recipes --binary-dir target/debug/examples --recipes fp32_io hubert_mixed_io --profile --trace
rtk proxy .venv/Scripts/python.exe scripts/benchmark-cpu-recipes.py --directory target/cpu-fusion-shapes --recipes tensor_io --iterations 32
```

大 trace 建议保存到空间充足的磁盘。脚本记录 binary、模型、fixture、input、
规则、线程/affinity、校准 recipe SHA256，resume 拒绝变化。
`--baseline-channel-norm` 单独验证 IO 收益。
`run-python-streaming-reference.py` 独立运行原始 FP32 模型，版本化边界选择
旧版或 speech-safe NumPy 参考；`validate-cpu-streaming-recipes.py` 配对输入、
噪声和控制，分别比较原始 Python WAV 和原生 FP32 trace。

本机详细产物在 `%LOCALAPPDATA%/vocal-fast/validation/` 的
`cpu-rolling-fusion`、`cpu-rolling-io`、`cpu-rolling-io-isolated`、
`cpu-fusion-shapes`、`cpu-extractor-quant`、`cpu-recipes-final`。
模型、trace 和二进制不进入 Git。

验证：CPU/Engine feature tests 75 passed / 1 ignored，workspace 229 passed /
14 ignored，workspace all-features check、Rust fmt check、Python py_compile
通过；独立边界参考的 onset、release、noise power、quiet offset 四个测试通过。
