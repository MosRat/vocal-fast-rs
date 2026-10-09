# CPU 推理配方与当前瓶颈

后续热点融合、最新默认与滚动量化的端到端验证见
[CPU 热点融合与滚动窗口量化](cpu-hotspot-fusion-recipes.md)。

后续已经接入 44960 的最近窗口 RMVPE + F0 缓存，FP32 真实引擎平均/P95 为
209.84 / 223.89 ms；详见 [实时 F0 缓存](cpu-realtime-f0-cache.md)。
下表保留此前 **全窗口 F0** 配方实验，不能与新测量跨作业直接相除。

## 建议配置

2026-10-08，本机 i5-13600K，standalone oneDNN 3.10.2。以当前已经优化的
dense/token-major FP32 图为控制，重新测试此前的量化组合。
旧实验使用旧 attention 输出布局，不能把旧加速率直接应用到当前默认图。

| 用途 | HuBERT | RMVPE | Synth | 状态 |
|---|---|---|---|---|
| 默认，优先对齐参考 | FP32 | FP32 | FP32 | 生产默认 |
| 减少权重内存 | FP32 | intermediate W8A8，其余 FP32 | FP32 | 显式实验配方，优先考虑 |
| 平衡内存与速度 | 混合 Linear W8A8，其余 FP32 | intermediate W8A8，其余 FP32 | FP32 | 显式实验配方，需试听与校准 |
| 扩大量化 Synth | 混合 W8A8 | intermediate W8A8 | flow/noise/第三阶段 residual W8A8 | 研究候选，低音量误差仍大 |

共同设置：固定 shape、NHWC/channels-last、静态权重一次打包、primitive/scratch 复用、
AVX2/FMA、Conv/ReLU 和 residual/Leaky 融合、dense attention 直接输出 token-major。
默认最多 6 个算子线程、spin=0、由 OS 调度；线程策略仍通过 Rust 配置传入。
本机实现的浮点计算路径为 FP32，低精度加速使用真实 VNNI W8A8。
CPU 配方需要按算子与硬件选择，当前不采用 TensorRT 的 FP16 配方。

## 测量契约

- 一个 RVC v2 / 40 kHz 声音模型，一段 Mandarin 语料；不宣称跨声音模型通用。
- 精度参考：原始 Applio FP32 Synth/RMVPE + HF HuBERT FP32，关闭 TF32。
  两端独立提取特征和 F0，只共享 waveform、latent noise 和 NSF noise。
  speaker=0、index=0、自动移调关闭；没有测量带检索的性能。
- 校准 offset：32000、112000；测试：208000、352000；额外验证：272000、416000。
  每个 offset 都测试两种长度，同 offset 的不同长度有重叠，不能视为独立语料。
- 每个 shape/候选在同一进程驻留完整 FP32 和候选管线，8 对预热、32 对交替执行。
  计时包含图输入输出复制、Q/layout/Smooth reorder、frontend 与最终输出复制；
  不包含加载编译、fixture IO、随机输入生成、SOLA 和音频设备缓冲。
  诊断输出另行执行，计时关闭 per-node profile。
- 6 个线程，进程限制到 Windows 报告的 P 核逻辑处理器 0–11，包括 SMT 兄弟。
  不是每个物理核固定一个线程，也不是生产默认的 OS 调度性能保证。
- 加速区间按完整四窗口周期做 paired block bootstrap。
  跨作业 FP32 均值有漂移，只比较每行成对的加速率；区间只反映本轮波动。
- 所有低音量验证窗口保留在误差统计内。`packed` 节省是原生打包参数核算，
  不等于进程 RSS；基准同时驻留两条管线，不能把基准 RSS 当成部署占用。

## 完整链路实测

配方名与 `scripts/benchmark-cpu-recipes.py` 一致。
`balanced_stage2` 包含 RMVPE intermediate、HuBERT mixed、Synth flow、noise Conv、
generator 第三阶段的 residual；不是只量化某一个孤立 Synth 算子。

| 输入 samples | 配方 | 同进程 FP32 → 候选 ms | 加速率 [95% 区间] | 候选 P95 ms | packed 节省 MiB |
|---:|---|---:|---:|---:|---:|
| 11200 | rmvpe_intermediate | 216.49 → 212.00 | 1.021× [0.991, 1.054] | 232.02 | 221.81 |
| 11200 | memory_attention | 216.35 → 208.69 | 1.037× [1.011, 1.063] | 226.03 | 303.79 |
| 11200 | balanced | 218.49 → 205.62 | 1.063× [1.027, 1.103] | 228.63 | 384.65 |
| 11200 | balanced_stage2 | 201.34 → 175.51 | 1.147× [1.111, 1.185] | 204.10 | 404.63 |
| 44960 | rmvpe_intermediate | 283.26 → 276.59 | 1.024× [0.991, 1.060] | 308.85 | 221.81 |
| 44960 | memory_attention | 299.10 → 287.97 | 1.039× [0.968, 1.085] | 311.76 | 303.79 |
| 44960 | balanced | 288.81 → 255.15 | 1.132× [1.101, 1.168] | 282.51 | 384.65 |
| 44960 | balanced_stage2 | 293.34 → 245.48 | 1.195× [1.160, 1.232] | 271.31 | 404.63 |

`memory_attention` 增加 HuBERT Q/K/V/out 和 feature projection 的 W8A8；
`balanced` 再加入全部 FFN 的 intermediate_dense，保留 FFN output_dense 为 FP32。
实际记录的整数内核为 `brg_conv_fwd:avx2_vnni`、`brgconv_1x1:avx2_vnni`、
`brg_matmul:avx2_vnni`，不是反量化后用 FP32 点积估算的加速。
`memory_attention/44960` 最大值为 528.84 ms，未剔除该离群值。
单独 RMVPE 量化的速度区间跨过 1，因此主要把它当作内存优化。

### 最终音频相对 Applio 的误差

cosine 为四个测试/验证窗口的最小值，RMSE 为逐窗口均值，max abs 为全窗口最大值。

| samples | 配方 | 最小 cosine | 平均 RMSE | max abs |
|---:|---|---:|---:|---:|
| 11200 | rmvpe_intermediate | 0.999999972 | 3.55e-5 | 0.000231 |
| 11200 | memory_attention | 0.999638799 | 0.002474 | 0.117027 |
| 11200 | balanced | 0.999670297 | 0.003368 | 0.072657 |
| 11200 | balanced_stage2 | 0.999627477 | 0.003700 | 0.070242 |
| 44960 | rmvpe_intermediate | 0.999999812 | 2.36e-5 | 0.000148 |
| 44960 | memory_attention | 0.999726635 | 0.001614 | 0.070267 |
| 44960 | balanced | 0.998758754 | 0.001521 | 0.049067 |
| 44960 | balanced_stage2 | 0.860796520 | 0.001821 | 0.055006 |

所有候选的 coarse F0 和 voicing mismatch 都是 0，F0 浮点最大差异约 0.000412 Hz。
原生 FP32 对 Applio、量化对原生 FP32 的误差也分别保存在报告内。
44960 / offset 416000 是低音量尾段，参考 RMS 为 0.000184431：

| 配方 | 输出 RMS | 尾段 RMSE | 尾段 max abs | cosine |
|---|---:|---:|---:|---:|
| rmvpe_intermediate | 0.000184436 | 1.13e-7 | 8.72e-7 | 0.999999812 |
| memory_attention | 0.000181837 | 5.01e-6 | 4.01e-5 | 0.999726635 |
| balanced | 0.000172891 | 1.46e-5 | 7.34e-5 | 0.998758754 |
| balanced_stage2 | 0.000188995 | 9.86e-5 | 0.000549 | 0.860796520 |

扩大量化 Synth 后，尾段 RMS 没有明显放大，但误差信号已达到参考 RMS 的约 53.5%。
不能仅凭语音窗口高 cosine 接受这个配方，尚未作为推荐 preset。
需要多声音、多音量、停顿和有声/无声切换的验证以及试听。

### 实时预算

| samples | HuBERT 帧 | Synth 帧 / skip | 实际播放 chunk | balanced 均值 / P95 |
|---:|---:|---:|---:|---:|
| 11200 | 34 | 70 / 8 | 8192@16 kHz = 512 ms | 205.62 / 228.63 ms |
| 44960 | 140 | 281 / 250 | 4000@16 kHz = 250 ms | 255.15 / 282.51 ms |

输入窗口包含历史和衔接区域，不能把 `samples / 16000` 当作可播放 chunk 时长。
11200 本轮纯推理预算充足；44960 的 balanced 平均 RTF 约 1.021，P95 超预算。
即使扩大量化 Synth 平均 RTF 为 0.982，P95 仍为 271.31 ms，质量也未通过。
本表全窗口 F0 配方未达到该 250 ms 长上下文预算。
这些不是麦克风采集到播放的实测延迟。

## Linear/GELU 融合实验

新增 `CpuConfig.fuse_linear_gelu`，默认 **false**。
只融合独占输出的 Linear/MatMul → exact GELU；保留标记的 preactivation 和其他消费者。
FP32 与 INT8 路径共用 typed activation post-op，保持 GELU 负输出。
重复执行、bias、诊断输出和量化融合都有算子测试。

oneDNN MatMul 支持 eltwise post-op；可用的融合不自动代表当前形状下更快。
见 [MatMul 文档](https://uxlfoundation.github.io/oneDNN/v3.10/dev_guide_matmul.html)
和 [Post-ops 文档](https://uxlfoundation.github.io/oneDNN/v3.10/dev_guide_attributes_post_ops.html)。

| samples | 当前 FP32 → 融合 ms | 加速率 [95% 区间] | 对未融合最终音频 max abs |
|---:|---:|---:|---:|
| 11200 | 209.51 → 214.87 | 0.975× [0.948, 1.006] | 0 |
| 44960 | 298.20 → 294.10 | 1.014× [0.980, 1.052] | 0 |

没有稳定加速，HuBERT tensor arena 也没有减少，保留独立实验开关。
本轮量化组合未开启这个开关。

## Rust 接口与校准

`CpuQuantizationPreset` 提供两个显式选择：

- `RmvpeIntermediate`：33 个指定 canonical RMVPE 算子，不使用 Smooth。
- `HubertMixedRmvpeIntermediate`：上述 33 个算子 + 61 个 HuBERT Linear。
  Q/K 无 Smooth；V/out 与 feature projection alpha=0.5；FFN input alpha=0.7。
  STFT、RMVPE encoder/decoder/head、GRU、attention 动态矩阵乘、norm、FFN output、Synth 保持 FP32。

静态 activation 每层一个 scale，权重逐输出通道 scale，zero point=0，S32 累加，
bias/输出 FP32。Smooth 输入通道缩放并入 Q/layout reorder，权重缩放在建图时完成。
详细合同见 [W8A8 实验](cpu-w8a8-expanded.md)。

```rust
use vocal_cpu::{CpuConfig, CpuPipeline, CpuQuantizationPreset};

let preset = CpuQuantizationPreset::RmvpeIntermediate;
let config = CpuConfig::default();
let recipe = preset.recipe(); // 只有规则，没有校准结果
let mut observer = CpuPipeline::from_sources_calibration(&sources, config.clone(), &recipe)?;
let conversion = vocal_types::ConversionConfig {
    index_ratio: 0.,
    auto_pitch: vocal_types::AutoPitchConfig::disabled(),
    ..Default::default()
};
// 独立 calibration windows，覆盖预期音量和内容，长度与 config 一致。
for waveform in &calibration_windows {
    observer.process(waveform, conversion, 0, false)?;
}
// 多 shape 校准时，分别观察后逐通道合并 absmax。
let calibration = observer.calibration()?;
drop(observer);
let pipeline = CpuPipeline::from_sources_preset(&sources, config, preset, calibration)?;
```

preset constructor 在建图前拒绝缺少或非法校准，编译时检查通道数量，建图后核对
实际量化算子的完整名称集合，拒绝缺失、额外和重复算子。
允许校准结果为已绑定的多配方超集；普通 `from_sources` 继续 FP32。
没有从实时输入或测试音频隐式校准，也没有改变 GUI 默认精度。

calibration map 本身不包含来源指纹：调用方必须绑定模型、图配置、shape 范围和校准语料。
不能跨模型直接复用本轮 map。实验脚本记录并验证这些指纹，但产物不是生产量化缓存。
两个 preset 已用真实模型分别测试两种 shape、各四个窗口，共 16 个输出，
对之前实测 JSON 规则的最大逐样本差异为 **0**，实际覆盖分别为 33 / 94 个算子。

## 瓶颈与接下来的优先级

1. **11200 的 Synth**：FP32 Synth 约 152 ms，占整链约 70%。优先找 Conv/Deconv
   layout、残差处理和内存边界优化，再验证选择性量化。此前全部 generator INT8
   会污染低音量输出，不能直接默认启用。
2. **44960 的 HuBERT**：FP32 约 122 ms；mixed 降到 94.67 ms，是本轮明确收益。
   140 帧继续使用 dense token-major；此前 online attention 省 scratch 但更慢。
   HuBERT 是双向 attention，滑动窗口简单复用旧 KV 不等价。
3. **44960 的 RMVPE 重算**：CPU 每次处理完整 44960 窗口，FP32 约 66.77 ms，
   量化后仍约 61.48 ms。比继续扩大 encoder/decoder INT8 更值得先研究
   最近窗口 + F0 history cache。它改变 F0 上下文合同，需对照 Python 实时参考，
   验证缓存推进、`3:-1` 裁剪、启动/重置、有声无声转换。后续实现和独立实时参考
   已完成，见 [实时 F0 缓存](cpu-realtime-f0-cache.md)；上表不混入后续结果。
4. **调度与内存**：继续固定 shape、共享持久线程池、一次权重打包和 workspace 复用。
   不直接改为全部 20 个逻辑核；先前线程/affinity/spin 测量没有足够稳定收益。
   W8A32/动态量化仍需真实 kernel/端到端测量，不以权重缩小比例估算速度。

## 复现与产物

先用既有脚本准备原始参考 fixtures：`experiment-cpu-w8a8-combinations.py`、
`experiment-hubert-quantization.py`。需要真实 VFRM 模型及
`target/cpu-w8a8/broad/inputs.json`；模型与产物不进入 Git。

```powershell
rtk cargo build -p vocal-cpu --example quantization_benchmark
rtk proxy .venv/Scripts/python.exe scripts/benchmark-cpu-recipes.py --directory target/cpu-recipes/fusion --recipes linear_gelu --iterations 32
rtk proxy .venv/Scripts/python.exe scripts/benchmark-cpu-recipes.py --directory target/cpu-recipes/precision --recipes rmvpe_intermediate memory_attention balanced balanced_stage2 --iterations 32
```

`--binary` 可指定 Cargo 输出位置；`--threads` 调整硬件策略并重新测量。
基准支持 `[config.json] [baseline_config.json]`，保留老脚本默认控制布局。
example 的 recipe 文件路径也可替换成 `preset:rmvpe_intermediate` 或
`preset:hubert_mixed_rmvpe_intermediate`。
`--fuse-linear-gelu` 在量化和控制图同时开启融合；改图配置后重新校准。

- `target/cpu-recipes/fusion/report.json`：FP32 融合的配对结果与误差。
- `target/cpu-recipes/precision/report.json`：32 对实验、全部误差、P95/max、实际 kernel、内存与指纹。
- `target/cpu-recipes/precision/calibration.json`：独立语料的原生校准，合并两种 shape。
- 各 shape/配方目录内 `raw.json`、FP32/候选 `.f32`、候选 `.wav`，可用于试听。
- `target/cpu-recipes/preset-api/report.json`：Rust preset 与实测规则的真实模型一致性检查。

`--resume` 检查 binary、模型、全部 waveform/noise/参考音频/F0 fixture、规则、
线程/affinity、迭代数和校准 SHA256。改变其中任一项应使用新的实验目录。

验证通过：Rust fmt check、CPU 30 个测试、workspace all-features check、
workspace 213 passed / 13 ignored、Python py_compile。
另对 8 组保存的量化基准逐项核对完整算子名称集合，并检查成对统计的固定比例样例。
