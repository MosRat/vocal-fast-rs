# CPU 实时 RMVPE 窗口与 F0 缓存

## 实现与默认选择

44960 实时入口现在采用参考流程的最近窗口 F0：

1. HuBERT 仍处理完整 44960 samples，输出 140 帧。
2. RMVPE 只处理最后 4960 samples，STFT/model 输出 32 帧。
3. 1024 帧原始 Hz 缓存左移 25 帧，末尾覆盖 RMVPE 的 `3:-1`，共 28 帧。
   新结果包含 3 帧重叠上下文，不能只追加最后 25 帧。
4. 从缓存末尾取 281 帧，再施加移调、coarse F0 与特征保护，送入 Synth。
5. Synth 裁掉前 250 帧，输出 31 帧；共同音频边界提取 4000@16 kHz 播放块，
   800 samples crossfade、160 samples SOLA 搜索。

合同依据本地 [Applio 实时 pipeline](../ref_repos/Applio/rvc/realtime/pipeline.py)
的最近窗口长度、缓存推进和 `3:-1` 裁剪。4000+800 经 RMVPE 32 帧对齐得到 4960。
所有模型算术继续 FP32；这次没有依赖量化实现加速。

- `CpuEngine::from_spec` 的 44960 配置默认启用 Rolling，CLI、Runtime/GUI 的同一入口均生效。
- 11200 实时入口仍使用 FullWindow，因为其 8192-sample 播放块不是 160-sample hop 的整数倍。
  不用向下取整的缓存推进，以免长期 F0 漂移。
- 普通 `CpuConfig::default()`、`CpuConfig::offline()`、几何 helper
  `CpuConfig::realtime_44960()` 本身继续 FullWindow，保持独立管线与旧实验的合同。
  通过 `CpuConfig.f0` 显式选择 Rolling；低层 `CpuEngine::from_sources` 也可以选择 FullWindow。
- 引擎拒绝缓存推进与播放块不同的配置；CPU 配置拒绝非整数 hop、超过 1024 帧历史、
  不足以覆盖推进的 RMVPE interior，以及离线使用 Rolling。

这会改变 F0 的上下文语义。正确精度参考是相同 Rolling 流程的 Python，
不是期望 FullWindow 和 Rolling 输出相同。

## 状态与代码层次

| 文件 | 职责 |
|---|---|
| `vocal-cpu/src/config.rs` | 序列化 `CpuF0Config`、几何与 hop 检查 |
| `vocal-cpu/src/pitch.rs` | RMVPE 图构建、原始输出、缓存准备/提交/重置 |
| `vocal-cpu/src/pipeline.rs` | 与 HuBERT、frontend、Synth 的 buffer 交接 |
| `vocal-engine/src/cpu_backend.rs` | 实时默认选择、播放块一致性、会话重置 |
| `vocal-engine/examples/cpu_streaming_benchmark.rs` | 独立诊断、成对计时、真实引擎文件回放 |
| `scripts/benchmark-cpu-streaming.py` | 调度、P 核 affinity、统计与来源摘要 |
| `scripts/run-python-streaming-reference.py` | CPU/DML/TRT 的原始模型与实时 DSP 对照 |

缓存从零开始；保留现有 13 块启动静音和共同 SOLA 边界。
历史存储原始 Hz，移调在读取历史后施加，因此改变移调无需重新填满缓存。
只构建一个 4960 RMVPE 图，没有额外保留完整 RMVPE 图或权重副本。

缓存更新先生成 pending history，Synth 成功且音频有限时才提交。
下游失败不会推进已提交历史。`reset_stream` 清空 committed/pending history 和自动音高状态；
引擎的会话重启、格式变化及捕获中断沿用共同 reset 入口。

诊断模式额外返回 `rmvpe_pitchf`（32 帧原始 F0）、`f0_cache`（1024 帧）和 `pitchf`
（移调后的 281 帧）。正常处理不导出这些张量，沿用持久输入和中间 buffers。

Rust 显式配置：

```rust
let config = vocal_cpu::CpuConfig {
    f0: vocal_cpu::CpuF0Config::rvc_realtime(),
    ..vocal_cpu::CpuConfig::realtime_44960()
};
let pipeline = vocal_cpu::CpuPipeline::from_sources(&sources, config)?;
```

Rolling 会改变 RMVPE activation 分布和 shape；量化候选需要重新校准，
不把此前 FullWindow 校准 map 当作已经验证的 Rolling 配方。

## 本机成对延迟

2026-10-08，i5-13600K，6 个算子线程，spin=0，P 核逻辑处理器 0–11（含 SMT）。
同一声音模型、27.1 秒中文 WAV、index=0、移调关闭、RMS mix=1。
完整回放 109 块，前 13 块用于历史启动，剩余 **96 对**交替测量。
固定 waveform 和相同随机序列；diagnostic 在计时前单独执行，之后重置 F0 历史。

`CpuPipeline::process` 计时包含随机输入生成、输入验证、阶段 IO、frontend、最终输出复制。
真实引擎另外测量共同重采样、SOLA、缓冲、输出封装。加载/建图、WAV IO 和音频设备不计入。
与此前 `process_exact` 的 32 对窗口实验合同不同，只比较本表同进程成对结果。

| 阶段 | FullWindow ms | Rolling ms |
|---|---:|---:|
| HuBERT | 105.78 | 104.57 |
| RMVPE + cache | 58.92 | 11.02 |
| 检索边界（index=0） | 0.032 | 0.031 |
| Frontend + Synth | 88.50 | 88.93 |
| 完整 pipeline | **254.88** | **206.18** |

| 指标 | FullWindow pipeline | Rolling pipeline | Rolling 真实 Engine |
|---|---:|---:|---:|
| 平均 ms | 254.88 | 206.18 | **209.84** |
| P95 ms | 273.66 | 218.34 | **223.89** |
| 最大 ms | 297.56 | 240.57 | 236.83 |
| RTF，相对 250 ms 播放块 | 1.020 | 0.825 | 0.839 |
| 超出 250 ms 的计时块 | 67/96 | 0/96 | 0/96 |

成对整链加速 **1.236×**，按 1 秒连续块重采样的 95% 区间为 **[1.224, 1.247]**。
置信区间只表示本轮波动。真实引擎阶段单独回放，不把两项均值相减当作精确边界成本。
本轮达到该硬件/模型的文件回放预算；不是带设备 IO 的持续直播保证，也不是采集到播放延迟。

RMVPE 原生 tensor arena 从 14.73 MiB 降到 1.94 MiB，workspace 从 7.14 MiB 降到 0.92 MiB。
固定 shape 也会影响 oneDNN 参数打包，本次 packed 参数反而增加约 1.34 MiB。
这些是原生分配核算，不能当作进程 RSS 或权重压缩比例。

## 与原始 Python FP32 对齐

原始 HF HuBERT + Applio RMVPE/Synth 独立提取特征和 F0；仅共享 native latent/NSF noise。
关闭 TF32，保留启动阶段的 history 统计；可听音频指标从 warmup 后开始。
真实 Engine 生成 16 kHz PCM16；模型输出为 40 kHz。

| 项目 | RMSE | 最大绝对误差 | cosine |
|---|---:|---:|---:|
| HuBERT | 1.57e-6 | 0.000267 | 0.999999999989 |
| 原始 RMVPE，Hz | 4.73e-5 | 0.000252 | 0.999999999999975 |
| F0 cache，Hz | 4.38e-5 | 0.000244 | 0.999999999999975 |
| Synth 音频尾段 | 3.53e-5 | 0.000931 | 0.999999980337 |
| 最终 WAV，含独立 SOLA | **3.66e-5** | **0.000841** | **0.999999979570** |

- 全部 109 块 coarse F0 mismatch=0；SOLA offset mismatch=0。
- 从 diagnostic 张量用 Python 重建的 PCM 与实际 Engine PCM 逐样本相同，max abs=0。
  这检查了模型到真实引擎的随机数、IO 和音频边界交接。
- 12 个低音量尾段块（参考 RMS < 0.01）保留在报告中；最小 cosine 为 0.999999579，
  最大 RMSE=2.22e-6、max abs=2.74e-5。
- 单独 Synth 输入对齐的 RMSE=3.47e-5；端到端误差不能只归结为 F0 cache。

只覆盖当前 default-v2 声音模型和中文片段；未测量检索、量化组合或所有声音的质量。

## 复现与产物

```powershell
rtk cargo build -p vocal-engine --features cpu-graph --example cpu_streaming_benchmark
rtk proxy .venv/Scripts/python.exe scripts/benchmark-cpu-streaming.py --directory target/cpu-f0
rtk proxy .venv/Scripts/python.exe scripts/run-python-streaming-reference.py --input target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav --trace target/cpu-f0/trace --output target/cpu-f0/applio-rolling.wav --native-output target/cpu-f0/cpu-rolling.wav --report target/cpu-f0/python-parity.json
```

支持 `--binary` 指定 Cargo target，`--threads` 改硬件策略；改动后重新测量。
已有 native 报告可用 `--summarize-only` 生成统计，来源 hash 明确记录为 summary 时的指纹，
不宣称它是事前绑定的 resume 合同。完整 debug trace 约数百 MiB，宜指定容量充足的目录。

本轮大 trace 留在 `%LOCALAPPDATA%/vocal-fast/validation/cpu-f0/trace`，小报告与试听文件复制到：

- `target/cpu-f0-rolling/summary.json`：延迟分布、逐阶段均值、内存与来源摘要。
- `target/cpu-f0-rolling/benchmark.json`：原始 96 对数据及独立 Engine 计时。
- `target/cpu-f0-rolling/python-parity.json`：完整阶段、最终 WAV、逐块和低音量误差。
- `target/cpu-f0-rolling/cpu-rolling.wav`：真实 CPU Engine 输出，27.1 秒，前 3.25 秒启动静音。
- `target/cpu-f0-rolling/applio-rolling.wav`：相同输入的 Python FP32 对照。

验证：CPU/Engine feature 测试 63 passed、workspace 217 passed / 13 ignored、
workspace all-features check、Rust fmt check、Python py_compile。
测试覆盖精确裁剪/重叠缓存、非法 hop、pending history 丢弃、会话重置和非有限 F0。
