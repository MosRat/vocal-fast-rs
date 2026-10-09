# CPU FP32 卷积融合与投影分派

2026-10-09，i5-13600K，Windows x64，独立 oneDNN 3.10.2，6 线程。
所有实验均为 FP32；未扩大 INT8 覆盖，也未改模型权重、F0 或音频边界。

## 实现

- `CpuConvolutionConfig.fuse_residual`：融合 Conv/Deconv → 可别名 reshape →
  residual Add，独占消费时还可融合 Leaky。要求另一输入在 Conv 执行前已存在、
  同形状、同类型，且布局可直接解释。标记的中间输出和分支保持其值；量化和
  calibration Conv 不参与。共享的 residual sum 保留在 activation 前，避免
  AddLeaky 降低阶段重复计算 Add、破坏 Conv 的单消费者条件。
- NHWC residual 的 singleton stride 在编译时规范化；否则 oneDNN 可以选择
  `ref:any`。若融合后的实际 implementation 含 `ref:`，撤销该处融合。
- `CpuLinearMode::InnerProduct`：仅处理连续 FP32 激活和固定二维权重，编译时
  枚举 primitive descriptor，优先选支持的 `brgemm:`。oneDNN 3.10.2 的该
  内核只接受 `forward_training`；这里只运行无状态 affine forward，不创建
  backward 或训练状态。使用固定版本的公共 API，不修改 oneDNN 源码。
- `CpuLinearMode::Pointwise1d`：把 token-major Linear 直接视为 NHWC 1×1
  Conv，选到 `brgconv_1x1:avx2`；激活不做 unfold、转置或复制，权重仅打包一次。
- `CpuLinearMode::Hybrid`：实验性形状分派。当 `rows >= 64 && K >= 1536`
  时保留 MatMul，其余支持的投影用 Pointwise1d。该阈值来自 34/140 帧的真实
  模型捕获，其他 CPU、较长离线序列须另测。动态权重、BMM 和非连续激活仍用
  原 MatMul。显式模式与实际 implementation 均可在 profile 中核查。
- `CpuConvolutionMode::Matmul1d/Pointwise1d`：另有有界 unfold 实验，workspace
  最大 96 MiB；这些针对非 1×1 Conv 的实验没有获得稳定收益，保留为可复现选项。

配置、Rust 所有权/FFI、原生算子与实验分别位于 `src/convolution.rs`、
`src/linear.rs`、`src/runtime.rs`、`native/linear.cpp`、
`native/convolution_matmul.cpp` 和 benchmark examples。权重、descriptor、
scratch、residual 存储和 arena 生命周期在编译时准备，replay 重用。
`CpuNodeTiming.implementation` 显示实际内核，不把配置名称当成加速证据。

默认仍为原有 FP32 Direct Conv + MatMul；新配方显式选择。保持历史配置和
校准图合同，不把单机单声音的实验分派作为全部硬件的默认。

## 实验合同

沿用 [热点融合测试](cpu-hotspot-fusion-recipes.md) 的原始 Python FP32 参考：
原始 HF HuBERT、Applio RMVPE/Synth；TF32 关闭，latent/NSF noise 相同。
SOLA 是当前 Rust `speech_safe_v1` 的独立 NumPy 参考，不是声称 Applio
完整实时 DSP 与本项目相同。

- 输入 257600 samples / 16.1 秒，单声道 16 kHz，default-v2 / 40 kHz 模型。
- 44960 输入窗口，4000 advance，250 ms 播放 chunk；HuBERT 140、Synth 281
  帧，skip 250；RMVPE 最近 4960 samples / 32 帧，缓存未变调的 Hz。
- index=0、pitch=0、speaker=0、RMS mix=1；65 块全部验证数值，前 13 块只从
  延迟统计排除，实际启动输出不静音。52 块计时包含低音量尾部。
- 控制是已优化的 channel norm + tensor IO + rolling F0 FP32，明确关闭新
  Conv residual/GELU 融合、Linear 替代。候选与控制在同进程交替执行。
- benchmark 进程限定 P 核逻辑处理器 0–11（包括 SMT），spin=0，pool pin=false。
  下表保持同一 affinity/线程配方；生产默认仍由 OS 调度。
- per-node profile、trace 与正常计时分离；参考验证和构建不与基准并行。
  同进程计时包括 RNG/模型间 IO；独立 Engine replay 包含重采样、SOLA 与
  buffer，不含音频设备。完整采集到播放延迟不等于这里的处理时间。
- 按连续一秒 chunk 块 bootstrap，报告 95% 区间，不剔除慢样本。
  作业之间绝对延迟有漂移，只在每个配对内比较速度。

## 投影和卷积快筛

真实 VFRM 形状/输入/权重，32 对交替计时，6 线程。局部速度比包含输出复制，
不能直接当整管线速度比；同形状代表算子也不能代表全部层的数值。

| 投影 | tokens | MatMul / Pointwise ms | 局部速度比 |
|---|---:|---:|---:|
| HuBERT Q/K/V/out 代表 | 34 | 0.09173 / 0.07713 | 1.189× |
| HuBERT Q/K/V/out 代表 | 140 | 0.43718 / 0.38745 | 1.128× |
| HuBERT FFN output | 34 | 0.37452 / 0.35442 | 1.057× |
| HuBERT FFN output | 140 | 1.34498 / 1.60763 | 0.837× |

因此保留长序列 FFN output 的 MatMul。BRGEMM InnerProduct 的这类投影同样
更慢，约 1.36968 → 1.48694 ms，不能整图强制选择 BRGEMM。

HuBERT extractor Conv1 的 unfold MatMul 为约 15.79 → 17.27 ms，unfold
1×1 Conv 为约 15.90 → 16.91 ms；主要 Synth Conv 也无一致收益。额外的
展开/访问量抵消了 GEMM 内核收益，保留直接卷积。

## 整链结果

首次有效配对及独立 Engine replay：

| 配方 | 控制 → 候选 ms | 速度比及 95% 区间 | Engine mean / P95 / max ms |
|---|---:|---:|---:|
| residual | 213.99 → 213.75 | 1.001× [0.975,1.030] | 196.94 / 220.66 / 234.85 |
| residual + InnerProduct BRGEMM | 198.90 → 193.70 | 1.027× [1.002,1.050] | 194.90 / 231.33 / 235.75 |
| residual + Hybrid | 200.43 → 193.20 | 1.037× [1.009,1.071] | 198.30 / 219.39 / 223.04 |

每项 Engine 为 0/52 超过 250 ms。这是单个声音、同语料 held-out 区间的文件
replay，尚未证明长时间真实设备采播或多声音性能。

第二轮 Hybrid 独立重复：226.28 → 218.21 ms，**1.037× [1.023,1.052]**；
Engine mean/P95/max = **225.97 / 238.03 / 241.45 ms**，0/52 超预算。
两轮配对收益均约 3.7%，但不能忽略绝对处理时间由 198 ms 漂移到 226 ms。

### GELU 融合的独立取舍

Pointwise Linear 的 exact erf GELU post-op 已通过数值检查，HuBERT 独立 GELU
节点从 20→8，去掉了 12 个 FFN activation dispatch。启用/关闭融合的 Engine
PCM16 逐字节一致。不过从各自相对 production 的速度比，不能推出两候选
之间的收益，因此另做以 residual + Hybrid 为控制的配对：

| 对照 | 控制 → 候选 ms | 速度比及 95% 区间 |
|---|---:|---:|
| Hybrid → Hybrid + Linear GELU | 220.00 → 221.13 | 0.995× [0.979,1.009] |

未证明额外加速，建议新实验配方也保持 `fuse_linear_gelu=false`。这一慢环境
测试的两个配对处理路径分别为 12/52、13/52 超过 250 ms；候选独立 Engine
mean/P95/max 为 **239.93 / 264.04 / 274.96 ms**，**30/52** 超预算。
不能把前两次未超预算当成稳定实时保证，也不能把不同作业的绝对耗时差
归因于 GELU；真正隔离这一因素的速度比区间包含 1。

基准结束后的 WMI 快照显示总 CPU 使用率 27%，有 MAA、MuMuNxDevice 等
其它进程负载，电源政策为 Balanced。该快照不是基准期间的同步采样，无法
定量证明调度/频率或某个进程是慢样本的原因；保留原始慢样本和统计，没有
停止其它进程或更改系统电源策略。当前六线程 FP32 的 250 ms 预算仍可能
在并发应用负载下不足。

Residual 单独并未证明总延迟收益；有明确的内存/调度收益：57 个 Conv binary
Add 融合，AddLeaky 37→1，Synth arena 28,766,164→23,052,244 bytes，减少
5,713,920 bytes（约 5.45 MiB），tensor allocations 272→266。

| 配方 | 最终 WAV 对 Python cosine | RMSE | max abs | 对原生 FP32 |
|---|---:|---:|---:|---|
| residual | 0.999999978699 | 3.60412e-5 | 0.000895962 | trace/PCM16 逐字节一致 |
| residual + InnerProduct | 0.999999978594 | 3.60993e-5 | 0.000926480 | RMSE 5.87934e-6 |
| residual + Hybrid | 0.999999978590 | 3.61020e-5 | 0.000926480 | RMSE 5.88365e-6 |

第二轮 Hybrid 和 Hybrid + Linear GELU 的上述精度指标相同，PCM16 逐字节
相等。GELU 的取舍由性能证据决定。

全部粗 F0 mismatch 为 0。投影替代引入不同 FP32 归约顺序，Hybrid 对原生
PCM16 的 max abs 为 6.10352e-5；最终 WAV 保持对原始 Python 参考的严格对齐。
不能把“相对内部 FP32 误差小”替代成“已对 Python 验证”。

## 仍需优化的热点

Hybrid 的独立 node profile（诊断时间，不是正常总延迟）：Synth Conv 合计
约 74.93 ms、HuBERT LinearBias 52.50 ms、HuBERT Conv 31.75 ms、RMVPE Conv
8.71 ms。HuBERT attention 3.55 ms、GELU 3.15 ms。

最大单层为 HuBERT extractor Conv1 约 14.47 ms、Conv2 7.41 ms；Synth 的
kernel=11 residual Conv 单层约 2.6–3.6 ms。优先研究这些计算块的分块、
权重缓存与 activation post-op，减少纯调度开销不足以消除主要 FLOPs。

## Rust 使用与复现

```rust,ignore
use vocal_cpu::{CpuConfig, CpuConvolutionConfig, CpuF0Config, CpuLinearMode};

let config = CpuConfig {
    linear_mode: CpuLinearMode::Hybrid,
    convolution: CpuConvolutionConfig {
        fuse_residual: true,
        ..Default::default()
    },
    f0: CpuF0Config::rvc_realtime(),
    ..CpuConfig::realtime_44960()
};
// CpuPipeline::from_sources(&sources, config)?;
// CpuEngine::from_sources(model, &sources, config, CpuStreamConfig::default(), speaker)?;
```

```powershell
rtk proxy powershell -NoProfile -File scripts/setup-cpu-operators.ps1 -SdkOnly
rtk cargo build --release -p vocal-cpu --example linear_benchmark --example convolution_benchmark
rtk cargo build --release -p vocal-engine --features cpu-graph --example cpu_streaming_benchmark
rtk proxy .venv/Scripts/python.exe scripts/benchmark-cpu-convolutions.py --directory target/cpu-kernels --binary-dir target/release/examples --sources sources.json --input heldout.wav --baseline-report fp32_io/benchmark.json --skip-operators --linear-operators --variants residual_hybrid_linear --trace --profile
rtk proxy .venv/Scripts/python.exe scripts/validate-cpu-streaming-recipes.py --reference-directory fp32_io --candidates target/cpu-kernels/residual_hybrid_linear --report target/cpu-kernels/parity.json
```

sources、baseline-report、已验证的参考和 fixture 必须属于同一实验合同。空间
较少时将 `CARGO_TARGET_DIR` 和报告目录设置到空间充足的磁盘。
报告包含模型、输入、fixture、二进制、oneDNN DLL 的 SHA256 和 affinity；
`--resume` 拒绝合同变化。产物在 `%LOCALAPPDATA%/vocal-fast/validation/` 的
`cpu-kernels-paired-v2`、`cpu-kernels-pointwise-screen`、`cpu-kernels-hybrid-v1/v2`、
`cpu-kernels-gelu-isolated`。`--control residual_hybrid_linear` 可隔离 GELU
相对已优化配方的收益；配对执行器要求相同线程/affinity/spin 政策，避免混用
线程实验与算子实验。

精简的合同/配对/精度汇总与第二轮 Hybrid 音频复制到
`target/compare/cpu-fp32-kernels/report.json`、`cpu-hybrid.wav`，仍不进入 Git。

本轮验证：CPU 45 passed；Engine cpu-graph 37 passed / 1 ignored；workspace
236 passed / 14 ignored；workspace all-features check、Rust fmt check 和变更
Python 的 py_compile 通过。独立 FP64 convolution/projection/GELU 测试覆盖
权重方向、多 batch、bias、padding/dilation、marked/branched 输出、共享 residual、
动态 BMM 回退和多次 replay。生成报告、音频、权重和 DLL 不进入 Git。
