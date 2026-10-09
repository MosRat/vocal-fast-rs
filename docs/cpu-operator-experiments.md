# 独立 CPU 算子库实验

2026-10-08，本机 i5-13600K，Windows x64、AVX2、FP32。

## 方法与边界

从现有 VFRM 手写图捕获真实输入、权重、bias 和输出，对比独立 oneDNN、
独立 XNNPACK 与当前 ATen 路径。没有使用随机矩阵代替真实模型数据。

- 模型：`default-v2/default.pth` 对应的规范化 Synth VFRM、HuBERT、RMVPE。
- 输入：已有中文离线测试的 `waveform.bin`，分别取 11200 / 44960 samples。
- 11200：HuBERT 34 frames，Synth 70 frames、skip 8，生成 62 frames。
- 44960：HuBERT 140 frames，Synth 281 frames、skip 250，生成 31 frames。
- 捕获 103 / 105 种 Conv、Deconv、静态 GEMM 形状；每个配置分别代表
  HuBERT 81、RMVPE 130、Synth 157 次算子执行。
- 同形状算子选一个真实实例，并记录出现次数。加权时间为
  `sum(代表实例的 median × 出现次数)`。
- 每个提供方使用独立进程，全部串行运行。通过 Windows CPU-set 的效率等级
  选择性能核心对应的 logical CPUs 0–11；没有假定 CPU 编号。
- 测试 1 / 6 线程。每个算子先暖机，7 个计时批次，按先导时间选择循环数。
  循环与计时都在 C++ 中，数值检查和输出复制在计时区间外。
- `prepared`：输入保持提供方所需布局，权重已预打包。
- `boundary`：每次从 canonical NCHW 转换输入，再将输出转换回 NCHW；
  GEMM 使用行主序，无额外布局变换。布局转换采用各路径实现，不保证相同内核。
- oneDNN / XNNPACK 复用输出和 workspace；ATen 包含其正常输出分配。

**这些加权时间是隔离算子实验，不是完整管线实测。** 它们没有模拟完整图的
缓存竞争、全部模型权重的工作集、逐元素操作、Attention、GRU、FFT、检索和
音频接缝。相同形状也可能因输入数值和布局不同而有不同表现。

捕获阶段为了保留中间输出关闭局部融合，捕获耗时不参与性能比较。
另行测试原有完整 ATen 管线，避免把热点时间误认为最终端到端时间。

## 六线程结果

最终数据：`target/compare/cpu-operators/final-report.json`。
表中都是上述加权隔离算子时间，单位 ms；XNNPACK 已使用 AVX2 转置。

| samples | 模块 | ATen prepared | oneDNN NHWC prepared | XNNPACK prepared |
|---:|---|---:|---:|---:|
| 11200 | HuBERT | 24.74 | 16.24 | 25.91 |
| 11200 | RMVPE | 21.75 | 10.83 | 18.78 |
| 11200 | Synth | 142.51 | 91.02 | 150.96 |
| 44960 | HuBERT | 94.25 | 61.20 | 105.34 |
| 44960 | RMVPE | 52.86 | 30.32 | 55.31 |
| 44960 | Synth | 83.04 | 50.65 | 85.66 |

Synth 热点合计，独立 oneDNN 分别减少约 36.1% / 39.0% 的时间。
两个窗口的 Synth 生成长度不同，不能直接用这两行比较同等输出的成本。

| samples | 模块 | ATen boundary | oneDNN NHWC boundary | XNNPACK boundary |
|---:|---|---:|---:|---:|
| 11200 | HuBERT | 25.82 | 18.11 | 26.42 |
| 11200 | RMVPE | 26.52 | 15.70 | 21.31 |
| 11200 | Synth | 172.13 | 107.45 | 161.18 |
| 44960 | HuBERT | 101.08 | 61.49 | 106.05 |
| 44960 | RMVPE | 68.26 | 38.11 | 60.34 |
| 44960 | Synth | 95.18 | 61.07 | 92.12 |

`format_tag::any` 与显式 NHWC 的 prepared 性能接近，没有一致优势。
显式 NCHW 在 RMVPE / Synth 上明显较慢：例如 11200 的 Synth 为 164.14 ms。
第一轮 XNNPACK 使用朴素串行转置，11200 的 Synth boundary 为 407.55 ms；
替换为 AVX2 8×8 分块转置、pthreadpool 调度后，复测为 162.23 ms，最终重复
测量为 161.18 ms。该差异来自实验中的转换实现，不能归咎于 XNNPACK 卷积内核。

单线程结果保存在 `report.json` 和 `xnnpack-simd-report.json`。11200 的
Synth prepared：ATen 406.14 ms，oneDNN NHWC 390.21 ms，XNNPACK 779.00 ms。
因此六线程收益不能直接外推到所有线程数和 CPU。

## 热点与实现

11200，六线程，单次代表算子：

| 算子 | 形状与参数 | ATen ms | oneDNN ms |
|---|---|---:|---:|
| Synth residual Conv | C=128、T=6200、K=11、dilation=1 | 5.1603 | 3.4011 |
| Synth residual Conv | C=128、T=6200、K=11、dilation=3 | 5.3400 | 3.2941 |
| Synth residual Conv | C=128、T=6200、K=11、dilation=5 | 5.1689 | 3.5659 |
| HuBERT FFN + bias | `[34,768] × [768,3072]` | 0.4188 | 0.2949 |

oneDNN 报告的实际实现为 `brg_conv_fwd:avx2` 和 `gemm:jit:f32`。
Synth 的普通 / 膨胀 Conv 合计为 ATen 138.11 ms、oneDNN 88.53 ms；
Deconv 合计为 4.35 → 2.40 ms。当前的主要收益来自卷积区域。

oneDNN 实验使用缓存的 primitive、预打包权重、应用提供的同步线程池、
`scratchpad_mode::user` 及固定输出内存。GEMM bias 通过 primitive bias 参数
计算。Deconv 使用 `iohw/giohw` 正确解释 Torch 的权重顺序；dilation 的 API
约定减一，output padding 映射到右侧 padding。

当前 ATen SDK 内部为 oneDNN 3.7.1，独立库为 3.10.2。观察到的差异同时包含
版本、布局、权重预打包、内存分配和线程池策略，不能全部解释为 ATen 调度成本。

## 数值

相对捕获的 FP32 图输出，六线程、NHWC / 优化后的 XNNPACK：

| samples | 提供方 | 最坏 max abs | 最坏 relative RMSE | 最低 cosine |
|---:|---|---:|---:|---:|
| 11200 | oneDNN | 1.6785e-4 | 1.0003e-6 | 0.9999999999995 |
| 11200 | XNNPACK | 1.9836e-4 | 1.7725e-6 | 0.9999999999984 |
| 44960 | oneDNN | 6.1035e-4 | 8.2645e-7 | 0.9999999999997 |
| 44960 | XNNPACK | 6.1035e-4 | 1.2020e-6 | 0.9999999999993 |

这些是不同内部算子的最坏值，不是最终音频误差。所有测试形状均被三种提供方
接受；prepared / boundary 输出都通过 relative RMSE < 1e-4 的检查。
完整替换执行器之后仍需独立特征、F0 和最终音频 parity。

## 完整 ATen 管线基线

同样的性能核心限制、六线程，暖机一次后测量十次；包含真实 HuBERT、RMVPE、
Synth、输入输出与 CPU 前处理，关闭检索。

| samples | HuBERT ms | RMVPE ms | Synth ms | 完整总时间 ms |
|---:|---:|---:|---:|---:|
| 11200 | 36.01 | 36.48 | 165.00 | 237.99 |
| 44960 | 117.94 | 83.76 | 101.10 | 304.75 |

数据在 `pipeline-11200-6.json` / `pipeline-44960-6.json`。
这些是原有 ATen 后端实测；本次没有获得完整 oneDNN / XNNPACK 后端的延迟。
它们也不包含设备采播、SOLA 等整个实时 engine 的开销。

## 依赖体积与来源

| 实测 artifact | 大小 | 说明 |
|---|---:|---|
| `torch_cpu.dll` | 250.49 MiB | 原有 ATen 主库，另有 c10 / OpenMP |
| `dnnl.dll` | 17.14 MiB | 3.10.2，选定 inference primitives，Graph / GPU 关闭 |
| `bench_xnnpack.dll` | 0.215 MiB | 本实验 FP32 算子与转置胶水静态链接后的 DLL |

`bench_xnnpack.dll` 使用 CPU torch 2.9.1 SDK 自带的 XNNPACK、microkernels-prod、
pthreadpool、cpuinfo 静态归档，运行时不依赖 Torch。这不是整个 XNNPACK SDK
或完整 RVC 后端的体积。oneDNN 的构建保留了默认 GEMM ISA 集合，运行时限制
AVX2；体积尚有进一步裁剪空间。

已用 `llvm-readobj --coff-imports` 检查 XNNPACK DLL 和 oneDNN DLL：只依赖
Windows / MSVC 运行库，没有 torch、c10、CUDA、MKL 或 OpenMP 导入。
SDK、源代码 checkout 与构建输出放在
`%LOCALAPPDATA%/vocal-fast/cpu-operators`，模型捕获与报告放在 `target/compare/`。

## 复现

CPU LibTorch SDK 用于捕获、对照，以及提供独立 XNNPACK 静态归档。先按
[CPU 后端文档](cpu-backend.md) 准备 SDK 与真实 VFRM / waveform 文件。

```powershell
rtk proxy powershell -NoProfile -File scripts/setup-cpu-operators.ps1
rtk cargo build -p vocal-cpu --examples
foreach ($samples in @(11200, 44960)) {
    rtk proxy target/debug/examples/export_operator_cases.exe `
        target/default-canonical-with-index.vfr `
        models/shared/hubert_base/hubert_features.vfr `
        models/shared/rmvpe/rmvpe.vfr `
        target/compare/cpu-final2-trace/0000/waveform.bin `
        $samples "target/compare/cpu-operators/$samples" 6
}
rtk proxy .venv/Scripts/python.exe scripts/benchmark-cpu-operators.py `
    --fixtures target/compare/cpu-operators `
    --output target/compare/cpu-operators/final-report.json `
    --threads 6 --samples 11200 44960 `
    --pipeline-baseline target/debug/examples/benchmark.exe
```

默认实验也覆盖单线程。`--providers xnnpack` 可单独复测转换策略，
`--all-cores` 可取消性能核心限制。报告保留批次时间、创建时间、每个形状的
错误指标、实际 oneDNN implementation 和 benchmark DLL 的 SHA256。

## 结论

本机六线程基线上，优先以独立 oneDNN 接管大块 Conv / Deconv / GEMM。
建立固定布局区域、复用 primitive 和 scratchpad，避免逐层转换为 canonical。
XNNPACK 已验证可提供很小的独立算子 artifact，但当前模型热点总体速度没有
超过 oneDNN，适合保留为按形状选择的候选。

FFT、GRU、Attention 与逐元素区域的替代尚未在本实验里计时，RustFFT 和
完整轻量执行器仍需单独实现及验证。现有 ATen 后端继续承担整链路数值对照。

验证：Rust fmt 检查、`vocal-cpu` 7 个测试、Python py_compile、原生 Release
构建，以及两组真实模型捕获的数值检查。

构建选项参考：[oneDNN 3.10 文档](https://uxlfoundation.github.io/oneDNN/v3.10/dev_guide_build_options.html)，
[XNNPACK 算子接口](https://github.com/google/XNNPACK/blob/master/include/xnnpack.h)。
