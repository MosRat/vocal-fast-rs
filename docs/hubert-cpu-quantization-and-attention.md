# HuBERT：CPU 量化与缓存分块 attention

本报告的旧布局控制与当前生产默认存在区别；相对当前优化后 FP32 的重新配对结果见
[CPU 推理配方与当前瓶颈](cpu-inference-recipes.md)。

## 实验范围

2026-10-08，i5-13600K，oneDNN 3.10.2，6 个算子线程，限制到 Windows
报告的性能核心。测试进程串行执行。精度筛选在本地 CUDA Torch 中运行，
关闭 TF32；fake W8A8 的乘加仍是 FP32，不能用其时间宣称 INT8 提速。

参考为本地 HF HuBERT eager FP32 与原始 Applio FP32 Synth。
复用 `target/cpu-w8a8/broad/inputs.json` 的中文输入、Applio F0、latent/NSF
噪声和参考音频。校准 offset 为 32000、112000，test 为 208000、352000，
validation 为 272000、416000；每组分别使用 11200、44960 samples。
校准与保留测试区间不重叠；同 offset 的两种长度存在包含关系，不能当作
独立说话人样本。8 个保留窗口包含一个低音量尾段，所有结果均保留它。

Python 实验只改变 HuBERT，F0 与 Synth 保持参考实现。
原生实验运行完整 `process_exact` 管线，RMVPE、Synth 保持原生 FP32，检索关闭，
自动变调关闭，同一输入与噪声；其音频独立与原始 Applio 对照。
两条管线同时驻留，8 组预热、24 组交替次序的 FP32/候选配对。
计时包含模型阶段 IO、activation Q/layout、frontend 与输出拷贝，不包含建图、
WAV 解码、文件写入、音频设备缓冲和 SOLA。没有在计时中下载诊断张量。

## 量化粒度与敏感性

权重为每输出通道静态 S8 `[-127,127]`，激活为每层静态 scalar S8
`[-128,127]`，零点为 0；原生使用 S32 累加、FP32 bias/输出，实际 kernel
包含 VNNI。SmoothQuant 的 `x/s` 与输入量化合入 oneDNN reorder，
`W*s` 在编译时量化、打包。这里没有量化 attention 的 QK/PV 乘法或 softmax。

完成 33 组区域/alpha 筛选、12 组单 block 筛选，以及 4 组混合/残差探索。
以下是 Python fake quant 的最终音频结果，不能与真实 CPU 耗时混为一列。

| 区域 | 层数 | alpha | 语音窗口最差 cosine | 全窗口最差 cosine | 平均音频 RMSE |
|---|---:|---:|---:|---:|---:|
| Q/K projection | 24 | 无 | 0.999892 | 0.999892 | 0.001092 |
| V/out projection | 24 | 0.5 | 0.999907 | 0.999907 | 0.001256 |
| feature projection | 1 | 0.5 | 0.999964 | 0.999964 | 0.000602 |
| FFN intermediate | 12 | 0.7 | 0.999833 | 0.998867 | 0.001607 |
| FFN output | 12 | 0.5 | 0.997492 | 0.997492 | 0.007371 |
| 全 Linear | 73 | 0.7 | 0.996917 | 0.996354 | 0.007347 |
| feature extractor Conv | 7 | 0.5 | 0.955790 | 0.954145 | 0.028134 |

FFN output 的输入已经过 GELU，校准样本里的 `absmax / p99.9` 在第 8、9、10
层分别约 10.37、22.85、25.59。第 10 层 absmax 为 44.44、p99.9 约 1.74。
这些分位数来自有界抽样。低 clipping 比例仍可伴随较大误差，说明量化舍入与
跨层累积也需要考虑，不能只检查是否溢出。单 block 筛选中第 8、10 层更敏感。

候选 `mixed` 覆盖 61 个 Linear、57,016,320 个权重：

- Q/K 不使用 Smooth；V/out 与 feature projection 使用 alpha=0.5。
- FFN intermediate 使用 alpha=0.7。
- FFN output、extractor、position Conv、归一化、GELU 和 attention core 保持 FP32。

另尝试 FFN output 的 1%、2%、4% outlier 通道走小型 FP32 残差 GEMM，剩余通道
静态 W8A8。当前校准和组合下的语音最差 cosine 为 0.993477、0.994352、0.991713，
没有改善，未接入原生后端。此结果仅否定当前配方，不否定所有 outlier 分解方法。

## 原生缓存分块在线 attention

HuBERT 是双向 attention。这里的“在线”指逐 key tile 累积 softmax，所有 query
仍然读取所有 key。没有改成 causal attention，也没有跨音频窗口复用 KV；
窗口、双向上下文和时间归一化变化会改变特征，直接复用旧 KV 不等价。

每个 query tile 维护 running maximum `m`、分母 `l`、未归一化 context `u`：

```text
S = Q_tile K_tile^T / sqrt(D)
m_new = max(m, rowmax(S))
decay = exp(m - m_new)
P = exp(S - m_new)
l_new = decay * l + rowsum(P)
u_new = decay * u + P V_tile
Y = u / l
```

原生实现按 head/query tile 分配到现有持久线程池，使用 AVX2/FMA QK 与 PV
内核；任意合法 feature stride 可走 scalar 路径。没有嵌套 oneDNN 线程调度。
每个逻辑 worker 独占 scratch，按 cache line 对齐。工作区在建图时分配，
同一图的 12 层共享其最大容量，replay 中不分配 tile buffer。

默认 tile 候选为 Q=4、K=32、D=64。Q/K/V、scores 和 numerator 的逻辑 tile
数据约 19 KiB；6 个 worker 的直接 scratch 合计 9600 bytes。
该预算没有包括硬件预取、输入 stride 所触及的其它 cache line 或线程运行栈。
完整 scores 的共享工作区则是 34 帧 55,488 bytes、140 帧 940,800 bytes。

### 真实 Q/K/V 的 12 层微基准

从 HF FP32 的每一层提取真实 projection，恢复图中 BTHD→BHTD 的 stride 视图。
128 个配对样本/层；下表为各层中位数之和，包含绑定、reshape 与输出拷贝，
不是完整 HuBERT 或音频管线延迟。

| 输出布局/算法 | 34 帧 / 11200 | 140 帧 / 44960 | attention 临时工作区 |
|---|---:|---:|---:|
| 旧 BHTD dense + transpose/reshape 拷贝 | 约 8.06 ms | 约 33.99 ms | 54.2 / 918.8 KiB |
| dense，直接输出 BTHD 物理布局 | 1.003 ms | 4.507 ms | 54.2 / 918.8 KiB |
| online Q4/K32，BTHD 输出 | 0.876 ms | 9.058 ms | 9.375 KiB |
| online Q4/K64，BTHD 输出 | 0.871 ms | 9.447 ms | 12.375 KiB |
| online Q8/K64，BTHD 输出 | 0.892 ms | 9.230 ms | 24.375 KiB |

主要收益来自移除布局拷贝。dense token 与旧 CPU 的 context/最终音频逐样本一致；
online context 对 HF FP32 的最大全局绝对误差为 `1.73e-6`。
Python online 完整音频平均 RMSE 约 `9e-8`；原生 online 对旧 CPU 最终音频的
最大绝对误差分别为 `8.37e-6 / 9.28e-7`。
140 帧下 dense 的 GEMM/softmax 已很有效，在线算法虽然省内存，却约慢 2 倍。
34 帧的 online 微基准有少量优势，完整管线没有足够证据支持将其设为默认。

## 完整原生管线结果

FP32 控制使用旧输出布局；每个候选都有自己的配对控制，不应直接比较不同
job 的绝对耗时来排名微小差异。下表的收益均由本次配对计算。

| 候选 | 输入 | HuBERT FP32→候选 ms | 全链 FP32→候选 ms | 全链加速 | 语音最差 cosine | 全窗口最差 cosine | 平均 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 dense token | 11200 | 45.23→37.98 | 237.66→228.78 | 1.039× | 0.999999959 | 0.999999959 | 3.75e-5 |
| FP32 dense token | 44960 | 168.26→136.23 | 357.14→325.74 | 1.096× | 0.999999988 | 0.999999812 | 2.38e-5 |
| mixed + dense token | 11200 | 41.36→25.83 | 217.80→199.97 | 1.089× | 0.999670 | 0.999670 | 0.003368 |
| mixed + dense token | 44960 | 152.88→93.34 | 321.12→258.90 | 1.240× | 0.999897 | 0.998759 | 0.001522 |
| mixed + online Q4/K32 | 11200 | 47.99→28.72 | 250.14→229.55 | 1.090× | 0.999585 | 0.999585 | 0.003539 |
| mixed + online Q4/K32 | 44960 | 158.20→101.94 | 332.00→276.49 | 1.201× | 0.999917 | 0.999007 | 0.001503 |

`mixed + dense` 的全链配对周期 bootstrap 95% 区间为 11200 的
`[1.054,1.128]×`、44960 的 `[1.218,1.265]×`。此区间只描述本次运行波动。
HuBERT packed 参数节省约 162.84 MiB；不包含所有模型、JIT 元数据、线程栈、
原始文件映射或 allocator 开销，因此不能当作进程 RSS 的下降。
最大音频绝对误差分别为 0.072657、0.049067；不能用 cosine 隐去瞬时误差。

44960 低音量尾段的参考 RMS 为 `0.000184431`，mixed dense RMS 为 `0.000172891`，
RMSE 为 `1.457e-5`、max abs 为 `7.335e-5`、cosine 为 `0.998759`，仍包含在统计里。
仅覆盖当前中文片段与 default-v2 声音模型，尚未做多语言/多声音的量化推广。

## 配置与代码层次

生产默认仍为 FP32，改为 dense token 输出布局。量化与 online 为显式选项。
Rust/JSON 的 `CpuConfig.attention` 不依赖环境变量：

```rust
use vocal_cpu::{CpuAttentionConfig, CpuAttentionMode, CpuConfig};

let config = CpuConfig {
    attention: CpuAttentionConfig {
        mode: CpuAttentionMode::Online,
        query_tile: 4,
        key_tile: 32,
        token_major_output: true,
    },
    ..CpuConfig::default()
};
```

`mode` 在 `fuse=true` 的 attention 模式降低后生效。dense 可通过
`token_major_output=false` 恢复旧布局；tile 参数在进入 FFI 前验证。
量化使用既有 `CpuPipeline::from_sources_quantized`，配方是 canonical 名称的规则
与 CPU 校准结果，不改变 VFRM schema，也不把校准数据塞进模型权重。

| 层次 | 文件 |
|---|---|
| 类型化配置 | `crates/vocal-cpu/src/attention.rs` |
| FFI 与 graph 生命周期 | `src/ffi.rs`、`src/runtime.rs`、`native/bridge.h`、`native/runtime.cpp` |
| dense attention | `native/attention.cpp` |
| online 调度、stride 与共享工作区 | `native/attention_online.cpp` |
| AVX2/FMA 数学内核 | `native/attention_simd.cpp`、`attention_simd.h` |
| Python 正确性原型、研究残差 | `scripts/hubert_attention.py`、`hubert_quantization.py` |
| 精度筛选、真实 Q/K/V 导出 | `scripts/experiment-hubert-quantization.py` |
| 原生微基准、全链配对 | `examples/attention_benchmark.rs`、`quantization_benchmark.rs` |
| 串行编排、统计 | `scripts/benchmark-hubert-cpu.py`、`summarize-hubert-cpu.py` |

## 复现与产物

使用已有真实模型、上述 fixture manifest，以及先前 CPU 校准
`target/cpu-w8a8/broad/native/calibration.json`（覆盖 HuBERT 全部 Linear，
按相同校准窗口合并两个 shape 的逐输入通道 absmax）。
首次准备 fixture 的脚本是 `experiment-cpu-w8a8-combinations.py`，
原生校准接口/步骤见 [扩大 CPU W8A8 覆盖实验](cpu-w8a8-expanded.md)。

```powershell
rtk proxy .venv/Scripts/python.exe scripts/experiment-hubert-quantization.py
rtk proxy .venv/Scripts/python.exe scripts/experiment-hubert-quantization.py --extend --regions block_00 block_01 block_02 block_03 block_04 block_05 block_06 block_07 block_08 block_09 block_10 block_11 --alpha 0.5 --rescue
rtk cargo build -p vocal-cpu --example attention_benchmark --example quantization_benchmark
rtk proxy .venv/Scripts/python.exe scripts/benchmark-hubert-cpu.py --iterations 24 --configs dense_token online_4_32
rtk proxy .venv/Scripts/python.exe scripts/summarize-hubert-cpu.py
```

基准控制显式使用旧布局，不随生产默认变化。编排脚本支持 `--only micro|pipeline`、
`--jobs`、`--resume`；resume 检查 binary、manifest、校准、迭代数与 affinity。
重建 binary 后应重新测量，不应混合旧结果。实测时先完成编译，避免争用 CPU。

- `target/hubert-cpu/accuracy.json`：49 组 Python 结果、逐窗口误差与校准分布。
- `recipes.json`：包括原生可用的 `mixed` 映射；research outlier 未导出为原生配方。
- `attention-cases.json` 与 `.f32`：24 组真实 Q/K/V/context，覆盖两种 shape 的 12 层。
- `native-attention.json`：96 组原生微基准配对与独立诊断。
- `native-pipeline.json`、`summary.json`：18 组完整管线、原始时间与 quiet-window 结果。
- `native/<shape>-<recipe>/raw.json` 与 `.f32`：原生完整输出和配对控制。
- 顶层 WAV 为 Python 精度筛选结果，文件名显式包含配方；不是 CPU INT8 的延迟测量。

验证：`cargo fmt --all -- --check`、workspace all-features check、CPU 26 个测试、
Python 2 个 online attention 测试与改动脚本的 `py_compile` 均通过。
测试包含不等 Q/K 长度、非连续 stride、tile 尾部、极端 logits、奇数 feature 宽度、
重复 replay 和 FP64 独立 softmax 参考。
