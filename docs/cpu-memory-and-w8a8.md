# CPU 内存、线程与 W8A8 实验

2026-10-08，本机 i5-13600K、Windows x64、oneDNN 3.10.2。正式 CPU 后端仍使用 FP32；
INT8 是独立实验工具，尚未作为推理精度配置接入正式管线。

## 已落地的内存和 IO 优化

| 模块 | 改动 |
|---|---|
| `native/memory.cpp` | 编译结束后，把所有缓存的 oneDNN workspace memory handle 重绑到每类最大 buffer，释放扩容留下的旧 buffer |
| `native/dnnl_ops.cpp`、`gru.cpp` | 标明已打包的参数输入，释放没有其它运行时消费者的原始权重；marked output、bias、norm 参数、共享视图仍按 storage 生命周期保留 |
| `native/attention.cpp` | 原地 softmax，score/probability 共用一个 T×T workspace |
| `src/pipeline.rs` | 按 HuBERT、RMVPE、Synth 顺序加载、编译、释放源 package，减少同时驻留的原始权重 |
| `vocal-engine/src/cpu_backend.rs` | 验证后释放源 package，再构建完整管线 |
| `CpuGraph::run_f32_into` | 直接输出到调用方的 FP32 Vec，复用容量，消除临时字节 tensor 和逐元素解码 |
| `PipelineBuffers` | 复用特征、F0、展开后的 Synth 输入、coarse pitch 字节 buffer；返回音频保持独立所有权 |

workspace 的释放发生在全部编译和常量折叠完成之后。THREADPOOL 执行为同步调用，因此顺序算子可以复用
scratch；没有把仍被 primitive 引用的地址直接释放。原始权重有额外消费者或被标记输出时，继续保留。

新增 `CpuGraph::memory_usage()` 和 `CpuPipeline::memory_usage()`，分项报告 tensor、workspace、packed
参数和辅助缓冲区字节数。它们不包含 index、FFT plan、JIT/cache metadata、Rust host buffer、线程栈和
allocator 开销，不能与整个进程的 private bytes 混用。

Windows 进程采样每 20 ms 一次，稳定值取构建结束后的推理区间中位数，排除退出清理期间。
最终记录以 `target/cpu-memory/final-memory.json` 为准；旧版本为 `baseline-verified.json`。
所有下表单位均为 **MiB（2^20 bytes）**。

| 窗口 | 旧常驻 private | 新常驻 private | 旧加载峰值 | 新加载峰值 |
|---|---:|---:|---:|---:|
| 11200 | 2059.2 | 1248.2 | 2892.0 | 1829.7 |
| 44960 | 2112.5 | 1298.2 | 2943.8 | 1853.5 |

常驻 private 减少约 39–40%，加载峰值减少约 37%。三张图分项统计的 owned bytes 为
888.6/918.6 MiB，其余包括 index、运行库、host buffer 和 allocator 等。

## 线程政策及实测

Rust `CpuConfig` 增加两个语义化参数，不依赖环境变量：

```rust
let config = vocal_cpu::CpuConfig {
    threads: 6,
    spin_wait_us: 0,
    pin_to_performance_cores: false,
    ..Default::default()
};
```

- 同线程数、spin 和 affinity 政策的图共用持久 worker pool，调用线程参与工作。
- `spin_wait_us` 为 0–1000 µs，预算耗尽后进入休眠，不在两次音频回调之间持续占用核心。
- `pin_to_performance_cores` 在 Windows 可查询的 processor group 0 内，先分配不同物理 P 核，再使用 SMT。
  遵守进程 affinity mask；调用线程仅在 compile/run 期间临时绑定，并恢复原 affinity。
  拓扑或系统调用不可用时使用 OS 调度。
- worker 异常可返回 Rust，之后可以再次运行；部分线程创建失败时也会停止并 join 已创建线程。

测过 2/4/6/8 线程、0/50/200 µs spin，以及 4/6 线程的物理 P 核绑定。不同轮次的 CPU 频率和
调度波动明显；spin 和强制绑定没有稳定收益。默认仍为最多 6 线程、spin=0、OS 调度，保留实测调参接口。
不要从一次线程数 sweep 把某个结果推广到其它 CPU 或持续播放状态。

最终连续对照：相同真实语音、VFRM、固定噪声、index=0，P 核进程 affinity，6 线程，各预热一次、30 次推理，
没有逐节点 timing 和 tensor trace。`target/cpu-memory/final-paired.json`：

| 窗口 | 旧平均 | 新平均 | 旧 p95 | 新 p95 |
|---|---:|---:|---:|---:|
| 11200 | 217.27 ms | 216.19 ms | 234.42 ms | 234.22 ms |
| 44960 | 326.63 ms | 321.87 ms | 348.57 ms | 345.85 ms |

收益主要是内存和减少分配。延迟差约 0.5%/1.5%，不视为稳定的大幅加速。
两种窗口最后一次推理音频与旧 oneDNN 逐样本一致：cosine=1、RMSE=0、max_abs=0。
11200 管线的播放 chunk 为 8192/16k=512 ms；44960 管线的播放 chunk 为 4000/16k=250 ms，
后者的本轮完整推理仍超过播放周期。

完整 WAV 与独立 Applio FP32 比较：最终 cosine **0.9999998644**、RMSE **1.064e-4**、
max_abs **9.702e-4**，coarse F0 **0/298** 不一致。见 `target/cpu-memory/applio-parity/`。
此项只共享 latent/NSF noise；特征和 F0 独立计算。

## W8A8 精度实验

新增 `scripts/cpu_quantization.py` 和 `scripts/experiment-cpu-w8a8.py`。
以原始 Applio FP32 Synth/RMVPE、本地 HF HuBERT 为参考，关闭 TF32；CUDA FP32 仅用于加快
fake quant 精度实验，不把它的耗时称为 CPU INT8 性能。

实现 scalar activation scale + per-output-channel weight scale，signed grid `[-127,127]`，
FP32 bias/输出。Conv、分组 Conv、Deconv、Linear 的权重轴分别处理。
SmoothQuant 的变换是 `x'=x/s, W'=W*s`，`s` 根据校准的逐输入通道最大值计算；
Conv 的分组和 Deconv 的 IO 布局也按输入/输出通道分别归一化。
采用这一方式是为了能映射到真正的 INT8 dot product，没有使用不可直接映射到普通 Conv/GEMM 的
每输入通道 activation 量化来宣称 W8A8 加速。

校准窗口：waveform 偏移 32000/112000；测试窗口：208000/352000。
每个偏移分别测试 11200 和 44960，两组互不重叠。使用一个真实声音模型、speaker=0、index=0；
每个窗口的 latent/NSF noise 固定。测试每个量化位置后都重新生成最终音频。
这里的最优 alpha 是小规模实验筛选结果，尚无独立第三组验证集或多声音模型验证。

RMVPE 的标准参考保留原始 Conv→BN。仅候选模型折叠 BN，并单独记录 FP32 折叠误差：
最终音频 RMSE 7.68e-7–3.51e-6、coarse F0 无变化。量化误差表均相对于原始参考。

63 组静态量化结果：`target/cpu-w8a8/applio-accuracy.json`。
表中 cosine 是四个测试窗口中的最低值，RMSE 是四个窗口的平均，max_abs 为最坏窗口。

| 量化位置 | 配方 | 最低最终 cosine | 平均音频 RMSE | 最大绝对误差 | coarse F0 变化 |
|---|---|---:|---:|---:|---:|
| RMVPE intermediate，33 Conv | 普通 W8A8 | 0.999999995 | 5.45e-6 | 1.27e-4 | 0 |
| RMVPE encoder | 普通 W8A8 | 0.976200 | 2.15e-2 | 0.309 | 14 |
| RMVPE decoder | Smooth α=0.7 | 0.998333 | 6.32e-3 | 0.0624 | 3 |
| RMVPE logits | 普通 W8A8 | 0.999941 | 8.21e-4 | 0.0130 | 2 |
| Synth noise Conv | 普通 W8A8 | 0.999999680 | 9.45e-5 | 0.00143 | 固定 F0 |
| Synth flow | Smooth α=0.7 | 0.999987927 | 4.98e-4 | 0.0167 | 固定 F0 |
| Synth 第三组 residual（stage 2） | Smooth α=0.7 | 0.999939580 | 1.21e-3 | 0.0106 | 固定 F0 |
| Synth 全部 residual | Smooth α=0.7 | 0.999229016 | 4.97e-3 | 0.0672 | 固定 F0 |
| HuBERT attention Linear | Smooth α=0.5 | 0.999858214 | 1.77e-3 | 0.0603 | 固定 F0 |
| HuBERT FFN | Smooth α=0.7 | 0.997318121 | 9.62e-3 | 0.129 | 固定 F0 |
| HuBERT feature extractor | Smooth α=0.5 | 0.955790090 | 3.15e-2 | 0.204 | 固定 F0 |

补测了 18 组每次调用动态 scalar activation scale（`dynamic-accuracy.json`，早期 folded-BN 参考）：
全部 Synth residual 最低 cosine 0.999350，stage 2 为 0.999947。动态 scale 没有普遍解决误差，
且运行时增加一次 max reduction。HuBERT FFN 的部分层 `max/p99.9` 达 10–26 倍，存在明显离群值；
SmoothQuant 能改善，但调高 alpha 也会恶化权重和测试窗口上的 clipping，不宜全局固定一个 alpha。

## 真正 CPU INT8 性能与 ISA

`bench_onednn_int8` 使用 oneDNN 的 S8 src/S8 packed weights、S32 accumulator、缩放后的 FP32 dst。
参数通过 `set_scales_mask` 和 `DNNL_ARG_ATTR_SCALES` 传入。权重在构建时量化并打包；
部分 packed GEMM 不支持逐输出通道的 FP32→S8 reorder，因此先量化 canonical 权重，再做 S8→S8 layout reorder。
输入 quantization 与 layout reorder 的成本包含在 boundary 时间内。

真实的 103+105 个 Conv/Deconv/静态 GEMM shape 均完成测试；每例独立核对采样的整数点积，
没有 scale/layout mismatch。报告 `target/cpu-w8a8/operators.json`，本机实现为
`brg_conv_fwd:avx2_vnni` / `brg_deconv:avx2_vnni` / `brg_matmul:avx2_vnni`，effective ISA=15。

| 窗口 | 阶段的 Conv/Deconv/静态 GEMM 加权 boundary 时间 | INT8 | 算子级比例 |
|---|---:|---:|---:|
| 11200 | HuBERT 20.29 ms | 8.14 ms | 2.49× |
| 11200 | RMVPE 17.44 ms | 12.57 ms | 1.39× |
| 11200 | Synth 139.02 ms | 60.94 ms | 2.28× |
| 44960 | HuBERT 80.80 ms | 31.80 ms | 2.54× |
| 44960 | RMVPE 44.16 ms | 24.27 ms | 1.82× |
| 44960 | Synth 72.55 ms | 35.59 ms | 2.04× |

这是按代表 shape 的 occurrences 求和的算子实验，输入来源为已捕获的模型 tensor。
部分 shape 将多个不同层合并，且每例独立调度、独立 Q/DQ；不包含所有算子，也没有部署 SmoothQuant
的通道缩放开销。校准使用该 case 本身，不能从这些数值推导完整管线的加速比例或精度。

更值得组合考察的两处：

- RMVPE 512 通道 3×3 小空间代表 Conv：11200 下 0.124→0.099 ms；44960 下 0.338→0.174 ms。
  对照手写图确认：intermediate 包含 1 个 256→512 的 1×1 Conv、1 个 256→512 的 3×3 Conv，
  以及 31 个 512→512 的 3×3 Conv；对应 node 138/140/142 及后续 residual，共 33 个 Conv。
  31 次 occurrences 均属于 intermediate，但代表输入不能反映每层不同的数值分布和缓存状态。
  intermediate 的 74,448,896 个权重若从 FP32 改成 INT8，原始参数存储有约 **213 MiB** 的节省空间。
- Synth stage 2 的代表 Conv 加权和：11200 为 26.12→11.53 ms，44960 为 14.14→7.51 ms。
  该范围有算子级收益，但当前音频误差未达到此前 0.99999 的数值标准。

强制普通 AVX2 的补测：

- signed Conv 的全正整数点积极值测试，正确结果 16.5，实际 16.625984。
  该路径对 signed 权重有额外调整/舍入；简单把范围降到 63 仍未解决该例。
- 使用 `[-126,126]` 偶数权重编码消除了额外整数点积误差，但权重实际上只有约 6 bit 幅值精度，
  需要另做音频精度校准。
- HuBERT FFN GEMM 即使用偶数编码也为 3.69 ms，而 FP32 0.232 ms；不能在没有 VNNI 的 CPU 上
  假设同样的 INT8 配方有收益。见 `avx2-full-range.json`、`avx2-reduced-range.json`、`avx2-even-grid.json`。

## 取舍

后续已经接入真实整链 W8A8，并扩大到 18 个跨模块配方、两个固定 shape；
最新实测与接口见 [CPU W8A8 扩大覆盖实验](cpu-w8a8-expanded.md)。下面的差值预算保留为原始估算，
不能替代后续整链成对测量。默认 CPU 配置仍为 FP32。

### 量化粒度与整管线收益估算

当前主实验是静态 PTQ：权重使用对称 S8、逐输出通道 scale；每层输入 activation 使用一个
对称 S8 scalar scale，由独立校准窗口决定。Conv 的权重 scale 覆盖该输出通道的全部输入通道
和 kernel；Deconv 按真实输出通道而非 PyTorch 存储第一维确定轴。补测的动态版本只是在每次
调用时重新计算整个输入 tensor 的 absmax，不是 per-token/per-channel/blockwise 量化。
SmoothQuant 的逐输入通道缩放来自静态校准，缩放后 activation 仍使用一个 scalar scale。
静态/动态 scale 与 11200/44960 的固定输入 shape 是不同配置维度。

以 `final-memory.json` 中 FP32 整管线 192.75/288.00 ms 为基线，使用算子 boundary 加权差值
估算 `T_new = T_baseline - (T_region_fp32 - T_region_int8)`：

| 候选 | 11200 估算总延迟 | 44960 估算总延迟 | 当前精度依据 |
|---|---:|---:|---|
| 仅 RMVPE intermediate | 191.97 ms，降低 0.4% | 282.83 ms，降低 1.8% | 最坏音频 cosine 0.999999995 |
| 仅 Synth stage 2 residual | 178.17 ms，降低 7.6% | 281.37 ms，降低 2.3% | Smooth α=0.7，最坏 cosine 0.999939580 |

RMVPE intermediate 的加权算子耗时为 3.986→3.200 ms / 10.725→5.553 ms。
以上是跨实验条件的算子差值预算，不是已部署 INT8 整图实测；stage 2 的通道缩放开销尚未计入，
实际收益可能更小。两种候选的误差和收益不能直接组合，仍需整图验证。正式 CPU 后端保持 FP32。
RMVPE intermediate 的主要收益是权重存储理论上减少 213 MiB，不能等同于进程内存必定减少同样大小。

作为 Amdahl 上限示例，假设整个 Synth 都能提速 2 倍、其余阶段不变，整管线也只能降到
128.13/242.71 ms，约 1.50×/1.19×。当前精度实验不足以支持整个 Synth 使用这一量化配方。

1. 优先继续验证 **RMVPE intermediate W8A8**：本轮音频误差小、粗 F0 不变，权重存储收益大。
   STFT、GRU、encoder/decoder 和 logits 保留 FP32，扩大语音/噪声/F0 范围后再接正式配置。
2. Synth stage 2 和 flow 是后续候选；先按层筛选、测试组合误差，并把 Smooth 的额外乘法融合进 Q/DQ，
   再测整图。不能将每个局部实验的 cosine 或速度直接相乘。
3. HuBERT attention Linear 可继续实验；feature extractor 和 FFN 不适合按当前配方整块启用。
4. CPU 量化策略必须记录目标 ISA、权重轴、量化范围、scale、必要的 Smooth scale、校准来源和参考版本。
   当前没有修改 VFRM schema，也没有把 fake quant 权重当作已经可用于生产的 INT8 cache。

接口依据：[oneDNN 3.10 量化属性](https://uxlfoundation.github.io/oneDNN/v3.10/dev_guide_attributes_quantization.html)，
[SmoothQuant 原论文](https://arxiv.org/abs/2211.10438)。Conv 的使用与收益是本项目实验结论，论文主要讨论 LLM。

## 复现

```powershell
rtk cargo build -p vocal-cpu --example benchmark
rtk proxy powershell -NoProfile -File scripts/setup-cpu-operators.ps1 -Int8BenchOnly
rtk proxy .venv/Scripts/python.exe scripts/benchmark-cpu-memory.py --threads 6 --output target/cpu-memory/current.json
rtk proxy .venv/Scripts/python.exe scripts/benchmark-cpu-memory.py --threads 4 6 8 --spin-us 0 50 200 --output target/cpu-memory/tune.json
rtk proxy .venv/Scripts/python.exe scripts/experiment-cpu-w8a8.py --output target/cpu-w8a8/applio-accuracy.json
rtk proxy .venv/Scripts/python.exe scripts/benchmark-cpu-int8.py --output target/cpu-w8a8/operators.json
rtk cargo build -p vocal-fast --features cpu-graph
rtk proxy .venv/Scripts/python.exe scripts/validate-cpu-migration.py --input target/compare/cpu-input.wav --output-dir target/cpu-memory/applio-parity --device cuda
```

`setup-cpu-operators.ps1 -Int8BenchOnly` 仅构建独立 oneDNN/INT8 probes，无需 CPU Torch SDK。
默认 `setup-cpu.ps1` 不安装 Torch runtime。原始 CPU benchmark 的 operator assets/FP32 reference
需要先按 `docs/cpu-operator-experiments.md` 捕获。真实权重、音频、DLL、trace 和 JSON 均留在本地，不入 Git。

验证：`cargo fmt --all -- --check`、workspace all-features check、workspace tests（187 passed，
13 ignored），Python 编译检查及 3 个量化轴/BN/零输入测试均通过。Rust 测试包括常量共享与 marked
output、重放输出 buffer、worker 异常恢复/跨线程移动/析构，以及 caller affinity 恢复。
