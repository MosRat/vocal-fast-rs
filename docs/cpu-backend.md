# VFRM CPU 后端

当前硬件配方、相对优化后 FP32 的量化结果、播放预算和 Rust preset 接口见
[CPU 推理配方与当前瓶颈](cpu-inference-recipes.md)。

后续默认已接入 HuBERT temporal norm 融合、预构建 reshape reorder 和按行
GatherElements；最新测量及量化/SOLA 敏感性见
[CPU 热点融合与滚动窗口量化](cpu-hotspot-fusion-recipes.md)。下面的 F0 缓存
延迟为这些优化之前的数据。

进一步的 FP32 residual post-op、BRGEMM/1×1 投影及形状分派见
[CPU FP32 卷积融合与投影分派](cpu-fp32-projection-fusion.md)。它们通过
`CpuConfig.convolution` / `linear_mode` 显式选择，当前默认保持 Direct/MatMul。

44960 实时入口已默认启用最近 4960 samples RMVPE + 1024 帧 F0 cache；
FP32 真实 Engine 平均/P95 为 209.84 / 223.89 ms，最终 WAV 对 Python cosine
0.999999980。实现、精度与复现见 [实时 F0 缓存](cpu-realtime-f0-cache.md)。

## 实现

执行完整的 `HuBERT → RMVPE → IVF 检索 → F0/特征对齐 → RVC Synth`。
直接读取规范化 VFRM 张量；运行时没有 Python、TorchScript、ONNX 或 CUDA。
目前支持 RVC v1 / 256 维和 v2 / 768 维特征，按 Synth 元数据自动选择
HuBERT 第 9 层加训练投影或第 12 层。转换与三后端实测见
[RVC v1 支持](rvc-v1-support.md)。基线计算精度为 FP32。VFRM 内的 FP16/BF16
权重在建图时扩展为 FP32。扩展不会恢复在缓存生成时丢失的精度。

执行层已迁移到独立 oneDNN 3.10.2 + RustFFT + 自有张量/SIMD 内核。
oneDNN 使用 THREADPOOL、GPU NONE、INFERENCE 配置；运行时不再链接 Torch、
c10、MKL、OpenMP 或 CUDA。Windows 的唯一算子 DLL 是约 17.34 MiB 的 `dnnl.dll`
（当前 SDK 包含实验用 INNER_PRODUCT）。
可通过 `VOCAL_CPU_ONEDNN_ROOT` 或 `ONEDNN_ROOT` 选择兼容的 THREADPOOL SDK。
独立 oneDNN / XNNPACK 的真实算子性能、数值和体积实验见
[CPU 算子库实验](cpu-operator-experiments.md)。

迁移实测、最终音频和可复现命令见 [独立 CPU 后端迁移](cpu-standalone-migration.md)。

| 层次 | 代码 | 职责 |
|---|---|---|
| 模型与 IR | `vocal-graph` | 固定形状、张量合同、精度描述、手写模型结构；DirectML 与 CPU 共用 |
| 配置 | `vocal-cpu/src/config.rs` | 线程数、融合、AVX2、激活布局、窗口与裁剪 |
| 验证与降低 | `validation.rs`、`planner.rs` | 拒绝非法描述，融合模式识别，删除不再需要的节点 |
| FFI 与所有权 | `ffi.rs`、`runtime.rs` | 错误边界、绑定长度、独占图、原生资源 RAII |
| 原生执行 | `native/runtime.cpp`、`tensor.cpp` | 常量折叠、stride 视图、生命周期 buffer 复用、绑定 |
| 算子缓存 | `native/dnnl_ops.cpp`、`attention.cpp`、`gru.cpp` | oneDNN primitive、权重打包、scratch 复用 |
| FFT | `native/stft.cpp`、`src/fft.rs` | RustFFT plan、每 worker 缓冲、mel/log |
| 基础算子 | `native/elements.cpp`、`simd.cpp`、`threadpool.cpp` | 广播、索引、归一化、AVX2、持久线程池 |
| 模型串联 | `pipeline.rs`、`retrieval.rs` | 控制、随机采样、F0、IVF k=8、模型阶段计时 |
| 音频边界 | `vocal-engine/src/cpu_backend.rs` | 公共重采样、固定 chunk、暖机、SOLA、crossfade |

### 优化和缓存

- HuBERT 的 FP32 attention 默认直接写入 token-major 物理布局，后续
  transpose/reshape 成为视图。`CpuConfig.attention` 可选择 dense 或缓存分块
  online softmax；后者省工作区但 140 帧下更慢，故保持 dense 默认。
  单独 HuBERT W8A8、在线算法和全链精度/延迟结果见
  [HuBERT 量化与缓存 attention](hubert-cpu-quantization-and-attention.md)。
- FP32 GEMM+bias 使用 oneDNN matmul；固定权重的连续 batch 展平为二维 GEMM。
- Conv/Deconv 权重在建图时打包为 primitive 指定布局；激活保留 NHWC。
  必要的边界 reorder 只构建一次。单消费者的 Conv → view → ReLU/Leaky 使用 post-op。
- 相同布局的 residual add + LeakyReLU 使用 AVX2 单遍内核，包含非整倍数尾部；
  广播按物理轴合并连续区域。无 AVX2 或不符合布局时使用基础算子。
- IVF 距离使用 Rust AVX2 分派，扫描选中列表，不创建全索引距离矩阵。
- GRU 的规范 z/r/n 顺序映射到 oneDNN LBR GRU 的 update/reset/output 门；
  保留两个 candidate bias，匹配 reset-after 语义和双向输出顺序。
- RMVPE 使用原生周期 Hann、reflect STFT、FFT、mel 和 log。
- VFRM 来源通过现有共享加载器缓存；原生图持有自有权重。各阶段固定图重复使用。
- 输入原生内存对齐并复用；输入切片仅在调用期间拷贝，不保留 Rust 指针。
  中间张量按最后一个消费者复用，视图延长源 buffer 生命周期。
  primitive scratch 和 attention 临时矩阵在顺序执行的节点之间共享。
  输出读取为调用者持有的字节/FP32 数据。
- 这是固定图和 primitive 缓存，仍逐算子调度，不是整图 JIT。

同线程数的图共享持久线程池；不同线程数的图可以共存。oneDNN 线程级 planning
策略在调用期间设置并恢复。图对象可移动到 worker，执行要求 `&mut self`，不能并发重入。
暂未提供持久化 CPU 编译计划；规范化权重缓存继续使用 VFRM。

## 构建与调用

```powershell
rtk proxy powershell -NoProfile -File scripts/setup-cpu.ps1
rtk cargo build -p vocal-fast --features cpu-graph
rtk proxy target/debug/vocal-fast.exe model run-pipeline cpu target/default-complete-with-index.vfrm target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav target/compare/cpu.wav
```

准备脚本编译固定 oneDNN 3.10.2 SDK 到 `%LOCALAPPDATA%/vocal-fast/cpu-operators/onednn`，
构建自动部署 `dnnl.dll`；Windows 打包包含许可证。RustFFT 静态链接。
Python/CUDA 环境只用于参考验证和旧算子库实验。

Rust 接口：`CpuPipeline::from_sources(&CpuSources, CpuConfig)`；生产音频引擎：
`EngineConfig::with_cpu_graph(model, CpuEngineSpec)` 或 `CpuEngine::from_sources`。
`CpuConfig` 可序列化，开关通过结构体语义化传入。CLI 额外接受
`VOCAL_CPU_THREADS`、`VOCAL_CPU_FUSE`、`VOCAL_CPU_CHANNELS_LAST`、`VOCAL_CPU_PROFILE`。
`VOCAL_OFFLINE_TRACE_DIR` 只用于诊断张量输出，不应在正常性能测量时启用。

Tauri 可构建 `--features cpu-graph`；Runtime 在载入完整 VFRM 后选择实际 CPU 引擎。
`scripts/build-windows.ps1 -Backend Cpu` 会选择 CPU 图、部署必要 DLL 和许可证。
未启用该 feature 时，CLI 与 Runtime 的真实模型转换明确报错。
旧 `EngineConfig::portable(RvcCpuPipeline, ...)` 仍用于无权重的流程 fixture。

### 实时几何

| 配置 | 输入窗口 | HuBERT 实际帧 | Synth 帧 | skip | 播放 chunk | crossfade | SOLA |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU 默认 | 11200 | 34 | 70（追加末帧） | 8 | 8192 / 512 ms | 1365 | 192 |
| 长窗口可选 | 44960 | 140 | 281 | 250 | 4000 / 250 ms | 800 | 160 |

表中样本单位均为 16 kHz。使用真实特征重复两次，并在实时模式追加末帧；离线
没有追加帧。512 ms 是播放 chunk 时长，不是计算延迟或完整采集到播放延迟。
默认 warmup 为 3 个 chunk；实际语音开始播放仍包含缓冲和历史准备时间。
44960 通过 `CpuConfig::realtime_44960()` 或 `CpuEngineSpec` 选择，CLI 流测试可设
`VOCAL_CPU_WINDOW_SAMPLES=44960`。

## 迁移前的 ATen 真实模型结果（历史基线）

2026-10-08，本机 i5-13600K，6 个算子线程，默认声音模型，FP32 计算。

### Applio 数值对齐

与原始 PTH + 原始 HuBERT/RMVPE 的 Applio CPU FP32 对照。双方独立提取特征、
F0 和检索；只共享 latent/NSF 随机张量，避免随机性混入模型误差。
离线采用公共 Applio 对齐的预处理、reflect padding 和裁剪，index=0、protect=0.5、
RMS mix=1、自动变调关闭。另测了 index=0.5。

| 测例 | 最终 WAV cosine | RMSE | max abs | 粗 F0 不一致 |
|---|---:|---:|---:|---:|
| 1 秒中文，index=0 | 0.999999917 | 8.55e-5 | 8.45e-4 | 0 / 298 |
| 1 秒中文，index=0.5 | 0.999999921 | 8.25e-5 | 7.90e-4 | 0 / 298 |
| 27.1 秒中文，index=0 | 0.999995204 | 4.68e-4 | 0.00924 | 0 / 2908 |

长音频 HuBERT RMSE `5.11e-7`，F0 RMSE `6.09e-5 Hz`。
同输入 Synth cosine `0.999999807`、RMSE `9.27e-5`。
端到端误差包含 F0 微小差异随时间累积的相位影响，不能用单个声学张量的误差
代替最终音频指标。以上仅覆盖当前声音模型，不是所有声音的准确性保证。

产物：

- `target/compare/cpu-full-final.wav`：完整 27.08 秒输出，40 kHz。
- `target/compare/cpu-full-applio.wav`：同随机输入的 Applio 对照。
- `target/compare/cpu-final2-parity.json`：完整数值报告。
- `target/compare/cpu-retrieval-parity.json`：带检索对齐。
- `target/compare/cpu-stream-11200.wav`：实际 engine/缓冲/SOLA 路径。

### 性能

固定图短测（暖机后、不下载调试张量）：11200 总耗时约 200 ms；44960 约
250–330 ms，持续运行存在约 440–450 ms 的慢段。不能据短测保证长时间实时性。
通道布局调整在最初同条件测量中，使 11200 从 269.5 降到 199.3 ms；44960 从
293.7 降到 265.3 ms。不同融合配置的差异较小，部分 Synth 测例没有收益。

实际公共 CPU engine 流测试：6 秒中文，11200 窗口，12 次推理，单次均值
`208.22 ms`、最大 `305.11 ms`；包含音频边界的处理总时间 `2.524 s`。
测试末块补齐为 98304 样本，写文件时裁回原来的 96000 样本。
44960 的同类流测试约 `9.185 s / 6 s`，当前 250 ms chunk 的持续实时余量不足，
因此 CPU Runtime 和 CLI 默认选择 11200。正式设备欠载/抖动仍需实际采播验证。

27.1 秒完整离线转换：处理约 `13.527 s`，建图/载入约 `1.147 s`，合计 `14.674 s`。
同随机输入的 Applio CPU 推理约 `16.510 s`；参考未对线程配置做同条件优化，
该时间仅用于记录，不能作为严格的算子库速度比。

当前热点仍为 Synth 的 dilated residual Conv 和上采样，44960 还有 HuBERT
投影/FFN GEMM 与 RMVPE 卷积。提供逐节点计时用于进一步定位；正常执行关闭它。

```powershell
rtk cargo build -p vocal-cpu --example benchmark
rtk proxy .venv/Scripts/python.exe scripts/benchmark-cpu.py --synth target/default-canonical-with-index.vfr --hubert models/shared/hubert_base/hubert_features.vfr --rmvpe models/shared/rmvpe/rmvpe.vfr --waveform target/compare/cpu-final2-trace/0000/waveform.bin --iterations 10 --output target/compare/cpu-benchmark.json
```

性能任务应串行执行，避免多个 CPU 推理进程互相干扰。基准使用固定波形和正常
随机采样；`--threads` 接受多个值，每组起独立进程。不要把一次短测与持续采播混同。
离线目前与其它原生适配器一样，仅支持一个 Applio segment；超过 41 秒会明确拒绝，
尚未加入跨 segment 的全文件 F0 缓存。

验证：workspace all-features check/test、fmt check、Tauri cpu-graph check。
184 个测试通过，18 个设备/资产测试忽略；CPU 原生算子测试实际运行，覆盖融合
归一化、注意力、AVX2 尾部、索引/相位、绑定错误恢复、非法描述与未对齐输入。
