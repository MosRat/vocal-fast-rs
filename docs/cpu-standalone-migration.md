# 独立 CPU 执行层迁移

2026-10-08，本机 i5-13600K，Windows x64。CPU 后端已从 ATen/LibTorch 迁移到
独立 oneDNN 3.10.2、RustFFT 6.4.1 和自有张量/基础算子/AVX2 内核。
VFRM、公共 `vocal-graph` IR、`CpuGraph`/`CpuPipeline`、engine/runtime/GUI 入口继续使用。

## 执行和资源

| 模块 | 职责 |
|---|---|
| `native/tensor.*` | 64 字节对齐 storage、dtype、stride、reshape/transpose/slice 视图、规范输出 |
| `native/runtime.cpp` | FFI 错误边界、静态生命周期 buffer 复用、常量折叠、Conv 激活融合、输入绑定 |
| `native/threadpool.*` | 持久 worker、同线程数 pool 共享、同步执行、异常传播与恢复 |
| `native/dnnl_ops.cpp` | 缓存 Conv/Deconv/GEMM/激活/LayerNorm/Softmax primitive、打包权重、边界 reorder |
| `native/attention.cpp` | QK → scale → Softmax → PV，顺序节点共享临时矩阵 |
| `native/gru.cpp` | oneDNN 双向 LBR GRU，规范 z/r/n 门、四组 bias、每次零初始状态 |
| `native/stft.cpp` + `src/fft.rs` | 1024 FFT、周期 Hann、center reflect、hop=160、幅值、128 mel 和 log |
| `native/elements.cpp` + `simd.cpp` | 广播、reduce、索引、pool、相位和单遍 AVX2 算子 |

计算保持 FP32。连续 batch 的固定权重 Linear 展平为二维 GEMM。
卷积激活保留 NHWC；singleton stride 使用显式格式标签，避免误选 reference Conv。
布局不兼容时使用已缓存的 reorder，临时存储也固定分配。
单消费者的 Conv → view → ReLU/Leaky 使用 oneDNN post-op；标记原始输出会阻止该融合。

primitive scratch 在顺序节点间共享。attention score/probability 的内存由图共享，
长音频不为每个 transformer block 独立分配两份完整 T×T 矩阵。
线程策略是作用域内的 oneDNN thread-local 设置，不需要进程内所有图使用相同线程数。
FFT plan 共享，但每个 worker 独占输入和 scratch；调用期间只写互不重叠的输出列。

接口对照：[oneDNN LBR GRU](https://uxlfoundation.github.io/oneDNN/v3.10/page_lbr_gru_example_cpp.html)、
[RustFFT scratch API](https://docs.rs/rustfft/6.4.1/rustfft/trait.Fft.html)。

## SDK 与部署

```powershell
rtk proxy powershell -NoProfile -File scripts/setup-cpu.ps1
rtk cargo build -p vocal-fast --features cpu-graph
rtk cargo build -p vocal-cpu --example benchmark
```

SDK 默认位于 `%LOCALAPPDATA%/vocal-fast/cpu-operators/onednn`。
覆盖顺序：`VOCAL_CPU_ONEDNN_ROOT` → `ONEDNN_ROOT` → 默认目录。
SDK 使用 THREADPOOL、GPU NONE、INFERENCE、关闭 oneDNN Graph，仅编译需要的 primitives。
本机 `dnnl.dll` 约 **17.14 MiB**，旧 `torch_cpu.dll` 单文件约 **250.49 MiB**。
RustFFT 和 SIMD 静态链接；PE imports 已确认 CLI 与 `dnnl.dll` 无 Torch、c10、
OpenMP、MKL 或 CUDA 依赖，另外仍需要 Windows/MSVC runtime。

Cargo 自动部署 `dnnl.dll` 到 binary、deps、examples 目录。
`scripts/build-windows.ps1 -Backend Cpu` 部署 DLL 与 oneDNN 许可证，安装包使用显式 allowlist。
旧 target 中残留的 Torch DLL 不代表当前二进制仍依赖它们。
CLI 离线报告包含实际加载的 `operator_library={name:oneDNN, version:3.10.2}`。

## 完整管线性能

实际语音取自中文输入的有效语音段；固定 VFRM 模型，FP32，index=0，自动变调关闭，
噪声由相同种子生成。旧 ATen 与新 oneDNN 独立进程**串行**运行，均限制到 CPU-set
返回的 P 核（logical 0–11）。每组预热一次、测 20 次；不启用逐节点计时或张量 trace。
Cargo dev 二进制，原生 C++ 与两个算子 SDK 均优化构建。下表是**完整模型管线**，不是微基准加权值。

| waveform | 线程 | 旧 ATen 总均值 | 新总均值 | HuBERT | RMVPE | Synth | 新 p95 / max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 11200 | 6 | 224.18 ms | **181.75 ms** | 35.91 | 25.31 | 120.09 | 193.85 / 198.49 ms |
| 44960 | 6 | 291.56 ms | **281.19 ms** | 132.96 | 57.82 | 88.71 | 325.75 / 341.93 ms |
| 11200 | 1 | 640.90 ms | 611.58 ms | 95.32 | 64.95 | 450.88 | 624.24 / 626.16 ms |
| 44960 | 1 | 801.37 ms | 829.59 ms | 373.23 | 172.75 | 281.91 | 843.66 / 850.94 ms |

6 线程下 11200 延迟降低 **18.9%**，44960 降低 **3.6%**。
HuBERT 仍比旧实现略慢；收益主要来自 RMVPE 与 Synth。单线程不是推荐配置。
同输入、同噪声的旧/新最终音频 cosine 分别为 0.9999999984、0.99999999999（6 线程）。
本次只覆盖一个声音模型。

11200 推理生成 62 帧/620 ms 音频，engine 实际推进/播放 **8192 samples / 512 ms**。
44960 生成 31 帧/310 ms，engine 推进 **4000 / 250 ms**。二者 skip 分别为 8 和 250，
不能将总时间当作相同输出长度的对比。11200 默认配置有计算余量；44960 当前均值和
p95 仍超过播放推进间隔，不能保证持续实时。

6 秒中文带 index=0.5 的实际公共 streaming engine 测试也已运行，12 次回调约 2.460 s，
写出 6 秒音频。CLI 报告中的 8192 是 runner 推进长度，join 的 crossfade/SOLA=0；
历史上下文、SOLA 和 crossfade 由 CPU engine 内部执行。尚未重新测量声卡采播欠载。

报告：`target/cpu-migration/speech-benchmark.json`。复现：

```powershell
rtk proxy .venv/Scripts/python.exe scripts/benchmark-cpu-migration.py --waveform target/cpu-migration/speech.f32 --threads 1 6 --iterations 20 --output target/cpu-migration/speech-benchmark.json
```

旧 baseline EXE 需要旧 CPU Torch DLL 可加载。脚本通过其 SDK 的 PATH 加载，
两种生产提供者只在各自进程执行。`speech.f32` 是旧 465600-sample trace 的 `[32000:76960]`。

## 对原始 Applio FP32 的音频验证

直接使用原始 PTH、HuBERT、RMVPE 和 FAISS index。双方独立提取特征、F0、检索；
只共享 latent/NSF 随机张量，使用公共 Applio 高通/reflect padding/离线裁剪。
自动变调关闭，protect=0.5，RMS mix=1，完整上下文；没有使用原生特征替代 Python 的提取。

| 输入 | index | 最终 WAV cosine | RMSE | max abs | 粗 F0 不一致 |
|---|---:|---:|---:|---:|---:|
| 1 秒中文 | 0 | **0.9999999176** | 8.52e-5 | 7.84e-4 | 0 / 298 |
| 27.1 秒中文 | 0.5 | **0.9999973402** | 3.56e-4 | 0.00697 | 0 / 2908 |

长音频 HuBERT RMSE 5.38e-7，F0 RMSE 4.55e-5 Hz；同输入 Synth cosine 0.9999997335，
RMSE 1.10e-4。最终误差还包含 F0 微差的相位累计，因此不能用单阶段误差代替最终 WAV。
长音频带诊断的处理约 13.74 s、含载入/建图约 15.23 s；该数据不是与 Applio 的严格速度比较。

可试听文件：`target/cpu-migration/onednn-full/native.wav`，
参考 `target/cpu-migration/onednn-full/applio.wav`。
误差报告 `onednn-full/parity.json`、`onednn-short/parity.json`。
早期 `short/`、`full/` 报告来自迁移前 CLI，已排除于以上迁移结果。

```powershell
rtk proxy .venv/Scripts/python.exe scripts/validate-cpu-migration.py --input target/test-assets/mandarin-rvc/chinese-input-offline-augmented.wav --output-dir target/cpu-migration/onednn-full --index-ratio 0.5
```

验证脚本拒绝旧 CLI；cosine < 0.99999 或粗 F0 不一致时失败。生成的模型、WAV、
traces、plans、benchmark reports 均保留在忽略的 target/ 或本地 SDK 目录。

## 当前热点和后续范围

11200 的 Synth 占约 66% 的推理时间，热点仍是生成器 dilated residual Conv。
44960 的 HuBERT 占约 47%，优化重点是 projection/FFN GEMM、feature extractor Conv
与局部布局重排。固定 primitive 缓存和静态 buffer 复用已完成，仍可尝试更少重排、
局部 blocked layout、更大的融合块；本次没有加入量化或降低基线计算精度。

迁移回归覆盖 NHWC 与规范输出、共享 buffer 的视图生命周期、重复推理、
Conv post-op、分组 Deconv/output padding、独立标量双向 LBR GRU、独立 DFT/STFT，
以及线程池异常跨 FFI 传播后的恢复。API、格式和现有设备相关限制见 [CPU 后端](cpu-backend.md)。

验证完成：CPU 包 14 项测试；`cargo fmt --all -- --check`；workspace all-features check；
workspace test（183 passed、13 ignored）；Tauri `--no-default-features --features cpu-graph`
check；Python py_compile 和 PowerShell parser。SDK setup 重跑无额外构建。
隔离部署测试只放置 CLI EXE + `dnnl.dll`，PATH 仅 Windows/System32，真实 WAV 转换成功，
确认无需开发目录中的旧 Torch DLL。未重新生成完整 GUI 安装包。
