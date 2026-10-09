# CPU W8A8 扩大覆盖实验

RMVPE 与 HuBERT 组合的最新测量、当前默认图控制、类型化配方接口和实时预算见
[CPU 推理配方与当前瓶颈](cpu-inference-recipes.md)。

## 实验契约

- 本机 i5-13600K，6 个线程，进程限制在 P 核，standalone oneDNN 3.10.2。
- 原始 Applio FP32 Synth/RMVPE + 本地 HF HuBERT 为精度标准，TF32 关闭。
- 一个真实 RVC v2 / 40 kHz 声音模型，speaker=0、index=0、自动移调关闭。
- 11200 / 44960 两个固定 waveform shape，Synth 分别裁去前 8 / 250 帧。
- 校准 offset 32000 / 112000；测试 offset 208000 / 352000；额外验证 offset 272000 / 416000。
  同一 offset 的两种长度包含重叠语音，各组 offset 之间不重叠，不将窗口数当作独立语料数。
- 每个窗口的 waveform、latent noise、NSF noise、参考音频写入共享 fixtures。
  Python fake quant 用于筛选精度；最终结果由真实 native W8A8 完整链路执行，直接对比 Applio 音频。
- native 延迟按同一进程、两个完整 pipeline、交替 FP32/INT8 顺序测量，8 对预热；
  首轮 18 个配方各 24 对，ConvTranspose/BN 对齐后对 4 个相关配方各复测 32 对。
  不启用 per-node profile，不保存中间张量。包括图 IO、activation Q/layout、Smooth 缩放、frontend 和最终输出复制。
  使用 `process_exact` 的阶段计时；固定随机输入的生成、输入有效性检查、加载/编译、WAV 解码和 SOLA 不计入。
  两条管线同时驻留以减少不同进程频率漂移，不能把该进程总内存当作单管线部署内存。

## 粒度与接口

权重逐输出通道量化到 `[-127,127]`，activation 每层一个静态 scalar scale。
oneDNN S8 reorder 的 activation 饱和范围为 `[-128,127]`；新 Python 筛选显式对齐这一范围，
旧 `experiment-cpu-w8a8.py` 保持原有 `[-127,127]` 实验契约。zero point=0，整数累加 S32，bias 和输出 FP32。
没有把反量化 FP32 点积当作 INT8 性能。

SmoothQuant：逐输入通道 `s`，静态变换 `x'=x/s, W'=W*s`。
权重变换在 compile 时完成；`x/s` 和 scalar activation quantization 合入同一个 layout-aware reorder。
普通量化使用 scalar reorder scale，Smooth 使用输入通道 reorder scale；整数 Conv/GEMM 的 source scale 始终为 scalar。
norm、softmax、attention 动态点积、STFT、GRU、NSF 相位/正弦、残差和非线性保持 FP32。
每个量化算子都记录实际 oneDNN implementation；没有 VNNI/AMX 时明确拒绝此实验路径。

Rust 显式实验接口；`fold_deconvolution_batchnorm` 默认 true，可明确关闭以复现原始未折叠图。
图优化开关改变后应重新校准。

```rust
let recipe = CpuQuantization {
    rules: vec![CpuQuantizationRule {
        prefix: "dec.resblocks.6".into(),
        smooth_alpha: Some(0.7),
    }],
    ..CpuQuantization::default()
};
// FP32 校准：多个独立输入执行后调用 calibration()。
let observer = CpuPipeline::from_sources_calibration(&sources, config.clone(), &recipe)?;
// 导出的 calibration 与 rules 一同构成新的 recipe，显式编译量化管线。
let pipeline = CpuPipeline::from_sources_quantized(&sources, config, &calibrated_recipe)?;
```

`CpuGraph` 也提供对应 `compile_calibration` / `compile_quantized` / `calibration`。
模型权重打包、native 参数内存释放、线程池、复用 workspace、FP32 外部 buffer 接口沿用现有 CPU 生命周期。
正式 `from_sources` 保持 FP32。校准需绑定同一模型和图优化配置；实验报告记录模型 SHA256，
resume 同时校验二进制、配方、fixtures、BN 配置和迭代数。
实验文件留在 `target/cpu-w8a8/broad/native/`，没有更改 VFRM schema 或生成未经验证的生产 INT8 cache。

## 覆盖范围

18 个配方：RMVPE intermediate；加入 NSF noise Conv、flow；Synth 第三级 residual；全部 convs1 或 convs2；
后两级 / 全部 residual；全部 generator；HuBERT attention Linear、最后四层 FFN、全部 Linear；
RMVPE 最后两级 / 全部 decoder、全部 U-Net；跨三个模型组合。

最激进配方包括 360 个 Conv/Deconv/Linear，202,086,480 个权重。
三个参考模型参数合计 212,307,407，量化权重占 **95.19%**。
校准实际调用的 Conv/Linear 共 369 层、211,004,497 个权重；排除 HuBERT extractor/position Conv 和 NSF source 小 Linear。
这一比例是参数数量，不是 FLOPs 或耗时占比。

native 每个配方严格核对 Python 选择的层数和权重元素数，避免只量化了部分规则却报告完整配方性能。
修复了 VFRM FP16/BF16 参数转 FP32 后名称丢失的问题，以及数字前缀 `node.34` 错匹配 `node.344` 的问题。

RMVPE decoder 的 ConvTranspose 后仍有独立 BN，Python PTQ 原先已折叠 BN。
CPU 量化/校准图现按同样方式，将独占且被选中的 ConvTranspose/BN 合并成有效权重和 FP32 bias；
默认 FP32 图不变。新测试验证折叠前后 FP32 输出，以及量化后的输出通道和 bias。
最后两级 decoder 的 11200 native cosine 从 0.783528 提高到 0.989998，但整个 decoder 仍有 F0 分叉，
不能据此启用整个 RMVPE decoder 的 INT8。native 静态校准来自实际 CPU 图，float 舍入、权重/scale 的
量化边界和 F0 argmax 仍会使结果与 Python fake quant 不同；采用配方时以 native 对 Applio 的直接结果为准。

## 完整链路实测结果

下表每格为同一进程的 **FP32→INT8 平均延迟 / 加速比例**。
所有 Synth 行均包含 RMVPE intermediate；residual 行还包含 noise Conv 和 flow。
“语音 cosine”取参考 RMS ≥0.01 的窗口最小值；全部窗口误差和低音量尾段另列，不丢弃尾段。

| 配方 | 11200 ms / 加速 | 44960 ms / 加速 | 11200 / 44960 语音 cosine | 11200 / 44960 平均音频 RMSE |
|---|---|---|---|---|
| RMVPE intermediate | 195.80→189.68 / 1.032× | 284.85→276.03 / 1.032× | 0.999999972 / 0.999999987 | 3.55e-5 / 2.36e-5 |
| +noise Conv | 187.11→190.32 / 0.983× | 280.35→280.49 / 0.999× | 0.999999695 / 0.999999853 | 9.25e-5 / 6.28e-5 |
| +flow/noise | 195.23→193.30 / 1.010× | 281.90→279.22 / 1.010× | 0.999956095 / 0.999992223 | 7.93e-4 / 4.29e-4 |
| +第三级 residual | 197.42→183.86 / 1.074× | 285.92→276.44 / 1.034× | 0.999924544 / 0.999974655 | 1.59e-3 / 8.66e-4 |
| +全部 residual 的 convs1 | 213.76→177.40 / 1.205× | 342.66→314.51 / 1.089× | 0.999690505 / 0.999824906 | 3.14e-3 / 2.47e-3 |
| +全部 residual 的 convs2 | 200.15→158.17 / 1.265× | 341.60→307.65 / 1.110× | 0.999237323 / 0.999611001 | 4.53e-3 / 2.72e-3 |
| +全部 residual | 202.60→130.56 / 1.552× | 347.01→296.06 / 1.172× | 0.998740263 / 0.999524379 | 5.73e-3 / 3.53e-3 |
| +全部 generator/flow | 198.24→129.52 / 1.531× | 282.43→246.74 / 1.145× | 0.998701212 / 0.999255396 | 6.55e-3 / 4.64e-3 |
| +HuBERT 全部 Linear，Synth flow/noise | 238.65→217.69 / 1.096× | 283.56→241.70 / 1.173× | 0.995731898 / 0.998762565 | 1.05e-2 / 5.79e-3 |
| 最广 95.19% 参数 | 198.10→111.24 / 1.781× | 285.15→192.40 / 1.482× | 0.950070709 / 0.989044152 | 3.51e-2 / 1.30e-2 |

跨配方的 FP32 绝对延迟仍随 CPU 状态变化，不能跨行相除。加速比例来自该行同进程相邻的成对测量。
按四窗口周期进行 block bootstrap，全部 residual 的本轮加速 95% 区间为 1.523–1.583× / 1.161–1.185×；
最广配方为 1.737–1.823× / 1.462–1.504×。区间仅表达本轮波动，不能推导其它 CPU/语音模型的表现。

### 低音量尾段

44960 / offset 416000 的参考输出 RMS 为 1.844e-4：

| 配方 | 输出 RMS | 音频 RMSE | 全部窗口最小 cosine |
|---|---:|---:|---:|
| RMVPE intermediate | 1.844e-4 | 1.13e-7 | 0.999999812 |
| +第三级 residual | 2.055e-4 | 9.98e-5 | 0.874369 |
| +全部 residual | 7.845e-4 | 7.30e-4 | 0.402789 |
| +全部 generator/flow | 3.102e-3 | 3.12e-3 | -0.075291 |
| 最广配方 | 2.468e-3 | 2.46e-3 | 0.074390 |

generator 全量量化在该尾段把 RMS 放大约 16.8 倍，不仅是 cosine 因低分母下降。
最终语音段的高 cosine 不足以证明尾段的噪声可接受，这些激进配方没有改为默认。

### RMVPE 与 F0

intermediate 单独量化的 coarse/voicing mismatch 为 0，F0 最大差约 0.000412 Hz。
BN 对齐后整个 decoder 在 11200 的最差 cosine 仍为 0.787973，最大音频绝对误差 0.752866；
44960 中最大 F0 差约 128.98 Hz。整个 U-Net / 最广配方也有粗 F0 变化。
小的 salience/argmax 差异可以显著改变 NSF 相位，不能只看局部 Conv 或 logits 的误差。

### 内存与取舍

RMVPE intermediate 的实际 packed 参数字节减少约 **221.8 MiB**；最广配方约 **589 MiB**。
这是 native 参数内存核算，不是进程 RSS，也不同于原始权重大小的理论节省量。

1. 当前要求接近 Applio FP32 的配置，优先只量化 RMVPE intermediate；本轮约 3.2% 的整链加速、内存收益较大。
2. 单独扩大 noise Conv 的覆盖没有收益，不将层数多作为性能改善的证据。
3. Synth residual 是主要的加速方向，但扩大后语音和低音量尾段误差都增加，需要再做按层保留/缩放校准。
4. HuBERT 大范围 Linear 更有利于 44960，不过全 Linear 的语音误差明显高于 attention-only。
5. RMVPE encoder/decoder/head/logits 继续保持 FP32；最广配方用于测试上限，不能作为质量达标的默认配置。

## 结果文件

- `target/cpu-w8a8/broad/accuracy.json`：18 个配方的 Python 完整链路 fake-quant 误差。
- `target/cpu-w8a8/broad/native-results.json`：真实 native INT8 配方，逐窗口 Applio/FP32 parity、F0、每对延迟、内存和实际 kernel。
- `target/cpu-w8a8/broad/folded-bn/native-results.json`：ConvTranspose/BN 对齐后的 4 个配方复测。
- `target/cpu-w8a8/broad/summary.json`：合并后 36 个 shape/配方结果、来源 hash、输出 RMS、F0 和本轮加速区间。
- `target/cpu-w8a8/broad/native/`：静态校准、配方、FP32/INT8 输出及可试听 WAV。

额外验证的 44960 / offset 416000 输出 RMS 约 0.000184，是低音量尾段。该片段仍计入完整误差。
另外报告参考 RMS ≥0.01 的语音窗口 cosine，且保留低音量尾段的绝对误差和输出 RMS；不能只用语音 cosine 掩盖噪声放大。
局部配方的误差和速度不能直接相加，CPU 上的实际 RMVPE F0 解码必须纳入最终音频比较。

## 复现

```powershell
rtk proxy .venv/Scripts/python.exe scripts/experiment-cpu-w8a8-combinations.py
rtk cargo build -p vocal-cpu --example quantization_benchmark
rtk proxy .venv/Scripts/python.exe scripts/benchmark-cpu-w8a8.py --iterations 24 --deconv-bn preserve
rtk proxy .venv/Scripts/python.exe scripts/benchmark-cpu-w8a8.py --directory target/cpu-w8a8/broad/folded-bn --inputs-from target/cpu-w8a8/broad --deconv-bn fold --recipes rmvpe_decoder_late rmvpe_decoder rmvpe_unet broad --iterations 32
rtk proxy .venv/Scripts/python.exe scripts/summarize-cpu-w8a8.py
```

测试包括分组 Conv/Deconv 输出通道轴、Smooth、GEMM 转置/bias、校准生命周期、非法校准、数字前缀隔离、
FP16/BF16 canonical 名称和 Python saturation 范围。下载权重、参考仓库、WAV、校准 JSON 和二进制输出均不入 Git。

验证：workspace all-features check、workspace tests（193 passed，13 ignored）、Rust fmt check、
Python 编译检查及 4 个量化单元测试通过。结果限于本次一个声音模型和一段 Mandarin 语料。
