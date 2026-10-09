# 连续中文男声：离线变声与实时调度诊断

日期：2026-10-07。测试设备：Windows、RTX 3060。

本文记录修复前的诊断。后续实现及真实推理节拍验证见
[实时 GUI 音频修复](realtime-gui-audio-fix.md)。

## 结论与范围

已用两段连续男声完成真实 DirectML 推理，并生成原始 Applio FP32 离线参考。
当前最明确的实时断续风险是 **推理突发耗时超过输出库存**；沙哑、机械音的重要
嫌疑是 **310 ms 滚动 F0 上下文**。另外，DirectML 输出中存在确定的 RMS 配置语义
问题，弱音会被进一步压低。

本轮完成了真实模型推理、逐窗口数值对照、DSP 单因素实验，以及使用真实 SPSC
队列的软件调度回放。机器没有可用 capture endpoint，未录制实际麦克风到扬声器
链路，也未做主观试听。这里的指标不能证明某一项是全部听感问题的唯一原因。
本轮没有更改用户保存的 GUI 配置或生产引擎行为。

## 1. 输入、配置与试听文件

输入来自先前下载的完整男声故事录音，截取的是连续时间段，没有拼接短句或删除
内部停顿。来源、许可、完整录音和截取记录保存在
`target/test-assets/mandarin-continuous/manifest.json` 及该目录的
`originals/`、`metadata/`。

| 输入 | 时长 | 完整录音中的截取位置 |
| --- | ---: | --- |
| `male-01-liangzhu-16k.wav` | 57.462 s | 6.596–64.058 s |
| `male-02-mulan-16k.wav` | 70.3870625 s | 106.748–177.135 s |

保存的 GUI 配置快照位于
`target/compare/continuous-male-quality/gui-config-at-run.json`：

- 模型：`target/default-complete-with-index.vfrm`，speaker 0。
- 后端：DirectML，RTX 3060；逻辑输入、输出均为 48 kHz 单声道。
- 控制：pitch 0、index ratio 0、protect 0.33、rmsMixRate 1。
- 设备回放块：480 frames / 10 ms；输入队列 3 块，GUI 输出队列最多 2 块 / 20 ms。
- 模型窗口：44960 个 16 kHz samples / 2.81 s；每次推进 4000 samples / 250 ms。
- 默认 F0：滚动 RMVPE，窗口 4960 samples / 310 ms。
- 默认混合精度：HuBERT FP16、feature extraction FP32、RMVPE FP32；Synth
  decoder stages 0–2 FP16，stage 3/post/encoder/flow FP32。

下列文件可直接对照播放。它们只增加整段恒定增益，目标 RMS 0.1、peak 上限 0.9；
没有改节奏、裁剪或删静音。两份当前输出保留了启动静音。

| 片段 | 当前 GUI 引擎回放 | Applio FP32 完整离线参考 |
| --- | --- | --- |
| 梁祝 | [当前输出](../target/compare/continuous-male-quality/listening/liangzhu-current.wav) | [Applio 参考](../target/compare/continuous-male-quality/listening/liangzhu-applio.wav) |
| 木兰 | [当前输出](../target/compare/continuous-male-quality/listening/mulan-current.wav) | [Applio 参考](../target/compare/continuous-male-quality/listening/mulan-applio.wav) |

另有只扩大 F0 窗口的 [梁祝诊断输出](../target/compare/continuous-male-quality/listening/liangzhu-full-f0.wav)。
这里的 `full-f0` 指每次使用完整 **2.81 s 实时模型窗口**，不是全文件 F0。

未加增益的主输出保存在 `target/compare/continuous-male-quality/`：

- `male-01-liangzhu-gui48-release.wav`、`male-02-mulan-gui48-release.wav`：
  Release RuntimeContext，实际 48 kHz / 10 ms 公共接口，无设备、无 pacing 的文件回放。
  输出与输入样本数相同，未额外 drain 流式尾音。
- `male-01-liangzhu-applio-offline.wav`、`male-02-mulan-applio-offline.wav`：
  原始 Applio FP32、原生 40 kHz 输出，使用 default-v2 PTH 和相同控制参数；完整
  离线流程包含高通、reflect padding、全文件 F0 和长段切分。

Applio 输出分别为 57.44 s、70.36 s。小幅时长差来自 feature extent 的离线裁剪，
不能直接解释为实时丢音。两条路线的上下文和 DSP 不同，也不适合直接按相位比较 WAV。

## 2. 断续：设备小块与模型批次的调度不匹配

`apps/tauri/src-tauri/src/audio.rs` 将输出 ring 限制到最多 2 个设备块，并使用
`WorkerInputPolicy::Latest`。引擎每累计 250 ms 输入执行一次推理，但公共接口每次
只返回当前 10 ms 所需样本。因此，引擎内部已有的待输出音频仍要通过同一个可能
阻塞于推理的 worker 逐块发布，render 无法直接取到整批结果。

Release 文件回放记录的 **超过 10 ms 的 callback** 耗时如下；不是全部 callback
的 percentile，大多数 callback 只装块或取已生成样本。

| 片段 | p50 | p95 | p99 | max | 总推理回放耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 梁祝 | 46.51 ms | 55.72 ms | 73.59 ms | 75.96 ms | 10.823 s |
| 木兰 | 58.36 ms | 66.34 ms | 69.00 ms | 72.29 ms | 16.262 s |

虽然平均 RTF 小于 1，约 47–58 ms 的典型推理阻塞仍明显超过 20 ms 的 ring 库存。
输出耗尽时补零；输入在此期间堆积，满队列及 latest 消费策略会丢弃捕获块。
连续语音模型的时间轴也会因此被压缩。

使用 `replay_queue_timing`，复用实际 `StreamIo`、`DuplexWorkerAdapter`、
`BlockDrainer`，按 10 ms tick 回放 10 s；输出预填 2 块。记录的 callback 耗时
通过 worker sleep 注入，**此实验没有神经推理和物理设备**。

| 软件回放 | 输出 underflow | 补零时长 | 输入满队列丢弃 | 输入 stale 丢弃 |
| --- | ---: | ---: | ---: | ---: |
| 无耗时对照 | 0 次 | 0 ms | 0 块 | 0 块 |
| Release 记录耗时 | 104 次 | 1040 ms | 34 块 | 70 块 |
| Debug 记录耗时 | 166 次 | 1660 ms | 100 块 | 66 块 |

两份 Release 对照的 device tick 都没有超过一个周期的迟到。
这重现了软件调度风险，**1040 ms 不是用户实际硬件录音测得的丢音量**。

还有两个需要一并处理的代码边界问题：

- `DuplexWorkerAdapter` 发现 sequence gap 后只 reset converter/assembler，没有通知
  engine 清理 history、F0 cache 和 SOLA 状态。
- ring sequence 在成功发布时推进；满队列拒绝写入不会形成 sequence gap，因此
  不能用当前 sequence 独立检测全部捕获时间轴缺口。GUI 另行递增的 processing
  sequence 也不携带捕获丢失信息。

建议把设备 period、模型 batch 和可播放库存分离，保持正常输入的 FIFO 时间轴，
让生成批次提前发布至 render 可消费的 FIFO。播放储备应覆盖推理突发 p99 和调度
裕量；真正无法跟上时，显式传递 discontinuity 和 reset，而不是静默压缩时间轴。
单纯扩大 ring 或降低平均推理耗时不足以完整解决这些边界。

依据：`male-*-gui48-release.json`、`queue-pacing-release.json`、`queue-pacing-debug.json`。

## 3. 沙哑、机械音：短 F0 上下文是重要嫌疑

梁祝同一份未过滤的 16 kHz 输入，比较完整文件 FP32 RMVPE 与 DirectML 310 ms
滚动 RMVPE；排除前 3.25 s startup，保留 5421 个 10 ms 帧。

| 相对全文件 F0 的差异 | 310 ms 滚动策略 | 2.81 s 窗口候选策略 |
| --- | ---: | ---: |
| 清浊音判定不同 | 856 帧 / 15.79% | 703 帧 / 12.97% |
| 双方都为浊音时，频率差 >6 semitones | 74 帧 | 16 帧 |
| 双方都为浊音时，频率差 p99 | 11.923 semitones | 2.089 semitones |

候选列是原始 **FP32 PyTorch 对实时窗口策略的独立评估**，并非候选 DML WAV
逐帧对全文件结果的直接统计。全文件 F0 也是算法结果，不是人工真值；这些是上下文
策略差异，不能称为已确定的 856 个识别错误。

滚动策略的 voiced fraction 为 0.7991，全文件为 0.6796；两者共同浊音帧中
还有 133 帧超过 1 semitone。对齐审计检查 ±10 帧，清浊音差异最小的位置是 lag 0，
不支持用简单的一两帧偏移解释全部差异。接近一个八度的频率跳变、不同的清浊音
判定会改变合成激励，是粗糙、机械听感的重要嫌疑。

已经单独生成实际 DML 的 2.81 s F0 窗口 WAV，其它配置保持 GUI 回放一致。
它的推理突发 p50/p95/p99 为 **60.10 / 81.03 / 87.48 ms**，max 122.31 ms。
更长上下文有数值依据，但增加耗时，不能直接作为当前调度模型的实时默认修复。

依据：`f0-context-report.json`、`f0-lag-audit.json`、
`f0-full-window-reference.json`、`male-01-liangzhu-full-f0-gui48.json`。

## 4. 神经网络数值对照：不支持整体算子或权重算错的解释

梁祝 230 个窗口，排除 13 个 warmup 后有 217 个可发声窗口。以同一 default-v2
来源权重的原始 FP32 HuBERT、Applio RMVPE/Synth 独立执行，并共享 native 随机输入：

| 中间结果 | 指标 |
| --- | --- |
| rolling F0 cache | 最大差 0.000305 Hz，RMSE 0.0000393 Hz |
| HuBERT | cosine 0.99997235 |
| 独立神经网络 audio tail | cosine 0.99995174 |
| 同输入 Synth | cosine 0.99996695，RMSE 0.00075041 |
| 固定 SOLA offset 后的全 WAV | cosine 0.99996178 |

这表明 **同一滚动和 DSP 策略下**，主要神经计算数值高度一致。它不能证明当前
流式策略等价于 Applio 完整离线，也不能排除局部误差或听感问题。

该 stage trace 使用 `target/dml-sources-rolling-with-index.json` 的 canonical
Synth；GUI 主输出使用 complete bundle。本轮没有重新逐 tensor 证明两份容器的
Synth bit-identical，因此不作“所有 GUI 权重已经逐字节验证”的声明。

独立计算 SOLA offset 时，178/217 个窗口选取的 offset 不同，全 WAV cosine
只有 0.3015；固定 offset 后恢复到 0.99996。周期信号的 correlation argmax 对
微小数值差异敏感，低全波形 cosine 不能独立证明音质差。后续可以检查 offset
稳定性及跨块相位，但当前证据不支持把它列为已经证实的唯一根因。

依据：`liangzhu-streaming-parity.json`、`dml-trace/`。

## 5. RMS 有确定的配置语义问题

`DirectMlEngine::infer` 在 `apply_rms_mix` 之前无条件执行
`converted *= rolling_input_rms`。即使 `rmsMixRate=1` 使后一个函数不修改输出，
前面的衰减仍然生效，违反了“1 保持 Synth 音量”的控制语义。

梁祝实际滚动 gain 的 p5/p50/p95 是 **0.00938 / 0.11259 / 0.16595**，约为
**−40.6 / −19.0 / −15.6 dB**。当前原始输出 RMS 为 0.01240，Applio 为 0.09620；
木兰分别为 0.01142 和 0.08403。

DSP 单因素实验使用相同已记录的神经 audio tails 和固定的 native-rate SOLA
offset，只删除这个乘法：RMS 从 **0.012406 增至 0.088250**。这证明了显著额外
衰减；弱音及包络会受影响，但低音量本身不能单独解释机械音。

依据：`dsp-ablations/dsp-analysis.json`，另保留 `native-rate-without-rms.wav`
和对应恒定增益试听副本。

## 6. 重采样、静音和启动行为

普通 `model run-streaming-pipeline` 使用 `MONO_16K` 公共格式，会经历 Synth
40 kHz → 16 kHz → CLI 输出 40 kHz 的额外往返。实际 GUI 48 kHz 回放没有这个
16 kHz 输出往返，因此本报告以 `offline_audio_probe` 的 GUI 格式输出为主。

对 Applio WAV 的纯线性往返实验中，4–8 kHz 能量比例从 0.009986 降为 0.003866，
SNR 为 21.98 dB；12 kHz 正弦下采样产生 4 kHz alias。引擎内部
`audio_adapter::to_internal_rate` 仍用 linear，逻辑 48 kHz → 模型 16 kHz
缺少抗混叠。外层 `StreamingFormatConverter` 已有 sinc FIR，但同采样率设备到
逻辑格式时会 bypass，不能替引擎内部这一步滤波。

当前 GUI 格式回放首次非零分别在 **3.4915 s / 3.5173 s**。13 次 250 ms
warmup 静音和首次装块解释了约 3.49 s 的启动静音；这不能作为后续持续 underflow
的证据。回放保留了实时启动语义，没有额外 flush 尾音。

3.5 s 之后，PCM16 WAV 有 355/213 个全零 10 ms 块，但以 envelope 粗对齐到
Applio 参考后，在参考 RMS >0.02 的位置，两个输出都没有对应全零块。
因此不能把这些静音块全部当作已测到的语音 dropout。无 pacing 文件推理也不会
自动重现设备 render 在 worker 阻塞期间的补零。

依据：`gui-audio-analysis.json`、`dsp-ablations/dsp-analysis.json`。

## 7. 修复优先顺序

1. 重做模型批次发布与播放库存管理，保留输入 FIFO 时间轴；明确 discontinuity
   通知和 engine reset。以推理突发 percentile 验证容量，而非只看平均 RTF。
2. 采用更充分的 F0 上下文策略，验证清浊音、倍频错误和耗时；配合第 1 项一起调整。
3. 移除无条件 input-RMS 乘法，统一 RMS mix 的配置语义和时间包络处理。
4. 修复引擎内部下采样的抗混叠；保持 native-rate 合成、明确重采样边界。
5. 再做 SOLA 稳定性诊断与实际物理 duplex 录音、长时间运行验证。

## 8. 最小复现

以下命令从仓库根目录运行。离线 probe 创建独立 RuntimeContext，不保存 GUI 配置，
也不打开音频设备；输入需匹配配置的单声道输入/输出格式。

```powershell
rtk cargo build --release -p vocal-fast --example offline_audio_probe --features directml
rtk target/release/examples/offline_audio_probe.exe target/compare/continuous-male-quality/gui-config-at-run.json target/compare/continuous-male-quality/male-01-liangzhu-input-48k.wav target/compare/continuous-male-quality/repro-liangzhu.wav target/compare/continuous-male-quality/repro-liangzhu.json
rtk cargo run -p vocal-fast --example replay_queue_timing --features directml -- target/compare/continuous-male-quality/repro-liangzhu.json target/compare/continuous-male-quality/repro-queue.json
rtk .venv/Scripts/python.exe scripts/run-applio-offline.py --input target/test-assets/mandarin-continuous/male-01-liangzhu-16k.wav --output target/compare/continuous-male-quality/repro-applio.wav --checkpoint models/voices/default-v2/default.pth --index models/voices/default-v2/added_IVF511_Flat_nprobe_1_default_v2.index --index-ratio 0 --protect 0.33 --volume-envelope 1 --report target/compare/continuous-male-quality/repro-applio.json
```

当前缓存、模型与 ref 依赖需保留；耗时会随驱动、系统负载、cache 状态而变化。
全部 WAV、trace、配置快照和实验 JSON 位于 ignored `target/`，没有加入 Git。
本轮验证包括 Release/Debug probe 实际执行、上述对照实验、
`cargo check -p vocal-fast --examples --features directml`、
`cargo test -p vocal-fast --features directml`、`cargo fmt --all -- --check`。
CLI package 本身没有单元测试，测试命令通过不能替代这些实际音频实验。
