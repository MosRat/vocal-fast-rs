# 实时 GUI 音频修复

2026-10-08 更新：正常会话已取消下文历史测试使用的 13 批启动静音；
三个神经后端共用新的起音、尾音与对齐 RMS 策略，详见
[实时语音过渡修复](realtime-speech-transitions.md)。

## 调度和数据边界

设备 slot 保持配置中的小周期，例如 48 kHz 下 480 frames / 10 ms。
`InferenceEngine::preferred_batch_frames` 提供当前模型真正的推进大小；Runtime
通过控制 actor 返回数值，engine 仍只由模型 worker 独占。

`DuplexWorkerAdapter::with_batch_frames` 在 worker 内完成设备格式转换和批次拼接，
以 FIFO 顺序消费输入。每次把完整模型批次交给 engine，再逐个发布至输出 ring。
输出满时保留未发布的样本；后续调用先送出这部分，再消费新输入。超时不会丢尾音。
设备 callback 只做非阻塞 SPSC 交换，不等待模型，不持有模型锁。

`RealtimeBufferConfig::for_batch` 根据设备 slot 和模型 batch 计算容量。当前 GUI
DirectML 默认 batch 为 12000 个 48 kHz frames / 250 ms，10 ms slot 对应：

| 项目 | 实际值 |
| --- | ---: |
| 输入容量 | 52 slots / 520 ms |
| 输出容量 | 77 slots / 770 ms |
| render 启动门槛 | 38 slots / 380 ms 音频库存 |

容量是允许的最大积压，不是固定附加延迟。模型整批发布，达到 380 ms 库存通常
需要两批，因此初次 render 会等待约两个 batch 加推理耗时。预缓冲期间输出静音，
不计为欠载；门槛打开后缺失的样本正常计入 underflow。设备引擎 period 不变成
250 ms。Passthrough 等逐设备块后端保留小队列、立即播放。

GUI 路由信息新增实际 `processingBatchFrames`、`playbackPrefillMicros`；诊断的
Processing / budget 以真实模型批次统计。这里的 prefill 是启动库存门槛，不是
硬件延迟测量或持续固定的 ring residency。

## 连续时间轴与会话重置

- GUI 输入策略使用 FIFO，避免推理突发时 `Latest` 丢掉连续语音上下文。
- 捕获满队列丢失会推进 sequence；未提交的 reservation 不会制造音频缺口。
- `BlockAssembler` 会处理并计数一个大 callback 中每个丢失块，保留末尾不完整块，
  而非遇到第一个 overflow 就丢弃剩余 callback。
- Native WASAPI 的 DATA_DISCONTINUITY 标志同样进入 sequence 缺口通知。
- Adapter 清理格式转换和不完整 batch，并在下一次完整 model batch 向 worker
  传递 discontinuity。worker 调用 `reset_stream`，清理 engine history、F0、
  SOLA、待输出数据。
- 每次启动都 reset engine，不复用上一会话残留。DirectML 和 TensorRT 实现真实
  stream reset；CPU pipeline 清理输入历史。

设备异常造成真实丢音后，模型仍需要重新建立上下文。预缓冲不能保证任意持续过载
都没有欠载；容量、queue drop、underflow 继续保留用于诊断。

## DirectML 音质处理

- 自动完整 bundle 的 44960-sample 默认图使用 **FullWindow RMVPE**，即完整
  2.81 s 模型窗口。明确指定 rolling 的外部 source manifest 仍尊重其配置。
- 删除 `converted *= rolling_input_rms`，使 `rmsMixRate=1` 保持 Synth 音量；其它
  RMS mix 参数继续通过共有的 `apply_rms_mix` 路径执行。
- public-rate 输入到模型 16 kHz 的转换采用有状态 sinc FIR，复用
  `StreamingFormatConverter`。相位、滤波历史跨 callback 保留，避免下采样混叠
  和分块舍入漂移；大调用分为有界子块，不重置滤波器。
- 当前 Synth 输出回到 public rate 的转换仍为原有 linear 路径。TensorRT 采用
  共用 GUI batch 调度及 stream reset；本轮没有改它的原生模型图、plan 或 F0 策略。

## 真实推理节拍验证

Windows、RTX 3060，Release，保存的 GUI DirectML 配置和完整 default-v2 bundle。
使用 `realtime_audio_probe` 执行真实 RuntimeContext / engine / batch adapter /
SPSC / BlockDrainer。文件作为输入设备，输入和 render 均按 10 ms 节拍推进；
**没有打开物理麦克风和扬声器**，没有用 sleep 模拟神经推理耗时。

| 测试 | 梁祝 | 木兰 |
| --- | ---: | ---: |
| 原始输入时长 | 57.462 s | 70.387 s |
| 捕获格式 | 48 kHz / mono | 44.1 kHz / stereo |
| 逻辑处理及输出格式 | 48 kHz / mono | 48 kHz / mono |
| 模型耗时 p50 | 60.06 ms | 52.57 ms |
| 模型耗时 p95 | 69.82 ms | 61.47 ms |
| 模型耗时 p99 | 90.86 ms | 69.31 ms |
| 模型耗时 max | 129.38 ms | 102.57 ms |
| 输入 full / stale drops | 0 / 0 | 0 / 0 |
| 输出 full drops | 0 | 0 |
| render underflow samples | 0 | 0 |
| model discontinuities | 0 | 0 |

回放 WAV 和逐项 JSON 位于 ignored `target/compare/realtime-gui-fix/`：

- [梁祝实时节拍输出](../target/compare/realtime-gui-fix/liangzhu-paced.wav)
- [木兰双声道输入实时节拍输出](../target/compare/realtime-gui-fix/mulan-44100-stereo-paced.wav)

文件末尾额外输入 1 s 静音，供最后不完整 batch 及 filter/history 处理。输出保留
现有模型 13 个 batch 的 startup mute 和播放预缓冲，不作静音裁剪。梁祝首次非零
约 3.80 s，原始输出 RMS 0.08694、peak 0.75345；启动等待仍是独立的体验限制。
这些结果验证软件实时路径及不同格式组合，不替代硬件 duplex 或主观音质验证。

```powershell
rtk cargo build --release -p vocal-fast --example realtime_audio_probe --features directml
rtk target/release/examples/realtime_audio_probe.exe target/compare/continuous-male-quality/gui-config-at-run.json target/compare/continuous-male-quality/male-01-liangzhu-input-48k.wav target/compare/realtime-gui-fix/liangzhu-paced.wav target/compare/realtime-gui-fix/liangzhu-paced.json
rtk target/release/examples/realtime_audio_probe.exe target/compare/continuous-male-quality/gui-config-at-run.json target/test-assets/mandarin-continuous/male-02-mulan-44100.wav target/compare/realtime-gui-fix/mulan-44100-stereo-paced.wav target/compare/realtime-gui-fix/mulan-44100-stereo-paced.json
```

单元测试覆盖 output timeout 尾音保留、partial batch 中断重置、初始预缓冲和真实
欠载的区别、捕获 loss sequence、同 callback 多次丢块计数，以及抗混叠和跨块相位。
运行完整工作区检查/测试、DML 相关 package 测试、Tauri 全 features 检查及前端测试。
