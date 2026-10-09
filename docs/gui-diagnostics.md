# 实时 IO 诊断

诊断页面展示最近 60 秒的有界曲线：音频年龄、每块推理耗时、worker
吞吐速率、分方向队列丢块和输出缺音时长。采用原生 SVG，支持深浅主题、
窄窗口和采样点悬停查看。停止后保留曲线，新会话清空历史；更新超过
3 秒未到达时暂停实时数值，不把旧值当作实时数据，也不跨空白时段连线。

## 指标的单位和边界

| 指标 | 含义 |
| --- | --- |
| Audio age | 最近一个带时间戳的音频块在 render callback 的年龄。Native WASAPI 当前不计尚未播放的输出 padding；CPAL 会加上后端报告的 playback timing。不是硬件 round-trip latency。 |
| Session p50 / p95 / p99 | 当前会话的音频年龄百分位，不是最近一秒的百分位。 |
| Processing time | 本次采样间隔内每块的平均推理耗时，不是累计耗时。 |
| Audio budget | 每个处理块对应的音频时长。Processing / budget 超过 100% 表示推理耗时超过所处理音频的时长；不是 CPU/GPU 利用率。 |
| Worker throughput | worker 消费的原生输入帧/秒及成功发布到输出 ring 的原生帧/秒。不代表实际硬件播放帧数；不同采样率的设备不应直接比较帧数。 |
| Input / Output full | 对应队列已满，无法发布新块，丢弃了该块。按 blocks/s 显示。 |
| Stale skipped | 为保持实时性主动跳过旧输入，单独统计，不能等同于队列满。 |
| Output underflow | 输出请求时数据不足，缺失部分补静音。显示每秒缺音的毫秒数，不是 underflow 事件次数。 |

GUI 神经后端按完整模型批次处理；Processing time / Audio budget 对应模型批次，
例如 250 ms，不再与 10 ms 的设备 slot 混为一个预算。播放初始预缓冲期间写入的
静音不计 underflow；播放启动后缺失的样本仍计入。GUI 正常输入使用 FIFO，
不主动跳过旧输入，stale 指标保留用于其它消费策略。

缺音时长使用物理输出格式换算：

```text
ms missing / s = Δinterleaved_samples × 1000 / channels / sample_rate / interval_seconds
```

例如 48 kHz 双声道输出，1 秒内缺 960 个交错样本，就是 10 ms/s 的缺音。
输入与输出 Queue full 从两个方向分别取增量；底层旧的全局计数为方向的
最大值，不能当作两者之和。

Endpoint details 折叠区显示会话 callback period / jitter、endpoint padding、
stream latency、queue high-water 和 clock drift。不支持的指标显示 `—`。

## 采样与内存

- 音频 worker 仅更新原子计数；图表、序列化和采样均在控制面执行。
- 每个 AudioSession 独占 AudioSampler，至少间隔 500 ms 更新一次，多个
  GUI 窗口读取同一份缓存，不清零共享计数。当前可见窗口通常每秒请求快照。
- GUI DTO 不再传递累计 input/output frames、processing micros、processed
  chunks 或 failures。Rust 内部 worker 计数使用模减计算增量，支持 u64
  回绕；profile 的饱和计数采用饱和减法。
- session ID 作为字符串发送，sample sequence 为可回绕 u32，并结合会话
  elapsed time 去重，避免 JavaScript 的 u64 精度问题。
- 每个窗口最多保留 128 点，并裁剪到最近 60 秒。重复或晚到的快照不产生
  新点、不延长新鲜度；采样序号回绕也不影响排序。

## 验证

Rust 测试覆盖 16 kHz / 48 kHz 帧速率、双声道缺音换算、两个方向的丢块、
块预算、空窗口、共享缓存和计数回绕。前端测试覆盖固定容量、时间窗、
重复/晚到快照、断线和停止/重启。Playwright 验证五张动态图表、深浅主题、
窄窗口、异常数值与折叠状态保留，报告和截图保存在 `target/windows-ui-probe/`。

本机只有 Remote Audio 输出，没有麦克风；本轮没有进行物理 capture-to-render
延迟实测。曲线验证使用模拟数据，不应作为实际设备性能报告。
