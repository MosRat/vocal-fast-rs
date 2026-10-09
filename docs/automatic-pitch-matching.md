# 自动音域匹配

## 配置与生效边界

共享 `ConversionConfig` 增加以下配置，GUI、CLI 和 Runtime 使用同一份数据：

```json
{
  "conversion": {
    "pitchSemitones": 0,
    "autoPitch": { "enabled": true, "targetHz": 255 },
    "indexRatio": 0,
    "protect": 0.33,
    "rmsMixRate": 1
  }
}
```

- 默认开启，目标 255 Hz，适合此前试听的女性模型；较低音域可选 155 Hz。
- 目标验证范围为有限的 50–1200 Hz。模型权重本身没有提供目标音域统计；目标由配置指定。
- 旧 JSON 缺少 `autoPitch` 时采用默认值；显式 `enabled: false` 保留。
- 手动 `pitchSemitones` 是自动补偿之外的额外偏移。首次尝试自动匹配时建议手动设为 0。
- idle 开关、目标和手动音高更新属于 `HotReload`，保留外层 engine 和已有权重/plan。
- Running、Stopping 及 worker 未归还 engine 时继续拒绝更新。GUI 的所有音高控件同时禁用。
- GUI 控件自动保存，沿用配置 revision、debounce、失败重试及跨窗口同步机制。

CLI aligned offline 使用相同默认值，可通过 `VOCAL_AUTO_PITCH=0` 关闭，
或 `VOCAL_AUTO_PITCH_TARGET=155` 更改目标。`VOCAL_OFFLINE_PITCH` 保持手动额外偏移。
手动单因素试听脚本显式关闭自动匹配，防止历史对照被重复升调。

## 算法

采用本地 Applio `c7665ac9a305b3683570ed914f1577d53d4b75c4` 的 Proposed Pitch 方法：

1. 对正的原始 F0 做线性插值，首尾延用最近的有效 F0。
2. 求插值轮廓中位数。
3. `automatic = clamp(round_ties_even(12 * log2(target / median)), -12, 12)`。
4. `effective = manual + automatic`，在 Synth 前同时改变 continuous F0 和 coarse pitch。

原始未发声位置仍为 0；插值只用于估计音域，不会把无声区变成浊音。
这项功能不将每帧音高吸附到固定音符，语调变化保持原比例。

实时使用校准后冻结策略：有效 F0 帧数至少为 `(context_frames / 2).clamp(2, 50)`
才校准，正常窗口需要约 500 ms 的浊音证据。静音不锁定，锁定后停顿不改变补偿。
会话重启、模型更换、开关/目标更改重新校准；手动偏移不需要重新估计音域。
离线按完整段落重新估计，少于两个有效 F0 时只应用手动偏移。

## 执行与内存

| 后端 | 实现 |
| --- | --- |
| CPU | 复用 scratch，线性选择中位数；Synth 前从原始 F0 生成一次缩放和 coarse pitch |
| DirectML 实时 | GPU graph 内 Cumsum、TopK、GatherElements 插值及中位数，反馈 state 保存锁定/offset；不下载 F0，不构造 N×N 矩阵 |
| DirectML 离线 | 复用该路径本来就会下载的 F0，CPU 估计后只上传一个比例标量 |
| TRT | 独立 CUDA kernel、固定 control/state/scaled-F0 缓冲；锁定后跳过插值/排序，仅复制并缩放；兼容现有 frontend plan |

TRT 的原始 RMVPE 输出和 rolling cache 不被修改。独立 scaled-F0 保证 CUDA graph
capture/replay 不会重复升调；开关和手动比例由地址稳定的设备标量读取。
常用轮廓在 shared memory 中排序，超过 8192 帧的自定义离线轮廓使用预分配的
device scratch 做选择。buffer 在创建 pipeline 时分配，正常执行不增加分配或 D2H 同步。
低层数值诊断 API 的显式默认控件保持原始 F0；生产 engine adapter 安装共享默认策略。

## 2026-10-08 实测

Windows、RTX 3060，GPU 操作串行。生成报告与 WAV 保持在 ignored 的
`target/compare/automatic-pitch/`。

- **完整离线**：爱莉希雅 complete VFRM、中文男声《梁祝》20 秒、target 255 Hz、
  manual 0、index 0.5、protect 0.33。自动补偿为 **+10 半音**；2198 帧 continuous F0、
  2198 帧 coarse pitch 及 1,055,040 个完整 decoder 样本与之前手动 +10 对照逐元素一致。
  原始 Applio auto255 对照见 [配置核对](applio-pitch-config-audit.md)。
- **真实 idle 更新**：TRT、DirectML 使用现有真实 VFRM 和 plan，按
  `on/255/0 → off/255/0 → on/155/0 → on/255/+2 → on/255/0` 往返。
  处理计数保持、持久化设置正确、没有失败；最后一次与第一次输出最大差为 **0**。
  每次 worker 持有 engine 时的更新均被拒绝。
- **GPU 单测**：奇偶轮廓、插值、静音、校准冻结、目标切换、手动叠加、重置，
  CUDA graph replay、原始 F0 不变以及 8193 帧选择路径通过。
- **匹配器 GPU 时间**：DirectML 281/282 帧小图约 **0.15 ms**，含标量/输出操作；
  CUDA 281 帧首次校准约 **0.061 ms**，锁定图回放平均 **0.0063 ms**（100 次）。
  单次测量依赖 GPU 时钟，不能视作硬件音频端到端延迟。
- **真实实时窗口回放**：两后端各处理 20 秒文件，输出长度与输入一致、全部有限。
  回放使用 Runtime engine 和真实窗口/缓存，未打开物理麦克风或扬声器。
- **GUI Playwright**：默认开启、开关/目标自动保存、手动 slider debounce、跨窗口目标同步，
  Running 时禁用全部音高控件通过，页面错误为 0；截图为 `gui-idle.png`、`gui-running.png`。

可复现检查：

```powershell
rtk cargo test -p vocal-types -p vocal-engine -p vocal-runtime --all-features -- --test-threads=1
rtk cargo test -p vocal-directml pitch_matching -- --ignored --nocapture --test-threads=1
rtk cargo test -p vocal-tensorrt pitch_matching -- --ignored --nocapture --test-threads=1
rtk cargo run -p vocal-fast --all-features --example idle_reload_probe -- runtime.json input-mono16k.wav target/pitch-idle
```
