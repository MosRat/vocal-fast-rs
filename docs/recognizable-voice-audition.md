# 中文特色模型离线试听与音高检查

日期：2026-10-08；Windows / RTX 3060。使用用户指定的
`%LOCALAPPDATA%/vocal-fast/model-library/recognizable-chinese-female/ready`。

用户随后确认 +12 半音更有特色。进一步的参考默认值、自动音高 255 Hz / +10
实测及 ContentVec 参数核对见 [音高配置核对](applio-pitch-config-audit.md)。

## 输入与输出

同一段连续中文男声：`target/test-assets/mandarin-continuous/male-01-liangzhu-16k.wav`
中的 5–25 秒，16 kHz 单声道，未拼句、未删内部停顿。源录音和来源记录保持在
`target/test-assets/mandarin-continuous/manifest.json`。

所有结果位于 `target/compare/recognizable-voices/`，不加入 Git。

- `audition.html`：本地试听页面，包含原输入、九份 DirectML 对照和两份 Applio 参考。
- `audition-report.json`：文件位置、模型位置、参数、时长、音量、哈希和实际耗时。
- `listening/`：仅做整段恒定增益的试听副本；目标 RMS 0.1，peak 不超过 0.9。
- 根目录 WAV：未做试听增益的转换输出；`traces/`：真实中间张量。

| 模型 | 原生输出采样率 | 生成的 pitch / index ratio | 包含加载的进程耗时 |
| --- | ---: | --- | --- |
| 爱莉希雅 | 48 kHz | 0 / 0；+12 / 0；+12 / 0.5 | 9.69；5.64；5.61 s |
| 洛天依说话 | 40 kHz | 0 / 0；+12 / 0.5 | 8.00；6.60 s |
| 洛天依歌唱 | 40 kHz | 0 / 0；+12 / 0.5 | 7.67；5.85 s |
| 银狼 | 40 kHz | 0 / 0；+12 / 0 | 8.30；6.02 s |

全部使用 speaker 0、protect 0.33、RMS mix 1。银狼没有 index，检索保持关闭。
九份 DirectML WAV 均为 19.98 秒、有限值、无削波；20 ms 长度差来自 Applio
离线 feature extent/crop 规则。没有实时 SOLA、启动静音或设备队列。不同模型输出
保留原生采样率，不能把不同采样率 WAV 直接逐样本比较。

## 当前配置与可验证的结论

只读检查 `%APPDATA%/com.vocalfast.app/config.json`：当前保存的是 DirectML、
pitch 0、index ratio 0。没有修改用户配置或干预已运行的 GUI 实例。

爱莉希雅 aligned F0 浊音 p5 / p50 / p95：

| 控制 | F0，Hz | voiced fraction |
| --- | --- | ---: |
| pitch 0 | 104.19 / 141.47 / 205.56 | 0.64877 |
| pitch +12 | 208.38 / 282.94 / 411.12 | 0.64877 |

升调在 Synth 前改变 F0，不是重采样输出 WAV。音高分布精确翻倍、时长保持一致。
男声的低 F0 原样保留在女性模型中，以及关闭模型检索，是值得试听核对的因素；
这些数值**不能单独证明**主观的“角色特色”问题已解决。+12 是实验参数，未改为
任何全局或模型默认配置。角色风格还受训练音频、模型质量、输入表达与其它参数影响。

## 原始 Applio 对照

使用该模型原始 `source/elysia-cn/elysiav3.pth`、对应原始 FAISS index、原始
Applio offline Pipeline / HuBERT / RMVPE / Synth，FP32 CUDA；禁用 TF32。
只共享 native latent / NSF 随机张量，参考链路独立提取声学特征与 F0。

| 指标 | pitch 0 / index 0 | pitch +12 / index 0.5 |
| --- | ---: | ---: |
| 输入预处理 waveform 最大差 | 0 | 0 |
| HuBERT cosine | 0.9999914 | 0.9999914 |
| coarse pitch 不同帧数 | 0 / 2198 | 0 / 2198 |
| continuous F0 最大差，Hz | 0.0002136 | 0.0005188 |
| 最终 WAV cosine | 0.9999673 | 0.9998584 |
| 最终 WAV RMSE | 0.0008422 | 0.0026920 |
| 同输入独立 Synth cosine | 0.9999832 | 0.9999933 |

依据：`elysia-p0-directml-applio-parity.json`、
`elysia-directml-applio-parity.json`。开启检索后 features cosine 为 0.9999394；
差异包含 HuBERT 混合精度及检索近邻选择。其它三个模型本轮生成了真实离线音频，
没有逐一完成原始 PTH stage parity，不能由一个模型推断全部模型精确一致。

该结果不支持“爱莉希雅离线图把不同音色算成同一种错误结果”的解释，也不能替代
主观试听或验证实时 GUI 的窗口/DSP。若原始 Applio 的 pitch 0 同样缺乏特色，
应先比较输入 F0 和模型检索；若离线与实时听感不同，继续比较相同控制的实时窗口。

## 本轮代码变更和复现

- 离线 CLI 接入 `VOCAL_OFFLINE_PITCH`，默认 0；在模型物化、GPU 初始化前验证
  ConversionConfig 的合法范围，避免无效配置触发耗时加载。
- `scripts/run-applio-offline.py` 接入 `--pitch`；允许无 index 且 ratio 0 的模型。
- `scripts/audition-voice-models.py` 提供连续剪切、串行推理、报告、统一音量试听和
  本地播放页。resume 校验输入哈希、EXE 哈希、模型位置/大小/修改时间及控制参数。
- 本轮离线试听使用 DirectML；现有 TensorRT frontend 仍拒绝非零 pitch，本轮没有
  改其图接口，也没有把 DirectML 音频标记为 TensorRT 输出。

```powershell
rtk cargo build -p vocal-fast --features directml,tensorrt-rtx
rtk .venv/Scripts/python.exe -X utf8 scripts/audition-voice-models.py --library $env:LOCALAPPDATA/vocal-fast/model-library/recognizable-chinese-female/ready --input target/test-assets/mandarin-continuous/male-01-liangzhu-16k.wav --output-dir target/compare/recognizable-voices --resume
```

Applio：

```powershell
rtk .venv/Scripts/python.exe -X utf8 scripts/run-applio-offline.py --input target/compare/recognizable-voices/input-male-liangzhu-20s.wav --output target/compare/recognizable-voices/elysia-cn-applio-p12-i0.5.wav --checkpoint $env:LOCALAPPDATA/vocal-fast/model-library/recognizable-chinese-female/source/elysia-cn/elysiav3.pth --index $env:LOCALAPPDATA/vocal-fast/model-library/recognizable-chinese-female/source/elysia-cn/added_IVF345_Flat_nprobe_1_elysiav3_v2.index --index-ratio 0.5 --pitch 12 --protect 0.33 --volume-envelope 1 --report target/compare/recognizable-voices/elysia-cn-applio-p12-i0.5.json --trace-dir target/compare/recognizable-voices/traces/elysia-cn-applio-p12-i0.5 --native-noise target/compare/recognizable-voices/traces/elysia-cn-directml-p12-i0.5/0000
```

初次运行保存的输入文件名如上；新版 suite 用 `input-male-20s.wav`，重新运行参考时
按新报告的 input 路径调整。噪声共享必须引用对应参数和输入的最新 native trace。

验证：九份 native 音频、两份真实 Applio 音频、两组逐阶段与独立 Synth 对照；
`cargo fmt --all -- --check`、`cargo test -p vocal-fast --features directml,tensorrt-rtx`
及两个 Python 脚本的 `py_compile`。CLI package 没有单元测试，测试命令成功不能替代
以上实际推理实验。本轮没有可用的主观试听工具，听感由用户试听核对。
