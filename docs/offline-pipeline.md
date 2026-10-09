# Shared offline pipeline test

离线测试入口位于 `vocal-engine::offline`，不属于 TensorRT 专用代码。它把
输入 WAV 解码后的单声道 16 kHz 样本分成固定窗口，处理最后一个补零窗口，
再裁掉补零输出，并返回统一的窗口数、耗时和引擎指标报告。

后端通过同一个 `EngineOfflinePipeline` 适配器接入：

| 后端 | 计算实现 | 需要的模型文件 |
| --- | --- | --- |
| `passthrough` | 原样复制 | 任意可读的 VFRM manifest |
| `cpu` | portable CPU RVC baseline（HuBERT/F0/Synth fixture 实现） | 任意 RVC VFRM |
| `tensorrt` | HuBERT → RMVPE → frontend → Synth TensorRT-RTX 图 | VFRM + `pipeline.json` |

未来后端只需实现 `OfflinePipeline::process_window`，窗口、padding、输出裁剪
和报告会自动复用。

## 命令

```powershell
# 只执行并打印统一报告
rtk cargo run -p vocal-fast -- model test-pipeline passthrough \
  target/test-assets/mandarin-rvc/maobailing-rvc.vfr \
  target/test-assets/mandarin-rvc/chinese-input.wav

# CPU baseline，保存 40 kHz（模型采样率）的 WAV
rtk cargo run -p vocal-fast -- model run-pipeline cpu \
  target/test-assets/mandarin-rvc/maobailing-rvc.vfr \
  target/test-assets/mandarin-rvc/chinese-input.wav target/mandarin-cpu.wav

# TensorRT RTX 使用相同的 runner；第五个参数是编译产物 manifest
rtk cargo run -p vocal-fast --features tensorrt-rtx -- model run-pipeline tensorrt \
  target/test-assets/mandarin-rvc/maobailing-rvc.vfr \
  target/test-assets/mandarin-rvc/chinese-input.wav target/mandarin-trt.wav \
  target/test-assets/mandarin-rvc/pipeline.json
```

`VOCAL_OFFLINE_WINDOW_SAMPLES` 可覆盖 portable 后端的窗口大小。TensorRT 后端
会把它限制在 `pipeline.json` 的 profile 范围内。旧的
`run-trt-pipeline` 命令保留作兼容入口。

## 准备真实中文测试资源

权重和音频不提交到 Git。运行：

```powershell
rtk powershell -ExecutionPolicy Bypass -File scripts/prepare-offline-assets.ps1
```

脚本下载并校验：

- `liangstein/Chinese-speech-to-text` 的 `1.wav`，16 kHz 单声道中文测试语音，
  Apache-2.0 仓库；SHA-256 为
  `30a610eb2027bc2cb59b88da6ad7ab0c2e39ca3e342590ebb455d490b4248e25`。
- `maotianchen/maobailing_rvc` 的 `maobailing_rvc.pth`，RVC v2、40 kHz、
  中文女声模型，模型卡标注 Apache-2.0；SHA-256 为
  `c386a153eace55a15cc50b02d102732fa08c1c6a144184be3424c2188a46867c`。

脚本会调用 Rust PTH 解析器写出 `maobailing-rvc.vfr`。模型卡说明该模型使用
768 维 HuBERT 和 RMVPE、`f0=1`，因此适合验证中文语音输入到女声模型的完整
RVC 链路。旧 `chinese-input.wav` 没有明确的男声标签，不作为男声测试依据。
实际 TensorRT 运行还需要本机的 HuBERT/RMVPE VFRM 共享模型和已编译
pipeline manifest。

## 连续完整男声输入（当前推荐）

```powershell
rtk .venv\Scripts\python.exe scripts/download-continuous-mandarin-audio.py
```

脚本从 [Multi-CAST Mandarin v2211](https://github.com/Multi-CAST/mcmandarin/tree/v2211)
下载两份完整故事录音，校验固定 SHA-256，并通过 `cldf/texts.csv` 验证说话人
性别为 `male`。来源由 Maria Vollmer 录制、整理，许可为 CC-BY-4.0；
[大学音频档案](https://fd-repo.uni-bamberg.de/records/bxcvm-m9e27) 保留完整录音。

输出在 `target/test-assets/mandarin-continuous/`：

| 内容 | 原录音范围 | 片段时长 | 直接播放文件 |
| --- | --- | --- | --- |
| 《梁山伯与祝英台》，MD03，22 岁男性 | 6.596–64.058 秒 | 57.462 秒 | `male-01-liangzhu-44100.wav` |
| 《花木兰》，MD01，23 岁男性 | 106.748–177.135 秒 | 70.387 秒 | `male-02-mulan-44100.wav` |

每个片段只截取同一个原文件上的一个连续区间，起止点来自逐句时间标注。
保留自然停顿、呼吸、口语重复及原录音背景声，不拼接句子、不压缩空白、
不调整响度、不变调。原采样率版本为 44.1 kHz 双声道 PCM16，写出后与
完整解码录音对应区间逐样本比较。另提供同名 `16k.wav`，仅做声道平均和
抗混叠重采样，供离线 pipeline 使用；直接试听优先使用 `44100.wav`。

`originals/` 保存约 8 分 13 秒的《梁祝》和约 10 分 25 秒的《花木兰》完整
MP3 及解码 WAV。`metadata/` 保存说话人资料、时间标注和许可；
`manifest.json` 记录下载地址、截取范围、格式、校验和及样本一致性结果。

## 逐句男声语料（旧样本）

```powershell
rtk .venv\Scripts\python.exe scripts/download-mandarin-male-audio.py
```

脚本从 Hugging Face 的 AISHELL-3 镜像按固定 revision 下载 9 段录音，
通过 `spk-info.txt` 验证 SSB0261、SSB0710、SSB0434 均标注 `male`，
并核对下载文件的 LFS SHA-256。资源来自 [AISHELL-3](https://www.openslr.org/93/)。

输出在 `target/test-assets/mandarin-male/`：

- `originals/*.wav`：9 段原始 44.1 kHz 单声道录音。
- `male-01-SSB0261-16k.wav`、`male-02-SSB0710-16k.wav`、
  `male-03-SSB0434-16k.wav`：每位说话人的三句话拼接为一段，以 300 ms
  静音分隔；抗混叠重采样到 16 kHz，峰值音量调整至 -1 dBFS（增益上限 4 倍）。
- 对应 `44100.wav`：原生采样率和原始音量，便于播放或检查录音。
- `manifest.json`、`README.md`：说话人性别和年龄组、逐句文本、来源、
  格式、时长、峰值/RMS 和 SHA-256。

不会对这些录音变调、合成或进行声音转换。原测试 WAV 保留，已有对比结果
继续对应原输入，不会因为下载新样本而变更。

原始语料包含较长的句内停顿。初版拼接还保留了录音首尾静音，实际句间停顿
约 0.8–1.1 秒；SSB0261 的不同句子整段 RMS 也相差约 7.5 dB。
这些是输入 WAV 本身的特征，不能直接据此判断实时 IO 是否 underflow。
旧样本的停顿压缩版本可通过以下命令复现；它仍由独立句子拼接，不作为
连续完整说话的测试输入：

```powershell
rtk .venv\Scripts\python.exe scripts/prepare-mandarin-male-audio.py
```

新文件为 `male-*-compact-16k.wav`，保留原始录音和初版 WAV。预处理仅缩短
RMS 低于 -55 dBFS、长度至少 350 ms 的空白中部，保留 180 ms 停顿；
裁剪录音首尾静音时保留 60 ms 余量，拼接处使用 5 ms 渐变。每句话采用
固定音量增益，以平衡句间响度。短停顿和语音顺序保留。
裁剪位置、增益、时长和 SHA-256 记录在 `compact-manifest.json` 中。
