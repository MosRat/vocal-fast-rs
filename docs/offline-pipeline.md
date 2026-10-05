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
768 维 HuBERT 和 RMVPE、`f0=1`，因此适合验证中文男声输入到女声模型的完整
RVC 链路。实际 TensorRT 运行还需要本机的 HuBERT/RMVPE VFRM 共享模型和已编译
pipeline manifest。
