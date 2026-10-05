# 测试音频

`fixtures/sine-440hz-16k-mono.wav` 是用标准库生成的 2 秒、16 kHz、单声道、16-bit PCM 正弦测试信号，适合做端到端延迟和格式检查，不含第三方录音版权。

需要重新生成时运行 `rtk python scripts/generate_test_audio.py`。

真实麦克风样本可以放在本目录，但默认测试应继续使用该确定性 fixture。

中文 RVC 离线测试资源使用 `scripts/prepare-offline-assets.ps1` 下载到
`target/test-assets/mandarin-rvc/`，不会进入版本库。三种后端的统一命令和
资源来源见 [`docs/offline-pipeline.md`](../docs/offline-pipeline.md)。
