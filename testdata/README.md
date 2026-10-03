# 测试音频

`fixtures/sine-440hz-16k-mono.wav` 是用标准库生成的 2 秒、16 kHz、单声道、16-bit PCM 正弦测试信号，适合做端到端延迟和格式检查，不含第三方录音版权。

需要重新生成时运行 `rtk python scripts/generate_test_audio.py`。

真实麦克风样本可以放在本目录，但默认测试应继续使用该确定性 fixture。
