# 模型目录

运行时按 RVC 常见的目录组织读取模型：

```text
models/<model-id>/
├── manifest.json
├── *.pth
└── *.index
```

VFRM v1 是新的紧凑缓存容器：固定 64 字节头、配置 JSON、typed tensor
目录和 64 字节对齐的张量数据区。每个张量保存名称、dtype、shape、偏移
和长度，便于 mmap、快速载入和按照 RVC state_dict 键名匹配。格式说明见
[`docs/vfrm-format.md`](../docs/vfrm-format.md)。

将原始目录导入成兼容缓存：

```powershell
rtk cargo run -p vocal-fast -- model pack models/<model-id> models/<model-id>/manifest.json models/<model-id>.vfr
```

对真实 RVC `.pth` 或 `.onnx` 目录直接做受限解析并打包，可使用：

```powershell
rtk cargo run -p vocal-fast -- model pack-rvc models/<model-id> models/<model-id>/manifest.json models/<model-id>.vfr
```

原始 `.pth` 会作为 `u8` tensor `rvc.pth` 保存，`.index` 会作为
`u8` tensor `retrieval.faiss` 保存。这保留了未转换模型的兼容性。

对 RVC `.pth` 可以直接使用 Rust 的受限解析器。它只允许标准 RVC
checkpoint 的 ZIP/pickle、`weight` state_dict、CPU storage 和已知 tensor
reducer，不执行任意 Python 对象：

```powershell
rtk cargo run -p vocal-fast -- model convert-pth `
  models/voices/default-v2 models/voices/default-v2/manifest.json `
  models/voices/default-v2/default.pth models/voices/default-v2/model.vfr
```

RVC Synthesizer ONNX 也可以由 Rust 直接导入。导入器按输入输出名称、rank、
固定 feature width、dtype、F0、采样率和 opset 检查，只提取 initializers：

```powershell
rtk cargo run -p vocal-fast -- model convert-onnx `
  models/voices/tsukuyomi-v2 models/voices/tsukuyomi-v2/manifest.json `
  models/voices/tsukuyomi-v2/model.onnx models/voices/tsukuyomi-v2/model.vfr
```

将可信 RVC checkpoint 的 `weight` state_dict 转换为真正的 typed tensors：

```powershell
rtk uv run --with torch --with numpy python scripts/convert-rvc-pth-to-vfrm.py `
  models/voices/default-v2 models/voices/default-v2/default.pth `
  models/voices/default-v2-converted.vfr
```

Python 转换器仍可用于交叉核对；Rust `.pth` 导入器会把 RVC `config`、`sr`、`f0`、`version` 和 `info` 放入配置区，
并保留原始 state_dict 名称、dtype 与 shape。VFRM 不包含 TensorRT engine；
engine 应按本机 GPU 和驱动生成。大体积模型文件由 `.gitignore` 忽略。
