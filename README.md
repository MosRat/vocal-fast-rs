# vocal-fast-rs

Rust workspace for a real-time voice converter. The baseline includes bounded audio blocks, RVC-compatible model metadata/cache loading, and a stage-based CPU RVC pipeline. TensorRT RTX and the Tauri shell are isolated so they can be enabled on an RTX development machine without coupling the core crates to a local SDK.

## 快速开始

```powershell
rtk cargo fmt --all
rtk cargo test --workspace
rtk cargo run -p vocal-fast -- doctor
rtk cargo run -p vocal-fast -- demo 8
rtk cargo run -p vocal-fast -- model inspect models/fixture-rvc-v2/manifest.json
rtk cargo run -p vocal-fast -- model pack models/fixture-rvc-v2 models/fixture-rvc-v2/manifest.json models/fixture-rvc-v2.vfr
```

`vocal-engine::tensorrt` 会从 canonical VFRM 权重生成固定的 `rvc-tensorrt-graph-v1` 描述，并校验每个权重 binding。RTX 机器上可设置 `VOCAL_TRTEXEC` 使用 `model compile-trt` 调用 `trtexec`；`vocal-tensorrt` 保留 SDK/plugin 边界，直接从 VFRM 参数创建 TensorRT network 需要 TensorRT-RTX SDK 文件可用后接入。

模型目录、VFRM 缓存和测试音频的说明分别在 [`models/README.md`](models/README.md)、[`docs/architecture.md`](docs/architecture.md) 和 [`testdata/README.md`](testdata/README.md)。
