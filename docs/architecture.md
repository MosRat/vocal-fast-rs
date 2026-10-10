# vocal-fast-rs 基线架构

```text
Tauri UI (apps/tauri)
        │ commands/events
vocal-cli / future app service
        │
vocal-audio ── bounded callback queue ── vocal-engine
        │                                      │
        └──────────── vocal-types ─────────────┘
                         │
                    vocal-model
              RVC .pth/.index → .vfr cache
```

## 分层职责

- `vocal-types`：跨 crate 的音频块、格式、模型描述和延迟指标；不依赖平台。
- `vocal-audio`：实时设备边界。`AudioRing` 是预分配的 bounded queue；`cpal-backend` feature 为后续设备 callback 保留接入点。
- `vocal-model`：读取 RVC 目录和 manifest；对 RVC `.pth` 使用不执行 pickle 的受限 Rust parser，对 RVC Synthesizer ONNX 解析图签名和 initializers，并写入 VFRM v1 typed tensors。未知格式直接报错。
- `vocal-engine`：`InferenceEngine` trait、CPU RVC pipeline、canonical RVC TensorRT graph builder 和 passthrough 基线。pipeline 按参考实现执行 16 kHz 预处理、内容特征、索引融合、F0、protect、合成、裁剪、RMS 和重采样；HuBERT/RMVPE/FAISS/Synthesizer 都是可替换 stage。`vocal-tensorrt` 负责本机 TensorRT-RTX/CUDA ABI、VFRM native graph build、运行时 context、CUDA buffer 和四段 HuBERT→RMVPE→frontend→Synth pipeline；ONNX 只在离线转换或兼容 fallback 中使用。
- `vocal-cli`：doctor、模型检查/打包和确定性 demo，用于建立第一条可验证流程。

## 基线验收

```powershell
rtk cargo fmt --all -- --check
rtk cargo test --workspace --exclude vocal-cpu --exclude vocal-tensorrt
rtk cargo run -p vocal-fast -- doctor
rtk cargo run -p vocal-fast -- demo 8
```

后续热点测试应围绕 callback 到 inference 的排队延迟、模型冷启动、索引检索、TensorRT engine 构建和 enqueue 执行时间添加 criterion/ETW 基准；量化和 CUDA graph 放在基线通过后进行。
