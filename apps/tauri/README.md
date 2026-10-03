# Tauri shell

这里保留前端和 `src-tauri` 的边界。核心 workspace 不依赖 WebView 或 GUI，方便在无桌面环境完成音频、模型和推理测试。

下一步接入顺序：

1. 在 `src-tauri` 加入 Tauri commands，把 `vocal-engine` 的会话生命周期暴露给前端。
2. 由 Rust 侧持有音频 callback 和模型缓存，前端只负责设备、模型和参数状态。
3. 使用事件流发送 `LatencyMetrics`，避免每个音频 block 跨 IPC 往返。
