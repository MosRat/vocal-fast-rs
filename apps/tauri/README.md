# Tauri GUI

GUI 是控制面。Rust 侧持有模型、engine、配置和生命周期，前端只通过命令修改配置或控制会话。

## 通信边界

- `subscribe` 建立一个 Tauri `Channel<BackendEvent>`，先发送完整快照，之后发送状态、指标和错误事件。
- `get_snapshot` 用于重新连接、窗口恢复或前端状态丢失后的全量同步。
- `update_config` 使用 `expectedRevision` 做乐观并发检查；运行中禁止改变会影响 engine 的配置。
- `load_model` 使用异步 Tauri 命令和 `spawn_blocking`，模型解析不会阻塞 WebView。
- 音频块不通过普通 `invoke` 持续传输；`process_block` 只用于控制面测试，生产音频路径应直接接入 `vocal-audio`。

## 配置和错误

配置写入应用配置目录下的 `config.json`，先写临时文件再替换目标文件。配置带 revision，前端提交旧 revision 时会收到 `config_conflict`。

命令错误统一为带 `kind` 和 `details` 的结构化响应，并同时通过 Channel 发出 `ErrorReport`。前端保留最后一条错误，用户可以 dismiss 后继续操作。

## 本地验证

```powershell
npm install
npm run build
npx tsc --noEmit
cargo test --manifest-path src-tauri/Cargo.toml
cargo check --manifest-path src-tauri/Cargo.toml
```
