# Tauri GUI

GUI 是控制面。Rust 侧持有模型、engine、音频设备流、配置和生命周期，前端只通过命令修改配置或控制会话。

## 通信边界

- `subscribe` 建立一个 Tauri `Channel<BackendEvent>`，先发送完整快照，之后发送状态、指标和错误事件。
- `get_snapshot` 用于重新连接、窗口恢复或前端状态丢失后的全量同步。
- `update_config` 使用 `expectedRevision` 做乐观并发检查；运行中禁止改变会影响 engine 的配置。
- `load_model` 使用异步 Tauri 命令和 `spawn_blocking`，模型解析不会阻塞 WebView。
- `list_audio_devices` 使用 CPAL 的推荐 host（Windows 上为 WASAPI）返回稳定 `DeviceId`、方向、默认设备和默认格式。
- `list_processing_backends` 返回共享 Runtime 的设备探测结果、禁用原因和推荐实现；模型卡片与 Settings 复用后端选择组件。
- DirectML 提供独立 GPU 选择，展示显存、推荐设备及虚拟/远程适配器类型；软件和不兼容设备禁用。选择按设备标识自动保存，已有模型会在 idle 状态应用配置；音频运行时禁止修改。
- `start_session` 创建 duplex capture/render stream 和 Rust worker；worker 通过 `WorkerIo::process_once` 驱动 engine，音频块不经过 WebView。
- `process_block` 只用于没有实时流时的控制面测试；实时运行时会拒绝它，避免两个 engine 同时消费模型。
- 前端每秒用 `get_snapshot` 补齐设备回调和 worker 指标，状态变化仍通过 Channel 推送。
- 页面通过 `view.ts` 的 DOM 差量更新保留控件和动画节点，相同内容不会重绘；保留节点的事件绑定会替换，避免重复处理按钮操作。
- Session configuration 当前只开放 Reference、Voice changer 11200、Applio 44960 三个 profile。其余 session 参数在 GUI 中固定，设置页复用同一组选项，启动 worker 后禁用配置调整。
- 音频路由、转换参数、模型目录及后端设置在 idle 状态自动保存；连续修改经过 350 ms 防抖，写入串行执行，失败保留修改并显示 Retry。只有三个 Session Profile 需要 Save profile。自动保存不提交未保存的 profile，Discard profile 也只撤销 preset 选择。
- 自动保存状态统一显示在侧栏左下角，跨页面复用：保存期间转动图标，成功后短暂显示勾选并恢复低亮度状态，失败保留 Retry/Reload；窄窗口保留状态图标和操作按钮，不占用页面内容区。
- 窗口之间的独立字段修改通过三方合并同步；重叠修改显示 Reload 提示。配置写入确认后保留请求期间的新修改，旧命令响应不会覆盖更高 revision 的快照。
- Processing backend 在模型选择下方独立开放。新配置默认 Automatic，已有手动选择保留；Load 可原子应用后端和 pipeline 路径，保留其它草稿。TensorRT 尚无 plans 时显示待准备状态并禁止启动；支持已有 pipeline 或 Build plans & load model。参见 [后端选择与验证](../../docs/processing-backend-selection.md)。
- 设置和诊断通过左侧导航进入，标题栏仅保留 Inspect 与窗口控制按钮，避免重复入口；托盘仍提供辅助窗口入口。
- 模型选择独立于配置草稿：选中目录条目后可直接 Load，Load / Unload 水平排列；模型操作保留其它未保存设置，外部配置变更仍触发 revision 冲突。Windows 路径显示隐藏 `\\?\` 前缀，文件操作保留原始路径。
- 启动阶段使用独立的 440 × 248 `splash` 窗口：单层不透明表面、品牌、状态文字及底部细进度条。只恢复配置、准备文件目录、检查模型元数据并探测音频设备与后端；不加载权重、读取 plan 缓存或构建引擎。随后显示主窗口，首次状态、设备列表和 Channel 连接就绪并完成绘制后，主页面自动恢复上次模型，在页面顶部呈现缓存、构建与引擎初始化进度。模型准备失败不改变应用启动就绪状态，也不会重复自动重试。
- 音频设备与后端探测在独立命名线程上并行执行，阻塞启动任务由异步 supervisor 等待并捕获异常。前端通过启动终态事件和窗口可见事件衔接首帧，不轮询启动状态；revision 防止迟到回复覆盖 Ready，关闭阶段禁止启动任务重新显示窗口。
- 主窗口关闭会隐藏到系统托盘，Rust 侧统一停止音频会话；托盘提供显示、启动/停止、设置、诊断和退出入口，退出前会走同一个 `AppState::shutdown` 路径。

## 配置和错误

配置写入应用配置目录下的 `config.json`，先写临时文件再替换目标文件。配置包括设备 ID、采样率、block/queue 和转换参数，带 revision；前端提交旧 revision 时会收到 `config_conflict`。模型可以单独卸载，卸载会清空 engine 和持久化模型路径。

命令错误统一为带 `kind` 和 `details` 的结构化响应，并同时通过 Channel 发出 `ErrorReport`。前端保留最后一条错误，用户可以 dismiss、展开技术详情或复制报告后继续操作；内部错误会自动打开错误对话框。启动状态与模型准备状态分开维护。

音频启动错误保留 HRESULT、设备方向、名称及恢复建议。设备占用、缺失、权限和服务错误直接报告；周期/格式冲突先按当前或默认共享周期有限重试，适合 fallback 的其它初始化错误才尝试 CPAL。打开设备失败保留已加载引擎 Ready，可更换设备或直接重试。线程边界、启动顺序和实测记录见 [Windows startup and audio recovery](../../docs/windows-startup-and-audio-errors.md)。

## 本地验证

```powershell
npm install
npm run build
npm test # Node 22.6+；配置草稿、保存确认与跨窗口合并回归
npx tsc --noEmit
cargo test --manifest-path src-tauri/Cargo.toml
cargo check --manifest-path src-tauri/Cargo.toml
```

## Windows 安装包与 Debug 实例

在 Windows x64、PowerShell 7、Visual Studio C++ Build Tools 和 Windows SDK 环境下，从本目录执行：

```powershell
# 默认构建含 TensorRT-RTX + DirectML 的 Release NSIS 安装包。
npm run build:windows

# 构建内嵌前端的 Debug 程序并启动，无需保持 Vite 运行。
npm run debug:windows

# DirectML 版本，不要求 CUDA/TensorRT SDK。
npm run build:windows -- -Backend DirectMl
npm run debug:windows -- -Backend DirectMl

# CPU/Passthrough 版本，不要求 GPU SDK。
npm run build:windows -- -Backend Cpu

# 为 Debug WebView2 开启仅监听本机的 CDP，供 Playwright/Inspect 使用。
npm run debug:windows -- -InspectPort 9223
```

脚本 `scripts/build-windows.ps1` 自动发现 Visual Studio，并从
`TENSORRT_RTX_ROOT`/`TRT_RTX_ROOT`/`TENSORRT_ROOT`/`TRT_ROOT` 和 `CUDA_PATH`
读取 SDK 路径；也可以使用 `-TensorRtRoot`、`-CudaRoot` 显式指定。
当前 native bridge 对应 TensorRT-RTX 1.6 和 CUDA 13.x。
默认同时启用 `tensorrt-rtx,directml,cpu-graph`；`-Backend DirectMl` 启用 `directml,cpu-graph`，
`-Backend Cpu` 仅启用 `cpu-graph`。普通 Cargo 和 `tauri dev` 默认启用 DirectML 和真实 CPU 推理；
首次运行前执行 `scripts/setup-directml.ps1` 安装固定版本的 SDK。
Windows helper 会自动安装缺失的默认 SDK，也支持 `-DirectMlRoot`/`DIRECTML_ROOT`。

`src-tauri/tauri.windows.conf.json` 启用当前用户安装、中英文安装界面、应用图标，
并保持动态 MSVC CRT，与 CUDA 的 `/MD` 一致。安装包附带 VC runtime、
TensorRT-RTX/ONNX parser、CUDA runtime、cuFFT 和 DirectML；模型权重及编译后的 plans 由用户单独管理。
WebView2 已安装时复用系统版本，缺失时由内嵌 bootstrapper 联网安装。
TensorRT 功能仍需要兼容的 NVIDIA 显卡及驱动。

| Windows 构建 | Cargo features | 分发清单的 GPU 运行库 |
| --- | --- | --- |
| TensorRt（默认） | `tensorrt-rtx,directml,cpu-graph` | TensorRT/ONNX parser、CUDA/cuFFT、`DirectML.dll`、`dnnl.dll` |
| DirectMl | `directml,cpu-graph` | `DirectML.dll`、`dnnl.dll` |
| Cpu | `cpu-graph` | `dnnl.dll`（无需 GPU SDK） |

Debug 的前两种构建额外部署 `DirectML.Debug.dll`；Release 不包含它。
DirectML 的 license/notice 放在 `licenses/directml/`，D3D12、DXGI 和 GPU 驱动由系统提供。
DLL 使用显式资源清单，不会把其它构建遗留的 DLL 混入安装包。
构建报告记录实际 features、资源目的地和运行库 SHA-256。
打包矩阵和实际验证记录见 [Windows runtime packaging](../../docs/windows-runtime-packaging.md)。

产物位于 `src-tauri/target/release/bundle/nsis/*-setup.exe`，Debug 程序位于
`src-tauri/target/debug/vocal-fast-tauri.exe`。构建记录、SHA-256 清单、启动 PID 和
标准输出/错误日志写到仓库 `target/windows-build/<Profile>-<Backend>/`，不加入 Git。
当前包未配置代码签名证书。

首次使用推荐通过上述脚本打包；直接 `tauri build --features tensorrt-rtx` 不会自动
生成 SDK DLL 的资源映射。DLL 暂存目录和临时资源配置都位于被忽略的 `target/` 目录。

## 完整模型与自动计划构建

模型列表展示 Synth、HuBERT、RMVPE 和可选检索 index；展开 Model contents 可以
查看每个组件的大小和来源。选择包含三部分权重的完整 VFRM 包并使用 TensorRT RTX
后，点击 Load 会自动提取声学权重、构建或复用 plans，再加载引擎。
Studio 和 Settings 的只读路径会同步指向实际缓存文件，无需填写子模型路径或 JSON。
模型加载使用已保存的 session profile，其他未保存配置仍保留在草稿中。

完整包的生成命令、格式和本机验证记录见
[complete-model-bundles.md](../../docs/complete-model-bundles.md)。
