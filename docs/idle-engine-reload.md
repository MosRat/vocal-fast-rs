# Idle 引擎参数更新与故障回滚

## 生效边界

`RuntimeContext` 的控制 actor 独占 idle engine；启动后所有权交给音频 worker。
Running、Stopping 或 worker 尚未归还 engine 时，一律拒绝配置更新。
配置在 idle 保存后，由下一次 session 使用；不支持运行中的参数替换。

| 更改 | idle 行为 |
| --- | --- |
| protect、RMS、index ratio、pitch、自动音域开关/目标 | 校验现有 engine 的能力后保存，保留 engine |
| 输入/输出格式、设备、队列 | 保留 engine；下一次启动重新创建 IO 转换器和队列 |
| processing backend | 先构造候选 engine，保存成功后替换 |
| DirectML sources、adapter、speaker | 先构造候选 DirectML engine，保存成功后替换 |
| TRT graph、window、speaker、已发布 manifest | 准备候选内部 pipeline，保存成功后提交，保留外层 engine |
| TRT precision、FFT、source、profile | 标记 `PlanBuildRequired`，阻止用旧 plan 启动 |
| model path | 标记 `LoadModelRequired`，必须完成模型载入 |

TRT 和 DirectML 均支持手动音高与默认开启的[自动音域匹配](automatic-pitch-matching.md)，
不为这些标量重建 plan。没有检索 index 时，会在保存阶段拒绝非零 index ratio，避免再次启动才损坏
engine 状态。Passthrough 只透传声音，转换参数不改变它的音频输出。

## protect

所有后端接受 `[0, 1]`，`protect >= 0.5` 关闭保护；有 F0 和检索时，
小于 0.5 的值对非浊音特征混合原始 HuBERT 特征。index ratio 为 0 时不应用保护。

实时 TRT adapter 将 `ConversionConfig.protect` 传给 native pipeline。
保护量保存在地址稳定的设备 scalar 中，已捕获 CUDA 图读取该地址。
重复配置同值不进行设备上传，也不为这项参数重建 plan 或 engine。
正常 session 重启仍按 streaming reset 策略清除历史和图状态。

index ratio 的多次 idle 编辑合并为下一次推理的一次上传。每次执行预留
4 个 pinned control staging 槽位，覆盖 ratio、p_len、return_length、speaker，
避免传输完成前覆盖之前的主机数据。

## 配置事务与状态

```text
revision / idle / 通用校验
  → 根据 engine 实际生效配置判断更新种类
  → 后端能力校验 / 准备候选 pipeline 或 engine
  → 原子保存配置
  → 不可失败的内存提交
  → 更新 revision 和 snapshot
```

`PreparedEngineReload` 借用原 engine；准备过程中原 pipeline 不变。文件写入
失败时丢弃候选资源，原 engine、配置、revision 和生命周期保持不变。
构造候选 pipeline 失败也不会先把新配置写入文件。

Runtime 单独记录 engine 实际生效配置，待构建的 plan 配置不会变成新比较基准。
普通控制参数保存不会清除 `PlanBuildRequired`；撤销构建参数可以恢复原 engine。
同时修改构建参数和执行参数时，先要求完成 plan 构建。

成功构建后调用 `apply_prepared_tensorrt_pipeline(config, revision)`。
这个接口也支持 manifest 路径不变的重新发布，并在提交前实际载入、检查候选
pipeline。Tauri 的 Build 流程使用该接口；模型不同仍需完成 Model Load。

每次 session 启动清理窗口、F0、SOLA、待输出数据和格式相关历史。CPU 和 TRT
直接处理不同格式的音频块时，也会清除旧格式历史。错误恢复在 device reset
成功后才把内部 engine 恢复为 Ready，并保留失败/处理累计计数。
重复 Start 和运行时 Clear Error 无法重置正在处理的 engine。

GUI 使用 `config_update_failed` 报告候选 GPU 更新失败，保留原 Runtime 状态。
前端显示具体原因和原配置仍有效的提示；revision 冲突继续使用原冲突类型。

## 验证

Windows / RTX 3060，真实 VFRM + FAISS index，中文男声前 6 秒，2026-10-08。
软件文件输入使用真实 TRT / DirectML 推理、Runtime 控制 actor、worker
所有权移交及独立配置文件；没有使用物理麦克风或扬声器。
CPU 与 Passthrough 验证的是各自实现的控制和音频边界。

- 四个后端完成 `0.33 → 0 → 0.5 → 1 → 0.33` 往返；保存不重置处理计数。
- TRT、DML 的最后一次 0.33 输出与第一次逐样本完全一致；0.5 与 1 也完全一致。
- TRT 的 protect=0 相对 0.33 最大样本差为 0.0500554；DML 为 0.314499，确认参数确实生效。
- index 开关、RMS、支持的 pitch，以及 44.1 kHz 双声道 / 16 kHz 单声道切换均成功。
- TRT 和 DML 各处理 264 个 batch，engine failures 均为 0。
- 缺失 manifest、非法 pitch 在提交前失败，旧 engine 不变。
- 将配置目标临时替换为目录，验证实际文件保存失败；TRT 候选内部 pipeline
  被丢弃，原配置和 revision 保持不变，之后同 revision 重试成功。
- CUDA GPU 测试在同一张 capture graph 上切换 protect，验证非浊音数值、浊音
  不变、0.5/1 关闭保护，以及 NaN 拒绝。

回归工具：`crates/vocal-cli/examples/idle_reload_probe.rs`。
生成配置、WAV 和报告在 `target/compare/idle-reload/{trt,dml,cpu,passthrough}/`，保持不跟踪。

```powershell
rtk cargo run -p vocal-fast --all-features --example idle_reload_probe -- runtime.json input-mono16k.wav target/idle-reload
rtk cargo test -p vocal-tensorrt --all-features retrieval::protection -- --include-ignored
rtk cargo test -p vocal-engine -p vocal-runtime --all-features
```
