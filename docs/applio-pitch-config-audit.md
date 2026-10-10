# Applio 音高与相关配置核对

后续实现已完成：[自动音域匹配](automatic-pitch-matching.md)默认开启，TRT 已支持手动/自动 F0 缩放。
本文保留接入前的配置审计与原始 Applio 数值对照，下面的“尚未接入/拒绝非零 pitch”描述为当时状态。

日期：2026-10-08。用户试听确认：此前同一段男声的 +12 半音版本比原音高版本更有
目标模型特色。本次核对本地 Applio checkout
`c7665ac9a305b3683570ed914f1577d53d4b75c4`；没有修改参考仓库或用户保存的 GUI 配置。

## 1. 默认值必须区分离线与实时

| 配置 | Applio 离线 UI | Applio 实时 UI | 当前 vocal-fast GUI / engine |
| --- | --- | --- | --- |
| F0 音高，半音 | 0，-24..24，step 1 | 0，-24..24，step 1 | 0，-24..24，step 0.1 |
| Proposed Pitch | 默认关闭；目标默认 155 Hz | 默认关闭；目标默认 155 Hz | 尚未接入 |
| 检索比例 | 0.75 | 0，源码注明用于减少延迟 | 0 |
| Protect | 0.5，UI 范围 0..0.5 | 0.33，UI 范围 0..0.5 | 0.33，UI 范围 0..1 |
| RMS / Volume Envelope | 1 | 1 | 1 |
| F0 算法 | RMVPE | FCPE | GPU 路径固定 RMVPE |
| Embedder 选择 | ContentVec | ContentVec | 模型内嵌或指定的 HuBERT/ContentVec 权重 |
| Autotune | 默认关闭 | 默认关闭 | 尚未接入 |

来源：

- `ref_repos/Applio/tabs/inference/inference.py:664,1076,1087,1107,1137,1152`
- `ref_repos/Applio/tabs/realtime/realtime.py:1224,1575,1586,1596,1606,1616,1625`
- `ref_repos/Applio/assets/presets/Default.json`
- `crates/vocal-types/src/config.rs`、`apps/tauri/src/main.ts`

默认 preset 的 index 0.75 / protect 0.5 和 callback 缺省值不等于实际实时 UI
选中的 index 0 / protect 0.33，不能只搜索函数默认参数后就断言 GUI 配置未对齐。
本轮 PyTorch 音频对照显式选择 RMVPE，并不声称使用了参考实时 UI 的 FCPE 默认。

## 2. Pitch 0 保留输入 F0；女性目标范围需要匹配

参考离线 `rvc/infer/pipeline.py:253` 和实时 `rvc/realtime/pipeline.py:244`：

```text
RMVPE / FCPE 提取原始 F0
  → 可选 Autotune，或 Proposed Pitch，或仅手动 Pitch
  → F0 × 2^(有效半音 / 12)
  → 生成 coarse pitch 和 continuous pitchf
  → Synth
```

同一个比例保留了语调变化，同时把激励移入另一音域。Pitch 0 并没有把 F0
变成常数；输入男声的低音域仍然被保留。模型的 speaker embedding / 检索特征
不会自动替用户选择女性音域。结合本次试听，音域与模型不适配是特色不足的主要
配置因素；这不能推断所有模型、输入或实时 DSP 都没有其它问题。

Applio UI 对 Proposed Pitch 的提示：男性目标常用 155 Hz，女性目标常用 255 Hz。
**目标由用户选择**，参考不读取模型文件来确定性别或训练 F0 范围。默认的 155 Hz
不能作为女性模型的自动推荐目标。

其算法使用原始 F0 的非零位置插值填补未发声位置，再取整条轮廓的中位数：

```text
valid = indices(F0 > 0)
median = median(interpolate(F0[valid], all_indices))
offset = clamp(round(12 × log2(target_hz / median)), -12, +12)
effective_pitch = manual_pitch + offset
```

少于两个有效 F0 时，参考的自动 offset 为 0。手动 Pitch 和自动 offset 会相加；
测试自动 255 Hz 时应把手动 Pitch 设为 0，以免在 +12 上再加约 +10。Autotune
是把音高往音阶网格吸附的另一个功能，在参考 `if/elif` 中优先于 Proposed Pitch，
本次说话音频保持关闭。

Applio 还有后处理的 `Pitch Shift Semitones`，经 Pedalboard 改输出波形。当前
`conversion.pitchSemitones` 对应前述 **Synth 前 F0 控制**；两者对 Synth 条件不同，
不能把后处理旋钮误认作已经接入了模型音高控制。

## 3. 自动音高实测

延续 `recognizable-voice-audition.md` 的 20 秒梁祝男声与爱莉希雅模型。

运行原始 Applio 的 Proposed Pitch，target 255 Hz、manual pitch 0、index 0.5、
protect 0.33、RMS 1。只共享已有 native latent / NSF 随机输入，F0 与 offset
由原始参考自己计算；runner 的观察钩子仅记录其输入输出。

- 原始 RMVPE 插值后 F0 中位数：139.428416 Hz。
- 原始参考打印 `calculated pitch offset: 10`；实测有效 Pitch：+10 半音。
- 对应当前 DirectML 手动 +10，输出浊音 F0 中位数约 252.074 Hz。
- 原来的手动 +12 对照为 282.943 Hz；不是自动 255 Hz 的精确同义配置。
- DML +10 与 Applio auto255 的 coarse pitch：2198 帧全部一致。
- continuous F0 最大差：0.0004883 Hz。
- 最终 WAV cosine：0.9998715；RMSE：0.0024157。
- 同输入独立 FP32 Synth cosine：0.9999933。
- 两份 19.98 秒输出均无削波；native noise 与此前 +12 trace 逐元素一致。

文件：

- `target/compare/recognizable-voices/elysia-cn-applio-auto255-i0.5.wav`
- `target/compare/recognizable-voices/listening/elysia-cn-applio-auto255-i0.5.wav`
- `target/compare/recognizable-voices/listening/elysia-cn-directml-p10-i0.5.wav`
- `target/compare/recognizable-voices/elysia-auto255-directml-applio-parity.json`

原音高控制回归另生成 `elysia-cn-applio-p0-observed-i0.wav`：有效 Pitch 0，
相对之前参考输出 cosine 0.999999994、RMSE 1.12e-5，最大差 0.0004501。
重复 CUDA 推理没有达到位相同，不能声称观察钩子回归的 WAV 字节完全一致。

## 4. Embedder 参数也已核对

参考 `rvc/lib/utils.py:130` 的默认 ContentVec 下载路径为
[Applio 官方发布的权重](https://huggingface.co/IAHispano/Applio/blob/70ed563897504c756ec94067c12c902c4fd42025/Resources/embedders/contentvec/pytorch_model.bin)。
本轮固定 Hugging Face revision
`70ed563897504c756ec94067c12c902c4fd42025`，下载至独立临时目录并验证 SHA256：

- 官方 FP32 文件 SHA256：`d8dd400e054ddf4e6be75dab5a2549db748cc99e756a097c496c099f65a4854e`。
- 本地 FP16 文件 SHA256：`cc8c20f4b90a520757260197a3ff2505705a7adbd20ad9eeaa4e1a9b38442ef5`。
- 统一旧 `weight_g/weight_v` 与新 parametrization 字段名后，213 / 213 个参数
  在 FP32 下逐元素相同；没有缺失参数。官方文件实际存的是可由本地 FP16 精确
  表示的值，容器大小与 dtype 不同不能作为权重不同的证据。

这排除了本地 PyTorch `hubert_base` 与该 ContentVec 默认 checkpoint 的参数错配。
该比较不重新证明所有完整 VFRM 的每个 acoustic tensor；爱莉希雅实际 VFRM 路径
已有本轮 stage parity。本轮没有改嵌入权重或重建用户模型文件。

依据：`target/compare/recognizable-voices/applio-contentvec-weights.json`。

## 5. 当前功能缺口与配置建议

1. DirectML 的手动 Pitch 符合参考：`pipeline.rs:107,358` 在 coarse pitch 之前
   缩放 continuous F0；idle 更新支持该标量。此次用户确认有效的 +12 可以保留为
   这些女性模型配合本段男声的显式设置，不修改全局默认。
2. 当前 Runtime 没有 Proposed Pitch / target Hz 配置。以后接入时应把目标作为
   模型使用配置，并区分手动 shift 与自动 offset。实时自动估计需要校准/冻结或
   稳定策略，直接在每个短窗口跳整数半音可能产生新的音高跳变。
3. 当前 TensorRT 的 realtime 与 aligned offline 都拒绝非零 Pitch：
   `tensorrt_backend.rs:706`、`offline/native.rs:334`。这是后端功能缺口，当前
   不能把 GUI 设为 +12 然后切到 TRT。实现应在 native frontend 提供动态缩放，
   确保 continuous/coarse F0 同步，并保持滚动原始 F0 cache 不被反复倍增。
4. 检索 0.5 是此次已试听和实测的选项；离线参考默认 0.75 不是所有模型的质量
   最优值。银狼没有 index，应保持 0，音高调整仍然有效。
5. Protect 0.33、RMS 1 可以保持。实际代码在 `protect >= 0.5` 时绕过 unvoiced
   feature protection，部分参考 UI 文案却描述成“最大保护”；以执行代码为准。
   当前 GUI 允许到 1，0.5..1 都相当于关闭保护，不提供额外音色控制。

## 复现与验证

```powershell
rtk .venv/Scripts/python.exe -X utf8 scripts/run-applio-offline.py --input target/compare/recognizable-voices/input-male-liangzhu-20s.wav --output target/compare/recognizable-voices/elysia-cn-applio-auto255-i0.5.wav --checkpoint $env:LOCALAPPDATA/vocal-fast/model-library/recognizable-chinese-female/source/elysia-cn/elysiav3.pth --index $env:LOCALAPPDATA/vocal-fast/model-library/recognizable-chinese-female/source/elysia-cn/added_IVF345_Flat_nprobe_1_elysiav3_v2.index --index-ratio 0.5 --proposed-pitch-target 255 --protect 0.33 --volume-envelope 1 --report target/compare/recognizable-voices/elysia-cn-applio-auto255-i0.5.json --trace-dir target/compare/recognizable-voices/traces/elysia-cn-applio-auto255-i0.5 --native-noise target/compare/recognizable-voices/traces/elysia-cn-directml-p12-i0.5/0000
rtk .venv/Scripts/python.exe -X utf8 scripts/check-applio-contentvec.py --cache $env:LOCALAPPDATA/Temp/vocal-fast-pitch-audit --report target/compare/recognizable-voices/applio-contentvec-weights.json
```

新参考 runner 参数 `--proposed-pitch-target` 直接调用原始算法；记录原始插值中位数、
实际有效半音和自动 offset。新参数只用于实验，没有声称应用 GUI 已支持它。

验证包含自动 255 Hz 实际推理、DML +10 实际推理、逐阶段/独立 Synth 数值对照、
原音高参考回归、213 个 embedder 参数比较、两个 Python 文件 `py_compile` 和
`git diff --check`。本轮没有改 Rust 或 GUI 行为，没有新增主观试听结论。
