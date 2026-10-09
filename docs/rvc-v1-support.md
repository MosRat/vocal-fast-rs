# RVC v1：转换、元数据与三种后端

CPU、DirectML 和 TensorRT-RTX 已支持 RVC v1，版本由已验证的 Synth
模型自动选择。实时、离线入口共用版本选择和规范化张量合同，运行时不需要
Python 或中间 ONNX。本次验证日期为 2026-10-09。

## 版本合同

| 项目 | RVC v1 | RVC v2 |
| --- | --- | --- |
| HuBERT Transformer 层数 | 9 | 12 |
| Transformer 内部宽度 | 768 | 768 |
| HuBERT 输出投影 | 训练得到的 `final_proj`，768 → 256 | 无 |
| 检索及 Synth 特征宽度 | 256 | 768 |
| `enc_p.emb_phone.weight` 规范布局 | `[256, hidden_channels]` | `[768, hidden_channels]` |
| F0 模型依赖 | RMVPE；与 v2 共用权重 | RMVPE |

v1 的特征不是截取 v2 的前 256 维。`final_proj` 的规范名称与形状为：

- `hubert.model.final_proj.weight`：`[768, 256]`，连续行优先。
- `hubert.model.final_proj.bias`：`[256]`。

HuBERT 完整包保存 12 层骨干和投影，同一个包可供 v1/v2 使用。建图时
v1 仅展开前 9 层，不计算其余 3 层。缺失投影、投影转置错误、明确声明只支持
v2 的 HuBERT、版本/特征宽度/检索维度矛盾会报错。

`RvcVersion::{V1,V2}` 提供语义化 Rust 接口：`feature_width()`、
`hubert_output_layer()`、`ModelPackage::validate_hubert_for_rvc()`。
独立建图可使用 `vocal_graph::models::hubert_for_rvc()`、
`DirectMlDevice::compile_hubert_for_rvc()`，或设置
`NativeBuildOptions.rvc_version`。三种完整 pipeline 从 Synth 推导版本，
不会要求 GUI 用户另选一个可能与权重冲突的版本。

## VFRM 与来源归一化

`rvc-synth-v1` 是张量 ABI 的版本，支持 RVC 模型 v1 和 v2。
两者共享 Synth 参数名称、卷积布局及线性层 `[in, out]` 布局；差异由
`RvcSynthConfig.feature_width` 和模型版本决定。PTH/ONNX 来源不会进入
后端的建图分支。

新转换的 Synth 元数据包含：

```json
{
  "canonical_schema": "rvc-synth-v1",
  "graph_schema": "rvc-v1-f0-feats",
  "rvc_features": {
    "version": "v1",
    "width": 256,
    "hubert_output_layer": 9,
    "hubert_projection": "hubert.model.final_proj"
  }
}
```

HuBERT 包的 `rvc_feature_modes` 声明各版本的层、宽度及投影。旧文件没有
这些字段时仍兼容，但 v1 的实际投影张量必须存在。嵌套 bundle 检查和
完整载入执行相同的版本选择校验。

PTH：Rust 解析受支持的 RVC checkpoint；没有 `version` 字段时，根据
真实 phone 权重的 256/768 维推导版本，再验证 config 和张量。

ONNX：验证输入输出签名、特征宽度、F0、采样率和标准算子。
voice-changer 的 `embOutputLayer` / `useFinalProj` 若存在，必须分别
符合 v1 的 `9/true` 或 v2 的 `12/false`。该应用的 `version: "2.1"`
不代表 RVC 模型版本。标准 MatMul/Conv 和简化后的 Gemm 参数都归一化
到固定名称；Gemm 的 alpha/beta 必须为 1，transA 必须为 0，transB
仅支持 0/1。其它形态报错。

实测将同一个青年 v1 checkpoint 经 voice-changer 导出并简化为 ONNX：
353 个 Synth 张量的名称、形状和 config 完全一致。最大权重差为
`0.00053453445`：PTH 归一化权重保留原 F16，而 ONNX 导出使用 FP32
折叠 weight norm，所以不承诺不同来源的数值逐位相同。ONNX → VFRM →
CPU 真实执行相对原 FP32 Synth 的最终 cosine 为 `0.9999999759`。

## 后端与缓存

- CPU/DirectML 共用 `vocal-graph` 手写模型，按版本展开 HuBERT，Synth
  特征输入、检索、插值和工作区按 256/768 宽度分配。
- CPU 量化 preset 也按版本选择 9/12 层覆盖；v1 新投影保留 FP32。
- TensorRT 的 Rust/C++ FFI 同步传入特征宽度，HuBERT、frontend、Synth
  的真实 binding 宽度在 buffer 分配之前校验。
- HuBERT/frontend plan 缓存区分 v1/v2；RMVPE 可跨版本复用。
  旧 pipeline manifest 按 v2 解读，不能用来执行 v1，需重建。
- stage cache 也区分 `NVIDIA_TF32_OVERRIDE`，避免 RTX 在另一 TF32
  策略下拒绝创建已缓存 plan 的 context。
- 离线 artifact 缓存只选取与当前声音版本一致的计划。
  完整离线 profile 即使长度恰好为 11200/44960，也关闭实时 tail dispatch。
- GUI 的已有 PTH 导入、完整 bundle、自动模型准备及三个后端选择流程均
  沿用；HuBERT 投影缺失时返回明确错误。

## 准备共享 HuBERT

Python 仅用于 acoustic 转换及参考验证。旧 v2 ONNX 不包含训练投影；
重新转换时附上与该骨干匹配的 HF/fairseq checkpoint：

```powershell
rtk .venv/Scripts/python.exe scripts/convert-acoustic-onnx-to-vfrm.py `
  --kind hubert --input models/shared/hubert_base/hubert_features.onnx `
  --hubert-checkpoint models/shared/hubert_base/pytorch_model.bin `
  --output models/shared/hubert_base/hubert_features.vfr
```

转换器验证共享 feature projection 与 checkpoint 精确匹配、12 层骨干
完整及新投影有限，然后原子发布文件。本地共享资产已重新转换，包含
229 个张量，兼容 v1/v2。声音模型可通过现有 GUI/CLI PTH 导入再打包，
或使用 `bundle-rvc` 示例将 Synth、HuBERT 和 RMVPE 合并。

## 真实模型精度验证

使用本地 `target/chinese-rvc/downloads/qingnian/男声-青年.pth`：
v1、F0 模型、40 kHz、109 个 speaker，测试 sid=0。
源仓库 `chaye741/RVC-Voice-Models`，固定 revision
`acfd9f836ca1fc862a594bc303cfccee8d565e94`；checkpoint SHA-256
`d40f4dcb7667a5c27fc270937eeab30bb768ec60409fa411a48fe89ed6d64a37`。
provenance、权重和报告保存在 `target/`，不纳入 Git。

以下为 11200 个真实语音输入 samples，完整 68 个 Synth frames、27200
个输出 samples；skip=0，不追加 HuBERT 边界帧。参考使用原 FP32
Applio Synth/RMVPE 与标准 RVC v1 的 layer 9 + trained projection。
每次比较共享该次 native 的 latent/NSF 随机输入。

| 后端/精度 | 检索 ratio | 最终 cosine | RMSE | max abs |
| --- | ---: | ---: | ---: | ---: |
| CPU FP32 | 0 | 0.9999999390 | 0.000068832 | 0.00078064 |
| DML FP32 | 0 | 0.9999998940 | 0.000090430 | 0.00081818 |
| TRT FP32 | 0 | 0.9999999066 | 0.000080933 | 0.00090280 |
| DML 默认混合精度 | 0 | 0.9999365561 | 0.00221109 | 0.0465571 |
| TRT HuBERT FP16 + 默认 Synth 混合精度 | 0 | 0.9999078069 | 0.00254945 | 0.0795970 |
| CPU FP32 | 0.5 | 0.9999999336 | 0.000078365 | 0.00105008 |
| DML FP32 | 0.5 | 0.9999998774 | 0.000106382 | 0.00112405 |
| TRT FP32 | 0.5 | 0.9999999254 | 0.000081544 | 0.00069801 |

三后端检索=0.5 的 coarse pitch mismatches 均为 0。v1 投影后的
HuBERT mixed RMSE 约 0.0042–0.0044，不应沿用 v2 的误差结论。
混合精度可用，但上述 max abs 是真实局部峰值，不能只看 cosine。

RTX 3060 上短窗口探针的主机处理延迟约：CPU FP32 165–215 ms；
DML FP32 45–50 ms、mixed 26–30 ms；TRT FP32 45–51 ms、mixed
19–25 ms。DML 探针包含诊断 readback，CPU/TRT 计时循环只读取生产输出。
这些是全 68 帧合成的探针数据，不是含声卡、SOLA片段与排队的实时延迟，
也不是严格同一计时口径下的后端速度排名。

### 完整离线 WAV

三种 CLI 后端均经过 WAV 解码、高通、两侧各 1 秒反射 padding、检索、
推理、裁剪及文件保存。43200 samples 的模型窗口最终得到 40 kHz / 27200
samples 的 WAV，全部有限且非静音。

| 后端/本次离线精度 | 最终 WAV cosine | RMSE | max abs |
| --- | ---: | ---: | ---: |
| CPU FP32 | 0.9999995618 | 0.000218009 | 0.00307596 |
| DML 默认混合精度 | 0.9998898605 | 0.003463774 | 0.03165503 |
| TRT HuBERT/RMVPE FP32 + 默认 Synth 混合精度 | 0.9999790436 | 0.001526466 | 0.01236680 |

参考适配须明确：本地 Applio checkout 的 v1 分支直接使用最后一层。
`run-applio-offline.py` 默认以 adapter 选择标准 RVC v1 的第 9 层，仍
执行原 Applio Synth、F0、DSP 和 FAISS。`--v1-hubert-layer 12` 可复现
该 checkout 原有行为。报告记录实际层和适配项。
Windows 中文 FAISS 路径通过内存反序列化读取，避免原库静默跳过检索。

音频和完整报告在 `target/compare/rvc-v1/`：

- `cpu-offline.wav`、`directml-offline.wav`、`tensorrt-offline.wav`。
- `*-offline-parity.json`、`*-reference.json`、`pth-onnx.json`、`summary.json`。
- 对应 Applio 音频共享各自 native 随机输入，可用于听音；不同 stochastic
  runs 的波形不能直接用于后端之间的数值比较。

v2 CPU 回归使用更新后的共享 HuBERT：最终 cosine `0.9999999704`，
HuBERT RMSE `1.5956e-6`，coarse pitch mismatches=0。

### 实际 Runtime / Engine 接入

另外通过 `offline_audio_probe` 回放实际 Runtime 的回调入口，三种后端
都使用完整 v1 bundle、`realtime_44960`、index ratio=0.5。6.3 秒输入由
同一 0.7 秒测试语音重复组成，验证目的为模型生命周期、实时上下文、
F0 缓存、buffer、SOLA 与输出交接。三个结果都为 16 kHz / 100800
samples，有限且含非零音频。配置、回调报告和 WAV 位于
`target/compare/rvc-v1/runtime/`。

TRT 固定 profile 实际编译了 140 帧 HuBERT、281 帧 frontend/Synth、
282 帧完整 RMVPE，以及独立的 4960 samples 实时 RMVPE。此回放不打开
声卡、不按墙钟节奏输入，也没有进行逐波形参考比较；它验证了真实引擎
接入，不能据此推断设备端到端延迟。

## 复现与检查

全管线探针需要三个 feature，最后参数依次为精度、检索 ratio、计时次数：

```powershell
rtk cargo run -p vocal-fast --all-features --example rvc_version_probe -- `
  cpu target/qingnian-v1-complete.vfrm target/hubert-rvc-v1-v2.vfrm `
  models/shared/rmvpe/rmvpe.vfr target/compare/dml-execution/voiced-11200.f32 `
  target/compare/rvc-v1/cpu-fp32.json fp32 0.5 5
```

后端可选 `cpu/directml/tensorrt`，GPU 可选 `fp32/mixed`；CPU 探针只接受
FP32。`compare-directml-python.py pipeline` 可验证三种输出，使用
`--checkpoint` 和 `--index` 指定配套原始资产。
ONNX 验证导出脚本 `export-rvc-validation-onnx.py` 需要本地
voice-changer checkout、Torch、onnx 与 onnxsim（本次 0.7.3）；
`vocal-model` 的 `check_rvc_formats` 示例比较 config/目录，并写出 VFRM。
这些工具不是原生推理的依赖。

已通过 workspace all-features check、workspace tests、all-features tests、
Rust 格式检查、Python 编译和投影单元测试。完整功能测试共 253 passed、
24 ignored；GPU 真实验证通过上述探针和 CLI 串行执行。
GUI 另外通过前端 `npm run build` 和独立 Tauri workspace 的
`cargo check --all-features`，包括本次版本选择与缓存检查。
