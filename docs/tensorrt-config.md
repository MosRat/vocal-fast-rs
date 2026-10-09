# TensorRT 配置、日志和错误边界

`vocal-tensorrt` 的环境变量只在 `TensorRtConfig::from_env()` 这一处解析。构建器和推理 pipeline 接收结构化配置，因此 Tauri 或其他 Rust 宿主可以绕过环境变量直接传入配置。

```rust
let mut config = vocal_tensorrt::TensorRtConfig::default();
config.build.precision = vocal_tensorrt::PrecisionMode::Fp16;
config.build.hubert_precision = vocal_tensorrt::PrecisionMode::Fp16;
config.build.rmvpe_precision = vocal_tensorrt::PrecisionMode::Fp32;
config.runtime.cuda_graph = true;
config.runtime.graph_mode = vocal_tensorrt::GraphReplayMode::NativeSubgraphs;
config.diagnostics.trace = false;
config.validate()?;

let artifacts = vocal_tensorrt::RvcTrtPipelineArtifacts::compile(
    config.build.clone(), package, hubert, rmvpe, output_dir,
    config.profile.clone(),
)?;
let pipeline = vocal_tensorrt::RvcTrtPipeline::load_with_runtime_config(
    &artifacts, synth_config, config.runtime, config.diagnostics,
)?;
```

## 环境变量分组

| 分组 | 变量 |
| --- | --- |
| SDK/profile | `TENSORRT_RTX_ROOT`, `TRT_RTX_ROOT`, `TENSORRT_ROOT`, `TRT_ROOT`, `VOCAL_TRT_PROFILE` |
| 构建 shape | `VOCAL_TRT_MIN/MAX/OPT_FRAMES`, `VOCAL_TRT_MIN/MAX/OPT_SAMPLES` |
| 精度 | `VOCAL_TRT_PRECISION`, `VOCAL_TRT_HUBERT_*_PRECISION`, `VOCAL_TRT_RMVPE_*_PRECISION`, `VOCAL_TRT_SYNTH_*` |
| 图和融合 | `VOCAL_TRT_GRAPH_MODE`, `VOCAL_TRT_CUDA_GRAPH`, `VOCAL_TRT_NATIVE_CUDA_GRAPH`, `VOCAL_TRT_GRAPH_CACHE_SIZE`, `VOCAL_TRT_PARALLEL_ACOUSTIC`, `VOCAL_TRT_FUSED_*`, `VOCAL_TRT_TACTIC_SOURCES` |
| 运行传输 | `VOCAL_TRT_ASYNC_AUDIO_D2H`, `VOCAL_TRT_RMVPE_FFT`, `VOCAL_TRT_TIMING_CACHE` |
| 诊断 | `VOCAL_TRT_TRACE`, `VOCAL_TRT_DEBUG_*`, `VOCAL_TRT_LAYER_PROFILE`, `VOCAL_TRT_LOG_BUILD_OPTIONS`, `VOCAL_TRT_ALLOW_NONFINITE` |
| 检索/CLI 测试 | `VOCAL_INDEX_RATIO`, `VOCAL_TRT_WARMUP`, `VOCAL_TRT_PIPELINE_ITERATIONS`, `VOCAL_TRT_REALTIME_F0`, `VOCAL_TRT_REALTIME_ADVANCE_FRAMES`, `VOCAL_TRT_TEST_FRAMES`, `VOCAL_TRT_DUMP_*` |

布尔值接受 `true/false`、`1/0`、`yes/no` 和 `on/off`。精度接受 `fp32`、`fp16`、`bf16` 及其常用别名；stage mask 和 tactic mask 接受十进制或 `0x` 十六进制。未知值会返回 `ConfigError`，不会静默回退到另一个精度或 profile。

## 日志和错误

## GPU FAISS 检索

RVC 目录中的 `.index` 文件会作为 `aux.retrieval.faiss` 写入 VFRM。当前
运行时接受 Flat-L2 和 IVF-Flat-L2（包括真实的 `IVF511,Flat,nprobe=1,d=768`
索引），加载后常驻显存，不依赖 Python 或 FAISS 动态库。CLI 的
`VOCAL_INDEX_RATIO=0..1` 只用于离线测试脚本；GUI 或 Rust 宿主应直接设置
`ConversionConfig.index_ratio`。TensorRT pipeline 也可以显式调用
`set_retrieval_ratio`，其设备地址在 CUDA Graph 捕获期间保持不变。

GUI / Runtime 的 idle 更新、protect 范围、内部 plan 提交以及失败回滚见
[Idle 引擎参数更新](idle-engine-reload.md)。

库代码使用 `tracing`，不再用 `eprintln!` 输出运行诊断。CLI 在入口初始化 `tracing-subscriber`；可以用标准 `RUST_LOG` 过滤，例如：

```text
RUST_LOG=vocal_tensorrt=trace,vocal_cli=info
```

`NativeBuildError::Stage` 为加载和各个 native graph build 阶段附加上下文，`NativeBuildError::Config` 保留环境变量名、原始值和期望格式。应用层可以直接匹配 `ConfigError`，在创建 CUDA/TensorRT 资源前报告配置错误。
