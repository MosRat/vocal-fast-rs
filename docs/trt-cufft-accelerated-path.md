# cuFFT 加速路径实测

当前 44960 固定 profile 的加速链路为：

```text
waveform
  -> CUDA reflect-pad + Hann
  -> cuFFT batched R2C (FP32)
  -> CUDA magnitude [1, 513, T]
  -> TensorRT mel + RMVPE (FP32)
  -> HuBERT feature extractor (FP32) + projection/encoder (FP16)
  -> Synth encoder/flow/source (FP32) + decoder (FP16)
```

cuFFT 只替换 RMVPE 的 STFT；mel 投影、U-Net、GRU、logits/decode 仍然是
TensorRT 原生图。所有计划在执行前创建，运行时没有中间张量回传主机，也没有
每帧重建 FFT plan。TensorRT-RTX 原生 CUDA Graph 负责四个 TensorRT 阶段，
cuFFT 前端与它们在同一 CUDA stream 上顺序执行。

## 复现

```powershell
$env:VOCAL_TRT_PROFILE = "realtime_44960"
$env:VOCAL_TRT_DISPATCH_TUNE = "44960"
$env:VOCAL_TRT_RMVPE_FFT = "cufft"
$env:VOCAL_TRT_HUBERT_PRECISION = "fp16"
$env:VOCAL_TRT_HUBERT_FEATURE_PRECISION = "fp32"
$env:VOCAL_TRT_RMVPE_PRECISION = "fp32"
$env:VOCAL_TRT_RMVPE_ENCODER_PRECISION = "fp32"
$env:VOCAL_TRT_RMVPE_DECODER_PRECISION = "fp32"
$env:VOCAL_TRT_SYNTH_PRECISION = "fp16"
$env:VOCAL_TRT_SYNTH_RESBLOCK_PRECISION = "fp32"
$env:VOCAL_TRT_SYNTH_RESBLOCK_STAGE_MASK = "0"

rtk cargo run -p vocal-fast --features tensorrt-rtx -- model compile-trt-pipeline `
  target/default-canonical.vfr models/shared/hubert_base/hubert_features.vfr `
  models/shared/rmvpe/rmvpe.vfr target/repro-cufft-h16feature32-r32-s16-44960

rtk cargo run -p vocal-fast --features tensorrt-rtx -- model profile-trt-pipeline `
  target/repro-cufft-h16feature32-r32-s16-44960/pipeline.json `
  target/default-canonical.vfr target/sine-44960.f32
```

精度对比需要单独开启 `VOCAL_TRT_DEBUG_INTERMEDIATES=1` 生成中间张量；该模式
会进行 D2H 下载，不能把它的单次延迟当作性能结果。之后使用：

```powershell
rtk uv run python scripts/compare-trt-end-to-end.py `
  --audio target/sine-44960.f32 `
  --debug target/repro-cufft-h16feature32-r32-s16-44960/trt-debug.json `
  --vfrm target/default-canonical.vfr --reference applio `
  --report target/repro-cufft-h16feature32-r32-s16-44960/end-to-end-applio.json
```

## RTX 3060 结果

固定形状为 waveform `44960`、HuBERT `140` 帧、RMVPE `282` 帧、RVC `281`
帧，10 次 warmup、100 次 steady replay：

| 阶段 | CUDA event |
| --- | ---: |
| H2D | `0.074 ms` |
| HuBERT | `4.613 ms` |
| RMVPE + cuFFT | `8.306 ms` |
| cuFFT 外部前端 | `0.014 ms` |
| Synth | `39.700 ms` |
| 阶段 event 总和 | `60.239 ms` |
| host mean | `60.817 ms` |

Native TensorRT CUDA Graph 报告一次 capture 和 109 次 replay。精度以 Applio
FP32 HuBERT/RMVPE/Synth 为参考：

| 指标 | 结果 |
| --- | ---: |
| HuBERT hidden RMSE | `0.002212` |
| RMVPE F0 RMSE | `0.004876 Hz` |
| coarse F0 一致率 | `100%` |
| 最终 waveform max abs | `5.76e-5` |
| 最终 waveform RMSE | `4.57e-6` |
| 最终 waveform cosine | `0.999492` |
| Synth-only cosine | `0.999798` |

同一输入的全 FP32 cuFFT 计划为约 `62.78 ms` event 阶段总和，最终 cosine
约 `0.99956`。因此当前安全加速路径约节省 `2.5 ms`（约 4%），而不会引入
全 HuBERT FP16 的明显误差。全 HuBERT FP16（包括 feature extractor）虽然
把阶段总和降到约 `58.66 ms`，但最终 cosine 只有约 `0.9578`，不作为默认路径。

## 当前策略

- `VOCAL_TRT_RMVPE_FFT=cufft` 是可用的 CUDA 加速开关；不需要 ONNX 或 TensorRT
  插件，插件 ABI 不稳定时也能部署。
- 默认保留 RMVPE FP32 和 HuBERT feature extractor FP32。HuBERT projection/encoder
  的 FP16、Synth decoder 的 FP16 仅在固定 profile 下启用，并应在目标声音模型上
  做一次 parity gate。
- cuFFT 前端不是热点：Nsight 中 gather、R2C、magnitude 每个约 `10 us` 量级。
  后续优化优先级是 RMVPE U-Net convolution、Synth decoder convolution、固定
  profile timing cache，以及 pinned H2D ring buffer。
