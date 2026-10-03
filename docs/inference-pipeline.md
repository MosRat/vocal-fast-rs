# 鎺ㄧ悊 pipeline 瀵圭収

`crates/vocal-engine/src/pipeline.rs` 瀵瑰簲鍙傝€?RVC 鐨?`infer/vc/pipeline.py`锛屾瘡涓彲鏇挎崲闃舵濡備笅锛?
| 鍙傝€冩楠?| Rust 鎺ュ彛 | 褰撳墠瀹炵幇 |
| --- | --- | --- |
| 16 kHz 闊抽銆佸弽灏?padding銆侀珮閫?| `resample_linear`銆乣reflect_pad`銆乣high_pass_filter` | CPU 鍩虹嚎 |
| HuBERT / ContentVec | `FeatureEncoder` | `FrameEnergyEncoder` fixture锛涙帴鍏ユā鍨嬫椂鏇挎崲 |
| FAISS added index | `FeatureIndex` | `InMemoryFeatureIndex`锛汧AISS/TensorRT adapter 棰勭暀 |
| RMVPE / FCPE / PM | `PitchExtractor` | `ZeroCrossingPitchExtractor`锛沗Rmvpe`銆乣Fcpe` 杩斿洖鏄庣‘鐨?adapter 閿欒 |
| `F.interpolate(..., scale_factor=2)` | `FeatureMatrix::interpolate_2x` | CPU 鍩虹嚎 |
| protect 铻嶅悎 | `RvcPipeline::run` | 鎸?voiced F0 瀵圭壒寰佸仛娣峰悎 |
| `net_g.infer` | `Synthesizer` | `PassthroughSynthesizer` fixture锛沗TensorRtRtxSynthesizer<E>` 鎺ユ敹 CUDA executor |
| trim / RMS / resample | `change_rms`銆乣resample_linear` | CPU 鍩虹嚎 |

## TensorRT RTX 鏋勫浘

`crates/vocal-engine/src/tensorrt.rs` 鏍规嵁 VFRM 鐨?`rvc_config` 鍜?canonical tensor 鍚嶇О鏋勫缓
`rvc-tensorrt-graph-v1` 鎻忚堪銆傛瀯鍥鹃『搴忓浐瀹氫负锛?
```text
feats/pitch -> enc_p -> reverse flow + emb_g -> dec -> audio
```

F0 妯″瀷杩樺寘鍚?`nsf_source -> noise_convs -> decoder` 鍒嗘敮锛涢殢鏈洪」浠?`rnd` 杈撳叆鏄惧紡缁戝畾锛屼究浜庡疄鏃剁澶嶇敤 CUDA buffer銆佸浐瀹?seed 鎴栨浛鎹负 TensorRT 闅忔満灞傘€傛墍鏈夊弬鏁板紩鐢ㄥ啓鎴?`@weight:<canonical-name>`锛屾瀯鍥剧粨鏉熸椂浼氭鏌ュ畠浠笌 VFRM 鍙傛暟闆嗗悎瀹屽叏涓€鑷淬€?
Realtime waveform profiles are derived from the complete application window, not from a fixed power-of-two chunk. RvcTrtProfile::reference covers min/opt/max = [11200, 35520, 44960] at 16 kHz; exact presets are available for voice-changer (11200) and RVC WebUI/Applio (44960). HuBERT, RMVPE and Synth frame extents are derived from those samples and passed to the native TensorRT builders.
褰撳墠浠撳簱鎻愪緵 `TrtExecCompiler` 浣滀负鏃х殑 graph/ONNX 鍏煎鍏ュ彛锛涚敓浜?TensorRT-RTX 璺緞鐢?`vocal-tensorrt` 鐩存帴浠?VFRM 鍒涘缓 native TensorRT network銆侶uBERT 鍜?RMVPE 浣跨敤 `acoustic-v1` 鑷湁鏍煎紡涓庢墜鍐?layer锛孲ynth 浣跨敤 `rvc-synth-v1` canonical tensor graph锛岃繍琛屾椂鍙姞杞芥満鍣ㄧ浉鍏崇殑 serialized plan銆侽NNX 鍙敤浜庣绾胯浆鎹㈡垨鏄惧紡鍏煎 fallback锛孷FRM 鏈韩浠嶄笉鎼哄甫鏈哄櫒鐩稿叧 plan銆?
```powershell
rtk cargo run -p vocal-fast -- model convert-onnx \
  models/voices/tsukuyomi-v2 models/voices/tsukuyomi-v2/manifest.json \
  models/voices/tsukuyomi-v2/model.onnx tsukuyomi.vfr
rtk cargo run -p vocal-fast -- model build-graph tsukuyomi.vfr tsukuyomi.graph.json
rtk cargo run -p vocal-fast -- model compile-trt tsukuyomi.vfr tsukuyomi.plan models/voices/tsukuyomi-v2/model.onnx
```

## 褰撳墠璋冪敤

```rust
use vocal_engine::{pipeline, BackendKind, EngineConfig, create_engine};

let mut engine = create_engine(EngineConfig {
    backend: BackendKind::RvcCpuPipeline,
    model: descriptor,
})?;
let output = engine.process(&input_chunk, conversion_config)?;
```

`RvcCpuPipeline` 鏄彲杩愯鍩虹嚎锛沗TensorRtRtxSynthesizer<E>` 宸茬粡鎻愪緵 pipeline 鍒?CUDA executor 鐨勮竟鐣岋紝鐪熷疄 RTX adapter 鍙互鐩存帴澶嶇敤 `TensorRtGraph` 鐨?binding 琛ㄣ€?
瀹屾暣鐨?native CUDA pipeline 浣嶄簬 `vocal-tensorrt::RvcTrtPipeline`锛氬畠鍦ㄥ悓涓€
CUDA stream 涓婂鐢ㄨ澶囩紦鍐插尯锛屼緷娆℃墽琛?HuBERT銆丷MVPE銆乫rontend 鍜?Synth锛?鍙湪杈撳叆涓婁紶涓庢渶缁堥煶棰戜笅杞芥椂璺ㄨ秺涓绘満杈圭晫銆俙NativeBuildOptions::precision`
鏀寔 acoustic FP32銆丗P16 鍜?BF16锛汼ynth 鍦?TensorRT-RTX 1.6 淇濇寔 FP32銆?
`PipelineOutput::timings` 浠ュ強 `PipelineEngine::last_timings()` 浼氫繚鐣欑壒寰併€佺储寮曘€侀煶楂樸€佸悎鎴愬拰鍚庡鐞嗚€楁椂锛屽悗缁彲浠ョ洿鎺ユ帴鍏?criterion 鎴?ETW 鐑偣娴嬭瘯銆?
RMVPE plans are emitted as two fixed-shape engines: `rmvpe-realtime.plan`
uses 4960 samples for streaming F0, while `rmvpe-full.plan` uses the selected
full window (11200 for voice-changer or 44960 for WebUI/Applio). The waveform
upload uses a three-slot pinned host ring and `cudaMemcpyAsync` on the pipeline
stream; the ring is reused only after the stream synchronization at the end of
each callback.
