# Acoustic VFRM bundles

HuBERT and RMVPE are now converted offline into `acoustic-v1` VFRM bundles:

```powershell
rtk .venv/Scripts/python.exe scripts/convert-acoustic-onnx-to-vfrm.py `
  --kind hubert --input models/shared/hubert_base/hubert_features.onnx `
  --output models/shared/hubert_base/hubert_features.vfr
rtk .venv/Scripts/python.exe scripts/convert-acoustic-onnx-to-vfrm.py `
  --kind rmvpe --input models/shared/rmvpe/rmvpe_20231006.onnx `
  --output models/shared/rmvpe/rmvpe.vfr
```

The bundle consists of the normal compact VFRM header, a JSON configuration,
an operator IR, and a contiguous tensor directory.  Initializers are named in
the `hubert.*` or `rmvpe.*` namespace. Anonymous exporter values are assigned a
content hash, while linear weights are recovered from their following bias
parameter (for example `hubert.model.encoder.layers.0.attention.q_proj.weight`).
Repeated scalar constants are stored once and all source names become aliases
in the IR. External ONNX data is loaded during conversion and is not required
by runtime loading.

`vocal-model::validate_acoustic` checks the graph signature, namespace, tensor
mapping, and node count. `vocal-tensorrt::AcousticGraphPlan` then classifies
the graph into TensorRT primitives, fused normalization/softmax candidates,
and explicit kernels. RMVPE's STFT and GRU are intentionally classified as
kernel stages because TensorRT-RTX 1.6 does not expose public STFT/GRU network
layers; the native builder emits the Fourier frontend and `ILoop` recurrence
directly from this boundary. No ONNX parser or silent identity fallback is
used at runtime.

Inspect a converted package with:

```powershell
rtk cargo run -p vocal-fast -- model inspect-acoustic models/shared/rmvpe/rmvpe.vfr
```

`RvcTrtPipelineArtifacts::compile` consumes these `.vfr` bundles directly for
the HuBERT and RMVPE plans. ONNX is needed only by the offline converter (or as
an explicit compatibility fallback) and is not opened while loading or
executing a native acoustic plan.
