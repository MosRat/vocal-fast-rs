# VFRM v1

VFRM v1 is the runtime cache format for this workspace. It is a compact
configuration plus tensor container, modeled after the useful parts of GGUF.
It does not contain a TensorRT engine.

## Layout

All integers are little endian.

```text
header (64 bytes)
config JSON (manifest + converter metadata)
tensor directory (binary entries)
zero padding to 64-byte alignment
tensor data (each tensor 64-byte aligned)
```

The header is `VFRM\0\0\0\1`, flags, header size, config length, tensor count,
directory length, data offset, and two reserved fields. A directory entry is
`name_len:u32, dtype:u8, rank:u8, reserved:u16, name bytes, shape:u64[rank],
offset:u64, length:u64`.

Supported dtypes are `u8`, `i32`, `i64`, `f16`, `f32`, and `bf16`. The loader
checks that `product(shape) * sizeof(dtype) == length`, rejects unknown dtypes
and out-of-range offsets. RVC model packages return tensors by their
canonical RVC Synthesizer names, so graph construction does not depend on
the source state_dict or ONNX initializer names.

## Conversion

Use the isolated `uv` environment to inspect a trusted RVC checkpoint and
convert its `weight` state_dict:

```powershell
rtk uv run --with torch --with numpy python scripts/convert-rvc-pth-to-vfrm.py `
  models/voices/default-v2 models/voices/default-v2/default.pth `
  models/voices/default-v2-converted.vfr
```

The native Rust importer now handles the RVC-only subset of modern PyTorch
checkpoints directly. It reads the ZIP `data.pkl` with a non-executing,
bounded pickle VM, accepts only CPU/standard tensor storage and the known RVC
reducers, and rejects arbitrary globals, non-contiguous views, unsupported
dtypes, malformed storage, and oversized objects:

```powershell
rtk cargo run -p vocal-fast -- model convert-pth `
  models/voices/default-v2 models/voices/default-v2/manifest.json `
  models/voices/default-v2/default.pth models/voices/default-v2/model.vfr
```

ONNX RVC Synthesizer exports are also parsed natively. The importer checks
the opset (13..18), recognized RVC input interfaces (`feats/p_len/sid` or
`phone/phone_lengths/ds`, with matching pitch inputs), fixed feature widths,
batch/rank/dtype, audio output shape, F0 metadata and sample rate. It extracts
only ONNX initializers into typed tensors and records a fixed `graph_schema`;
unsupported graphs, external initializers, dynamic feature widths and unknown
input/output contracts fail immediately:

```powershell
rtk cargo run -p vocal-fast -- model convert-onnx `
  models/voices/tsukuyomi-v2 models/voices/tsukuyomi-v2/manifest.json `
  models/voices/tsukuyomi-v2/model.onnx models/voices/tsukuyomi-v2/model.vfr
```

The older Python converter remains useful as a cross-check, but is no longer
required for the RVC `.pth` or RVC ONNX path.
 
## Canonical RVC tensor ABI

RVC packages use the fixed "rvc-synth-v1" namespace. Tensor names are stable
dot-separated graph parameter names under enc_p, flow, dec, and emb_g. The
directory is sorted by canonical name and every tensor is contiguous,
row-major, and little-endian.

The phone projection is always stored as [feature_width, 192]. This is the
ONNX MatMul layout; the PyTorch Linear checkpoint layout [192, feature_width]
is transposed during conversion. PyTorch weight normalization pairs are
folded from weight_g and weight_v into one weight tensor. ONNX folded Conv and
ConvTranspose constants are assigned from their consuming node path.

Metadata includes canonical_schema, canonical_layout, graph_schema,
canonical_tensor_count, and tensor_map. tensor_map records source initializer
names and transforms for inspection, while runtime graph construction uses
only canonical names and shapes.
