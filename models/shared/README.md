# Shared inference models

This directory contains the shared models used by an RVC pipeline. They are
separate from a voice checkpoint (`model.pth`) and can be shared by every
voice:

```text
models/shared/
├── hubert_base/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
└── rmvpe/
    ├── rmvpe.pt
    └── rmvpe.onnx       # optional TensorRT/ONNX path
```

The files are downloaded from the `lj1995/VoiceConversionWebUI` Hugging Face
repository. The exact URLs, revisions, byte sizes and SHA-256 values are kept
in [`manifest.json`](manifest.json). The binary files are intentionally
ignored by git; run the standard-library downloader to fetch or verify them:

```powershell
python scripts/download-shared-models.py
python scripts/download-shared-models.py --verify-only
```

The current Rust baseline does not deserialize these PyTorch/ONNX files yet.
They are prepared as real inputs for the HuBERT/RMVPE adapters and subsequent
TensorRT conversion work.
