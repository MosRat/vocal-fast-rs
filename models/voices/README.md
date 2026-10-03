# RVC voice fixtures

The workspace contains two local, ignored ONNX fixtures downloaded from the
public `wok000/vcclient_model` catalog to test the strict importer:

```text
tsukuyomi-v2/          # RVC v2, 40 kHz, F0, feats interface
amitaro-v2-nof0/       # RVC v2, 16 kHz, no-F0, feats interface
```

The `.onnx` files are runtime assets and are not committed. Their manifests
record the expected sample rate and RVC family. Both have been converted by
the Rust importer to VFRM caches locally. The default `.pth` voice is under
`default-v2/` and exercises the restricted PyTorch ZIP/pickle parser.
