# RMVPE mixed precision

The native RMVPE graph now has explicit precision regions. The parity-safe
configuration is:

| region | type |
| --- | --- |
| STFT and mel frontend | FP32 |
| U-Net encoder | FP32 |
| U-Net decoder | FP32 |
| bidirectional GRU and recurrent state | FP32 |
| salience/logits and weighted decode | FP32 |

The GRU boundary is an explicit cast, so TensorRT cannot accidentally mix a
half activation with a float recurrent state. The public `pitchf` binding stays
FP32. The default keeps the U-Net in FP32. Setting
`VOCAL_TRT_RMVPE_PRECISION=fp16` selects FP16 for both U-Net regions; the more
specific `VOCAL_TRT_RMVPE_ENCODER_PRECISION` and
`VOCAL_TRT_RMVPE_DECODER_PRECISION` variables still override them individually.
The sensitive STFT, GRU, and logits regions default to FP32.

The FP16 U-Net is a throughput experiment. Enable it only after comparing the
native F0 track on real speech; the Python sensitivity matrix alone does not
cover TensorRT tactic and boundary differences.

## Region split and native gate

The Applio reference was also measured with the U-Net split into its trained
encoder, intermediate, and decoder modules. On the 44960-sample fixture,
lowering the encoder produced F0 RMSE `1.14e-2 Hz` and two coarse-bin
mismatches. Lowering only the intermediate module produced `2.19e-4 Hz` with
all coarse bins equal; lowering only the decoder produced `1.22e-3 Hz` with
all coarse bins equal. These results identify the encoder as the sensitive
U-Net region, but they are only a Python reference bound.

A native 44960 TensorRT plan was built with HuBERT extractor FP32,
HuBERT projection/encoder FP16, RMVPE encoder FP32, RMVPE decoder FP16, and
FP32 GRU/logits/decode. The decoder boundary now has an explicit cast before
zero-insertion/deconvolution, so this mixed graph builds successfully. It
reduced steady RMVPE time from about `9.55 ms` to `8.22 ms` on the RTX 3060,
but the real native F0 gate was worse: F0 RMSE `0.1074 Hz`, coarse exact
`98.93%` (three mismatches), and complete waveform cosine `0.48987` against
the Applio FP32 path. The Synth-only cosine stayed `0.999999864`, confirming
that the change is confined to RMVPE. Consequently the production profile
keeps both RMVPE U-Net regions in FP32; the decoder-FP16 plan remains an
opt-in throughput experiment until it passes a multi-voice F0 gate.

`RvcTrtPipelineArtifacts` records all five region choices and the FFT frontend
mode in `pipeline.json`.
The 11200 test build is in
`target/trt-h16-r32-s16-fixed-mixed`; its full path measured HuBERT/RMVPE/
frontend/Synth GPU times of approximately 7.0/3.6/0.03/19.5 ms on the local
RTX system. The realtime F0 path uses the separate 4960 plan and measured
approximately 2.3 ms for RMVPE.

An A/B build with HuBERT and Synth both fixed at FP16 measured RMVPE 3.64 ms
with an FP32 U-Net and 2.30 ms with the mixed U-Net after warmup. On the sine
fixture the mixed path changed pitch by two coarse bins; pitchf RMSE was
`0.02735` (relative L2 `6.22e-5`) and audio RMSE was `0.00546` (relative L2
`0.0415`). These numbers are recorded in
`target/compare/rmvpe-mixed-precision-11200.json`; they are a sensitivity
check, not an Applio parity claim.

## Frontend precision and numerical reduction

The VFRM `rmvpe.mel_extractor.mel_basis` tensor matches Applio exactly.  The
native frontend still uses an FP32 convolution to spell out the 1024-point
Fourier transform, while PyTorch uses a cuFFT/FFT implementation.  The two
paths have the same operator semantics but different cancellation error on
near-zero high-frequency bins.  On the 44960-sample sine fixture this gives
mel RMSE `0.05734`, F0 RMSE `0.10238 Hz`, and two coarse-bin mismatches even
when every RMVPE region is FP32.

`VOCAL_TRT_RMVPE_FFT=1` enables an opt-in radix-2 FFT graph.  Its ten
butterfly stages reduce the corresponding Python float32 simulation to mel
RMSE `0.000418` and F0 RMSE `0.000687 Hz`.  TensorRT-RTX can build this graph,
but the current gather/butterfly expansion has an excessive first-use JIT
cost, so it is not the default realtime path.  The next implementation step
is to replace the expansion with one CUDA FFT plugin or a compact native FFT
operator while retaining the same FP32 boundary.
