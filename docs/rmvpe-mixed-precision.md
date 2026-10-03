# RMVPE mixed precision

The native RMVPE graph now has explicit precision regions. The shipped realtime
configuration is:

| region | type |
| --- | --- |
| STFT and mel frontend | FP32 |
| U-Net encoder | FP16 |
| U-Net decoder | FP16 |
| bidirectional GRU and recurrent state | FP32 |
| salience/logits and weighted decode | FP32 |

The GRU boundary is an explicit cast, so TensorRT cannot accidentally mix a
half activation with a float recurrent state. The public `pitchf` binding stays
FP32. Override the U-Net regions with `VOCAL_TRT_RMVPE_ENCODER_PRECISION` and
`VOCAL_TRT_RMVPE_DECODER_PRECISION`; the sensitive regions default to FP32.

`RvcTrtPipelineArtifacts` records all five region choices in `pipeline.json`.
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
