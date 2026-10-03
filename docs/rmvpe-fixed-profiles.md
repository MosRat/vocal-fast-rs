# Fixed RMVPE profiles

Every newly compiled pipeline contains two RMVPE engines:

* `rmvpe-realtime.plan`: `min=opt=max=4960` samples (the streaming F0 window).
* `rmvpe-full.plan`: fixed at the selected application window. The broad
  legacy `reference` profile keeps its `[11200, 35520, 44960]` range for
  compatibility; exact presets use `min=opt=max`.

For the voice-changer profile the second value is 11200. For the current
WebUI/Applio profile it is 44960. `rmvpe_plan` remains a serialized alias to
the full plan so older manifests continue to load. `process_realtime()` binds
the 4960 context and `process()` binds the full context.

The fixed 44960 build was exercised with both eager full-window and realtime
F0 calls. The latter reported `pitch_frames=32`, while the full call reported
`pitch_frames=282`; both paths captured and replayed the TensorRT native CUDA
graph at the fixed shape.
