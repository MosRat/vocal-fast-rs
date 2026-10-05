# Realtime 44960 budget

`realtime_44960` is the model context window, not the amount of audio that a
realtime worker should publish on every callback.

## Window and playback chunk

The Applio/WebUI preset is defined at the internal 16 kHz rate:

| component | samples at 16 kHz | duration |
|---|---:|---:|
| new playback chunk | 4,000 | 250 ms |
| retained extra context | 40,000 | 2,500 ms |
| crossfade | 800 | 50 ms |
| SOLA search | 160 | 10 ms |
| TensorRT input window | **44,960** | **2,810 ms** |

For a 40 kHz RVC v2 model, one RVC frame is 400 output samples.  The steady
state callback advances 25 RVC frames, so the publishable output is:

```text
25 frames * 400 samples/frame = 10,000 samples at 40 kHz = 250 ms
```

If the audio device runs at 48 kHz, that chunk is resampled to 12,000 device
samples.  The 44,960-sample input is still submitted on every callback so
HuBERT sees the full context; realtime RMVPE uses its separate fixed 4,960
sample plan and appends the new F0 frames to the 1,024-frame cache.

## RTX 3060 measurement

The following fixed-profile run used `realtime_f0=1`, `advance_frames=25`,
native TensorRT-RTX graph strategy, and the shared HuBERT/RMVPE plans.

| Synth plan | total | HuBERT | RMVPE realtime | frontend | Synth | Synth tail |
|---|---:|---:|---:|---:|---:|---:|---:|
| static tail FP32 | 69.48 ms | 26.89 ms | 22.81 ms | 0.042 ms | 19.42 ms | 12,400 samples |

The steady callback budget is therefore about 69 ms of GPU/host pipeline time
for 250 ms of newly playable audio. This leaves roughly 3.6x compute headroom
on the RTX 3060, before device scheduling and audio-driver buffering.

## Current API boundary

`RvcTrtPipeline::process_realtime()` keeps the 44960-sample HuBERT/RMVPE
context. The fixed realtime Synth graph keeps the 281-frame encoder context,
then crops flow/decoder at `skip_head=250` and returns the 31-frame playable
tail (`4000 + 800 + 160 = 4960` internal samples),
performs normalized correlation over the 160-sample SOLA search, blends the
800-sample overlap, and publishes exactly the callback block (4000 samples at
16 kHz, or 12000 at 48 kHz).  This keeps the encoder/flow context identical to
Applio/WebUI; a tail-only Synth input changes that context and can produce
mechanical artifacts.  The Synth artifact's `min/opt/max_synth_frames`
profile now describes the complete RVC frame range used by the graph.

The `12,400` samples in the benchmark table are the device output copied from
the Synth tail (`31 * 400` at the model's 40 kHz rate). The engine adapter then
resamples that extended tail, runs SOLA/crossfade, and emits the 250 ms
callback block; it does not publish the whole Synth tail as a callback.

The 11200 voice-changer preset uses `skip_head=8`, `return_length=62`, and a
fixed 70-frame Synth context. On the same RTX 3060, one eager fixed-profile
call measured 55.9 ms (HuBERT 16.4 ms, RMVPE 11.1 ms, Synth 27.9 ms); after
native subgraph capture the steady last call measured 32.4 ms. The first
capture is included in the reported mean when a run uses multiple iterations.

Pipeline manifests produced before the full-context realtime change may still
contain the legacy tail-oriented min/opt values. They remain loadable when
their maximum covers the complete RVC frame count, but should be recompiled so
TensorRT's optimum tactic is selected for the actual realtime context shape.
