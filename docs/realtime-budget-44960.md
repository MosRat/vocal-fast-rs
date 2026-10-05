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

The following run used 10 warmups, 30 iterations, `realtime_f0=1`,
`advance_frames=25`, native TensorRT CUDA Graph replay, and the shared HuBERT /
RMVPE plans.  Only the Synth plan changed between rows.

| Synth plan | mean total | last total | HuBERT | RMVPE realtime | frontend | Synth | output returned |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed FP16 | 47.00 ms | 46.73 ms | 4.53 ms | 3.34 ms | 0.012 ms | 37.05 ms | 112,400 samples |
| coherent FP16 `0x2` | 40.17 ms | 37.70 ms | 5.02 ms | 3.34 ms | 0.012 ms | 28.44 ms | 112,400 samples |

The steady callback budget is therefore about 40--47 ms of GPU/host pipeline
time for 250 ms of newly playable audio.  This leaves roughly 5.3--6.2x
compute headroom on the RTX 3060, before device scheduling and audio-driver
buffering.

## Current API boundary

`RvcTrtPipeline::process_realtime()` currently returns the complete Synth audio
window (`112400` samples for 281 RVC frames).  It maintains the F0 cache and
accepts `advance_frames`, but it does not yet crop the returned waveform or
apply host-side overlap, crossfade, and SOLA alignment.  A production playback
worker must publish the 25-frame/10,000-sample tail (after its overlap policy)
instead of enqueueing all 112,400 samples.  Until that output joiner is added,
the graph benchmark demonstrates compute real-time capability but is not the
final device callback integration.
