# Real-time audio I/O

`vocal-audio` separates the device callback from model work:

```text
capture callback -> BlockAssembler -> input SPSC ring -> worker -> output SPSC ring
                                                                    |
render callback <- BlockDrainer  <----------------------------------+
```

The callback side only copies f32 samples into pre-allocated slots, advances an
atomic index, and returns. It does not allocate, take a mutex, wait for model
inference, or perform resampling. The worker consumes a fixed block as an
owned `AudioChunk`, runs the inference engine, and publishes an output block.
`BlockAssembler` and `BlockDrainer` handle a device callback size that differs
from the model block size. Output underflow is rendered as silence. Queue full
and underflow counters are exposed for telemetry.

`StreamIo::split()` returns two endpoint pairs:

- `DeviceIo`: input producer and output consumer for the native callback.
- `WorkerIo`: input consumer and output producer for the inference thread.

The ring is intentionally SPSC. Do not clone endpoint handles or run two
producers/consumers against one queue. A monitor path can call
`AudioConsumer::try_copy_latest_into` to drop stale blocks and bound latency;
the main render path should use FIFO draining to preserve every block.

## Windows backend choice

The native Windows low-latency API is WASAPI. On Windows 10 and later,
`IAudioClient3` exposes the legal shared-mode engine periods through
`GetSharedModeEnginePeriod` and lets an application request one with
`InitializeSharedAudioStream`. The requested period must be a multiple of the
driver's fundamental period and inside its minimum/maximum range. An
event-driven client then reads from `IAudioCaptureClient` and writes to
`IAudioRenderClient` when the endpoint signals the event.

Use shared mode with the requested engine period when interoperability with
other applications matters. Use exclusive mode only after probing the device's
format and period support; it can reduce conversion latency but prevents other
clients from using that endpoint in the selected format. The worker/ring layer
is independent of this choice.

The optional `cpal-backend` adapter is a portable first integration. It requests
f32 and a fixed callback size, then still uses the assembler/drainer because a
driver may provide another size. On Windows this uses cpal's WASAPI host, but
the adapter does not promise WASAPI exclusive mode. A native `wasapi` adapter
can replace only the device layer while keeping `StreamIo` unchanged.

## Initial tuning

Start with the device's native sample rate (commonly 48 kHz), 10 ms callback
period, and two to four queued blocks. For 48 kHz this is 480 frames per block.
Increase the queue only when telemetry shows capture overflow or render
underflow; a deeper queue increases round-trip latency. The RVC model can
resample its worker input to 16 kHz while the device remains at its native
rate. Keep resampling and channel conversion outside the native callback.

The reference realtime implementations follow the same shape: a callback
submits one block, a worker processes it, and a monitor/output queue drops old
data when it falls behind. They also configure WASAPI/ASIO-specific settings
at stream creation rather than changing them from the callback.

