# Real-time audio I/O

`vocal-audio` separates the device callback from model work:

```text
capture callback -> BlockAssembler -> input SPSC ring -> worker -> output SPSC ring
                                                                    |
render callback <- BlockDrainer  <----------------------------------+
```

The callback side only copies f32 samples into pre-allocated slots, advances an
atomic index, and returns. It does not allocate, take a mutex, wait for model
inference, or perform resampling. The worker can acquire an input lease and
read the slot in place, then acquire an output lease and write the inference
result directly into the output queue. `AudioChunk` is only needed at an
explicit compatibility boundary; the hot path does not create one per block.
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

`DeviceIo::stats()` and `WorkerIo::stats()` expose queue depth, pushes, pops,
overflow counters, stale-block drops, and the high-water mark. The worker can
choose `wait_input()` for FIFO processing or `wait_latest_input()` when bounded
age is more important than preserving every block. `wait_latest_input()` only
drops blocks from the consumer side, so a producer never overwrites a slot that
the worker is reading. `AudioIoStats::estimated_queue_latency()` converts the
current ring depth to a lower-bound queue budget. For example, a 480-frame
block at 48 kHz is 10 ms, so three queued input blocks plus three queued output
blocks represent 60 ms of ring residency before device safety buffers and
model execution are included.

`StreamIo::new_with_queue_config` and the WASAPI `*_with_queue_config` open
methods accept independent `StreamQueueConfig` capacities. The default
`StreamIo::new` remains symmetric for simple callers; a low-latency duplex
setup can use three input blocks and two output blocks so render backlog is
bounded more tightly than capture scheduling slack. Both WASAPI `open` entry
points use that 3/2 profile by default; pass an explicit queue config to
override it.

The producer publishes a generation after every commit. On Windows the worker
waits with `WaitOnAddress` and the callback wakes it with
`WakeByAddressSingle`; the callback never takes a mutex or sleeps. Portable
builds retain a bounded polling fallback.

The zero-copy worker loop is shaped like this:

```rust,ignore
let (input, mut output) = worker.try_input_output()?;
convert(input.samples(), output.samples_mut());
output.commit();
```

When the render queue is intentionally shallow, a worker can use
`WorkerIo::wait_output(timeout)` instead of retrying `try_output()`. The
consumer only signals the space generation when it releases a slot from a
full ring, so the normal render callback path does not issue wakeups.

The input lease is released after the model has finished reading it. Dropping
an output lease aborts its reservation without publishing a partially written
block.

For an actual capture-to-render profile, opt into timestamps only in the
profile build. Create one `AudioClock` per stream, call
`AudioProducer::try_acquire_at(clock.now())` (or `push_at`) when a capture
block is published, copy `AudioReadLease::timestamp()` to the output lease
with `set_timestamp`, and read the timestamp from the render-side lease. The
render thread can then report `clock.elapsed(timestamp)` as the measured age.
When using `BlockDrainer`, the same source timestamp is returned in
`DrainReport::source_timestamp`.
`cpal_backend::build_duplex_profiled` and
`wasapi_backend::WasapiStream::open_profiled` collect latest, mean, maximum,
and approximate p50/p95/p99/p99.9 ages in their `latency()` snapshots. The
snapshot also reports underflow samples, packet timestamp validity,
discontinuities, queue high-water, overflow/stale-drop counts, and maximum
endpoint padding. Profiled device callbacks also expose callback period
count/mean and callback jitter percentiles, so scheduling jitter can be
distinguished from queue residency. Native WASAPI compares callback intervals
with the actual period selected from `IAudioClient3`, rather than assuming the
model block size. A zero sample count means the inference worker did not
propagate timestamps, rather than that the device measured zero latency.
The default methods write `AudioTimestamp::ZERO`, so production streams do not
pay for a clock read on every callback.

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

`wasapi_backend::WasapiStream::open(..., WasapiMode::SharedEvent)` now selects
the Windows 10+ `NativeWasapiStream` path. It validates the endpoint's native
f32 mix format, chooses a period aligned to `GetSharedModeEnginePeriod`, and
initializes the stream with `InitializeSharedAudioStream` and event callbacks.
It intentionally does not enable format conversion; callers should use the
device's native sample rate and channel count or handle conversion outside the
callback. The older wrapper path is available only through
`WasapiStream::open_legacy*`; `ExclusiveEvent` continues to use that path.

The optional `cpal-backend` adapter is a portable first integration. It requests
f32 and a fixed callback size, then still uses the assembler/drainer because a
driver may provide another size. It rejects devices whose default sample rate
or channel count does not match the stream format, avoiding an implicit
conversion path. On Windows this uses cpal's WASAPI host, but the adapter does
not promise WASAPI exclusive mode. A native `wasapi` adapter
can replace only the device layer while keeping `StreamIo` unchanged.
The native adapter keeps one pre-allocated `f32` endpoint buffer per event
thread and views it as bytes only at the WASAPI FFI boundary, so capture and
render avoid per-sample format conversion. When the endpoint packet is exactly
one model block, capture writes directly into a ring lease and render writes a
ring lease directly through `GetBuffer`/`ReleaseBuffer`'s high-level wrapper;
the assembler/drainer staging copy is then skipped. Variable packet sizes and
underflow still use the bounded staging buffers.

Capture and render event threads register with Windows MMCSS under the
`Pro Audio` task category. Registration is scoped to each thread and is
reverted on exit. A model worker can call `WorkerIo::register_mmcss("Audio")`
on its own inference thread when it must share the same scheduling policy; the
guard must be retained until that loop exits.

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
