# Independent audio endpoint formats

The GUI now opens capture and render using each endpoint's native shared mix
format. Sample rates and channel counts do not have to match. An explicit
WASAPI device selection also uses the native IAudioClient3 graph; CPAL is a
reported fallback for compatible recovery cases such as unsupported formats.
Device busy, unavailable, permission and service failures are reported directly.

## Ownership and data flow

```text
native capture callback / thread
  -> capture SPSC ring (capture rate, channels and slot size)
  -> DuplexWorkerAdapter (worker-owned filter state and scratch)
  -> engine (RuntimeConfig.input_format)
  -> optional RuntimeConfig.output_format conversion
  -> render format conversion and block assembler
  -> render SPSC ring (render rate, channels and slot size)
  -> native render callback / thread
```

`StreamIo::new_duplex` constructs independent ring configurations.
`StreamConfig::for_endpoint` translates a reference duration into native frames.
Rounding the slot size does not round the resampling ratio: the converter keeps
an integer rational phase across calls and produces variable sample counts.
The worker assembler publishes complete render slots, including when one
capture block produces zero or multiple render slots. A full render queue is
waited on before consuming input; endpoint callbacks never wait for the worker.
Latest-input scheduling resets converter/assembler history after skipped ring
sequences, avoiding interpolation across discarded audio.

Runtime input and output formats describe logical processing formats, rather
than requiring that both physical devices expose those exact formats. The GUI
continues to keep session format controls fixed. Selecting devices is sufficient
to negotiate their native sample rates and channel counts.

## Streaming conversion

`StreamingFormatConverter` uses a cached 1,024-phase Blackman-windowed sinc FIR.
The base filter has 64 taps, with length scaled by the decimation factor to
retain a useful stopband for higher input rates. Filter history, phase and scratch
are prepared before engine ownership is transferred. No per-block allocation is
performed by the converters. Equal sample rates bypass the FIR.

Supported converter bounds are 8–384 kHz and 1–64 channels. Equal channel counts
preserve channel order. Different counts use a normalized mono average followed
by replication into each destination channel. Thus a two-channel microphone can
feed a one-channel render device, and a mono voice can feed stereo or additional
channels. This is a voice routing policy; it does not infer surround speaker
positions or a beamforming matrix from the channel count. Averaging opposite
phase microphone channels can cancel their signal.

Native WASAPI exchanges f32 mix buffers directly with its rings. CPAL additionally
supports signed/unsigned integer PCM and f64, using bounded callback scratch for
sample representation conversion. Rate/channel conversion stays on the worker.
Native clock drift correction remains device-clock-based frame slip. The CPAL
fallback does not currently expose that native clock correction; bounded queues
and latest-input scheduling limit backlog, but it is not a continuous clock-servo
ASRC implementation.

Converter group delay is included in route metadata and source timestamps.
CPAL profiling uses the actual callback size and its capture/playback timestamps,
rather than assuming its driver period equals the ring slot duration.

## UI and failure handling

`AudioTelemetry.route` reports the actual audio backend, capture/render/worker
formats, conversion delay and the native startup failure if fallback occurred.
Studio session controls, Audio routing, Settings and Diagnostics share the same
route component. CPAL fallback has an amber notice, an accessible alert and an
expandable reason. This state is separate from the inference backend selection.
If an eligible native failure is followed by a CPAL failure, the error includes
both technical reasons. Startup reports retain the HRESULT, endpoint direction
and device identity, with recovery suggestions and an Audio routing shortcut.
An endpoint startup failure preserves the loaded idle engine and permits retry.
Period conflicts are negotiated separately from device occupancy; details and
current validation are in [Windows startup and audio recovery](windows-startup-and-audio-errors.md).

The model worker is spawned before the engine lease is transferred. Startup
failure therefore leaves the idle engine owned by the runtime. COM apartment
guards outlive the endpoint interfaces. IAudioClock is acquired only after
InitializeSharedAudioStream, when WASAPI services become available.

## Verification on 2026-10-07

- Root workspace: 141 tests passed, 9 ignored.
- Audio with all features: 39 tests passed; 3 live-device tests are opt-in.
- Tauri with DML + TensorRT: 11 library tests passed.
- Frontend type checking, production build and route UI assertions passed.
- Simulated duplex rings verified 16 kHz/2 ch -> 48 kHz/1 ch,
  48 kHz/1 ch -> 16 kHz/2 ch, and 44.1 kHz/2 ch -> 48 kHz/6 ch.
  Tests cover fractional block sizes, independent logical output conversion,
  constant-signal gain, duration preservation and render backpressure.
- FIR tests cover arbitrary callback partitioning, alias rejection, history
  reset, prepared-buffer reuse and partial-frame errors.
- Live Windows Remote Audio endpoint: 44.1 kHz/2 ch. Native initialization
  selected a 441-frame period and a 970-frame endpoint buffer; IAudioClock
  reported frequency 352,800. Explicit endpoint ID selection and wrong-direction
  rejection passed. CPAL received three real native-format output callbacks.
- Light/dark/620 px route component fixtures were checked with Playwright.
- The real Tauri instance was started twice without a capture endpoint. Both
  backend failure reasons appeared in the error report; dismissing it restored
  Ready with the loaded engine and saved configuration preserved.

There is no capture endpoint in this remote session. The actual microphone to
speaker path, hardware latency and successful fallback of a complete duplex
session cannot be measured here; the input cases above use simulated capture
blocks and real output endpoint checks are recorded separately.

### Converter profile

Run the reproducible profile without a device or model:

```powershell
rtk proxy cargo run --release -p vocal-audio --example profile_io_conversion
```

Each measured call represents 10 ms of input; 4,000 calls were sampled after
warmup. These are converter CPU times, not capture-to-playback latency.

| Input -> output | p50 | p95 | p99 | FIR delay | Steady allocations |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 kHz/2 ch -> 48 kHz/1 ch | 15.6 us | 18.9 us | 25.6 us | 2.000 ms | 0 |
| 48 kHz/2 ch -> 16 kHz/1 ch | 14.6 us | 15.8 us | 19.9 us | 2.000 ms | 0 |
| 44.1 kHz/1 ch -> 48 kHz/2 ch | 33.0 us | 36.5 us | 46.7 us | 0.726 ms | 0 |
| 48 kHz/1 ch -> 48 kHz/2 ch | 0.5 us | 0.6 us | 0.6 us | 0 ms | 0 |

Local catalogs, UI fixtures and build manifests stay untracked under `target/`.
