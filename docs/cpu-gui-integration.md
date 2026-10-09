# CPU runtime and GUI integration

## Configuration and lifecycle

`vocal-engine::CpuExecutionConfig` is the portable execution policy shared by
Runtime, CLI and Tauri. The CPU SDK is optional in portable builds. Existing
configuration files default to:

```json
"cpu": {
  "threads": 0,
  "spinWaitUs": 0,
  "pinToPerformanceCores": false,
  "profile": false
}
```

Zero threads chooses `min(available logical processors, 6)`, with a minimum of
one. This bounds the persistent intra-operator pool and avoids automatically
taking every logical processor. Manual counts 1–256 are accepted. The scheduler
can prefer Windows performance cores; absent topology falls back to OS policy.
Spin wait is bounded to 1000 µs and defaults to sleep. Profiling adds per-node
clock reads and logs the ten slowest operators per inference chunk. FP32,
fusion, CPU ISA dispatch and channels-last layout stay enabled; experimental
quantization is not enabled by GUI selection.

The three existing session profiles also select CPU streaming geometry:

| Profile | CPU context at 16 kHz | Output chunk | Initial context warmup |
| --- | ---: | ---: | ---: |
| Reference | 11200 | 512 ms | 1536 ms |
| Voice changer | 11200 | 512 ms | 1536 ms |
| Applio | 44960 | 250 ms | 3250 ms |

These buffering intervals do not measure capture-to-render latency. The first
voiced output additionally waits for the next inference chunk and device
buffering. Reference chooses the compact CPU shape to give inference more
time. Full-context CPU execution can exceed its 250 ms budget on desktop CPUs.

Scheduling edits save automatically while idle. Session profiles remain staged
until **Save profile**. Thread policy, profiling, speaker and effective context
changes prepare a replacement CPU engine. Pitch, automatic pitch matching,
protect, index ratio, RMS and device routing reuse the current engine. Runtime
validates and prepares candidates before saving/replacing state. Failed
preparation retains the engine, persisted configuration and revision. A running
worker or a worker awaiting ownership return rejects configuration updates.

### Speaker selection

`speakerId` is a shared session setting (default `0`) for CPU, DirectML and
TensorRT, editable in both Studio and Settings. Canonical VFRM inspection
reports `components.speakerCount` from the validated Synth configuration;
inspection reads metadata and tensor directories without allocating weights.
For a loaded canonical model, the GUI accepts only integer IDs in
`0..speakerCount`. Legacy files and external manifests report an unknown count;
the backend validates the speaker embedding when loading. Choosing a different
model does not silently reset the saved speaker ID.

Speaker edits use the existing revisioned idle autosave and candidate-engine
validation, like thread settings. Load is disabled until pending configuration
edits are saved. CPU passes the saved thread, scheduling, context and speaker
selection into its graph engine; active sessions lock these controls. Invalid
IDs show an error and retain the previous engine and saved configuration.

## Model preparation and availability

The real CPU path accepts complete canonical RVC v2 / 768-feature bundles,
materializes HuBERT/RMVPE sources in the shared extraction cache, and compiles
oneDNN graphs from the weights. It requires no TensorRT plans, Torch, MKL,
OpenMP or GPU driver. Retrieval is optional; nonzero index ratio requires an
included index. The main page reports weight/operator/buffer preparation through
the existing preparation Channel and progress display. Startup still probes
devices and files without building the engine before showing the workspace.

Automatic selection prefers TensorRT RTX, a compatible physical DirectML GPU,
real CPU inference, then Passthrough. Virtual/remote DML adapters are selectable
but do not displace CPU automatically. Builds without `cpu-graph` disable the
CPU choice and report why; portable CPU fixtures are not advertised as voice
conversion. The device catalog includes logical processor count, suggested
threads and the loaded oneDNN version.

All Windows build variants now include `cpu-graph`, `dnnl.dll` and its license
and third-party notices. Default Cargo/Tauri development builds include CPU
and DirectML. The CPU-only distribution requires no GPU SDK. The build helper
checks the CPU headers, import library, DLL and notices before building and
uses an explicit resource allowlist to exclude stale dependencies.

## Validation

`cpu_runtime_probe` uses the same public Runtime/worker handoff API as the GUI,
with a private config store and file input. It checks finite audible output,
protect updates preserving the engine, thread-policy reconstruction, failed
speaker preparation retaining state, active update rejection, restart and
unload. Reports and WAVs remain untracked under `target/cpu-runtime-integration/`.

```powershell
rtk cargo run -p vocal-fast --release --features cpu-graph --example cpu_runtime_probe -- <complete.vfrm> <input.wav> target/cpu-runtime-integration
```

Playwright with simulated Tauri IPC also verifies thread/affinity/spin/profile
automatic saves, manual session-profile saving, remote configuration snapshots
and active-session control locks. Its report and screenshot are in
`target/windows-ui-probe/cpu-ui-report.json` and `cpu-settings-dark.png`.

File replay validates the engine/control lifecycle. It is not a live hardware
latency measurement or a listening quality score.

On 2026-10-08 the complete Elysia Chinese bundle and the continuous Liangzhu
male speech WAV passed the probe with automatic six threads, then two threads
with performance-core scheduling. Loading took 1.073 s and replacement took
1.558 s. Observed automatic chunk processing was 233–408 ms for 512 ms chunks;
two threads reached 642 ms and exceeded that budget. These are observations
from one file replay with machine background workload, rather than an isolated
performance benchmark. The GUI also derives larger capture/render queues and
playback prefill from `preferred_batch_frames`, through the existing
`RealtimeBufferConfig::for_batch`, so a long CPU inference does not run on the
three-slot monitoring queue.

Verification gates: workspace/all-feature checks; 196 default workspace tests
(13 hardware/fixture tests ignored); 94 CPU/engine/runtime library and integration
tests; 26 Tauri tests (one hardware test ignored); 25 frontend tests; frontend
build and CPU Playwright interactions. Generated evidence remains under `target/`.

### Speaker control regression checks

After connecting the shared speaker control, verification passed: formatting,
workspace all-feature checking, 200 default workspace tests (13 ignored), 69
engine/runtime tests with all features (one ignored), 28 frontend tests, and the
frontend production build. Model fixtures cover single-speaker and three-speaker
canonical bundles, acoustic-only files, and unknown legacy speaker counts.

Headless Edge with simulated Tauri IPC verified both Studio and Settings,
CPU/DirectML/TensorRT switching without losing the speaker selection, thread and
speaker autosave, Load waiting for pending saves, negative/fractional/empty and
out-of-range ID rejection, remote synchronization, and active-session locks.
The report is `target/windows-ui-probe/cpu-speaker-ui-report.json`; the screenshot
is `target/windows-ui-probe/cpu-speaker-settings.png`. This checks GUI behavior;
real CPU inference and worker ownership are covered separately by the probe above.
