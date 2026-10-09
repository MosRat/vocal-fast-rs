# Processing backend selection

The shared `vocal-runtime` capability API reports backend availability and a
recommendation. Studio shows the selector below the selected voice model;
Settings uses the same component. Audio sessions lock backend and plan edits.

## Selection and preparation

Complete bundles now prepare plans automatically on Load; see
[complete voice bundles](complete-model-bundles.md) for the component display,
container format, memory policy, and measured workflow.

- Fresh GUI configurations use Automatic. Startup resolves it to TensorRT RTX
  when the compiled backend, current CUDA device, driver and CUDA runtime probe
  are usable; otherwise it prefers a compatible physical DirectML GPU, then
  pretrained CPU inference, then Passthrough. Legacy configurations without
  `backendAuto` preserve their manual choice.
- An unavailable saved backend falls back during startup, before restoring the
  model, with a visible startup notice. Refresh devices updates the selector;
  it never replaces an active engine.
- RVC CPU executes pretrained HuBERT, RMVPE, retrieval and RVC v2 Synth graphs
  with standalone oneDNN and RustFFT. Builds without `cpu-graph` disable this
  choice; the portable test fixture is never advertised as voice conversion.
  CPU scheduling settings, buffering hints and idle reload semantics are
  described in [CPU GUI integration](cpu-gui-integration.md).
- Availability does not mean plans are ready. TensorRT can be selected and a
  model package loaded before plans exist. The parsed package stays resident,
  the engine is absent, and Start is rejected until plans are prepared.
- Load atomically applies an edited backend/pipeline and loads the voice. Its
  revision check protects backend edits from concurrent configuration updates.
  Unrelated folder/profile/voice-control drafts are retained. When the backend
  is unchanged, loading keeps the existing model-only command semantics.
- Existing `pipeline.json` files can be supplied directly. Load validates the
  native engine; the plans must have been built for the selected voice model.
  Alternatively, supply HuBERT and RMVPE VFRM paths and use **Build plans & load
  model**. This saves edited settings, builds or reuses the model-specific cache,
  and then loads the chosen model.

## Windows detection

The probe loads `nvcuda.dll` from System32, queries CUDA device zero (the device
used by the current engine), its compute capability and driver API version,
then creates and synchronizes a temporary CUDA stream. It does not deserialize
models or plans. Discovery runs on the control plane; GUI requests run on a
blocking executor. It is never invoked for an audio block.

The current portable plan builder accepts SM 8.6, 8.9, 12.0 and 12.1. Turing
needs separate build flags and precision policies that this application does
not yet set. This matches NVIDIA's [default engine portability policy](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/support-matrix.html).
Build support, missing driver/device, unsupported GPU and CUDA runtime failures
produce disabled options with reasons. Model loading, backend changes, plan
preparation and audio startup also validate availability in Rust. No driver DLL
is bundled; the CUDA/TensorRT runtime libraries remain application dependencies.

## Validation on Windows

The local RTX 3060 (SM 8.6, CUDA driver API 13.4) was detected and recommended.
Native WebView tests covered CPU/Passthrough switching, pending TensorRT plans,
real TensorRT engine loading using existing Default Voice plans, failed manifest
loading without replacing persisted configuration, retry, refresh, Settings
reuse, and preservation of unrelated drafts. Headless UI tests simulated missing
hardware and checked disabled GPU options plus recovery on refresh. Runtime
tests cover missing compiled support, failed probes, revision conflicts and
startup fallback. This is engine/control validation, not a microphone loopback
or physical audio latency measurement.

Generated reports and screenshots are in `target/windows-build/Debug-TensorRt/`:
`backend-regression.json`, `backend-unavailable-preview.json`, and
`processing-backend-after.png`.

The initial TensorRT integration passed frontend TypeScript/Vite checks and its
affected Runtime/Tauri tests. The subsequent native DML integration now passes
the workspace gates, as recorded below.

## DirectML device selection — 2026-10-07

The shared Processing backend section now presents implementation selection
first, then a DML GPU card with memory information and model preparation state.
The optional source manifest is in an **Advanced sources** disclosure. Studio,
Settings and the separate settings window reuse the same component and handlers.

- **Automatic** prefers a compatible physical GPU in Windows high-performance
  order. Virtual/remote adapters have distinct labels, including when their
  driver exposes the same GPU name as the hardware adapter.
- Software/incompatible adapters are disabled with reasons. A disconnected
  saved device remains visible as unavailable; Load/Save are blocked until the
  user chooses a valid device or Automatic.
- GPU selection is saved as `directmlDeviceId`. Load applies it atomically with
  backend/model selection and retains other draft edits. Settings Save also
  applies it to an idle engine. Changes require an engine rebuild only for DML;
  an audio worker prevents all such edits.
- Source manifests retain their graph/precision policy, while the GUI's GPU
  selection overrides their adapter setting. No TensorRT plans are built for
  a DML device change.

Validation on the local RTX 3060 included:

| Check | Result |
| --- | --- |
| TypeScript/Vite build | Passed |
| Root formatting and all-feature workspace check | Passed |
| Full workspace tests | 135 passed, 9 ignored |
| Tauri library tests, TensorRT + default DML | 11 passed |
| Explicit native adapter ID with an invalid index | Correct GPU selected |
| Missing native adapter ID / software adapter | Rejected; no silent fallback |
| Native WebView GPU selection and Load | Saved exact GPU ID; model/engine Ready |
| Restart with an explicit saved GPU | Config and model restored successfully |
| Settings page | Same selected GPU as Studio |
| Narrow viewport (680 CSS pixels) | Card stacks vertically; no horizontal overflow |
| UI logic for stale/unsupported/absent GPUs, running lock, escaped labels and draft rebase | Passed |

The physical GPU and its virtual/remote aliases were tested; a second physical
GPU was not available. This is device/model integration validation and does not
measure microphone-to-speaker latency. The user's original automatic GPU
preference was restored and the current DML model remains Ready in Debug.

Generated evidence is under `target/windows-build/`: `dml-device-ui.json`,
`dml-device-restart.json`, `dml-device-final.json`, `dml-device-final.png`,
`dml-device-settings.png` and `dml-device-narrow.png`. These files remain untracked.
