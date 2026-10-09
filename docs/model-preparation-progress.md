# Model preparation feedback and Chinese model bundles

## Visible progress

Studio and Settings share the `preparation-progress` component. Loading a model,
saving idle engine configuration and preparing TensorRT plans show a highlighted
status card, a progress bar and elapsed time. Related buttons show a spinner and
an action caption. Library scans show feedback next to the model list.

Preparation reports these real stages through the Tauri channel:

1. Inspect model metadata and component directories.
2. Check/extract embedded HuBERT and RMVPE sources.
3. Read Synth weights and the retrieval index.
4. Check the pipeline fingerprint and cached plan files.
5. Compile HuBERT, RMVPE realtime, RMVPE full window, frontend and Synth.
6. Publish plans and the manifest to the cache.
7. Load the processing engine and prepare device memory.
8. Complete, or report the failing stage/cause.

The builder reports completed plans out of five. This measures completed work,
not predicted wall time; stages with no measurable completion count use an
indeterminate bar. A cache hit skips compilation and remains visible during
engine initialization. Progress appears at the top of the main workspace and
next to the relevant backend controls, including automatic model restoration.
The splash only reports configuration, filesystem metadata and device probes;
it never waits for model weights, plan caches or engine creation.

Shell bootstrap publishes an immutable device/backend/metadata catalog. After
subscribing, reading its initial snapshot and painting, the visible main window
claims the saved model restore once. A manual model action or saved configuration
change cancels that pending claim. Reloads and auxiliary windows cannot start a
second restore, and model preparation failure leaves the shell `Ready`.
`boot.elapsedMs` stops at shell readiness. Browser performance entries named
`vocal-workspace-first-paint`, `vocal-model-restore-requested` and
`vocal-model-restore-duration` expose the separate frontend/engine timings.

`PreparationTracker` has its own small status lock. Progress publication never
asks the Runtime actor for a snapshot. The webview advances its timer locally
while engine initialization occupies that actor, and suspends normal periodic
snapshot polling during preparation. A guard marks interrupted operations as failed;
operation IDs prevent an older handle from changing a newer operation's status.
Each stage has a revision so delayed snapshots cannot replace a terminal update;
elapsed time is never used as a state version. Auxiliary windows request one
recovery snapshot after five seconds without a progress update.

Register each JavaScript `Channel` once. Creating two Rust senders by passing
the same JS Channel to two registration commands resets its ordering counter and
the first sender's drop can end the JS callback. The UI registers with
`subscribe_preparation`, then retrieves the initial full snapshot separately.

### Startup verification (2026-10-07)

Windows Debug, DirectML, `default-complete-with-index.vfrm`:

- Shell initialization completed in 1,071 ms; no engine existed at shell readiness.
- The main page painted before sending its saved-model restore request.
- Weight/engine preparation then took 3,797 ms with progress on the main page.
- The immutable startup catalog query returned in 1.6 ms while preparation was visible.
- Studio/Diagnostics navigation remained usable; audio Start stayed disabled.
- Reloading the main webview did not issue another restore or advance its operation ID.
- Boot elapsed time remained frozen; the final engine was Ready with audio inactive and no errors.

The Playwright/CDP report and loading/ready screenshots are generated under
`target/windows-build/Debug-TensorRt/workspace-startup-verification.json` and
`workspace-model-{loading,ready}.png`. These timings describe this local DirectML
run; they are not TensorRT compilation or hardware audio latency measurements.
Existing and newly opened auxiliary windows can receive preparation events.

## Downloaded complete Chinese bundles

Artifacts remain untracked under `target/chinese-rvc/`:

| Bundle | Voice / rate | File size | Retrieval index |
| --- | --- | --- | --- |
| `maobailing-complete.vfrm` | 猫百灵, Chinese synthetic female voice, RVC v2 / 40 kHz | 846.3 MiB | 88.3 MiB |
| `wanxin-complete.vfrm` | 婉心, Chinese female voice, RVC v2 / 48 kHz | 947.9 MiB | 187.7 MiB |

Each includes the Synth checkpoint, complete canonical HuBERT and RMVPE weights,
and the populated FAISS index. They use the automatic TensorRT preparation path.
Choose `F:\Workspace\RustProjects\vocal-fast-rs\target\chinese-rvc` as the
library folder, scan, select a `*-complete.vfrm` entry, and click Load. No acoustic
source paths or pipeline JSON need to be supplied.

Sources:

- [maotianchen/maobailing_rvc](https://huggingface.co/maotianchen/maobailing_rvc),
  revision `e56357e07e6788617318733dce1e10eb92889633`, Apache-2.0. The publisher
  describes training audio generated with index-tts2.
- [chaye741/RVC-Voice-Models](https://huggingface.co/chaye741/RVC-Voice-Models),
  revision `acfd9f836ca1fc862a594bc303cfccee8d565e94`, `女声-婉心.pth` and its
  paired index. The publisher provides no license declaration in the repository
  metadata; this fact is recorded in the provenance.

The initially evaluated 青年 male checkpoint is v1/256 features and was excluded
from the delivered complete bundles because the shared acoustic package uses
768 features. Its verified source files remain in `downloads/qingnian/`.

`*-provenance.json` records repository revisions, source URLs, file sizes and
SHA-256 hashes for downloads, the bundle and acoustic sources.
`*-inspection.json` records component validation. Checkpoints are inspected with
PyTorch `weights_only=True`; canonical conversion uses the Rust restricted
checkpoint loader, and bundling streams acoustic payloads.

Reproduce from the repository root:

```powershell
rtk cargo build -p vocal-model --example prepare-rvc
rtk proxy .venv/Scripts/python.exe scripts/download-chinese-rvc.py
```

## Windows verification, 2026-10-07

RTX 3060 / SM 8.6, TensorRT RTX 1.6.1.120, CUDA driver API 13.4,
`realtime_44960`, native graph replay, audio inactive:

| Operation | Build cache | Preparation + engine load |
| --- | --- | --- |
| 猫百灵 first load | Miss | 17.481 s |
| 猫百灵 repeated load | Hit | 8.040 s |
| 婉心 first load | Miss | 20.584 s |

Playwright observed native compilation stages, disabled action controls,
continuous elapsed time during engine load, and synchronized Studio/auxiliary
Settings feedback. An invalid path produced a terminal error while retaining
the previous model/config revision; the subsequent load recovered. The original
default model was restored after validation.

Both complete bundles also completed a real TensorRT offline streaming smoke
test: 480 blocks of 480 mono samples at 48 kHz per model, fixed output block size,
all values finite and nonzero output after warmup. Observed maximum output peaks
were 0.00937454 (猫百灵) and 0.01325126 (婉心). The evidence is
`target/windows-build/Debug-TensorRt/chinese-model-inference.json`. This used a
synthetic input signal; listening quality and physical microphone latency were
not measured.

Generated evidence: `target/windows-build/Debug-TensorRt/progress-report.json`
and `progress-*.png` screenshots. These timings are local observations and vary
with GPU load, filesystem caches, profile and precision choices.

The final cache-hit/window-open regression completed in 7.626 s. A Settings
window opened during engine initialization showed the cache-hit badge and
transitioned to Ready along with Studio. See `progress-final.json`,
`progress-cache-hit.png`, and `progress-settings.png`. The final UI retains the
persisted `target` library folder, has both complete bundles in its refreshed
catalog and no test configuration draft; the debug process remains running.

Checks: frontend build, Python syntax, workspace formatting, workspace
all-features check, 128 workspace tests passed / 6 ignored, and 11 Tauri tests
passed. An unrestricted workspace test invocation was blocked by another active
DirectML validation executable at link time; the successful workspace run used
`--lib --bins --tests` to avoid overwriting that executable.
