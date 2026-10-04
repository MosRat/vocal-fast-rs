# Shared runtime context

`vocal-runtime` is the application core shared by the CLI and Tauri. It owns
configuration, model selection, engine construction, processing, lifecycle,
metrics, and stable runtime errors. The two frontends only provide control and
transport:

| Layer | Owns |
| --- | --- |
| `vocal-runtime` | `RuntimeConfig`, atomic `ConfigStore`, model loading/cache ownership, engine, lifecycle and snapshots |
| Tauri | commands, channel events, device enumeration, CPAL session and UI DTOs |
| CLI | argument parsing, terminal output, model packaging/inspection and CLI-only benchmarks |

`RuntimeContext` is cloneable. Clones share the same state and operation gate;
only one model or processing operation can mutate the engine at a time. Audio
callbacks do not acquire the GUI state lock. The Tauri worker holds a context
clone and calls `process` after taking a block from the audio ring.

Model packages are loaded through `ModelLoader::load_cache_shared`. The runtime
keeps a strong `Arc<ModelPackage>` while a model is selected, while the process
wide model cache keeps weak entries. This makes repeated selections from GUI
and other controllers reuse parsed tensor data without keeping unused models
alive forever.

The lifecycle is shared by both controllers:

```text
Created -> Loading -> Ready -> Running -> Stopped
             |          |       |
             +--------> Failed <-+
```

`Stopped` can return to `Running` when the engine remains loaded. A failed
runtime can be returned to `Ready` with `clear_error` when an engine is still
available. Configuration updates use an optimistic `config_revision` so a
stale GUI or CLI command cannot overwrite a newer revision.

Use `load_model_descriptor` for a controller-provided descriptor (for example
the CPU demo fixture); file based controllers should use `load_model`, which
also persists the selected model path through `ConfigStore`.
