# GUI model management

The **Models** workspace edits VFRM display metadata and imports voice models
from local files, Hugging Face, HF Mirror and ModelScope public repositories.
Downloaded weights and generated models remain outside Git.

## Editing details

Select a VFRM from the configured library. Name, notes, attribution/source,
author, language, license and tags are editable. Model ID, architecture,
sample rate, speakers and tensor shapes stay read-only because they describe
the actual weights. Metadata drafts are independent of session configuration
autosave. Click **Save model details** while audio is idle.

The model crate validates the original directory, streams tensor bytes into a
temporary file, relocates tensor offsets, validates the replacement, then
renames it over the original. Computational metadata, index and embedded
acoustic payloads are preserved. A file revision token rejects stale edits.
Runtime updates its summary/descriptor without rebuilding the engine or
cloning the resident tensor package. Active sessions reject metadata edits.
Editing the container changes its file identity; future file-based cache
lookups can invalidate. Saving display information never triggers plan builds.

## Download / import

1. Choose HF Mirror, Hugging Face or ModelScope and enter `owner/model` or the
   provider's repository URL. An explicit branch/tag/revision is optional.
2. Browse files, choose a PTH checkpoint, VFRM or ZIP, and optionally choose
   the matching retrieval `.index` for a PTH checkpoint.
3. Enter a display name and destination. The default is
   `%LOCALAPPDATA%/vocal-fast/model-library`.
4. Choose **Download & convert**. Progress, transferred bytes, errors,
   cancellation and completion appear in the model workspace. Completion
   refreshes the library. **Use in Studio** selects the result; it does not
   automatically replace a running engine.

The same native conversion supports local PTH, VFR/VFRM, or a ZIP containing
one voice model and at most one retrieval index. No Python or PyTorch runtime
is required. PTH metadata determines RVC version, F0 use and sample rate.
Non-RVC checkpoints, incompatible indices and ambiguous archives produce
actionable errors. ZIP paths, symlinks, extraction size, checkpoint storage
and metadata sizes are bounded.

Complete VFRM files keep their embedded HuBERT/RMVPE/index. PTH imports can
reuse acoustics from a complete model already in the library, or use an
explicit complete bundle / canonical HuBERT and RMVPE VFRM pair. Without
acoustic sources, the output is clearly marked Synth-only. Raw acoustic PT
and ONNX exports are not converted by this importer. Some models also lack
a retrieval index; the UI reports that separately.

The GUI lists supported *file extensions*, not a guarantee that every PTH in
an arbitrary repository is an RVC model. Conversion validates the checkpoint.
Private/gated repository authentication is not implemented.

## Ownership, communication and persistence

- `vocal-model::metadata`: bounded inspection and atomic metadata edits.
- `vocal-runtime::model_library`: providers, download cache, conversion and
  an independent job worker, shared with CLI/integration tools.
- `src-tauri/model_commands.rs`: filesystem commands and native file picker.
- `model-management.ts`, `model-library-types.ts`, `model-management.css`:
  separate drafts, task state, presentation and interaction.
- A single active import per process; cancellation includes job ID. GUI
  accepts only increasing job/revision pairs. Tauri uses the existing Channel.
- Destination travels in task status so reconnects and multiple windows
  refresh the correct library.
- Download cache uses repository/revision/file identity and a per-file OS
  lock. Partial downloads resume only after validating Content-Range; exact
  size and supplied SHA-256 are checked before publishing.
- HF branches resolve to immutable commit SHA. ModelScope downloads use
  each file's commit revision and SHA-256. Its unpaginated directory endpoint
  is traversed per directory; potentially truncated lists fail visibly.
- Network work never holds the runtime engine operation lock. Conversion
  stages in a private directory; one directory rename publishes model and
  provenance together. Cancelled/failed jobs leave original files intact.
  During a native conversion step cancellation waits for that step to finish.
- Hidden staging/download/acoustic-cache entries are excluded from scanning.
  Imported folders retain independent records even when source model IDs match.

## Verification (2026-10-08)

- Metadata tests: growing/shrinking metadata preserves payloads and unknown
  computational fields; stale/invalid writes preserve the original file.
- Download fixtures: interrupted transfer/resume, cache reuse, checksum
  failure, cancellation, invalid range and archive traversal.
- Real local cat/maobailing PTH + index -> complete VFRM: **7.33 seconds** on
  first acoustic extraction. Runtime summary sync and active-session guard
  passed.
- Real Hugging Face `maotianchen/maobailing_rvc`, pinned revision
  `e56357e07e6788617318733dce1e10eb92889633`: **55,225,035 bytes** checkpoint
  and **92,566,459 bytes** index downloaded, SHA-256 checked and converted to
  complete VFRM in **14.86 seconds**. Reused local canonical acoustic weights.
- Native HF and ModelScope repository traversal verified. HF Mirror returned
  HTTP 403 in this environment; errors are visible and source can be switched.
- Playwright: metadata save, drafts retained across snapshots, matching index
  suggestion, task progress, duplicate start prevention, cancel, stale event
  rejection, completion refresh, active-session locks, provider errors and
  1380/1080/860 px layouts. Reports/screenshots are untracked under
  `target/windows-ui-probe/model-management-*`.
- Workspace tests: **213 passed / 13 ignored**; frontend **30 passed**; Tauri
  with all backend features **26 passed / 1 ignored**. Workspace all-features
  check, frontend build and runtime Clippy (`--no-deps`) passed.
- Embedded-frontend Windows debug application built with CPU, DirectML and
  TensorRT at
  `C:/Users/taizun/AppData/Local/vocal-fast/build-model-library-tauri/debug/vocal-fast-tauri.exe`.
  The directory includes the backend runtime DLLs. This is a development build,
  not a new installer; the existing running application was preserved.

Reproduce using `crates/vocal-runtime/examples/model-library-smoke.rs`:
`cargo run -p vocal-runtime --example model-library-smoke -- list hf owner/model`
or pass a JSON `ModelImportRequest` path to exercise the complete import path.
