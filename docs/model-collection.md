# RVC conversion and ModelScope collection

The public collection is [MosRat/vfrm-models](https://modelscope.cn/models/MosRat/vfrm-models).
It contains 25 real RVC v2 voices from Hugging Face and ModelScope, shared
HuBERT/RMVPE VFRMs, and a complete Mandarin voice bundle. All three execution
backends consume the same canonical tensor names and layouts.

## Rust conversion CLI

Build the converter without GPU features, CUDA, TensorRT, Python or oneDNN:

```powershell
cargo build -p vocal-fast --release
$cli = 'target/release/vocal-fast.exe'
& $cli model convert voice.pth voice.vfrm --name 'My voice' --id my-voice
& $cli model convert voice.pth voice-with-index.vfrm --index added.index
& $cli model convert voice.onnx voice.vfrm --manifest manifest.json
& $cli model describe voice.vfrm
& $cli model verify voice.vfrm
& $cli model bundle voice.vfrm hubert.vfrm rmvpe.vfrm complete.vfrm
```

`convert` accepts RVC exported Synth checkpoints, not arbitrary PyTorch/ONNX
models or training G/D checkpoints. PTH import uses the bounded Rust pickle
reader without executing Python. The phone projection, checkpoint configuration,
sample rate and f0 flag determine RVC v1/v2 and f0/non-f0 architecture. All
parameters are normalized and validated before output. ONNX import requires
an explicit manifest: sample rate is not inferred from an ambiguous waveform
output. The existing strict RVC ONNX operator and interface checks still apply.

`--index` embeds FAISS Flat-L2/IVF-Flat-L2 data, validates its structure and
checks 256/768 feature width against the model. Unsupported index families
are rejected. A trained-only empty index does not replace the populated index
needed for retrieval.

Optional `--source URL`, `--provenance source.json` and `--library-info info.json`
attach public source and display metadata. Both JSON files must be objects
of at most 1 MiB. Library info uses `author`, `language`, `license`, `tags`.
Never include credentials or signed download URLs in these files.

Existing conversion outputs are rejected. Writes use a same-directory temporary
VFRM, validate it, flush it, and publish with an atomic create-if-absent hard
link. This requires a filesystem supporting hard links, such as NTFS or ext4.
An unsupported filesystem returns an I/O error without replacing another file.
The older `convert-pth`/`convert-onnx` command signatures remain available.

`describe` inspects only the bounded directory. `verify` also loads and validates
tensor payload structure and embedded retrieval; its memory usage includes the
loaded weights. Neither command proves acoustic parity or benchmark performance.

## Reproducible collection workflow

Python orchestrates hub APIs and the Rust executable. Synth conversion itself
remains Rust. Acoustic preparation uses the existing offline exporter/converter;
runtime inference does not require ONNX or Python.

```powershell
uv sync
uv pip install --python .venv/Scripts/python.exe `
  -r scripts/model_collection/requirements.txt
$py = '.venv/Scripts/python.exe'
$cli = 'target/release/vocal-fast.exe'
```

1. Search and integrate repository candidates from both hubs. The unified
   search inspects each result and marks exported checkpoints, FAISS indexes,
   ONNX files, missing licenses and shape-validation warnings. It does not
   download model weights:

   ```powershell
   & $py scripts/model-collection.py search hf RVC --output target/hf-rvc.json
   & $py scripts/model-collection.py search ms RVC --output target/ms-rvc.json
   & $py scripts/model-collection.py search all RVC --limit 20 `
     --output target/rvc-candidates.json
   & $py scripts/model-collection.py merge-search target/hf-rvc.json target/ms-rvc.json `
     --output target/rvc-candidates-merged.json
   & $py scripts/model-collection.py discover hf maotianchen/maobailing_rvc `
     --output target/maobailing-candidates.json
   & $py scripts/model-collection.py discover ms NaClNeZn/RVC-Model `
     --output target/ms-rvc-candidates.json
   ```

   The `all` form is the preferred cross-hub scan. `merge-search` is useful
   when separate scans were run with different queries. Candidates are keyed
   by `(provider, repository)`; same names on HF and ModelScope remain separate
   sources because their bytes and licenses are independent. Review the
   `warnings` field and select exact files in `selection.json` before locking.

2. Review model cards/license restrictions and select exact checkpoint/index
   paths in `scripts/model_collection/selection.json`. Pin the selection:

   ```powershell
   & $py scripts/model-collection.py lock `
     --selection scripts/model_collection/selection.json `
     --output target/model-collection/sources.lock.json
   ```

   The checked-in `scripts/model_collection/sources.lock.json` is the first
   published selection. HF revisions are commit IDs. ModelScope currently
   exposes branch revisions for these sources; SHA-256 and byte sizes pin
   the actual content. A changed branch file fails verification.

3. Build from the checked-in lock, optionally reusing raw files under `models/`
   only when their complete size and SHA-256 match:

   ```powershell
   & $py scripts/model-collection.py build `
     --lock scripts/model_collection/sources.lock.json `
     --output target/model-collection/distribution `
     --cli $cli --seed-root models --bundle maobailing
   ```

   The first output directory must be new. Raw downloads and exporter intermediates
   stay under `target/model-collection/cache/`; source notices, VFRMs and catalog
   go into the distribution directory. Only requested files are downloaded,
   never a multi-gigabyte repository snapshot. The generated `catalog.json` is
   schema v2 and records a source-lock digest, source fingerprint, artifact
   hashes and bundle dependency hashes.

   To maintain an existing directory after changing `sources.lock.json`, use
   the same command with `--update`:

   ```powershell
   & $py scripts/model-collection.py build `
     --lock scripts/model_collection/sources.lock.json `
     --output target/model-collection/distribution `
     --cli $cli --seed-root models --bundle maobailing --update
   ```

   A voice or shared acoustic VFRM is reused when its locked source revision,
   complete source fingerprint, size and SHA-256 still match. New or changed
   inputs alone are fetched and converted. Add `--prune` when obsolete generated
   voice, shared or bundle files should be removed; provenance files are kept
   for audit history unless removed explicitly.

   An older checked-out distribution can be upgraded without touching its large
   VFRM files:

   ```powershell
   & $py scripts/model-collection.py migrate `
     --lock scripts/model_collection/sources.lock.json `
     --output target/model-collection/distribution-large
   ```

   Migration validates the existing inventory, adds source fingerprints and
   bundle dependency hashes, rewrites the README, and regenerates checksums.

   HuBERT is exported from the locked HF checkpoint/config before acoustic
   conversion. RMVPE uses the locked **waveform-input** ONNX; the more common
   mel-input ONNX is not interchangeable. For reuse of the previously verified
   local acoustic VFRMs, add `--hubert-seed PATH --rmvpe-seed PATH`; their hashes
   must match `seed_vfrm_sha256` in the lock. ONNX exporter versions can alter
   graph metadata and binary file hashes while preserving canonical weights.

4. Verify, authenticate using the ModelScope SDK's local credential cache or
   `MODELSCOPE_API_TOKEN`, and publish:

   ```powershell
   & $py scripts/model-collection.py verify `
     --output target/model-collection/distribution --cli $cli
   & $py scripts/model-collection.py publish `
     --output target/model-collection/distribution `
     --repo MosRat/vfrm-models --cli $cli
   ```

   Publish requires the destination to match the lock. It checks an exact file
   inventory and token patterns in text files, validates catalog v2 against the
   inventory, uploads an explicit allow-list, and reports local-only, remote-only
   and changed files before and after upload. Remote files are never deleted by
   default. Use `--prune` only for an intentional exact mirror:

   ```powershell
   & $py scripts/model-collection.py publish `
     --output target/model-collection/distribution `
     --repo MosRat/vfrm-models --cli $cli `
     --report target/model-collection/publish-report.json --roundtrip sample
   & $py scripts/model-collection.py publish `
     --output target/model-collection/distribution `
     --repo MosRat/vfrm-models --cli $cli --prune --roundtrip none
   ```

   `--roundtrip sample` validates both shared VFRMs, one voice and every
   complete bundle through the public download API and Rust parser. `all`
   validates every voice too; `none` only checks remote metadata. SDK credentials
   stay outside source and distribution trees.

## Using the published files in the GUI

The collection's `bundles/` directory is the simplest GUI path. Download a
complete bundle, place it in the GUI model library, select it in Studio, choose
Automatic, CPU, DirectML or TensorRT-RTX, and click **Load**. The GUI detects
the embedded Synth, HuBERT, RMVPE and optional retrieval index. For DirectML
and TensorRT-RTX it extracts the acoustic VFRMs into its local cache and builds
or reuses the backend graph. PTH, ONNX and `pipeline.json` are not needed.

The smaller `voices/` files contain Synth and optional retrieval only. The GUI
now resolves the two shared acoustic dependencies while loading: it first
checks the per-user cache, then asks before downloading `shared/hubert.vfrm`
and `shared/rmvpe.vfrm` from the public ModelScope registry. The files are
SHA-256 verified and bound into a derived complete VFRM under
`%LOCALAPPDATA%/vocal-fast/model-library/bound`; the original voice file is
never modified. Choosing local acoustic files remains available from the same
setup card. This means CPU, DirectML and TensorRT load the same resolved model
without backend-specific source fields. The GUI component badges make the
distinction visible: a voice-only file reports HuBERT/RMVPE as missing, while
a complete bundle reports all required components as embedded. The optional
retrieval index is used automatically
when it is present in the selected voice.

The runtime keeps the public collection manifest under
`%LOCALAPPDATA%/vocal-fast/model-library/catalog-cache/` for six hours. Shared
dependency resolution reads `catalog.json` first and falls back to a repository
listing when the manifest is unavailable or uses an older schema. Downloaded
payloads remain content verified and are keyed by immutable repository/file
identity, so refreshing the manifest never invalidates a valid local file.

## Distribution layout and licensing

```text
voices/                 Synth + optional FAISS; shared acoustics not duplicated
shared/                 hubert.vfrm, rmvpe.vfrm
bundles/                complete voice + retrieval + embedded acoustics
provenance/<source>/     original cards/licenses and source.json
catalog.json            v2 manifests, source fingerprints and artifact hashes
sources.lock.json       upstream revisions, file hashes and conversion recipes
inventory.json          complete distribution file sizes/hashes
SHA256SUMS              readable checksum list
configuration.json      ModelScope format/task declaration
README.md               model card and usage
```

Licenses are per source, not inherited from vocal-fast-rs. Initial voices declare
Apache-2.0 or MIT upstream; the selected waveform RMVPE declares GPL-3.0.
The complete bundle contains these differently licensed components. Collection
metadata therefore uses `other`, preserves upstream declarations/cards and does
not label all weights MIT. A hub license label is recorded as an upstream
declaration, not a new grant by this project.

## Current validation (2026-10-11)

| Voice | Provider | Version | Sample rate | Embedded retrieval |
|---|---|---|---:|---|
| maobailing | HF | v2 | 40000 | yes |
| malay-male-1 | HF | v2 | 40000 | no |
| malay-female-1 | HF | v2 | 40000 | no |
| default-v2 | HF | v2 | 40000 | yes |
| Star Rail 21 voices | ModelScope | v2 | 48000 | no |

All 25 conversions passed canonical Synth and index validation. The two
shared acoustic files and complete bundle passed Rust loading. A separate
rebuild from original acoustic sources produced identical names, shapes,
dtypes and payload hashes for all 229 HuBERT and 310 RMVPE tensors. The large
upload committed 48 distribution files (28 LFS VFRMs), checked every remote
size/hash, and round-tripped maobailing through the download API and Rust
parser. Reports remain untracked under `target/model-collection/`.

Checks: `cargo fmt --all -- --check`, `cargo test -p vocal-fast`,
`cargo test -p vocal-model`, Python `py_compile`, and collection integrity
unit tests. These are conversion/integrity checks, not new inference parity
or audio quality measurements.
