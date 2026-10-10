# GitHub release workflow

The repository publishes Windows installers from two workflows:

1. `release-dispatch.yml` is started with `workflow_dispatch`. It validates the
   synchronized version declarations, computes an exact version or a major/minor/patch
   bump, updates the lockfiles, commits the release version on `main`, and
   creates the matching annotated `vX.Y.Z` tag.
2. `release.yml` listens for `v*.*.*` tags. It builds `Cpu` and `DirectMl`
   installers on `windows-latest`, uploads the installer and build manifest,
   creates `SHA256SUMS.txt`, and publishes the GitHub Release. TensorRT-RTX is
   an optional package built on GitHub-hosted runners.

`release.yml` also accepts `workflow_dispatch` with `publish=false`, which runs
both Windows packaging jobs without publishing a release. This is the smoke
test path for changes to the packaging workflow.

The dispatch workflow explicitly starts `release.yml` after creating the tag.
This is required because GitHub does not start a new workflow from events
created with the default `GITHUB_TOKEN`.

TensorRT-RTX is downloaded in an Ubuntu container job and passed to a
GitHub-hosted Windows runner as a one-day artifact. The Windows job installs
CUDA 13.4 with `scripts/setup-cuda-ci.ps1`, verifies the installer's NVIDIA
Authenticode signature, installs nvcc/compiler dependencies and cudart/cuFFT
development files, and builds the application. The SDK artifact expires after
one day. Configure these repository variables:

- `TENSORRT_RTX_DOWNLOAD_URL`: an authorized NVIDIA TensorRT-RTX archive URL.
- `TENSORRT_RTX_DOWNLOAD_SHA256`: the exact SHA-256 of that archive.

An optional `NVIDIA_API_KEY` repository secret is sent as a bearer token when
the download endpoint requires it. The manual workflows accept
`tensorrt_url` and `tensorrt_sha256` inputs, which override the repository
variables. The URL must point to a `.zip`, `.tar.gz`, `.tgz`, or `.tar.zst`
archive containing the Windows SDK layout (`bin/`, `include/`, `lib/`, and
`doc/`). The downloader verifies the hash before extraction and checks the
TensorRT runtime DLL, headers, import library, and notice files.

The build copies the TensorRT SDK's `doc/Acknowledgements.txt` and
`doc/README.txt` into `licenses/tensorrt-rtx/`. Hosted CUDA installations
bundle NVIDIA's official CUDA EULA page as `licenses/cuda/EULA.html`, with a
`README` recording the toolkit version and license source. Local builds with
the full CUDA toolkit copy its `EULA.txt`, `LICENSE`, and `README` instead.
The installer also includes oneDNN and DirectML notices for the enabled
backends. Enable redistribution only when the applicable SDK licenses permit it.

## Validate without publishing

The dispatch workflow has a dry-run input. It calculates and validates the
next version without changing the branch or creating a tag:

```powershell
gh workflow run release-dispatch.yml --ref main `
  --field bump=patch `
  --field dry_run=true
gh run watch
```

To exercise the Windows packaging matrix without creating a tag or release:

```powershell
gh workflow run release.yml --ref main --field publish=false
gh run watch
```

To include the optional TensorRT-RTX package, use the dispatch workflow with
the hosted download/build enabled. The repository variables described above
must be configured first:

```powershell
gh workflow run release-dispatch.yml --ref main `
  --field bump=patch `
  --field include_tensorrt=true `
  --field dry_run=false
gh run watch
```

For a packaging smoke test without publishing, run `release.yml` directly and
set both boolean inputs explicitly:

```powershell
gh workflow run release.yml --ref main `
  --field publish=false `
  --field include_tensorrt=true `
  --field tensorrt_url="$env:TENSORRT_RTX_DOWNLOAD_URL" `
  --field tensorrt_sha256="$env:TENSORRT_RTX_DOWNLOAD_SHA256"
gh run watch
```

## Publish a release

The default patch bump reads the synchronized version from the repository. An
exact version can be supplied instead; an optional `v` prefix is accepted:

```powershell
gh workflow run release-dispatch.yml --ref main `
  --field version=0.2.0 `
  --field dry_run=false
gh run watch
```

The workflow refuses to overwrite an existing tag. Once the tag push starts
`release.yml`, the final installer and checksum assets appear on the Release
page. The version helper can be used locally as well:

```powershell
python scripts/release-version.py read
python scripts/release-version.py bump 0.1.0 patch
python scripts/release-version.py verify --expected 0.1.0
```

All workflow actions use versioned major releases (`checkout@v4`, setup
actions `@v4`/`@v5`, artifact actions `@v4`, and `action-gh-release@v2`).
