# GitHub release workflow

The repository publishes Windows installers from two workflows:

1. `release-dispatch.yml` is started with `workflow_dispatch`. It validates the
   synchronized version declarations, computes an exact version or a major/minor/patch
   bump, updates the lockfiles, commits the release version on `main`, and
   creates the matching annotated `vX.Y.Z` tag.
2. `release.yml` listens for `v*.*.*` tags. It builds `Cpu` and `DirectMl`
   installers on `windows-latest`, uploads the installer and build manifest,
   creates `SHA256SUMS.txt`, and publishes the GitHub Release. TensorRT-RTX is
   an optional package built on a self-hosted Windows runner.

`release.yml` also accepts `workflow_dispatch` with `publish=false`, which runs
both Windows packaging jobs without publishing a release. This is the smoke
test path for changes to the packaging workflow.

The dispatch workflow explicitly starts `release.yml` after creating the tag.
This is required because GitHub does not start a new workflow from events
created with the default `GITHUB_TOKEN`.

TensorRT-RTX is intentionally excluded from hosted runners. Its SDK and CUDA
redistributables are not distributed by this repository. The optional package
requires a self-hosted runner labeled `self-hosted`, `windows`, `x64`, and
`trt-rtx`, with `TENSORRT_RTX_ROOT` (or one of its documented aliases) and
`CUDA_PATH` configured. The build copies the SDK's `doc/Acknowledgements.txt`,
`doc/README.txt`, `EULA.txt`, `LICENSE`, and `README` into the installer under
`licenses/tensorrt-rtx/` and `licenses/cuda/`. Enable this only when the
applicable NVIDIA SDK and CUDA licenses permit that redistribution.

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
the self-hosted runner enabled:

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
  --field include_tensorrt=true
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
