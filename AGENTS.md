# Repository contribution rules

## Commit boundaries

Use one commit for one completed functional change in one module. Keep these
changes separate when they can be reviewed independently:

- workspace and dependency configuration
- `vocal-types` and `vocal-audio` foundations
- `vocal-model` loaders, VFRM format, and model schemas
- `vocal-engine` CPU pipeline and backend dispatch
- `vocal-tensorrt` CUDA, FFI, native graph, and TensorRT pipeline work
- `vocal-cli` and Tauri application integration
- analysis and conversion scripts
- tests, fixtures, and documentation

Do not mix generated build output, downloaded weights, or reference checkouts
with a source change. A change that crosses modules should be split when the
individual parts build and review independently; otherwise use the narrowest
scope that describes the integrated change.

## Commit message format

Every commit uses:

```text
type(scope): short imperative description
```

The subject is concise, starts with a lowercase type, and does not end with a
period. Use these types:

- `feat` for a new capability
- `fix` for a behavior or correctness repair
- `refactor` for structure-preserving code changes
- `perf` for a measured performance change
- `test` for tests or validation tooling
- `docs` for documentation
- `build` for workspace, dependency, or build-system changes
- `chore` for repository maintenance

Use a module name as the scope, for example `feat(model): add vfrm tensor
directory`, `feat(tensorrt): build native rvc graph`, or
`docs(repo): document commit boundaries`. Keep the body optional, but include
it when the reason, measured result, or compatibility impact is not obvious
from the subject.

## Verification before committing

Run the smallest relevant checks before each commit. For Rust changes, run
`rtk cargo fmt --all -- --check` and the affected package tests; before a
release or a cross-crate change run `rtk cargo check --workspace --all-features`
and `rtk cargo test --workspace`. For Python scripts, run `rtk .venv\\Scripts\\python.exe
-m py_compile` on changed files. Record important benchmark or parity results
in `docs/` or `target/` reports, with generated `target/` files remaining
untracked.

## Repository contents

Keep source, manifests, lightweight fixtures, and reproducible scripts in Git.
Do not commit `target/`, virtual environments, downloaded model weights,
TensorRT plans, or cloned reference repositories. Update `.gitignore` when a
new generated or local-only asset class is introduced.
