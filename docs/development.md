# Development

Development needs [uv](https://docs.astral.sh/uv/) and a Rust toolchain
with `clippy` and `rustfmt`. The repository pins the Rust version in
`rust-toolchain.toml`; [rustup](https://rustup.rs) installs it on first
use.

```sh
git clone https://github.com/mairas/kibends
cd kibends
./run sync       # dependencies, both extras, and the Rust core
./run test       # Rust tests and the fast Python tests
./run check      # lint, type checks and every test, as CI does
```

Run `./run sync` in every fresh checkout or worktree before testing. Run
`./run check` before committing.

## Commands

| Command | What it does |
|---|---|
| `./run sync` | Create or update the virtualenv with the dev dependencies and the `plot` and `gui` extras, and build the Rust extension `kibends._core`. |
| `./run test` | Run the Rust core's tests and the fast Python tests, rebuilding first. Extra arguments go to pytest. |
| `./run test-all` | Run every test, including the slow ones. |
| `./run lint` | Check linting and formatting with ruff, rustfmt and clippy. |
| `./run format` | Format the code and apply ruff's safe lint fixes. |
| `./run stubs` | Regenerate the type stubs of `kibends._core` from the Rust sources. |
| `./run typecheck` | Check that the stubs are current, and type-check with pyright in strict mode. |
| `./run check` | Run lint, typecheck and every test, as CI does. |
| `./run bench` | Measure the optimizer on benchmark boards (see `scripts/benchmark.py`). |
| `./run bumpversion patch\|minor\|major` | Bump the version in `VERSION`, `pyproject.toml` and `Cargo.toml`, and refresh both lockfiles. Needs [bump2version](https://github.com/c4urself/bump2version) (`uv tool install bump2version`). |
| `./run help` | List the commands. |

Tests never need a running KiCad: `tests/conftest.py` builds fake boards.

## Releasing

Every merge to main stages a release: a pre-release tagged
`v<VERSION>+<N>_pre` and a draft `v<VERSION>+<N>`, where N counts the
builds of that version. Publishing the draft on GitHub builds the wheels
and the sdist and uploads them to PyPI as version `<VERSION>`. PyPI
takes each version once, so a release of a version already there uploads
nothing.

The first pull request that changes the package after a release bumps
the version with `./run bumpversion`; the PR checks fail until it does.
Later pull requests before the next release leave the version alone.

## Design

The design reference is in the code: the Python module docstrings under
`src/kibends` and the Rust module docs in `rust/core`. They describe how
the optimizer works, which the user documentation leaves out.
