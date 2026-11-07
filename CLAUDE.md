# kibends

`kibends smooth` replaces the chain of track segments selected in KiCad's PCB editor with a short, smooth path of straight segments that keeps clearance to other copper. It talks to a running KiCad 10 through the IPC API (kicad-python, imported as `kipy`). The README is the user documentation.

The repository is public. Never write host names, addresses or other details of the author's machines into files, commits, issues or PRs.

## Commands

- `./run sync` installs dependencies, including the `plot` extra that `TestPlot` needs. Run it in every fresh worktree before testing.
- `./run test` runs the fast tier; `./run test-all` runs everything, including `slow` and real kicad-cli tests.
- `./run lint` runs `ruff check` and `ruff format --check`; `./run format` formats and applies ruff's fixes.
- `./run typecheck` runs pyright in strict mode over `src`, `tests` and `scripts`. Keep code fully annotated and never use `Any`.
- `./run check` runs lint, typecheck and `test-all`, as CI does. Run it before committing. CI has no kicad-cli, so the real-KiCad DRC tests run only locally: a change to `drc.py` needs a local `./run check` with KiCad installed.
- `./run help` lists the commands. `run` follows the halos convention: one bash function per command, documented with `#@` docstrings.

Ruff and pyright are configured in `pyproject.toml`. CI (`.github/workflows/pr.yml`) calls the halos-org shared checks workflow, which runs `.github/actions/run-tests`; kicad-cli tests skip there.

## Pipeline

`cli.smooth` runs the stages in this order:

1. `snapshot.read_selection` and `LayerSnapshot.from_board` read the selection and the chain's copper layer. Everything they return is free of kipy.
2. `chain.extract_chain` orders the selected segments into one chain and pins its ends and every point where same-net copper (pads, vias, tracks, copper shapes) touches it.
3. `obstacles.build_obstacles` turns other-net copper, holes, keepouts and the board edge into obstacles inflated by the required clearance.
4. `optimize.optimize` minimises length + stiffness² × bending + repulsion with L-BFGS-B. Free vertices move only inside their bubble, a box in the free space around them, and an edge check rejects edges that newly cross an obstacle. Together these keep the chain on the same side of every obstacle.
5. `simplify.simplify` merges the optimised polyline back into few straight segments within `--chord-tolerance`, re-checking each merged edge.
6. `drc` runs `kicad-cli pcb drc` on the board text before and after the replacement and refuses the commit if violations rise. `--no-drc` skips it.
7. `commit.commit_chain` deletes the old tracks and creates the new ones in one KiCad commit.

## Invariants

- One run is one undo step in KiCad. kibends never refills zones; the user refills them or lets KiCad's DRC do it.
- Locked tracks, other nets' copper, vias and pads are never modified.
- Pinned points stay where they are. KiCad connects two tracks when their copper overlaps, not only when an end lies on the other's centreline, so pinning uses both tracks' half widths.
- Any failure before the push drops the commit. A connection loss during the push is reported as "may have landed", never as success.
- Arc tracks are refused; output is always straight tracks.

## Units and coordinates

Segment endpoints and widths from KiCad are integer nanometres, so endpoint matching is exact. Everything after the snapshot is millimetre floats and shapely geometry. All coordinates are in KiCad's board frame, with y pointing down.

## KiCad API limits

- kicad-python 0.8.0 cannot read board design rules: no minimum clearance, no custom rules, no edge or hole clearance. The optimizer uses netclass clearances plus `--edge-clearance` and `--hole-clearance`, and the kicad-cli DRC gate catches everything else. Recheck newer kicad-python releases for a rules getter.
- An intermittent "KiCad is busy" error (`AS_BUSY`) means the user has a tool or dialog open in the PCB editor. It is not a code regression. Every IPC failure ends the run with one log line through `cli._exit_on_failure`, except a copper-text render KiCad refuses for another reason, which only warns and leaves the text to the DRC check.
- The DRC gate edits the board as text and depends on KiCad 10's file format (`drc.inject`, `drc.remove_segments`).

## Provenance

Parts of `snapshot`, `commit`, `drc` and `cli` are copied and trimmed from kistitch (https://github.com/hatlabs/kistitch, MIT, Copyright (c) 2025 Hat Labs Oy). Each such module says so in its docstring; keep those notices.

## Tests

Tests never need a running KiCad. `tests/conftest.py` provides kipy object builders and fake boards (`FakeBoard`, `CommitBoard`). `tests/scenarios.py` holds optimizer scenarios, which `scripts/plot_scenarios.py` also renders for tuning. Tests marked `kicad_cli` run the real kicad-cli on `tests/fixtures/gate` and skip when it is not installed.

### Test speed

A slow suite slows every edit-test cycle for the person running it. Treat test run time as a requirement, not a detail.

- `./run test` (the fast tier) finishes in under 10 s wall-clock.
- No test in the fast tier takes more than 1 s on its own, setup included. Measure with `.venv/bin/python -m pytest -n0 -m "not slow" --durations=10`; the parallel run hides slow tests.
- Make a slow test faster before marking it `slow`: coarser optimizer settings, a smaller scenario, a fake instead of the real kicad-cli. Mark it `slow` only when it needs the real thing, such as the kicad-cli tests.
- Every change that adds or modifies tests checks the durations above and reports the fast-tier time.
