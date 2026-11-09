# kibends

`kibends smooth` replaces each chain of track segments selected in KiCad's PCB editor (several chains per run, on any copper layers) with a short, smooth path of straight segments that keeps clearance to other copper and to the other selected chains. It talks to a running KiCad 10 through the IPC API (kicad-python, imported as `kipy`). The README is the user documentation.

The repository is public. Never write host names, addresses or other details of the author's machines into files, commits, issues or PRs.

## Commands

- `./run sync` installs dependencies, including the `plot` extra that `TestPlot` needs. Run it in every fresh worktree before testing.
- `./run test` runs the fast tier; `./run test-all` runs everything, including `slow` tests.
- `./run lint` runs `ruff check` and `ruff format --check`; `./run format` formats and applies ruff's fixes.
- `./run typecheck` runs pyright in strict mode over `src`, `tests` and `scripts`. Keep code fully annotated and never use `Any`.
- `./run check` runs lint, typecheck and `test-all`, as CI does. Run it before committing.
- `./run help` lists the commands. `run` follows the halos convention: one bash function per command, documented with `#@` docstrings.

Ruff and pyright are configured in `pyproject.toml`. CI (`.github/workflows/pr.yml`) calls the halos-org shared checks workflow, which runs `.github/actions/run-tests`.

## Where design detail lives

Module docstrings are the design reference; function and class docstrings hold the details. When behaviour changes, update the docstrings it touches, not this file. This file changes only when an instruction, command or invariant changes, so a work brief should not ask for a CLAUDE.md edit otherwise.

## Pipeline

`run.run_smooth` runs the stages in order and prints nothing; `cli.smooth` presents its reports. A GUI is meant to call the same function.

1. Read: `snapshot` reads the selection and the board once per run; nothing after it uses kipy until the commit.
2. Extract: `chain` splits the selection into chains and pins them (contacts, junctions, stubs joining tracks); `vias` judges which selected vias move, stitching vias and zone containment included.
3. Obstacles: `obstacles` builds each chain's and each moving via's static field, and the via's limits.
4. Optimize: `optimize` minimises length plus repulsion with L-BFGS-B bursts (`bursts`), one problem per layer or per group of layers that moving vias tie together, and returns `results`. `members`, `neighbours`, `seen` and `via_moves` give what the members see of each other and their energy; `polylines` measures, resamples and checks polylines; `contacts` keeps vertices on their contacts; `pairs`, `pair_layout` and `pair_blocks` handle differential pairs.
5. Simplify: `simplify` merges each result into few straight segments, rechecking clearance.
6. Unchanged chains are left out of the commit (`run`).
7. Commit: `commit` re-reads what it replaces, refuses if it changed, and replaces tracks and vias in one KiCad commit.

## Invariants

- One run is one undo step in KiCad.
- kibends never refills zones and never runs KiCad's DRC. The user refills the zones, or lets the DRC do it, and runs the DRC for everything the optimizer does not model (custom rules, board minimums, thermal reliefs, island removal, hatched fills, minimum-width necks).
- A moved via stays where the zones of its net that reached its ring connect it, taken together, on each layer where one did, and keeps the zone's clearance and margin from other-net copper there, so that the refill reconnects it.
- Locked tracks, other nets' copper, unselected or fixed vias and pads are never modified.
- Pinned points stay where they are, except chain ends at a moving via, which keep their offset from its centre.
- A vertex that keeps a pad or a staying via may move but stays inside it, shrunk by `CONTACT_MARGIN`; the chain keeps touching every pad and via it touched.
- The loose ends of a junction's branches move together and stay one point.
- Pinning uses both tracks' half widths: KiCad connects two tracks when their copper overlaps, not only when an end lies on the other's centreline.
- Any failure before the push, in any chain, drops the whole commit, and an interrupt ends the run with "nothing committed". A connection loss or an interrupt during the push is reported as "may have landed", never as success.
- Arc tracks are refused; output is always straight tracks.

## Units and coordinates

Segment endpoints and widths from KiCad are integer nanometres, so endpoint matching is exact. Everything after the snapshot is millimetre floats and shapely geometry, in KiCad's board frame with y pointing down. Printed positions are relative to the board's grid origin instead: every printed point goes through the run's `coords.UserFrame`. Reports keep positions in the board frame and carry the frame; their descriptions are text already formatted in it.

## KiCad API limits

- kicad-python 0.8.0 cannot read board design rules: no minimum clearance, no custom rules, no edge or hole clearance. The optimizer uses netclass clearances plus `--edge-clearance` and `--hole-clearance`; the user's KiCad DRC checks everything else. Recheck newer kicad-python releases for a rules getter.
- An intermittent "KiCad is busy" error (`AS_BUSY`) means the user has a tool or dialog open in the PCB editor. It is not a code regression.
- Every IPC failure ends the run with one log line through `cli._exit_on_failure`, except a copper-text render KiCad refuses for another reason, which only warns and leaves the text to KiCad's DRC, and the selection after a push (`commit._select`), which is logged at debug since the commit has landed.

## Provenance

Parts of `snapshot`, `commit` and `cli` are copied and trimmed from kistitch (https://github.com/hatlabs/kistitch, MIT, Copyright (c) 2025 Hat Labs Oy). Each such module says so in its docstring; keep those notices.

## Tests

Tests never need a running KiCad: `tests/conftest.py` provides kipy object builders and fake boards, and `tests/scenarios.py` holds optimizer scenarios, which `scripts/plot_scenarios.py` renders for tuning.

The optimizer's floating-point result differs between platforms (BLAS, CPU), so a test compares what passes through it within a named tolerance that says what it stands for (`NM_ROUNDING`, `REPORTED`, `STRAIGHT_TOLERANCE`), or by a property simplification makes exact, such as a straight route merging into one segment. It compares exactly only what the optimizer does not produce (pins, frozen and unchanged copper, ids, widths, nets, commit inputs) or two results of the same machine.

### Test speed

A slow suite slows every edit-test cycle for the person running it. Treat test run time as a requirement, not a detail.

- `./run test` (the fast tier) finishes in under 10 s wall-clock.
- No test in the fast tier takes more than 1 s on its own, setup included. Measure with `.venv/bin/python -m pytest -n0 -m "not slow" --durations=10`; the parallel run hides slow tests.
- Make a slow test faster before marking it `slow`: coarser optimizer settings, a smaller scenario, a fake instead of the real thing. Mark it `slow` only when it needs the real thing.
- Every change that adds or modifies tests checks the durations above and reports the fast-tier time.
