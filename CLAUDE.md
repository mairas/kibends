# kibends

`kibends smooth` replaces each chain of track segments selected in KiCad's PCB editor (several chains per run, on any copper layers) with a short, smooth path of straight segments that keeps clearance to other copper and to the other selected chains. It talks to a running KiCad 10 through the IPC API (kicad-python, imported as `kipy`). The README is the user documentation.

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

`run.run_smooth` runs the stages in this order without printing. Before the DRC check and the commit, either of which may raise, it hands an `OptimizerReport` (per-chain statistics, frozen vertices, abnormal bursts, unchanged chains, each movable via's start and end, termination) to its `on_optimized` callback; at the end it returns a `RunReport`, which adds the DRC verdict, whether it committed and any refusal. `cli.smooth` prints the statistics and warnings from the callback and the DRC verdict from the returned report. A GUI is meant to call the same function.

1. `snapshot.read_selection` and `BoardSnapshot.read` read the selected tracks and vias and the board once per run, for the copper layers of the selection, or for every enabled copper layer when a selected via may move (`vias.may_move`: unlocked and through); `BoardSnapshot.slice` gives one `LayerSnapshot` per layer without further requests. Everything they return is free of kipy.
2. `chain.split_locked` and `chain.split_chains` (together `extract_chains`) split the selection into chains, one per connected piece of one layer, net and width, and pin each chain's ends and every point where same-net copper (pads, vias, tracks, copper shapes) on its layer touches it. A route through a via is one chain per layer, each pinned at the via. A via is copper and a hole only on the layers its drill spans. The copper of the selected vias that may move does not pin, so that it does not tether the chains it carries.
   `vias.movable_vias` then judges the selected vias: an unlocked through via is movable when only selected chain ends touch it, on any layer, at least two do, and no other copper touches those ends; `BoardSnapshot.read` finds that copper, zone fills included and teardrops left out, since KiCad reapplies them on the next refill, from the end discs `vias.carried_copper` gives for chains extracted without pins. The others get a warning, and if any of them could have moved, the chains are extracted again with its copper pinning, so a fixed via pins as an unselected one does. Pins never change a chain's order or ends, so the via's chain ends stay valid.
3. `obstacles.build_obstacles` turns other-net copper, holes, keepouts and the board edge into obstacles inflated by the required clearance: one static field per (layer, net, width), from that layer's snapshot, with every selected track and every moving via removed. `obstacles.build_via_obstacles` gives each moving via a field over every layer it spans.
4. `optimize.optimize_chains` minimises length + stiffness² × bending + repulsion with L-BFGS-B, in alternating bursts, one chain at a time. Each burst sees its chain's static field plus the other chains on the layer at their current shapes (`obstacles.CompositeField`, `ChainField`). Free vertices move only inside their bubble, a box in the free space around them, and an edge check rejects edges that newly cross an obstacle. Together these keep every chain on the same side of every obstacle and of every other chain. Each moving via gets a burst after the chains, moving with the chains it joins; the others see it where it currently is.
5. `simplify.simplify_chains` merges each optimised polyline back into few straight segments within `--chord-tolerance`, re-checking each merged edge against the static field, the other chains and the moved vias at their final positions.
6. A chain whose result equals its selected tracks is unchanged: it is left out of the DRC check and the commit, and if every chain is unchanged nothing is committed.
7. `drc.check` runs `kicad-cli pcb drc` once on the board text without its teardrop zones, which KiCad reapplies on the next refill, and once with every changed chain replaced and every moved via's block rewritten at its new position under a new uuid, and the run is refused if violations rise. A moved via counts like a chain, so violations it already had are not new. `--no-drc` skips it.
8. `commit.commit_chains` re-reads the changed chains' tracks and the moved vias in one request and refuses if any changed, then deletes the old tracks and vias and creates the new tracks and a moved copy of each via in one KiCad commit.

## Invariants

- One run is one undo step in KiCad. kibends never refills zones; the user refills them or lets KiCad's DRC do it.
- Locked tracks, other nets' copper, unselected or fixed vias and pads are never modified.
- Pinned points stay where they are, except chain ends at a moving via, which keep their offset from its centre. KiCad connects two tracks when their copper overlaps, not only when an end lies on the other's centreline, so pinning uses both tracks' half widths.
- Any failure before the push, in any chain, drops the whole commit. A connection loss during the push is reported as "may have landed", never as success.
- Arc tracks are refused; output is always straight tracks.

## Units and coordinates

Segment endpoints and widths from KiCad are integer nanometres, so endpoint matching is exact. Everything after the snapshot is millimetre floats and shapely geometry. All coordinates are in KiCad's board frame, with y pointing down.

## KiCad API limits

- kicad-python 0.8.0 cannot read board design rules: no minimum clearance, no custom rules, no edge or hole clearance. The optimizer uses netclass clearances plus `--edge-clearance` and `--hole-clearance`, and the kicad-cli DRC gate catches everything else. Recheck newer kicad-python releases for a rules getter.
- An intermittent "KiCad is busy" error (`AS_BUSY`) means the user has a tool or dialog open in the PCB editor. It is not a code regression. Every IPC failure ends the run with one log line through `cli._exit_on_failure`, except a copper-text render KiCad refuses for another reason, which only warns and leaves the text to the DRC check.
- The DRC gate edits the board as text and depends on KiCad 10's file format (`drc.inject`, `drc.remove_segments`, `drc.move_vias`, `drc.remove_teardrops`).

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
