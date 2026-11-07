# kibends

Smooths KiCad tracks. `kibends smooth` takes the chain of track segments
selected in the PCB editor and replaces it with a short, smooth path of
plain straight segments that keeps clearance to other copper. The result is
not made of KiCad native arcs, so the tracks stay editable.

The new path minimises a weighted sum of three goals:

- **length**: in free space, two endpoints give a straight track;
- **bending**: sharp corners cost more than gentle ones;
- **closeness to other copper**: the track drifts toward the middle of the
  free space around it.

The original route is only the starting point. Bends remain where other
copper or the board edge forces them, and the track stays on the same side
of every obstacle it started next to.

Uses the KiCad IPC API through
[kicad-python](https://pypi.org/project/kicad-python/).

Status: early development. One chain per run.

## Requirements

- KiCad 10 or later, with the API server enabled
  (Preferences > Plugins > Enable KiCad API).
- Python 3.11 or later.
- `kicad-cli`, which comes with KiCad, for the DRC check.

## Installation

```sh
uv tool install git+https://github.com/mairas/kibends
```

With `--plot` support, which adds matplotlib:

```sh
uv tool install 'kibends[plot] @ git+https://github.com/mairas/kibends'
```

## Usage

1. Open the board in KiCad's PCB editor and save it: the DRC check reads
   the project and rule files from disk.
2. Select the track segments of one chain: connected end to end, on one
   layer, one net and one width, without branches. Selected vias, pads and
   other items are ignored; arc tracks are refused.
3. Run:

```sh
kibends smooth --dry-run --plot chain.png   # look first
kibends smooth                              # then commit
```

The command prints the length, track count, tightest bend radius and
clearance margins before and after, and the optimizer's outcome. Without
`--dry-run`, and if the DRC check passes, the selected tracks are replaced
in one commit, which one undo
in KiCad reverts. kibends does not refill zones: refill them yourself (B)
or let KiCad's DRC refill them.

Locked tracks in the selection are left untouched. The chain's two ends
stay where they are, as does every point where unselected copper of the same
net (a pad, via, track or copper shape) touches the chain.

### Options

| Option | Default | Effect |
|---|---|---|
| `--stiffness MM` | 0.3 | Weight of bending, as a length. Larger values give wider, rounder bends at the cost of length. Obstacles can still force tighter bends. |
| `--repulsion WEIGHT` | 0.5 | Weight of keeping away from other copper, relative to length. 0 disables it, so the track pulls tight against obstacles at exactly the clearance. |
| `--range MM` | 1.0 | Distance beyond the required clearance at which repulsion fades to zero. Copper further away has no effect. |
| `--edge-clearance MM` | 0.3 | Clearance to the board edge. |
| `--hole-clearance MM` | 0.25 | Clearance to drilled holes. |
| `--chord-tolerance MM` | 0.025 | How far the committed tracks may depart from the optimised curve; larger values give fewer, longer tracks. |
| `--dry-run`, `-n` | | Print the statistics and the DRC verdict without changing the board. |
| `--no-drc` | | Commit without the DRC check. |
| `--kicad-cli PATH` | | The kicad-cli to run the DRC check with. By default kibends uses the one the running KiCad reports, then `kicad-cli` on `PATH`, then the usual install locations. |
| `--plot FILE` | | Draw the original and smoothed chain over the area its centreline must avoid; the format follows the extension. Works with or without `--dry-run`. |

The defaults give a modest rounding: in the test geometries, a 90° corner
wrapped around a pad gets a bend radius of roughly 0.7 to 1.4 mm. Bending
enters with stiffness squared, so the bends widen quickly; try steps such as
0.5 or 1.0 before larger values. A longer `--range` spreads tracks further
from their neighbours where space allows.

### Clearances

Copper clearances come from the board's netclasses, including pad and
footprint clearance overrides. KiCad's API does not expose the board's edge
and hole clearances, custom rules or board minimums: the edge and hole
clearances come from the options above, and custom rules and board
minimums are not seen by the optimizer.

The DRC check covers what the optimizer cannot see. Before committing,
kibends runs `kicad-cli pcb drc` twice on copies of the board in a scratch
directory, with the project's `.kicad_pro` and `.kicad_dru`: once on the
board as it is, and once with the chain replaced. If the result adds a
violation, nothing is committed, and each new violation is listed with its
type, description and position. Violations the board already had do not
count, including ones the old chain had against the same item. A path that
splits another net's pour shows up as more unconnected items on that net and
is refused, naming the net.

Save the board before running: kicad-cli reads the board's project and rule
files from disk, so rule or netclass edits not yet saved in KiCad are not
seen. The check takes two full-board DRC runs with zone refill; the log shows
how long each took.

### Editing a smoothed track

Change the route by hand or with the router, select the chain again and
re-run. Re-running on kibends' own output with nothing else changed moves it
only slightly, but still replaces its tracks.

## Development

```sh
./run sync
./run test       # fast tests, in parallel
./run test-all   # also the slow and kicad-cli tests; run before committing
```
