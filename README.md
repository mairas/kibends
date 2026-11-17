# kibends

kibends smooths the tracks of a KiCad board. It replaces each selected
route with a short, smooth path that keeps its clearance to other copper,
and it does so in the running PCB editor as one undo step.

![HALMET before smoothing](docs/images/halmet-before.png)
![HALMET after smoothing](docs/images/halmet-after.png)

Status: early development.

## Install

```sh
uv tool install 'kibends[gui] @ git+https://github.com/mairas/kibends'
```

You need:

- KiCad 10 or later, with the API server enabled (Preferences > Plugins >
  Enable KiCad API).
- Python 3.14 or later.
- A Rust toolchain, 1.90 or later, since the install builds kibends from
  source. Without `cargo` on the path, the build downloads one into its
  own cache; [rustup](https://rustup.rs) installs one to keep. An older
  `cargo` on the path, such as a distribution's package, stops the build.

The `gui` extra adds the panel and needs macOS 14 or later, or on Linux
glibc 2.34 or later (2.39 on aarch64). Without it you get only the
[command line](#command-line).

## Using it

1. Open the board in KiCad's PCB editor.
2. Start `kibends-panel`. A small window opens and stays open between runs.
3. Select the tracks to smooth, on any copper layers, with the vias that
   may move with them. Select nothing to smooth the whole board. The top
   line of the panel says what Run will touch.
4. Press Run. The panel shows the progress and a graph of how much shorter
   the tracks get. Stop ends optimizing and commits the best result so
   far.
5. Look at the result in KiCad. The report in the panel says how much
   length the run saved, how many chains and vias it changed, and any
   warnings. One Ctrl+Z in KiCad undoes the whole run.
6. Refill the zones (B) and run KiCad's DRC.

<img src="docs/images/panel.png" width="400" alt="The panel after a whole-board run on HALMET">

To change a smoothed route, edit it by hand or with the router, select it
again and run again. After a run on a selection, the new tracks and moved
vias are selected, so a second run needs no new selection.

[The panel](docs/panel.md) describes its fields and report, and
[What gets smoothed](docs/selection.md) how kibends reads a selection.

## What it changes and what it does not

- A run is one undo step in KiCad. If any part fails, nothing is
  committed.
- Locked tracks, pads, other nets' copper and the tracks of selected
  groups, such as length-tuning patterns, are never modified.
- A via moves only when it is selected, or the run covers the whole
  board, and only where kibends can tell the move is safe;
  [which vias move](docs/selection.md#vias) has the rules. A moved via
  stays inside the zones of its net that connected it.
- A route keeps its connections: its ends stay on the pads, vias and
  tracks they meet, and it keeps touching every pad and via it touched.
  Tracks of one net that only cross each other may come apart; see
  [crossing tracks](docs/selection.md#tracks-of-one-net-that-cross).
- kibends never refills zones and never runs the DRC. Refill the zones and
  run KiCad's DRC after every run.

## Limits

- The output is straight track segments. A selected arc is replaced by
  straight tracks; kibends never draws KiCad arcs.
- kibends cannot read custom DRC rules (`.kicad_dru`) or board minimums.
  It keeps the netclass clearances and the edge, hole and hole-to-hole
  clearances from the project file. [Clearances](docs/clearances.md) lists
  what only KiCad's DRC checks.
- A selection whose copper already violates a clearance is refused, with
  each place listed.
- Skew between the two tracks of a differential pair is not corrected.
  See [Differential pairs](docs/differential-pairs.md).

## How it works

kibends looks for the shortest path for each route that still keeps away
from the copper around it. Tracks drift toward the middle of the free space
and bend smoothly round what they have to avoid. It refines all selected
routes and vias together, coarse to fine, so neighbouring tracks make room
for each other. Signal vias are also drawn toward their ground return vias
([Return vias](docs/return-vias.md)).

## Command line

`kibends smooth` does the same as the panel's Run. It needs a
selection; `--all` smooths the whole board:

```sh
kibends smooth --dry-run   # optimize and report, change nothing
kibends smooth             # optimize and commit
kibends smooth --all       # the whole board, whatever is selected
```

Ctrl-C stops a run and commits the best result so far.
[Command line](docs/cli.md) lists every option.

## Documentation

- [The panel](docs/panel.md)
- [What gets smoothed](docs/selection.md)
- [Differential pairs](docs/differential-pairs.md)
- [Return vias](docs/return-vias.md)
- [Clearances](docs/clearances.md)
- [Command line](docs/cli.md)
- [Development](docs/development.md)

kibends uses the KiCad IPC API through
[kicad-python](https://pypi.org/project/kicad-python/).
