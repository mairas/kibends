# kibends

Smooths KiCad tracks. `kibends smooth` takes the chains of track segments
selected in the PCB editor and replaces each with a short, smooth path of
plain straight segments that keeps clearance to other copper, including the
other selected chains. The result is not made of KiCad native arcs, so the
tracks stay editable.

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

Status: early development.

## Requirements

- KiCad 10 or later, with the API server enabled
  (Preferences > Plugins > Enable KiCad API).
- Python 3.14 or later.
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
2. Select the track segments to smooth, on any copper layers. Each
   connected piece of one net and one width on one layer is a chain: its
   segments join end to end without branches. Two widths of one net may meet
   only end to end, as at a neck-down. A route that changes layers through a
   via is one chain per layer. Chains on different layers do not affect
   each other, except through the vias that move with them.
   An unselected via stays where it is and holds the chain ends on both
   sides. Select a via with its tracks to let it move with them: it moves
   when it is an unlocked through via, at least two selected chain ends meet
   it, and no other copper touches it or those chain ends on any layer. Any
   other selected via stays where it is, with a warning naming it and the
   reason: it is locked, it is not a through via, other copper touches it
   or a chain end it would carry (an unselected track, a pad, a copper
   shape, another via or a zone of its net, on any layer), or fewer than
   two selected chain ends meet it. Teardrops are ignored, here and in the
   DRC check: KiCad reapplies them on the next refill. Selected pads and
   other items are ignored; arc tracks are refused.
   A box selection catches tracks on every visible layer, and all of them
   are smoothed. Check what a run would change with `--dry-run` first, or
   set non-active layers to Dim or Hide (Ctrl+H), which limits selection to
   the active layer.
3. Run:

```sh
kibends smooth --dry-run --plot chain.png   # look first
kibends smooth                              # then commit
```

The chains are optimised together, so neighbouring chains make room for
each other: chains of different nets keep their clearance to each other and
repel each other like other copper, while chains of one net may touch but
not overlap, except where they already connect. Only selected chains and
vias move. A moving via keeps its clearances on every copper layer it spans,
including layers no selected chain is on, and the chains it joins move with
it.

The command prints, for each chain, the length, track count, tightest bend
radius and clearance margins before and after, then for each differential
pair its gap and skew (see below), then for each selected via that may move
how far it moved and where to, then the optimizer's outcome: whether it
converged or stopped at its iteration cap or the time limit, after how many
iterations and how many seconds.
While a run lasts longer than a few seconds, it reports its progress every
few seconds: the iteration, the time so far, the selected tracks' summed
length, the energy it minimises and how many chains are still moving. Set
`LOGURU_LEVEL=DEBUG` to see that line after every iteration.
Every position kibends prints, in errors, warnings, statistics and DRC
lines, is in millimetres relative to the board's grid origin, with y
pointing down: as the PCB editor shows coordinates with the grid origin
chosen under Preferences > PCB Editor > Origins & Axes and neither axis
inverted.
A chain the optimizer leaves as it was is marked unchanged and is neither
checked nor replaced. Without `--dry-run`, and if the DRC check passes, the
changed chains are replaced and the moved vias moved in one commit, which
one undo in KiCad reverts; if any chain or via fails, nothing is committed.
A track or via edited in KiCad since the run read it stops the commit.
kibends does not refill zones: refill them yourself (B) or let KiCad's DRC
refill them.

Locked tracks in the selection are left untouched, and a locked track that
splits a selected piece makes two chains. Each chain's two ends stay where
they are, unless it ends at a moving via, as does every point where other
copper of the same net (a pad, via, track, copper shape or another selected
chain) touches the chain.

A vertex that starts inside another item's clearance stays where it is,
with a warning. When that item is another selected chain, the warning names
it, and a re-run after that chain has moved may free the vertex.

### Differential pairs

Two selected chains on one layer whose nets KiCad pairs by name, a common
base with a `+`/`-` or `P`/`N` suffix (trailing digits and underscores
ignored, as KiCad's router does), are smoothed as a pair where they run
coupled: side by side at a near-constant gap for at least five pitches and
1 mm. Each such stretch is a coupled section; a pair that splits around a
pad or via and rejoins has several. The pair's centreline is optimised and
both tracks follow it at the gap the routing had, measured as the median
over the coupled sections, so a pair costs about as much as one chain. The
rest of each chain, such as its fan-outs to pads or vias, is smoothed as an
ordinary chain, and each section meets it where and in the direction it
did. A section's centreline never bends tighter than its inner track
allows, half the pitch plus half the track width. A section whose routing
already bends tighter keeps its own tracks, with a warning that says so.
So does one whose corner the copper around it keeps from rounding, such as
a pad inside a sharp corner, or its own parts beyond the corner: the
warning says which, and the run is repeated with that section held, so
that the other chains keep clear of its tracks.

A coupled section is simplified as its centreline, within the chord
tolerance, and both tracks follow the simplified centreline as they
followed the optimised one: at the gap, tapering to an end's own gap near
an end routed at another, segment for segment in parallel outside such a
taper, and meeting the fan-outs in the direction they did. While
optimising, a section's parts more than a turn apart along it, such as the
legs of a U or the laps of a spiral, do not move closer to each other than
the pair's outer extent plus the larger clearance, and parts that already
lie closer do not move closer still. The run refuses, naming the pair, to commit
a section whose tracks come closer to each other than both its nets'
clearance and its routing kept them, where they run side by side or
across a turn.
Both nets' tracks go into the one DRC check and the one commit.

The report adds a block per pair: the measured gap and the range of the
gap along the result's coupled sections, and the skew, the length
difference of the two chains, before and after. Bending a pair makes its
inner track shorter, so the skew changes; kibends reports it but does not
correct it, which KiCad's length tuner does. A coupled section that
same-net copper touches between its ends stays as it is, and the report
counts it as held, as it does a section kept for bending too tightly.

A run warns when two chains whose nets pair do not run coupled or have
different widths, when a chain runs coupled with one of its pair net that
pairs with another chain, and when a pair's measured gap differs by more
than a tenth from its netclass's differential pair gap. Each of these is
smoothed as separate chains, or, for the last, at its measured gap. A pair
is paired on each layer separately, and a via that joins a pair's track
stays fixed, with a warning. kibends does not route uncoupled fan-outs
into coupling.

### Options

| Option | Default | Effect |
|---|---|---|
| `--stiffness MM` | 0.3 | Weight of bending, as a length. Larger values give wider, rounder bends at the cost of length. Obstacles can still force tighter bends. |
| `--repulsion WEIGHT` | 0.5 | Weight of keeping away from other copper, relative to length. 0 disables it, so the track pulls tight against obstacles at exactly the clearance. |
| `--range MM` | 1.0 | Distance beyond the required clearance at which repulsion fades to zero. Copper further away has no effect. |
| `--edge-clearance MM` | 0.3 | Clearance to the board edge. |
| `--hole-clearance MM` | 0.25 | Clearance between drilled holes and other nets' copper. A moving via's hole also keeps 0.25 mm from every other hole. |
| `--chord-tolerance MM` | 0.01 | How far the committed tracks may depart from the optimised curve; larger values usually give fewer, longer tracks, though beside copper a pair may keep more. |
| `--time-limit SECONDS` | none | Stop optimizing after this long and go on with the best result found so far: simplification, the DRC check and the commit run as usual, so a partial result can be inspected and undone in one step. The optimizer checks the limit after each chain's or via's burst, so it may run over by one burst, or by one outer iteration when it must run again with a differential pair's section held as routed. |
| `--settled-change MM` | 0.001 | A chain settles when a burst moves its shape by less than this. Raising it alone saves little time and can leave visibly different tracks. |
| `--stall-tolerance MM` | 0.001 | A chain also settles when its energy, in mm of track length, improves by less than this over `--stall-window` iterations. Raising it saves little time and lets some tracks end slightly longer. |
| `--stall-window N` | 3 | Outer iterations the stall check looks back over. |
| `--max-iterations N` | 200 | Outer iterations after which the optimizer stops and goes on with the best result found. |
| `--burst-iterations N` | 300 | L-BFGS-B iterations per burst. Fewer are the strongest way to shorten a run, but some layouts then settle into a noticeably different shape. |
| `--lbfgs-ftol TOL` | 1e-15 | L-BFGS-B ends a burst early when the energy's relative reduction falls below this; the default effectively never does, so bursts run to `--burst-iterations`. A larger value, such as 1e-9, shortens runs, most of all on boards with differential pairs, at the cost of small changes in the result; much larger values change it a lot. |
| `--lbfgs-gtol TOL` | 1e-10 | L-BFGS-B ends a burst early when the projected gradient falls below this. It rarely ends one before `--lbfgs-ftol` or `--burst-iterations` do. |
| `--allow-via-in-pad` | | Let a moving via settle inside a pad of its own net. Without it a moving via may touch a pad of its net but never overlaps one. KiCad's DRC does not check via-in-pad, so this option is the only guard: use it only where the fabricator fills and caps vias in pads. Holes keep their clearances either way. |
| `--dry-run`, `-n` | | Print the statistics and the DRC verdict without changing the board. |
| `--no-drc` | | Commit without the DRC check. |
| `--kicad-cli PATH` | | The kicad-cli to run the DRC check with. By default kibends uses the one the running KiCad reports, then `kicad-cli` on `PATH`, then the usual install locations. |
| `--plot FILE` | | Draw each original and smoothed chain over the area its centreline must avoid, one subplot per chain, with the start and end of each moving via it joins; the format follows the extension. Works with or without `--dry-run`. |

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
minimums are not seen by the optimizer. Neither are copper zones, which
reflow after an edit, nor rule areas that keep out vias.

The DRC check covers what the optimizer cannot see. Before committing,
kibends runs `kicad-cli pcb drc` twice on copies of the board in a scratch
directory, with the project's `.kicad_pro` and `.kicad_dru`: once on the
board as it is, and once with every changed chain replaced and every moved
via in its new place. If the result
adds a violation, nothing is committed: the message names the nets involved,
and each new violation is listed with its type, description and position.
Violations the board already had do not count, including ones a chain's old
tracks or a moved via had against the same item; another chain's or via's
old violation does not excuse a new one. A path that
splits another net's pour shows up as more unconnected items on that net and
is refused, naming the net.

Save the board before running: kicad-cli reads the board's project and rule
files from disk, so rule or netclass edits not yet saved in KiCad are not
seen. The check takes two full-board DRC runs with zone refill; the log shows
how long each took.

### Editing a smoothed track

Change the route by hand or with the router, select the chains again and
re-run. Re-running on kibends' own output with nothing else changed moves it
only slightly, but still replaces its tracks.

## Development

```sh
./run sync
./run test       # fast tests, in parallel
./run test-all   # also the slow and kicad-cli tests; run before committing
```
