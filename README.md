# kibends

Smooths KiCad tracks. `kibends smooth` takes the chains of track segments
selected in the PCB editor and replaces each with a short, smooth path of
plain straight segments that keeps clearance to other copper, including the
other selected chains. The result is not made of KiCad native arcs, so the
tracks stay editable.

The new path minimises a weighted sum of two goals:

- **length**: in free space, two endpoints give a straight track;
- **closeness to other copper**: the track drifts toward the middle of the
  free space around it.

The original route is only the starting point. Bends remain where other
copper or the board edge forces them, rounded by the track keeping away
from the copper it turns around, and the track stays on the same side of
every obstacle it started next to.

Uses the KiCad IPC API through
[kicad-python](https://pypi.org/project/kicad-python/).

Status: early development.

## Requirements

- KiCad 10 or later, with the API server enabled
  (Preferences > Plugins > Enable KiCad API).
- Python 3.14 or later.

## Installation

```sh
uv tool install git+https://github.com/mairas/kibends
```

With `--plot` support, which adds matplotlib:

```sh
uv tool install 'kibends[plot] @ git+https://github.com/mairas/kibends'
```

## Usage

1. Open the board in KiCad's PCB editor.
2. Select the track segments to smooth, on any copper layers. Each
   connected piece of one net and one width on one layer is a chain: its
   segments join end to end. Where three or more selected tracks of one net
   meet at their ends, the selection branches: it is split there into chains
   that end at the junction, the junction moves with them unless other
   copper of the net holds it, and the branches meet at one point again in
   the result. A track that ends in the middle of another, anywhere its
   copper reaches the other's, branches there too: the other is split at
   the point of its centreline nearest the end, and the end moves onto it.
   That holds when every track ending there meets the other from one side,
   or ends past its centreline by no more than the other's half width, and
   leaves its copper through its side, and no other copper of the net
   touches the end where it was drawn. A track that crosses the other, runs
   along it, already connects to it through the selection, or has other
   copper on its end keeps its end, and the two stay connected where their
   copper overlaps; a track that lies on another of its width merges into
   it. Tracks of two widths that overlap, as at a neck-down, stay separate
   chains of their own widths, held where their copper overlaps. A
   route that changes layers through a via is one chain per layer. Chains on different layers do not affect
   each other, except through the vias that move with them.
   An unselected via stays where it is and holds the chain ends on both
   sides. Select a via with its tracks to let it move with them: it moves
   when it is an unlocked through via, at least two selected chain ends meet
   it, and no other copper touches it or those chain ends on any layer,
   except a zone of its net. A selected stitching via, an unlocked through
   via that no selected track ends at and no copper but a zone of its net
   touches, moves too: it has no length to shorten, so it stays where it
   is unless copper within `--range` of its clearance crowds it (a selected
   track cutting a corner, another via, a pad or other static copper, or
   the edge of its zone), and then backs off until the pushes balance or
   nothing is in range. One that no zone of its net reaches stays where it
   is. A via in a plane stays inside the zone on each
   layer where the zone's region reaches its ring, so that the next refill
   connects it again. kibends never reads a zone's fill, which may be out of date with
   the outlines or the copper; it works from outlines and priorities alone.
   A zone's region on a layer is its outline, out of the zone's own
   cut-outs, zones of other nets of higher or equal priority (grown by the
   larger of the two zones' clearances) and rule areas keeping out copper
   pour. A via whose ring such a region reaches on a layer it spans counts
   as connected to that zone, even where the current fill misses it; a via
   whose ring no such region reaches moves as in free space. A moving
   via's centre stays half the zone's minimum width plus 0.005 mm (KiCad's
   default arc error) inside the region, and it keeps the zone's clearance plus that margin from
   other nets' copper on those layers, where that is more than the netclass
   clearance, including tracks and vias that move in the same run. Thermal
   reliefs, island removal, hatched fills and a zone's minimum width
   through narrow necks are not modelled: a via can end up in a part of the
   zone the refill leaves out, in a gap of a hatch, or connected only
   through a neck, which only KiCad's DRC
   sees. Any other selected via stays where it is; a
   debug log line names it and the reason: it is locked, it is not a
   through via, other copper touches it or a chain end it would carry (an
   unselected track, a pad, a copper shape or another via of its net, on
   any layer), it sits at the edge of a zone's region (closer to it than
   that margin, so moving might lose a connection it has now), only one
   selected track ends at it, or none does and no zone of its net reaches
   it. A moving via that starts closer to other
   copper than its clearance stays where it is too, connected as it is, and other chains keep only the netclass clearance from it. Teardrops are ignored:
   KiCad reapplies them on the next refill. Selected pads and
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
including layers no selected chain is on, stays out of rule areas that keep
out vias on those layers, and the chains it joins move with it. A selected
via that starts inside its clearance or inside such a rule area stays where
it is. Vias of one net keep only the hole-to-hole distance from each other
and repel each other within `--range` of it, so a dense grid of selected
stitching vias spreads, and each moving via adds to the run's time: leave
stitching vias out of the selection, lock them, or pass `--fixed-vias`
where they should stay.

All the chains of one layer are one problem, which a single optimizer run
moves at once; a moving via joins every layer it spans, and the chains on
them, into one problem. The optimizer works in bursts: each burst moves
every vertex and via of a problem, each only within its own free space and
a share of its distance to the other chains, so that no chain jumps over
copper or over another chain, and it undoes any move that would bring an
edge newly into another item's clearance. An outer iteration gives every
problem that has not settled one burst. A problem settles when a burst
barely changes it or its energy stops improving.

The command prints the optimizer's outcome: whether it converged or stopped
at its iteration cap or the time limit, after how many iterations and how
many seconds and how many edge moves it rejected.
Per-chain figures are not printed: `--plot` titles each chain's subplot
with its length before and after and its minimum clearance margin.
While a run lasts longer than a few seconds, it reports its progress every
few seconds: the iteration, the time so far, the selected tracks' summed
length, the energy it minimises and the lowest it has kept, and how far the
iteration moved any track or via. Set `LOGURU_LEVEL=DEBUG` to see that line
after every iteration.
Every position kibends prints, in errors and warnings, is in millimetres relative to the board's grid origin, with y
pointing down: as the PCB editor shows coordinates with the grid origin
chosen under Preferences > PCB Editor > Origins & Axes and neither axis
inverted.
A chain the optimizer leaves as it was is not replaced, and a run that
changes no chain still commits the stitching vias it moved. Without `--dry-run`, the
changed chains are replaced and the moved vias moved in one commit, which
one undo in KiCad reverts; if any chain or via fails, nothing is committed.
A track or via edited in KiCad since the run read it stops the commit.
kibends does not refill zones and does not run KiCad's DRC: refill the
zones yourself (B), or run the DRC, which refills them and checks the result
against every rule of the board.

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
already bends tighter keeps its own tracks. So does one whose corner the
copper around it keeps from rounding, such as a pad inside a sharp corner,
or its own parts beyond the corner: the run is repeated with that section
held, so that the other chains keep clear of its tracks. The repeat goes on from
where the first run left every other chain and via, and `--time-limit`
covers both runs.

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
Both nets' tracks go into the one commit.

Bending a pair makes its inner track shorter, so the skew, the length
difference of the two chains, changes; kibends does not correct it, which
KiCad's length tuner does. Same-net copper that touches a coupled section
between its ends, such as a pad the pair runs past or a stub that branches
off one track, fixes the pair there. Where both tracks run straight for
at least two pitches (track width plus gap) to either side of that
point, the section is split there, and square across the pair from it on
the other track, into sections that each move on their own and meet where
the copper is; a part too short to change shape between two such points
stays as it is. Copper anywhere else, such as near a corner, leaves the
whole section as it is.

A run warns when two chains whose nets pair run coupled but have
different widths, when a chain runs coupled with one of its pair net that
pairs with another chain, and when a pair's measured gap differs by more
than a tenth from its netclass's differential pair gap. Each of these is
smoothed as separate chains, or, for the last, at its measured gap. A pair
is paired on each layer separately, and a via that joins a pair's track
stays fixed. kibends does not route uncoupled fan-outs
into coupling.

### Options

| Option | Default | Effect |
|---|---|---|
| `--repulsion WEIGHT` | 0.5 | Weight of keeping away from other copper, relative to length. 0 disables it, so the track pulls tight against obstacles at exactly the clearance. |
| `--range MM` | 1.0 | Distance beyond the required clearance at which repulsion fades to zero. Copper further away has no effect. |
| `--edge-clearance MM` | 0.3 | Clearance to the board edge. |
| `--hole-clearance MM` | 0.25 | Clearance between drilled holes and other nets' copper. A moving via's hole also keeps 0.25 mm from every other hole. |
| `--chord-tolerance MM` | 0.01 | How far the committed tracks may depart from the optimised curve; larger values usually give fewer, longer tracks, though beside copper a pair may keep more. |
| `--time-limit SECONDS` | none | Stop optimizing after this long and go on with the best result found so far: simplification and the commit run as usual, so a partial result can be inspected and undone in one step. The optimizer checks the limit after each problem's burst, so it may run over by one burst, or by one outer iteration when it must run again with a differential pair's section held as routed. |
| `--settled-change MM` | 0.001 | A problem settles when a burst moves none of its tracks or vias by this much. Raising it alone saves little time and can leave visibly different tracks. |
| `--stall-tolerance MM` | 0.001 | A problem also settles when its energy, in mm of track length, improves by less than this over `--stall-window` of its bursts. Raising it saves little time and lets some tracks end slightly longer. |
| `--stall-window N` | 3 | Bursts of a problem the stall check looks back over. |
| `--max-iterations N` | 200 | Outer iterations, each one burst of every problem not yet settled, after which the optimizer stops and goes on with the best result found. |
| `--burst-iterations N` | 50 | L-BFGS-B iterations per burst. Each burst re-checks how far every track and via may move, so on a large selection shorter bursts reach a given result sooner. Fewer shorten a run further, but at 25 a differential pair in the test scenarios settles visibly short of its smooth shape: `--stall-window` counts bursts, so with shorter bursts it looks back over less work. More get a little closer to the exact optimum at several times the cost. |
| `--lbfgs-ftol TOL` | 1e-9 | L-BFGS-B ends a burst early when the energy's relative reduction falls below this. Smaller values let bursts run to `--burst-iterations`; in the test scenarios they changed a scenario's summed track length by up to about 15 µm. Much larger values shorten runs but change the result a lot: 1e-6 moved some tracks by tenths of a millimetre in [an earlier measurement](https://github.com/mairas/kibends/issues/27). |
| `--lbfgs-gtol TOL` | 1e-10 | L-BFGS-B ends a burst early when the projected gradient falls below this. It rarely ends one before `--lbfgs-ftol` or `--burst-iterations` do. |
| `--no-move-bounds` | | Experimental: let every vertex and via move as far as it likes in each burst, instead of a box no larger than the free space around it. Without these bounds a track or via may pass through another one within a burst, which the optimizer's check of each new edge does not always catch, so the result can route a track on the other side of a pad, via or other track than before. KiCad's DRC catches only a result inside another item's clearance or with a broken connection; a track that lands cleanly on the other side of a pad passes it, so inspect the result. |
| `--allow-via-in-pad` | | Let a moving via settle inside a pad of its own net. Without it a moving via may touch a pad of its net but never overlaps one. KiCad's DRC does not check via-in-pad, so this option is the only guard: use it only where the fabricator fills and caps vias in pads. Holes keep their clearances either way. |
| `--fixed-vias` | | Keep every selected stitching via (one no selected track ends at) where it is, as if it were not selected, and spend no time moving it. Vias that carry selected tracks still move. |
| `--dry-run`, `-n` | | Optimize and print the outcome without changing the board. |
| `--plot FILE` | | Draw each original and smoothed chain over the area its centreline must avoid, one subplot per chain, with the start and end of each moving via it joins; the format follows the extension. Works with or without `--dry-run`. |

Repulsion alone rounds the bends: with the defaults, a 90° corner wrapped
around a pad in the test geometries gets a bend radius of about 1.3 mm.
More `--repulsion` or a longer `--range` widens it and spreads tracks
further from their neighbours where space allows; with `--repulsion 0` a
track bends at the corner of the clearance around the pad. A track is
straight between the copper it turns around, so it may leave a pinned end,
such as the end of a differential pair's coupled section, at a corner.

### Clearances

kibends keeps only the clearances its optimizer models: the board's
netclass clearances, including pad and footprint clearance overrides; the
edge and hole clearances of `--edge-clearance` and `--hole-clearance`,
since KiCad's API does not expose the board's own; rule areas that keep
out tracks or vias, which moving vias stay out of as well; and, for a
moving via, the outlines and clearances of the zones it connects to. Rule
areas' other keepouts, of pads, copper pours and footprints, are not seen.
Nor are custom DRC rules (the project's `.kicad_dru`), board minimums,
thermal reliefs, island removal, hatched fills or minimum-width necks, or
copper zones,
which reflow after an edit, beyond what holds a moving via. A custom rule
stricter than the netclass clearances (a larger clearance for vias, say)
can therefore be violated by the smoothed tracks. Those are left to
KiCad's DRC: run it after smoothing.

### Editing a smoothed track

Change the route by hand or with the router, select the chains again and
re-run. Re-running on kibends' own output with nothing else changed moves it
only slightly, but still replaces its tracks.

## Development

```sh
./run sync
./run test       # fast tests, in parallel
./run test-all   # also the slow tests; run before committing
```
