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
from the copper it turns around. Copper of its own net is no obstacle.

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
   copper on its end keeps its end, and the other is held where that end's
   copper reaches it; a track that lies on another of its width merges into
   it. Tracks of two widths that overlap, as at a neck-down, stay separate
   chains of their own widths, held where the end of one reaches the other.
   Where selected tracks of one net overlap or cross away from their ends,
   nothing holds them: copper of one net does not interact, and the run may
   move them apart, even where that overlap was their only connection. So a
   stub drawn past the track it crosses, its end beyond that track's copper,
   may come off it: draw a T to end on the track, or check the ratsnest
   after the run. A
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
   track cutting a corner, a via of another net, a pad or other static
   copper of another net), and then backs off until the pushes balance or
   nothing is in range. Copper of its own net does not push it, and the
   edge of its zone, the holes of its net's vias and pads and its net's
   pads only stop it. One that no zone of its net reaches stays where it
   is. A via in a plane stays inside the zones of its net on each layer
   where a zone's region reaches its ring, so that the next refill
   connects it again. kibends never reads a zone's fill, which may be out of date with
   the outlines or the copper; it works from outlines and priorities alone.
   A zone's region on a layer is its outline, out of the zone's own
   cut-outs, zones of other nets of higher or equal priority (grown by the
   larger of the two zones' clearances) and rule areas keeping out copper
   pour. A via whose ring such a region reaches on a layer it spans counts
   as connected to that zone, even where the current fill misses it; a via
   whose ring no such region reaches moves as in free space. A moving
   via's centre stays half the zone's minimum width plus 0.005 mm (KiCad's
   default arc error) inside the region. Zones of its net that overlap or
   meet on a layer count together there: the centre stays that deep inside
   one of them by its own minimum width, or inside the union of their
   regions by the largest minimum width among them, so a via deep inside a
   large plane moves although a smaller zone of the same net drawn over it
   ends at the via, and one on the seam of two zones meeting edge to edge
   moves too. It keeps the zone's clearance plus that margin from
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
repel each other like other copper. Copper of one net does not interact at
all: chains of one net neither repel nor hold each other back, and may come
close, overlap or cross, as may a chain and a via of its net, and a chain
and the static copper of its net. Only selected chains and vias move. A moving via keeps its clearances on every copper layer it spans,
including layers no selected chain is on, stays out of rule areas that keep
out vias on those layers, and the chains it joins move with it. A selected
via that starts inside its clearance or inside such a rule area stays where
it is. Vias of one net, moving or not, keep only the hole-to-hole distance
from a moving via, their rings free to overlap, and do not repel it, so a
grid of selected stitching vias keeps its spacing, as does a selected one
beside an unselected one; a moving via does not repel from the edge of a
zone it stays inside either, which only stops it, while the board edge and
other nets' copper still repel it. Tracks and copper shapes of the via's own
net do not affect it, and its net's pads only stop it from overlapping them
(see `--allow-via-in-pad`). Each moving via adds to the run's time: leave stitching vias out
of the selection, lock them, or pass `--fixed-vias` where they should
stay.

All the chains of one layer are one problem, which a single optimizer run
moves at once; a moving via joins every layer it spans, and the chains on
them, into one problem. The optimizer works in bursts: each burst moves
every vertex and via of a problem at once, and undoes any move that would
bring an edge or a via newly into another item's clearance. An outer iteration gives every
problem that has not settled one burst. A problem settles when a burst
barely changes it or its energy stops improving.

The command prints the optimizer's outcome: whether it converged or stopped
at its iteration cap or the time limit, after how many iterations and how
many seconds and how many edge moves it rejected.
Per-chain figures are not printed: `--plot` titles each chain's subplot
with its length before and after and its minimum clearance margin.
It reports when it starts setting up the optimizer, and before the first
iteration how long setup took, how many problems it split the selection into
(the layers or groups of layers the moving tracks and vias tie together), and
their starting summed length and energy. A run repeated to hold a
differential pair's section reports both again, saying so.
While a run lasts longer than a few seconds, it reports its progress every
few seconds: the iteration, the time so far, the selected tracks' summed
length, the energy it minimises, and how far the iteration moved any track or
via. Pass `--verbose` (`-v`) to see that line
after every iteration, with the other debug lines.
Every position kibends prints, in errors and warnings, is in millimetres relative to the board's grid origin, with y
pointing down: as the PCB editor shows coordinates with the grid origin
chosen under Preferences > PCB Editor > Origins & Axes and neither axis
inverted.
A chain the optimizer leaves as it was is not replaced, and a run that
changes no chain still commits the stitching vias it moved. Without `--dry-run`, the
changed chains are replaced and the moved vias moved in one commit, which
one undo in KiCad reverts; if any chain or via fails, nothing is committed.
After the commit, the new tracks and moved vias are added to the selection,
so it covers the same copper as before the run and a second run needs no
new selection; selected items the run left unchanged stay selected. A dry
run, or a run that commits nothing, leaves the selection as it was.
A track or via edited in KiCad since the run read it stops the commit.
kibends does not refill zones and does not run KiCad's DRC: refill the
zones yourself (B), or run the DRC, which refills them and checks the result
against every rule of the board.

Locked tracks in the selection are left untouched, and a locked track that
splits a selected piece makes two chains. Each chain's two ends stay where
they are, unless it ends at a moving via or slides within a pad or via
(below), as does every point where other copper of the same net (a track
or copper shape, or the end of another selected chain) touches the chain.

A pad of the chain's net, or a via of its net that does not move, only
needs to keep touching the chain somewhere. A chain that ends inside one
keeps its end inside it, 5 µm within its outline, but the end may slide
anywhere there, and the rest of the chain, including what ran inside the
pad, is free. A chain that runs over one partway along keeps one vertex
inside it, where two of its new tracks meet, free to slide there in the
same way, however many times it entered the pad. KiCad connects a track
to a pad or via wherever their copper overlaps, so that vertex keeps both
its tracks connected. A vertex of the chain that lies exactly on a pad's
anchor or a via's centre stays there, and needs no other vertex in that pad
or via: KiCad drags only the track ends there along with a footprint or a
via, so moving them would leave the tracks behind when you drag it later.
A pad or via pins the chain where it touches it, as other
copper does, where the chain only grazes it without its centreline
entering, where it also touches an end that other copper holds, and at the
junction of branching tracks. An end whose copper reaches another selected
track of its net stays where it is too, since that track keeps it there. The tracks of a differential pair keep the
vertex inside it where it is.

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

A chain of one of the pair's nets, such as a fan-out, keeps clear of the
pair's track of the other net. The coupled section cannot tell which of its
tracks faces the chain, so it keeps its whole width clear of the chain as of
any other, except near where the two connect. In effect, beyond its fan-out
such a chain keeps off the pair's track of its own net too: it neither comes
close to it nor crosses it there.

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
| `--burst-iterations N` | 50 | L-BFGS-B iterations per burst. Each burst resamples the tracks and checks every move, so on a large selection shorter bursts reach a given result sooner. Fewer shorten a run further, but at 25 a differential pair in the test scenarios settles visibly short of its smooth shape: `--stall-window` counts bursts, so with shorter bursts it looks back over less work. More get a little closer to the exact optimum at several times the cost. |
| `--lbfgs-ftol TOL` | 1e-9 | L-BFGS-B ends a burst early when the energy's relative reduction falls below this. Smaller values let bursts run to `--burst-iterations`; in the test scenarios they changed a scenario's summed track length by up to about 15 µm. Much larger values shorten runs but change the result a lot: 1e-6 moved some tracks by tenths of a millimetre in [an earlier measurement](https://github.com/mairas/kibends/issues/27). |
| `--lbfgs-gtol TOL` | 1e-10 | L-BFGS-B ends a burst early when the projected gradient falls below this. It rarely ends one before `--lbfgs-ftol` or `--burst-iterations` do. |
| `--allow-via-in-pad` | | Let a moving via settle inside a pad of its own net. Without it a moving via may touch a pad of its net but never overlaps one; the pad stops it without pushing it. KiCad's DRC does not check via-in-pad, so this option is the only guard: use it only where the fabricator fills and caps vias in pads. Holes keep their clearances either way. |
| `--fixed-vias` | | Keep every selected stitching via (one no selected track ends at) where it is, as if it were not selected, and spend no time moving it. Vias that carry selected tracks still move. |
| `--dry-run`, `-n` | | Optimize and print the outcome without changing the board. |
| `--plot FILE` | | Draw each original and smoothed chain over the area its centreline must avoid, one subplot per chain, with the start and end of each moving via it joins; the format follows the extension. Works with or without `--dry-run`. |
| `--verbose`, `-v` | | Log at DEBUG: progress after every iteration and the debug lines, such as why a selected via stays where it is. Overrides the level the `LOGURU_LEVEL` environment variable sets; without either, kibends logs at INFO. |

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
