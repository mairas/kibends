# kibends

Smooths KiCad tracks. `kibends smooth` takes the chains of track segments
selected in the PCB editor and replaces each with a short, smooth path of
plain straight segments that keeps clearance to other copper, including the
other selected chains. Selected arc tracks are smoothed too, as straight
pieces along the arc. The result is not made of KiCad native arcs, so the
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
- A Rust toolchain, 1.90 or later, since installing builds kibends'
  optimizer core from source. Without `cargo` on the path, the build
  downloads one into its own cache; [rustup](https://rustup.rs) installs
  one to keep. An older `cargo` on the path, such as a distribution's
  package, stops the build.

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
   segments join end to end. A selected arc track counts as straight
   pieces along it, each turning at most 5°, and is replaced by straight
   tracks like the rest of its chain. Only a chain the run leaves exactly
   as drawn keeps its arc, which in practice is rare. A locked arc stays
   an arc and holds the chains that end on it, as a locked track does.
   Where three or more selected tracks of one net meet at their ends, or
   tracks of two widths meet end to end, as where a route narrows, the
   selection branches: it is split there into chains
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
   it. Tracks of two widths that overlap, as at a neck-down whose narrow
   track ends inside the wide one's copper rather than on its end, stay
   separate chains of their own widths, held where the end of one reaches
   the other.
   Where selected tracks of one net overlap or cross away from their ends,
   nothing holds them: copper of one net does not interact, and the run may
   move them apart, even where that overlap was their only connection. So a
   stub drawn past the track it crosses, its end beyond that track's copper,
   may come off it: draw a T to end on the track, or check the ratsnest
   after the run. A closed loop is cut where copper of its net touches it,
   a junction, another track, a pad or a via, and smoothed from there; one
   that nothing of its net touches is cut at its first point. A loop that
   surrounds nothing shrinks: to nothing, and its tracks are deleted, when
   it is held at one point, or to a line when held at two. A
   route that changes layers through a via is one chain per layer. Chains on different layers do not affect
   each other, except through the vias that move with them.
   An unselected via stays where it is and holds the chain ends on both
   sides. Select a via with its tracks to let it move with them: it moves
   when it is unlocked, it is not on a net of a differential pair, a
   selected chain end meets it or a zone of its net reaches it, and no
   other copper touches it or those chain ends on a layer it spans, except a
   zone of its net. One chain end is enough: a pad's fan-out via moves
   along its track toward the pad until the pad stops it, unless
   `--allow-via-in-pad` is given. Selected tracks that run through it count: where two of
   them meet at its centre, the chain is cut there into two that end at it,
   both carried, but a selected track that passes over it with no vertex at
   its centre holds it like other copper. Through, blind, buried and micro
   vias move alike, each on the layers it spans: a track, pad or other
   copper on a layer it does not reach neither moves with it nor holds it.
   Stacked vias of one net, such as a micro via on a buried via, touch on
   the layer they share, so each holds the other where it is, selected or
   not. A selected stitching via, an unlocked via that no selected track
   ends at and no copper but a zone of its net touches, moves too, also when
   the selection holds no tracks but locked ones: it has no length to
   shorten, so only the copper around it moves it. Copper within `--range`
   of its clearance pushes it (a selected track cutting a corner, a via of
   another net, a pad or other static copper of another net), and it backs
   off until the pushes balance or nothing is in range. A ground via and a
   via of any other net within `--attraction-range` also pull each other, and ground vias of one net
   push each other apart (see [Return vias](#return-vias)). Other copper of
   its own net does not push it, and the edge of its zone, the holes of its
   net's vias and pads and its net's pads only stop it. One that no
   zone of its net reaches stays where it is. A via in a plane stays inside the zones of its net on each layer
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
   sees. Any other selected via stays where it is; a debug log line names it
   and the reason: it is locked, it is on a net of a differential pair (see
   [Differential pairs](#differential-pairs)), a track of a selected group
   ends at it, other copper touches it or a chain end it
   would carry (an unselected track, a selected track that passes it with
   no vertex at its centre, a pad, a copper shape or another via of its
   net, on a layer it spans), it sits at the edge of a zone's region
   (closer to it than that margin, so moving might lose a connection it has
   now), or no selected track ends at it and no zone of its net reaches
   it. A moving via that
   starts closer to other copper than its clearance moves out of it like any
   other (see below). Teardrops are ignored: KiCad reapplies them on the
   next refill. Selected pads and other items are ignored.
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
repel each other like other copper. Copper of one net does not interact,
but for the vias of a ground net (below): chains of one net neither repel
nor hold each other back, and may come
close, overlap or cross, as may a chain and a via of its net, and a chain
and the static copper of its net. Nor do the selected tracks of the two
nets of a differential pair keep clearance from each other (below). Only
selected chains and vias move. A moving via keeps its clearances on every copper layer it spans,
including layers no selected chain is on, stays out of rule areas that keep
out vias on those layers, and the chains it joins move with it. A moving
via that starts inside its clearance or inside such a rule area moves out
of it where there is room, and never ends deeper inside than it started.
Starting inside the clearance of one chain or via does not let it into
another's: each chain and each via is judged on its own, and so is each
piece of static copper around it, the outside of its zones and each of its
net's pads and holes. Static copper of one net that overlaps on one
layer, such as two tracks meeting at a width change or a T-junction, is
one piece, judged by the via's depth in the whole of it: inside one track
of a piece, the via may move into another of its tracks no deeper. The
optimizer keeps vias so while it runs; kibends does not check the result
again before the commit, and KiCad's DRC reports any clearance violation
left after the run.
Two vias keep their clearances and hole-to-hole distance from each other
only where their spans overlap; KiCad's DRC judges any stricter rule for
blind, buried or micro vias.
Vias of one net, moving or not, keep only the hole-to-hole distance
from a moving via, their rings free to overlap. Vias of one net that is
not a ground net do not repel each other, so a grid of selected stitching
vias of such a net keeps its spacing, as does a selected one beside an
unselected one; vias of one ground net do (see
[Return vias](#return-vias)). A moving via does not repel from the edge of a
zone it stays inside either, which only stops it, while the board edge and
other nets' copper still repel it. Tracks and copper shapes of the via's own
net do not affect it, and its net's pads only stop it from overlapping them
(see `--allow-via-in-pad`). Each moving via adds to the run's time.

All the chains of one layer are one problem, which a single optimizer run
moves at once; a moving via joins every layer it spans, and the chains on
them, into one problem. The optimizer works in bursts: each burst moves
every vertex and via of a problem at once, and undoes any move that would
bring an edge or a via newly into another item's clearance. An outer iteration gives every
problem that has not settled one burst. A problem settles when a burst
barely changes it or its energy stops improving. Each evaluation of a
burst spreads over every logical CPU; set the environment variable
`RAYON_NUM_THREADS=N` to use N threads instead, with the same result.

The command prints the optimizer's outcome, after how many iterations and
how many seconds, and how many edge moves it rejected. It converged, had
nothing to do, or stopped at its iteration cap or the time limit, each
keeping the best result found; the warning of a run that stopped while some
problem was still at a coarse spacing says it stopped before reaching the
final spacing. It stopped on abnormal bursts when L-BFGS
ended three bursts of some problem in a row abnormally after the run had
improved the tracks: it keeps the best result found, which may fall short
of the smoothest. A run that ends so without improving anything prints
"L-BFGS terminated abnormally" and commits nothing.
Per-chain figures are not printed: `--plot` titles each chain's subplot
with its length before and after.
It reports when it starts setting up the optimizer, and before the first
iteration how long setup took, how many problems it split the selection into
(the layers or groups of layers the moving tracks and vias tie together), and
their starting summed length and energy. With coarse spacings, it says
when each problem starts at each of them ("Optimizing the chains on F.Cu
coarsely, at 1.6 mm spacing") and at the final spacing ("Refining the chains
on F.Cu at 0.1 mm spacing"); a problem that settles in its first burst,
which runs at the final spacing, skips the coarse spacings and logs neither. When it holds a differential
pair's section, it says how many chains and vias it put back.
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
against every rule of the board. kibends checks no clearance of its own
after optimizing: the DRC is the check for clearance after a run.

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

A vertex that starts inside another item's clearance moves like any other,
and the clearance pushes it out where there is room, another selected
chain moving aside too if it must. Where there is none, as in a channel
narrower than the track's clearance asks, the result has no more clearance
violations than the selection had, and KiCad's DRC reports what remains.

### Differential pairs

Two selected chains on one layer whose nets KiCad pairs by name, a common
base with a `+`/`-` or `P`/`N` suffix (trailing digits and underscores
ignored, as KiCad's router does), are smoothed as a pair where they run
coupled: side by side and parallel at a near-constant gap, however short
the stretch, as in KiCad's own diff pair coupling check. Either track may
be drawn in any number of segments, such as an arc split finely or the
straights of a length-tuning pattern. Each such stretch is a coupled
section; a pair that splits around a
pad or via and rejoins has several. The pair's centreline is optimised and
both tracks follow it at the gap the routing had, measured as the median
over the coupled sections, so a pair costs about as much as one chain. The
rest of each chain, such as its fan-outs to pads or vias, is smoothed as an
ordinary chain, and each section meets it where and in the direction it
did. A section's centreline never bends tighter than its inner track
allows, half the pitch plus half the track width. A section whose routing
already bends tighter keeps its own tracks; a jog or sliver of less than a
micrometre, such as KiCad leaves on its nanometre grid, does not count. So
does one whose corner the copper around it keeps from rounding, such as a
pad inside a sharp corner: when the run would end, that section goes back
to its tracks and the run goes on with it held, so that the other chains
keep clear of its tracks. Every other chain and via goes on from where the
run had taken it, except those that had moved into the section's clearance,
or into a chain or via that had to go back, which go back to where they
started; with time left, the run then smooths them again. `--time-limit`
still covers the whole run: if the time is up when the section is held, the
run ends there, once it has also held any section that going back left bent
too tightly.

A coupled section is simplified as its centreline, within the chord
tolerance, and both tracks follow the simplified centreline as they
followed the optimised one: at the gap, segment for segment in parallel,
and meeting the fan-outs in the direction they did. Where an end was
routed at another gap, one track's end edge, about a spacing long, takes up
the difference and turns slightly off the direction it had. kibends does
not check the committed tracks' gap again; KiCad's DRC reports a pair's tracks that end up closer
than their clearance.
Both nets' tracks go into the one commit.

The selected tracks of a diff pair's two nets do not keep clearance from
each other while optimising: a coupled section and the chains of its two
nets, such as its fan-outs, ignore each other, and so do a chain of one net
and a chain of the other, at the fan-outs and wherever the two tracks run
apart, such as around a via. Nothing keeps one net's track from being
pulled across the other's there; kibends does not check for it, and KiCad's
DRC reports it. The diff pair's vias, and the copper of either net that is
not selected, still keep clearance from the other net's tracks.

Bending a pair makes its inner track shorter, so the skew, the length
difference of the two chains, changes; kibends does not correct it, which
KiCad's length tuner does. Same-net copper that touches a coupled section
between its ends, such as a pad the pair runs through or a stub that
branches off one track, fixes the pair there: the track it touches keeps
passing through that point in the direction it did, the other track runs
beside it at the gap, and the rest of the section, such as a corner beyond
the pad, is smoothed. Copper close to a corner keeps that corner sharp, and
when that bends more tightly than the inner track allows, or leaves too
little of the section to change shape, the whole section stays as it is.

A selected stub, such as a test pad's or an ESD diode's, splits the track
it branches off into separate chains. The other track of the diff pair is
then cut opposite each end where the two run coupled up to it, or to within
one pitch of it, and each part pairs with the part beside it. The stub's
junction and the points opposite it stay where they are. On a straight the
diff pair stays coupled at its gap up to the stub on both tracks. Within
about a pitch of a corner, coupling ends at the corner's vertex, and the
stretch of either track between it and the stub stays as drawn, at its
routed gap.

A selected group, such as a length-tuning pattern, counts as its tracks,
arcs and vias and those of every group inside it. They stay exactly as
drawn, and the tracks they touch end there. They still pair like any
track: where a tuning pattern's straight runs beside the other track of
its diff pair, that track keeps its routed gap, and the diff pair on
either side of the pattern pairs up to it as at a stub. Beside the
pattern's bumps, the other track is smoothed on its own.

A run warns when two chains whose nets pair run coupled but have
different widths; they are smoothed as separate chains. A chain whose
partner beside it pairs with another chain of its net is smoothed on its
own, without a warning. A diff pair keeps its measured gap, whatever its netclass sets. A
pair is paired on each layer separately. kibends does not route uncoupled
fan-outs into coupling.

A diff pair's vias do not move. A selected via on either net of a diff
pair stays where it is, whether or not the selection pairs its tracks. As
in KiCad, a net is of a diff pair when its name ends in `+`, `-`, `P` or
`N`, before any trailing digits and underscores, and the net with the
opposite suffix exists on the board: a via on `VIN` moves when the board has
no `VIP`. The tracks that end at a diff pair's via keep their ends there:
moving the two nets' vias one by one would break the symmetry of the diff
pair's via structure and its length tuning.

### Return vias

A via that changes layers needs a ground via near it for its return
current. kibends draws ground vias and the other vias toward each other, so
that a moving signal via keeps its return vias near and a moving stitching
via stays near the signal vias it serves.

The ground nets are those given with `--ground-nets`; without it, the nets
with copper zones on at least two inner layers; without those, `GND` if it
has zones, vias or pads; else there are none, and the run goes on without
attraction. A power net poured on two inner layers counts as ground: name
the ground nets with `--ground-nets` where that is wrong. When a selected
via may move, one line names the ground nets and where they came from, and
a net given with `--ground-nets` that has no copper gets a warning. A via
on a ground net is a ground via; a via on any other net, power nets
included, is a regular one. Every via takes part, selected or not, moving
or not, but two vias interact only where they share a layer.

A ground via and a regular via within `--attraction-range` of each other,
centre to centre, pull each other, and the pulls of all such vias add up.
Each still keeps its clearance from the other and is repelled by it as by
other copper. A via's tracks and the copper around it hold it well beyond
its clearance: with the defaults, a signal via on a straight route with a
return via 1.5 mm across it moves about 0.13 mm off the route toward it,
ending about 1.37 mm from it, and the route grows by a few micrometres. A
pair with nothing else near settles about 0.02 mm beyond its clearance,
since the default pull is the stronger at every distance past it; a lower
`--attraction`, such as 0.25, balances the two farther out, two 0.6 mm
vias with 0.2 mm clearance about 1.6 mm apart, centre to centre.

Ground vias of one net push each other apart within `--range` beyond their
rings and the clearance, so that stitching vias drawn toward a signal via
spread around it instead of bunching up; the hole-to-hole distance stays
their only limit. A selected grid of ground stitching vias closer than
that therefore spreads, and its vias move into gaps in it, as far as the
zones they connect to let them. Ground vias of two ground nets keep their
clearance and repel each other as other copper does.

A selected grid of ground stitching vias that signal routes pass through
moves out of their way, and the chains on their layers wait for it, since
moving vias tie those layers into one problem. On synthetic two-layer
planes with 29 to 47 stitching vias the routes ended up to a fifth
shorter than with the stitching vias left unselected, and the runs took
1.8 to 5.8 times as long, the most with the most vias. Once the routes
stop shortening, the vias may still creep toward their balance by tens
of micrometres a burst for little gain, so a problem's stall check
allows one more `--stall-tolerance` for each of its moving vias.

`--attraction 0` turns the pull and the ground vias' push off together:
ground nets are not looked for, `--ground-nets` is ignored, and vias of one
ground net leave each other alone, as vias of any other net do.

### Options

| Option | Default | Effect |
|---|---|---|
| `--repulsion WEIGHT` | 0.5 | Weight of keeping away from other copper, relative to length. 0 disables it, so the track pulls tight against obstacles at exactly the clearance. |
| `--range MM` | 1.0 | Distance beyond the required clearance at which repulsion fades to zero. Copper further away has no effect. |
| `--attraction WEIGHT` | 1.15 | Weight of ground vias and regular vias pulling each other, relative to length (see [Return vias](#return-vias)). More weight moves the vias farther toward each other against the copper and tracks around them. 0 turns the pull and the push between vias of one ground net off, and `--ground-nets` is then ignored. |
| `--attraction-range MM` | 3.0 | Centre distance at which the pull between a ground via and a regular via fades to zero. |
| `--ground-nets NET ...` | detected | The ground nets, instead of the nets with copper zones on at least two inner layers, or `GND`. |
| `--edge-clearance MM` | 0.3 | Clearance to the board edge. |
| `--hole-clearance MM` | 0.25 | Clearance between drilled holes and other nets' copper. A moving via's hole also keeps 0.25 mm from every other hole. |
| `--chord-tolerance MM` | 0.01 | How far the committed tracks may depart from the optimised curve; larger values usually give fewer, longer tracks, though beside copper a pair may keep more. |
| `--time-limit SECONDS` | none | Stop optimizing after this long and go on with the best result found so far: simplification and the commit run as usual, so a partial result can be inspected and undone in one step. The optimizer checks the limit after each L-BFGS iteration and ends the running burst there, so it runs over by about one L-BFGS iteration and the checks after the burst, and by the hold of a differential pair's section if one falls due when the time is up. |
| `--spacing MM ...` | 0.1 | Distances between a chain's vertices that each problem runs at, coarsest first; each hands over to the next as the problem settles. The last is the final spacing, at most 0.1, and one value runs at it alone. At most 8 values, each smaller than the one before. |
| `--settled-change MM` | half `--chord-tolerance` | A problem settles when a burst moves none of its tracks or vias by this much. A smaller move cannot change the committed tracks by more than simplification already may. |
| `--stall-tolerance MM` | 0.001 | A problem also settles when, over `--stall-window` of its bursts, its energy, in mm of track length, improves by less than this for each of its moving chains and vias, and its tracks' summed length falls by less than this. The energy's share counts per chain and via, so a problem and far-apart copies of it stall together: copies may run longer, never stop sooner. Raising it saves little time and lets some tracks end slightly longer. |
| `--stall-window N` | 3 | Bursts of a problem the stall check looks back over. |
| `--max-iterations N` | 200 | Outer iterations, each one burst of every problem not yet settled, after which the optimizer stops and goes on with the best result found. |
| `--burst-iterations N` | 50 | L-BFGS iterations per burst. Each burst resamples the tracks and checks every move, so on a large selection shorter bursts reach a given result sooner. Fewer shorten a run further, but at 25 a differential pair in the test scenarios settles visibly short of its smooth shape: `--stall-window` counts bursts, so with shorter bursts it looks back over less work. More get a little closer to the exact optimum at several times the cost. |
| `--lbfgs-ftol TOL` | 1e-9 | L-BFGS ends a burst early when the energy's relative reduction falls below this. Smaller values let bursts run to `--burst-iterations`; in the test scenarios they changed a scenario's summed track length by up to about 15 µm. Much larger values shorten runs but change the result a lot: 1e-6 moved some tracks by tenths of a millimetre in [an earlier measurement](https://github.com/mairas/kibends/issues/27). |
| `--lbfgs-gtol TOL` | 1e-10 | L-BFGS ends a burst early when the largest gradient component falls below this. It rarely ends one before `--lbfgs-ftol` or `--burst-iterations` do. |
| `--allow-via-in-pad` | | Let a moving via settle inside a pad of its own net. Without it a moving via may touch a pad of its net but never overlaps one; the pad stops it without pushing it. KiCad's DRC does not check via-in-pad, so this option is the only guard: use it only where the fabricator fills and caps vias in pads. Holes keep their clearances either way. |
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

Development needs a Rust toolchain with `clippy` and `rustfmt`.

```sh
./run sync       # also builds the Rust core, kibends._core
./run test       # Rust tests and fast Python tests, rebuilding first
./run test-all   # also the slow tests; run before committing
```
