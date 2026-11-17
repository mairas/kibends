# Clearances

kibends keeps the clearances it can read. Everything else is left to
KiCad's DRC, which you should run after every run.

## What kibends keeps

- The netclass clearances, including pad and footprint clearance
  overrides.
- The minimum copper to edge clearance, minimum hole clearance and minimum
  hole to hole from the project's Board Setup.
- Rule areas that keep out tracks or vias. Moving vias stay out of them
  too.
- For a moving via, the outlines and clearances of the zones it connects
  to (see [Vias in zones](selection.md#vias-in-zones)).

A moving via keeps its clearances on every copper layer it spans, also on
layers where no selected track is. Two vias keep their clearances and
hole-to-hole distance from each other only where their spans overlap.

## Where the values come from

KiCad's API does not give access to the design rules, so kibends reads the
three Board Setup values from the project file (`.kicad_pro`) as KiCad
last saved it. Save the board after changing them.

| Rule | Without a project value |
|---|---|
| Copper to edge clearance | 0.3 mm |
| Hole clearance | 0.25 mm |
| Hole to hole | 0.25 mm |

The command line's `--edge-clearance`, `--hole-clearance` and
`--hole-to-hole` override the project file. A project value below the
default is used, as KiCad's DRC uses it too, with a warning naming the
option that overrides it: many project files hold 0 for a rule nobody
set.

## The start check

A run checks the selected copper as drawn before it changes anything. If
any of it is already inside a netclass, edge, hole or hole-to-hole
clearance, the run stops,
commits nothing and lists each place: the chain or via, what it is too
close to, how far apart they are, what is required, and where. Move the
copper apart or relax the rules, then run again. A moving via that starts
inside its zone's clearance is not refused: the run moves it out.

Positions are in millimetres from the board's grid origin, with y
pointing down, as the PCB editor shows them with the grid origin chosen
under Preferences > PCB Editor > Origins & Axes and neither axis
inverted.

The check allows:

- a chain end on a pad or via of its own net;
- copper within a nanometre or two of a clearance, which KiCad's grid
  rounding leaves;
- a selected arc that bulges into a clearance, as long as the straight
  pieces kibends replaces it with are clear.

A hole that is not round, such as a slot, is not checked; KiCad's DRC
judges it. Arcs on the board edge and in copper graphics are judged by
short straight pieces along them, so copper inside their bend at exactly
the clearance may be refused.

Once a run has started, it never moves copper into a clearance, so a run
that starts clean ends clean by these rules.

## What only KiCad's DRC checks

- Custom DRC rules (the project's `.kicad_dru`). A custom rule stricter
  than the netclass clearances, such as a larger clearance for vias, can
  be broken by the smoothed tracks.
- Board minimums from Board Setup other than the three above.
- Rule areas' keepouts of pads, copper pours and footprints.
- Stricter rules for blind, buried or micro vias.
- Zones: kibends does not refill them. Thermal reliefs, island removal,
  hatched fills and minimum width through narrow necks are not modelled.
- The gap between the two tracks of a differential pair (see
  [Differential pairs](differential-pairs.md#clearance-between-the-two-tracks)).

KiCad's DRC does not check via-in-pad at all. kibends keeps a moving via
out of the pads of its net unless you allow it (see
[Vias](selection.md#vias)).
