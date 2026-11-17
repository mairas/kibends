# Differential pairs

kibends smooths the two tracks of a differential pair together, as one
route at a constant gap, wherever they run side by side.

## Which tracks pair

Two nets pair as they do in KiCad's router: a common name with a `+`/`-`
or `P`/`N` suffix, trailing digits and underscores ignored, such as
`USB_D+` and `USB_D-`. A net counts as part of a pair only when the net
with the opposite suffix exists on the board: a via on `VIN` moves when
the board has no `VIP`.

Two selected chains of a pair's nets on one layer are smoothed as a pair
where they run coupled: side by side and parallel at a near-constant gap,
however short the stretch, as in KiCad's own coupling check. Either track
may be drawn in any number of segments, such as an arc split finely or the
straights of a length-tuning pattern. Each coupled stretch is a coupled
section; a pair that splits around a pad or via and rejoins has several.
Each layer is paired on its own.

The rest of each chain, such as its fan-outs to pads or vias, is smoothed
as an ordinary chain. kibends does not route uncoupled fan-outs into
coupling.

## What a pair keeps

- **Its gap.** Both tracks follow the smoothed centreline at the gap they
  were routed at, whatever the netclass sets.
- **How it meets its fan-outs.** Each coupled section meets the rest of
  its chains where and in the direction it did.
- **Its bends.** A section never bends tighter than its inner track
  allows. A section whose routing already bends tighter keeps its tracks
  as drawn, and so does one whose corner the copper around it keeps from
  rounding, such as a pad inside a sharp corner.
- **Points other copper fixes.** Same-net copper that touches a coupled
  section between its ends, such as a pad the pair runs through or a stub
  off one track, fixes the pair there. The rest of the section is still
  smoothed.

Bending a pair shortens its inner track, so the skew between the two
tracks changes. kibends does not correct it; KiCad's length tuner does.

## Clearance between the two tracks

The selected tracks of a pair's two nets do not keep clearance from each
other while the run optimizes them, at the fan-outs and wherever the two
run apart, such as around a via. Nothing keeps one net's track from being
pulled across the other's there, and kibends does not check the gap of
the committed tracks. KiCad's DRC reports either. The pair's vias, and the
unselected copper of either net, still keep clearance from the other
net's tracks.

## Stubs and tuning patterns

A selected stub, such as a test pad's or an ESD diode's, splits the track
it branches off. The other track is cut opposite the stub, and each part
pairs with the part beside it. The stub's junction and the points opposite
it stay where they are. Near a corner, the short stretch between the
corner and the stub stays as drawn.

A length-tuning pattern is a group, so its tracks stay as drawn (see
[What gets smoothed](selection.md#what-stays-where-it-is)). Where the
pattern's straight runs beside the other track, that track keeps its
routed gap, and the pair on either side of the pattern pairs up to it as
at a stub.

## Warnings

A run warns when two chains whose nets pair run coupled but have different
widths. They are smoothed as separate chains.

## Vias

A pair's vias do not move. A selected via on either net of a pair stays
where it is, whether or not the selection includes the pair's tracks, and
the tracks that end at it keep their ends there. Moving the two nets' vias
one by one would break the symmetry of the pair's via layout and its
length tuning.
