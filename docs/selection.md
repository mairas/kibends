# What gets smoothed

kibends smooths what is selected in the PCB editor, or the whole board
when the panel runs with nothing selected (or the command line runs with
`--all`). This page describes how it turns that selection into routes and
which selected vias move with them.

## Chains

kibends works on chains. A chain is a connected piece of selected tracks
of one net and one width on one layer, its segments joined end to end. A
route that changes layers through a via is one chain on each layer. Each
chain is replaced by a new chain of straight tracks.

All chains of a run are smoothed together, on every layer, so neighbouring
chains make room for each other. Chains of different nets keep their
clearance to each other. Chains of one net neither push nor hold each
other, and may come close, overlap or cross.

A selected arc track is read as short straight pieces along the arc and
replaced by straight tracks like the rest of its chain. A locked arc stays
an arc.

Teardrops are ignored: KiCad redraws them when the zones are refilled.
Selected pads and other items are ignored.

## What stays where it is

- Locked tracks are never modified. A locked track in the middle of a
  selected route splits it into two chains.
- The tracks, arcs and vias of a selected group, such as a length-tuning
  pattern, stay as drawn, as do those of groups inside it. The tracks that
  touch the group end there.
- The two ends of each chain stay where they are, unless they end at a via
  that moves, or slide inside a pad or via (below).
- Every point where other copper of the chain's net touches it stays where
  it is: an unselected track, a copper shape, or the end of another
  selected chain.

## Pads and vias a chain touches

A chain keeps touching every pad and via of its net that it touched, but
it need not touch it in the same place:

- An end inside a pad or a via that does not move may slide anywhere
  inside it, a few micrometres within its outline.
- A chain that runs across a pad partway along keeps one corner inside the
  pad, free to slide there in the same way. The rest of the chain is free,
  including what ran inside the pad before.
- A point exactly on a pad's anchor or a via's centre stays there. KiCad
  drags only the track ends there along when you later move the footprint
  or via, so moving them would leave the tracks behind.

KiCad connects a track to a pad or via wherever their copper overlaps, so
these rules keep every connection.

## Junctions and T-ends

Where three or more selected tracks of one net meet at their ends, the
selection branches. It is split into chains that end at the junction. The
junction moves with them unless other copper of the net holds it, and the
branches meet at one point again in the result. Tracks of two widths that
meet end to end, as where a route narrows, branch the same way.

A track that ends on the middle of another, a T, branches there too: the
other track is split at the point nearest the end, and the end moves onto
it. If that is not clean, for example the track crosses the other instead
of ending on it, runs along it, or other copper is on its end, the end
keeps its place and the other track is held where they touch.

Tracks of two widths that overlap, as at a neck-down whose narrow track
ends inside the wide one's copper rather than on its end, stay separate
chains, held where one reaches the other.

## Tracks of one net that cross

Where selected tracks of one net overlap or cross away from their ends,
nothing holds them together, and the run may move them apart, even where
that overlap was their only connection. A stub drawn past the track it
crosses may come off it. Draw such a stub to end on the track, as a T, or
check the ratsnest after the run.

## Loops

A closed loop is cut where copper of its net touches it (a junction,
another track, a pad or a via) and smoothed from there. A loop that
nothing of its net touches is cut at its first point. A loop that
surrounds nothing shrinks: to nothing when it is held at one point, and
its tracks are deleted, or to a line when it is held at two.

## Vias

An unselected via stays where it is and holds the chain ends on its
layers.

Select a via together with its tracks to let it move with them. A selected
via moves when all of these hold:

- it is not locked;
- it is not on a net of a differential pair (see
  [Differential pairs](differential-pairs.md#vias));
- every copper item touching it, on a layer it spans, is the end of a
  selected chain or a zone of its net, and no other copper touches those
  ends;
- a selected chain ends at it, or a zone of its net reaches it.

One chain end is enough. A pad's fan-out via moves along its track toward
the pad until the pad stops it, unless the via may sit in the pad (Allow
via in pad in the panel, `--allow-via-in-pad` on the command line). Use
that only where the fabricator fills and caps vias in pads: KiCad's DRC
does not check via-in-pad.

Where two selected tracks meet at the centre of a moving via, the chain is
cut there into two chains that end at the via, and both move with it. A
selected track that passes over a via without a corner at its centre
counts as other copper touching it, and the via stays.

Through, blind, buried and micro vias move alike, each on the layers it
spans. Copper on a layer a via does not reach neither moves with it nor
holds it. Stacked vias of one net, such as a micro via on a buried via,
hold each other where they are.

### Why a selected via stays

A selected via that does not move stays because:

- it is locked;
- it is on a net of a differential pair;
- a track of a selected group ends at it;
- other copper touches it, or touches a chain end it would carry: an
  unselected track, a selected track that passes over it, a pad, a copper
  shape or another via of its net;
- it sits at the edge of a zone of its net, so moving it might lose the
  connection it has now;
- no selected track ends at it and no zone of its net reaches it.

The command line's `--verbose` names each such via and its reason.

### Stitching vias

A selected via that no selected track ends at, and that no copper but a
zone of its net touches, is a stitching via. It moves too, even when the
selection holds no tracks or only locked ones. It has no length to
shorten, so only the copper around it moves it: it backs away from other
nets' copper close by until the pushes balance. Ground vias are also drawn
toward signal vias near them ([Return vias](return-vias.md)).

### Vias in zones

A moved via stays inside the zones of its net that reached its ring, on
each layer where one did, and keeps the zone's clearance from other nets'
copper there, so that the next refill connects it again. kibends works
from the zone outlines and priorities, not from the current fill, which
may be out of date.

Thermal reliefs, island removal, hatched fills and a zone's minimum width
through narrow necks are not modelled. A via can end up where the refill
leaves no copper, in a gap of a hatch, or connected only through a narrow
neck. KiCad's DRC reports those.
