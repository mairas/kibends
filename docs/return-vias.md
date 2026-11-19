# Return vias

A signal via that changes layers needs a ground via near it to carry its
return current. kibends draws ground vias and signal vias toward each
other, so that a moving signal via keeps its return vias near and a
moving stitching via stays near the signal vias it serves. The barrel of
a plated through-hole pad joins the layers as a via does, so pads take
part too.

## Ground nets

kibends finds the ground nets on its own:

1. the nets with copper zones on at least two inner layers;
2. otherwise `GND`, if it has zones, vias or pads;
3. otherwise none, and vias are not drawn together.

A power net poured on two inner layers counts as ground by the first
rule. Where that is wrong, type the ground nets in the panel's Ground nets
field, or give them with `--ground-nets` on the command line. The panel
shows the nets it found and how it found them.

A via on a ground net is a ground via. A via on any other net, power nets
included, is a signal via here. Every via takes part, selected or not, but
only selected vias that may move do move, and two vias affect each other
only where they share a layer.

Plated through-hole pads take part as well, and never move. A pad on a
ground net is a return for signal vias. A pad on any other net draws
ground vias only where it changes layers: when tracks or zones of its net
connect to it on at least two copper layers. A pin with a track on one
layer alone leaves ground vias where they are. SMD pads, unplated holes
and pads with no net take no part.

## How vias are drawn together

A ground via and a signal via within the attraction range of each other,
centre to centre, pull each other (3 mm by default). Each still keeps its
clearance from the other. The tracks and copper around a via hold it well
beyond its clearance: with the defaults, a signal via on a straight route
with a return via 1.5 mm across moves about 0.13 mm toward it, and the
route grows by a few micrometres. Two vias with nothing else near settle
just beyond their clearance.

A pad pulls a via of the other kind in the same way, but the distance runs
from the via's centre to the wall of the pad's hole, so a large pin
reaches as far beyond its hole as a via does beyond its centre. A pad only
pulls: the via keeps its clearance from the pad's copper as from any other
copper, and one drawn toward a pad may settle pressed against that
clearance instead of at a balance.

Raise Attraction to draw vias closer against the copper and tracks around
them, or lower it to let them balance farther apart.

## Ground vias spread apart

Ground vias of one net push each other apart, within the repulsion range
beyond their rings and clearance. Stitching vias drawn toward a signal via
then spread around it instead of bunching up. A selected grid of ground
stitching vias that is denser than that spreads, as far as the zones it
connects to allow, and moves out of the way of signal routes that pass
through it. Selecting the stitching vias lets the routes become shorter,
but the run takes longer, the more so the more vias move.

A ground pad pushes the ground vias of its net away in the same way,
from the wall of its hole. A pad of another ground net is only copper to
keep clear of.

Ground vias of two different ground nets keep their clearance and repel
each other like other copper.

## Turning it off

Attraction 0 turns off both the pull and the push between ground vias,
and the pads' pull and push with them. Ground nets are then not looked
for, and the Ground nets field (`--ground-nets`) is ignored.
