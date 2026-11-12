"""What the members and moving vias of a problem see of each other, searched
for all of them at once; Rust's (`kibends._core`; the design is in
`kibends_core::seen`, `kibends_core::neighbours` and
`kibends_core::vias::sight`).

A member sees the edges the others show (`members.shown`) by a rule with
sparse exceptions (`Seen`): every edge, on its layer, of a track of a net
it does not ignore, and the vias near it with their rings. One tree per
layer answers every member on it: the smooth distance for the energy, the
nearest piece for the checks (`Seen.distances`), and whether a segment
enters one (`Seen.segments_clear`, `EdgesClear`). A `NeighbourList` keeps a
layer's pairs between evaluations. A moving via sees whole members and the
other vias (`ViaSight`).
"""

from kibends import _core

Seen = _core.Seen
EdgesClear = _core.EdgesClear
SmoothAnswer = _core.SmoothAnswer
NeighbourList = _core.NeighbourList
ViaSight = _core.ViaSight
ViaSmooth = _core.ViaSmooth
ObstacleDistances = _core.ObstacleDistances

SKIN_LEAST = _core.SKIN_LEAST
"""The skin each layer of a `NeighbourList` starts with, mm."""
SKIN_MOST = _core.SKIN_MOST
"""The widest skin a layer of a `NeighbourList` takes, mm."""
STATIC_SKIN = _core.STATIC_SKIN
"""The skin of every asker of a static list, mm."""
