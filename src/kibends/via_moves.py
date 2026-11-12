"""Moving vias in the optimizer: the chain ends they carry, the fields they
make and their free distances; `MovingVias`, `free_distances` and
`newly_inside` are Rust's (their design is in `kibends_core::vias::moves`
and `kibends_core::vias::free`).

The optimizer (`kibends.optimize`) makes each moving via's centre a
variable of its problem and carries the chain ends it joins at their
offsets from it.

On a zone's layers a moving via keeps the zone's clearance from the
chains and vias moving with it.

A via's clearance is judged obstacle by obstacle (`newly_inside`) in the
optimizer's checks, so that a via inside one obstacle is still kept out of
the others: each member, each other via, and each obstacle of its static
copper and of each part of its stops, connected copper of one net on one
layer as one (`FreeDistances`).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from kibends import _core
from kibends.energy import DistanceField, FloatArray, ObstacleField
from kibends.neighbours import ChainInput
from kibends.obstacles import ChainField, core_field, core_via, core_zones
from kibends.seen import ViaSight
from kibends.snapshot import layer_ordinal
from kibends.vias import ViaInput

FreeDistances = _core.FreeDistances
"""The free distance of each via to each obstacle: each of what stands still
(`static`), and each member and via its sight finds (`by_obstacle`)."""

newly_inside = _core.newly_inside
"""Which of the (via, obstacle) pairs of what stands still for the vias and
of the members and vias they see, at `now`, are newly inside by more than
`tolerance`, or more than `slack` deeper inside than at `start`. Each
obstacle is judged by the via's nearest distance to it against that
distance at the start, so that a via inside one is still kept out of the
others. `start` holds every pair inside by more than `tolerance`."""


@dataclass(frozen=True)
class _End:
    """A chain end that moves with a via."""

    chain: int
    index: int
    """0 for the chain's first point, -1 for its last."""
    offset: FloatArray
    """From the via's centre to the end, kept as the via moves: an end
    joined off-centre stays where it overlaps the via's copper."""


Positions = Sequence[FloatArray]
"""Where each moving via is, in the order of the run's vias."""


class MovingVias:
    """The moving vias of a run: the chain ends they carry, and the fields
    they make and what they see, with the vias at the positions the caller
    holds (`kibends._core.MovingVias`; its design is in
    `kibends_core::vias::moves`).

    A via and a chain are near when the chain is on a layer the via spans
    and is of another net: each is then in the other's field. Copper of one
    net does not interact, whether the chain ends at the via or not.
    """

    def __init__(
        self,
        chains: Sequence[ChainInput],
        vias: Sequence[ViaInput],
        hole_clearance: float,
        hole_to_hole: float,
    ) -> None:
        self.chains = chains
        self.vias = tuple(vias)
        self.hole_clearance = hole_clearance
        self.hole_to_hole = hole_to_hole
        lines = [c.centreline for c in chains]
        self.core = _core.MovingVias(
            [c.net for c in lines],
            [layer_ordinal(c.layer) for c in chains],
            np.array([c.width for c in lines], dtype=float),
            np.array([c.clearance for c in lines], dtype=float),
            np.array([c.points[0] for c in lines], dtype=float).reshape(-1, 2),
            np.array([c.points[-1] for c in lines], dtype=float).reshape(-1, 2),
            [core_via(v.via) for v in self.vias],
            [v.clearance for v in self.vias],
            [core_zones(v.zones) for v in self.vias],
            [[(end.chain, end.last) for end in v.ends] for v in self.vias],
            hole_clearance,
            hole_to_hole,
        )
        self.ends = [
            tuple(
                _End(chain, -1 if last else 0, np.array(offset, dtype=float))
                for chain, last, offset in self.core.ends(v)
            )
            for v in range(len(self.vias))
        ]

    def via_obstacles_for_chain(
        self, i: int, positions: Positions
    ) -> ObstacleField | None:
        """The vias near chain i at `positions` as obstacles to it; None when
        none is one."""
        if not self.vias:
            return None
        at = np.array(positions, dtype=float).reshape(-1, 2)
        found = self.core.via_obstacles_for_chain(i, at)
        return None if found is None else ChainField.of(found)

    def apart(self, v: int, u: int) -> float | None:
        """The distance via v's centre keeps from via u's; None if no rule
        applies. It is the same as u's from v's, the larger of both vias'
        zone distances."""
        return self.core.apart(v, u)

    def rings(self, v: int) -> dict[int, float]:
        """The chains near via v, in what it sees and seeing it, and how far
        each keeps its centreline from v's centre, by v's zones on that
        chain's layer."""
        return dict(self.core.rings(v))


StaticParts = Sequence[DistanceField | None]
"""What stands still for a via, in parts."""


def free_distances(
    statics: Sequence[StaticParts],
    sight: ViaSight,
    polylines: Sequence[FloatArray],
    positions: Positions,
    below: float = math.inf,
) -> FreeDistances:
    """The free distance of each via to each obstacle nearer than `below`:
    to each of its `statics`, each obstacle of a part as a clearance check
    judges it, and to each of the chains or members at `polylines` and the
    vias at `positions` that `sight` finds. Without a finite `below`, of
    each part of its statics at least the nearest obstacle."""
    return _core.free_distances(
        [[None if f is None else core_field(f) for f in parts] for parts in statics],
        sight,
        [np.asarray(p, dtype=float).reshape(-1, 2) for p in polylines],
        np.array(positions, dtype=float).reshape(-1, 2),
        below,
    )
