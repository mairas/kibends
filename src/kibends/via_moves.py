"""Moving vias in the optimizer: the chain ends they carry, the fields they
make and what they see.

The optimizer (`kibends.optimize`) makes each moving via's centre a
variable of its problem and carries the chain ends it joins at their
offsets from it.

On a zone's layers a moving via keeps the zone's clearance from the
chains and vias moving with it.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from kibends.bursts import (
    point_distance,
)
from kibends.energy import (
    BoolArray,
    DistanceField,
    FloatArray,
    ObstacleField,
)
from kibends.neighbours import ChainInput
from kibends.obstacles import (
    moving_via_obstacles,
    via_chain_inflation,
    via_via_inflation,
    zones_on,
)
from kibends.seen import ObstacleDistances, ViaSight
from kibends.snapshot import Via
from kibends.vias import ViaInput


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
    holds.

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
        self.ends = [
            tuple(
                _End(
                    end.chain,
                    -1 if end.last else 0,
                    chains[end.chain].centreline.points[-1 if end.last else 0]
                    - np.array(via.via.position, dtype=float),
                )
                for end in via.ends
            )
            for via in self.vias
        ]
        self._zones_by_id = {v.via.id: v.zones for v in self.vias}
        clearances = {c.centreline.net: c.centreline.clearance for c in chains}
        clearances |= {v.via.net: v.clearance for v in self.vias}
        self.clearance_of: Callable[[str], float] = clearances.__getitem__
        self._chains_near = [
            [
                i
                for i, chain in enumerate(chains)
                if chain.centreline.net != via.via.net and via.via.reaches(chain.layer)
            ]
            for via in self.vias
        ]
        self._vias_near = [
            [v for v, near in enumerate(self._chains_near) if i in near]
            for i in range(len(chains))
        ]

    def at(self, v: int, positions: Positions) -> Via:
        x, y = positions[v]
        return replace(self.vias[v].via, position=(float(x), float(y)))

    def via_obstacles_for_chain(
        self, i: int, positions: Positions
    ) -> ObstacleField | None:
        """The vias near chain i at `positions` as obstacles to it; None when
        none is one."""
        if not self.vias:
            return None
        c = self.chains[i].centreline
        return moving_via_obstacles(
            self.chains[i].layer,
            c.net,
            c.width,
            c.clearance,
            [self.at(v, positions) for v in self._vias_near[i]],
            self.clearance_of,
            hole_clearance=self.hole_clearance,
            zones=self._zones_by_id,
        )

    def apart(self, v: int, u: int) -> float | None:
        """The distance via v's centre keeps from via u's; None if no rule
        applies. It is the same as u's from v's, the larger of both vias'
        zone distances."""
        return via_via_inflation(
            self.vias[v].via,
            self.vias[u].via,
            self.clearance_of,
            hole_clearance=self.hole_clearance,
            hole_to_hole=self.hole_to_hole,
            zones=self.vias[v].zones,
            other_zones=self.vias[u].zones,
        )

    def rings(self, v: int) -> dict[int, float]:
        """The chains near via v, in what it sees and seeing it, and how far
        each keeps its centreline from v's centre, by v's zones on that
        chain's layer."""
        via, zones = self.vias[v].via, self.vias[v].zones
        rings: dict[int, float] = {}
        for i in self._chains_near[v]:
            ring = via_chain_inflation(
                via,
                self.chains[i].centreline,
                self.clearance_of,
                hole_clearance=self.hole_clearance,
                zones=zones_on(zones, [self.chains[i].layer]),
            )
            assert ring is not None
            rings[i] = ring
        return rings


class FreeDistances(NamedTuple):
    """The free distance of each via to each obstacle: each part of what
    stands still, and each member and via its sight finds."""

    static: FloatArray
    """(vias, parts): each via's, in order, to each part of what stands
    still for it, +inf where it has no such part."""
    by_obstacle: ObstacleDistances


StaticParts = Sequence[DistanceField | None]
"""What stands still for a via, in parts that are judged apart."""


def free_distances(
    statics: Sequence[StaticParts],
    sight: ViaSight,
    polylines: Sequence[FloatArray],
    positions: Positions,
    below: float = math.inf,
) -> FreeDistances:
    """The free distance of each via to each obstacle: to each of its
    `statics`, and to each of the chains or members at `polylines` and the
    vias at `positions` that `sight` finds nearer than `below`."""
    parts = max((len(p) for p in statics), default=0)
    static = np.full((len(positions), parts), np.inf)
    for row, fields, at in zip(static, statics, positions, strict=True):
        for k, field in enumerate(fields):
            if field is not None:
                row[k] = point_distance(field, at)
    return FreeDistances(static, sight.obstacle_distances(polylines, positions, below))


def newly_inside(
    now: FreeDistances, start: FreeDistances, tolerance: float
) -> tuple[BoolArray, BoolArray]:
    """Which vias are newly inside some part of what stands still for them
    by more than `tolerance`, or deeper inside it than at `start`, and which
    of the (via, obstacle) pairs of `now` are so. Each obstacle is judged by
    the via's nearest distance to it against that distance at the start, so
    that a via inside one is still kept out of the others; an edge of the
    obstacle other than its nearest may move deeper while the nearest stays
    the deepest. `start` holds every pair inside by more than `tolerance`."""
    static = (now.static < np.minimum(start.static, -tolerance)).any(axis=1)
    at = np.searchsorted(start.by_obstacle.key, now.by_obstacle.key)
    found = at < len(start.by_obstacle.key)
    found[found] = start.by_obstacle.key[at[found]] == now.by_obstacle.key[found]
    before = np.full(len(at), np.inf)
    before[found] = start.by_obstacle.distance[at[found]]
    return static, now.by_obstacle.distance < np.minimum(before, -tolerance)
