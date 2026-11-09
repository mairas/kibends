"""Moving vias in the optimizer: the chain ends they carry, the fields they
make and what they see.

The optimizer (`kibends.optimize`) makes each moving via's centre a
variable of its problem and carries the chain ends it joins at their
offsets from it.

On a zone's layers a moving via keeps the zone's clearance from the
chains and vias moving with it. A via that starts inside its clearance
stays, and is then seen as an unselected via (`MovingVias.distances`,
`MovingVias.standing`).
"""

import math
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace

import numpy as np

from kibends.bursts import (
    point_distance,
)
from kibends.energy import (
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
from kibends.seen import ViaSight
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
        starts = self.starting_positions()
        self.ends = [
            tuple(
                _End(
                    end.chain,
                    -1 if end.last else 0,
                    chains[end.chain].centreline.points[-1 if end.last else 0]
                    - starts[k],
                )
                for end in via.ends
            )
            for k, via in enumerate(self.vias)
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

    def standing(self, still: Collection[int]) -> MovingVias:
        """These vias with those in `still`, which stay where they are, as
        unselected vias: without the zones they would keep the clearance of
        if they moved: standing still, they connect as they do now."""
        return MovingVias(
            self.chains,
            [
                replace(v, zones=()) if k in still else v
                for k, v in enumerate(self.vias)
            ],
            self.hole_clearance,
            self.hole_to_hole,
        )

    def starting_positions(self) -> list[FloatArray]:
        return [np.array(v.via.position, dtype=float) for v in self.vias]

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

    def apart(self, v: int, u: int, *, own_zones_only: bool = False) -> float | None:
        """The distance via v's centre keeps from via u's; None if no rule
        applies. It is the same as u's from v's, the larger of both vias'
        zone distances, unless `own_zones_only` counts v's zones alone."""
        return via_via_inflation(
            self.vias[v].via,
            self.vias[u].via,
            self.clearance_of,
            hole_clearance=self.hole_clearance,
            hole_to_hole=self.hole_to_hole,
            zones=self.vias[v].zones,
            other_zones=() if own_zones_only else self.vias[u].zones,
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

    def distances(
        self, positions: Positions, polylines: Sequence[FloatArray]
    ) -> list[float]:
        """Each via's free distance (`free_distances`) with the chains near
        it at `polylines` and the other vias at `positions`, by its own zones
        alone: two vias keep the larger of their zone distances, so when a
        via starts inside the other's, the other starts inside it too,
        stays, and is seen without its zones (`standing`)."""
        count = len(self.vias)
        apart = [
            {
                u: d
                for u in range(count)
                if u != v and (d := self.apart(v, u, own_zones_only=True)) is not None
            }
            for v in range(count)
        ]
        sight = ViaSight(
            range(count),
            [self.rings(v) for v in range(count)],
            apart,
            len(self.chains),
        )
        return free_distances(self.vias, sight, polylines, positions)


def free_distances(
    vias: Sequence[ViaInput],
    sight: ViaSight,
    polylines: Sequence[FloatArray],
    positions: Positions,
) -> list[float]:
    """The free distance of each via `sight` looks from, in order: to its
    static copper and stops (`ViaInput.limits`), and to what `sight` finds
    of the chains or members at `polylines` and the vias at `positions`."""
    cross = sight.nearest(polylines, positions, math.inf).distance
    return [
        min(point_distance(vias[k].limits, positions[k]), float(d))
        for k, d in zip(map(int, sight.looking), cross, strict=True)
    ]
