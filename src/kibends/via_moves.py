"""Moving vias in the optimizer: the chain ends they carry, and the fields
they make and see.

The optimizer (`kibends.optimize`) makes each moving via's centre a
variable of its problem and carries the chain ends it joins at their
offsets from it.
"""

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
    CompositeField,
    chain_obstacles_for_via,
    moving_via_obstacles,
    via_chain_inflation,
    via_obstacles_for_via,
    via_via_inflation,
    zones_on,
)
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
    they make and see with the vias at the positions the caller holds.

    A via and a chain are near when the chain is on a layer the via spans
    and does not end at it: each is then in the other's field, a chain of
    the via's net at touching distance, so that the via makes no new
    connection.
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
        self.incident = [frozenset(end.chain for end in ends) for ends in self.ends]
        self._zones_by_id = {v.via.id: v.zones for v in self.vias}
        clearances = {c.centreline.net: c.centreline.clearance for c in chains}
        clearances |= {v.via.net: v.clearance for v in self.vias}
        self.clearance_of: Callable[[str], float] = clearances.__getitem__
        self._chains_near = [
            [
                i
                for i, chain in enumerate(chains)
                if i not in self.incident[v] and via.via.reaches(chain.layer)
            ]
            for v, via in enumerate(self.vias)
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

    def chains_near(self, v: int) -> list[int]:
        """The chains near via v: in its field, and it in theirs."""
        return self._chains_near[v]

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

    def field(
        self,
        v: int,
        positions: Positions,
        polylines: Sequence[FloatArray],
    ) -> ObstacleField:
        """Via v's field with the chains near it at `polylines` and the other
        vias at `positions`, by v's own zones alone: two vias keep the
        larger of their zone distances, so when v starts inside the other's,
        the other starts inside it too, stays, and is seen without its
        zones (`standing`)."""
        via, zones = self.vias[v].via, self.vias[v].zones
        layers = dict.fromkeys(self.chains[i].layer for i in self._chains_near[v])
        parts: list[ObstacleField] = [
            self.vias[v].limits,
            *(
                chain_obstacles_for_via(
                    via,
                    [
                        replace(self.chains[i].centreline, points=polylines[i])
                        for i in self._chains_near[v]
                        if self.chains[i].layer == layer
                    ],
                    self.clearance_of,
                    hole_clearance=self.hole_clearance,
                    zones=zones_on(zones, [layer]),
                )
                for layer in layers
            ),
        ]
        moving = via_obstacles_for_via(
            via,
            [self.at(u, positions) for u in range(len(self.vias)) if u != v],
            self.clearance_of,
            hole_clearance=self.hole_clearance,
            hole_to_hole=self.hole_to_hole,
            zones=zones,
        )
        if moving is not None:
            parts.append(moving)
        return CompositeField(parts)

    def ring(self, v: int, i: int) -> float:
        """The distance chain i's centreline keeps from via v's centre."""
        return via_chain_inflation(
            self.vias[v].via,
            self.chains[i].centreline,
            self.clearance_of,
            hole_clearance=self.hole_clearance,
            zones=zones_on(self.vias[v].zones, [self.chains[i].layer]),
        )

    def apart(self, v: int, u: int) -> float | None:
        """The distance via v's centre keeps from via u's, the same as u's
        from v's; None if no rule applies."""
        return via_via_inflation(
            self.vias[v].via,
            self.vias[u].via,
            self.clearance_of,
            hole_clearance=self.hole_clearance,
            hole_to_hole=self.hole_to_hole,
            zones=self.vias[v].zones,
            other_zones=self.vias[u].zones,
        )

    def distance(
        self, v: int, positions: Positions, polylines: Sequence[FloatArray]
    ) -> float:
        return point_distance(self.field(v, positions, polylines), positions[v])
