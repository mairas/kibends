"""Moving vias in the optimizer: the chain ends they carry, the fields they
make and see, and the contact between a via and a chain pressing on it.

The optimizer (`kibends.optimize`) makes each moving via's centre a
variable of its problem and carries the chain ends it joins at their
offsets from it. A chain that presses on a via without ending at it is
left out of the via's bubble and the via out of the chain's, so that the
two move together instead of creeping after each other by the clearance
hinge's give; a check that neither passed through the other (`Contact`)
replaces them.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from kibends.bursts import (
    INSIDE_TOLERANCE,
    point_distance,
)
from kibends.energy import (
    BoolArray,
    FloatArray,
    ObstacleField,
)
from kibends.neighbours import ChainInput
from kibends.obstacles import (
    ChainField,
    CompositeField,
    chain_obstacles_for_via,
    moving_via_obstacles,
    via_chain_inflation,
    via_obstacles_for_via,
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
        # A moving via joins chains of its net, so every net asked about has one.
        clearances = {c.centreline.net: c.centreline.clearance for c in chains}
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
        )

    def field(
        self,
        v: int,
        positions: Positions,
        polylines: Sequence[FloatArray],
    ) -> ObstacleField:
        """Via v's field with the chains near it at `polylines` and the other
        vias at `positions`."""
        via = self.vias[v].via
        others = [
            replace(self.chains[i].centreline, points=polylines[i])
            for i in self._chains_near[v]
        ]
        parts: list[ObstacleField] = [
            self.vias[v].field,
            chain_obstacles_for_via(
                via, others, self.clearance_of, hole_clearance=self.hole_clearance
            ),
        ]
        moving = via_obstacles_for_via(
            via,
            [self.at(u, positions) for u in range(len(self.vias)) if u != v],
            self.clearance_of,
            hole_clearance=self.hole_clearance,
            hole_to_hole=self.hole_to_hole,
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
        )

    def distance(
        self, v: int, positions: Positions, polylines: Sequence[FloatArray]
    ) -> float:
        return point_distance(self.field(v, positions, polylines), positions[v])


def presses(
    position: FloatArray, points: FloatArray, ring: float, limit: float
) -> bool:
    """Whether a chain at `points`, its centreline keeping `ring` from the
    centre of a via at `position`, presses on it: clear of it, yet within
    `limit`. A chain with an edge as long as the ring is wide does not, as
    the check that neither passed through the other (`Contact.kept`) relies
    on the via not fitting between two of its vertices."""
    if np.linalg.norm(np.diff(points, axis=0), axis=1).max() >= 2 * ring:
        return False
    gap = ChainField(
        points[:-1], points[1:], np.full(len(points) - 1, ring)
    ).signed_distance(position[None])[0][0]
    return 0.0 <= gap < limit


@dataclass(frozen=True)
class Contact:
    """A chain and a via in contact as a burst found them, to tell whether
    either passed through the other."""

    via: FloatArray
    points: FloatArray
    ring: float
    edges_clear: BoolArray
    """Which edges were clear of the via's ring."""
    vertices_clear: BoolArray
    """Which vertices were clear of it."""

    @staticmethod
    def before(via: FloatArray, points: FloatArray, ring: float) -> Contact:
        disc = _ring_field(via, ring)
        return Contact(
            via,
            points,
            ring,
            disc.segments_clear(points[:-1], points[1:], INSIDE_TOLERANCE),
            disc.segments_clear(points, points, INSIDE_TOLERANCE),
        )

    def kept(self, via: FloatArray, points: FloatArray) -> bool:
        """Whether neither passed through the other on the way to `via` and
        `points`, both moving in straight lines: each vertex clear of the
        ring before stays clear of it all the way, as seen from the via, and
        so does each edge clear before at the end. The via cannot pass
        between two vertices only while the edge between them is shorter
        than the ring is wide, so an edge stretched that far counts as
        passed through."""
        if np.linalg.norm(np.diff(points, axis=0), axis=1).max() >= 2 * self.ring:
            return False
        clear = self.vertices_clear
        paths = _ring_field(np.zeros(2), self.ring).segments_clear(
            self.points[clear] - self.via, points[clear] - via, INSIDE_TOLERANCE
        )
        edges = _ring_field(via, self.ring).segments_clear(
            points[:-1], points[1:], INSIDE_TOLERANCE
        )
        return bool(paths.all() and np.all(edges | ~self.edges_clear))


def _ring_field(centre: FloatArray, ring: float) -> ChainField:
    """The disc of radius `ring` around `centre`, as a field."""
    return ChainField(centre[None], centre[None], np.array([ring]))
