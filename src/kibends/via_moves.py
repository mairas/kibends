"""Moving vias in the optimizer: the chain ends they carry, the fields they
make and what they see.

The optimizer (`kibends.optimize`) makes each moving via's centre a
variable of its problem and carries the chain ends it joins at their
offsets from it.

On a zone's layers a moving via keeps the zone's clearance from the
chains and vias moving with it.

A via's clearance is judged obstacle by obstacle (`newly_inside`) in the
optimizer's checks, so that a via inside one obstacle is still kept out of
the others: each member, each other via, and each obstacle of its static
copper and of each part of its stops, connected copper of one net on one
layer as one (`StaticFree`).
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
    IntArray,
    ObstacleField,
)
from kibends.neighbours import ChainInput
from kibends.obstacles import (
    moving_via_obstacles,
    static_distances,
    static_parts,
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


class StaticFree(NamedTuple):
    """(via, obstacle) pairs of what stands still for each via, in
    ascending order of `key`, and each via's exact distance to its
    obstacle. A via's obstacles are those of each of its parts, numbered
    one part after another: of a field that gives edges
    (`obstacles.static_parts`), each as a clearance check judges it, its
    connected copper of one net on one layer or other item
    (`obstacles.StaticEdges.clearance_obstacles`), and the whole of one
    that does not."""

    key: IntArray
    """Each pair's via times the most obstacles any via has, plus its
    obstacle."""
    via: IntArray
    distance: FloatArray


class FreeDistances(NamedTuple):
    """The free distance of each via to each obstacle: each of what stands
    still, and each member and via its sight finds."""

    static: StaticFree
    by_obstacle: ObstacleDistances


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
    to each of its `statics` (`StaticFree`), and to each of the chains or
    members at `polylines` and the vias at `positions` that `sight` finds.
    Without a finite `below`, of each part of its statics at least the
    nearest obstacle."""
    found = [
        _static_free(fields, at, below)
        for fields, at in zip(statics, positions, strict=True)
    ]
    stride = max((f.obstacles for f in found), default=0)
    none = np.empty(0, dtype=np.intp)
    via = np.concatenate(
        [
            none,
            *(np.full(len(f.obstacle), v, dtype=np.intp) for v, f in enumerate(found)),
        ]
    )
    obstacle = np.concatenate([none, *(f.obstacle for f in found)])
    distance = np.concatenate([np.empty(0), *(f.distance for f in found)])
    return FreeDistances(
        StaticFree(via * stride + obstacle, via, distance),
        sight.obstacle_distances(polylines, positions, below),
    )


class _ViaStatic(NamedTuple):
    """The obstacles of what stands still for one via that are near it, in
    ascending order, their distances, and how many obstacles there are in
    all."""

    obstacle: IntArray
    distance: FloatArray
    obstacles: int


def _static_free(fields: StaticParts, at: FloatArray, below: float) -> _ViaStatic:
    """The obstacles of `fields` nearer than `below` to a via at `at`."""
    obstacles: list[IntArray] = [np.empty(0, dtype=np.intp)]
    distances: list[FloatArray] = [np.empty(0)]
    first = 0
    for field in fields:
        if field is None:
            continue
        edges = static_parts(field)
        if edges is None:
            obstacle = np.zeros(1, dtype=np.intp)
            distance = np.array([point_distance(field, at)])
            count = 1
        else:
            found = static_distances(edges, at[None], below)
            obstacle, distance = found.obstacle, found.distance
            count = sum(e.obstacle_count for e in edges)
        near = distance < below
        obstacles.append(obstacle[near] + first)
        distances.append(distance[near])
        first += count
    return _ViaStatic(
        np.concatenate(obstacles),
        np.concatenate(distances),
        first,
    )


def deeper_inside(
    now_key: IntArray,
    now: FloatArray,
    start_key: IntArray,
    start: FloatArray,
    tolerance: float,
    slack: float,
) -> BoolArray:
    """Which pairs, by their keys `now_key` at distances `now`, are newly
    inside by more than `tolerance`, or more than `slack` deeper inside
    than the pair of the same key among `start_key` at `start`, ascending."""
    at = np.searchsorted(start_key, now_key)
    found = at < len(start_key)
    found[found] = start_key[at[found]] == now_key[found]
    before = np.full(len(at), np.inf)
    before[found] = start[at[found]]
    return now < np.minimum(before - slack, -tolerance)


def newly_inside(
    now: FreeDistances, start: FreeDistances, tolerance: float, slack: float = 0.0
) -> tuple[BoolArray, BoolArray]:
    """Which of the (via, obstacle) pairs of what stands still for the vias
    and of the members and vias they see, at `now`, are newly inside by
    more than `tolerance`, or more than `slack` deeper inside than at
    `start`. Each obstacle is judged by the via's nearest distance to it
    against that distance at the start, so that a via inside one is still
    kept out of the others; an edge of the obstacle other than its nearest
    may move deeper while the nearest stays the deepest. `start` holds
    every pair inside by more than `tolerance`."""
    return (
        deeper_inside(
            now.static.key,
            now.static.distance,
            start.static.key,
            start.static.distance,
            tolerance,
            slack,
        ),
        deeper_inside(
            now.by_obstacle.key,
            now.by_obstacle.distance,
            start.by_obstacle.key,
            start.by_obstacle.distance,
            tolerance,
            slack,
        ),
    )
