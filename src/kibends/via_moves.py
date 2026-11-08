"""Moving vias in the optimizer: their fields, and the burst of each via.

A via's burst moves its position and the free vertices of the chains that
end at it together, each inside its bubble, the chains' ends keeping their
offsets from its centre. A chain that presses on the via without ending at
it joins the burst too, so that the two move together instead of creeping
after each other by the clearance hinge's give; the two leave each other
out of their bubbles, and a check that neither passed through the other
replaces them.
"""

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from kibends.bursts import (
    INSIDE_TOLERANCE,
    OptimizerSettings,
    bounded_burst,
    half_side,
    point_distance,
    revert_until_clear,
)
from kibends.energy import (
    BoolArray,
    DistanceField,
    EnergyParams,
    FloatArray,
    ObstacleField,
    obstacle_energy,
    obstacle_integrand,
    obstacle_reach,
    point_energy,
    total_energy,
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


@dataclass(frozen=True)
class Carried:
    """A chain in a via's burst: its free vertices move with the via's
    position, and its ends at the via keep their offsets from it."""

    chain: int
    points: FloatArray
    free: BoolArray
    ends: list[int]
    """Indices of its vertices at the via."""
    offsets: FloatArray
    """From the via's centre to each of those vertices."""
    field: ObstacleField
    ring: float | None = None
    """For a chain in contact with the via, the distance between its
    centreline and the via's centre at clearance. The two are left out of
    each other's fields, and of each other's bubbles, so that they move
    together; the burst counts them against each other."""

    def at(self, via: FloatArray, free: FloatArray) -> FloatArray:
        """The polyline with the via at `via` and the free vertices, flat."""
        points = self.points.copy()
        points[self.free] = free.reshape(-1, 2)
        points[self.ends] = via + self.offsets
        return points


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
        params: EnergyParams,
        hole_clearance: float,
        hole_to_hole: float,
    ) -> None:
        self.chains = chains
        self.vias = tuple(vias)
        self.params = params
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

    def vias_near(self, i: int) -> list[int]:
        """The vias near chain i: in its field, and it in theirs."""
        return self._vias_near[i]

    def chains_near(self, v: int) -> list[int]:
        """The chains near via v: in its field, and it in theirs."""
        return self._chains_near[v]

    def via_obstacles_for_chain(
        self, i: int, positions: Positions, without: int | None = None
    ) -> ObstacleField | None:
        """The vias near chain i at `positions`, but via `without`, as
        obstacles to it; None when none is one."""
        if not self.vias:
            return None
        c = self.chains[i].centreline
        return moving_via_obstacles(
            self.chains[i].layer,
            c.net,
            c.width,
            c.clearance,
            [self.at(v, positions) for v in self._vias_near[i] if v != without],
            self.clearance_of,
            hole_clearance=self.hole_clearance,
        )

    def field(
        self,
        v: int,
        positions: Positions,
        polylines: Sequence[FloatArray],
        without: Collection[int] = (),
    ) -> ObstacleField:
        """Via v's field with the chains near it at `polylines`, but those in
        `without`, and the other vias at `positions`."""
        via = self.vias[v].via
        others = [
            replace(self.chains[i].centreline, points=polylines[i])
            for i in self._chains_near[v]
            if i not in without
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

    def contacts(
        self,
        v: int,
        positions: Positions,
        polylines: Sequence[FloatArray],
        moving: Collection[int],
        settings: OptimizerSettings,
    ) -> list[int]:
        """The moving chains near via v that it presses on at `positions`
        and `polylines`: clear of it, yet within reach of its energy terms.

        Each would hold the via to a step of half their gap, and itself to
        the same, so that the two creep after each other by the hinge's give
        in every outer iteration; in one burst they move together. A chain
        with an edge as long as the ring is wide is left out: the check that
        neither passed through the other relies on the via not fitting
        between two of its vertices.
        """
        at = positions[v][None]
        limit = obstacle_reach(self.params) + settings.bubble_tolerance
        contacts: list[int] = []
        for i in self._chains_near[v]:
            if i not in moving:
                continue
            points = polylines[i]
            ring = self.ring(v, i)
            if np.linalg.norm(np.diff(points, axis=0), axis=1).max() >= 2 * ring:
                continue
            gap = ChainField(
                points[:-1], points[1:], np.full(len(points) - 1, ring)
            ).signed_distance(at)[0][0]
            if 0.0 <= gap < limit:
                contacts.append(i)
        return contacts

    def carried(
        self,
        v: int,
        contacts: Sequence[int],
        polylines: Sequence[FloatArray],
        anchors: Sequence[BoolArray],
        fields: Mapping[int, ObstacleField],
    ) -> list[Carried]:
        """The chains via v's burst moves, at `polylines` with their
        `anchors` and in `fields`: those ending at it, then `contacts`."""
        carried: list[Carried] = []
        for i in sorted(self.incident[v]):
            points = polylines[i]
            ends = [e for e in self.ends[v] if e.chain == i]
            carried.append(
                Carried(
                    i,
                    points,
                    ~anchors[i],
                    [e.index % len(points) for e in ends],
                    np.array([e.offset for e in ends]),
                    fields[i],
                )
            )
        carried += [
            Carried(
                i,
                polylines[i],
                ~anchors[i],
                [],
                np.empty((0, 2)),
                fields[i],
                self.ring(v, i),
            )
            for i in contacts
        ]
        return carried

    def distance(
        self, v: int, positions: Positions, polylines: Sequence[FloatArray]
    ) -> float:
        return point_distance(self.field(v, positions, polylines), positions[v])

    def energy(
        self, v: int, positions: Positions, polylines: Sequence[FloatArray]
    ) -> float:
        """Via v's own obstacle terms, its ring standing for a track as long
        as it is wide."""
        return point_energy(
            positions[v],
            self.field(v, positions, polylines),
            self.params,
            self.vias[v].via.diameter,
        )[0]


@dataclass(frozen=True)
class ViaBurst:
    position: FloatArray
    rejected_edges: int
    polylines: dict[int, FloatArray]
    """The chains the burst moved as it left them, by index."""
    anchors: dict[int, BoolArray]
    """Their pinned and frozen vertices, by index."""


def via_burst(
    position: FloatArray,
    carried: Sequence[Carried],
    field: ObstacleField,
    diameter: float,
    params: EnergyParams,
    settings: OptimizerSettings,
) -> ViaBurst:
    """One bounded L-BFGS-B run on a via's position and its chains' free
    vertices together, retried with halved bounds where edges crossed.

    As in a chain's burst, every vertex moves inside its bubble. The via's
    bubble is free in its own field and, at each chain end it carries, in
    that chain's field. An edge that newly crosses an obstacle moves its
    ends back, the via with every end it carries if one of them is the
    via's. A chain in contact with the via is missing from its field and
    the via from the chain's, so the two may close on each other; where
    either passes through the other, both move back.
    """
    via_distance = min(
        point_distance(field, position),
        *(point_distance(c.field, p) for c in carried for p in c.points[c.ends]),
    )
    halves = [np.repeat(half_side(np.array([via_distance]), settings), 2)]
    slices: list[slice] = []
    offset = 2
    for c in carried:
        distance, _ = c.field.signed_distance(c.points[c.free])
        halves.append(np.repeat(half_side(distance, settings), 2))
        slices.append(slice(offset, offset + 2 * int(c.free.sum())))
        offset = slices[-1].stop
    start = np.concatenate([position, *(c.points[c.free].ravel() for c in carried)])
    clear_before = [
        c.field.segments_clear(c.points[:-1], c.points[1:]) for c in carried
    ]
    contacts_before = [
        None if c.ring is None else Contact.before(position, c.points, c.ring)
        for c in carried
    ]

    def revert(candidate: FloatArray) -> tuple[BoolArray, int]:
        return _revert_carried_crossings(
            candidate, start, carried, slices, clear_before, contacts_before
        )

    bounded = bounded_burst(
        _via_objective(carried, slices, field, diameter, params, settings),
        start,
        np.concatenate(halves),
        revert,
        settings,
    )
    via, x = bounded.points[:2], bounded.points
    moved = {c.chain: c.at(via, x[s]) for c, s in zip(carried, slices, strict=True)}
    anchors = {c.chain: ~c.free for c in carried}
    return ViaBurst(via, bounded.rejected_edges, moved, anchors)


def _revert_carried_crossings(
    candidate: FloatArray,
    start: FloatArray,
    carried: Sequence[Carried],
    slices: Sequence[slice],
    clear_before: Sequence[BoolArray],
    contacts_before: Sequence[Contact | None],
) -> tuple[BoolArray, int]:
    """Move the ends of newly crossing edges back, in place, until none
    cross, as `_revert_new_crossings` does for one chain; the via's
    coordinates come first in `candidate`.

    Returns the mask of reverted coordinates and the number of edges that
    crossed in the candidate as given.
    """

    def crossings(candidate: FloatArray) -> tuple[int, BoolArray]:
        count = 0
        back = np.zeros(len(candidate), dtype=bool)
        for c, part, before, contact in zip(
            carried, slices, clear_before, contacts_before, strict=True
        ):
            points = c.at(candidate[:2], candidate[part])
            if contact is not None and not contact.kept(candidate[:2], points):
                count += 1
                back[:2] = True
                back[part] = True
                continue
            crossing = before & ~c.field.segments_clear(points[:-1], points[1:])
            if not crossing.any():
                continue
            count += int(crossing.sum())
            ends = np.zeros(len(points), dtype=bool)
            ends[:-1] |= crossing
            ends[1:] |= crossing
            back[:2] |= bool(ends[c.ends].any())
            back[part.start + np.flatnonzero(np.repeat(ends[c.free], 2))] = True
        return count, back

    return revert_until_clear(candidate, start, crossings)


def _via_objective(
    carried: Sequence[Carried],
    slices: Sequence[slice],
    field: DistanceField,
    diameter: float,
    params: EnergyParams,
    settings: OptimizerSettings,
) -> Callable[[FloatArray], tuple[float, FloatArray]]:
    """The energy of a via and its chains as a function of the via's
    position followed by the chains' free vertices, for minimize(jac=True)."""
    still = [~c.free for c in carried]
    for c, fixed in zip(carried, still, strict=True):
        fixed[c.ends] = False

    def objective(x: FloatArray) -> tuple[float, FloatArray]:
        via = x[:2]
        value, via_gradient = point_energy(via, field, params, diameter)
        gradients: list[FloatArray] = []
        for c, part, fixed in zip(carried, slices, still, strict=True):
            points = c.at(via, x[part])
            chain_value, gradient = total_energy(
                points, fixed, c.field, params, settings.short_edge
            )
            value += chain_value
            via_gradient = via_gradient + gradient[c.ends].sum(axis=0)
            if c.ring is not None:
                pair_value, pair_via, pair_points = _pair_energy(
                    via, points, c.ring, diameter, params
                )
                value += pair_value
                via_gradient = via_gradient + pair_via
                gradient = gradient + pair_points
            gradients.append(gradient[c.free].ravel())
        return value, np.concatenate([via_gradient, *gradients])

    return objective


def _pair_energy(
    via: FloatArray,
    points: FloatArray,
    ring: float,
    diameter: float,
    params: EnergyParams,
) -> tuple[float, FloatArray, FloatArray]:
    """A via and a chain in contact counted against each other, as the run
    energy counts them: the chain's obstacle terms along its length from
    the via's ring, and the via's from the chain. Returns the value and its
    gradients at the via and at every vertex.
    """
    value, chain_gradient = obstacle_energy(points, _ring_field(via, ring), params)
    # Moving the chain and the via together leaves the chain's terms as they
    # are, so the via's share is the vertices' sum, reversed.
    via_gradient = -chain_gradient.sum(axis=0)
    starts, ends = points[:-1], points[1:]
    along = ends - starts
    squared = np.einsum("ij,ij->i", along, along)
    t: FloatArray = np.clip(
        np.einsum("ij,ij->i", via - starts, along)
        / np.where(squared > 0.0, squared, 1.0),
        0.0,
        1.0,
    )
    offsets: FloatArray = via - (starts + t[:, None] * along)
    lengths: FloatArray = np.linalg.norm(offsets, axis=1)
    k = int(np.argmin(lengths))
    term, slope = obstacle_integrand(params)(np.array([lengths[k] - ring]))
    direction = offsets[k] / lengths[k] if lengths[k] > 0.0 else np.zeros(2)
    pull = diameter * slope[0] * direction
    via_gradient = via_gradient + pull
    chain_gradient = chain_gradient.copy()
    chain_gradient[k] -= (1 - t[k]) * pull
    chain_gradient[k + 1] -= t[k] * pull
    return value + diameter * float(term[0]), via_gradient, chain_gradient


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
