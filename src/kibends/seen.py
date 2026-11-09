"""What the members and moving vias of a problem see of each other, searched
for all of them at once.

A member sees pieces of the others' edges, each inflated by its own amount
(`SeenParts`), and each evaluation of the energy asks for the piece nearest
each of its quadrature points. One tree per layer, over every edge that a
member on that layer sees, finds the edges near the points of all of them,
and each pair of a point and an edge stands for the pieces of that edge
that the point's member sees. The pairs within reach are those a tree over
the member's own pieces would find, measured the same way, so the answers
are the same as a field of each member's own pieces gives.

A moving via sees whole members and the other vias (`ViaSight`): one tree
over their edges and centres answers every via, each pair inflated by the
distance its via keeps from that member or via.
"""

import itertools
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple, overload

import numpy as np

from kibends.energy import BoolArray, FloatArray, IntArray
from kibends.obstacles import (
    NearestEdge,
    candidate_pairs,
    complex_points,
    near_segments,
    nearest_of_pairs,
)


@dataclass(frozen=True)
class SeenParts:
    """What one chain of a layer sees of the others: pieces of their edges,
    each inflated by its own amount."""

    vertex: IntArray
    """Each piece's edge's first vertex, numbered across what the others
    see of every chain and section (`members.shown`), one after another,
    then two for each via (`members.shown_flat`)."""
    start: FloatArray
    """Where along its edge each piece starts, from 0 to 1."""
    end: FloatArray
    inflation: FloatArray


@dataclass(frozen=True)
class SeenLayer:
    """The members on one layer and the edges they see."""

    members: list[int]
    edges: IntArray
    """Each edge some member sees, by its first vertex, ascending."""
    first: IntArray
    """Per member and edge, at member * len(edges) + the edge's position,
    where the member's pieces of the edge start in `pieces`."""
    count: IntArray
    """How many pieces of the edge the member sees, likewise."""
    pieces: IntArray
    """The members' pieces, by their index in `Seen`, by member and edge."""


class Seen(Sequence[SeenParts]):
    """What each member of a problem sees of the others (`SeenParts`), and
    one search per layer that answers every member on it."""

    def __init__(self, parts: Sequence[SeenParts], layers: Sequence[str]) -> None:
        """`parts` per member, each on the layer `layers` gives."""
        counts = [len(part.vertex) for part in parts]
        self.vertex: IntArray = np.concatenate(
            [np.empty(0, dtype=np.intp), *(part.vertex for part in parts)]
        )
        self.start: FloatArray = np.concatenate(
            [np.empty(0), *(part.start for part in parts)]
        )
        self.end: FloatArray = np.concatenate(
            [np.empty(0), *(part.end for part in parts)]
        )
        self.inflation: FloatArray = np.concatenate(
            [np.empty(0), *(part.inflation for part in parts)]
        )
        self._bounds = bounds = list(itertools.accumulate(counts, initial=0))
        self._parts = [
            SeenParts(
                self.vertex[a:b], self.start[a:b], self.end[a:b], self.inflation[a:b]
            )
            for a, b in itertools.pairwise(bounds)
        ]
        self.within: FloatArray = np.array(
            [float(part.inflation.max(initial=0.0)) for part in parts]
        )
        """Each member's largest inflation."""
        self._on: dict[str, list[int]] = {}
        for i, layer in enumerate(layers):
            self._on.setdefault(layer, []).append(i)

    @cached_property
    def layers(self) -> list[SeenLayer]:
        """The members of each layer and what they see, indexed for the
        search; built on first use, as the checks that only look at one
        member's `SeenParts` need none."""
        return [self._layer(members) for members in self._on.values()]

    def _layer(self, members: list[int]) -> SeenLayer:
        bounds = self._bounds
        counts = [bounds[i + 1] - bounds[i] for i in members]
        pieces = np.concatenate(
            [
                np.empty(0, dtype=np.intp),
                *(np.arange(bounds[i], bounds[i + 1], dtype=np.intp) for i in members),
            ]
        )
        vertex = self.vertex[pieces]
        seen = np.zeros(int(vertex.max(initial=-1)) + 1, dtype=bool)
        seen[vertex] = True
        edges = np.flatnonzero(seen)
        position = np.cumsum(seen) - 1
        slot = np.repeat(np.arange(len(members)), counts) * len(edges)
        slot += position[vertex]
        count = np.bincount(slot, minlength=len(members) * len(edges))
        order = np.argsort(slot, kind="stable")
        return SeenLayer(members, edges, np.cumsum(count) - count, count, pieces[order])

    @overload
    def __getitem__(self, index: int) -> SeenParts: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[SeenParts]: ...
    def __getitem__(self, index: int | slice) -> SeenParts | Sequence[SeenParts]:
        return self._parts[index]

    def __len__(self) -> int:
        return len(self._parts)

    def __iter__(self) -> Iterator[SeenParts]:
        return iter(self._parts)

    def segments(
        self, pieces: IntArray, flat: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Where each of `pieces` starts and ends, with what the members
        show at `flat`."""
        a = flat[self.vertex[pieces]]
        along = flat[self.vertex[pieces] + 1] - a
        return (
            a + self.start[pieces, None] * along,
            a + self.end[pieces, None] * along,
        )

    def nearest(
        self,
        queries: Sequence[FloatArray],
        flat: FloatArray,
        reach: float,
        listed: NeighbourList | None = None,
    ) -> list[NearestEdge]:
        """Per member, the piece it sees nearest each of its `queries`, as
        `ChainField.nearest` with `reach` gives it over the member's own
        pieces at `flat`; the edge of each answer is the piece's index.
        With `listed`, each layer looks only at the pairs it lists."""
        answers: list[NearestEdge | None] = [None] * len(self)
        for n, layer in enumerate(self.layers):
            members = layer.members
            sizes = [len(queries[i]) for i in members]
            points = np.concatenate([np.empty((0, 2)), *(queries[i] for i in members)])
            owner = np.repeat(np.arange(len(members)), sizes)
            limit = reach + self.within[layer.members]
            if listed is None:
                point, piece = self._pairs(layer, points, owner, flat, limit)
            else:
                point, piece = listed.pairs(n, points, owner, flat)
            a, b = self.segments(piece, flat)
            starts = complex_points(a)
            found = nearest_of_pairs(
                complex_points(points),
                point,
                starts,
                complex_points(b) - starts,
                self.inflation[piece],
                limit[owner[point]],
            )
            answered = found.edge >= 0
            edge = np.full(len(points), -1, dtype=np.intp)
            edge[answered] = piece[found.edge[answered]]
            for i, (first, last) in zip(
                members,
                itertools.pairwise(itertools.accumulate(sizes, initial=0)),
                strict=True,
            ):
                answers[i] = NearestEdge(
                    found.distance[first:last],
                    found.gradient[first:last],
                    edge[first:last],
                )
        complete = [answer for answer in answers if answer is not None]
        # Every member is on exactly one layer.
        assert len(complete) == len(self)
        return complete

    def candidates(
        self,
        n: int,
        points: FloatArray,
        owner: IntArray,
        flat: FloatArray,
        reach: float,
        slack: float,
    ) -> tuple[IntArray, IntArray]:
        """The (point, piece) pairs of layer n, with its members' quadrature
        points `points`, each of the member `owner` gives by its place on the
        layer, that `nearest` with `reach` can answer a point with once the
        distance from each point to each piece has changed by at most
        `slack`, mm (`obstacles.candidate_pairs`)."""
        layer = self.layers[n]
        within = reach + self.within[layer.members]
        point, piece = self._pairs(layer, points, owner, flat, within + slack)
        a, b = self.segments(piece, flat)
        starts = complex_points(a)
        keep = candidate_pairs(
            complex_points(points),
            point,
            starts,
            complex_points(b) - starts,
            self.inflation[piece],
            within[owner[point]],
            slack,
        )
        return point[keep], piece[keep]

    def _pairs(
        self,
        layer: SeenLayer,
        points: FloatArray,
        owner: IntArray,
        flat: FloatArray,
        limit: FloatArray,
    ) -> tuple[IntArray, IntArray]:
        """(point, piece) pairs of `layer`, among them every piece within
        limit[m] of a point of member m, or more than limit[m] beyond."""
        if not len(layer.edges):
            empty = np.empty(0, dtype=np.intp)
            return empty, empty
        point, edge = near_segments(
            flat[layer.edges],
            flat[layer.edges + 1],
            points,
            float(limit.max()) + _ROUND_OFF,
        )
        visible = self._visible(layer, owner[point], edge)
        point, slot = point[visible], (owner[point] * len(layer.edges) + edge)[visible]
        count = layer.count[slot]
        point = np.repeat(point, count)
        rank = np.arange(len(point)) - np.repeat(np.cumsum(count) - count, count)
        return point, layer.pieces[np.repeat(layer.first[slot], count) + rank]

    @staticmethod
    def _visible(layer: SeenLayer, owner: IntArray, edge: IntArray) -> BoolArray:
        """Which (member, edge) pairs of `layer`, the member by its place on
        the layer and the edge by its position in `layer.edges`, the member
        sees any piece of: the pairs the search keeps before each stands for
        the member's pieces of its edge."""
        return layer.count[owner * len(layer.edges) + edge] > 0


# The trees are searched a little beyond the limits, so that rounding in a
# sum of them drops no pair that the exact distance keeps.
_ROUND_OFF = 1e-9


@dataclass
class _Found:
    points: FloatArray
    ends: FloatArray
    pairs: tuple[IntArray, IntArray]


class NeighbourList:
    """A Verlet list: per layer, the pairs of its members' quadrature points
    and the pieces they see of the others (`SeenParts`) that can answer
    each other while every one of those points and every vertex of the
    edges they see stays within `skin` of where it was when the pairs were
    found. Once any has moved farther, the layer finds its pairs again where
    they are. Either way the energy is the one a search over every piece
    gives; the list only spares the search."""

    def __init__(self, seen: Seen, reach: float, skin: float) -> None:
        self.seen = seen
        self.reach = reach
        self.skin = skin
        self._found: list[_Found | None] = [None] * len(seen.layers)

    def pairs(
        self, n: int, points: FloatArray, owner: IntArray, flat: FloatArray
    ) -> tuple[IntArray, IntArray]:
        """Layer n's (point, piece) pairs with its members' quadrature
        points at `points`, each of the member `owner` gives, and what they
        see of the others at `flat`."""
        edges = self.seen.layers[n].edges
        ends = flat[np.concatenate([edges, edges + 1])]
        found = self._found[n]
        if (
            found is None
            or found.points.shape != points.shape
            or _moved(points, found.points) > self.skin
            or _moved(ends, found.ends) > self.skin
        ):
            # A point's distance to a piece changes by at most its own move
            # plus the larger move of the piece's ends.
            pairs = self.seen.candidates(
                n, points, owner, flat, self.reach, 2 * self.skin
            )
            found = self._found[n] = _Found(points, ends, pairs)
        return found.pairs


def _moved(points: FloatArray, start: FloatArray) -> float:
    """The largest distance from start[i] to points[i], mm."""
    if not len(points):
        return 0.0
    delta = points - start
    return math.sqrt(float(np.max(np.einsum("ij,ij->i", delta, delta))))


class ViaNearest(NamedTuple):
    """What answers each moving via: its distance and unit gradient, and
    the member and its edge's first vertex, or -1 and the via, that answers
    it, and where that edge starts and ends."""

    distance: FloatArray
    gradient: FloatArray
    member: IntArray
    vertex: IntArray
    starts: FloatArray
    ends: FloatArray


class ViaSight:
    """What each moving via sees, for its energy and its free distance:
    every edge of the members it names, and the other vias it names as edges
    of no length, each inflated by the distance it gives; searched with one
    tree over the edges of every member some via sees and every via."""

    def __init__(
        self,
        looking: Sequence[int],
        rings: Sequence[Mapping[int, float]],
        apart: Sequence[Mapping[int, float]],
        members: int,
    ) -> None:
        """The vias `looking`, by their index, each seeing the members its
        `rings` names and the vias its `apart` names, at those distances."""
        self.looking = np.array(looking, dtype=np.intp)
        self.rings = np.full((len(looking), members), np.nan)
        self.apart = np.full((len(looking), len(rings)), np.nan)
        for n, k in enumerate(looking):
            for j, ring in rings[k].items():
                self.rings[n, j] = ring
            for u, distance in apart[k].items():
                self.apart[n, u] = distance
        self.within = np.array(
            [max([0.0, *rings[k].values(), *apart[k].values()]) for k in looking]
        )
        """Each via's largest inflation."""
        self.seen = np.flatnonzero(~np.isnan(self.rings).all(axis=0))
        """The members some via sees."""

    def nearest(
        self,
        polylines: Sequence[FloatArray],
        positions: Sequence[FloatArray],
        reach: float,
    ) -> ViaNearest:
        """What answers each via of `looking`, in order, with the members at
        `polylines` and the vias at `positions`, as `ChainField.nearest`
        with `reach` gives it over the via's own view."""
        at = np.array(positions, dtype=float).reshape(-1, 2)
        lengths = np.array([len(points) for points in polylines], dtype=np.intp)
        firsts = np.cumsum(lengths) - lengths
        counts = lengths[self.seen] - 1
        owner = np.repeat(self.seen, counts)
        vertex = np.arange(len(owner)) - np.repeat(np.cumsum(counts) - counts, counts)
        flat = np.concatenate([np.empty((0, 2)), *polylines])
        first = firsts[owner] + vertex
        a = np.concatenate([flat[first], at])
        b = np.concatenate([flat[first + 1], at])
        member = np.concatenate([owner, np.full(len(at), -1, dtype=np.intp)])
        vertex = np.concatenate([vertex, np.arange(len(at), dtype=np.intp)])
        queries = at[self.looking]
        limit = reach + self.within
        point, edge = near_segments(
            a, b, queries, float(limit.max(initial=0.0)) + _ROUND_OFF
        )
        of_member = member[edge] >= 0
        inflation = np.empty(len(edge))
        inflation[of_member] = self.rings[point[of_member], member[edge[of_member]]]
        inflation[~of_member] = self.apart[point[~of_member], vertex[edge[~of_member]]]
        keep = ~np.isnan(inflation)
        point, edge, inflation = point[keep], edge[keep], inflation[keep]
        starts = complex_points(a[edge])
        found = nearest_of_pairs(
            complex_points(queries),
            point,
            starts,
            complex_points(b[edge]) - starts,
            inflation,
            limit[point],
        )
        answered = found.edge >= 0
        e = np.full(len(queries), -1, dtype=np.intp)
        e[answered] = edge[found.edge[answered]]
        return ViaNearest(
            found.distance,
            found.gradient,
            np.where(e >= 0, member[e], -1),
            np.where(e >= 0, vertex[e], -1),
            a[e],
            b[e],
        )
