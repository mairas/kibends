"""What the members and moving vias of a problem see of each other, searched
for all of them at once.

A member sees the edges the others show (`members.shown`) by a rule with
sparse exceptions (`Seen`). By the rule it sees every edge, on its layer,
of a track of a net it does not ignore, inflated by the distance the two
kinds keep, a kind being a width and a clearance (`Seen.table`). A chain
ignores its own net and the other net of its diff pair, a coupled section
its two nets (`members.viewer_of`). The exceptions are edges, each inflated
by its own amount (`Pieces`): a member sees the vias near it with their
rings.

Every question about what the members see, the smooth distance from a
point for the energy, each track and each via an obstacle
(`energy.smooth_minimum`), how far a point lies from the nearest piece
for the checks, or whether a segment enters one, is answered for all
members of a layer with one tree over the edges they see: each pair of a
point and an edge stands for the pieces of that edge the point's member
sees, by the rule or its exceptions. The pairs within reach are those a
tree over the member's own pieces would find, measured the same way, so
the answers are the same as a field of each member's own pieces gives. A
distance wanted however far is found beyond that tree from fields that
hold only what a member sees (`Seen.distances`), so that many edges a
member does not see, such as its own net's, cost nothing.

A moving via sees whole members and the other vias (`ViaSight`): one tree
over their edges and centres answers every via, each pair inflated by the
distance its via keeps from that member or via, each member and each via
an obstacle of the smooth distance.

What the members and moving vias see of static copper is searched once
per asker and kept between evaluations (`StaticList`), as `NeighbourList`
keeps what the members see of each other.
"""

import itertools
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from kibends.energy import (
    BoolArray,
    FloatArray,
    IntArray,
    ObstacleField,
    smooth_reach,
)
from kibends.obstacles import (
    NO_FOUND,
    ChainField,
    ComplexArray,
    Found,
    SmoothAnswer,
    StaticEdges,
    StaticItems,
    candidate_pairs,
    complex_points,
    join_found,
    near_segments,
    nearest_of_pairs,
    obstacle_order,
    pair_depths,
    pairs_inside,
    search_parts,
    smooth_of_pairs,
    static_parts,
    static_smooth,
)


class Track(NamedTuple):
    """A track a member shows the others."""

    first: int
    """Its first vertex, numbered across what every member shows, one after
    another, then two for each via (`members.shown_flat`)."""
    vertices: int
    layer: str
    net: int
    """Its net, numbered."""
    kind: int
    """Its width and clearance, numbered as `Seen.table` takes them."""


class Viewer(NamedTuple):
    """A member as it sees the others."""

    layer: str
    net: int
    kind: int
    exempt: tuple[int, ...]
    """The nets besides its own whose tracks it does not see: the other
    net of a chain's diff pair, a section's two. Its own tracks are of its
    net or of these."""


class Pieces(NamedTuple):
    """Edges, each inflated by its own amount."""

    vertex: IntArray
    """Each piece's edge's first vertex, numbered as `Track.first`."""
    inflation: FloatArray


@dataclass(frozen=True)
class SeenLayer:
    """The members on one layer, the edges they see and what decides which
    pieces of them each sees."""

    members: list[int]
    first: int
    """Its first piece, by its index in `Seen`; its pieces are the next
    `pieces`."""
    pieces: int
    edges: IntArray
    """Each edge some member sees, by its first vertex, ascending."""
    whole: IntArray
    """Each edge's piece covering all of it, by its index in `Seen`."""
    net: IntArray
    """Each edge's track's net, -1 for a via."""
    kind: IntArray
    """Each edge's track's kind, -1 for a via."""
    viewer_net: IntArray
    """Per member, its net."""
    viewer_kind: IntArray
    exempt: IntArray
    """Per member, a row of the nets besides its own whose tracks it does
    not see, padded with NO_NET."""
    slot: IntArray
    """The members' exceptions, each at member * len(edges) + the position
    of its edge, ascending."""
    piece: IntArray
    """Each exception's piece, by its index in `Seen`."""
    inflation: FloatArray
    """Each exception's inflation."""


NO_NET = -2
"""Pads `SeenLayer.exempt`: no track's net, nor a via's."""


class Seen:
    """What each member of a problem sees of the others at one set of
    shapes, and one search per layer that answers every member on it."""

    def __init__(
        self,
        tracks: Sequence[Track],
        viewers: Sequence[Viewer],
        table: FloatArray,
        exceptions: Sequence[Pieces],
    ) -> None:
        """The members as `viewers` and their `exceptions`, in order,
        seeing the `tracks`, two tracks of kinds k and l of two nets kept
        table[k, l] apart."""
        self.table = table
        size = max(
            [
                0,
                *(t.first + t.vertices for t in tracks),
                *(int(p.vertex.max(initial=-1)) + 2 for p in exceptions),
            ]
        )
        net = np.full(size, -1, dtype=np.intp)
        kind = np.full(size, -1, dtype=np.intp)
        # A vertex of no track is a via's first, an obstacle of its own.
        obstacle = len(tracks) + np.arange(size, dtype=np.intp)
        for k, t in enumerate(tracks):
            net[t.first : t.first + t.vertices - 1] = t.net
            kind[t.first : t.first + t.vertices - 1] = t.kind
            obstacle[t.first : t.first + t.vertices] = k
        self.within: FloatArray = np.zeros(len(viewers))
        """Each member's largest inflation."""
        vertex: list[IntArray] = [np.empty(0, dtype=np.intp)]
        on: dict[str, list[int]] = {}
        for i, viewer in enumerate(viewers):
            on.setdefault(viewer.layer, []).append(i)
        self.layers: list[SeenLayer] = []
        pieces = 0
        for layer, members in on.items():
            seeing = [viewers[i] for i in members]
            own = [exceptions[i] for i in members]
            shown = [t for t in tracks if t.layer == layer]
            blocked = Counter(n for v in seeing for n in {v.net, *v.exempt})
            ex_vertex = np.concatenate(
                [np.empty(0, dtype=np.intp), *(p.vertex for p in own)]
            )
            edges = np.unique(
                np.concatenate(
                    [
                        ex_vertex,
                        *(
                            np.arange(t.first, t.first + t.vertices - 1, dtype=np.intp)
                            for t in shown
                            if blocked[t.net] < len(members)
                        ),
                    ]
                )
            )
            count = len(edges)
            ex_member = np.repeat(np.arange(len(members)), [len(p.vertex) for p in own])
            slot = ex_member * count + np.searchsorted(edges, ex_vertex)
            order = np.argsort(slot, kind="stable")
            ex_piece = pieces + count + np.arange(len(ex_vertex), dtype=np.intp)
            ex_inflation = np.concatenate([np.empty(0), *(p.inflation for p in own)])
            vertex += [edges, ex_vertex]
            width = max([0, *(len(v.exempt) for v in seeing)])
            exempt = np.full((len(members), width), NO_NET, dtype=np.intp)
            for row, v in zip(exempt, seeing, strict=True):
                row[: len(v.exempt)] = v.exempt
            self.layers.append(
                SeenLayer(
                    members=members,
                    first=pieces,
                    pieces=count + len(ex_vertex),
                    edges=edges,
                    whole=pieces + np.arange(count, dtype=np.intp),
                    net=net[edges],
                    kind=kind[edges],
                    viewer_net=np.array([v.net for v in seeing], dtype=np.intp),
                    viewer_kind=np.array([v.kind for v in seeing], dtype=np.intp),
                    exempt=exempt,
                    slot=slot[order],
                    piece=ex_piece[order],
                    inflation=ex_inflation[order],
                )
            )
            pieces += count + len(ex_vertex)
            nets_of: dict[int, set[int]] = {}
            for t in shown:
                nets_of.setdefault(t.kind, set()).add(t.net)
            for i, v, p in zip(members, seeing, own, strict=True):
                excluded = {v.net, *v.exempt}
                # Stops at the first net not excluded, so after at most
                # len(excluded) + 1 of them, however many nets a kind has.
                self.within[i] = max(
                    [
                        0.0,
                        *(
                            float(table[v.kind, k])
                            for k, nets in nets_of.items()
                            if any(n not in excluded for n in nets)
                        ),
                        float(p.inflation.max(initial=0.0)),
                    ]
                )
        self.vertex: IntArray = np.concatenate(vertex)
        """Each piece's edge's first vertex, numbered as `Track.first`."""
        self.obstacle: IntArray = obstacle[self.vertex]
        """Each piece's obstacle for the smooth minimum: its track, or its
        via."""

    def segments(
        self, pieces: IntArray, flat: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Where each of `pieces` starts and ends, with what the members
        show at `flat`."""
        a = flat[self.vertex[pieces]]
        return a, a + (flat[self.vertex[pieces] + 1] - a)

    def smooth_answers(
        self,
        queries: Sequence[FloatArray],
        flat: FloatArray,
        reach: float,
        listed: NeighbourList | None = None,
    ) -> list[SmoothAnswer]:
        """Per member, the smooth distance from each of its `queries` to
        the pieces it sees, each track and each via an obstacle, and what
        answers it, as `ChainField.smooth_answers` with `reach` gives it over the
        member's own pieces at `flat`; each answer's point is by its place
        among the member's queries and its edge is the piece's index. With
        `listed`, each layer looks only at the pairs it lists."""
        answers: list[SmoothAnswer | None] = [None] * len(self.within)
        for n, layer in enumerate(self.layers):
            points, owner, bounds = _gathered(queries, layer.members)
            limit = self._smooth_limit(layer, reach)
            if listed is None:
                point, piece, inflation = self._in_obstacle_order(
                    self._pairs(layer, points, owner, flat, float(limit.max()))
                )
            else:
                point, piece, inflation = listed.pairs(n, points, owner, flat)
            starts, along = self._shapes(layer, flat, piece)
            found = smooth_of_pairs(
                complex_points(points),
                point,
                starts,
                along,
                inflation,
                limit[owner[point]],
                self.obstacle[piece],
            )
            # The answers come in the order of their points, so each
            # member's are one stretch.
            at, edge = found.point, piece[found.edge]
            share, direction = found.share, found.direction
            for i, (first, last) in zip(layer.members, bounds, strict=True):
                low, high = np.searchsorted(at, [first, last])
                answers[i] = SmoothAnswer(
                    found.distance[first:last],
                    found.gradient[first:last],
                    at[low:high] - first,
                    edge[low:high],
                    share[low:high],
                    direction[low:high],
                )
        return _complete(answers)

    def distances(
        self, queries: Sequence[FloatArray], flat: FloatArray, below: float = math.inf
    ) -> list[FloatArray]:
        """Per member, the signed distance from each of its `queries` to the
        nearest piece it sees, as `ChainField.signed_distance` gives it over
        the member's own pieces, +inf where it sees none. `below` is the
        threshold the caller compares the distances against: below it they
        are exact, elsewhere at least `below`; by default exact everywhere.
        `flat` is what the members show (`members.shown_flat`) at the shapes
        this was built at, with the vias where the caller holds them.

        One search within the members' largest inflation plus `below`, or
        twice that inflation, answers every point whose nearest piece lies
        within it. Without a finite `below`, the rest are answered from the
        pieces they see alone (`_beyond`), so that no edge a member does not
        see, such as its own net's, is looked at however far its nearest
        piece lies."""
        answers: list[FloatArray | None] = [None] * len(self.within)
        for layer in self.layers:
            points, owner, bounds = _gathered(queries, layer.members)
            distance = np.full(len(points), np.inf)
            within = self.within[layer.members]
            widest = float(within.max())
            if len(layer.edges) and widest > 0.0:
                finite = math.isfinite(below)
                radius = widest + below if finite else 2 * widest
                if radius >= 0.0:
                    point, piece, inflation = self._pairs(
                        layer, points, owner, flat, radius
                    )
                    starts, along = self._shapes(layer, flat, piece)
                    distance = nearest_of_pairs(
                        complex_points(points), point, starts, along, inflation, radius
                    ).distance
                if not finite:
                    # A piece beyond the radius lies farther than the radius
                    # less the member's largest inflation.
                    far = np.flatnonzero(distance > radius - within[owner])
                    distance[far] = self._beyond(layer, points[far], owner[far], flat)
            for i, (first, last) in zip(layer.members, bounds, strict=True):
                answers[i] = distance[first:last]
        return _complete(answers)

    def _beyond(
        self, layer: SeenLayer, points: FloatArray, owner: IntArray, flat: FloatArray
    ) -> FloatArray:
        """`distances` at `points` of `layer`, each of the member `owner`
        gives by its place on the layer, exactly and however far, each point
        looking only at what its member sees.

        A member's exceptions form a field of their own. The tracks a member
        sees by the rule are those of every net but its own and those it
        ignores besides: a member that ignores more than its own net has a
        field per distinct set of nets left out. The others' are the union,
        over the bits of the nets' numbers, of the tracks whose net differs
        from its own in that bit, so each edge is in one field per bit and
        every other net in one field at least."""
        distance = np.full(len(points), np.inf)
        if not len(points):
            return distance
        a, b = self.segments(layer.whole, flat)
        tracks = np.flatnonzero(layer.kind >= 0)
        net = layer.net[tracks]
        nets = [
            {int(n) for n in (layer.viewer_net[m], *layer.exempt[m]) if n != NO_NET}
            for m in range(len(layer.members))
        ]

        def nearest(edges: IntArray, asking: IntArray) -> None:
            """Lower `distance` at the points `asking` to what the track
            `edges` give, by their positions in `layer.edges`."""
            kinds = layer.viewer_kind[owner[asking]]
            for kind in np.unique(kinds):
                at = asking[kinds == kind]
                # Only exact distances are asked of it, which obstacles do
                # not change.
                field = ChainField(
                    a[edges],
                    b[edges],
                    self.table[kind, layer.kind[edges]],
                    np.arange(len(edges), dtype=np.intp),
                )
                d = field.signed_distance(points[at])[0]
                distance[at] = np.minimum(distance[at], d)

        single = np.array([len(n) == 1 for n in nets])[owner]
        own = np.array([min(n) if len(n) == 1 else NO_NET for n in nets])[owner]
        # Every bit any track's or asking chain's net sets: a net that agrees
        # with a chain's own in fewer bits would never be asked for.
        widest = max(int(net.max(initial=0)), int(own[single].max(initial=0)))
        for bit in range(max(1, widest.bit_length())):
            for value in (0, 1):
                asking = np.flatnonzero(single & ((own >> bit) & 1 != value))
                edges = tracks[(net >> bit) & 1 == value]
                if len(asking) and len(edges):
                    nearest(edges, asking)
        groups: dict[frozenset[int], list[int]] = {}
        for m in range(len(nets)):
            if len(nets[m]) > 1:
                groups.setdefault(frozenset(nets[m]), []).append(m)
        for left_out, members in groups.items():
            asking = np.flatnonzero(np.isin(owner, members))
            edges = tracks[~np.isin(net, list(left_out))]
            if len(asking) and len(edges):
                nearest(edges, asking)
        count = len(layer.edges)
        for m in range(len(layer.members)):
            low, high = np.searchsorted(layer.slot, [m * count, (m + 1) * count])
            if low == high:
                continue
            c, d = self.segments(layer.piece[low:high], flat)
            at = np.flatnonzero(owner == m)
            if not len(at):
                continue
            field = ChainField(
                c, d, layer.inflation[low:high], np.arange(high - low, dtype=np.intp)
            )
            distance[at] = np.minimum(
                distance[at], field.signed_distance(points[at])[0]
            )
        return distance

    def segments_clear(
        self,
        polylines: Sequence[FloatArray],
        flat: FloatArray,
        tolerance: float = 0.0,
    ) -> list[EdgesClear]:
        """Per member at `polylines`, whether each of its edges stays out of
        the pieces it sees at `flat`, as `ChainField.segments_clear` with
        `tolerance` gives it over the member's own pieces, and what each
        edge that does not enters."""
        answers: list[EdgesClear | None] = [None] * len(self.within)
        for layer in self.layers:
            a, owner, bounds = _gathered([p[:-1] for p in polylines], layer.members)
            b, _, _ = _gathered([p[1:] for p in polylines], layer.members)
            crossing = np.zeros(len(a), dtype=bool)
            segment = np.empty(0, dtype=np.intp)
            vertex = np.empty(0, dtype=np.intp)
            obstacle = np.empty(0, dtype=np.intp)
            depth = np.empty(0)
            reach = float(self.within[layer.members].max()) - tolerance
            if len(layer.edges) and len(a) and reach > 0.0:
                ca, cb = complex_points(a), complex_points(b)
                radius = reach + float(np.abs(cb - ca).max()) / 2
                segment, piece, inflation = self._pairs(
                    layer, (a + b) / 2, owner, flat, radius
                )
                starts, along = self._shapes(layer, flat, piece)
                inside = pairs_inside(
                    ca[segment],
                    cb[segment] - ca[segment],
                    starts,
                    along,
                    inflation,
                    tolerance,
                )
                taken = np.flatnonzero(inside)
                taken = taken[np.argsort(segment[taken], kind="stable")]
                segment, entered = segment[taken], piece[taken]
                vertex, obstacle = self.vertex[entered], self.obstacle[entered]
                depth = pair_depths(
                    ca[segment],
                    cb[segment] - ca[segment],
                    starts[taken],
                    along[taken],
                    inflation[taken],
                )
                crossing[segment] = True
            for i, (first, last) in zip(layer.members, bounds, strict=True):
                low, high = np.searchsorted(segment, [first, last])
                answers[i] = EdgesClear(
                    ~crossing[first:last],
                    segment[low:high] - first,
                    vertex[low:high],
                    obstacle[low:high],
                    depth[low:high],
                )
        return _complete(answers)

    def candidates(
        self,
        n: int,
        points: FloatArray,
        owner: IntArray,
        flat: FloatArray,
        reach: float,
        slack: float,
    ) -> Pairs:
        """The (point, piece) pairs of layer n, with its members' quadrature
        points `points`, each of the member `owner` gives by its place on the
        layer, that `smooth_answers` with `reach` can answer a point with once the
        distance from each point to each piece has changed by at most
        `slack`, mm (`obstacles.candidate_pairs`), and each pair's
        inflation."""
        layer = self.layers[n]
        within = self._smooth_limit(layer, reach)
        point, piece, inflation = self._pairs(
            layer, points, owner, flat, float(within.max()) + slack
        )
        starts, along = self._shapes(layer, flat, piece)
        keep = candidate_pairs(
            complex_points(points),
            point,
            starts,
            along,
            inflation,
            within[owner[point]],
            slack,
        )
        return self._in_obstacle_order(Pairs(point[keep], piece[keep], inflation[keep]))

    def _smooth_limit(self, layer: SeenLayer, reach: float) -> FloatArray:
        """How far from a point of each member of `layer` the smooth search
        with `reach` looks at what the member sees, its largest inflation
        included: the same for the search and the neighbour list, so that
        both give the same answers."""
        return smooth_reach(reach) + self.within[layer.members]

    def _in_obstacle_order(self, pairs: Pairs) -> Pairs:
        """`pairs` in the order `smooth_of_pairs` takes them, searched or
        listed alike."""
        order = obstacle_order(pairs.point, self.obstacle[pairs.piece], pairs.piece)
        return Pairs(pairs.point[order], pairs.piece[order], pairs.inflation[order])

    def _shapes(
        self, layer: SeenLayer, flat: FloatArray, pieces: IntArray
    ) -> tuple[ComplexArray, ComplexArray]:
        """Where each of `pieces`, all of `layer`, starts and the way to its
        end, as complex numbers, with what the members show at `flat`; each
        piece of the layer is measured once, however many pairs it is in."""
        a, b = self.segments(
            np.arange(layer.first, layer.first + layer.pieces, dtype=np.intp), flat
        )
        starts = complex_points(a)
        along = complex_points(b) - starts
        local = pieces - layer.first
        return starts[local], along[local]

    def _pairs(
        self,
        layer: SeenLayer,
        points: FloatArray,
        owner: IntArray,
        flat: FloatArray,
        radius: float,
    ) -> Pairs:
        """(point, piece) pairs of `layer`, among them every piece within
        `radius` of a point that the point's member, by its place on the
        layer in `owner`, sees, and each pair's inflation."""
        if not len(layer.edges) or not len(points):
            return _NO_PAIRS
        point, edge = near_segments(
            flat[layer.edges], flat[layer.edges + 1], points, radius + _ROUND_OFF
        )
        member = owner[point]
        net = layer.net[edge]
        ruled = (
            (layer.kind[edge] >= 0)
            & (net != layer.viewer_net[member])
            & ~(net[:, None] == layer.exempt[member]).any(axis=1)
        )
        if not len(layer.slot):
            point, edge, member = point[ruled], edge[ruled], member[ruled]
            return Pairs(
                point,
                layer.whole[edge],
                self.table[layer.viewer_kind[member], layer.kind[edge]],
            )
        slot = member * len(layer.edges) + edge
        first = np.searchsorted(layer.slot, slot)
        count = np.where(
            ruled, 1, np.searchsorted(layer.slot, slot, side="right") - first
        )
        point = np.repeat(point, count)
        rank = np.arange(len(point)) - np.repeat(np.cumsum(count) - count, count)
        ruled, edge, member = (np.repeat(x, count) for x in (ruled, edge, member))
        excepted = np.repeat(first, count)[~ruled] + rank[~ruled]
        piece = np.empty(len(point), dtype=np.intp)
        inflation = np.empty(len(point))
        piece[ruled] = layer.whole[edge[ruled]]
        inflation[ruled] = self.table[
            layer.viewer_kind[member[ruled]], layer.kind[edge[ruled]]
        ]
        piece[~ruled] = layer.piece[excepted]
        inflation[~ruled] = layer.inflation[excepted]
        return Pairs(point, piece, inflation)


class EdgesClear(NamedTuple):
    """Whether each edge of a member stays out of the pieces it sees, and
    what those that do not enter: per (edge, piece) pair, the edge, the
    first vertex of the piece's edge, numbered as `Track.first`, so that a
    via's is its first vertex in `members.shown_flat`, the piece's
    obstacle, its track or its via (`Seen.obstacle`), and how deep the edge
    is inside the piece (`obstacles.pair_depths`)."""

    clear: BoolArray
    edge: IntArray
    vertex: IntArray
    obstacle: IntArray
    depth: FloatArray


class Pairs(NamedTuple):
    """(point, piece) pairs, and the inflation each pair's member sees its
    piece with."""

    point: IntArray
    piece: IntArray
    inflation: FloatArray


_NO_PAIRS = Pairs(np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp), np.empty(0))


def _gathered(
    queries: Sequence[FloatArray], members: Sequence[int]
) -> tuple[FloatArray, IntArray, list[tuple[int, int]]]:
    """The `queries` of `members` one after another, each one's member by
    its place among them, and where each member's lie."""
    sizes = [len(queries[i]) for i in members]
    points = np.concatenate([np.empty((0, 2)), *(queries[i] for i in members)])
    owner = np.repeat(np.arange(len(members)), sizes)
    return (
        points,
        owner,
        list(itertools.pairwise(itertools.accumulate(sizes, initial=0))),
    )


def _complete[T](answers: Sequence[T | None]) -> list[T]:
    complete = [answer for answer in answers if answer is not None]
    # Every member is on exactly one layer.
    assert len(complete) == len(answers)
    return complete


# The trees are searched a little beyond the limits, so that rounding in a
# sum of them drops no pair that the exact distance keeps.
_ROUND_OFF = 1e-9


SKIN_LEAST = 0.05
"""The skin each layer of a `NeighbourList` starts with, mm."""
SKIN_MOST = 0.15
"""The widest skin a layer of a `NeighbourList` takes, mm."""


@dataclass
class _Found:
    points: FloatArray
    ends: FloatArray
    pairs: Pairs
    skin: float


class NeighbourList:
    """A Verlet list: per layer, the pairs of its members' quadrature points
    and the pieces they see of the others (`Seen`) that can answer
    each other while every one of those points and every vertex of the
    edges they see stays within the layer's skin of where it was when the
    pairs were found. Once any has moved farther, the layer finds its pairs
    again where they are. Either way the energy is the one a search over
    every piece gives, whatever the skin: a point's distance to a piece
    changes by at most its own move plus the larger move of the piece's
    ends, so the pairs are found with twice the skin to spare. The list
    only spares the search.

    A layer's skin starts at SKIN_LEAST and, each time the layer finds its
    pairs again, becomes how far its points and pieces had moved since it
    last found them, at most SKIN_MOST. Pieces that moved that far are
    likely to move as far again, so a list lasts about as many evaluations
    whether they move far or barely, and one made for a burst that hardly
    moves stays small. A wider skin costs at every evaluation: a pair is
    listed while its edge may come within SMOOTH_REACH of the point's
    nearest, so the band listed behind the nearest is SMOOTH_REACH plus four
    skins wide. A search costs about six evaluations' answers.

    Measured with `./run bench --only bus`, a fixed skin of 0.2 mm listed
    more than twice the pairs of SKIN_LEAST and ran no faster than one of
    0.1 mm. SKIN_MOST ran fastest on the Z-shaped buses (`z_bus`), whose
    tracks move far. It ran a tenth slower than a fixed 0.1 mm on 262
    short chains at 0.6 mm pitch (`short_chains`), where the band reaches
    a second track behind the nearest. On a real board of 100 chains and
    96 moving vias, a 60 s run took 81 searches in 166 evaluations instead
    of 170, and its objective 58.3 s instead of 70.8 s."""

    def __init__(self, seen: Seen, reach: float) -> None:
        self.seen = seen
        self.reach = reach
        self._found: list[_Found | None] = [None] * len(seen.layers)

    def pairs(
        self, n: int, points: FloatArray, owner: IntArray, flat: FloatArray
    ) -> Pairs:
        """Layer n's (point, piece) pairs with its members' quadrature
        points at `points`, each of the member `owner` gives, and what they
        see of the others at `flat`, with each pair's inflation."""
        edges = self.seen.layers[n].edges
        ends = flat[np.concatenate([edges, edges + 1])]
        found = self._found[n]
        skin = SKIN_LEAST
        if found is not None and found.points.shape == points.shape:
            moved = max(_moved(points, found.points), _moved(ends, found.ends))
            if moved <= found.skin:
                return found.pairs
            skin = min(moved, SKIN_MOST)
        # A point's distance to a piece changes by at most its own move plus
        # the larger move of the piece's ends.
        pairs = self.seen.candidates(n, points, owner, flat, self.reach, 2 * skin)
        self._found[n] = _Found(points, ends, pairs, skin)
        return pairs


STATIC_SKIN = SKIN_LEAST
"""The skin of every asker of a `StaticList`, mm."""


class _StaticFound(NamedTuple):
    points: FloatArray
    found: Found


class StaticList:
    """The static fields of several askers, members or moving vias, each
    at its own points, answered from their edges (`obstacles.StaticEdges`)
    with one search per asker, kept between evaluations as `NeighbourList`
    keeps a layer's pairs.

    Static edges never move, so a point's distance to one changes by at
    most the point's own move: an asker searches again once one of its
    points has moved farther than STATIC_SKIN since its last search, with
    that skin to spare. Either way the answers are those of a search at the
    points; the list only spares the search. The askers' pairs are
    measured together, in one pass (`obstacles.static_smooth`).

    Unlike a layer's in `NeighbourList`, the skin does not follow how far
    the points moved: a wider skin lists more edges of each item and more
    items behind the nearest at every evaluation. Measured with
    `./run bench --only bus` on the pad rows, a skin following the moves up
    to SKIN_MOST listed 1.6 times the pairs and took a tenth longer.

    A field that gives no edges (`obstacles.static_parts`) answers by its
    own `smooth_distance`, and an asker with no field sees nothing. Not
    `listed`, every call searches without a skin.
    """

    def __init__(
        self,
        fields: Sequence[ObstacleField | None],
        reach: float,
        *,
        listed: bool = True,
    ) -> None:
        self.fields = tuple(fields)
        self.reach = reach
        self.listed = listed
        every: dict[int, int] = {}
        parts: list[StaticEdges] = []
        given = [None if f is None else static_parts(f) for f in self.fields]
        for found in given:
            for part in found or ():
                if every.setdefault(id(part), len(parts)) == len(parts):
                    parts.append(part)
        self.items = StaticItems(parts)
        """Every asker's items, one field after another, each field once."""
        self._parts: list[list[tuple[StaticEdges, int]] | None] = [
            None
            if found is None
            else [(p, int(self.items.starts[every[id(p)]])) for p in found]
            for found in given
        ]
        self._found: list[_StaticFound | None] = [None] * len(self.fields)
        self._joined: Found = NO_FOUND

    def smooth(
        self, queries: Sequence[FloatArray]
    ) -> list[tuple[FloatArray, FloatArray]]:
        """Each asker's smooth distance at its `queries`, (n, 2) points, and
        its gradient, as its field's `smooth_distance` with the list's reach
        gives them."""
        edged = [k for k, parts in enumerate(self._parts) if parts is not None]
        reach = smooth_reach(self.reach)
        changed = not self.listed or not any(self._found)
        for k in edged:
            parts = self._parts[k]
            assert parts is not None
            points = queries[k]
            last = self._found[k]
            if (
                self.listed
                and last is not None
                and last.points.shape == points.shape
                and _moved(points, last.points) <= STATIC_SKIN
            ):
                continue
            slack = STATIC_SKIN if self.listed else 0.0
            found = search_parts(parts, points, reach, slack)
            self._found[k] = _StaticFound(points, found)
            changed = True
        sizes = [len(queries[k]) for k in edged]
        firsts = list(itertools.accumulate(sizes, initial=0))
        if changed:
            founds: list[Found] = []
            for k, first in zip(edged, firsts, strict=False):
                last = self._found[k]
                assert last is not None
                founds.append(last.found._replace(point=last.found.point + first))
            self._joined = join_found(founds) if founds else NO_FOUND
        coords = np.concatenate([np.empty((0, 2)), *(queries[k] for k in edged)])
        distance, gradient = static_smooth(
            complex_points(coords), self._joined, self.items
        )
        answers: list[tuple[FloatArray, FloatArray]] = []
        place = dict(zip(edged, firsts, strict=False))
        for k, (field, points) in enumerate(zip(self.fields, queries, strict=True)):
            first = place.get(k)
            if first is not None:
                last = first + len(points)
                answers.append((distance[first:last], gradient[first:last]))
            elif field is None:
                free = np.full(len(points), np.inf)
                answers.append((free, np.zeros((len(points), 2))))
            else:
                answers.append(field.smooth_distance(points, self.reach))
        return answers


def _moved(points: FloatArray, start: FloatArray) -> float:
    """The largest distance from start[i] to points[i], mm."""
    if not len(points):
        return 0.0
    delta = points - start
    return math.sqrt(float(np.max(np.einsum("ij,ij->i", delta, delta))))


class ViaSight:
    """What each moving via sees, for its energy and its free distance:
    every edge of the members it names, and the other vias it names as edges
    of no length, each inflated by the distance it gives; searched with one
    tree over the edges of every member some via sees and every via."""

    def __init__(
        self,
        rings: Sequence[Mapping[int, float]],
        apart: Sequence[Mapping[int, float]],
        members: int,
    ) -> None:
        """Each via, by its index, seeing the members its `rings` names and
        the vias its `apart` names, at those distances."""
        self.rings = np.full((len(rings), members), np.nan)
        self.apart = np.full((len(rings), len(rings)), np.nan)
        for k, (seen, kept) in enumerate(zip(rings, apart, strict=True)):
            for j, ring in seen.items():
                self.rings[k, j] = ring
            for u, distance in kept.items():
                self.apart[k, u] = distance
        self.within = np.array(
            [
                max([0.0, *seen.values(), *kept.values()])
                for seen, kept in zip(rings, apart, strict=True)
            ]
        )
        """Each via's largest inflation."""
        self.seen = np.flatnonzero(~np.isnan(self.rings).all(axis=0))
        """The members some via sees."""

    def distances(
        self,
        polylines: Sequence[FloatArray],
        positions: Sequence[FloatArray],
        reach: float,
    ) -> FloatArray:
        """Each via's exact distance, in order, to the nearest of what it
        sees, with the members at `polylines` and the vias at `positions`,
        as `ChainField.signed_distance` with `reach` gives it over the via's
        own view."""
        view = self._view(polylines, positions, reach)
        starts = complex_points(view.a[view.edge])
        return nearest_of_pairs(
            complex_points(view.at),
            view.point,
            starts,
            complex_points(view.b[view.edge]) - starts,
            view.inflation,
            view.limit[view.point],
        ).distance

    def obstacle_distances(
        self,
        polylines: Sequence[FloatArray],
        positions: Sequence[FloatArray],
        below: float,
    ) -> ObstacleDistances:
        """Each via's exact distance to each member and each via it sees,
        as `distances` gives it over that one alone, where it is below
        `below`."""
        view = self._view(polylines, positions, below)
        starts = complex_points(view.a[view.edge])
        distance = nearest_of_pairs(
            complex_points(view.at)[view.point],
            np.arange(len(view.point)),
            starts,
            complex_points(view.b[view.edge]) - starts,
            view.inflation,
            view.limit[view.point],
        ).distance
        near = np.flatnonzero(distance < below)
        edge = view.edge[near]
        via, distance = view.point[near], distance[near]
        member, vertex = view.member[edge], view.vertex[edge]
        obstacle = np.where(member >= 0, member, len(polylines) + vertex)
        key = via * (len(polylines) + len(view.at)) + obstacle
        order = np.lexsort((distance, key))
        nearest = order[np.flatnonzero(np.diff(key[order], prepend=-1) != 0)]
        return ObstacleDistances(
            key[nearest],
            via[nearest],
            obstacle[nearest],
            distance[nearest],
            vertex[nearest],
        )

    def smooth_answers(
        self,
        polylines: Sequence[FloatArray],
        positions: Sequence[FloatArray],
        reach: float,
    ) -> ViaSmooth:
        """Each via's smooth distance, in order, to what it sees, each
        member and each via an obstacle, and what answers it, as
        `ChainField.smooth_answers` with `reach` gives it over the via's own view."""
        view = self._view(polylines, positions, smooth_reach(reach))
        member = view.member[view.edge]
        obstacle = np.where(
            member >= 0, member, len(polylines) + view.vertex[view.edge]
        )
        order = obstacle_order(view.point, obstacle, view.edge)
        point, edge = view.point[order], view.edge[order]
        starts = complex_points(view.a[edge])
        found = smooth_of_pairs(
            complex_points(view.at),
            point,
            starts,
            complex_points(view.b[edge]) - starts,
            view.inflation[order],
            view.limit[point],
            obstacle[order],
        )
        e = edge[found.edge]
        return ViaSmooth(
            found.distance,
            found.gradient,
            found.point,
            view.member[e],
            view.vertex[e],
            view.a[e],
            view.b[e],
            found.share,
            found.direction,
        )

    def _view(
        self,
        polylines: Sequence[FloatArray],
        positions: Sequence[FloatArray],
        reach: float,
    ) -> _ViaView:
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
        limit = reach + self.within
        radius = float(limit.max(initial=0.0)) + _ROUND_OFF
        point, edge = near_segments(a, b, at, radius)
        of_member = member[edge] >= 0
        inflation = np.empty(len(edge))
        inflation[of_member] = self.rings[point[of_member], member[edge[of_member]]]
        inflation[~of_member] = self.apart[point[~of_member], vertex[edge[~of_member]]]
        keep = ~np.isnan(inflation)
        return _ViaView(
            at, a, b, member, vertex, point[keep], edge[keep], inflation[keep], limit
        )


class _ViaView(NamedTuple):
    """What the vias see: every edge some via may see, from a along to b,
    each a member's edge by its first vertex or a via, and the (via, edge)
    pairs within `limit` of each via that it sees, and their inflations."""

    at: FloatArray
    a: FloatArray
    b: FloatArray
    member: IntArray
    vertex: IntArray
    point: IntArray
    edge: IntArray
    inflation: FloatArray
    limit: FloatArray


class ObstacleDistances(NamedTuple):
    """(via, obstacle) pairs, in ascending order of `key`, and each via's
    exact distance to its obstacle, a member or a via. Of a member, the
    first vertex of its edge nearest the via comes too."""

    key: IntArray
    """Each pair's via times the number of obstacles, plus its obstacle."""
    via: IntArray
    obstacle: IntArray
    """A member by its index, a via by the number of members plus its."""
    distance: FloatArray
    vertex: IntArray
    """The member's edge's first vertex, or the via's index."""


class ViaSmooth(NamedTuple):
    """Each moving via's smooth distance and its gradient, and what answers
    it: per member or via that has a share, the via it answers (`point`, as
    in `obstacles.SmoothAnswer`), the member and its edge's first vertex, or
    -1 and the via, where that edge starts and ends, its share and its unit
    gradient."""

    distance: FloatArray
    gradient: FloatArray
    point: IntArray
    member: IntArray
    vertex: IntArray
    starts: FloatArray
    ends: FloatArray
    share: FloatArray
    direction: FloatArray
