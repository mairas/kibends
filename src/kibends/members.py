"""The members of a problem, what they see of each other and their summed
energy, independent of the problem that holds them (`optimize._Problem`).

A `Member` is a chain or a coupled section as its centreline. The others
see a chain as itself and a section as its two tracks (`shown`), laid one
after another, then each via's centre twice (`shown_flat`). `seen_parts`
gives what each member sees of the others on its layer, by the rule
`neighbours.neighbour_field` builds its field by, as a `seen.Seen`: the
rule by net and kind, and as exceptions only a section's view of its own
nets' tracks and the vias, so that it grows with the members, not with
pairs of them. `members_energy` sums each member's energy in its static
field plus what it sees of the others, the distance its obstacle term
takes the smooth minimum over both (`energy.smooth_minimum`), each track
and each via an obstacle, and gives its gradient at every vertex and via
centre: where a member's obstacle term pulls on another's edge, each
obstacle's share of the gradient reaches the ends of its edge nearest the
point, and through a section's offset its centreline
(`PairBlock.offset_pullback`). `layer_energy` is its form for one layer
without vias, and `pull_answers` takes a via's push back to what answers
it, by the same shares.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np

from kibends.energy import (
    BoolArray,
    EnergyParams,
    FloatArray,
    IntArray,
    arc_lengths,
    bend_radius_energy,
    combine_fields,
    edge_integral,
    obstacle_integrand,
    obstacle_reach,
    quadrature_points,
    shape_energy,
)
from kibends.neighbours import ChainInput, obstacle_spans
from kibends.obstacles import ChainCentreline, track_inflation
from kibends.pair_blocks import PairBlock
from kibends.seen import NeighbourList, Pieces, Seen, Track, ViaSmooth, Viewer

Member = ChainInput | PairBlock
"""What a problem optimises: a chain, or a coupled section as its
centreline."""


def layer_energy(
    chains: Sequence[Member],
    polylines: Sequence[FloatArray],
    masks: Sequence[BoolArray],
    params: EnergyParams,
    short_edge: float,
    seen: Seen | None = None,
) -> tuple[float, list[FloatArray]]:
    """The summed energy of chains and sections of one layer, each in its
    static field plus what it sees of the others at `polylines`, by default
    as `seen_parts` finds it there, a section with its bend-radius hinge,
    and its gradient at every vertex of every chain and centreline, with the
    rows of `masks` zeroed; raises ValueError if it is not finite.

    A section's centreline is what it keeps clear as, and its two tracks,
    offset from it, are what the others keep clear of: their part of the
    gradient is taken back through the offset (`PairBlock.offset_pullback`).
    """
    if seen is None:
        seen = seen_parts(chains, polylines, obstacle_reach(params))
    value, gradients, _ = members_energy(
        chains, polylines, masks, params, short_edge, seen, ()
    )
    return value, gradients


def members_energy(
    members: Sequence[Member],
    polylines: Sequence[FloatArray],
    masks: Sequence[BoolArray],
    params: EnergyParams,
    short_edge: float,
    seen: Seen,
    positions: Sequence[FloatArray],
    listed: NeighbourList | None = None,
) -> tuple[float, list[FloatArray], FloatArray]:
    """`layer_energy` of members on any layers, each seeing the others on its
    own as `seen` shows them, with vias at `positions` among them; and the
    gradient at each via's centre. With `listed`, made for `seen`, each
    layer looks only at the pieces it lists."""
    reach = obstacle_reach(params)
    integrand = obstacle_integrand(params)
    k = params.points_per_edge
    lines = [
        shown(member, points) for member, points in zip(members, polylines, strict=True)
    ]
    flat = _flat(lines, positions)
    at_shown = np.zeros_like(flat)
    gradients: list[FloatArray] = []
    value = 0.0
    every_query = [quadrature_points(points, k) for points in polylines]
    answers = seen.smooth_answers(every_query, flat, reach, listed)
    statics = [
        body_of(member).field.smooth_distance(queries, reach)
        for member, queries in zip(members, every_query, strict=True)
    ]
    # Every member's points at once, one after another.
    bounds = np.cumsum([0, *(len(queries) for queries in every_query)])
    distance, (static_share, seen_share) = combine_fields(
        np.concatenate([np.empty(0), *(d for d, _ in statics)]),
        np.concatenate([np.empty(0), *(near.distance for near in answers)]),
    )
    direction = static_share[:, None] * np.concatenate(
        [np.empty((0, 2)), *(g for _, g in statics)]
    ) + seen_share[:, None] * np.concatenate(
        [np.empty((0, 2)), *(near.gradient for near in answers)]
    )
    slopes = np.empty(len(distance))
    for i, (member, points) in enumerate(zip(members, polylines, strict=True)):
        mine = slice(bounds[i], bounds[i + 1])
        obstacle, obstacle_gradient, slopes[mine] = edge_integral(
            points, distance[mine], direction[mine], integrand, k
        )
        shape, shape_gradient = shape_energy(points, masks[i], short_edge)
        value += shape + obstacle
        gradient = shape_gradient + obstacle_gradient
        if isinstance(member, PairBlock):
            bend, bend_gradient = bend_radius_energy(
                points, masks[i], member.limit, params
            )
            value += bend
            gradient += bend_gradient
        gradients.append(gradient)
    point = np.concatenate(
        [
            np.empty(0, dtype=np.intp),
            *(near.point + bounds[i] for i, near in enumerate(answers)),
        ]
    )
    pull = (slopes * seen_share)[point] * np.concatenate(
        [np.empty(0), *(near.share for near in answers)]
    )
    pulled = np.flatnonzero(pull != 0.0)
    if len(pulled):
        piece = np.concatenate(
            [np.empty(0, dtype=np.intp), *(near.edge for near in answers)]
        )[pulled]
        toward = np.concatenate([np.empty((0, 2)), *(n.direction for n in answers)])
        start = seen.vertex[piece]
        at = np.concatenate([np.empty((0, 2)), *every_query])[point[pulled]]
        along = _parameters(at, *seen.segments(piece, flat))
        t = seen.start[piece] + along * (seen.end[piece] - seen.start[piece])
        push = pull[pulled, None] * toward[pulled]
        np.add.at(at_shown, start, -push * (1 - t)[:, None])
        np.add.at(at_shown, start + 1, -push * t[:, None])
    first = 0
    for member, points, gradient, parts in zip(
        members, polylines, gradients, lines, strict=True
    ):
        pieces: list[FloatArray] = []
        for _, part in parts:
            pieces.append(at_shown[first : first + len(part)])
            first += len(part)
        if isinstance(member, PairBlock):
            gradient += member.offset_pullback(points, *pieces)
        else:
            gradient += pieces[0]
    via_gradients = at_shown[first:].reshape(-1, 2, 2).sum(axis=1)
    if not (math.isfinite(value) and all(np.all(np.isfinite(g)) for g in gradients)):
        raise ValueError(f"energy is not finite: {value}")
    for g, mask in zip(gradients, masks, strict=True):
        g[mask] = 0.0
    return value, gradients, via_gradients


def shown(member: Member, points: FloatArray) -> list[tuple[ChainInput, FloatArray]]:
    """What the others see of a member at `points`: a chain itself, a
    section its two tracks."""
    if isinstance(member, PairBlock):
        return list(zip(member.tracks, member.offset(points), strict=True))
    return [(member, points)]


def body_of(member: Member) -> ChainInput:
    """The chain a member's own energy and pins are those of."""
    return member.chain if isinstance(member, PairBlock) else member


def viewer_of(member: Member) -> tuple[ChainInput, Mapping[str, ChainInput] | None]:
    """The chain a member keeps clear of the others as, and, for a section,
    that chain under each of its nets, as `neighbours.obstacle_spans` takes them."""
    if isinstance(member, PairBlock):
        return member.facing, member.as_net
    return member, None


def shown_flat(
    members: Sequence[Member],
    polylines: Sequence[FloatArray],
    positions: Sequence[FloatArray] = (),
) -> FloatArray:
    """What the others see of every member, polylines one after another,
    then each via's centre twice, an edge of no length."""
    return _flat(
        [shown(member, line) for member, line in zip(members, polylines, strict=True)],
        positions,
    )


def _flat(
    lines: Sequence[Sequence[tuple[ChainInput, FloatArray]]],
    positions: Sequence[FloatArray],
) -> FloatArray:
    """`shown_flat` of the members `shown` as `lines`."""
    return np.concatenate(
        [
            *(points for parts in lines for _, points in parts),
            *(np.array([at, at]) for at in positions),
        ]
    )


def seen_parts(
    chains: Sequence[Member],
    polylines: Sequence[FloatArray],
    reach: float,
    rings: Sequence[Mapping[int, float]] = (),
) -> Seen:
    """For each chain or section at its polyline, the others on its layer at
    theirs as `neighbours.obstacle_spans` shows them to it, the rule
    `neighbours.neighbour_field` builds its field by. A section sees as its
    centreline of the pair's outer extent under either of its nets
    (`PairBlock.as_net`), and is seen as its two tracks. Each
    via of `rings` is seen by the members it names, inflated by the distance
    it gives.

    The tracks of another net than a member's are seen whole, by the rule
    `seen.Seen` applies; only a section's view of its own nets' tracks and
    the vias are listed, as its exceptions, so the work grows with the
    members and their edges, not with pairs of members."""
    nets: dict[str, int] = {}
    kinds: dict[tuple[float, float], int] = {}

    def kind(centreline: ChainCentreline) -> int:
        return kinds.setdefault((centreline.width, centreline.clearance), len(kinds))

    def net(name: str) -> int:
        return nets.setdefault(name, len(nets))

    lines = [
        shown(member, points) for member, points in zip(chains, polylines, strict=True)
    ]
    first = np.cumsum([0, *(len(points) for parts in lines for _, points in parts)])
    tracks: list[Track] = []
    owned: dict[tuple[str, str], list[tuple[int, ChainInput, FloatArray, int]]] = {}
    for j, parts in enumerate(lines):
        for other, points in parts:
            them, start = other.centreline, int(first[len(tracks)])
            owned.setdefault((other.layer, them.net), []).append(
                (j, other, points, start)
            )
            tracks.append(
                Track(start, len(points), other.layer, net(them.net), kind(them))
            )
    viewers: list[Viewer] = []
    exceptions: list[list[Pieces]] = [[] for _ in chains]
    for i, member in enumerate(chains):
        own, as_net = viewer_of(member)
        me = own.centreline
        viewers.append(
            Viewer(
                own.layer, net(me.net), kind(me), tuple(net(n) for n in as_net or {})
            )
        )
        for name in as_net or {}:
            for j, other, points, start in owned.get((own.layer, name), []):
                if j == i:
                    continue
                kept = obstacle_spans(
                    own, other, points, reach, polylines[i], as_net=as_net
                )
                if kept is None:
                    continue
                edge, begin, end = _pieces(points, kept.spans)
                exceptions[i].append(
                    Pieces(start + edge, begin, end, np.full(len(edge), kept.inflation))
                )
    for k, ring in enumerate(rings):
        at = np.array([first[-1] + 2 * k], dtype=np.intp)
        for i, inflation in ring.items():
            exceptions[i].append(
                Pieces(at, np.zeros(1), np.ones(1), np.array([inflation]))
            )
    every = list(kinds)
    table = np.array(
        [[track_inflation(*a, *b) for b in every] for a in every], dtype=float
    ).reshape(len(every), len(every))
    return Seen(tracks, viewers, table, [_joined(pieces) for pieces in exceptions])


def _joined(pieces: Sequence[Pieces]) -> Pieces:
    """`pieces` one after another."""
    return Pieces(
        np.concatenate([np.empty(0, dtype=np.intp), *(p.vertex for p in pieces)]),
        np.concatenate([np.empty(0), *(p.start for p in pieces)]),
        np.concatenate([np.empty(0), *(p.end for p in pieces)]),
        np.concatenate([np.empty(0), *(p.inflation for p in pieces)]),
    )


def _pieces(
    points: FloatArray, spans: Sequence[tuple[float, float]] | None
) -> tuple[IntArray, FloatArray, FloatArray]:
    """The edges of `points` that the arc-length `spans` cover, by default
    all of them, and where along each the cover starts and ends."""
    edges = np.arange(len(points) - 1, dtype=np.intp)
    if spans is None:
        return edges, np.zeros(len(edges)), np.ones(len(edges))
    arc = arc_lengths(points)
    lengths = np.diff(arc)
    found: list[IntArray] = [np.empty(0, dtype=np.intp)]
    starts: list[FloatArray] = [np.empty(0)]
    ends: list[FloatArray] = [np.empty(0)]
    for left, right in spans:
        k = np.flatnonzero((arc[:-1] < right) & (arc[1:] > left))
        length = np.where(lengths[k] > 0.0, lengths[k], 1.0)
        found.append(k)
        starts.append(np.clip((left - arc[k]) / length, 0.0, 1.0))
        ends.append(np.clip((right - arc[k]) / length, 0.0, 1.0))
    return np.concatenate(found), np.concatenate(starts), np.concatenate(ends)


def _parameters(points: FloatArray, starts: FloatArray, ends: FloatArray) -> FloatArray:
    """Where along each segment from starts[i] to ends[i] its point nearest
    points[i] lies, from 0 to 1."""
    along = ends - starts
    squared = np.einsum("ij,ij->i", along, along)
    projection = np.einsum("ij,ij->i", points - starts, along)
    return np.clip(projection / np.where(squared > 0.0, squared, 1.0), 0.0, 1.0)


def pull_answers(
    near: ViaSmooth,
    slope: FloatArray,
    at: FloatArray,
    gradients: list[FloatArray],
    via_gradients: FloatArray,
) -> None:
    """Take from what `near` found answering each via its share of the
    via's push, its unit gradient times the via's `slope`: from the other
    via, or from the ends of a member's edge in proportion to where along it
    the via's centre at `at` is nearest."""
    push = (slope[near.point] * near.share)[:, None] * near.direction
    on_via = near.member < 0
    np.subtract.at(via_gradients, near.vertex[on_via], push[on_via])
    for j in np.unique(near.member[~on_via]):
        edge = np.flatnonzero(near.member == j)
        first = near.vertex[edge]
        t = _parameters(at[near.point[edge]], near.starts[edge], near.ends[edge])
        gradient = gradients[int(j)]
        np.subtract.at(gradient, first, (1 - t)[:, None] * push[edge])
        np.subtract.at(gradient, first + 1, t[:, None] * push[edge])
