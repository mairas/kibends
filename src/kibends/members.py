"""The members of a problem, what they see of each other and their summed
energy, independent of the problem that holds them (`optimize._Problem`).

A `Member` is a chain or a coupled section as its centreline. The others
see a chain as itself and a section as its two tracks (`shown`), laid one
after another, then each via's centre twice (`shown_flat`). `seen_parts`
finds, for each member, the pieces of the others on its layer that it
sees, by the rule `neighbours.neighbour_field` builds its field by, as a
`seen.Seen`. `members_energy` sums each member's energy in its static
field plus what it sees of the others, and gives its gradient at every
vertex and via centre: where a member's obstacle term pulls on another's
edge, the gradient reaches that edge's ends, and through a section's
offset its centreline (`PairBlock.offset_pullback`). `layer_energy` is its
form for one layer without vias, and `pull_answers` takes a via's push
back to what answers it.
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
    edge_integral,
    obstacle_integrand,
    obstacle_reach,
    quadrature_points,
    shape_energy,
)
from kibends.neighbours import ChainInput, obstacle_spans
from kibends.obstacles import ChainField
from kibends.pair_blocks import PairBlock
from kibends.seen import NeighbourList, Seen, SeenParts, ViaNearest

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
    nearest = seen.nearest(every_query, flat, reach, listed)
    for i, (member, points) in enumerate(zip(members, polylines, strict=True)):
        queries = every_query[i]
        static, static_direction = body_of(member).field.signed_distance(queries, reach)
        near = nearest[i]
        nearer = near.distance < static
        distance = np.where(nearer, near.distance, static)
        direction = np.where(nearer[:, None], near.gradient, static_direction)
        obstacle, obstacle_gradient, slopes = edge_integral(
            points, distance, direction, integrand, k
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
        pulled = np.flatnonzero(nearer & (slopes != 0.0))
        if len(pulled):
            piece = near.edge[pulled]
            start = seen.vertex[piece]
            along = _parameters(queries[pulled], *seen.segments(piece, flat))
            t = seen.start[piece] + along * (seen.end[piece] - seen.start[piece])
            push = slopes[pulled, None] * near.gradient[pulled]
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
    it gives."""
    lines = [
        shown(member, points) for member, points in zip(chains, polylines, strict=True)
    ]
    first = np.cumsum([0, *(len(points) for parts in lines for _, points in parts)])
    seen: list[SeenParts] = []
    for i, member in enumerate(chains):
        own, as_net = viewer_of(member)
        vertices: list[IntArray] = [np.empty(0, dtype=np.intp)]
        starts: list[FloatArray] = [np.empty(0)]
        ends: list[FloatArray] = [np.empty(0)]
        inflations: list[FloatArray] = [np.empty(0)]
        line = 0
        for j, parts in enumerate(lines):
            for other, points in parts:
                line += 1
                if j == i or other.layer != own.layer:
                    continue
                kept = obstacle_spans(
                    own, other, points, reach, polylines[i], as_net=as_net
                )
                if kept is None:
                    continue
                edge, start, end = _pieces(points, kept.spans)
                vertices.append(first[line - 1] + edge)
                starts.append(start)
                ends.append(end)
                inflations.append(np.full(len(edge), kept.inflation))
        seeing = [(k, ring[i]) for k, ring in enumerate(rings) if i in ring]
        vertices.append(np.array([first[-1] + 2 * k for k, _ in seeing], dtype=np.intp))
        starts.append(np.zeros(len(seeing)))
        ends.append(np.ones(len(seeing)))
        inflations.append(np.array([inflation for _, inflation in seeing]))
        seen.append(
            SeenParts(
                np.concatenate(vertices),
                np.concatenate(starts),
                np.concatenate(ends),
                np.concatenate(inflations),
            )
        )
    return Seen(seen, [viewer_of(member)[0].layer for member in chains])


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


def pieces_field(seen: SeenParts, flat: FloatArray) -> ChainField:
    """The pieces of `seen` as obstacles, with the layer's chains at `flat`,
    their polylines one after another."""
    a, b = flat[seen.vertex], flat[seen.vertex + 1]
    along = b - a
    return ChainField(
        a + seen.start[:, None] * along, a + seen.end[:, None] * along, seen.inflation
    )


def _parameters(points: FloatArray, starts: FloatArray, ends: FloatArray) -> FloatArray:
    """Where along each segment from starts[i] to ends[i] its point nearest
    points[i] lies, from 0 to 1."""
    along = ends - starts
    squared = np.einsum("ij,ij->i", along, along)
    projection = np.einsum("ij,ij->i", points - starts, along)
    return np.clip(projection / np.where(squared > 0.0, squared, 1.0), 0.0, 1.0)


def pull_answers(
    near: ViaNearest,
    nearer: BoolArray,
    at: FloatArray,
    push: FloatArray,
    gradients: list[FloatArray],
    via_gradients: FloatArray,
) -> None:
    """Subtract each via's `push` from what `near` found answering it, where
    that is `nearer` than its static copper: the other via, or the ends of a
    member's edge in proportion to where along it the via's centre at `at`
    is nearest."""
    answered = np.flatnonzero(nearer)
    member, vertex = near.member[answered], near.vertex[answered]
    on_via = member < 0
    np.subtract.at(via_gradients, vertex[on_via], push[answered[on_via]])
    for j in np.unique(member[~on_via]):
        edge = answered[member == j]
        first = near.vertex[edge]
        t = _parameters(at[edge], near.starts[edge], near.ends[edge])[:, None]
        gradient = gradients[int(j)]
        np.subtract.at(gradient, first, (1 - t) * push[edge])
        np.subtract.at(gradient, first + 1, t * push[edge])
