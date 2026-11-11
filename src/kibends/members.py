"""The members of a problem, what they see of each other and their summed
energy, independent of the problem that holds them (`optimize._Problem`).

A `Member` is a chain or a coupled section as its centreline. The others
see a chain as itself and a section as its two tracks (`shown`), laid one
after another, then each via's centre twice (`shown_flat`). `seen_parts`
gives what each member sees of the others on its layer, by the rule
`neighbours.neighbour_field` builds its field by, as a `seen.Seen`: the
rule by net and kind, and as exceptions only the vias, so that it grows
with the members, not with pairs of them. `members_energy` sums each
member's energy in its static field plus what it sees of the others, the
distance its obstacle term takes the smooth minimum over both
(`energy.smooth_minimum`), each track and each via an obstacle, and gives
its gradient at every vertex and via centre: where a member's obstacle
term pulls on another's edge, each obstacle's share of the gradient
reaches the ends of its edge nearest the point, and through a section's
offset its centreline (`PairBlock.offset_pullback`). `pull_answers` takes
a via's push back to what answers it, by the same shares.
`members_stiffness` gives each vertex's stiffness under its obstacle term,
by which a burst scales its variables (`optimize._scale`).
"""

import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

from kibends.energy import (
    BoolArray,
    EnergyParams,
    FloatArray,
    IntArray,
    bend_radius_energy,
    combine_fields,
    edge_curvature,
    edge_integral,
    obstacle_curvature,
    obstacle_integrand,
    obstacle_reach,
    quadrature_points,
    shape_energy,
)
from kibends.neighbours import ChainInput
from kibends.obstacles import ChainCentreline, SmoothAnswer, track_inflation
from kibends.pair_blocks import PairBlock
from kibends.pairs import diff_pair_nets
from kibends.seen import (
    NeighbourList,
    Pieces,
    Seen,
    StaticList,
    Track,
    ViaSmooth,
    Viewer,
)

Member = ChainInput | PairBlock
"""What a problem optimises: a chain, or a coupled section as its
centreline."""


def members_energy(
    members: Sequence[Member],
    polylines: Sequence[FloatArray],
    masks: Sequence[BoolArray],
    params: EnergyParams,
    short_edge: float,
    seen: Seen,
    positions: Sequence[FloatArray],
    listed: NeighbourList | None = None,
    statics: StaticList | None = None,
) -> tuple[float, list[FloatArray], FloatArray]:
    """The summed energy of chains and sections on any layers, each in its
    static field plus what it sees of the others on its own layer at
    `polylines` as `seen` shows them, with vias at `positions` among them,
    a section with its bend-radius hinge; its gradient at every vertex of
    every chain and centreline, with the rows of `masks` zeroed; and the
    gradient at each via's centre. Raises ValueError if it is not finite.

    A section's centreline is what it keeps clear as, and its two tracks,
    offset from it, are what the others keep clear of: their part of the
    gradient is taken back through the offset (`PairBlock.offset_pullback`).

    With `listed`, made for `seen`, each layer looks only at the pieces it
    lists; with `statics`, made for the members' static fields with the
    energy's reach, the static fields answer from it, by default from one
    search each."""
    integrand = obstacle_integrand(params)
    k = params.points_per_edge
    lines = [
        shown(member, points) for member, points in zip(members, polylines, strict=True)
    ]
    flat = _flat(lines, positions)
    at_shown = np.zeros_like(flat)
    gradients: list[FloatArray] = []
    value = 0.0
    every_query, answers, bounds, distance, direction, seen_share = _fields(
        members, polylines, params, seen, flat, listed, statics
    )
    no_points = np.empty((0, 2))
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
    pull = (slopes * seen_share)[point] * _joined_rows(
        [near.share for near in answers], np.empty(0)
    )
    pulled = np.flatnonzero(pull != 0.0)
    if len(pulled):
        piece = np.concatenate(
            [np.empty(0, dtype=np.intp), *(near.edge for near in answers)]
        )[pulled]
        toward = _joined_rows([near.direction for near in answers], no_points)
        start = seen.vertex[piece]
        at = _joined_rows(every_query, no_points)[point[pulled]]
        along = _parameters(at, *seen.segments(piece, flat))
        push = pull[pulled, None] * toward[pulled]
        np.add.at(at_shown, start, -push * (1 - along)[:, None])
        np.add.at(at_shown, start + 1, -push * along[:, None])
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


def members_stiffness(
    members: Sequence[Member],
    polylines: Sequence[FloatArray],
    params: EnergyParams,
    seen: Seen,
    positions: Sequence[FloatArray],
    listed: NeighbourList | None = None,
    statics: StaticList | None = None,
) -> list[FloatArray]:
    """Per member, each vertex's stiffness under its obstacle term at
    `polylines` (`energy.edge_curvature`), with what it sees of its static
    field and the others as `members_energy` finds it, `listed` and
    `statics` as there. Only the member's own quadrature points count:
    where its edges are what another member's term sees, the curvature of
    that pull is left out."""
    flat = shown_flat(members, polylines, positions)
    _, _, bounds, distance, direction, _ = _fields(
        members, polylines, params, seen, flat, listed, statics
    )
    integrand = obstacle_integrand(params)
    curvature = obstacle_curvature(params)
    return [
        edge_curvature(
            points,
            distance[bounds[i] : bounds[i + 1]],
            direction[bounds[i] : bounds[i + 1]],
            integrand,
            curvature,
            params.points_per_edge,
        )
        for i, points in enumerate(polylines)
    ]


class _Fields(NamedTuple):
    """What every member's quadrature points see, one member after another."""

    queries: list[FloatArray]
    """Per member, its quadrature points."""
    answers: list[SmoothAnswer]
    """Per member, what its points see of the others."""
    bounds: IntArray
    """Where each member's points start, and where the last one's end."""
    distance: FloatArray
    """The smooth minimum over the static field and the others."""
    direction: FloatArray
    """Its gradient."""
    seen_share: FloatArray
    """The others' share of it."""


def _fields(
    members: Sequence[Member],
    polylines: Sequence[FloatArray],
    params: EnergyParams,
    seen: Seen,
    flat: FloatArray,
    listed: NeighbourList | None,
    statics: StaticList | None,
) -> _Fields:
    """The distances of every member's quadrature points at `polylines`,
    the others shown as `flat`, for `members_energy` and
    `members_stiffness`."""
    reach = obstacle_reach(params)
    k = params.points_per_edge
    every_query = [quadrature_points(points, k) for points in polylines]
    answers = seen.smooth_answers(every_query, flat, reach, listed)
    if statics is None:
        statics = static_list(members, reach, listed=False)
    static = statics.smooth(every_query)
    bounds = np.cumsum([0, *(len(queries) for queries in every_query)])
    distance, (static_share, seen_share) = combine_fields(
        _joined_rows([d for d, _ in static], np.empty(0)),
        _joined_rows([near.distance for near in answers], np.empty(0)),
    )
    no_points = np.empty((0, 2))
    direction = static_share[:, None] * _joined_rows(
        [g for _, g in static], no_points
    ) + seen_share[:, None] * _joined_rows(
        [near.gradient for near in answers], no_points
    )
    return _Fields(every_query, answers, bounds, distance, direction, seen_share)


def static_list(
    members: Sequence[Member], reach: float, *, listed: bool = True
) -> StaticList:
    """The members' static fields as a `StaticList` with the energy's
    `reach`, for `members_energy`."""
    return StaticList(
        [body_of(member).field for member in members], reach, listed=listed
    )


def _joined_rows(arrays: Sequence[FloatArray], empty: FloatArray) -> FloatArray:
    """`arrays` one after another, `empty` if there are none; one of them
    as it is."""
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate([empty, *arrays])


def shown(member: Member, points: FloatArray) -> list[tuple[ChainInput, FloatArray]]:
    """What the others see of a member at `points`: a chain itself, a
    section its two tracks."""
    if isinstance(member, PairBlock):
        return list(zip(member.tracks, member.offset(points), strict=True))
    return [(member, points)]


def body_of(member: Member) -> ChainInput:
    """The chain a member's own energy and pins are those of."""
    return member.chain if isinstance(member, PairBlock) else member


def viewer_of(member: Member) -> tuple[ChainInput, frozenset[str]]:
    """The chain a member keeps clear of the others as, and the nets whose
    tracks it ignores, as `neighbours.neighbour_field` takes them: a
    chain's own and the other net of its diff pair, a section's two."""
    if isinstance(member, PairBlock):
        return member.facing, member.nets
    return member, diff_pair_nets(member.centreline.net)


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
    rings: Sequence[Mapping[int, float]] = (),
) -> Seen:
    """For each chain or section at its polyline, the others on its layer at
    theirs as `neighbours.neighbour_field` shows them to it. A section sees
    as its centreline of the diff pair's outer extent, and is seen as its two
    tracks. Each via of `rings` is seen by the members it names, inflated
    by the distance it gives.

    The tracks of the nets a member does not ignore (`viewer_of`) are seen
    whole, by the rule `seen.Seen` applies; only the vias are listed, as
    its exceptions, so the work grows with the members and their edges, not
    with pairs of members."""
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
    for parts in lines:
        for other, points in parts:
            them, start = other.centreline, int(first[len(tracks)])
            tracks.append(
                Track(start, len(points), other.layer, net(them.net), kind(them))
            )
    viewers: list[Viewer] = []
    for member in chains:
        own, ignored = viewer_of(member)
        me = own.centreline
        # A net no member shows has no tracks to ignore.
        exempt = tuple(sorted(nets[n] for n in ignored if n != me.net and n in nets))
        viewers.append(Viewer(own.layer, net(me.net), kind(me), exempt))
    exceptions: list[list[Pieces]] = [[] for _ in chains]
    for k, ring in enumerate(rings):
        at = np.array([first[-1] + 2 * k], dtype=np.intp)
        for i, inflation in ring.items():
            exceptions[i].append(Pieces(at, np.array([inflation])))
    every = list(kinds)
    table = np.array(
        [[track_inflation(*a, *b) for b in every] for a in every], dtype=float
    ).reshape(len(every), len(every))
    return Seen(tracks, viewers, table, [_joined(pieces) for pieces in exceptions])


def _joined(pieces: Sequence[Pieces]) -> Pieces:
    """`pieces` one after another."""
    return Pieces(
        np.concatenate([np.empty(0, dtype=np.intp), *(p.vertex for p in pieces)]),
        np.concatenate([np.empty(0), *(p.inflation for p in pieces)]),
    )


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
