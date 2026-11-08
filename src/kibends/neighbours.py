"""Selected chains as obstacles to each other.

Each chain has a static field of everything on the board but the selected
chains. The other selected chains on its layer are added to it as obstacles,
one field each, at whatever shape the caller holds them in. Where two
chains of one net touch, each leaves the other out near the touch, so that
neither keeps clear of the copper it connects to.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from kibends.energy import (
    MIN_EDGE_LENGTH,
    BoolArray,
    FloatArray,
    ObstacleField,
    arc_lengths,
)
from kibends.obstacles import (
    ChainCentreline,
    ChainField,
    CompositeField,
    chain_obstacles,
)

# Pins of two chains of one net up to this much farther apart than their half
# widths still overlap: touch points are projected in floats. 1 nm, KiCad's
# coordinate resolution.
TOUCH_TOLERANCE = 1e-6


@dataclass(frozen=True, eq=False)
class ChainInput:
    """One chain of a run, with the field of everything but the selected chains."""

    centreline: ChainCentreline
    """Its description, net, width, clearance and starting polyline."""
    layer: str
    """Only chains on the same layer see each other."""
    fixed: BoolArray
    field: ObstacleField
    """Obstacles for its net and width, without any selected chain."""


def same_layer_neighbours(chains: Sequence[ChainInput]) -> list[list[int]]:
    """For each chain, the indices of the other chains on its layer."""
    return [
        [j for j, other in enumerate(chains) if j != i and other.layer == chain.layer]
        for i, chain in enumerate(chains)
    ]


def chain_field(
    static: ObstacleField, neighbours: Iterable[ObstacleField | None]
) -> ObstacleField:
    """A chain's field: `static` alone, or with the neighbour fields that are
    not None."""
    others = [field for field in neighbours if field is not None]
    chains = [field for field in others if isinstance(field, ChainField)]
    if len(chains) > 1:
        rest = [field for field in others if not isinstance(field, ChainField)]
        others = [ChainField.merged(chains), *rest]
    return CompositeField([static, *others]) if others else static


def neighbour_fields(
    chains: Sequence[ChainInput],
    i: int,
    others: Iterable[int],
    polylines: Sequence[FloatArray],
    reach: float,
    own_points: FloatArray | None = None,
) -> list[ObstacleField]:
    """The chains `others`, at `polylines`, as obstacles to chain i at
    `own_points`, by default its starting polyline; those left out entirely
    are dropped."""
    return [
        field
        for j in others
        if (
            field := neighbour_field(
                chains[i], [(chains[j], polylines[j])], reach, own_points
            )
        )
        is not None
    ]


def with_moved_ends(start: FloatArray, current: FloatArray) -> FloatArray:
    """`start` with its ends where `current` has them, which differ only
    where a moving via carried them."""
    moved = start.copy()
    moved[0], moved[-1] = current[0], current[-1]
    return moved


def neighbour_field(
    own: ChainInput,
    shown: Iterable[tuple[ChainInput, FloatArray]],
    reach: float,
    own_points: FloatArray | None = None,
    *,
    as_net: Mapping[str, ChainInput] | None = None,
) -> ObstacleField | None:
    """The chains of `shown`, each at its points, as obstacles to `own` at
    `own_points`, by default its starting polyline, in one field; None if
    nothing is left.

    A chain of a net in `as_net`, by default only `own`'s, is left out near
    where it touches `own` as that net's chain (`neighbour_centrelines`).
    Where that stands for `own` under another net, as a pair's centreline
    stands for either track, the chain keeps the larger of the two
    clearances from the rest of it.
    """
    me = own.centreline
    nets = {me.net: own} if as_net is None else as_net
    parts: list[ChainCentreline] = []
    for other, points in shown:
        them = other.centreline
        mine = nets.get(them.net)
        if mine is None:
            parts.append(replace(them, points=points))
            continue
        margin = 0.0 if them.net == me.net else max(me.clearance, them.clearance)
        parts += neighbour_centrelines(
            mine, other, points, reach, own_points, margin=margin
        )
    if not parts:
        return None
    return chain_obstacles(me.net, me.width, me.clearance, parts)


def neighbour_centrelines(
    own: ChainInput,
    other: ChainInput,
    points: FloatArray,
    reach: float,
    own_points: FloatArray | None = None,
    margin: float = 0.0,
) -> list[ChainCentreline]:
    """The parts of `other` at `points` that are obstacles to `own`.

    Chains of one net connect where their copper overlaps, and both are
    pinned there: at a neck-down both end at one point, at a T one ends in
    the other's copper. Near such a touch both must leave it, so the other
    is left out on each side of its pin as far as it lies within its
    inflation plus `reach` of `own` at `own_points`, by default its starting
    polyline. A caller that keeps a clearance from the rest of `other`
    widens its inflation by that `margin`.
    """
    me, them = own.centreline, other.centreline
    whole = [replace(them, points=points)]
    if them.net != me.net:
        return whole
    touches = _touches(own, other)
    if not touches:
        return whole
    mine = me.points if own_points is None else own_points
    clear = (me.width + them.width) / 2 + reach + margin
    arc = arc_lengths(points)
    left_out: list[tuple[float, float]] = []
    for at_own, at_other in touches:
        own_pin = _vertex_at(me.points, mine, at_own)
        other_pin = _vertex_at(them.points, points, at_other)
        own_headings = [heading for _, heading in _departures(mine, own_pin, clear)]
        offset: FloatArray = points[other_pin] - mine[own_pin]
        along = float(arc[other_pin])
        for forward, heading in _departures(points, other_pin, clear):
            length = _leave_out(own_headings, offset, heading, clear)
            left_out.append(
                (along, along + length) if forward else (along - length, along)
            )
    return [
        replace(them, points=_cut(points, start, end))
        for start, end in _complement(left_out, float(arc[-1]))
    ]


def _touches(own: ChainInput, other: ChainInput) -> list[tuple[FloatArray, FloatArray]]:
    """Pairs of pinned points, one of each chain, whose copper overlaps."""
    mine = own.centreline.points[own.fixed]
    theirs = other.centreline.points[other.fixed]
    overlap = (own.centreline.width + other.centreline.width) / 2 + TOUCH_TOLERANCE
    gaps = np.linalg.norm(mine[:, None] - theirs[None], axis=2)
    i, j = np.nonzero(gaps <= overlap)
    return [(mine[a], theirs[b]) for a, b in zip(i, j, strict=True)]


def _vertex_at(original: FloatArray, points: FloatArray, pin: FloatArray) -> int:
    """The vertex of `points` at `pin`, a pinned point of the polyline that
    started as `original`.

    A pinned point stays a vertex throughout, and stays put, except an end
    that a moving via carries away: the ends are found by being ends.
    """
    if np.array_equal(pin, original[0]):
        return 0
    if np.array_equal(pin, original[-1]):
        return len(points) - 1
    return int(np.argmin(np.sum((points - pin) ** 2, axis=1)))


def _departures(
    points: FloatArray, k: int, length: float
) -> list[tuple[bool, FloatArray]]:
    """Each way the polyline continues from vertex k: whether it is forward,
    and its unit direction."""
    departures: list[tuple[bool, FloatArray]] = []
    if k < len(points) - 1:
        departures.append((True, _heading(points[k:], length)))
    if k > 0:
        departures.append((False, _heading(points[k::-1], length)))
    return departures


def _leave_out(
    own_headings: Sequence[FloatArray],
    offset: FloatArray,
    other_heading: FloatArray,
    clear: float,
) -> float:
    """How far along `other_heading` the other chain, leaving its pin at
    `offset` from this chain's pin, stays within `clear` of this one.

    Each stretch of this chain leaving its pin, along one of `own_headings`, is
    taken as a ray, and the other chain as a ray from its pin: the length is
    where it last leaves the band of width `clear` around any of them. From
    one shared point, at an angle θ below 90° between the two, that is
    clear / sin θ, else clear.
    """
    along_other = float(offset @ other_heading)
    # The disc of radius `clear` around this chain's pin.
    discriminant = along_other**2 - float(offset @ offset) + clear**2
    length = -along_other + math.sqrt(discriminant) if discriminant >= 0.0 else 0.0
    for heading in own_headings:
        cosine = float(other_heading @ heading)
        start = float(offset @ heading)
        side = _cross(offset, heading)
        spread = _cross(other_heading, heading)
        if spread == 0.0:
            if abs(side) <= clear and cosine > 0.0:
                return math.inf
            continue
        # The band beside the ray: |side + t·spread| <= clear, start + t·cosine >= 0.
        last = max((clear - side) / spread, (-clear - side) / spread)
        if cosine < 0.0:
            last = min(last, -start / cosine)
        elif cosine == 0.0 and start < 0.0:
            continue
        first = min((clear - side) / spread, (-clear - side) / spread)
        if cosine > 0.0:
            first = max(first, -start / cosine)
        if first <= last:
            length = max(length, last)
    return max(length, 0.0)


def _cross(u: FloatArray, v: FloatArray) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def _complement(
    intervals: Sequence[tuple[float, float]], total: float
) -> list[tuple[float, float]]:
    """The parts of [0, total] outside every interval, each of positive length."""
    kept: list[tuple[float, float]] = []
    start = 0.0
    for left, right in sorted(intervals):
        if left > start:
            kept.append((start, left))
        start = max(start, right)
        if start >= total:
            return kept
    kept.append((start, total))
    return kept


def _heading(points: FloatArray, length: float) -> FloatArray:
    """Unit direction from the first vertex to the point `length` along."""
    arc = arc_lengths(points)
    at = min(length, float(arc[-1]))
    ahead = np.stack(
        [np.interp([at], arc, points[:, 0]), np.interp([at], arc, points[:, 1])],
        axis=1,
    )
    step: FloatArray = ahead[0] - points[0]
    return step / max(float(np.linalg.norm(step)), MIN_EDGE_LENGTH)


def _cut(points: FloatArray, start: float, end: float) -> FloatArray:
    """The part of a polyline between two arc lengths from its first vertex."""
    arc = arc_lengths(points)
    inside = (arc > start) & (arc < end)
    ends = np.stack(
        [
            np.interp([start, end], arc, points[:, 0]),
            np.interp([start, end], arc, points[:, 1]),
        ],
        axis=1,
    )
    return np.concatenate([ends[:1], points[inside], ends[1:]])
