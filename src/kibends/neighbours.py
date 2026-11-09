"""Selected chains as obstacles to each other.

Each chain has a static field of everything on the board but the selected
chains. The other selected chains on its layer are added to it as obstacles,
one field each, at whatever shape the caller holds them in. Chains of one
net are no obstacles to each other: copper of one net may come close,
overlap or cross. A coupled section's centreline does keep clear of its own
nets' chains, as its other track must, and leaves each out near where it
touches it (`kept_spans`), so that it does not keep clear of the fan-outs
it connects to.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from kibends.energy import (
    MIN_EDGE_LENGTH,
    BoolArray,
    DistanceField,
    FloatArray,
    IntArray,
    ObstacleField,
    arc_lengths,
)
from kibends.obstacles import (
    ChainCentreline,
    ChainField,
    CompositeField,
    chain_inflation,
    chains_field,
)

# Pins of a section and a chain of its net up to this much farther apart than
# their half widths still overlap: touch points are projected in floats. 1 nm,
# KiCad's coordinate resolution.
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
    contacts: tuple[tuple[int, DistanceField], ...] = ()
    """The vertices that keep a pad or via, by index, each with the outside
    of the region it may lie in, held as the board edge is: positive inside.
    Such a vertex is free, but a stop keeps it inside and a burst never
    leaves it outside. A fixed vertex keeps its contact by staying."""


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
) -> list[ObstacleField]:
    """The chains `others`, at `polylines`, as obstacles to chain i; those
    left out entirely are dropped."""
    return [
        field
        for j in others
        if (field := neighbour_field(chains[i], [(chains[j], polylines[j])], reach))
        is not None
    ]


def neighbour_field(
    own: ChainInput,
    shown: Iterable[tuple[ChainInput, FloatArray]],
    reach: float,
    own_points: FloatArray | None = None,
    *,
    as_net: Mapping[str, ChainInput] | None = None,
) -> ObstacleField | None:
    """The chains of `shown`, each at its points, as obstacles to `own` at
    `own_points`, by default its starting polyline, in one field, each as
    `obstacle_spans` gives it; None if nothing is left."""
    parts: list[tuple[ChainCentreline, float]] = []
    for other, points in shown:
        kept = obstacle_spans(own, other, points, reach, own_points, as_net=as_net)
        if kept is not None:
            parts += (
                (part, kept.inflation)
                for part in _centrelines(other, points, kept.spans)
            )
    if not parts:
        return None
    return chains_field(parts)


class ObstacleSpans(NamedTuple):
    """The stretches of another chain that are obstacles to a chain, and
    how far it keeps from them."""

    spans: list[tuple[float, float]] | None
    """Arc lengths along the other's points, as `kept_spans` gives them;
    None for the whole of it."""
    inflation: float
    """The distance kept between the two centrelines, mm."""


def obstacle_spans(
    own: ChainInput,
    other: ChainInput,
    points: FloatArray,
    reach: float,
    own_points: FloatArray | None = None,
    *,
    as_net: Mapping[str, ChainInput] | None = None,
) -> ObstacleSpans | None:
    """The stretches of `other` at `points` that are obstacles to `own` at
    `own_points`, by default its starting polyline; None if none is.

    A chain of `own`'s net is no obstacle. Another net's chain is seen whole,
    inflated by both half widths and the larger clearance
    (`obstacles.chain_inflation`). A coupled section's centreline, `own`
    with `as_net` giving it under either of the pair's nets, keeps clear of
    those nets' chains as of any other, but leaves each out near where it
    touches it as that net's chain (`kept_spans`), widening its leave-out by
    the larger clearance it keeps from the rest.
    """
    me, them = own.centreline, other.centreline
    inflation = chain_inflation(me.net, me.width, me.clearance, them)
    if inflation is None:
        return None
    mine = (as_net or {}).get(them.net)
    if mine is None:
        return ObstacleSpans(None, inflation)
    margin = max(me.clearance, them.clearance)
    spans = kept_spans(mine, other, points, reach, own_points, margin=margin)
    return ObstacleSpans(spans, inflation)


def _centrelines(
    other: ChainInput, points: FloatArray, spans: Sequence[tuple[float, float]] | None
) -> list[ChainCentreline]:
    """`other` at `points`, cut to the arc-length `spans`, by default whole."""
    them = other.centreline
    if spans is None:
        return [replace(them, points=points)]
    return [replace(them, points=_cut(points, start, end)) for start, end in spans]


def kept_spans(
    own: ChainInput,
    other: ChainInput,
    points: FloatArray,
    reach: float,
    own_points: FloatArray | None = None,
    *,
    margin: float,
) -> list[tuple[float, float]] | None:
    """The stretches of `other` at `points` that are obstacles to `own`, as
    arc lengths along `points`; None for the whole of it.

    `own` is a coupled section's centreline as a chain of `other`'s net
    (`PairBlock.as_net`). The two connect where their copper overlaps, and
    both are pinned there: at a fan-out the chain ends at the section's end.
    Near such a touch the section must leave it, so the other is left out
    on each side of its pin as far as it lies within its inflation plus
    `reach` of `own` at `own_points`, by default its starting polyline. The
    section keeps a clearance from the rest of `other`, so its inflation is
    widened by that `margin`.
    """
    me, them = own.centreline, other.centreline
    own_touch, other_touch = _touches(own, other)
    if not own_touch.size:
        return None
    mine = me.points if own_points is None else own_points
    clear = (me.width + them.width) / 2 + reach + margin
    arc = arc_lengths(points)
    own_pins = _pins(me.points, own.fixed, mine, arc_lengths(mine), own_touch, clear)
    other_pins = _pins(them.points, other.fixed, points, arc, other_touch, clear)
    left_out: list[tuple[float, float]] = []
    for a, b in zip(own_touch.tolist(), other_touch.tolist(), strict=True):
        own_pin, other_pin = own_pins[a], other_pins[b]
        own_headings = [heading for _, heading in own_pin.departures]
        offset: FloatArray = points[other_pin.vertex] - mine[own_pin.vertex]
        along = float(arc[other_pin.vertex])
        for forward, heading in other_pin.departures:
            length = _leave_out(own_headings, offset, heading, clear)
            left_out.append(
                (along, along + length) if forward else (along - length, along)
            )
    return _complement(left_out, float(arc[-1]))


def _touches(own: ChainInput, other: ChainInput) -> tuple[IntArray, IntArray]:
    """Pairs of pinned points, one of each chain, whose copper overlaps, by
    their order among each chain's pinned points."""
    mine = own.centreline.points[own.fixed]
    theirs = other.centreline.points[other.fixed]
    overlap = (own.centreline.width + other.centreline.width) / 2 + TOUCH_TOLERANCE
    gaps = np.linalg.norm(mine[:, None] - theirs[None], axis=2)
    i, j = np.nonzero(gaps <= overlap)
    return i, j


class _Pin(NamedTuple):
    """A pinned point of a chain where it touches another."""

    vertex: int
    """Its vertex of the chain's current polyline."""
    departures: list[tuple[bool, FloatArray]]
    """Each way the chain leaves it, as `_departures` gives them."""


def _pins(
    original: FloatArray,
    fixed: BoolArray,
    points: FloatArray,
    arc: FloatArray,
    touching: IntArray,
    clear: float,
) -> dict[int, _Pin]:
    """Each pinned point in `touching`, by its order among the pinned points
    of the polyline that started as `original`, found on `points`, whose
    arc lengths are `arc`."""
    pinned = original[fixed]
    found: dict[int, _Pin] = {}
    for a in np.unique(touching).tolist():
        k = _vertex_at(original, points, pinned[a])
        found[a] = _Pin(k, _departures(points, arc, k, clear))
    return found


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
    points: FloatArray, arc: FloatArray, k: int, length: float
) -> list[tuple[bool, FloatArray]]:
    """Each way the polyline, with arc lengths `arc`, continues from vertex
    k: whether it is forward, and its unit direction to the point `length`
    along that way, or to the end if it is shorter."""
    departures: list[tuple[bool, FloatArray]] = []
    if k < len(points) - 1:
        at = min(float(arc[k]) + length, float(arc[-1]))
        departures.append((True, _heading(points, arc, k, at)))
    if k > 0:
        departures.append(
            (False, _heading(points, arc, k, max(float(arc[k]) - length, 0.0)))
        )
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


def _heading(points: FloatArray, arc: FloatArray, k: int, at: float) -> FloatArray:
    """Unit direction from vertex k to the point at arc length `at`."""
    after = int(np.searchsorted(arc, at))
    ahead = points[after]
    if after > 0 and arc[after] > at:
        before = after - 1
        part = (at - arc[before]) / (arc[after] - arc[before])
        ahead = points[before] + (points[after] - points[before]) * part
    step: FloatArray = ahead - points[k]
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
