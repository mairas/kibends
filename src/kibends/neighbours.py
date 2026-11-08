"""Selected chains as obstacles to each other.

Each chain has a static field of everything on the board but the selected
chains. The other selected chains on its layer are added to it as obstacles,
one field each, at whatever shape the caller holds them in.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from kibends.energy import (
    MIN_EDGE_LENGTH,
    BoolArray,
    EnergyParams,
    FloatArray,
    ObstacleField,
    arc_lengths,
)
from kibends.obstacles import ChainCentreline, CompositeField, chain_obstacles


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


def obstacle_reach(params: EnergyParams) -> float:
    """Distance from an obstacle beyond which both obstacle terms of the
    energy are zero, mm."""
    return max(params.standoff, params.falloff_range if params.repulsion else 0.0)


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
    return CompositeField([static, *others]) if others else static


def neighbour_fields(
    chains: Sequence[ChainInput],
    i: int,
    others: Iterable[int],
    polylines: Sequence[FloatArray],
    reach: float,
) -> list[ObstacleField]:
    """The chains `others`, at `polylines`, as obstacles to chain i at its
    starting polyline; those left out entirely are dropped."""
    return [
        field
        for j in others
        if (field := neighbour_field(chains[i], chains[j], polylines[j], reach))
        is not None
    ]


def neighbour_field(
    own: ChainInput,
    other: ChainInput,
    points: FloatArray,
    reach: float,
    own_points: FloatArray | None = None,
) -> ObstacleField | None:
    """`other` at `points` as an obstacle to `own` at `own_points`, by default
    its starting polyline, or None if nothing is left."""
    centreline = neighbour_centreline(own, other, points, reach, own_points)
    if centreline is None:
        return None
    me = own.centreline
    return chain_obstacles(me.net, me.width, me.clearance, [centreline])


def neighbour_centreline(
    own: ChainInput,
    other: ChainInput,
    points: FloatArray,
    reach: float,
    own_points: FloatArray | None = None,
) -> ChainCentreline | None:
    """The part of `other` at `points` that is an obstacle to `own`, or None
    if nothing is left.

    A chain of the same net that ends where this one ends continues it, as at
    a neck-down: near that junction both must leave the shared point, so the
    other is left out as far as it lies within its inflation plus `reach` of
    `own` at `own_points`, by default its starting polyline.
    """
    me, them = own.centreline, other.centreline
    if them.net == me.net:
        mine = me.points if own_points is None else own_points
        clear = (me.width + them.width) / 2 + reach
        ends = me.points[[0, -1]]

        def shared(point: FloatArray) -> bool:
            return bool(np.any(np.all(ends == point, axis=1)))

        total = float(arc_lengths(points)[-1])
        start = _junction_cut(mine, points, clear) if shared(them.points[0]) else 0.0
        end = (
            total - _junction_cut(mine, points[::-1], clear)
            if shared(them.points[-1])
            else total
        )
        if start >= end:
            return None
        points = _cut(points, start, end)
    return replace(them, points=points)


def _junction_cut(own: FloatArray, other: FloatArray, clear: float) -> float:
    """Length of `other`, which starts at an end of `own`, to leave out there.

    Beyond it `other` lies at least `clear` from `own`'s first stretch: at an
    angle θ below 90° between the two that takes clear / sin θ, else clear.
    """
    if not np.array_equal(own[0], other[0]):
        own = own[::-1]
    a, b = _heading(own, clear), _heading(other, clear)
    if float(a @ b) <= 0.0:
        return clear
    sine = abs(float(a[0] * b[1] - a[1] * b[0]))
    return clear / sine if sine > 0.0 else math.inf


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
