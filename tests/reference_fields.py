"""Fields built from chains' centrelines and vias by the optimizer's
inflation rules, for tests that check those rules against the board's
obstacles or use them as a reference."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace

import numpy as np

from kibends.energy import FloatArray, ObstacleField
from kibends.neighbours import ChainInput
from kibends.obstacles import (
    HOLE_CLEARANCE,
    HOLE_TO_HOLE,
    ChainCentreline,
    ChainField,
    CompositeField,
    chain_inflation,
    chains_field,
    via_chain_inflation,
    via_via_inflation,
)
from kibends.pairs import diff_pair_nets
from kibends.snapshot import Via, ViaZone


def chain_field(
    static: ObstacleField, neighbours: ObstacleField | None
) -> ObstacleField:
    """A chain's field: `static` alone, or with `neighbours` if not None."""
    return static if neighbours is None else CompositeField([static, neighbours])


def neighbour_field(
    own: ChainInput,
    shown: Iterable[tuple[ChainInput, FloatArray]],
    *,
    ignored: frozenset[str] | None = None,
) -> ObstacleField | None:
    """The chains of `shown`, each at its points and an obstacle of its own,
    as obstacles to `own` in one field, inflated as `obstacles.chain_inflation`
    gives it; None if nothing is left. Chains of the nets `ignored`, by
    default `own`'s and the other net of its diff pair, are no obstacles.
    The reference for what a member sees of the others
    (`kibends_core::members`)."""
    me = own.centreline
    nets = diff_pair_nets(me.net) if ignored is None else ignored
    parts: list[tuple[list[ChainCentreline], float]] = []
    for other, points in shown:
        them = other.centreline
        inflation = chain_inflation(nets, me.width, me.clearance, them)
        if inflation is not None:
            parts.append(([replace(them, points=points)], inflation))
    if not parts:
        return None
    return chains_field(parts)


def chain_obstacles(
    net: str, width: float, clearance: float, others: Sequence[ChainCentreline]
) -> ChainField:
    """`others` as obstacles to a track of `net`, `width` and netclass
    `clearance` (mm), each inflated as `chain_inflation` gives it."""
    ignored = diff_pair_nets(net)
    return chains_field(
        (o, chain_inflation(ignored, width, clearance, o)) for o in others
    )


def chain_obstacles_for_via(
    via: Via,
    others: Sequence[ChainCentreline],
    clearance_of: Callable[[str], float],
    *,
    hole_clearance: float = HOLE_CLEARANCE,
    zones: Sequence[ViaZone] = (),
) -> ChainField:
    """`others` as obstacles to `via`, each inflated as
    `via_chain_inflation` gives it."""
    return chains_field(
        (
            o,
            via_chain_inflation(
                via, o, clearance_of, hole_clearance=hole_clearance, zones=zones
            ),
        )
        for o in others
    )


def via_obstacles_for_via(
    via: Via,
    others: Iterable[Via],
    clearance_of: Callable[[str], float],
    *,
    hole_clearance: float = HOLE_CLEARANCE,
    hole_to_hole: float = HOLE_TO_HOLE,
    zones: Sequence[ViaZone] = (),
) -> ChainField | None:
    """`others`, at their positions, as obstacles to `via` by its own
    `zones` alone, each inflated as `via_via_inflation` gives it; `via`
    itself among them is left out, and None if none is one."""
    centres: list[tuple[float, float]] = []
    inflations: list[float] = []
    for other in others:
        if other.id == via.id:
            continue
        inflation = via_via_inflation(
            via,
            other,
            clearance_of,
            hole_clearance=hole_clearance,
            hole_to_hole=hole_to_hole,
            zones=zones,
        )
        if inflation is not None:
            centres.append(other.position)
            inflations.append(inflation)
    if not centres:
        return None
    at = np.array(centres, dtype=float)
    # Each via is an obstacle of its own.
    return ChainField(
        at,
        at,
        np.array(inflations, dtype=float),
        np.arange(len(centres), dtype=np.intp),
    )
