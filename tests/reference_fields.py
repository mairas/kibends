"""Fields built from chains' centrelines and vias by the optimizer's
inflation rules, for tests that check those rules against the board's
obstacles or use them as a reference."""

from collections.abc import Callable, Iterable, Sequence

import numpy as np

from kibends.obstacles import (
    HOLE_CLEARANCE,
    HOLE_TO_HOLE,
    ChainCentreline,
    ChainField,
    chain_inflation,
    chains_field,
    via_chain_inflation,
    via_via_inflation,
)
from kibends.pairs import diff_pair_nets
from kibends.snapshot import Via, ViaZone


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
