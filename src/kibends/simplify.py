"""Obstacle-aware simplification of optimised polylines.

A chord around a convex obstacle always deviates toward it, so merging by
chord tolerance alone would cut into the clearance at every wrap; each merged
segment must also pass the exact edge check.

Several chains are simplified in turn, each against its own field plus the
other chains on its layer at their latest shapes.
"""

from collections.abc import Sequence

import numpy as np
from loguru import logger

from kibends.energy import BoolArray, EnergyParams, FloatArray, ObstacleField
from kibends.neighbours import (
    ChainInput,
    chain_field,
    neighbour_fields,
    obstacle_reach,
    same_layer_neighbours,
)
from kibends.obstacles import CompositeField
from kibends.optimize import RunResult

DEFAULT_CHORD_TOLERANCE = 0.025
"""mm: well below fabrication tolerances; at 5 µm a 23 mm chain on a real
board came out as 63 tracks."""


def simplify(
    polyline: FloatArray,
    fixed: BoolArray,
    field: ObstacleField,
    tolerance: float = DEFAULT_CHORD_TOLERANCE,
) -> FloatArray:
    """Merge consecutive segments while the chord stays within `tolerance` of
    the merged vertices and clear of obstacles; fixed vertices are kept."""
    return polyline[_kept(polyline, fixed, field, tolerance)]


def simplify_chains(
    chains: Sequence[ChainInput],
    run: RunResult,
    params: EnergyParams,
    tolerance: float = DEFAULT_CHORD_TOLERANCE,
) -> list[FloatArray]:
    """Simplify the optimised chains of `run`, in the order of `chains`, each
    against its field and the other chains on its layer at their latest shapes.

    Since the pairwise clearance is symmetric, every merged edge keeps
    clear of the final neighbours; a chain whose merged edges do not is
    logged as an error and left as optimised.

    Neighbours of one net are trimmed near shared ends by the reach of
    `params`, measured against each chain's starting polyline, where the
    optimizer measures against the chain's current one. That is deliberate:
    trimming against the optimised polyline left more segments at acute
    junctions when tried (https://github.com/mairas/kibends/issues/17).
    """
    if len(run.chains) != len(chains):
        raise ValueError(
            f"{len(run.chains)} optimised chains for {len(chains)} chain inputs"
        )
    reach = obstacle_reach(params)
    neighbours = same_layer_neighbours(chains)
    optimised = [c.polyline for c in run.chains]
    current = list(optimised)
    kept: list[list[int]] = []
    for i, result in enumerate(run.chains):
        field = chain_field(
            chains[i].field,
            neighbour_fields(chains, i, neighbours[i], current, reach),
        )
        kept.append(_kept(result.polyline, result.fixed, field, tolerance))
        current[i] = result.polyline[kept[i]]
    for i in cutting_chains(chains, optimised, current, kept, reach):
        logger.error(
            f"Chain {chains[i].centreline.description} left unsimplified at"
            f" {len(optimised[i]) - 1} tracks: its simplified edges cut into"
            " a neighbour's clearance"
        )
        current[i] = optimised[i]
    return current


def cutting_chains(
    chains: Sequence[ChainInput],
    optimised: Sequence[FloatArray],
    simplified: Sequence[FloatArray],
    kept: Sequence[Sequence[int]],
    reach: float,
) -> list[int]:
    """Chains whose merged edges, `optimised[i][kept[i]]` spanning more than
    one optimised edge, enter a neighbour at its `simplified` shape."""
    cutting: list[int] = []
    neighbours = same_layer_neighbours(chains)
    for i, indices in enumerate(np.asarray(k) for k in kept):
        merged = np.diff(indices) > 1
        if not (merged.any() and neighbours[i]):
            continue
        starts = optimised[i][indices[:-1][merged]]
        ends = optimised[i][indices[1:][merged]]
        field = CompositeField(
            neighbour_fields(chains, i, neighbours[i], simplified, reach)
        )
        if not field.segments_clear(starts, ends).all():
            cutting.append(i)
    return cutting


def _kept(
    polyline: FloatArray, fixed: BoolArray, field: ObstacleField, tolerance: float
) -> list[int]:
    kept = [0]
    last = len(polyline) - 1
    start = 0
    while start < last:
        end = start + 1
        while (
            end < last
            and not fixed[end]
            and _mergeable(polyline, start, end + 1, field, tolerance)
        ):
            end += 1
        kept.append(end)
        start = end
    return kept


def _mergeable(
    polyline: FloatArray, start: int, end: int, field: ObstacleField, tolerance: float
) -> bool:
    a: FloatArray = polyline[start]
    b: FloatArray = polyline[end]
    chord = b - a
    squared = max(float(chord @ chord), np.finfo(float).tiny)
    inner = polyline[start + 1 : end]
    t = np.clip((inner - a) @ chord / squared, 0.0, 1.0)
    deviation = np.linalg.norm(inner - (a + t[:, None] * chord), axis=1)
    if deviation.max() > tolerance:
        return False
    return bool(field.segments_clear(a[None], b[None])[0])
