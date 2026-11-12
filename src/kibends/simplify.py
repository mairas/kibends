"""Obstacle-aware simplification of optimised polylines, Rust's
(`kibends_core::simplify`, whose docs hold its design).

Each merged segment stays within the chord tolerance of the vertices it
replaces and clear of obstacles; fixed vertices are kept. Several chains
are simplified in turn, each against its own field plus the other chains
on its layer at their latest shapes, those it ignores left out, as the
optimizer leaves them out. A coupled section of a differential pair is
simplified as its centreline, and its two tracks follow it as they do in
the optimizer.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from kibends import _core
from kibends.bursts import DEFAULT_CHORD_TOLERANCE
from kibends.energy import BoolArray, FloatArray, ObstacleField
from kibends.members import core_member
from kibends.neighbours import ChainInput
from kibends.obstacles import core_field

if TYPE_CHECKING:
    from kibends.pair_blocks import PairBlock
    from kibends.results import RunResult

_NO_POINTS = np.empty((0, 2))
"""A chain's direction pins: it has none."""


def simplify(
    polyline: FloatArray,
    fixed: BoolArray,
    field: ObstacleField,
    tolerance: float = DEFAULT_CHORD_TOLERANCE,
) -> FloatArray:
    """Merge consecutive segments while the chord stays within `tolerance` of
    the merged vertices and clear of obstacles; fixed vertices are kept."""
    points = np.asarray(polyline, dtype=float)
    return polyline[
        _core.kept_vertices(
            points, fixed, core_field(field), np.full(len(points), tolerance)
        )
    ]


def simplify_chains(
    chains: Sequence[ChainInput],
    run: RunResult,
    tolerance: float = DEFAULT_CHORD_TOLERANCE,
    *,
    sections: Mapping[int, PairBlock] | None = None,
) -> list[FloatArray]:
    """Simplify the optimised chains of `run`, each against its field and the
    other chains on its layer that it does not ignore, at their latest
    shapes.

    Each of `sections` stands in for the chain of its index, its polyline
    a coupled section's centreline, which keeps clear of the others as the
    block's `facing` in the field of its chain of `chains`. The others see
    its two tracks, offset from its centreline at its latest shape.
    """
    if len(run.chains) != len(chains):
        raise ValueError(
            f"{len(run.chains)} optimised chains for {len(chains)} chain inputs"
        )
    sections = sections or {}
    members: list[_core.Member] = []
    directions: list[FloatArray] = []
    for i, given in enumerate(chains):
        block = sections.get(i)
        if block is None:
            members.append(core_member(given))
            directions.append(_NO_POINTS)
        else:
            chain = replace(block.chain, field=given.field)
            members.append(core_member(replace(block, chain=chain), hinged=False))
            directions.append(block.directions)
    return _core.simplify_chains(
        members,
        [c.polyline for c in run.chains],
        [c.fixed for c in run.chains],
        directions,
        tolerance,
    )
