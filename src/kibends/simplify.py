"""Obstacle-aware simplification of an optimised polyline.

A chord around a convex obstacle always deviates toward it, so merging by
chord tolerance alone would cut into the clearance at every wrap; each merged
segment must also pass the exact edge check.
"""

import numpy as np

from kibends.energy import BoolArray, FloatArray
from kibends.optimize import ObstacleField

DEFAULT_CHORD_TOLERANCE = 0.005
"""mm: above the optimiser's residual waviness on straight runs (about 2 µm
with the default settings), far below fabrication tolerances."""


def simplify(
    polyline: FloatArray,
    fixed: BoolArray,
    field: ObstacleField,
    tolerance: float = DEFAULT_CHORD_TOLERANCE,
) -> FloatArray:
    """Merge consecutive segments while the chord stays within `tolerance` of
    the merged vertices and clear of obstacles; fixed vertices are kept."""
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
    return polyline[kept]


def _mergeable(
    polyline: FloatArray, start: int, end: int, field: ObstacleField, tolerance: float
) -> bool:
    a, b = polyline[start], polyline[end]
    chord = b - a
    squared = max(float(chord @ chord), np.finfo(float).tiny)
    inner = polyline[start + 1 : end]
    t = np.clip((inner - a) @ chord / squared, 0.0, 1.0)
    deviation = np.linalg.norm(inner - (a + t[:, None] * chord), axis=1)
    if deviation.max() > tolerance:
        return False
    return bool(field.segments_clear(a[None], b[None])[0])
