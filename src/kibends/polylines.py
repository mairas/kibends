"""Measures and checks on a single polyline that the optimizer
(`optimize`) applies to every member, independent of the problem.

`resample` lays each span between fixed vertices out at uniform arc-length
spacing, as every burst starts. `measure` gives a polyline's length and
bend statistics (`results.PolylineStats`), and `count_violations`
counts where it enters its clearance away from its anchors, which the best
state may not add to. `shape_change` is how far a burst moved a
shape, not counting vertices sliding along it, which settles a problem.
`SelfCheck` is what a burst may not do to a coupled section's centreline:
bend an edge newly below the limit. All but `measure` are Rust's
(`kibends_core::polylines`).
"""

import numpy as np

from kibends import _core
from kibends.energy import FloatArray
from kibends.results import PolylineStats

# Below this turning angle a vertex counts as straight (infinite radius).
STRAIGHT_TURN = 1e-9

resample = _core.resample
"""Each span between fixed vertices resampled to uniform arc-length
spacing, the fixed vertices kept exactly, and the new fixed mask. The
first vertex must be fixed."""

count_violations = _core.count_violations
"""Places where a polyline enters the clearance away from its `anchors`,
given the vertices `inside` the clearance and the edges that `cut` into
it, each by more than INSIDE_TOLERANCE, so that a track routed at exactly
the clearance starts with no violations.

Each corner cut, an edge with both ends clear that enters the clearance,
counts once, and so does each run of consecutive vertices inside the
clearance that holds no anchor; only the cut edges with both ends clear
count. A run that holds an anchor in the clearance cannot move away and is
not counted. A run is consecutive in the chain, so inside vertices in two
touching or overlapping obstacles form one. The count says nothing about
how deep a violation goes, and an edge from any run to a clear vertex adds
nothing to it whatever obstacles the edge passes through."""

shape_change = _core.shape_change
"""Local symmetric distance between two polylines, mm: the largest
distance from each vertex of either to the other's edges near the same
arc-length fraction. Vertices sliding along an unchanged shape do not
count."""

SelfCheck = _core.SelfCheck
"""What a burst may not do to a coupled section's centreline: bend an edge
below the limit that was not already there where the burst started."""


def measure(polyline: FloatArray, spacing: float) -> PolylineStats:
    """Length and bend statistics of a polyline.

    The bend radius is the discrete curvature radius l/(4·tan(φ/2)) of the
    polyline resampled at `spacing` with every vertex kept, so a sharp corner
    reports a radius below the spacing however long its edges are.
    """
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    return PolylineStats(
        length=float(lengths.sum()),
        segment_count=len(polyline) - 1,
        tightest_bend_radius=_tightest_bend_radius(
            resample(polyline, np.ones(len(polyline), dtype=bool), spacing)[0]
        ),
    )


def _tightest_bend_radius(points: FloatArray) -> float:
    edges = np.diff(points, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    incoming, outgoing = edges[:-1], edges[1:]
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    turn = np.abs(np.arctan2(cross, np.einsum("ij,ij->i", incoming, outgoing)))
    bending = turn > STRAIGHT_TURN
    if not bending.any():
        return float("inf")
    span = (lengths[:-1] + lengths[1:])[bending]
    return float(np.min(span / (4 * np.tan(turn[bending] / 2))))
