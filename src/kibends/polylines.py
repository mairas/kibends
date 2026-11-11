"""Measures and checks on a single polyline that the optimizer
(`optimize`) applies to every member, independent of the problem.

`resample` lays each span between fixed vertices out at uniform arc-length
spacing, as every burst starts. `measure` gives a polyline's length and
bend statistics (`results.PolylineStats`), and `violations`
counts where it enters its static clearance away from its anchors, which
the best state may not add to. `shape_change` is how far a burst moved a
shape, not counting vertices sliding along it, which settles a problem.
`SelfCheck` is what a burst may not do to a coupled section's centreline:
bend an edge newly below the limit.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from kibends.bursts import INSIDE_TOLERANCE
from kibends.energy import (
    MIN_EDGE_LENGTH,
    BoolArray,
    FloatArray,
    ObstacleField,
    arc_lengths,
)
from kibends.pairs import bend_radii, too_tight
from kibends.results import PolylineStats

# A span only slightly longer than a multiple of the spacing gets no extra
# vertex, so float noise in re-measured lengths does not change the count.
RESAMPLE_SLACK = 1e-6

# Below this turning angle a vertex counts as straight (infinite radius).
STRAIGHT_TURN = 1e-9

# Shape change is measured from each vertex to the other polyline's edges
# within this many edges of the same index.
SHAPE_WINDOW = 3

# Offsets from edge k of the vertices its bend radius depends on: the turns
# at both its ends take the vertices on either side.
BEND_REACH = np.arange(-1, 3)


def violations(points: FloatArray, anchors: BoolArray, field: ObstacleField) -> int:
    """Places where the polyline enters the clearance away from its anchors.

    Each corner cut, an edge with both ends clear that passes through an
    obstacle, counts once, and so does each run of consecutive vertices
    inside the clearance that holds no anchor. A run that holds an anchor
    in the clearance cannot move away and is not counted.
    A run is consecutive in the chain, so inside vertices in two touching
    or overlapping obstacles form one. The count says nothing about how deep
    a violation goes, and an edge from any run to a clear vertex adds nothing
    to it whatever obstacles the edge passes through. Vertices and edges
    within INSIDE_TOLERANCE of the clearance count as clear, so a track
    routed at exactly the clearance starts with no violations.

    This is the count's definition, against one field. The optimizer asks
    its fields for every member at once and counts with `count_violations`
    (`optimize._Problem.violations`); the tests hold the two to the same
    answer.
    """
    distance, _ = field.signed_distance(points)
    inside = distance < -INSIDE_TOLERANCE
    ends_clear = ~inside[:-1] & ~inside[1:]
    cut = np.zeros(len(ends_clear), dtype=bool)
    cut[ends_clear] = ~field.segments_clear(
        points[:-1][ends_clear], points[1:][ends_clear], INSIDE_TOLERANCE
    )
    return count_violations(inside, cut, anchors)


def count_violations(inside: BoolArray, cut: BoolArray, anchors: BoolArray) -> int:
    """`violations` of a polyline whose vertices `inside` lie in the
    clearance and whose edges `cut` enter it, each by more than
    INSIDE_TOLERANCE; only the cut edges with both ends clear count."""
    ends_clear = ~inside[:-1] & ~inside[1:]
    cuts = np.count_nonzero(cut & ends_clear)
    run_starts = inside & ~np.concatenate(([False], inside[:-1]))
    run = np.cumsum(run_starts)
    loose_runs = np.count_nonzero(run_starts & ~np.isin(run, run[inside & anchors]))
    return int(cuts + loose_runs)


def resample(
    points: FloatArray, fixed: BoolArray, spacing: float
) -> tuple[FloatArray, BoolArray]:
    """Resample each span between fixed vertices to uniform arc-length spacing.

    Fixed vertices are kept exactly; returns the new points and fixed mask.
    """
    anchor_indices = np.flatnonzero(fixed)
    pieces = [points[:1]]
    masks = [np.ones(1, dtype=bool)]
    for start, end in pairwise(anchor_indices):
        span = points[start : end + 1]
        arc = arc_lengths(span)
        total = float(arc[-1])
        count = max(1, int(np.ceil(total / spacing - RESAMPLE_SLACK)))
        targets = np.linspace(0.0, total, count + 1)[1:-1]
        interior = np.stack(
            [np.interp(targets, arc, span[:, 0]), np.interp(targets, arc, span[:, 1])],
            axis=1,
        )
        pieces += [interior, points[end : end + 1]]
        masks += [np.zeros(count - 1, dtype=bool), np.ones(1, dtype=bool)]
    return np.concatenate(pieces), np.concatenate(masks)


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


def shape_change(before: FloatArray, after: FloatArray) -> float:
    """Local symmetric distance between two polylines, mm.

    Vertices sliding along an unchanged shape do not count: resampling to
    uniform spacing and the energy's indifference to where along the track
    a vertex sits would otherwise keep moving vertices forever without
    changing the track.
    """
    return max(_local_distance(after, before), _local_distance(before, after))


def _local_distance(points: FloatArray, polyline: FloatArray) -> float:
    """Largest distance from each point to the polyline's edges near it.

    "Near" is within SHAPE_WINDOW edges of the edge at the same arc-length
    fraction, so the far leg of a hairpin never counts as close.
    """
    last_edge = len(polyline) - 2
    starts, ends = polyline[:-1], polyline[1:]
    edge_starts = _arc_fractions(polyline)[:-1]
    indices = np.searchsorted(edge_starts, _arc_fractions(points), side="right") - 1
    nearest = np.full(len(points), np.inf)
    for offset in range(-SHAPE_WINDOW, SHAPE_WINDOW + 1):
        edge = np.clip(indices + offset, 0, last_edge)
        start, delta = starts[edge], ends[edge] - starts[edge]
        squared = np.einsum("ij,ij->i", delta, delta)
        t = np.einsum("ij,ij->i", points - start, delta) / np.maximum(
            squared, MIN_EDGE_LENGTH**2
        )
        closest: FloatArray = start + np.clip(t, 0.0, 1.0)[:, None] * delta
        nearest = np.minimum(nearest, np.linalg.norm(points - closest, axis=1))
    return float(nearest.max())


def _arc_fractions(points: FloatArray) -> FloatArray:
    arc = arc_lengths(points)
    return arc / max(float(arc[-1]), MIN_EDGE_LENGTH)


@dataclass(frozen=True)
class SelfCheck:
    """What a burst may not do to a coupled section's centreline: bend an
    edge below `bend_limit` that was not already there where the burst
    started (`tight_before`)."""

    bend_limit: float
    tight_before: BoolArray

    @staticmethod
    def of(original: FloatArray, bend_limit: float) -> SelfCheck:
        return SelfCheck(bend_limit, too_tight(bend_radii(original), bend_limit))

    def __call__(self, candidate: FloatArray) -> tuple[BoolArray, int]:
        """The vertices to move back, and how many edges newly bent too
        tightly."""
        back = np.zeros(len(candidate), dtype=bool)
        tight = too_tight(bend_radii(candidate), self.bend_limit) & ~self.tight_before
        around = np.flatnonzero(tight)[:, None] + BEND_REACH
        back[np.clip(around, 0, len(candidate) - 1).ravel()] = True
        return back, int(tight.sum())
