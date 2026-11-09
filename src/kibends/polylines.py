"""Measures and checks on a single polyline that the optimizer
(`optimize`) applies to every member, independent of the problem.

`resample` lays each span between fixed vertices out at uniform arc-length
spacing, as every burst starts. `measure` gives a polyline's length, bend
and clearance statistics (`results.PolylineStats`), and `violations`
counts where it enters its static clearance away from its anchors, which
the best state may not add to. `shape_change` is how far a burst moved a
shape, not counting vertices sliding along it, which settles a problem.
`SelfCheck` is what a burst may not do to a coupled section's centreline:
bend an edge newly below the limit, or bring its parts beyond a turn
closer than they may (`close_to_itself`).
"""

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.spatial import cKDTree

from kibends.bursts import INSIDE_TOLERANCE
from kibends.energy import (
    MIN_EDGE_LENGTH,
    BoolArray,
    DistanceField,
    FloatArray,
    IntArray,
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

MEASURE_SAMPLES_PER_SPACING = 4

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
    inside the clearance that holds no anchor. A run that holds a frozen
    vertex or a pin in the clearance cannot move away and is not counted.
    A run is consecutive in the chain, so inside vertices in two touching
    or overlapping obstacles form one. The count says nothing about how deep
    a violation goes, and an edge from any run to a clear vertex adds nothing
    to it whatever obstacles the edge passes through. Vertices and edges
    within INSIDE_TOLERANCE of the clearance count as clear, so a track
    routed at exactly the clearance starts with no violations.
    """
    distance, _ = field.signed_distance(points)
    inside = distance < -INSIDE_TOLERANCE
    ends_clear = ~inside[:-1] & ~inside[1:]
    cuts = np.count_nonzero(
        ~field.segments_clear(
            points[:-1][ends_clear], points[1:][ends_clear], INSIDE_TOLERANCE
        )
    )
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


def measure(
    polyline: FloatArray, field: DistanceField, spacing: float
) -> PolylineStats:
    """Length, bend and clearance statistics of a polyline.

    The bend radius is the discrete curvature radius l/(4·tan(φ/2)) of the
    polyline resampled at `spacing` with every vertex kept, so a sharp corner
    reports a radius below the spacing however long its edges are.
    """
    edges = np.diff(polyline, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    return PolylineStats(
        length=float(lengths.sum()),
        segment_count=len(polyline) - 1,
        tightest_bend_radius=_tightest_bend_radius(
            resample(polyline, np.ones(len(polyline), dtype=bool), spacing)[0]
        ),
        **_clearance(polyline, lengths, field, spacing / MEASURE_SAMPLES_PER_SPACING),
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


def _clearance(
    polyline: FloatArray, lengths: FloatArray, field: DistanceField, step: float
) -> dict[str, float]:
    counts = np.maximum(1, np.ceil(lengths / step).astype(int))
    distances: list[FloatArray] = []
    weights: list[FloatArray] = []
    for start, end, count, length in zip(
        polyline[:-1], polyline[1:], counts, lengths, strict=True
    ):
        t = np.linspace(0.0, 1.0, count + 1)
        distances.append(field.signed_distance(start + t[:, None] * (end - start))[0])
        edge_weights = np.full(count + 1, length / count)
        edge_weights[[0, -1]] /= 2
        weights.append(edge_weights)
    distance = np.concatenate(distances)
    weight = np.concatenate(weights)
    total = weight.sum()
    mean = float(distance @ weight / total) if total > 0.0 else float(distance.mean())
    return {"min_clearance": float(distance.min()), "mean_clearance": mean}


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


def close_to_itself(points: FloatArray, apart: float) -> tuple[IntArray, FloatArray]:
    """The pairs of vertices closer than `apart` that lie more than half a
    circle `apart` across from each other along the polyline, and their
    distances: there it comes back onto itself rather than turning, as a
    turn tighter than that circle would."""
    pairs = cKDTree(points).query_pairs(apart, output_type="ndarray")
    if not len(pairs):
        return pairs, np.empty(0)
    arc = arc_lengths(points)
    i, j = pairs[:, 0], pairs[:, 1]
    beyond = np.abs(arc[i] - arc[j]) > math.pi / 2 * apart
    pairs = pairs[beyond]
    i, j = pairs[:, 0], pairs[:, 1]
    return pairs, np.hypot(*(points[i] - points[j]).T)


@dataclass(frozen=True)
class SelfCheck:
    """What a burst may not do to a coupled section's centreline, measured
    against `original`, where the burst started: with a `bend_limit`, bend
    an edge newly below it; with `apart`, bring vertices closer than that
    to its parts beyond a turn (`close_to_itself`), or than either started
    from them where it started closer, but never than the two started."""

    original: FloatArray
    bend_limit: float | None
    apart: float | None
    tight_before: BoolArray | None
    nearest: FloatArray
    """Per vertex, how close it started to its parts beyond a turn, at most
    `apart`."""

    @staticmethod
    def of(
        original: FloatArray,
        bend_limit: float | None = None,
        apart: float | None = None,
    ) -> SelfCheck:
        tight_before = (
            None if bend_limit is None else too_tight(bend_radii(original), bend_limit)
        )
        nearest = np.full(len(original), apart or 0.0)
        if apart is not None:
            pairs, distance = close_to_itself(original, apart)
            np.minimum.at(nearest, pairs[:, 0], distance)
            np.minimum.at(nearest, pairs[:, 1], distance)
        return SelfCheck(original, bend_limit, apart, tight_before, nearest)

    def __call__(self, candidate: FloatArray) -> tuple[BoolArray, int, int]:
        """The vertices to move back, how many edges newly bent too tightly
        and vertex pairs came too close, and how many of those were pairs."""
        back = np.zeros(len(candidate), dtype=bool)
        count = close = 0
        if self.bend_limit is not None and self.tight_before is not None:
            tight = too_tight(bend_radii(candidate), self.bend_limit)
            tight &= ~self.tight_before
            around = np.flatnonzero(tight)[:, None] + BEND_REACH
            back[np.clip(around, 0, len(candidate) - 1).ravel()] = True
            count += int(tight.sum())
        if self.apart is not None:
            pairs, distance = close_to_itself(candidate, self.apart)
            i, j = pairs[:, 0], pairs[:, 1]
            # At most the pair's own starting distance, so a pair moved back
            # whole is never too close, even one that lies beyond a turn only
            # along the candidate.
            started = np.hypot(*(self.original[i] - self.original[j]).T)
            floor = np.minimum(np.maximum(self.nearest[i], self.nearest[j]), started)
            closer = distance < floor - INSIDE_TOLERANCE
            back[i[closer]] = back[j[closer]] = True
            close = int(closer.sum())
            count += close
        return back, count, close
