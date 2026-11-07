"""Outer optimisation loop: resample, bubble bounds, L-BFGS-B burst, edge check.

Free vertices move only inside their bubble, a box inscribed in the free disc
around them, so they never jump over an obstacle; an exact edge check after
each burst rejects edges that newly cross an obstacle. Together they keep the
chain on the same side of every obstacle (topology preservation).
"""

from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import Protocol

import numpy as np
from scipy.optimize import minimize

from kibends.chain import Point2
from kibends.energy import (
    MIN_EDGE_LENGTH,
    BoolArray,
    DistanceField,
    EnergyParams,
    FloatArray,
    free_vertex_objective,
    total_energy,
)

# A span only slightly longer than a multiple of the spacing gets no extra
# vertex, so float noise in re-measured lengths does not change the count.
RESAMPLE_SLACK = 1e-6

# Below this turning angle a vertex counts as straight (infinite radius).
STRAIGHT_TURN = 1e-9

MEASURE_SAMPLES_PER_SPACING = 4

# Halving the bounds of rejected vertices this many times within one outer
# iteration shrinks their step to 1/16; beyond that they wait for the next one.
MAX_BOUND_HALVINGS = 4

LBFGSB_ABNORMAL = 2

# The chain's length term is ill-conditioned (tension spreads one vertex per
# iteration), so L-BFGS-B's default stopping tolerances end a burst long before
# the shape settles; with these, burst_maxiter bounds the burst instead.
LBFGSB_FTOL = 1e-15
LBFGSB_GTOL = 1e-10

# Shape change is measured from each vertex to the other polyline's edges
# within this many edges of the same index.
SHAPE_WINDOW = 3


class ObstacleField(DistanceField, Protocol):
    """Distance field that can also check segments exactly."""

    def segments_clear(self, starts: FloatArray, ends: FloatArray) -> BoolArray:
        """Return per segment whether it stays out of every inflated obstacle.

        starts and ends are (n, 2) arrays in mm. A segment is clear when it
        does not enter the region where the signed distance is negative;
        touching the boundary is clear.
        """
        ...


class Termination(Enum):
    CONVERGED = "converged"
    NOTHING_TO_DO = "nothing to do"
    ITERATION_CAP = "iteration cap reached"
    ABNORMAL = "L-BFGS-B terminated abnormally"


@dataclass(frozen=True)
class OptimizerSettings:
    spacing: float = 0.1
    """Target distance between resampled vertices, mm."""
    bubble_tolerance: float = 0.01
    """δ, mm: bubbles are measured to obstacles inflated by δ less than the
    clearance, and vertices deeper than δ inside an obstacle are frozen."""
    displacement_tolerance: float = 1e-2
    """Convergence threshold on a burst's largest shape change, as a
    fraction of the spacing."""
    max_iterations: int = 200
    stall_tolerance: float = 1e-3
    """Converged when the best energy improved by less than this over
    `stall_window` iterations. The energy is in mm of track length, so this
    is 1 µm of length: in flat stretches the shape keeps wandering by tens
    of µm while the energy creeps down by far less."""
    stall_window: int = 3
    burst_maxiter: int = 300
    """L-BFGS-B iterations per burst."""

    def __post_init__(self) -> None:
        if self.spacing <= 0.0:
            raise ValueError(f"spacing must be positive, got {self.spacing}")


DEFAULT_SETTINGS = OptimizerSettings()


@dataclass(frozen=True)
class FrozenVertex:
    """A vertex left in place because it starts inside an obstacle."""

    position: Point2
    distance: float
    """Signed distance to the clearance boundary, negative, mm."""


@dataclass(frozen=True)
class PolylineStats:
    length: float
    segment_count: int
    tightest_bend_radius: float
    """Smallest discrete radius of curvature after resampling at the spacing;
    infinite for a straight polyline, mm."""
    min_clearance: float
    """Smallest signed distance to the clearance boundary along the edges, mm."""
    mean_clearance: float
    """Length-weighted mean of that distance, mm."""


@dataclass(frozen=True)
class OptimizeResult:
    polyline: FloatArray
    fixed: BoolArray
    """Pinned and frozen vertices of `polyline`; simplification keeps them."""
    termination: Termination
    iterations: int
    rejected_edges: int
    """Edges whose burst moves were undone because they crossed an obstacle."""
    frozen: tuple[FrozenVertex, ...]
    before: PolylineStats
    after: PolylineStats


@dataclass(frozen=True)
class _Burst:
    points: FloatArray
    rejected_edges: int
    abnormal: bool


def optimize(
    polyline: FloatArray,
    fixed: BoolArray,
    field: ObstacleField,
    params: EnergyParams,
    settings: OptimizerSettings = DEFAULT_SETTINGS,
) -> OptimizeResult:
    """Minimise the chain's energy with its fixed vertices held in place."""
    if not (fixed[0] and fixed[-1]):
        raise ValueError("both endpoints of the chain must be fixed")
    distance, _ = field.signed_distance(polyline)
    frozen_mask = ~fixed & (distance < -settings.bubble_tolerance)
    frozen = tuple(
        FrozenVertex((float(x), float(y)), float(d))
        for (x, y), d in zip(polyline[frozen_mask], distance[frozen_mask], strict=True)
    )
    anchors = fixed | frozen_mask
    before = measure(polyline, field, settings.spacing)

    def finish(
        points: FloatArray,
        mask: BoolArray,
        termination: Termination,
        iterations: int,
        rejected_edges: int = 0,
    ) -> OptimizeResult:
        return OptimizeResult(
            polyline=points,
            fixed=mask,
            termination=termination,
            iterations=iterations,
            rejected_edges=rejected_edges,
            frozen=frozen,
            before=before,
            after=before
            if points is polyline
            else measure(points, field, settings.spacing),
        )

    points, mask = resample(polyline, anchors, settings.spacing)
    if mask.all():
        return finish(polyline, anchors, Termination.NOTHING_TO_DO, 0)

    # Resampling can lay a chord across an obstacle corner, and a burst only
    # rejects crossings its own moves create, so the result is limited to
    # polylines that cut no more corners than the input did.
    allowed_cuts = _cut_corners(points, field)
    best, best_mask = points, mask
    best_energy = total_energy(points, mask, field, params)[0]
    # The stall check follows every iteration, not just the ones the guard
    # accepts, or a briefly cut corner would end the run early.
    lowest_energies = [best_energy]
    rejected_edges = 0
    accepted = polyline
    termination = Termination.ITERATION_CAP
    iteration = 0
    for iteration in range(1, settings.max_iterations + 1):
        burst = _burst(points, mask, field, params, settings)
        rejected_edges += burst.rejected_edges
        shape_change = _shape_change(accepted, burst.points)
        accepted = burst.points
        energy = total_energy(burst.points, mask, field, params)[0]
        if energy <= best_energy and _cut_corners(burst.points, field) <= allowed_cuts:
            best, best_mask, best_energy = burst.points, mask, energy
        lowest_energies.append(min(lowest_energies[-1], energy))
        converged = burst.rejected_edges == 0 and (
            shape_change < settings.displacement_tolerance * settings.spacing
            or _stalled(lowest_energies, settings)
        )
        if burst.abnormal:
            termination = Termination.ABNORMAL
            break
        if converged and iteration == 1:
            return finish(polyline, anchors, Termination.NOTHING_TO_DO, iteration)
        if converged:
            termination = Termination.CONVERGED
            break
        points, mask = resample(burst.points, mask, settings.spacing)
    return finish(best, best_mask, termination, iteration, rejected_edges)


def _cut_corners(points: FloatArray, field: ObstacleField) -> int:
    """Edges with both ends clear that still pass through an obstacle.

    Edges with an end inside an obstacle belong to a violation the input
    already had, such as the neighbourhood of a frozen vertex.
    """
    distance, _ = field.signed_distance(points)
    ends_clear = (distance[:-1] >= 0.0) & (distance[1:] >= 0.0)
    crossing = ~field.segments_clear(points[:-1], points[1:])
    return int(np.count_nonzero(ends_clear & crossing))


def _stalled(lowest_energies: list[float], settings: OptimizerSettings) -> bool:
    if len(lowest_energies) <= settings.stall_window:
        return False
    earlier = lowest_energies[-1 - settings.stall_window]
    return earlier - lowest_energies[-1] <= settings.stall_tolerance


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
        arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(span, axis=0), axis=1))]
        )
        count = max(1, int(np.ceil(arc[-1] / spacing - RESAMPLE_SLACK)))
        targets = np.linspace(0.0, arc[-1], count + 1)[1:-1]
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
    distances = []
    weights = []
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


def _burst(
    points: FloatArray,
    fixed: BoolArray,
    field: ObstacleField,
    params: EnergyParams,
    settings: OptimizerSettings,
) -> _Burst:
    """One bounded L-BFGS-B run, retried with halved bounds where edges crossed."""
    free = ~fixed
    distance, _ = field.signed_distance(points)
    bubble = np.maximum(distance[free] + settings.bubble_tolerance, 0.0) / 2
    half_side = bubble / np.sqrt(2)
    clear_before = field.segments_clear(points[:-1], points[1:])
    start = points[free]
    scale = np.ones(free.sum())
    current = points.copy()
    rejected_edges = 0
    abnormal = False
    for _ in range(MAX_BOUND_HALVINGS + 1):
        reach = (half_side * scale)[:, None]
        bounds = list(
            zip(
                (start - reach).ravel(),
                (start + reach).ravel(),
                strict=True,
            )
        )
        result = minimize(
            free_vertex_objective(current, fixed, field, params),
            current[free].ravel(),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": settings.burst_maxiter,
                "ftol": LBFGSB_FTOL,
                "gtol": LBFGSB_GTOL,
            },
        )
        abnormal = result.status == LBFGSB_ABNORMAL
        candidate = current.copy()
        candidate[free] = result.x.reshape(-1, 2)
        reverted, crossings = _revert_new_crossings(
            candidate, points, free, clear_before, field
        )
        current = candidate
        if crossings == 0 or abnormal:
            break
        rejected_edges += crossings
        scale[reverted[free]] /= 2
    return _Burst(current, rejected_edges, abnormal)


def _shape_change(before: FloatArray, after: FloatArray) -> float:
    """Local symmetric distance between two polylines, mm.

    Vertices sliding along an unchanged shape do not count: resampling to
    uniform spacing and the bending term's spreading of vertices at bends
    would otherwise keep moving vertices forever without changing the track.
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
        closest = start + np.clip(t, 0.0, 1.0)[:, None] * delta
        nearest = np.minimum(nearest, np.linalg.norm(points - closest, axis=1))
    return float(nearest.max())


def _arc_fractions(points: FloatArray) -> FloatArray:
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    return np.asarray(arc / max(arc[-1], MIN_EDGE_LENGTH))


def _revert_new_crossings(
    candidate: FloatArray,
    original: FloatArray,
    free: BoolArray,
    clear_before: BoolArray,
    field: ObstacleField,
) -> tuple[BoolArray, int]:
    """Move the free ends of newly crossing edges back, in place, until none cross.

    Returns the mask of reverted vertices and the number of edges that crossed
    in the candidate as given. Reverting can make a neighbouring
    edge cross, hence the loop; it ends because fully reverted edges are the
    original, clear ones.
    """
    reverted = np.zeros(len(candidate), dtype=bool)
    initial_crossings: int | None = None
    while True:
        crossing = clear_before & ~field.segments_clear(candidate[:-1], candidate[1:])
        if initial_crossings is None:
            initial_crossings = int(crossing.sum())
        if not crossing.any():
            return reverted, initial_crossings
        ends = np.zeros(len(candidate), dtype=bool)
        ends[:-1] |= crossing
        ends[1:] |= crossing
        ends &= free & ~reverted
        candidate[ends] = original[ends]
        reverted |= ends
