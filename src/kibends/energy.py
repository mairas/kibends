"""Energy terms of a track polyline, each returning its value and gradient.

Polylines are (n, 2) float arrays in mm. Every term returns the gradient with
respect to all vertex coordinates as an (n, 2) array, with the rows of fixed
vertices zeroed.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
TermValue = tuple[float, FloatArray]
Integrand = Callable[[FloatArray], tuple[FloatArray, FloatArray]]

# 1 nm, KiCad's coordinate resolution: shorter edges are treated as degenerate.
MIN_EDGE_LENGTH = 1e-6

# Beyond this turning angle the DER curvature 2·tan(φ/2) is continued linearly
# in φ, keeping hairpins and full reversals finite, costly and differentiable.
MAX_SMOOTH_TURN = 0.95 * np.pi


class DistanceField(Protocol):
    """Signed distance to the nearest inflated obstacle boundary."""

    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """Return the distance (n,) and its unit gradient (n, 2) at (n, 2) points.

        The distance is positive in free space and negative inside an inflated
        obstacle; the gradient points toward increasing distance. Distances
        beyond `reach` may be returned as +inf with a zero gradient.
        """
        ...


class ObstacleField(DistanceField, Protocol):
    """Distance field that can also check segments exactly."""

    def segments_clear(
        self, starts: FloatArray, ends: FloatArray, tolerance: float = 0.0
    ) -> BoolArray:
        """Return per segment whether it stays out of every inflated obstacle.

        starts and ends are (n, 2) arrays in mm. A segment is clear when it
        does not enter the region where the signed distance is below
        -tolerance; touching the boundary is clear.
        """
        ...


@dataclass(frozen=True)
class EnergyParams:
    """Weights and shape parameters of the total energy; length has weight 1."""

    stiffness: float
    """Bending is weighted by stiffness², mm."""
    repulsion: float
    falloff_range: float
    """Distance beyond the clearance boundary at which repulsion reaches 0, mm."""
    clearance_weight: float
    standoff: float
    """Distance from the clearance boundary at which the hinge starts, mm."""
    points_per_edge: int
    """Interior quadrature points per edge, in addition to its endpoints."""


def arc_lengths(points: FloatArray) -> FloatArray:
    """Cumulative arc length at each vertex, starting at 0."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate((np.zeros(1), np.cumsum(steps)))


def length_energy(points: FloatArray, fixed: BoolArray) -> TermValue:
    value, gradient = _length(points, fixed, 0.0)
    return value, _zero_fixed(gradient, fixed)


def bending_energy(points: FloatArray, fixed: BoolArray) -> TermValue:
    """Discrete elastic rod bending Σ (2·tan(φᵢ/2))² / (lᵢ/2), unweighted."""
    value, gradient = _bending(points)
    return value, _zero_fixed(gradient, fixed)


def repulsion_energy(
    points: FloatArray,
    fixed: BoolArray,
    field: DistanceField,
    falloff_range: float,
    points_per_edge: int,
) -> TermValue:
    """∫ g(d(s)) ds with g = (1 - d/range)³ in [0, range), 0 beyond."""
    value, gradient = _edge_integral(
        points, field, _falloff(falloff_range), points_per_edge, falloff_range
    )
    return value, _zero_fixed(gradient, fixed)


def clearance_energy(
    points: FloatArray,
    fixed: BoolArray,
    field: DistanceField,
    standoff: float,
    points_per_edge: int,
) -> TermValue:
    """∫ c(d(s)) ds with the CHOMP hinge c of the given standoff."""
    value, gradient = _edge_integral(
        points, field, _hinge(standoff), points_per_edge, standoff
    )
    return value, _zero_fixed(gradient, fixed)


def total_energy(
    points: FloatArray,
    fixed: BoolArray,
    field: DistanceField,
    params: EnergyParams,
    short_edge: float = 0.0,
) -> TermValue:
    """Weighted sum of all terms; raises ValueError if it is not finite.

    A positive `short_edge`, mm, adds a barrier Σ (s - l)²/l over the edges
    shorter than s = `short_edge` that have a free end. The discrete bending
    energy drops as two neighbouring vertices close on each other, and across
    a vanishing edge its gradient is meaningless; the barrier grows without
    bound as an edge shrinks, so free vertices never pile up. It is 0 with
    zero slope at s, so it leaves longer edges alone.
    """
    length_value, length_gradient = _length(points, fixed, short_edge)
    bending_value, bending_gradient = _bending(points)
    obstacle_value, obstacle_gradient = obstacle_energy(points, field, params)
    bending_weight = params.stiffness**2
    value = length_value + bending_weight * bending_value + obstacle_value
    gradient = length_gradient + bending_weight * bending_gradient + obstacle_gradient
    if not (np.isfinite(value) and np.all(np.isfinite(gradient))):
        raise ValueError(f"energy is not finite: {value}")
    return value, _zero_fixed(gradient, fixed)


def point_energy(
    point: FloatArray, field: DistanceField, params: EnergyParams, length: float
) -> TermValue:
    """The obstacle terms of `total_energy` for copper at `point` that stands
    for `length` mm of track, and their gradient (2,) at `point`."""
    reach = obstacle_reach(params)
    distance, direction = field.signed_distance(point[None], reach=reach)
    value, slope = obstacle_integrand(params)(distance)
    return length * float(value[0]), length * slope[0] * direction[0]


def obstacle_energy(
    points: FloatArray, field: DistanceField, params: EnergyParams
) -> TermValue:
    """The obstacle terms of `total_energy` alone, with the gradient of every
    vertex, fixed ones included."""
    integrand = obstacle_integrand(params)
    reach = obstacle_reach(params)
    return _edge_integral(points, field, integrand, params.points_per_edge, reach)


def free_vertex_objective(
    points: FloatArray,
    fixed: BoolArray,
    field: DistanceField,
    params: EnergyParams,
    short_edge: float = 0.0,
) -> Callable[[FloatArray], tuple[float, FloatArray]]:
    """Total energy as a function of the flattened free vertex coordinates.

    The result suits scipy.optimize.minimize(..., jac=True); fixed vertices
    keep their positions from `points`.
    """
    work = points.copy()
    free = ~fixed

    def objective(free_coordinates: FloatArray) -> tuple[float, FloatArray]:
        work[free] = free_coordinates.reshape(-1, 2)
        value, gradient = total_energy(work, fixed, field, params, short_edge)
        return value, gradient[free].ravel()

    return objective


def obstacle_integrand(params: EnergyParams) -> Integrand:
    """Repulsion plus clearance, weighted, and its slope, at given distances."""
    falloff = _falloff(params.falloff_range)
    hinge = _hinge(params.standoff)

    def obstacle(distance: FloatArray) -> tuple[FloatArray, FloatArray]:
        falloff_value, falloff_slope = falloff(distance)
        hinge_value, hinge_slope = hinge(distance)
        return (
            params.repulsion * falloff_value + params.clearance_weight * hinge_value,
            params.repulsion * falloff_slope + params.clearance_weight * hinge_slope,
        )

    return obstacle


def obstacle_reach(params: EnergyParams) -> float:
    """Distance from an obstacle beyond which both obstacle terms of the
    energy are zero, mm, so farther copper need not be looked up."""
    return max(params.standoff, params.falloff_range if params.repulsion else 0.0)


def _zero_fixed(gradient: FloatArray, fixed: BoolArray) -> FloatArray:
    gradient[fixed] = 0.0
    return gradient


def _edges(points: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Edge vectors, their lengths, and unit directions (zero for degenerate edges)."""
    edges = np.diff(points, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    units = edges / np.maximum(lengths, MIN_EDGE_LENGTH)[:, None]
    return edges, lengths, units


def _scatter_edge_gradient(
    n: int, start_gradient: FloatArray, end_gradient: FloatArray
) -> FloatArray:
    gradient = np.zeros((n, 2))
    gradient[:-1] += start_gradient
    gradient[1:] += end_gradient
    return gradient


def _length(points: FloatArray, fixed: BoolArray, short_edge: float) -> TermValue:
    """Length plus the short-edge barrier of `total_energy`."""
    _, lengths, units = _edges(points)
    value = float(lengths.sum())
    short = lengths < short_edge
    if short.any():
        short &= ~(fixed[:-1] & fixed[1:])
        clamped = np.maximum(lengths[short], MIN_EDGE_LENGTH)
        value += float(np.sum((short_edge - clamped) ** 2 / clamped))
        slope = np.ones(len(lengths))
        slope[short] += 1 - (short_edge / clamped) ** 2
        units = units * slope[:, None]
    return value, _scatter_edge_gradient(len(points), -units, units)


def _perpendicular(vectors: FloatArray) -> FloatArray:
    return np.stack([-vectors[:, 1], vectors[:, 0]], axis=1)


def _curvature(turn: FloatArray) -> tuple[FloatArray, FloatArray]:
    """2·tan(φ/2) and its derivative, continued linearly beyond MAX_SMOOTH_TURN."""
    clipped = np.clip(turn, -MAX_SMOOTH_TURN, MAX_SMOOTH_TURN)
    tangent = np.tan(clipped / 2)
    slope = 1 + tangent**2
    return 2 * tangent + slope * (turn - clipped), slope


def _bending(points: FloatArray) -> TermValue:
    edges, lengths, units = _edges(points)
    incoming, outgoing = edges[:-1], edges[1:]
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    turn = np.arctan2(cross, np.einsum("ij,ij->i", incoming, outgoing))
    curvature, curvature_slope = _curvature(turn)

    raw_span = lengths[:-1] + lengths[1:]
    span = np.maximum(raw_span, MIN_EDGE_LENGTH)
    span_varies = (raw_span > MIN_EDGE_LENGTH)[:, None]

    value = float(np.sum(2 * curvature**2 / span))
    d_turn = (4 * curvature * curvature_slope / span)[:, None]
    d_span = (-2 * curvature**2 / span**2)[:, None] * span_varies

    min_square = MIN_EDGE_LENGTH**2
    incoming_square = np.maximum(lengths[:-1] ** 2, min_square)[:, None]
    outgoing_square = np.maximum(lengths[1:] ** 2, min_square)[:, None]
    d_incoming = (
        -d_turn * _perpendicular(incoming) / incoming_square + d_span * units[:-1]
    )
    d_outgoing = (
        d_turn * _perpendicular(outgoing) / outgoing_square + d_span * units[1:]
    )

    gradient = np.zeros_like(points)
    gradient[:-2] -= d_incoming
    gradient[1:-1] += d_incoming - d_outgoing
    gradient[2:] += d_outgoing
    return value, gradient


def _falloff(falloff_range: float) -> Integrand:
    """(1 - d/range)³ on [0, range), 0 beyond, linear continuation below 0.

    Zero value and slope at range, so farther copper has no effect. Inside an
    obstacle the clearance hinge dominates; the linear continuation keeps the
    repulsion bounded there instead of growing cubically.
    """

    def falloff(distance: FloatArray) -> tuple[FloatArray, FloatArray]:
        remaining = 1 - np.clip(distance, 0.0, falloff_range) / falloff_range
        boundary_slope = -3 / falloff_range
        inside = np.minimum(distance, 0.0)
        value = remaining**3 + boundary_slope * inside
        slope = np.where(distance < 0.0, boundary_slope, boundary_slope * remaining**2)
        return value, slope

    return falloff


def _hinge(standoff: float) -> Integrand:
    """CHOMP clearance hinge with standoff ε.

    0 beyond ε, (ε-d)²/(2ε) on [0, ε), ε/2 - d inside.
    """

    def hinge(distance: FloatArray) -> tuple[FloatArray, FloatArray]:
        depth = standoff - np.minimum(distance, standoff)
        quadratic = depth**2 / (2 * standoff)
        value = np.where(distance < 0.0, standoff / 2 - distance, quadratic)
        slope = np.where(distance < 0.0, -1.0, -depth / standoff)
        return value, slope

    return hinge


def _edge_integral(
    points: FloatArray,
    field: DistanceField,
    integrand: Integrand,
    points_per_edge: int,
    reach: float,
) -> TermValue:
    """∫ f(d(s)) ds along the polyline by the trapezoid rule on each edge.

    Each quadrature point's gradient goes to the edge's endpoints by its linear
    interpolation weights; the edge-length factor contributes f's mean times
    the edge direction.
    """
    n = len(points)
    interior_t = np.arange(1, points_per_edge + 1) / (points_per_edge + 1)
    interior = (
        points[:-1, None, :] * (1 - interior_t)[None, :, None]
        + points[1:, None, :] * interior_t[None, :, None]
    )
    distance, distance_gradient = field.signed_distance(
        np.concatenate([points, interior.reshape(-1, 2)]), reach=reach
    )
    vertex_distance, interior_distance = distance[:n], distance[n:].reshape(n - 1, -1)
    vertex_gradient = distance_gradient[:n]
    interior_gradient = distance_gradient[n:].reshape(n - 1, -1, 2)

    edge_distance = np.concatenate(
        [vertex_distance[:-1, None], interior_distance, vertex_distance[1:, None]],
        axis=1,
    )
    edge_distance_gradient = np.concatenate(
        [vertex_gradient[:-1, None], interior_gradient, vertex_gradient[1:, None]],
        axis=1,
    )
    t = np.concatenate([[0.0], interior_t, [1.0]])
    weights = np.full(points_per_edge + 2, 1 / (points_per_edge + 1))
    weights[[0, -1]] /= 2

    value_at, slope_at = integrand(edge_distance)
    _, lengths, units = _edges(points)
    edge_mean = value_at @ weights

    point_gradient = (weights * slope_at)[:, :, None] * edge_distance_gradient
    start_gradient = lengths[:, None] * np.einsum("k,ekj->ej", 1 - t, point_gradient)
    end_gradient = lengths[:, None] * np.einsum("k,ekj->ej", t, point_gradient)
    measure_gradient = edge_mean[:, None] * units
    gradient = _scatter_edge_gradient(
        n, start_gradient - measure_gradient, end_gradient + measure_gradient
    )
    return float(lengths @ edge_mean), gradient
