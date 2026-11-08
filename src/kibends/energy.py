"""Energy terms of a track polyline, each returning its value and gradient.

Polylines are (n, 2) float arrays in mm. Every term returns the gradient with
respect to all vertex coordinates as an (n, 2) array, with the rows of fixed
vertices zeroed.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.intp]
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


def edge_bend_radii(
    lengths: FloatArray, half_turns: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Each edge's bend radius, from the edges' `lengths` and tan(φ/2) of the
    turn φ at each interior vertex, and the signed sum of those tangents at
    each edge's two ends.

    The bend radius of an edge is the largest offset at which its mitred
    offset still has positive length: its length over the absolute sum,
    infinite where the sum is 0.
    """
    half_turn = np.concatenate([[0.0], half_turns, [0.0]])
    spread = half_turn[:-1] + half_turn[1:]
    size = np.abs(spread)
    radii = np.divide(
        lengths, size, out=np.full_like(lengths, math.inf), where=size > 0.0
    )
    return radii, spread


def arc_lengths(points: FloatArray) -> FloatArray:
    """Cumulative arc length at each vertex, starting at 0."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate((np.zeros(1), np.cumsum(steps)))


def length_energy(points: FloatArray, fixed: BoolArray) -> TermValue:
    value, gradient = _length(points, fixed, 0.0)
    return value, _zero_fixed(gradient, fixed)


def bending_energy(points: FloatArray, fixed: BoolArray) -> TermValue:
    """Discrete elastic rod bending Σ (2·tan(φᵢ/2))² / (lᵢ/2), unweighted."""
    value, gradient = _bending(points, _bends(points))
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


def bend_radius_energy(
    points: FloatArray, fixed: BoolArray, limit: float, params: EnergyParams
) -> TermValue:
    """Σ lₑ·c(rₑ - `limit`) over the edges, weighted like the clearance
    hinge c and with its standoff, where rₑ is edge e's bend radius: its
    length lₑ over |tan(φ/2)| summed over its two ends' turns φ, signed, 0
    at the polyline's ends.

    That is the radius `pairs.check_bend_radius` measures: below it the
    polyline's mitred offset to the inside folds back on itself, so with
    `limit` set to `pairs.bend_radius_limit` it keeps a pair's inner track
    from folding. A bend at the limit is as hard a constraint as the
    clearance, and has nothing to do with how smooth a track should be,
    hence the clearance's weight and standoff and not the stiffness.
    """
    value, gradient = _bend_radius(points, _bends(points), limit, params)
    return value, _zero_fixed(gradient, fixed)


def total_energy(
    points: FloatArray,
    fixed: BoolArray,
    field: DistanceField,
    params: EnergyParams,
    short_edge: float = 0.0,
    bend_limit: float | None = None,
) -> TermValue:
    """Weighted sum of all terms; raises ValueError if it is not finite.

    A positive `short_edge`, mm, adds a barrier Σ (s - l)²/l over the edges
    shorter than s = `short_edge` that have a free end. The discrete bending
    energy drops as two neighbouring vertices close on each other, and across
    a vanishing edge its gradient is meaningless; the barrier grows without
    bound as an edge shrinks, so free vertices never pile up. It is 0 with
    zero slope at s, so it leaves longer edges alone.

    A `bend_limit`, mm, adds `bend_radius_energy` with that limit.
    """
    bends = _bends(points)
    shape_value, shape_gradient = _shape(points, fixed, bends, params, short_edge)
    obstacle_value, obstacle_gradient = obstacle_energy(points, field, params)
    value = shape_value + obstacle_value
    gradient = shape_gradient + obstacle_gradient
    if bend_limit is not None:
        bend_value, bend_gradient = _bend_radius(points, bends, bend_limit, params)
        value += bend_value
        gradient = gradient + bend_gradient
    if not (np.isfinite(value) and np.all(np.isfinite(gradient))):
        raise ValueError(f"energy is not finite: {value}")
    return value, _zero_fixed(gradient, fixed)


def shape_energy(
    points: FloatArray, fixed: BoolArray, params: EnergyParams, short_edge: float
) -> TermValue:
    """The terms of `total_energy` that depend on the polyline alone: length,
    the short-edge barrier and the weighted bending."""
    value, gradient = _shape(points, fixed, _bends(points), params, short_edge)
    return value, _zero_fixed(gradient, fixed)


def _shape(
    points: FloatArray,
    fixed: BoolArray,
    bends: _Bends,
    params: EnergyParams,
    short_edge: float,
) -> TermValue:
    length_value, length_gradient = _length(points, fixed, short_edge)
    bending_value, bending_gradient = _bending(points, bends)
    bending_weight = params.stiffness**2
    return (
        length_value + bending_weight * bending_value,
        length_gradient + bending_weight * bending_gradient,
    )


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
    bend_limit: float | None = None,
) -> Callable[[FloatArray], tuple[float, FloatArray]]:
    """Total energy as a function of the flattened free vertex coordinates.

    The result suits scipy.optimize.minimize(..., jac=True); fixed vertices
    keep their positions from `points`.
    """
    work = points.copy()
    free = ~fixed

    def objective(free_coordinates: FloatArray) -> tuple[float, FloatArray]:
        work[free] = free_coordinates.reshape(-1, 2)
        value, gradient = total_energy(
            work, fixed, field, params, short_edge, bend_limit
        )
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


def _turns(edges: FloatArray) -> FloatArray:
    """Signed turning angle at each interior vertex."""
    incoming, outgoing = edges[:-1], edges[1:]
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    return np.arctan2(cross, np.einsum("ij,ij->i", incoming, outgoing))


class _Bends(NamedTuple):
    """A polyline's edges and the curvature at its interior vertices."""

    edges: FloatArray
    lengths: FloatArray
    units: FloatArray
    curvature: FloatArray
    """2·tan(φ/2) of each turn φ, as `_curvature` continues it."""
    curvature_slope: FloatArray


def _bends(points: FloatArray) -> _Bends:
    edges, lengths, units = _edges(points)
    return _Bends(edges, lengths, units, *_curvature(_turns(edges)))


def _bending(points: FloatArray, bends: _Bends) -> TermValue:
    edges, lengths, units, curvature, curvature_slope = bends

    raw_span = lengths[:-1] + lengths[1:]
    span = np.maximum(raw_span, MIN_EDGE_LENGTH)
    span_varies = (raw_span > MIN_EDGE_LENGTH)[:, None]

    value = float(np.sum(2 * curvature**2 / span))
    d_turn = (4 * curvature * curvature_slope / span)[:, None]
    d_span = (-2 * curvature**2 / span**2)[:, None] * span_varies
    return value, _turn_gradient(points, edges, lengths, units, d_turn, d_span)


def _turn_gradient(
    points: FloatArray,
    edges: FloatArray,
    lengths: FloatArray,
    units: FloatArray,
    d_turn: FloatArray,
    d_span: FloatArray,
) -> FloatArray:
    """The gradient at every vertex of a sum over interior vertices, given its
    derivatives (n-2, 1) by each vertex's turn and by the summed length of
    the two edges meeting there."""
    incoming, outgoing = edges[:-1], edges[1:]
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
    return gradient


def _bend_radius(
    points: FloatArray, bends: _Bends, limit: float, params: EnergyParams
) -> TermValue:
    edges, lengths, units, curvature, curvature_slope = bends
    radii, spread = edge_bend_radii(lengths, curvature / 2)
    size = np.abs(spread)
    # Wider bends, straight edges among them, are beyond the hinge.
    active = lengths < (limit + params.standoff) * size
    if not active.any():
        return 0.0, np.zeros_like(points)
    safe_size = np.where(active, size, 1.0)
    radius = np.where(active, radii, math.inf)
    hinge_value, hinge_slope = _hinge(params.standoff)(radius - limit)
    weight = params.clearance_weight
    value = weight * float(lengths @ hinge_value)
    d_length = weight * (hinge_value + lengths * hinge_slope / safe_size)
    d_size = np.where(active, -weight * (lengths / safe_size) ** 2 * hinge_slope, 0.0)
    d_half_turn = np.sign(spread) * d_size
    d_turn = ((d_half_turn[:-1] + d_half_turn[1:]) * curvature_slope / 2)[:, None]
    gradient = _scatter_edge_gradient(
        len(points), -d_length[:, None] * units, d_length[:, None] * units
    )
    no_span = np.zeros_like(d_turn)
    return value, gradient + _turn_gradient(
        points, edges, lengths, units, d_turn, no_span
    )


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
    """∫ f(d(s)) ds along the polyline by the trapezoid rule on each edge."""
    distance, direction = field.signed_distance(
        quadrature_points(points, points_per_edge), reach=reach
    )
    value, gradient, _ = edge_integral(
        points, distance, direction, integrand, points_per_edge
    )
    return value, gradient


def quadrature_points(points: FloatArray, points_per_edge: int) -> FloatArray:
    """The points at which `edge_integral` takes distances: the vertices,
    then each edge's interior points in order."""
    interior_t = np.arange(1, points_per_edge + 1) / (points_per_edge + 1)
    interior = (
        points[:-1, None, :] * (1 - interior_t)[None, :, None]
        + points[1:, None, :] * interior_t[None, :, None]
    )
    return np.concatenate([points, interior.reshape(-1, 2)])


def edge_integral(
    points: FloatArray,
    distance: FloatArray,
    direction: FloatArray,
    integrand: Integrand,
    points_per_edge: int,
) -> tuple[float, FloatArray, FloatArray]:
    """∫ f(d(s)) ds along the polyline by the trapezoid rule on each edge,
    from the distances and their unit gradients at `quadrature_points`.

    Each quadrature point's gradient goes to the edge's endpoints by its linear
    interpolation weights; the edge-length factor contributes f's mean times
    the edge direction. Returns the value, its gradient at every vertex, and
    its derivative by the distance at each quadrature point, for a caller
    whose distances also depend on other coordinates.
    """
    n = len(points)
    interior_t = np.arange(1, points_per_edge + 1) / (points_per_edge + 1)
    vertex_distance, interior_distance = distance[:n], distance[n:].reshape(n - 1, -1)
    vertex_gradient = direction[:n]
    interior_gradient = direction[n:].reshape(n - 1, -1, 2)

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

    weighted_slope = weights * slope_at
    point_gradient = weighted_slope[:, :, None] * edge_distance_gradient
    start_gradient = lengths[:, None] * np.einsum("k,ekj->ej", 1 - t, point_gradient)
    end_gradient = lengths[:, None] * np.einsum("k,ekj->ej", t, point_gradient)
    measure_gradient = edge_mean[:, None] * units
    gradient = _scatter_edge_gradient(
        n, start_gradient - measure_gradient, end_gradient + measure_gradient
    )
    by_edge = lengths[:, None] * weighted_slope
    slopes = np.zeros(len(distance))
    slopes[: n - 1] += by_edge[:, 0]
    slopes[1:n] += by_edge[:, -1]
    slopes[n:] = by_edge[:, 1:-1].ravel()
    return float(lengths @ edge_mean), gradient, slopes
