"""Energy terms of a track polyline, each returning its value and gradient.

Polylines are (n, 2) float arrays in mm. Every term returns the gradient with
respect to all vertex coordinates as an (n, 2) array, with the rows of fixed
vertices zeroed.

The distance the obstacle terms take is a smooth minimum over the
obstacles near a point, -τ·log Σ exp(-d/τ) with τ = SMOOTH_WIDTH
(`smooth_minimum`), and not the nearest one's alone: where a track lies
equally near two obstacles the nearest-only distance has a kink, its
gradient that of one side, and a track centred between its neighbours
could not move. An obstacle is one chain or one track of a section, one
via, the board edge, or static copper as `obstacles.Obstacles` groups
it. Within one obstacle the distance stays its nearest edge's, so a
straight neighbour made of many edges adds nothing, but where two parts
of one obstacle are equally near, the kink of the nearest-only distance
remains: inside the corner of a bent track, or between two arms of one,
as in a U or a meander, or of copper that one obstacle joins elsewhere.
Where n obstacles are equally near, the energy sees the point τ·ln n
nearer than it is. Every clearance judgement takes the exact distance
(`DistanceField.signed_distance`).
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
Curvature = Callable[[FloatArray], FloatArray]
"""An integrand's second derivative by the distance."""
PairIntegrand = Callable[[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]

DEFAULT_ATTRACTION_RANGE = 3.0
"""Centre distance at which a ground via and a regular one stop attracting
each other, mm. Beyond the 1.8 mm over which two 0.6 mm ground vias of one
net with 0.2 mm clearance push each other at the default range, so that
one pushed out by its neighbours is still drawn back. At an attraction of
0.25, a pair keeping up to 1.5 mm between centres still balances beyond its
clearance, 2.5 mm and 4 mm move that balance by 0.02 mm, and stitching
grids took about as many iterations to settle at ranges from 2 mm to 3 mm."""

# 1 nm, KiCad's coordinate resolution: shorter edges are treated as degenerate.
MIN_EDGE_LENGTH = 1e-6

# Beyond this turning angle the curvature 2·tan(φ/2) is continued linearly
# in φ, keeping hairpins and full reversals finite, costly and differentiable.
MAX_SMOOTH_TURN = 0.95 * np.pi

SMOOTH_WIDTH = 0.005
"""τ of the smooth minimum over obstacles (`smooth_minimum`), mm: two
equally near obstacles look τ·ln 2, 3.5 µm, nearer than either."""
SMOOTH_REACH = 10 * SMOOTH_WIDTH
"""How much farther than the nearest an obstacle still counts in the smooth
minimum, and how far beyond the energy's reach obstacles are looked for,
mm: one farther would lower it by less than τ·e⁻¹⁰, 0.2 nm. Looking
beyond the reach keeps a second obstacle's arrival at the reach from
lowering the distance by up to τ·ln 2 at once."""


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

    def smooth_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """`signed_distance` as the energy takes it: the smooth minimum over
        the field's obstacles (`smooth_minimum`) of each one's exact
        distance, and its gradient, the obstacles' unit gradients weighted
        by their shares. Obstacles farther than `reach` plus SMOOTH_REACH
        may be left out, and a point with none gets +inf."""
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

    repulsion: float
    falloff_range: float
    """Distance beyond the clearance boundary at which repulsion reaches 0, mm."""
    clearance_weight: float
    standoff: float
    """Distance from the clearance boundary at which the hinge starts, mm."""
    points_per_edge: int
    """Interior quadrature points per edge, in addition to its endpoints."""
    attraction: float = 0.0
    """Weight of a ground via and a regular one pulling together; 0 turns
    it off (`attraction_integrand`)."""
    attraction_range: float = DEFAULT_ATTRACTION_RANGE
    """Centre distance at which that pull reaches 0, mm."""


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
    hence the clearance's weight and standoff.
    """
    value, gradient = _bend_radius(points, _bends(points), limit, params)
    return value, _zero_fixed(gradient, fixed)


def shape_energy(points: FloatArray, fixed: BoolArray, short_edge: float) -> TermValue:
    """The terms of a member's energy that depend on the polyline alone:
    its length and, with a positive `short_edge`, mm, a barrier
    Σ (s - l)²/l over the edges shorter than s = `short_edge` that have a
    free end.

    Nothing else keeps neighbouring vertices apart: sliding a vertex along
    the track barely changes length or repulsion, while an edge's
    direction, and with it the gradient of its length and of the turns the
    bend-radius hinge measures, is undefined as it vanishes. The barrier
    grows without bound as an edge shrinks, so free vertices never pile up.
    It is 0 with zero slope at s, so it leaves longer edges alone."""
    value, gradient = _length(points, fixed, short_edge)
    return value, _zero_fixed(gradient, fixed)


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


def obstacle_curvature(params: EnergyParams) -> Curvature:
    """The second derivative of `obstacle_integrand` by the distance, and
    inside an obstacle that at its boundary. Inside, the integrand is
    linear, but a step back out meets the boundary's curvature, and a
    track pressed at the clearance lies a round-off inside it as often as
    outside."""
    falloff = _falloff_curvature(params.falloff_range)
    hinge = _hinge_curvature(params.standoff)

    def curvature(distance: FloatArray) -> FloatArray:
        outside = np.maximum(distance, 0.0)
        return params.repulsion * falloff(outside) + params.clearance_weight * hinge(
            outside
        )

    return curvature


def hinge_curvature(params: EnergyParams) -> Curvature:
    """The second derivative of `hinge_integrand` by the distance, inside
    an obstacle taken at its boundary as in `obstacle_curvature`."""
    hinge = _hinge_curvature(params.standoff)

    def curvature(distance: FloatArray) -> FloatArray:
        return params.clearance_weight * hinge(np.maximum(distance, 0.0))

    return curvature


def hinge_integrand(params: EnergyParams) -> Integrand:
    """The clearance hinge of `obstacle_integrand` alone, weighted: for what
    only limits where a thing may go and does not repel it."""
    hinge = _hinge(params.standoff)

    def limit(distance: FloatArray) -> tuple[FloatArray, FloatArray]:
        value, slope = hinge(distance)
        return params.clearance_weight * value, params.clearance_weight * slope

    return limit


def repulsion_integrand(params: EnergyParams) -> Integrand:
    """The repulsion of `obstacle_integrand` alone, weighted: for what pushes
    away without being a wall."""
    falloff = _falloff(params.falloff_range)

    def repulsion(distance: FloatArray) -> tuple[FloatArray, FloatArray]:
        value, slope = falloff(distance)
        return params.repulsion * value, params.repulsion * slope

    return repulsion


def attraction_integrand(params: EnergyParams) -> PairIntegrand:
    """The pull of two vias on each other, and its slope, at centre distances
    r with limit distances r_min: the shifted-force
    -w·(1/r - 1/R) - w·(r - R)/R² of weight w = `attraction` and range R =
    `attraction_range`, which has zero value and slope at R and is 0
    beyond, continued linearly below r_min, so that it stays finite and
    leaves the clearance hinge the wall there."""
    weight, cutoff = params.attraction, params.attraction_range

    def attraction(
        distance: FloatArray, limit: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        if not weight:
            return np.zeros_like(distance), np.zeros_like(distance)
        at = np.minimum(np.maximum(distance, limit), cutoff)
        slope = weight * (1 / at**2 - 1 / cutoff**2)
        value = -weight * (1 / at - 1 / cutoff) - weight * (at - cutoff) / cutoff**2
        return value + slope * (distance - at), slope

    return attraction


def smooth_minimum(
    point: IntArray, distance: FloatArray, count: int
) -> tuple[FloatArray, FloatArray]:
    """-τ·log Σ exp(-d/τ), τ = SMOOTH_WIDTH, at each of `count` points over
    the obstacles' distances d, obstacle k's distance[k] at point[k], +inf
    at a point with none; and each obstacle's share of the gradient there,
    exp(-d/τ) over the sum, the shares of a point summing to 1. Obstacles
    farther than the nearest by more than SMOOTH_REACH, and those at +inf,
    have no share, so that a search can leave them out and give the same.
    A point with one obstacle gets its distance and a share of 1."""
    if np.bincount(point, minlength=count).max(initial=0) <= 1:
        # The sum has one term at most: log 1 is 0, without the work.
        smooth = np.full(count, np.inf)
        smooth[point] = distance
        return smooth, np.isfinite(distance).astype(float)
    lowest = np.full(count, np.inf)
    np.minimum.at(lowest, point, distance)
    base = _finite(lowest)
    share = _share(distance - base[point])
    total = np.bincount(point, share, minlength=count)
    found = total > 0.0
    return _smoothed(base, total, found), share / np.where(found, total, 1.0)[point]


def _finite(lowest: FloatArray) -> FloatArray:
    """`lowest`, 0 where it is +inf: what each obstacle's distance is taken
    from, so that one at +inf stays at +inf and none gives NaN."""
    return np.where(np.isfinite(lowest), lowest, 0.0)


def _share(above: FloatArray) -> FloatArray:
    """exp(-a/τ) of how far each obstacle lies beyond the nearest, 0 beyond
    SMOOTH_REACH."""
    return np.where(above <= SMOOTH_REACH, np.exp(-above / SMOOTH_WIDTH), 0.0)


def _smoothed(base: FloatArray, total: FloatArray, found: BoolArray) -> FloatArray:
    """The smooth minimum from the nearest distance `base` and the `total`
    of the shares, +inf where none is `found`."""
    return np.where(
        found, base - SMOOTH_WIDTH * np.log(np.where(found, total, 1.0)), np.inf
    )


def weighted_sum(
    point: IntArray, share: FloatArray, direction: FloatArray, count: int
) -> FloatArray:
    """At each of `count` points, the sum of the (m, 2) `direction`s at it,
    each weighted by its `share`; the smooth minimum's gradient."""
    return np.column_stack(
        [np.bincount(point, share * direction[:, k], minlength=count) for k in (0, 1)]
    ).reshape(count, 2)


def combine_fields(*distances: FloatArray) -> tuple[FloatArray, list[FloatArray]]:
    """`smooth_minimum` over fields, each of `distances` one field's (n,)
    smooth distances, and each field's share at each point. A smooth
    minimum of smooth minima is the one over all their obstacles, but for
    the obstacles SMOOTH_REACH leaves out."""
    answering = [k for k, d in enumerate(distances) if not np.isinf(d).all()]
    if len(answering) == 1:
        # As in `smooth_minimum`, one term at each point is its own minimum,
        # and fields often have no other in reach.
        alone = distances[answering[0]]
        shares = [np.zeros(len(alone)) for _ in distances]
        shares[answering[0]] = np.isfinite(alone).astype(float)
        return alone, shares
    base = _finite(np.minimum.reduce(distances))
    shares = [_share(d - base) for d in distances]
    total = np.add.reduce(shares)
    found = total > 0.0
    scale = np.where(found, total, 1.0)
    return _smoothed(base, total, found), [share / scale for share in shares]


def smooth_reach(reach: float) -> float:
    """How far from a point the smooth distance looks for obstacles, for
    an energy whose terms end at `reach`, mm (SMOOTH_REACH)."""
    return reach + SMOOTH_REACH


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
    """Length plus the short-edge barrier of `shape_energy`."""
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


def _turn_gradient(
    points: FloatArray, edges: FloatArray, lengths: FloatArray, d_turn: FloatArray
) -> FloatArray:
    """The gradient at every vertex of a sum over interior vertices, given its
    derivatives (n-2, 1) by each vertex's turn."""
    incoming, outgoing = edges[:-1], edges[1:]
    min_square = MIN_EDGE_LENGTH**2
    incoming_square = np.maximum(lengths[:-1] ** 2, min_square)[:, None]
    outgoing_square = np.maximum(lengths[1:] ** 2, min_square)[:, None]
    d_incoming = -d_turn * _perpendicular(incoming) / incoming_square
    d_outgoing = d_turn * _perpendicular(outgoing) / outgoing_square

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
    return value, gradient + _turn_gradient(points, edges, lengths, d_turn)


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


def _falloff_curvature(falloff_range: float) -> Curvature:
    """The second derivative of `_falloff`: 0 where it is linear or 0."""

    def curvature(distance: FloatArray) -> FloatArray:
        within = (distance >= 0.0) & (distance < falloff_range)
        remaining = 1 - np.clip(distance, 0.0, falloff_range) / falloff_range
        return np.where(within, 6 / falloff_range**2 * remaining, 0.0)

    return curvature


def _hinge_curvature(standoff: float) -> Curvature:
    """The second derivative of `_hinge`: 1/ε on its quadratic part."""

    def curvature(distance: FloatArray) -> FloatArray:
        return np.where((distance >= 0.0) & (distance < standoff), 1 / standoff, 0.0)

    return curvature


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


def quadrature_points(points: FloatArray, points_per_edge: int) -> FloatArray:
    """The points at which `edge_integral` takes distances: the vertices,
    then each edge's interior points in order."""
    interior_t = np.arange(1, points_per_edge + 1) / (points_per_edge + 1)
    interior = (
        points[:-1, None, :] * (1 - interior_t)[None, :, None]
        + points[1:, None, :] * interior_t[None, :, None]
    )
    return np.concatenate([points, interior.reshape(-1, 2)])


class _EdgeSamples(NamedTuple):
    """The quadrature points of each edge, its ends included, as
    `edge_integral` weights them."""

    distance: FloatArray
    """(edges, points_per_edge + 2) distances."""
    gradient: FloatArray
    """(edges, points_per_edge + 2, 2) their gradients."""
    t: FloatArray
    """Where along its edge each point lies, from 0 to 1."""
    weights: FloatArray
    """The trapezoid rule's weight of each point, summing to 1."""


def _edge_samples(
    n: int, distance: FloatArray, direction: FloatArray, points_per_edge: int
) -> _EdgeSamples:
    """The distances at `quadrature_points` of a polyline of `n` vertices,
    and their gradients, by edge."""
    interior_t = np.arange(1, points_per_edge + 1) / (points_per_edge + 1)
    vertex_distance, interior_distance = distance[:n], distance[n:].reshape(n - 1, -1)
    vertex_gradient = direction[:n]
    interior_gradient = direction[n:].reshape(n - 1, -1, 2)
    edge_distance = np.concatenate(
        [vertex_distance[:-1, None], interior_distance, vertex_distance[1:, None]],
        axis=1,
    )
    edge_gradient = np.concatenate(
        [vertex_gradient[:-1, None], interior_gradient, vertex_gradient[1:, None]],
        axis=1,
    )
    t = np.concatenate([[0.0], interior_t, [1.0]])
    weights = np.full(points_per_edge + 2, 1 / (points_per_edge + 1))
    weights[[0, -1]] /= 2
    return _EdgeSamples(edge_distance, edge_gradient, t, weights)


def point_stiffness(
    distance: FloatArray,
    direction: FloatArray,
    integrand: Integrand,
    curvature: Curvature,
) -> FloatArray:
    """The diagonal second derivative of `integrand` at points by x and by
    y, from their smooth distances (...) and gradients (..., 2), as the
    terms of it that cannot be negative: f''(d)·g², with g the gradient,
    and |f'(d)|·(1 - |g|²)/τ on both axes, the smooth minimum's own
    curvature (`smooth_minimum`) where obstacles in different directions
    are about as near, the spread of their unit gradients measured by the
    shortfall of |g| from 1. It leaves out each obstacle's own curvature."""
    _, slope = integrand(distance)
    squared = direction**2
    spread = np.maximum(1 - squared.sum(axis=-1), 0.0)
    smoothing = np.abs(slope) * spread / SMOOTH_WIDTH
    return curvature(distance)[..., None] * squared + smoothing[..., None]


def edge_curvature(
    points: FloatArray,
    distance: FloatArray,
    direction: FloatArray,
    integrand: Integrand,
    curvature: Curvature,
    points_per_edge: int,
) -> FloatArray:
    """Each vertex's stiffness under `edge_integral` of `integrand`: the
    larger over x and y of its diagonal second derivative, from the
    distances and their gradients at `quadrature_points`.

    Each quadrature point adds its `point_stiffness` times lₑ·wₖ and the
    square of the vertex's interpolation weight, and each edge its mean f̄
    times the curvature of its length, f̄·(1 - u²)/lₑ across its direction
    u. It leaves out the product of the length's and the distance's
    slopes, which can be negative, so that it is exact for edges parallel
    to flat obstacles."""
    n = len(points)
    samples = _edge_samples(n, distance, direction, points_per_edge)
    value_at, _ = integrand(samples.distance)
    _, lengths, units = _edges(points)
    measure = (lengths[:, None] * samples.weights)[:, :, None]
    per_point = measure * point_stiffness(
        samples.distance, samples.gradient, integrand, curvature
    )
    start = np.einsum("k,ekj->ej", (1 - samples.t) ** 2, per_point)
    end = np.einsum("k,ekj->ej", samples.t**2, per_point)
    edge_mean = value_at @ samples.weights
    across = (edge_mean / np.maximum(lengths, MIN_EDGE_LENGTH))[:, None] * (
        1 - units**2
    )
    return _scatter_edge_gradient(n, start + across, end + across).max(axis=1)


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
    edge_distance, edge_distance_gradient, t, weights = _edge_samples(
        n, distance, direction, points_per_edge
    )

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
