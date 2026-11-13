import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from conftest import (
    CURVATURE_ATOL,
    CURVATURE_RTOL,
    CURVATURE_STEP,
    obstacle_energy,
    total_energy,
)
from distance_fields import Box, Disc, FloatArray, NanField, Walls
from numpy.typing import NDArray

from kibends.energy import (
    SMOOTH_REACH,
    SMOOTH_WIDTH,
    DistanceField,
    EnergyParams,
    TermValue,
    attraction_integrand,
    bend_radius_energy,
    edge_curvature,
    edge_integral,
    hinge_curvature,
    hinge_integrand,
    obstacle_curvature,
    obstacle_integrand,
    quadrature_points,
    shape_energy,
    smooth_minimum,
)
from kibends.obstacles import ChainCentreline, ChainField, chains_field

Term = Callable[[FloatArray, NDArray[np.bool_]], tuple[float, FloatArray]]

FD_STEP = 1e-6
GRADIENT_RTOL = 1e-5
POINTS_PER_EDGE = 3
FALLOFF_RANGE = 1.5
STANDOFF = 0.2

DISC = Disc(0.0, 0.0, 1.0)
BOX = Box(0.0, 0.0, 1.0, 0.6)
PARAMS = EnergyParams(
    repulsion=0.7,
    falloff_range=FALLOFF_RANGE,
    clearance_weight=50.0,
    standoff=STANDOFF,
    points_per_edge=POINTS_PER_EDGE,
)


def length_energy(points: FloatArray, fixed: NDArray[np.bool_]) -> TermValue:
    return shape_energy(points, fixed, 0.0)


def weighted(
    points: FloatArray,
    fixed: NDArray[np.bool_],
    field: DistanceField,
    params: EnergyParams,
) -> TermValue:
    value, gradient = obstacle_energy(points, field, params)
    gradient[fixed] = 0.0
    return value, gradient


def repulsion_energy(
    points: FloatArray, fixed: NDArray[np.bool_], field: DistanceField
) -> TermValue:
    """The repulsion term alone, unweighted."""
    return weighted(
        points, fixed, field, replace(PARAMS, repulsion=1.0, clearance_weight=0.0)
    )


def clearance_energy(
    points: FloatArray, fixed: NDArray[np.bool_], field: DistanceField
) -> TermValue:
    """The clearance hinge alone, unweighted."""
    return weighted(
        points, fixed, field, replace(PARAMS, repulsion=0.0, clearance_weight=1.0)
    )


def free(n: int) -> NDArray[np.bool_]:
    return np.zeros(n, dtype=bool)


def numeric_gradient(
    energy: Callable[[FloatArray], float], points: FloatArray
) -> FloatArray:
    gradient = np.zeros_like(points)
    for index in np.ndindex(points.shape):
        step = np.zeros_like(points)
        step[index] = FD_STEP
        gradient[index] = (energy(points + step) - energy(points - step)) / (
            2 * FD_STEP
        )
    return gradient


def assert_gradient_matches(term: Term, points: FloatArray) -> None:
    fixed = free(len(points))
    _, analytic = term(points, fixed)
    numeric = numeric_gradient(lambda p: term(p, fixed)[0], points)
    error = np.linalg.norm(analytic - numeric) / max(np.linalg.norm(numeric), 1e-12)
    assert error < GRADIENT_RTOL, (analytic, numeric)


def wiggly_polyline(
    start: tuple[float, float], end: tuple[float, float], n: int, seed: int
) -> FloatArray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)[:, None]
    line: FloatArray = (1 - t) * np.array(start) + t * np.array(end)
    return line + rng.normal(scale=0.15, size=(n, 2))


def arc(radius: float, angle: float, n: int) -> FloatArray:
    theta = np.linspace(0.0, angle, n)
    return radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)


def horizontal_edge(y: float) -> FloatArray:
    return np.array([[-1.0, y], [1.0, y]])


def with_field(
    term: Callable[
        [FloatArray, NDArray[np.bool_], DistanceField], tuple[float, FloatArray]
    ],
    field: DistanceField,
) -> Term:
    return lambda points, fixed: term(points, fixed, field)


def repulsion(field: DistanceField) -> Term:
    return with_field(repulsion_energy, field)


def clearance(field: DistanceField) -> Term:
    return with_field(clearance_energy, field)


def total(field: DistanceField) -> Term:
    return with_field(lambda p, f, d: total_energy(p, f, d, PARAMS), field)


# Polylines passing beside the disc and box, partly within range and standoff,
# with some quadrature points inside the obstacle.
NEAR_DISC = [wiggly_polyline((-2.5, 1.1), (2.5, 0.9), 9, seed) for seed in range(3)]
NEAR_BOX = [
    wiggly_polyline((-2.5, 0.9), (2.5, 0.8), 9, seed) for seed in range(3, 6)
] + [wiggly_polyline((-2.5, 0.7), (2.5, 0.6), 9, seed=6)]


class TestLength:
    def test_value_is_sum_of_edge_lengths(self) -> None:
        points = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 10.0]])
        value, _ = length_energy(points, free(3))
        assert value == pytest.approx(11.0)

    @pytest.mark.parametrize("points", NEAR_DISC)
    def test_gradient(self, points: FloatArray) -> None:
        assert_gradient_matches(length_energy, points)


class TestRepulsion:
    @pytest.mark.parametrize("points", NEAR_DISC)
    def test_gradient_near_disc(self, points: FloatArray) -> None:
        assert_gradient_matches(repulsion(DISC), points)

    @pytest.mark.parametrize("points", NEAR_BOX)
    def test_gradient_near_box(self, points: FloatArray) -> None:
        assert_gradient_matches(repulsion(BOX), points)

    def test_zero_beyond_range(self) -> None:
        points = wiggly_polyline((-3.0, 4.0), (3.0, 4.0), 7, seed=0)
        value, gradient = repulsion(DISC)(points, free(7))
        assert value == 0.0
        np.testing.assert_array_equal(gradient, 0.0)

    def test_value_and_slope_vanish_at_range(self) -> None:
        walls = Walls(half_gap=10.0)
        just_inside = horizontal_edge(10.0 - FALLOFF_RANGE + 1e-4)
        value, gradient = repulsion(walls)(just_inside, free(2))
        assert value < 1e-10
        assert np.abs(gradient).max() < 1e-6

    def test_edge_at_constant_distance_integrates_falloff(self) -> None:
        walls = Walls(half_gap=10.0)
        value, _ = repulsion(walls)(horizontal_edge(10.0 - FALLOFF_RANGE / 2), free(2))
        assert value == pytest.approx(2.0 * (1 - 0.5) ** 3)

    def test_continues_inside_obstacle(self) -> None:
        walls = Walls(half_gap=1.0)
        at_boundary, _ = repulsion(walls)(horizontal_edge(1.0), free(2))
        inside, gradient = repulsion(walls)(horizontal_edge(1.5), free(2))
        assert inside > at_boundary
        assert np.all(gradient[:, 1] > 0)


class TestClearance:
    @pytest.mark.parametrize("points", NEAR_DISC)
    def test_gradient_near_disc(self, points: FloatArray) -> None:
        assert_gradient_matches(clearance(DISC), points)

    @pytest.mark.parametrize("points", NEAR_BOX)
    def test_gradient_near_box(self, points: FloatArray) -> None:
        assert_gradient_matches(clearance(BOX), points)

    def test_zero_beyond_standoff(self) -> None:
        walls = Walls(half_gap=1.0)
        value, gradient = clearance(walls)(
            horizontal_edge(1.0 - STANDOFF - 0.01), free(2)
        )
        assert value == 0.0
        np.testing.assert_array_equal(gradient, 0.0)

    def test_smooth_at_standoff(self) -> None:
        walls = Walls(half_gap=1.0)
        _, gradient = clearance(walls)(horizontal_edge(1.0 - STANDOFF + 1e-6), free(2))
        assert np.abs(gradient).max() < 1e-4

    def test_quadratic_within_standoff(self) -> None:
        walls = Walls(half_gap=1.0)
        depth = STANDOFF / 2
        value, _ = clearance(walls)(horizontal_edge(1.0 - STANDOFF + depth), free(2))
        assert value == pytest.approx(2.0 * depth**2 / (2 * STANDOFF))

    def test_linear_inside_obstacle(self) -> None:
        walls = Walls(half_gap=1.0)
        values = [
            clearance(walls)(horizontal_edge(1.0 + d), free(2))[0]
            for d in (0.5, 1.0, 1.5)
        ]
        assert values[1] - values[0] == pytest.approx(2.0 * 0.5)
        assert values[2] - values[1] == pytest.approx(2.0 * 0.5)

    def test_edge_crossing_obstacle_between_clear_endpoints(self) -> None:
        edge = np.array([[-2.0, 0.5], [2.0, 0.5]])
        assert DISC.signed_distance(edge)[0].min() > STANDOFF
        value, _ = clearance(DISC)(edge, free(2))
        assert value > 0.0


class TestTotal:
    @pytest.mark.parametrize("points", NEAR_DISC + NEAR_BOX)
    def test_gradient(self, points: FloatArray) -> None:
        assert_gradient_matches(total(DISC), points)

    def test_is_weighted_sum_of_terms(self) -> None:
        points = NEAR_DISC[0]
        fixed = free(len(points))
        terms = [
            (1.0, length_energy(points, fixed)),
            (
                PARAMS.repulsion,
                repulsion_energy(points, fixed, DISC),
            ),
            (
                PARAMS.clearance_weight,
                clearance_energy(points, fixed, DISC),
            ),
        ]
        value, gradient = total_energy(points, fixed, DISC, PARAMS)
        assert value == pytest.approx(sum(w * v for w, (v, _) in terms))
        np.testing.assert_allclose(gradient, sum(w * g for w, (_, g) in terms))

    def test_non_finite_energy_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            total_energy(NEAR_DISC[0], free(len(NEAR_DISC[0])), NanField(), PARAMS)


BEND_LIMIT = 1.8
"""Tight enough that the wiggly polylines' edges reach every part of the
hinge: clear, within the standoff, and inside the limit."""
BEND_SPACING = 0.1
"""The spacing each edge's hinge weighs by, mm."""


def bend_radius(limit: float = BEND_LIMIT) -> Term:
    return lambda points, fixed: bend_radius_energy(
        points, fixed, limit, BEND_SPACING, PARAMS
    )


def edge_radii(points: FloatArray) -> FloatArray:
    """Each edge's length over its ends' summed half-turn tangents."""
    edges = np.diff(points, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    cross = edges[:-1, 0] * edges[1:, 1] - edges[:-1, 1] * edges[1:, 0]
    turn = np.arctan2(cross, np.einsum("ij,ij->i", edges[:-1], edges[1:]))
    half = np.concatenate([[0.0], np.tan(turn / 2), [0.0]])
    return lengths / np.abs(half[:-1] + half[1:])


class TestBendRadius:
    def test_zero_while_every_bend_is_wider_than_the_limit(self) -> None:
        points = arc(1.0, np.pi / 2, 33)
        value, gradient = bend_radius_energy(
            points, free(33), 0.5, BEND_SPACING, PARAMS
        )
        assert value == 0.0
        np.testing.assert_array_equal(gradient, 0.0)

    def test_straight_edges_cost_nothing_however_short(self) -> None:
        points = np.stack([np.linspace(0.0, 0.01, 11), np.zeros(11)], axis=1)
        value, _ = bend_radius_energy(
            points, free(11), BEND_LIMIT, BEND_SPACING, PARAMS
        )
        assert value == 0.0

    def test_is_the_clearance_hinge_of_the_radii_by_the_edge(self) -> None:
        points = NEAR_DISC[0]
        radius = edge_radii(points)
        depth = radius - BEND_LIMIT
        assert depth.min() < 0.0 < STANDOFF < depth.max()
        hinge = np.where(
            depth < 0.0,
            STANDOFF / 2 - depth,
            (STANDOFF - np.minimum(depth, STANDOFF)) ** 2 / (2 * STANDOFF),
        )
        value, _ = bend_radius_energy(
            points, free(len(points)), BEND_LIMIT, BEND_SPACING, PARAMS
        )
        assert value == pytest.approx(
            PARAMS.clearance_weight * BEND_SPACING * hinge.sum()
        )

    def test_tighter_bend_costs_more(self) -> None:
        values = [
            bend_radius_energy(
                arc(radius, np.pi / 2, 33), free(33), 1.0, BEND_SPACING, PARAMS
            )[0]
            for radius in (1.0, 0.8, 0.5)
        ]
        assert 0.0 < values[0] < values[1] < values[2]

    @pytest.mark.parametrize("points", NEAR_DISC + NEAR_BOX)
    def test_gradient(self, points: FloatArray) -> None:
        assert_gradient_matches(bend_radius(), points)

    def test_gradient_beyond_smooth_turn_limit(self) -> None:
        hairpin = np.array([[0.0, 0.0], [2.0, 0.0], [0.1, 0.05], [0.0, 1.0]])
        assert_gradient_matches(bend_radius(), hairpin)

    def test_total_energy_adds_it_with_a_limit(self) -> None:
        points = NEAR_DISC[0]
        fixed = free(len(points))
        plain = total_energy(points, fixed, DISC, PARAMS)
        bend = bend_radius_energy(points, fixed, BEND_LIMIT, BEND_SPACING, PARAMS)
        value, gradient = total_energy(
            points, fixed, DISC, PARAMS, hinge=(BEND_LIMIT, BEND_SPACING)
        )
        assert value == pytest.approx(plain[0] + bend[0])
        np.testing.assert_allclose(gradient, plain[1] + bend[1])


ALL_TERMS: dict[str, Term] = {
    "length": length_energy,
    "repulsion": repulsion(DISC),
    "clearance": clearance(DISC),
    "total": total(DISC),
    "bend radius": bend_radius(),
}


@pytest.mark.parametrize("name", ALL_TERMS)
def test_coincident_consecutive_vertices_are_finite(name: str) -> None:
    points = np.array([[-2.0, 1.1], [0.0, 1.1], [0.0, 1.1], [2.0, 1.2], [2.0, 1.2]])
    value, gradient = ALL_TERMS[name](points, free(5))
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize("name", ALL_TERMS)
def test_fixed_vertices_get_zero_gradient(name: str) -> None:
    points = NEAR_DISC[1]
    fixed = free(len(points))
    fixed[[0, 4, -1]] = True
    _, gradient = ALL_TERMS[name](points, fixed)
    np.testing.assert_array_equal(gradient[fixed], 0.0)
    assert np.abs(gradient[~fixed]).max() > 0.0


SHORT_EDGE = 0.1


def short_edge_barrier(points: FloatArray, fixed: NDArray[np.bool_]) -> TermValue:
    """The barrier's share of the total energy."""
    with_value, with_gradient = total_energy(points, fixed, DISC, PARAMS, SHORT_EDGE)
    without_value, without_gradient = total_energy(points, fixed, DISC, PARAMS)
    return with_value - without_value, with_gradient - without_gradient


def pinched(gap: float) -> FloatArray:
    """An arc with its two middle vertices `gap` apart along the chord between them."""
    points = arc(2.0, 1.0, 9)
    middle = (points[4] + points[5]) / 2
    chord: FloatArray = points[5] - points[4]
    along = chord / float(np.linalg.norm(chord))
    points[4], points[5] = middle - gap / 2 * along, middle + gap / 2 * along
    return points


class TestShortEdgeBarrier:
    def test_gradient_matches_finite_differences(self) -> None:
        assert_gradient_matches(short_edge_barrier, pinched(0.03))

    def test_is_zero_when_no_edge_is_short(self) -> None:
        points = arc(2.0, 1.0, 9)
        value, gradient = short_edge_barrier(points, free(len(points)))
        assert value == 0.0
        np.testing.assert_array_equal(gradient, 0.0)

    def test_grows_without_bound_as_two_vertices_close(self) -> None:
        gaps = [0.08, 0.03, 1e-3, 1e-5]
        values = [short_edge_barrier(pinched(g), free(9))[0] for g in gaps]
        assert values == sorted(values)
        assert values[-1] > 1e2

    def test_pulling_two_vertices_together_costs_energy(self) -> None:
        def energy(gap: float) -> float:
            return total_energy(pinched(gap), free(9), DISC, PARAMS, SHORT_EDGE)[0]

        assert energy(1e-6) > energy(0.03)

    def test_leaves_an_edge_between_fixed_vertices_alone(self) -> None:
        points = pinched(1e-3)
        fixed = free(len(points))
        fixed[[4, 5]] = True
        assert short_edge_barrier(points, fixed)[0] == 0.0


ATTRACTION = 0.3
ATTRACTION_RANGE = 3.0
LIMIT = 0.8
"""A pair's limit distance r_min, mm."""
ATTRACTING = EnergyParams(
    repulsion=0.7,
    falloff_range=FALLOFF_RANGE,
    clearance_weight=50.0,
    standoff=STANDOFF,
    points_per_edge=POINTS_PER_EDGE,
    attraction=ATTRACTION,
    attraction_range=ATTRACTION_RANGE,
)


def attraction(r: float, limit: float = LIMIT) -> tuple[float, float]:
    value, slope = attraction_integrand(ATTRACTING)(np.array([r]), np.array([limit]))
    return float(value[0]), float(slope[0])


class TestAttraction:
    def test_is_the_shifted_force_form_between_the_limit_and_the_range(
        self,
    ) -> None:
        w, cutoff = ATTRACTION, ATTRACTION_RANGE
        for r in (LIMIT, 1.2, 2.5):
            value, slope = attraction(r)
            assert value == pytest.approx(
                -w * (1 / r - 1 / cutoff) - w * (r - cutoff) / cutoff**2
            )
            assert slope == pytest.approx(w * (1 / r**2 - 1 / cutoff**2))
            # Its slope is positive: it pulls the pair together.
            assert value < 0.0 < slope

    def test_value_and_slope_vanish_at_the_range_and_beyond(self) -> None:
        near, near_slope = attraction(ATTRACTION_RANGE - 1e-4)
        assert abs(near) < 1e-8
        assert near_slope < 1e-4
        for r in (ATTRACTION_RANGE, ATTRACTION_RANGE + 0.5, 100.0):
            assert attraction(r) == (0.0, 0.0)

    def test_continues_linearly_inside_the_limit(self) -> None:
        at_limit, limit_slope = attraction(LIMIT)
        for r in (LIMIT - 0.1, 0.2, 0.0):
            value, slope = attraction(r)
            assert slope == pytest.approx(limit_slope)
            assert value == pytest.approx(at_limit + limit_slope * (r - LIMIT))

    def test_slope_is_the_derivative_of_the_value(self) -> None:
        for r in (0.3, LIMIT + 0.01, 1.7, ATTRACTION_RANGE - 0.01):
            numeric = (attraction(r + FD_STEP)[0] - attraction(r - FD_STEP)[0]) / (
                2 * FD_STEP
            )
            assert attraction(r)[1] == pytest.approx(numeric, rel=1e-6)

    def test_a_limit_beyond_the_range_attracts_nowhere(self) -> None:
        for r in (0.5, 3.5, 5.0):
            assert attraction(r, limit=ATTRACTION_RANGE + 1.0) == (0.0, 0.0)

    def test_without_weight_it_is_zero(self) -> None:
        off = attraction_integrand(PARAMS)(np.array([0.5, 1.0]), np.array([LIMIT] * 2))
        np.testing.assert_array_equal(off[0], 0.0)
        np.testing.assert_array_equal(off[1], 0.0)


SMOOTH_POINT = np.array([[5.0, 0.4]])
"""Beside the middle of `smooth_chain`, nearest one of its vertices, mm."""
EXACT = 1e-12
"""How far the smooth distance of one obstacle may part from the exact one,
mm: they take the same differences."""
MICROMETRE_THOUSANDTH = 1e-6
"""How far an obstacle farther than SMOOTH_REACH may move the smooth
distance, mm."""


def smooth_chain() -> ChainField:
    """A straight track along y = 0 of 100 edges 0.1 mm long, one obstacle,
    kept 0.2 mm from."""
    points = np.column_stack([np.linspace(0.0, 10.0, 101), np.zeros(101)])
    track = ChainCentreline("track", "T", 0.2, 0.2, points)
    return chains_field([(track, 0.2)])


class TestSmoothMinimum:
    def test_one_obstacle_is_its_exact_distance(self) -> None:
        smooth, share = smooth_minimum(np.array([0]), np.array([0.3]), 1)
        assert smooth[0] == pytest.approx(0.3, abs=EXACT)
        assert share[0] == 1.0

    def test_a_farther_obstacle_beyond_the_reach_leaves_the_nearer(self) -> None:
        far = 0.1 + SMOOTH_REACH + SMOOTH_WIDTH
        smooth, share = smooth_minimum(np.array([0, 0]), np.array([0.1, far]), 1)
        assert smooth[0] == pytest.approx(0.1, abs=MICROMETRE_THOUSANDTH)
        np.testing.assert_array_equal(share, [1.0, 0.0])

    def test_two_equally_near_are_seen_tau_ln_2_nearer(self) -> None:
        smooth, share = smooth_minimum(np.array([0, 0]), np.array([0.2, 0.2]), 1)
        assert smooth[0] == pytest.approx(0.2 - SMOOTH_WIDTH * math.log(2), abs=EXACT)
        np.testing.assert_allclose(share, [0.5, 0.5])

    def test_a_point_with_none_is_infinitely_far(self) -> None:
        smooth, _ = smooth_minimum(np.array([1]), np.array([0.2]), 2)
        assert smooth[0] == math.inf
        assert smooth[1] == pytest.approx(0.2)

    def test_a_straight_track_of_many_edges_is_one_obstacle(self) -> None:
        field = smooth_chain()

        smooth, toward = field.smooth_distance(SMOOTH_POINT, 1.0)

        exact, exact_toward = field.signed_distance(SMOOTH_POINT)
        assert smooth[0] == pytest.approx(exact[0], abs=EXACT)
        np.testing.assert_allclose(toward, exact_toward, atol=EXACT)


def test_a_point_between_two_walls_has_the_slope_its_gradient_gives() -> None:
    """A straight track midway between two walls: the energy's slope across
    it, to either side, is the gradient's, 0, where the nearest wall alone
    would give a kink."""
    points = np.column_stack([np.linspace(0.0, 2.0, 5), np.zeros(5)])
    fixed = np.array([True, False, False, False, True])
    walls = Walls(0.3)
    _, gradient = total_energy(points, fixed, walls, PARAMS)

    def energy(at: FloatArray) -> float:
        return total_energy(at, fixed, walls, PARAMS)[0]

    step = np.zeros_like(points)
    step[2, 1] = FD_STEP
    up = (energy(points + step) - energy(points)) / FD_STEP
    down = (energy(points) - energy(points - step)) / FD_STEP
    # One-sided differences err by the curvature times the step.
    scale = abs(energy(points)) * FD_STEP / SMOOTH_WIDTH
    assert gradient[2, 1] == pytest.approx(0.0, abs=EXACT)
    assert up == pytest.approx(gradient[2, 1], abs=scale)
    assert down == pytest.approx(gradient[2, 1], abs=scale)


WIDE = 5.0
"""Half the gap of a channel whose far wall is beyond every term's reach, mm."""
NARROW = 0.1
"""Half the gap of a channel within the standoff of both walls, mm."""


def channel_gradient(points: FloatArray, walls: Walls) -> FloatArray:
    """The gradient of `edge_integral` of PARAMS' obstacle terms at `points`
    in the channel of `walls`."""
    at = quadrature_points(points, POINTS_PER_EDGE)
    distance, direction = walls.smooth_distance(at)
    return edge_integral(
        points, distance, direction, obstacle_integrand(PARAMS), POINTS_PER_EDGE
    )[1]


@pytest.mark.parametrize(
    ("half_gap", "y"),
    [
        (WIDE, -WIDE + 0.1),
        (WIDE, -WIDE + 0.7),
        (WIDE, -WIDE + 2.0),
        (NARROW, 0.0),
        (NARROW, 0.004),
    ],
    ids=["within standoff", "within falloff", "beyond", "midway", "off midway"],
)
def test_edge_curvature_is_the_diagonal_curvature_between_flat_walls(
    half_gap: float, y: float
) -> None:
    # Edges parallel to flat walls: each wall's distance is linear, and moves
    # across the walls do not change edge lengths to first order, so the
    # diagonal curvature across them is exactly the terms edge_curvature
    # keeps, the smooth minimum's between the walls among them.
    walls = Walls(half_gap)
    points = np.column_stack([[0.0, 0.3, 0.5, 0.9, 1.0], np.full(5, y)])
    at = quadrature_points(points, POINTS_PER_EDGE)
    distance, direction = walls.smooth_distance(at)

    stiffness = edge_curvature(
        points,
        distance,
        direction,
        obstacle_integrand(PARAMS),
        obstacle_curvature(PARAMS),
        POINTS_PER_EDGE,
    )

    numeric = np.zeros(len(points))
    for i in range(len(points)):
        step = np.zeros_like(points)
        step[i, 1] = CURVATURE_STEP
        numeric[i] = (
            channel_gradient(points + step, walls)[i, 1]
            - channel_gradient(points - step, walls)[i, 1]
        ) / (2 * CURVATURE_STEP)
    np.testing.assert_allclose(
        stiffness, numeric, rtol=CURVATURE_RTOL, atol=CURVATURE_ATOL
    )


def test_inside_an_obstacle_the_curvature_is_the_boundary_s() -> None:
    inside, boundary = obstacle_curvature(PARAMS)(np.array([-0.05, 0.0]))
    hinge_inside, hinge_boundary = hinge_curvature(PARAMS)(np.array([-0.05, 0.0]))

    assert inside == boundary > 0.0
    assert hinge_inside == hinge_boundary > 0.0


def test_the_hinge_curvature_is_its_slope_s_derivative_within_the_standoff() -> None:
    distance = PARAMS.standoff * np.array([0.1, 0.5, 0.9])
    _, up = hinge_integrand(PARAMS)(distance + CURVATURE_STEP)
    _, down = hinge_integrand(PARAMS)(distance - CURVATURE_STEP)

    np.testing.assert_allclose(
        hinge_curvature(PARAMS)(distance),
        (up - down) / (2 * CURVATURE_STEP),
        rtol=CURVATURE_RTOL,
        atol=CURVATURE_ATOL,
    )
