from collections.abc import Callable

import numpy as np
import pytest
from distance_fields import Box, Disc, FloatArray, Walls
from numpy.typing import NDArray

from kibends.energy import (
    DistanceField,
    EnergyParams,
    bending_energy,
    clearance_energy,
    free_vertex_objective,
    length_energy,
    repulsion_energy,
    total_energy,
)

Term = Callable[[FloatArray, NDArray[np.bool_]], tuple[float, FloatArray]]

FD_STEP = 1e-6
GRADIENT_RTOL = 1e-5
POINTS_PER_EDGE = 3
FALLOFF_RANGE = 1.5
STANDOFF = 0.2

DISC = Disc(0.0, 0.0, 1.0)
BOX = Box(0.0, 0.0, 1.0, 0.6)
PARAMS = EnergyParams(
    stiffness=2.0,
    repulsion=0.7,
    falloff_range=FALLOFF_RANGE,
    clearance_weight=50.0,
    standoff=STANDOFF,
    points_per_edge=POINTS_PER_EDGE,
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
    return with_field(
        lambda p, f, d: repulsion_energy(p, f, d, FALLOFF_RANGE, POINTS_PER_EDGE), field
    )


def clearance(field: DistanceField) -> Term:
    return with_field(
        lambda p, f, d: clearance_energy(p, f, d, STANDOFF, POINTS_PER_EDGE), field
    )


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


class TestBending:
    def test_straight_polyline_is_zero(self) -> None:
        points = np.stack([np.linspace(0, 5, 6), np.linspace(1, 3, 6)], axis=1)
        value, gradient = bending_energy(points, free(6))
        assert value == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(gradient, 0.0, atol=1e-12)

    @pytest.mark.parametrize("vertices", [33, 65])
    def test_circle_arc_approximates_integral_of_curvature_squared(
        self, vertices: int
    ) -> None:
        radius, angle = 5.0, np.pi / 2
        value, _ = bending_energy(arc(radius, angle, vertices), free(vertices))
        # The two end vertices carry no turning, an O(1/n) shortfall.
        assert value == pytest.approx(angle / radius, rel=0.05)

    def test_resolution_independent(self) -> None:
        radius, angle = 5.0, np.pi / 2
        coarse, _ = bending_energy(arc(radius, angle, 33), free(33))
        fine, _ = bending_energy(arc(radius, angle, 65), free(65))
        assert fine == pytest.approx(coarse, rel=0.03)

    @pytest.mark.parametrize("points", NEAR_DISC + NEAR_BOX)
    def test_gradient(self, points: FloatArray) -> None:
        assert_gradient_matches(bending_energy, points)

    def test_gradient_beyond_smooth_turn_limit(self) -> None:
        hairpin = np.array([[0.0, 0.0], [2.0, 0.0], [0.1, 0.05], [0.0, 1.0]])
        assert_gradient_matches(bending_energy, hairpin)

    def test_full_reversal_is_finite_and_costly(self) -> None:
        reversal = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        right_angle = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        value, gradient = bending_energy(reversal, free(3))
        assert np.isfinite(value)
        assert np.all(np.isfinite(gradient))
        assert value > bending_energy(right_angle, free(3))[0]

    def test_turn_cost_grows_monotonically_up_to_reversal(self) -> None:
        angles = np.linspace(0.1, np.pi, 50)
        values = [
            bending_energy(
                np.array([[-1.0, 0.0], [0.0, 0.0], [np.cos(a), np.sin(a)]]), free(3)
            )[0]
            for a in angles
        ]
        assert np.all(np.diff(values) > 0)


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
            (PARAMS.stiffness**2, bending_energy(points, fixed)),
            (
                PARAMS.repulsion,
                repulsion_energy(points, fixed, DISC, FALLOFF_RANGE, POINTS_PER_EDGE),
            ),
            (
                PARAMS.clearance_weight,
                clearance_energy(points, fixed, DISC, STANDOFF, POINTS_PER_EDGE),
            ),
        ]
        value, gradient = total_energy(points, fixed, DISC, PARAMS)
        assert value == pytest.approx(sum(w * v for w, (v, _) in terms))
        np.testing.assert_allclose(gradient, sum(w * g for w, (_, g) in terms))

    def test_non_finite_energy_is_an_error(self) -> None:
        class NanField:
            def signed_distance(
                self, points: FloatArray
            ) -> tuple[FloatArray, FloatArray]:
                return np.full(len(points), np.nan), np.zeros_like(points)

        with pytest.raises(ValueError, match="finite"):
            total_energy(NEAR_DISC[0], free(len(NEAR_DISC[0])), NanField(), PARAMS)


ALL_TERMS: dict[str, Term] = {
    "length": length_energy,
    "bending": bending_energy,
    "repulsion": repulsion(DISC),
    "clearance": clearance(DISC),
    "total": total(DISC),
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


class TestFreeVertexObjective:
    def test_matches_total_on_free_vertices(self) -> None:
        points = NEAR_BOX[0]
        fixed = free(len(points))
        fixed[[0, -1]] = True
        objective = free_vertex_objective(points, fixed, BOX, PARAMS)
        value, flat_gradient = objective(points[~fixed].ravel())
        expected_value, expected_gradient = total_energy(points, fixed, BOX, PARAMS)
        assert value == pytest.approx(expected_value)
        np.testing.assert_allclose(flat_gradient, expected_gradient[~fixed].ravel())

    def test_moves_only_free_vertices(self) -> None:
        points = NEAR_BOX[1]
        fixed = free(len(points))
        fixed[[0, -1]] = True
        objective = free_vertex_objective(points, fixed, BOX, PARAMS)
        moved = points.copy()
        moved[~fixed] += 0.01
        value, _ = objective(moved[~fixed].ravel())
        assert value == pytest.approx(total_energy(moved, fixed, BOX, PARAMS)[0])
        np.testing.assert_array_equal(points, NEAR_BOX[1])
