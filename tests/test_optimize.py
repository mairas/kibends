import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scenarios
from distance_fields import Box, Disc, FloatArray, FreeSpace, Union, Walls
from scenarios import PARAMS, Field, Scenario, chain, path
from shapely import wkt

import kibends.optimize as optimize_module
from kibends.obstacles import Obstacle, Obstacles
from kibends.optimize import (
    OptimizeResult,
    OptimizerSettings,
    Termination,
    measure,
    optimize,
)
from kibends.simplify import DEFAULT_CHORD_TOLERANCE

SETTINGS = OptimizerSettings()
# Positions of a straight result within this of the ideal line, mm: the
# energy cannot resolve smaller bows, and simplify flattens them anyway.
STRAIGHT_TOLERANCE = DEFAULT_CHORD_TOLERANCE


def run(scenario: Scenario, settings: OptimizerSettings = SETTINGS) -> OptimizeResult:
    return optimize(
        scenario.polyline, scenario.fixed, scenario.field, scenario.params, settings
    )


def all_edges_clear(points: FloatArray, field: Field) -> bool:
    return bool(field.segments_clear(points[:-1], points[1:]).all())


def distance_to_line(
    points: FloatArray, start: FloatArray, end: FloatArray
) -> FloatArray:
    direction = (end - start) / np.linalg.norm(end - start)
    offset = points - start
    return np.asarray(np.abs(offset[:, 0] * direction[1] - offset[:, 1] * direction[0]))


def y_at(points: FloatArray, x: float) -> float:
    """y where the polyline crosses the vertical line at x (x increasing)."""
    return float(np.interp(x, points[:, 0], points[:, 1]))


def test_l_path_in_free_space_becomes_straight() -> None:
    polyline, fixed = chain((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))

    result = optimize(polyline, fixed, FreeSpace(), PARAMS, SETTINGS)

    assert result.termination is Termination.CONVERGED
    start, end = polyline[0], polyline[-1]
    assert np.array_equal(result.polyline[[0, -1]], polyline[[0, -1]])
    assert distance_to_line(result.polyline, start, end).max() < STRAIGHT_TOLERANCE
    assert result.after.length == pytest.approx(np.hypot(10.0, 10.0), abs=1e-4)
    assert result.before.length == pytest.approx(20.0)
    assert result.after.segment_count == len(result.polyline) - 1


SOFT, STIFF = 0.1, 1.5


@pytest.fixture(scope="module")
def soft() -> OptimizeResult:
    return run(scenarios.wrap(stiffness=SOFT))


@pytest.fixture(scope="module")
def stiff() -> OptimizeResult:
    return run(scenarios.wrap(stiffness=STIFF))


class TestWrap:
    def test_converges(self, soft: OptimizeResult, stiff: OptimizeResult) -> None:
        assert soft.termination is Termination.CONVERGED
        assert stiff.termination is Termination.CONVERGED

    def test_every_edge_stays_outside_the_inflated_box(
        self, soft: OptimizeResult, stiff: OptimizeResult
    ) -> None:
        for result in (soft, stiff):
            assert all_edges_clear(result.polyline, scenarios.WRAP_BOX)
            assert result.after.min_clearance >= 0.0

    def test_wraps_the_box_and_is_shorter(self, soft: OptimizeResult) -> None:
        corner = np.array([8.5, 1.5])
        distance, _ = scenarios.WRAP_BOX.signed_distance(soft.polyline)
        assert distance.min() < PARAMS.standoff
        nearest = soft.polyline[np.argmin(distance)]
        assert np.linalg.norm(nearest - corner) < 1.0
        assert soft.after.length < soft.before.length

    def test_stiffness_widens_the_bend(
        self, soft: OptimizeResult, stiff: OptimizeResult
    ) -> None:
        assert soft.after.tightest_bend_radius == pytest.approx(
            scenarios.ENVELOPE, rel=0.3
        )
        assert stiff.after.tightest_bend_radius > 2 * soft.after.tightest_bend_radius

    def test_rerun_on_own_output_is_near_no_op(self, soft: OptimizeResult) -> None:
        again = optimize(
            soft.polyline, soft.fixed, scenarios.WRAP_BOX, scenarios.wrap(SOFT).params
        )

        assert again.rejected_edges == 0
        assert again.termination is Termination.NOTHING_TO_DO
        assert np.array_equal(again.polyline, soft.polyline)


class TestChannel:
    def test_repulsion_centres_the_track(self) -> None:
        result = run(scenarios.channel(repulsion=1.0))

        assert result.termination is Termination.CONVERGED
        middle = result.polyline[
            (result.polyline[:, 0] > 7.0) & (result.polyline[:, 0] < 13.0)
        ]
        assert np.abs(middle[:, 1]).max() < 0.05
        assert result.after.mean_clearance > result.before.mean_clearance

    def test_without_repulsion_length_keeps_it_straight(self) -> None:
        scenario = scenarios.channel(repulsion=0.0)

        result = run(scenario)

        assert result.termination is Termination.NOTHING_TO_DO
        assert np.array_equal(result.polyline, scenario.polyline)


def test_path_left_of_disc_stays_left() -> None:
    result = run(scenarios.topology())

    assert result.termination is Termination.CONVERGED
    assert all_edges_clear(result.polyline, scenarios.TOPOLOGY_DISC)
    assert y_at(result.polyline, 0.0) > scenarios.TOPOLOGY_DISC.radius
    assert result.after.length < result.before.length


def test_path_at_exact_clearance_is_not_frozen_and_moves() -> None:
    scenario = scenarios.pad_row()

    result = run(scenario)

    assert result.frozen == ()
    assert result.termination is Termination.CONVERGED
    assert y_at(result.polyline, 5.0) < -0.05
    assert all_edges_clear(result.polyline, scenario.field)


def test_vertex_deep_inside_obstacle_is_frozen_and_kept() -> None:
    scenario = scenarios.frozen()

    result = run(scenario)

    assert len(result.frozen) == 1
    frozen = result.frozen[0]
    assert frozen.position == scenarios.FROZEN_VERTEX
    assert frozen.distance == pytest.approx(-0.5)
    kept = np.all(result.polyline == scenarios.FROZEN_VERTEX, axis=1)
    assert kept.sum() == 1
    assert result.fixed[kept].all()
    assert result.after.length < result.before.length


# Enough outer iterations for the detour to reach the pocket's corners, where
# edges cut across them and are rejected.
DETOUR_REJECTING_ITERATIONS = 6


def test_edges_cutting_a_corner_are_rejected() -> None:
    scenario = scenarios.detour()

    result = run(
        scenario, replace(SETTINGS, max_iterations=DETOUR_REJECTING_ITERATIONS)
    )

    assert result.rejected_edges > 0
    assert all_edges_clear(result.polyline, scenario.field)


@pytest.mark.parametrize("cap", [8, 10])
def test_result_stopped_at_the_cap_adds_no_crossing(cap: int) -> None:
    # Resampling lays chords across the pocket's corners in these iterations.
    scenario = scenarios.detour()

    result = run(scenario, replace(SETTINGS, max_iterations=cap))

    assert all_edges_clear(result.polyline, scenario.field)


FROZEN_DISC_RADIUS = 0.3


@pytest.mark.slow
@pytest.mark.parametrize("x", [3.0, 9.0, 16.0])
def test_detour_with_a_frozen_vertex_is_still_pulled_tight(x: float) -> None:
    detour = scenarios.detour()
    field = Union([*detour.field.shapes, Disc(x, 0.0, FROZEN_DISC_RADIUS)])
    polyline = np.insert(detour.polyline, 1 if x < 9.85 else 5, [x, 0.0], axis=0)

    result = optimize(
        polyline, scenarios.endpoints_fixed(polyline), field, detour.params, SETTINGS
    )

    assert result.termination is Termination.CONVERGED
    assert len(result.frozen) == 1
    assert result.after.length == pytest.approx(scenarios.DETOUR_LENGTH, abs=1e-2)


@pytest.mark.slow
def test_detour_in_narrow_channel_is_pulled_tight() -> None:
    scenario = scenarios.detour()

    result = run(scenario)

    assert result.termination is Termination.CONVERGED
    assert result.rejected_edges > 0
    assert all_edges_clear(result.polyline, scenario.field)
    assert result.before.length == pytest.approx(scenarios.DETOUR_LENGTH + 5.0)
    assert result.after.length == pytest.approx(scenarios.DETOUR_LENGTH, abs=1e-3)
    # Bubble-limited: about 62 outer iterations with the default settings.
    assert result.iterations <= 70


class TestNothingToDo:
    def test_free_vertices_already_straight_between_pins(self) -> None:
        polyline = path((0.0, 0.0), (2.0, 0.0), (5.0, 0.0), (7.0, 0.0), (10.0, 0.0))
        fixed = np.array([True, False, True, False, True])

        result = optimize(polyline, fixed, FreeSpace(), PARAMS, SETTINGS)

        assert result.termination is Termination.NOTHING_TO_DO
        assert np.array_equal(result.polyline, polyline)
        assert result.after == result.before

    def test_chain_shorter_than_the_spacing(self) -> None:
        polyline, fixed = chain((0.0, 0.0), (0.03, 0.04))

        result = optimize(polyline, fixed, FreeSpace(), PARAMS, SETTINGS)

        assert result.termination is Termination.NOTHING_TO_DO
        assert result.iterations == 0
        assert np.array_equal(result.polyline, polyline)


def test_narrow_gap_is_threaded_not_jumped() -> None:
    scenario = scenarios.narrow_gap()

    result = run(scenario)

    assert result.termination is Termination.CONVERGED
    assert all_edges_clear(result.polyline, scenario.field)
    upper, lower = scenarios.GAP_DISCS
    through = y_at(result.polyline, upper.cx)
    assert lower.cy + lower.radius < through < upper.cy - upper.radius


def test_endpoints_must_be_fixed() -> None:
    polyline = path((0.0, 0.0), (1.0, 0.0))
    with pytest.raises(ValueError, match="endpoints"):
        optimize(polyline, np.array([True, False]), FreeSpace(), PARAMS)


def test_measure_reports_clearance_and_bend() -> None:
    disc = Disc(0.0, 0.0, 1.0)
    angles = np.linspace(0.0, np.pi, 200)
    radius = 1.5
    arc = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)

    stats = measure(arc, disc, spacing=0.05)

    assert stats.min_clearance == pytest.approx(radius - 1.0, abs=1e-3)
    assert stats.mean_clearance == pytest.approx(radius - 1.0, abs=1e-3)
    assert stats.tightest_bend_radius == pytest.approx(radius, rel=1e-2)
    assert stats.length == pytest.approx(np.pi * radius, rel=1e-4)


def test_measure_gives_a_sharp_corner_a_small_radius() -> None:
    polyline = path((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))

    stats = measure(polyline, Box(50.0, 50.0, 1.0, 1.0), spacing=0.1)

    assert stats.tightest_bend_radius < 0.1


def test_settings_reject_nonpositive_spacing() -> None:
    with pytest.raises(ValueError, match="spacing"):
        replace(SETTINGS, spacing=0.0)


def test_iteration_cap_is_reported_with_a_valid_polyline() -> None:
    scenario = scenarios.wrap()

    result = run(scenario, replace(SETTINGS, max_iterations=2))

    assert result.termination is Termination.ITERATION_CAP
    assert result.iterations == 2
    assert all_edges_clear(result.polyline, scenario.field)
    assert result.after.length < result.before.length


class MisleadingWalls(Walls):
    """Walls whose distance gradient points the wrong way."""

    def signed_distance(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        distance, gradient = super().signed_distance(points)
        return distance, -gradient


def test_abnormal_lbfgsb_termination_is_reported() -> None:
    polyline, fixed = chain((0.0, 0.0), (5.0, 0.8), (10.0, 0.0))
    params = replace(PARAMS, repulsion=5.0)

    result = optimize(polyline, fixed, MisleadingWalls(1.0), params)

    assert result.termination is Termination.ABNORMAL
    assert np.array_equal(result.polyline[[0, -1]], polyline[[0, -1]])


STALL_FIXTURE = Path(__file__).parent / "fixtures" / "stall" / "vin_sense_chain.json"
STALL_ITERATION_LIMIT = 30


@pytest.mark.slow
def test_flat_energy_on_a_real_board_stops_when_it_stalls() -> None:
    # A real chain whose free stretch sits at the fringe of the repulsion
    # range: the shape keeps wandering by tens of µm while the energy no
    # longer improves, so only an energy-based stop ends it.
    data = json.loads(STALL_FIXTURE.read_text())
    field = Obstacles(
        [
            Obstacle("fixture", wkt.loads(o["geometry"]), o["inflation"])
            for o in data["obstacles"]
        ]
    )
    params = replace(PARAMS, repulsion=0.5, falloff_range=1.0)

    result = optimize(
        np.array(data["points"]),
        np.array(data["pinned"]),
        field,
        params,
        OptimizerSettings(max_iterations=STALL_ITERATION_LIMIT),
    )

    assert result.termination is Termination.CONVERGED
    assert result.iterations < STALL_ITERATION_LIMIT
    assert result.after.min_clearance >= 0.0


class TestStalled:
    # Binary-exact values, so the boundary comparison is not a rounding test.
    SETTINGS = OptimizerSettings(stall_tolerance=0.5, stall_window=3)

    def test_needs_a_full_window_of_history(self) -> None:
        assert not optimize_module._stalled([5.0, 5.0, 5.0], self.SETTINGS)

    def test_improvement_of_exactly_the_tolerance_is_a_stall(self) -> None:
        assert optimize_module._stalled([5.0, 4.75, 4.625, 4.5], self.SETTINGS)

    def test_larger_improvement_over_the_window_keeps_going(self) -> None:
        assert not optimize_module._stalled([5.0, 4.75, 4.625, 4.25], self.SETTINGS)
