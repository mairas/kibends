"""Single chains through `optimize.optimize`: shapes, clearance, frozen
vertices and stop rules, on the scenarios of `tests/scenarios.py`."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scenarios
import shapely
from distance_fields import Box, Disc, FloatArray, FreeSpace, MisleadingWalls, Union
from scenarios import PARAMS, Scenario, chain, path
from shapely import wkt

import kibends.optimize as optimize_module
from kibends.bursts import INSIDE_TOLERANCE, OptimizerSettings
from kibends.energy import BoolArray, ObstacleField, total_energy
from kibends.obstacles import Obstacle, Obstacles
from kibends.optimize import MAX_ABNORMAL_BURSTS, optimize
from kibends.polylines import measure, resample, violations
from kibends.results import OptimizeResult, Termination

SETTINGS = OptimizerSettings()
# For long runs: about 20x faster than the defaults and, on the detour and
# stall scenarios, still converging within 1 µm of the default result's length.
COARSE = replace(SETTINGS, spacing=0.3, settled_change=0.003, burst_maxiter=30)
# Positions of a straight result within this of the ideal line, mm: the
# optimizer leaves the L path bowed by 7 µm at L_PATH_SETTINGS.
STRAIGHT_TOLERANCE = 0.01


def run(scenario: Scenario, settings: OptimizerSettings = SETTINGS) -> OptimizeResult:
    return optimize(
        scenario.polyline, scenario.fixed, scenario.field, scenario.params, settings
    )


def all_edges_clear(points: FloatArray, field: ObstacleField) -> bool:
    return bool(field.segments_clear(points[:-1], points[1:]).all())


def distance_to_line(
    points: FloatArray, start: FloatArray, end: FloatArray
) -> FloatArray:
    direction = (end - start) / np.linalg.norm(end - start)
    offset = points - start
    cross: FloatArray = offset[:, 0] * direction[1] - offset[:, 1] * direction[0]
    return np.abs(cross)


def y_at(points: FloatArray, x: float) -> float:
    """y where the polyline crosses the vertical line at x (x increasing)."""
    return float(np.interp(x, points[:, 0], points[:, 1]))


# Stops on a residual bow of about 15 µm, as the defaults do, four times
# faster.
L_PATH_SETTINGS = replace(
    SETTINGS, spacing=0.2, settled_change=0.002, burst_maxiter=100
)


def test_l_path_in_free_space_becomes_straight() -> None:
    polyline, fixed = chain((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))

    result = optimize(polyline, fixed, FreeSpace(), PARAMS, L_PATH_SETTINGS)

    assert result.termination is Termination.CONVERGED
    start, end = polyline[0], polyline[-1]
    assert np.array_equal(result.polyline[[0, -1]], polyline[[0, -1]])
    assert distance_to_line(result.polyline, start, end).max() < STRAIGHT_TOLERANCE
    assert result.after.length == pytest.approx(np.hypot(10.0, 10.0), abs=1e-4)
    assert result.before.length == pytest.approx(20.0)
    assert result.after.segment_count == len(result.polyline) - 1


# Resolves the 0.35 mm corner radius to within 5%, several times faster than
# the defaults.
WRAP_SETTINGS = replace(
    SETTINGS, spacing=0.25, settled_change=0.0025, burst_maxiter=100
)
# A rerun on the wrap's own output wanders along the flat stretches before
# it settles, by amounts that vary with the platform's floating point.
# Measured: 4 iterations on macOS and Linux; 8.2 µm and 6.6 µm moved; the
# length changed by -0.16 µm, -67 nm and +82 nm.
RERUN_ITERATIONS = 6
RERUN_DRIFT = 0.015
RERUN_LENGTH_SLACK = 1e-3


@pytest.fixture(scope="module")
def wrapped() -> OptimizeResult:
    return run(scenarios.wrap(), WRAP_SETTINGS)


class TestWrap:
    def test_converges(self, wrapped: OptimizeResult) -> None:
        assert wrapped.termination is Termination.CONVERGED

    def test_every_edge_stays_outside_the_inflated_box(
        self, wrapped: OptimizeResult
    ) -> None:
        assert all_edges_clear(wrapped.polyline, scenarios.WRAP_BOX)
        assert wrapped.after.min_clearance >= 0.0

    def test_wraps_the_box_and_is_shorter(self, wrapped: OptimizeResult) -> None:
        corner = np.array([8.5, 1.5])
        distance, _ = scenarios.WRAP_BOX.signed_distance(wrapped.polyline)
        # Sampled along the edges: at this spacing no vertex need sit in the
        # corner, but the track between them still hugs it.
        assert wrapped.after.min_clearance < PARAMS.standoff
        nearest = wrapped.polyline[np.argmin(distance)]
        assert np.linalg.norm(nearest - corner) < 1.0
        assert wrapped.after.length < wrapped.before.length

    def test_bends_at_the_corner_s_radius(self, wrapped: OptimizeResult) -> None:
        assert wrapped.after.tightest_bend_radius == pytest.approx(
            scenarios.ENVELOPE, rel=0.3
        )

    def test_rerun_on_own_output_is_near_no_op(self, wrapped: OptimizeResult) -> None:
        again = optimize(
            wrapped.polyline,
            wrapped.fixed,
            scenarios.WRAP_BOX,
            scenarios.wrap().params,
            WRAP_SETTINGS,
        )

        assert again.rejected_edges == 0
        assert again.termination is Termination.CONVERGED
        assert again.iterations <= RERUN_ITERATIONS
        moved = shapely.LineString(again.polyline).hausdorff_distance(
            shapely.LineString(wrapped.polyline)
        )
        assert moved < RERUN_DRIFT
        assert again.after.length <= wrapped.after.length + RERUN_LENGTH_SLACK


class TestChannel:
    def test_repulsion_centres_the_track(self) -> None:
        result = run(scenarios.channel(repulsion=1.0), COARSE)

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
    result = run(scenarios.topology(), COARSE)

    assert result.termination is Termination.CONVERGED
    assert all_edges_clear(result.polyline, scenarios.TOPOLOGY_DISC)
    assert y_at(result.polyline, 0.0) > scenarios.TOPOLOGY_DISC.radius
    assert result.after.length < result.before.length


def test_path_at_exact_clearance_is_not_frozen_and_moves() -> None:
    scenario = scenarios.pad_row()

    result = run(scenario, COARSE)

    assert result.frozen == ()
    assert result.termination is Termination.CONVERGED
    assert y_at(result.polyline, 5.0) < -0.05
    assert all_edges_clear(result.polyline, scenario.field)


# The free vertices beside the run stuck inside the disc stand off at the
# clearance, so the track jogs out of the run at both its ends. That costs up
# to 44 µm over the 10.002 mm wrap, unevenly with the spacing: 10.022, 10.007,
# 10.041, 10.005 and 10.044 mm at 0.1, 0.15, 0.2, 0.25 and 0.3 mm.
JOG_ALLOWANCE = 0.05


def test_input_that_cuts_a_corner_is_still_pulled_tight() -> None:
    # The edge from (4, -0.9) to (6, -0.9) cuts 100 µm into the disc;
    # resampling puts free vertices inside it.
    field = Disc(5.0, 0.0, 1.0)
    polyline, fixed = chain(
        (0.0, -0.9), (2.5, -3.0), (4.0, -0.9), (6.0, -0.9), (10.0, -0.9)
    )

    result = optimize(polyline, fixed, field, PARAMS, COARSE)

    assert result.after.length == pytest.approx(10.0, abs=JOG_ALLOWANCE)
    clear = field.segments_clear(result.polyline[:-1], result.polyline[1:])
    crossing_starts = ~clear & np.concatenate(([True], clear[:-1]))
    # At most the one run resampling laid inside the disc still crosses it;
    # the bursts usually move it out entirely.
    assert crossing_starts.sum() <= 1


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


# Too weak to hold the taut wrap off the box's corner: bursts cut across it,
# and only the edge check moves them back.
WEAK_CLEARANCE_WEIGHT = 1.0


def test_edges_cutting_a_corner_are_rejected() -> None:
    scenario = scenarios.wrap()
    params = replace(scenario.params, clearance_weight=WEAK_CLEARANCE_WEIGHT)

    result = run(replace(scenario, params=params), COARSE)

    assert result.termination is Termination.CONVERGED
    assert result.rejected_edges > 0
    assert all_edges_clear(result.polyline, scenario.field)


# Resampling lays chords across the pocket's corners at caps 8 and 10; at 13
# to 20 the lowest-energy burst so far has vertices up to 31 µm inside the
# clearance and, at 17 and 20, an edge 52 µm inside. A sweep of caps 1 to 30
# takes about 6 s, so these stand in for it.
@pytest.mark.parametrize("cap", [8, 10, 13, 15, 17, 20])
def test_result_stopped_at_the_cap_adds_no_crossing(cap: int) -> None:
    scenario = scenarios.detour()

    result = run(scenario, replace(COARSE, max_iterations=cap))

    assert all_edges_clear(result.polyline, scenario.field)


FROZEN_DISC_RADIUS = 0.3


# At 0.3 mm, with the frozen vertex at x = 16, every burst from the third
# presses the track into the pocket's sharp corners, so none becomes the best
# state and the run settles 3.4 mm long; at 0.2 mm every x pulls tight.
FROZEN_DETOUR = replace(COARSE, spacing=0.2)


@pytest.mark.parametrize("x", [3.0, 9.0, 16.0])
def test_detour_with_a_frozen_vertex_is_still_pulled_tight(x: float) -> None:
    detour = scenarios.detour()
    assert isinstance(detour.field, Union)
    field = Union([*detour.field.shapes, Disc(x, 0.0, FROZEN_DISC_RADIUS)])
    polyline = np.insert(detour.polyline, 1 if x < 9.85 else 5, [x, 0.0], axis=0)

    result = optimize(
        polyline,
        scenarios.endpoints_fixed(polyline),
        field,
        detour.params,
        FROZEN_DETOUR,
    )

    assert result.termination is Termination.CONVERGED
    assert len(result.frozen) == 1
    assert result.after.length == pytest.approx(scenarios.DETOUR_LENGTH, abs=1e-2)


# Without the guard, the result at this cap has a vertex 16 µm inside the
# pocket's clearance.
FROZEN_DETOUR_CAP = 13


def test_frozen_vertex_does_not_excuse_a_violation_elsewhere() -> None:
    detour = scenarios.detour()
    assert isinstance(detour.field, Union)
    frozen_at = (3.0, 0.0)
    field = Union([*detour.field.shapes, Disc(*frozen_at, FROZEN_DISC_RADIUS)])
    polyline = np.insert(detour.polyline, 1, frozen_at, axis=0)

    result = optimize(
        polyline,
        scenarios.endpoints_fixed(polyline),
        field,
        detour.params,
        replace(COARSE, max_iterations=FROZEN_DETOUR_CAP),
    )

    assert [vertex.position for vertex in result.frozen] == [frozen_at]
    assert np.all(result.polyline == frozen_at, axis=1).sum() == 1
    assert result.after.length < result.before.length
    assert all_edges_clear(result.polyline, detour.field)


SHALLOW_DEPTH = 0.001


def test_inside_run_holding_anchors_may_split() -> None:
    # Pins at both ends and a frozen vertex between them share one run of
    # vertices 1 µm inside the box; the standoff pushes the free ones out,
    # leaving a separate run at each anchor.
    frozen_at = (6.0, 0.0)
    field = Union(
        [
            Box(5.0, SHALLOW_DEPTH - 5.0, 6.0, 5.0),
            Disc(*frozen_at, FROZEN_DISC_RADIUS),
        ]
    )
    polyline, fixed = chain((0.0, 0.0), frozen_at, (10.0, 0.0))

    result = optimize(polyline, fixed, field, PARAMS, COARSE)

    assert [vertex.position for vertex in result.frozen] == [frozen_at]
    assert np.all(result.polyline == frozen_at, axis=1).sum() == 1
    assert y_at(result.polyline, 3.0) > SHALLOW_DEPTH
    assert y_at(result.polyline, 8.0) > SHALLOW_DEPTH


# A pad off the origin, so its clearance boundary is not binary-exact and a
# track routed along it resamples to vertices a few 1e-15 mm inside.
PAD_TOP = 45.35
PAD_INFLATION = 0.327
EXACT_PAD = Obstacles(
    [Obstacle("pad", shapely.box(11.9, PAD_TOP, 12.7, 45.85), PAD_INFLATION)]
)
EXACT_PAD_CAP = 2


def test_track_at_exact_clearance_starts_with_no_violation() -> None:
    at_clearance = PAD_TOP - PAD_INFLATION
    polyline, fixed = chain(
        (10.3, 45.6), (11.3, at_clearance), (13.3, at_clearance), (14.3, 45.6)
    )
    settings = replace(SETTINGS, max_iterations=EXACT_PAD_CAP)
    resampled, mask = resample(polyline, fixed, settings.spacing)

    result = optimize(polyline, fixed, EXACT_PAD, PARAMS, settings)

    assert violations(resampled, mask, EXACT_PAD) == 0
    distance, _ = EXACT_PAD.signed_distance(result.polyline)
    assert distance.min() >= -INSIDE_TOLERANCE
    assert all_edges_clear(result.polyline, EXACT_PAD)


# Measured: 6 outer iterations at COARSE.
DETOUR_ITERATIONS = 10


def test_detour_in_narrow_channel_is_pulled_tight() -> None:
    scenario = scenarios.detour()

    result = run(scenario, COARSE)

    assert result.termination is Termination.CONVERGED
    assert all_edges_clear(result.polyline, scenario.field)
    assert result.before.length == pytest.approx(scenarios.DETOUR_LENGTH + 5.0)
    assert result.after.length == pytest.approx(scenarios.DETOUR_LENGTH, abs=1e-3)
    assert result.iterations <= DETOUR_ITERATIONS


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


def test_bursts_that_all_fail_without_progress_end_the_run_abnormally() -> None:
    polyline, fixed = chain((0.0, 0.0), (5.0, 0.8), (10.0, 0.0))
    params = replace(PARAMS, repulsion=5.0)

    result = optimize(polyline, fixed, MisleadingWalls(1.0), params)

    assert result.termination is Termination.ABNORMAL
    assert result.abnormal_bursts == MAX_ABNORMAL_BURSTS
    assert np.array_equal(result.polyline[[0, -1]], polyline[[0, -1]])


def shortest_edge(points: FloatArray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).min())


# Starts at which two neighbouring vertices once collapsed onto each other in a
# burst, the gradient across the vanishing edge between them failed
# L-BFGS-B's line search, and the run ended abnormally. Starts 1 µm away from
# these converged, so the shifts matter.
@pytest.mark.parametrize("y", [0.8 + 1e-6, 0.5 + 2e-6, 0.7])
def test_channel_starts_that_collapsed_two_vertices_converge(y: float) -> None:
    channel = scenarios.channel(1.0)
    polyline, fixed = chain((0.0, y), (20.0, y))

    result = optimize(polyline, fixed, channel.field, channel.params)

    assert result.termination is Termination.CONVERGED
    assert result.abnormal_bursts == 0
    assert shortest_edge(result.polyline) > SETTINGS.short_edge


STALL_FIXTURE = Path(__file__).parent / "fixtures" / "stall" / "vin_sense_chain.json"
STALL_ITERATION_LIMIT = 30


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
        # With the stall stop disabled these settings run into the limit.
        replace(COARSE, max_iterations=STALL_ITERATION_LIMIT),
    )

    assert result.termination is Termination.CONVERGED
    assert result.iterations < STALL_ITERATION_LIMIT
    assert result.after.min_clearance >= 0.0


class TestStalled:
    # Binary-exact values, so the boundary comparison is not a rounding test.
    SETTINGS = OptimizerSettings(stall_tolerance=0.5, stall_window=3)

    def test_needs_a_full_window_of_history(self) -> None:
        assert not optimize_module.stalled([5.0, 5.0, 5.0], self.SETTINGS)

    def test_improvement_of_exactly_the_tolerance_is_a_stall(self) -> None:
        assert optimize_module.stalled([5.0, 4.75, 4.625, 4.5], self.SETTINGS)

    def test_larger_improvement_over_the_window_keeps_going(self) -> None:
        assert not optimize_module.stalled([5.0, 4.75, 4.625, 4.25], self.SETTINGS)


class ReachIgnored:
    """The stall fixture's obstacles, always answering exactly."""

    def __init__(self, field: Obstacles) -> None:
        self.field = field

    def signed_distance(
        self, points: FloatArray, reach: float = np.inf
    ) -> tuple[FloatArray, FloatArray]:
        return self.field.signed_distance(points)

    def segments_clear(
        self, starts: FloatArray, ends: FloatArray, tolerance: float = 0.0
    ) -> BoolArray:
        return self.field.segments_clear(starts, ends, tolerance)


def test_energy_with_a_reach_limited_field_is_unchanged() -> None:
    data = json.loads(STALL_FIXTURE.read_text())
    field = Obstacles(
        [
            Obstacle("fixture", wkt.loads(o["geometry"]), o["inflation"])
            for o in data["obstacles"]
        ]
    )
    params = replace(PARAMS, repulsion=0.5, falloff_range=1.0)
    points = np.array(data["points"])
    fixed = np.array(data["pinned"])

    value, gradient = total_energy(points, fixed, field, params)
    exact_value, exact_gradient = total_energy(
        points, fixed, ReachIgnored(field), params
    )

    assert value == pytest.approx(exact_value, rel=1e-12)
    assert gradient == pytest.approx(exact_gradient, abs=1e-12)


def test_a_settled_change_beyond_the_first_burst_still_keeps_it() -> None:
    scenario = scenarios.channel(repulsion=1.0)
    # Over twice what the first burst moves.
    lax = replace(COARSE, settled_change=2.0)

    result = optimize(
        scenario.polyline, scenario.fixed, scenario.field, scenario.params, lax
    )

    assert result.termination is Termination.CONVERGED
    assert result.iterations == 1
    assert result.after.mean_clearance > result.before.mean_clearance
