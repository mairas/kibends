"""Single chains through `optimize.optimize`: shapes, clearance, copper
that starts inside it and stop rules, on the scenarios of `tests/scenarios.py`."""

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import scenarios
import shapely
from conftest import clearances, total_energy, violations
from distance_fields import (
    Box,
    Disc,
    FloatArray,
    FreeSpace,
    MisleadingWalls,
    ReachIgnored,
    Union,
)
from scenarios import PARAMS, Scenario, chain, path
from shapely import wkt

import kibends.optimize as optimize_module
from kibends.bursts import (
    INSIDE_TOLERANCE,
    MAX_COARSE_LEVELS,
    Burst,
    Check,
    Objective,
    OptimizerSettings,
    Stop,
)
from kibends.energy import ObstacleField
from kibends.neighbours import ChainInput
from kibends.obstacles import ChainCentreline, Obstacle, Obstacles
from kibends.optimize import MAX_ABNORMAL_BURSTS, optimize, optimize_chains
from kibends.polylines import measure, resample
from kibends.results import OptimizeResult, Termination
from kibends.via_moves import MovingVias

SETTINGS = OptimizerSettings()
ONE_SPACING = replace(SETTINGS, coarse_spacings=())
"""The defaults at the final spacing alone, for a premise that counts
bursts at one spacing: a coarse level hands an abnormal streak over, and the
first burst after a handover settles if it moves nothing, abnormal or not."""
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
# A rerun at the defaults on the wrap's own output settles within these.
# Measured: 1 iteration, 1.2 µm moved, the length 4 nm longer. At
# WRAP_SETTINGS' coarser final spacing, resampling the output cuts its
# chords into the repulsion, and pushing them out runs the coarse levels
# again, rejecting edge moves.
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
        assert clearances(wrapped.polyline, scenarios.WRAP_BOX).min() >= 0.0

    def test_wraps_the_box_and_is_shorter(self, wrapped: OptimizeResult) -> None:
        corner = np.array([8.5, 1.5])
        distance, _ = scenarios.WRAP_BOX.signed_distance(wrapped.polyline)
        # Sampled along the edges: at this spacing no vertex need sit in the
        # corner, but the track between them still hugs it.
        assert clearances(wrapped.polyline, scenarios.WRAP_BOX).min() < PARAMS.standoff
        nearest = wrapped.polyline[np.argmin(distance)]
        assert np.linalg.norm(nearest - corner) < 1.0
        assert wrapped.after.length < wrapped.before.length

    def test_bends_at_the_corner_s_radius(self, wrapped: OptimizeResult) -> None:
        assert wrapped.after.tightest_bend_radius == pytest.approx(
            scenarios.ENVELOPE, rel=0.3
        )

    def test_rerun_on_own_output_is_near_no_op(self) -> None:
        wrapped = run(scenarios.wrap(), SETTINGS)
        again = optimize(
            wrapped.polyline,
            wrapped.fixed,
            scenarios.WRAP_BOX,
            scenarios.wrap().params,
            SETTINGS,
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
        scenario = scenarios.channel(repulsion=1.0)

        result = run(scenario, COARSE)

        assert result.termination is Termination.CONVERGED
        middle = result.polyline[
            (result.polyline[:, 0] > 7.0) & (result.polyline[:, 0] < 13.0)
        ]
        assert np.abs(middle[:, 1]).max() < 0.05
        assert (
            clearances(result.polyline, scenario.field).mean()
            > clearances(scenario.polyline, scenario.field).mean()
        )

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


def test_path_at_exact_clearance_moves() -> None:
    scenario = scenarios.pad_row()

    result = run(scenario, COARSE)

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


def test_a_vertex_deep_inside_an_obstacle_moves_out() -> None:
    scenario = scenarios.starts_inside()

    result = run(scenario, COARSE)

    assert not np.all(result.polyline == scenarios.INSIDE_VERTEX, axis=1).any()
    assert clearances(result.polyline, scenario.field).min() >= -INSIDE_TOLERANCE
    assert result.after.length < result.before.length


PAD_DEPTH = 0.02
"""How far the vertex starts inside the pad's clearance, mm."""


def test_a_vertex_just_inside_a_pad_s_clearance_ends_clear_of_it() -> None:
    pad = Box(5.0, 1.0, 1.0, 1.0, corner_radius=scenarios.ENVELOPE)
    polyline, fixed = chain((0.0, -1.0), (5.0, PAD_DEPTH), (10.0, -1.0))

    result = optimize(polyline, fixed, pad, PARAMS, COARSE)

    assert clearances(polyline, pad).min() == pytest.approx(-PAD_DEPTH)
    assert clearances(result.polyline, pad).min() >= -INSIDE_TOLERANCE


def test_a_pinned_end_inside_a_clearance_stays() -> None:
    polyline, fixed = chain((0.0, 0.0), (5.0, 2.0), (10.0, 0.0))

    result = optimize(polyline, fixed, Disc(0.0, 0.0, 0.5), PARAMS, COARSE)

    np.testing.assert_array_equal(result.polyline[[0, -1]], polyline[[0, -1]])


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


DISC_CLIP = 0.12
"""Radius of two discs either side of the detour's pocket, which its legs
cut into by 20 µm, mm."""


# A run stopped at its cap can trade the input's violation for one at another
# obstacle (https://github.com/mairas/kibends/issues/14). The run converges
# in 5 iterations at these settings, so every cap here stops it unsettled.
@pytest.mark.parametrize("cap", [1, 2, 3, 4])
def test_a_clipped_detour_stopped_at_the_cap_adds_no_violation(cap: int) -> None:
    detour = scenarios.detour()
    assert isinstance(detour.field, Union)
    field = Union(
        [
            *detour.field.shapes,
            Disc(9.75, 1.2, DISC_CLIP),
            Disc(10.25, 1.2, DISC_CLIP),
        ]
    )

    result = optimize(
        detour.polyline,
        detour.fixed,
        field,
        detour.params,
        replace(COARSE, max_iterations=cap),
    )

    assert result.termination is Termination.ITERATION_CAP
    assert violations(result.polyline, result.fixed, field) <= violations(
        detour.polyline, detour.fixed, field
    )


GUARDED_DISC = Disc(5.0, -1.0, 0.5)
"""Clear of the chain the guard test starts with."""


@pytest.mark.parametrize(("inside", "taken"), [(False, True), (True, False)])
def test_the_guard_takes_no_more_violations_than_the_start(
    inside: bool, taken: bool
) -> None:
    polyline, fixed = chain((0.0, 0.0), (5.0, 2.0), (10.0, 0.0))
    alone = ChainInput(
        ChainCentreline("chain", "", 0.0, 0.0, polyline), "", fixed, GUARDED_DISC
    )
    (placed,) = optimize_module._problems(  # pyright: ignore[reportPrivateUsage]
        [alone], [], MovingVias([alone], [], 0.0, 0.0), PARAMS, COARSE
    )
    problem = placed.problem
    moved = problem.points[0].copy()
    if inside:
        # One free vertex in the disc: a run of one inside, no anchor in it.
        free = np.flatnonzero(~problem.masks[0])
        moved[free[len(free) // 2]] = (GUARDED_DISC.cx, GUARDED_DISC.cy)

    assert problem.allowed == [0]
    seen = problem.seeing([moved])
    assert problem.guard([moved], problem.masks, problem.positions, seen) is taken


NARROW = 0.05
"""How far each wall of the slot reaches past the middle, mm: no line
through it is clear."""


def test_a_chain_through_a_slot_narrower_than_its_clearance_adds_no_violation() -> None:
    # The walls reach far beyond the ends, so the chain cannot go around.
    field = Union(
        [
            Box(5.0, 50.0 - NARROW, 1.0, 50.0),
            Box(5.0, NARROW - 50.0, 1.0, 50.0),
        ]
    )
    polyline, fixed = chain((0.0, 0.0), (5.0, 0.5), (10.0, 0.0))

    result = optimize(polyline, fixed, field, PARAMS, COARSE)

    assert violations(result.polyline, result.fixed, field) <= violations(
        polyline, fixed, field
    )


SHALLOW_DEPTH = 0.001


def test_inside_run_holding_anchors_may_split() -> None:
    # Pins at both ends and one between them share one run of vertices 1 µm
    # inside the box; the standoff pushes the free ones out, leaving a
    # separate run at each pin.
    pinned_at = (6.0, 0.0)
    field = Box(5.0, SHALLOW_DEPTH - 5.0, 6.0, 5.0)
    polyline = path((0.0, 0.0), pinned_at, (10.0, 0.0))

    result = optimize(polyline, np.ones(3, dtype=bool), field, PARAMS, COARSE)

    assert np.all(result.polyline == pinned_at, axis=1).sum() == 1
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


def test_the_optimizer_refuses_more_coarse_spacings_than_it_holds() -> None:
    polyline, fixed = chain((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    too_many = tuple(0.2 * 2.0**k for k in reversed(range(MAX_COARSE_LEVELS + 1)))
    settings = replace(SETTINGS)
    # Past OptimizerSettings' own check, to reach the binding's.
    object.__setattr__(settings, "coarse_spacings", too_many)
    with pytest.raises(ValueError, match="coarse spacings"):
        optimize(polyline, fixed, FreeSpace(), PARAMS, settings)


def test_measure_reports_bend_and_length() -> None:
    angles = np.linspace(0.0, np.pi, 200)
    radius = 1.5
    arc = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)

    stats = measure(arc, spacing=0.05)

    assert stats.tightest_bend_radius == pytest.approx(radius, rel=1e-2)
    assert stats.length == pytest.approx(np.pi * radius, rel=1e-4)


def test_measure_gives_a_sharp_corner_a_small_radius() -> None:
    polyline = path((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))

    stats = measure(polyline, spacing=0.1)

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


BURST_LINE = re.compile(
    r"^DEBUG Burst of the chains on \S*: (\d+) L-BFGS iterations, "
    r"(\d+) evaluations, (\d+\.\d{3}) s in evaluations, "
    r'(\d+\.\d{3}) s in the rest of the step, ended with "([^"]+)"'
    r"(?:, at (\d+(?:\.\d+)?) mm spacing)?$"
)
LOGGED_BURSTS = 3


class BurstCost(NamedTuple):
    iterations: int
    evaluations: int
    evaluating: float
    rest: float
    ending: str
    coarse: float | None
    """The spacing of the coarse level it ran at, mm; None at the final
    spacing."""


def burst_costs(log: list[str]) -> list[BurstCost]:
    """Each burst's cost and ending, as logged."""
    lines = [line.strip() for line in log if "evaluations" in line]
    matches = [BURST_LINE.match(line) for line in lines]
    assert all(matches), lines
    return [
        BurstCost(
            int(m[1]),
            int(m[2]),
            float(m[3]),
            float(m[4]),
            m[5],
            None if m[6] is None else float(m[6]),
        )
        for m in matches
        if m
    ]


def test_every_burst_logs_what_it_cost_at_debug_and_nothing_at_info(
    log: list[str],
) -> None:
    settings = replace(COARSE, max_iterations=LOGGED_BURSTS)

    result = run(scenarios.wrap(), settings)

    costs = burst_costs(log)
    assert len(costs) == result.iterations == LOGGED_BURSTS
    for cost in costs:
        assert 1 <= cost.iterations <= settings.burst_maxiter
        assert cost.evaluations >= cost.iterations
    # Each figure is rounded to a millisecond; the evaluations of three
    # bursts together take more than one.
    assert sum(cost.evaluating for cost in costs) > 0
    info = [line for line in log if line.startswith("INFO")]
    assert info and not any("Burst" in line for line in info)


COARSE_LEVELS = (1.2, 0.6)
"""Coarse spacings above COARSE's final 0.3 mm."""
BOWED = ((0.0, 0.0), (5.0, 3.0), (10.0, 0.0))
"""A chain the probe at the final spacing does not settle."""


def stage_lines(log: list[str]) -> list[str]:
    """The lines that say a problem enters a level."""
    return [line.strip() for line in log if "coarsely" in line or "Refining" in line]


def on_two_layers(*vertices: tuple[float, float]) -> list[ChainInput]:
    """A chain on F.Cu and a copy on B.Cu: two problems."""
    return [
        scenarios.selected(layer, layer, FreeSpace(), *vertices, layer=layer)
        for layer in ("F.Cu", "B.Cu")
    ]


def test_each_problem_logs_each_level_it_enters_once_at_info(log: list[str]) -> None:
    settings = replace(COARSE, coarse_spacings=COARSE_LEVELS)

    result = optimize_chains(on_two_layers(*BOWED), PARAMS, settings)

    assert result.termination is Termination.CONVERGED
    for layer in ("F.Cu", "B.Cu"):
        assert [line for line in stage_lines(log) if f" on {layer} " in line] == [
            f"INFO Optimizing the chains on {layer} coarsely, at 1.2 mm spacing",
            f"INFO Optimizing the chains on {layer} coarsely, at 0.6 mm spacing",
            f"INFO Refining the chains on {layer} at 0.3 mm spacing",
        ]


@pytest.mark.parametrize(
    ("vertices", "coarse"),
    [(BOWED, ()), (((0.0, 0.0), (10.0, 0.0)), COARSE_LEVELS)],
    ids=["one spacing", "nothing to do"],
)
def test_a_run_without_levels_to_enter_logs_no_stage(
    log: list[str],
    vertices: tuple[tuple[float, float], ...],
    coarse: tuple[float, ...],
) -> None:
    optimize_chains(
        on_two_layers(*vertices), PARAMS, replace(COARSE, coarse_spacings=coarse)
    )

    assert stage_lines(log) == []


def test_a_burst_at_a_coarse_level_logs_its_spacing(log: list[str]) -> None:
    polyline, fixed = chain(*BOWED)
    settings = replace(COARSE, coarse_spacings=COARSE_LEVELS)

    optimize(polyline, fixed, FreeSpace(), PARAMS, settings)

    levels = [cost.coarse for cost in burst_costs(log)]
    # The probe at the final spacing, then each level in turn.
    assert levels[0] is None
    assert levels[-1] is None
    assert [c for c in dict.fromkeys(levels) if c is not None] == list(COARSE_LEVELS)


@pytest.mark.parametrize(
    ("cap", "before_final"),
    # Of BOWED's run: the probe, two bursts at 1.2 mm, one at 0.6 mm, then
    # the final spacing.
    [(2, True), (4, False)],
    ids=["at a coarse level", "at the final level"],
)
def test_a_capped_run_says_whether_it_stopped_before_the_final_spacing(
    cap: int, before_final: bool
) -> None:
    settings = replace(COARSE, coarse_spacings=COARSE_LEVELS, max_iterations=cap)

    result = optimize_chains(on_two_layers(*BOWED), PARAMS, settings)

    assert result.termination is Termination.ITERATION_CAP
    assert result.before_final_spacing is before_final


def test_bursts_that_all_fail_without_progress_end_the_run_abnormally() -> None:
    # Straight and off the walls' midline, the chain is as short as it can
    # be, so every step the misleading gradient takes, towards the nearer
    # wall, raises the energy, however a burst scales its variables.
    polyline, fixed = chain((0.0, 0.3), (5.0, 0.3), (10.0, 0.3))
    params = replace(PARAMS, repulsion=5.0)

    result = optimize(polyline, fixed, MisleadingWalls(1.0), params, ONE_SPACING)

    assert result.termination is Termination.ABNORMAL
    assert result.abnormal_bursts == MAX_ABNORMAL_BURSTS
    assert np.array_equal(result.polyline[[0, -1]], polyline[[0, -1]])


def test_bursts_that_all_fail_after_progress_end_the_run_given_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = optimize_module.run_burst

    def failing(
        objective: Objective,
        start: FloatArray,
        revert: Check,
        settings: OptimizerSettings,
        stop: Stop | None = None,
    ) -> Burst:
        return replace(real(objective, start, revert, settings, stop), abnormal=True)

    monkeypatch.setattr(optimize_module, "run_burst", failing)
    polyline, fixed = chain((0.0, 0.0), (5.0, 0.8), (10.0, 0.0))

    result = optimize(
        polyline, fixed, FreeSpace(), PARAMS, replace(COARSE, coarse_spacings=())
    )

    assert result.termination is Termination.GAVE_UP
    assert result.abnormal_bursts == MAX_ABNORMAL_BURSTS
    # The bursts before the streak ended were kept: the best state is shorter.
    assert result.after.length < result.before.length
    assert np.array_equal(result.polyline[[0, -1]], polyline[[0, -1]])


def test_a_burst_that_ends_abnormally_logs_its_evaluations_and_ending(
    log: list[str],
) -> None:
    # Near the upper wall its misleading gradient outweighs the length's, so
    # every trial of a burst's first line search raises the energy: each
    # burst fails there, from the same start, whatever the last bits.
    polyline, fixed = chain((0.0, 0.0), (5.0, 0.8), (10.0, 0.0))
    params = replace(PARAMS, repulsion=5.0)

    optimize(polyline, fixed, MisleadingWalls(1.0), params, ONE_SPACING)

    costs = burst_costs(log)
    assert len(costs) == MAX_ABNORMAL_BURSTS
    for cost in costs:
        assert cost.iterations == 0
        assert cost.evaluations > 0
        assert cost.ending.startswith("ABNORMAL")


def shortest_edge(points: FloatArray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).min())


# Starts at which two neighbouring vertices once collapsed onto each other in a
# burst, the gradient across the vanishing edge between them failed
# L-BFGS's line search, and the run ended abnormally. Starts 1 µm away from
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
    assert clearances(result.polyline, field).min() >= 0.0


class TestStalled:
    # Binary-exact values, so the boundary comparison is not a rounding test.
    SETTINGS = OptimizerSettings(stall_tolerance=0.5, stall_window=3)
    STILL = (3.0, 3.0, 3.0, 3.0)
    """Track lengths that do not change."""

    def stalled(
        self, lowest: list[float], lengths: tuple[float, ...], members: int, vias: int
    ) -> bool:
        return optimize_module.stalled(lowest, lengths, self.SETTINGS, members, vias)

    def test_needs_a_full_window_of_history(self) -> None:
        assert not self.stalled([5.0, 5.0, 5.0], self.STILL[:3], 1, 0)

    def test_improvement_of_exactly_the_tolerance_is_a_stall(self) -> None:
        assert self.stalled([5.0, 4.75, 4.625, 4.5], self.STILL, 1, 0)

    def test_larger_improvement_over_the_window_keeps_going(self) -> None:
        assert not self.stalled([5.0, 4.75, 4.625, 4.25], self.STILL, 1, 0)

    def test_each_member_adds_a_tolerance(self) -> None:
        # 1.0 over the window: two tolerances.
        lowest = [5.0, 4.5, 4.25, 4.0]
        assert not self.stalled(lowest, self.STILL, 1, 0)
        assert self.stalled(lowest, self.STILL, 2, 0)

    def test_each_moving_via_adds_a_tolerance(self) -> None:
        lowest = [5.0, 4.5, 4.25, 4.0]
        assert self.stalled(lowest, self.STILL, 1, 1)

    def test_tracks_shortening_by_more_than_a_tolerance_keep_going(self) -> None:
        lowest = [5.0, 4.5, 4.25, 4.0]
        # 0.75 over the window: more than one tolerance, however many members.
        shortening = (3.0, 2.75, 2.5, 2.25)
        assert not self.stalled(lowest, shortening, 1, 1)
        assert not self.stalled(lowest, shortening, 4, 0)

    @pytest.mark.parametrize(
        "lowest",
        [[5.0, 4.75, 4.625, 4.5], [5.0, 4.75, 4.625, 4.25], [5.0, 4.5, 4.25, 4.0]],
    )
    def test_two_far_apart_copies_stall_on_energy_as_one_does(
        self, lowest: list[float]
    ) -> None:
        # Far apart, two copies sum to twice one's energy.
        doubled = [2 * e for e in lowest]
        assert self.stalled(doubled, self.STILL, 2, 2) == self.stalled(
            lowest, self.STILL, 1, 1
        )

    @pytest.mark.parametrize(
        "lengths",
        [(3.0, 3.0, 3.0, 3.0), (3.0, 3.0, 2.75, 2.5), (3.0, 2.75, 2.5, 2.25)],
    )
    def test_two_far_apart_copies_never_stall_sooner_than_one(
        self, lengths: tuple[float, ...]
    ) -> None:
        lowest = [5.0, 4.5, 4.25, 4.0]
        doubled = [2 * e for e in lowest]
        twice = tuple(2 * length for length in lengths)
        one = self.stalled(lowest, lengths, 1, 1)
        assert one or not self.stalled(doubled, twice, 2, 2)

    def test_two_copies_run_on_while_their_summed_length_still_falls(self) -> None:
        lowest = [5.0, 4.5, 4.25, 4.0]
        # One tolerance of shortening for one copy, two for both.
        lengths = (3.0, 3.0, 2.75, 2.5)
        twice = tuple(2 * length for length in lengths)
        assert self.stalled(lowest, lengths, 1, 1)
        assert not self.stalled([2 * e for e in lowest], twice, 2, 2)


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
        points, fixed, ReachIgnored(field.core), params
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
    assert (
        clearances(result.polyline, scenario.field).mean()
        > clearances(scenario.polyline, scenario.field).mean()
    )
