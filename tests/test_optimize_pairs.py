import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pytest
import scenarios
from distance_fields import BoolArray, FloatArray, FreeSpace
from scenarios import (
    DP_GAP,
    DP_HALF_EXTENT,
    DP_PAD,
    PAIR_CLEARANCE,
    PAIR_WIDTH,
    PairsScenario,
    selected,
)
from shapely.geometry import LineString, Point

import kibends.optimize as optimize_module
from kibends.bursts import Burst, Objective
from kibends.energy import EnergyParams, ObstacleField
from kibends.optimize import (
    INSIDE_TOLERANCE,
    OptimizerSettings,
    PairResult,
    RunResult,
    Termination,
    _revert_new_crossings,  # pyright: ignore[reportPrivateUsage]
    close_to_itself,
    optimize_chains,
    resample,
)
from kibends.pair_blocks import PairBlock, PairInput, pair_block
from kibends.pairs import (
    BEND_RADIUS_TOLERANCE,
    GAP_TOLERANCE,
    bend_radius_limit,
    check_bend_radius,
    gap_range,
)

COARSE = OptimizerSettings(spacing=0.3, settled_change=0.003, burst_maxiter=30)

GAP_SLACK = GAP_TOLERANCE * DP_GAP
"""How far the tracks' gap may stray from the pair's, as pair detection
allows."""

DIRECTION_TOLERANCE = 1e-9
EXTENT_SLACK = 1e-9


def run(scenario: PairsScenario, settings: OptimizerSettings = COARSE) -> RunResult:
    return optimize_chains(
        scenario.chains, scenario.params, settings, pairs=scenario.pairs
    )


def length(points: FloatArray) -> float:
    return float(LineString(points).length)


def heading(points: FloatArray) -> FloatArray:
    """Unit direction of the first edge."""
    step: FloatArray = points[1] - points[0]
    return step / float(np.linalg.norm(step))


def assert_meets_its_ends(result: PairResult, given: PairInput) -> None:
    """The tracks start and end where the section's did, in its directions."""
    for track, original in ((result.a, given.a.points), (result.b, given.b.points)):
        np.testing.assert_array_equal(track[[0, -1]], original[[0, -1]])
        for points, before in ((track, original), (track[::-1], original[::-1])):
            np.testing.assert_allclose(
                heading(points), heading(before), atol=DIRECTION_TOLERANCE
            )


def assert_gap_kept(result: PairResult) -> None:
    low, high = gap_range(result.a, result.b, PAIR_WIDTH)
    assert DP_GAP - GAP_SLACK <= low <= high <= DP_GAP + GAP_SLACK


def assert_within_extent(
    result: PairResult, given: PairInput, settings: OptimizerSettings = COARSE
) -> None:
    """Both tracks' copper lies within the width the centreline keeps clear
    as, so that its clearance is theirs."""
    reach = pair_block(given, settings.spacing).chain.centreline.width / 2
    centreline = LineString(result.centreline)
    for track in (result.a, result.b):
        farthest = max(centreline.distance(Point(v)) for v in track)
        assert farthest + PAIR_WIDTH / 2 <= reach + EXTENT_SLACK


@pytest.fixture(scope="module")
def over_pad() -> RunResult:
    return run(scenarios.pair_over_pad())


class TestPairOverPad:
    def test_converges(self, over_pad: RunResult) -> None:
        assert over_pad.termination is Termination.CONVERGED

    def test_pulls_the_pair_down_onto_the_pad(self, over_pad: RunResult) -> None:
        (pair,) = over_pad.pairs
        assert length(pair.centreline) < pair.before.length - 1.0
        assert pair.centreline[:, 1].max() < 1.5

    def test_keeps_its_gap_along_the_whole_section(self, over_pad: RunResult) -> None:
        (pair,) = over_pad.pairs
        assert_gap_kept(pair)

    def test_both_tracks_keep_clear_of_the_pad(self, over_pad: RunResult) -> None:
        (pair,) = over_pad.pairs
        pad = Point(scenarios.PAIR_PAD)
        for track in (pair.a, pair.b):
            assert LineString(track).distance(pad) >= (
                DP_PAD + PAIR_CLEARANCE + PAIR_WIDTH / 2 - INSIDE_TOLERANCE
            )

    def test_meets_its_fan_outs_where_and_as_it_did(self, over_pad: RunResult) -> None:
        (given,) = scenarios.pair_over_pad().pairs
        (pair,) = over_pad.pairs
        assert_meets_its_ends(pair, given)

    def test_tracks_lie_within_the_centreline_s_extent(
        self, over_pad: RunResult
    ) -> None:
        (given,) = scenarios.pair_over_pad().pairs
        (pair,) = over_pad.pairs
        assert_within_extent(pair, given)

    def test_tracks_lie_half_a_pitch_off_the_centreline(
        self, over_pad: RunResult
    ) -> None:
        (pair,) = over_pad.pairs
        half_pitch = (DP_GAP + PAIR_WIDTH) / 2
        for track in (pair.a, pair.b):
            spread = gap_range(track, pair.centreline, PAIR_WIDTH)
            assert spread == pytest.approx(
                (half_pitch - PAIR_WIDTH,) * 2, abs=GAP_SLACK
            )


@pytest.fixture(scope="module")
def tent() -> RunResult:
    return run(scenarios.pair_tent())


class TestPairBesideAChain:
    def test_both_tracks_keep_clear_of_the_chain(self, tent: RunResult) -> None:
        (pair,) = tent.pairs
        (bar,) = tent.chains
        for track in (pair.a, pair.b):
            assert LineString(track).distance(LineString(bar.polyline)) >= (
                PAIR_WIDTH + PAIR_CLEARANCE - INSIDE_TOLERANCE
            )

    def test_the_pair_rests_on_the_chain(self, tent: RunResult) -> None:
        (pair,) = tent.pairs
        (bar,) = tent.chains
        inner = min(
            LineString(track).distance(LineString(bar.polyline))
            for track in (pair.a, pair.b)
        )
        (given,) = scenarios.pair_tent().pairs
        extent = pair_block(given, COARSE.spacing).chain.centreline.width
        widening = (extent - 2 * PAIR_WIDTH - DP_GAP) / 2
        assert inner < PAIR_WIDTH + PAIR_CLEARANCE + widening + 0.05

    def test_keeps_its_gap_and_ends(self, tent: RunResult) -> None:
        (given,) = scenarios.pair_tent().pairs
        (pair,) = tent.pairs
        assert_gap_kept(pair)
        assert_meets_its_ends(pair, given)
        assert_within_extent(pair, given)

    def test_is_deterministic(self, tent: RunResult) -> None:
        again = run(scenarios.pair_tent())
        (pair,), (repeat,) = tent.pairs, again.pairs
        for first, second in (
            (pair.centreline, repeat.centreline),
            (pair.a, repeat.a),
            (pair.b, repeat.b),
            (tent.chains[0].polyline, again.chains[0].polyline),
        ):
            np.testing.assert_array_equal(first, second)
        assert again.iterations == tent.iterations


UNHINGED_ITERATIONS = 2
CORNER_SHORTCUT = 0.05
"""mm: a centreline rounding the corner at the limit is at least this much
shorter."""
CORNER_SETTINGS = OptimizerSettings()
"""The defaults: fine enough that a soft centreline pulled taut against its
direction pins bends tighter than the limit."""


@pytest.fixture(scope="module")
def corner() -> RunResult:
    return run(scenarios.pair_corner(), CORNER_SETTINGS)


class TestTightCorner:
    def test_keeps_the_bend_radius_limit(self, corner: RunResult) -> None:
        (pair,) = corner.pairs
        assert not check_bend_radius(pair.centreline, DP_GAP, PAIR_WIDTH).too_tight
        assert pair.bend is not None and not pair.bend.too_tight

    def test_bends_close_to_the_limit(self, corner: RunResult) -> None:
        (pair,) = corner.pairs
        radius = check_bend_radius(pair.centreline, DP_GAP, PAIR_WIDTH).radius
        assert radius < 1.5 * bend_radius_limit(DP_GAP, PAIR_WIDTH)

    def test_folds_without_the_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def unlimited(given: PairInput, spacing: float) -> PairBlock:
            return replace(pair_block(given, spacing), limit=0.0)

        monkeypatch.setattr(optimize_module, "pair_block", unlimited)
        (pair,) = run(scenarios.pair_corner(), CORNER_SETTINGS).pairs
        assert check_bend_radius(pair.centreline, DP_GAP, PAIR_WIDTH).too_tight
        assert pair.bend is not None and pair.bend.too_tight

    def test_bursts_reject_bends_below_the_limit_without_its_hinge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = optimize_module.free_vertex_objective

        def unhinged(
            points: FloatArray,
            fixed: BoolArray,
            field: ObstacleField,
            params: EnergyParams,
            short_edge: float = 0.0,
            bend_limit: float | None = None,
        ) -> Objective:
            return real(points, fixed, field, params, short_edge)

        monkeypatch.setattr(optimize_module, "free_vertex_objective", unhinged)
        # Pressing on the limit, every burst has edges to reject, so the
        # section never settles: a few iterations show it.
        settings = replace(CORNER_SETTINGS, max_iterations=UNHINGED_ITERATIONS)
        (pair,) = run(scenarios.pair_corner(), settings).pairs
        assert length(pair.centreline) < pair.before.length - CORNER_SHORTCUT
        assert pair.bend is not None and not pair.bend.too_tight

    def test_inner_track_runs_the_centreline_s_way(self, corner: RunResult) -> None:
        (pair,) = corner.pairs
        along = np.diff(pair.centreline, axis=0)
        for track in (pair.a, pair.b):
            assert np.all(np.einsum("ij,ij->i", np.diff(track, axis=0), along) > 0)

    def test_keeps_its_gap(self, corner: RunResult) -> None:
        (pair,) = corner.pairs
        assert_gap_kept(pair)

    def test_outer_track_lies_within_the_centreline_s_extent(
        self, corner: RunResult
    ) -> None:
        (given,) = scenarios.pair_corner().pairs
        (pair,) = corner.pairs
        assert_within_extent(pair, given, CORNER_SETTINGS)


def test_an_end_at_another_gap_keeps_both_tracks_directions() -> None:
    (given,) = scenarios.pair_uneven_end().pairs

    (pair,) = run(scenarios.pair_uneven_end()).pairs

    assert_meets_its_ends(pair, given)
    assert_gap_kept(pair)


def test_a_short_chamfer_at_an_end_gets_no_sliver() -> None:
    (given,) = scenarios.pair_chamfered().pairs

    block = pair_block(given, COARSE.spacing)

    points, fixed = block.chain.centreline.points, block.chain.fixed
    assert len(points) == len(given.a.points)
    assert fixed[[0, 1, -2, -1]].all()
    (pair,) = run(scenarios.pair_chamfered()).pairs
    assert_meets_its_ends(pair, given)


@pytest.fixture(scope="module")
def split() -> RunResult:
    return run(scenarios.pair_split())


class TestTwoCoupledSections:
    def test_split_gives_two_sections_and_a_detour_on_each_track(self) -> None:
        scenario = scenarios.pair_split()
        assert len(scenario.pairs) == 2
        assert [c.centreline.net for c in scenario.chains] == ["P", "N"]

    def test_optimizes_each_section(self, split: RunResult) -> None:
        for pair in split.pairs:
            assert length(pair.centreline) < pair.before.length - 0.05
            assert_gap_kept(pair)

    def test_sections_keep_their_ends_and_directions(self, split: RunResult) -> None:
        for given, pair in zip(scenarios.pair_split().pairs, split.pairs, strict=True):
            assert_meets_its_ends(pair, given)

    def test_detours_still_join_the_sections(self, split: RunResult) -> None:
        first, second = split.pairs
        p, n = (c.polyline for c in split.chains)
        np.testing.assert_array_equal(p[[0, -1]], [first.a[-1], second.a[0]])
        np.testing.assert_array_equal(n[[0, -1]], [first.b[-1], second.b[0]])

    def test_detours_keep_clear_of_the_pad(self, split: RunResult) -> None:
        pad = Point(scenarios.SPLIT_PAD)
        for chain in split.chains:
            assert LineString(chain.polyline).distance(pad) >= (
                DP_PAD + PAIR_CLEARANCE + PAIR_WIDTH / 2 - INSIDE_TOLERANCE
            )


def test_a_section_pinned_inside_stays_as_it_is() -> None:
    (given,) = scenarios.pair_over_pad().pairs
    pinned = given.b_pinned.copy()
    pinned[2] = True
    held = replace(given, b_pinned=pinned)

    result = optimize_chains((), scenarios.PARAMS, COARSE, pairs=(held,))

    assert result.termination is Termination.NOTHING_TO_DO
    (pair,) = result.pairs
    np.testing.assert_array_equal(pair.a, given.a.points)
    np.testing.assert_array_equal(pair.b, given.b.points)
    assert pair.block.held
    assert pair.bend is None


SHORT_CHAMFER = 0.1
"""Of a 90° corner taken in two 45° turns: a bend radius well below the
limit."""


def test_a_section_routed_too_tightly_stays_as_it_is() -> None:
    corner = 2.0
    given = scenarios.coupled(
        FreeSpace(),
        (0.0, 0.0),
        (corner, 0.0),
        (corner + SHORT_CHAMFER, SHORT_CHAMFER),
        (corner + SHORT_CHAMFER, corner),
    )

    result = optimize_chains((), scenarios.PARAMS, COARSE, pairs=(given,))

    assert result.termination is Termination.NOTHING_TO_DO
    (pair,) = result.pairs
    np.testing.assert_array_equal(pair.a, given.a.points)
    np.testing.assert_array_equal(pair.b, given.b.points)
    assert pair.block.held
    assert pair.bend is not None and pair.bend.too_tight


def test_a_section_inside_a_chain_s_clearance_throughout_stays_put() -> None:
    section = scenarios.coupled(FreeSpace(), (0.0, 0.0), (5.0, 0.0))
    half_extent = pair_block(section, COARSE.spacing).chain.centreline.width / 2
    # Every vertex resampling adds to the centreline lies deeper than δ
    # inside the chain's clearance, so every bound of its bursts is zero.
    y = half_extent + PAIR_CLEARANCE + PAIR_WIDTH / 2 - 2 * COARSE.bubble_tolerance
    chain = selected("C", "C", FreeSpace(), (0.0, y), (5.0, y))

    result = optimize_chains([chain], scenarios.PARAMS, COARSE, pairs=(section,))

    (pair,) = result.pairs
    assert LineString(pair.a).equals(LineString(section.a.points))
    assert LineString(pair.b).equals(LineString(section.b.points))
    assert pair.abnormal_bursts == 0


SPLIT_APART = 10.0
"""mm between two sections that do not see each other's fields."""
SHARP_LEG = 1.0
"""mm: each leg of a 135° corner, which resampling at the coarse spacing
cuts into a bend tighter than the limit."""


def sharp_corner(field: ObstacleField, below: float = 0.0) -> PairInput:
    """A section turning 135° at (`SHARP_LEG`, `below`)."""
    back = SHARP_LEG / math.sqrt(2)
    return scenarios.coupled(
        field, (0.0, below), (SHARP_LEG, below), (SHARP_LEG - back, below + back)
    )


def test_a_section_kept_tight_does_not_hold_back_one_that_can_round_its_corner() -> (
    None
):
    inside = np.array([-1.0 - 1 / math.sqrt(2), 1 / math.sqrt(2)])
    inside /= np.linalg.norm(inside)
    pad = np.array([SHARP_LEG, 0.0]) + inside * (
        DP_PAD + PAIR_CLEARANCE + DP_HALF_EXTENT
    )
    stuck = sharp_corner(scenarios.pad_for_pair(*pad))
    free = sharp_corner(FreeSpace(), below=SPLIT_APART)

    result = optimize_chains((), scenarios.PARAMS, COARSE, pairs=(stuck, free))

    kept, rounded = result.pairs
    block = pair_block(stuck, COARSE.spacing)
    start, _ = resample(
        block.chain.centreline.points, block.chain.fixed, COARSE.spacing
    )
    tightest = block.bend(start).radius
    assert kept.bend is not None and kept.bend.too_tight
    assert kept.bend.radius >= tightest * (1.0 - BEND_RADIUS_TOLERANCE)
    assert rounded.bend is not None and not rounded.bend.too_tight


PAIR_COST = 1.5
"""A pair may query its field for this many times as many points as one
chain along its centreline. Measured: 1.03, and 0.93 at the defaults. Wall
time, which a busy machine makes too noisy to test, measured 1.1 times a
chain's at these settings and 0.95 at the defaults."""


@dataclass
class Counting:
    """A field that counts the points it is asked about."""

    field: ObstacleField
    points: int = 0

    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        self.points += len(points)
        return self.field.signed_distance(points, reach)

    def segments_clear(
        self, starts: FloatArray, ends: FloatArray, tolerance: float = 0.0
    ) -> BoolArray:
        self.points += 2 * len(starts)
        return self.field.segments_clear(starts, ends, tolerance)


def test_a_pair_costs_about_one_chain() -> None:
    pair, chain = scenarios.pair_over_pad(), scenarios.centreline_over_pad()
    (section,), (centreline,) = pair.pairs, chain.chains
    pair_field, chain_field = Counting(section.field), Counting(centreline.field)

    run(replace(pair, pairs=(replace(section, field=pair_field),)))
    optimize_chains([replace(centreline, field=chain_field)], chain.params, COARSE)

    assert pair_field.points < PAIR_COST * chain_field.points


APART = 0.7
"""mm a centreline keeps its own parts beyond a turn, as a pair's does."""
ARC_STEPS = 64


def close_pairs(points: FloatArray) -> set[tuple[int, int]]:
    pairs, _ = close_to_itself(points, APART)
    return {(int(i), int(j)) for i, j in pairs}


def hairpin(legs: float) -> FloatArray:
    """Two 3 mm legs `legs` apart along x < 0, 0.1 mm between vertices,
    joined by a half circle; the first leg runs at y = legs / 2."""
    angles = np.linspace(-math.pi / 2, math.pi / 2, ARC_STEPS)
    turn = (legs / 2) * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    back = np.stack([np.linspace(0.0, -3.0, 31), np.full(31, -legs / 2)], axis=1)
    forth = np.stack([np.linspace(-3.0, 0.0, 31), np.full(31, legs / 2)], axis=1)
    return np.concatenate([forth[:-1], turn[::-1][:-1], back])


def test_a_straight_centreline_is_not_close_to_itself() -> None:
    line = np.stack([np.linspace(0.0, 10.0, 101), np.zeros(101)], axis=1)

    assert close_pairs(line) == set()


def test_a_turn_as_wide_as_the_parts_keep_apart_is_not_close_to_itself() -> None:
    angles = np.linspace(0.0, math.pi, ARC_STEPS)
    half = APART / 2 + EXTENT_SLACK
    turn = half * np.stack([np.cos(angles), np.sin(angles)], axis=1)

    assert close_pairs(turn) == set()


def test_legs_beyond_a_tight_turn_are_close_to_each_other() -> None:
    points = hairpin(APART / 2)

    assert (0, len(points) - 1) in close_pairs(points)


def at(points: FloatArray, x: float, y: float) -> int:
    return int(np.argmin(np.hypot(points[:, 0] - x, points[:, 1] - y)))


LEGS = 0.6
"""mm between a hairpin's legs, less than they keep: closer than that they
may not come."""


def revert(candidate: FloatArray, original: FloatArray) -> tuple[BoolArray, int]:
    free = np.ones(len(original), dtype=bool)
    free[[0, -1]] = False
    clear = np.ones(len(original) - 1, dtype=bool)
    reverted, rejected = _revert_new_crossings(
        candidate, original, free, clear, FreeSpace(), apart=APART
    )
    assert rejected.total == rejected.too_close
    return reverted, rejected.too_close


def test_two_vertices_closer_than_they_started_both_move_back() -> None:
    original = hairpin(LEGS)
    top, bottom = at(original, -2.0, LEGS / 2), at(original, -2.0, -LEGS / 2)
    elsewhere = at(original, -0.5, -LEGS / 2)
    candidate = original.copy()
    candidate[top, 1] -= 0.05
    candidate[bottom, 1] += 0.05
    candidate[elsewhere, 0] -= 0.02

    reverted, too_close = revert(candidate, original)

    assert reverted[top] and reverted[bottom] and not reverted[elsewhere]
    assert (candidate[[top, bottom]] == original[[top, bottom]]).all()
    assert candidate[elsewhere, 0] == original[elsewhere, 0] - 0.02
    assert too_close > 0


@pytest.mark.parametrize("backwards", [False, True])
def test_a_vertex_too_close_to_one_that_stayed_moves_back_alone(
    backwards: bool,
) -> None:
    original = hairpin(LEGS)
    if backwards:
        original = original[::-1].copy()
    kink = at(original, -2.0, LEGS / 2)
    original[kink, 1] = 0.0
    moved = at(original, -2.0, -LEGS / 2)
    elsewhere = at(original, -0.5, -LEGS / 2)
    candidate = original.copy()
    # Still farther from the kink than it started, but closer to the other
    # leg's vertex at x = -2.3, which stays, than either started from the
    # other leg.
    candidate[moved] = (-2.3, -0.15)
    candidate[elsewhere, 0] -= 0.02

    reverted, too_close = revert(candidate, original)

    assert reverted[moved] and not reverted[elsewhere] and too_close > 0
    assert (candidate[moved] == original[moved]).all()


REJECTED = 5
STILL_ITERATIONS = 3


@pytest.mark.parametrize(
    ("too_close", "settles"), [(REJECTED, True), (REJECTED - 1, False)]
)
def test_a_section_settles_on_rejections_only_when_they_are_its_own_parts(
    monkeypatch: pytest.MonkeyPatch, too_close: int, settles: bool
) -> None:
    def still(points: FloatArray, *_: object) -> object:
        return optimize_module._ChainBurst(  # pyright: ignore[reportPrivateUsage]
            points.copy(), REJECTED, False, too_close
        )

    monkeypatch.setattr(optimize_module, "_burst", still)
    settings = replace(COARSE, max_iterations=STILL_ITERATIONS)

    result = run(scenarios.pair_corner(), settings)

    if settles:
        assert result.termination is Termination.NOTHING_TO_DO
    else:
        assert result.termination is Termination.ITERATION_CAP


@pytest.mark.parametrize(
    ("abnormal", "counted"), [(False, 2 * REJECTED), (True, REJECTED)]
)
def test_only_a_normally_ended_burst_counts_its_parts_too_close(
    monkeypatch: pytest.MonkeyPatch, abnormal: bool, counted: int
) -> None:
    def rejecting(
        candidate: FloatArray, *_: object
    ) -> tuple[BoolArray, optimize_module._Rejected]:  # pyright: ignore[reportPrivateUsage]
        reverted = np.zeros(len(candidate), dtype=bool)
        return reverted, optimize_module._Rejected(REJECTED, REJECTED)  # pyright: ignore[reportPrivateUsage]

    def one_check(
        _objective: Objective,
        start: FloatArray,
        _half_sides: FloatArray,
        revert: Callable[[FloatArray], tuple[BoolArray, int]],
        _settings: OptimizerSettings,
    ) -> Burst:
        _, first = revert(start.copy())
        _, last = revert(start.copy())
        return Burst(start, first + (0 if abnormal else last), abnormal)

    monkeypatch.setattr(optimize_module, "_revert_new_crossings", rejecting)
    monkeypatch.setattr(optimize_module, "bounded_burst", one_check)
    points = hairpin(LEGS)
    fixed = np.zeros(len(points), dtype=bool)
    fixed[[0, -1]] = True

    burst = optimize_module._burst(  # pyright: ignore[reportPrivateUsage]
        points, fixed, FreeSpace(), scenarios.PARAMS, COARSE, apart=APART
    )

    assert burst.too_close == counted


TILT = 0.3
"""rad by which a whole hairpin turns."""


def test_a_hairpin_turned_whole_keeps_its_parts_as_far_apart() -> None:
    original = hairpin(LEGS)
    c, s = math.cos(TILT), math.sin(TILT)
    candidate = original @ np.array([[c, s], [-s, c]])

    _, too_close = revert(candidate, original)

    assert too_close == 0
