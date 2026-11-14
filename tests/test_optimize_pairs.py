"""Coupled sections of differential pairs through
`optimize.optimize_chains`.

The problem's energy is checked against a reference built member by member
(`reference_energy`), and the offset pullback against finite
differences."""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pytest
import scenarios
from conftest import (
    BurstClock,
    all_abnormal_bursts,
    first_check,
    gap_range,
    layer_energy,
    total_energy,
)
from distance_fields import BoolArray, FloatArray, FreeSpace
from reference_fields import chain_field, neighbour_field
from scenarios import (
    DP_GAP,
    DP_PAD,
    PAIR_CLEARANCE,
    PAIR_WIDTH,
    PairsScenario,
    selected,
)
from shapely.geometry import LineString, Point

import kibends.members as members_module
import kibends.optimize as optimize_module
from kibends import _core
from kibends.bursts import (
    DEFAULT_CHORD_TOLERANCE,
    FALLBACK_WARNING,
    INSIDE_TOLERANCE,
    Burst,
    Check,
    Objective,
    OptimizerSettings,
    Stop,
    run_burst,
)
from kibends.chain import Chain
from kibends.contacts import no_contacts
from kibends.energy import EnergyParams, ObstacleField, arc_lengths
from kibends.members import Member, seen_parts
from kibends.neighbours import ChainInput
from kibends.obstacles import ChainCentreline
from kibends.optimize import MAX_ABNORMAL_BURSTS, optimize_chains
from kibends.pair_blocks import Hold, PairBlock, PairInput, pair_block
from kibends.pair_layout import cut_parts, find_pairs, lay_out
from kibends.pairs import (
    GAP_TOLERANCE,
    bend_radius_limit,
    check_bend_radius,
    offset_pair,
)
from kibends.polylines import SelfCheck, resample
from kibends.results import ChainResult, PairResult, RunResult, Termination
from kibends.simplify import simplify_chains
from kibends.vias import ViaInput

COARSE = OptimizerSettings(spacing=0.3, settled_change=0.003, burst_maxiter=30)
ONE_SPACING = replace(COARSE, coarse_spacings=())
"""COARSE at its final spacing alone, for a premise that counts bursts at one
spacing or compares with the inputs resampled at it."""
CORNER = OptimizerSettings(burst_maxiter=30)
"""The default spacing, at which a centreline pulled taut against its
direction pins bends tighter than the limit, with short bursts."""

GAP_SLACK = GAP_TOLERANCE * DP_GAP
DIRECTION_TOLERANCE = 1e-9
EXTENT_SLACK = 1e-9


def run(scenario: PairsScenario, settings: OptimizerSettings = COARSE) -> RunResult:
    return optimize_chains(
        scenario.chains, scenario.params, settings, pairs=scenario.pairs
    )


def length(points: FloatArray) -> float:
    return float(LineString(points).length)


def heading(points: FloatArray) -> FloatArray:
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
    reach = pair_block(given, settings.spacing).chain.centreline.width / 2
    centreline = LineString(result.centreline)
    for track in (result.a, result.b):
        farthest = max(centreline.distance(Point(v)) for v in track)
        assert farthest + PAIR_WIDTH / 2 <= reach + EXTENT_SLACK


def shown(member: Member, points: FloatArray) -> list[tuple[ChainInput, FloatArray]]:
    """What the others see of a chain or a section."""
    if isinstance(member, PairBlock):
        return list(zip(member.tracks, member.offset(points), strict=True))
    return [(member, points)]


def reference_energy(
    members: Sequence[Member],
    polylines: Sequence[FloatArray],
    masks: Sequence[BoolArray],
    params: EnergyParams,
    settings: OptimizerSettings = COARSE,
) -> float:
    """The energy at `polylines`, built member by member: each chain and
    section in its static field plus the others as `neighbour_field` shows
    them, a section with its bend-radius hinge."""
    value = 0.0
    for i, (member, points, mask) in enumerate(
        zip(members, polylines, masks, strict=True)
    ):
        others = [
            part
            for j, other in enumerate(members)
            if j != i
            for part in shown(other, polylines[j])
        ]
        if isinstance(member, PairBlock):
            near = neighbour_field(member.facing, others, ignored=member.nets)
            static = member.chain.field
            hinge: tuple[float, float] | None = (member.limit, member.spacing)
        else:
            near = neighbour_field(member, others)
            static, hinge = member.field, None
        field: ObstacleField = chain_field(static, near)
        value += total_energy(points, mask, field, params, settings.short_edge, hinge)[
            0
        ]
    return value


STEP = 1e-7
WOBBLE = 0.05
"""mm by which free centreline vertices are moved off their resampled
places, so that every vertex turns."""


def wobbled(block: PairBlock, seed: int) -> tuple[FloatArray, BoolArray]:
    points, fixed = resample(
        block.chain.centreline.points, block.chain.fixed, COARSE.spacing
    )
    rng = np.random.default_rng(seed)
    points = points.copy()
    points[~fixed] += rng.uniform(-WOBBLE, WOBBLE, (int((~fixed).sum()), 2))
    return points, fixed


@pytest.mark.parametrize(
    "scenario",
    [scenarios.pair_uneven_end(), scenarios.pair_corner()],
    ids=lambda s: s.name,
)
def test_offset_pullback_matches_finite_differences(
    scenario: PairsScenario,
) -> None:
    (given,) = scenario.pairs
    block = pair_block(given, COARSE.spacing)
    points, fixed = wobbled(block, 1)
    rng = np.random.default_rng(2)
    at_a, at_b = rng.normal(size=points.shape), rng.normal(size=points.shape)

    def value(centreline: FloatArray) -> float:
        a, b = block.offset(centreline)
        return float(np.sum(at_a * a) + np.sum(at_b * b))

    analytic = block.offset_pullback(points, at_a, at_b)

    # The tracks' end points are the section's own, which no free vertex
    # moves.
    free = np.flatnonzero(~fixed)
    numeric = np.zeros_like(points)
    for k in free:
        for axis in (0, 1):
            up, down = points.copy(), points.copy()
            up[k, axis] += STEP
            down[k, axis] -= STEP
            numeric[k, axis] = (value(up) - value(down)) / (2 * STEP)
    np.testing.assert_allclose(analytic[free], numeric[free], rtol=1e-5, atol=1e-6)


class TestEnergy:
    def params(self) -> EnergyParams:
        return replace(scenarios.PARAMS, repulsion=1.0, falloff_range=0.5)

    def state(
        self, scenario: PairsScenario
    ) -> tuple[list[Member], list[FloatArray], list[BoolArray]]:
        """The scenario's chains and sections, the sections resampled and
        wobbled so that they bend everywhere."""
        blocks = [pair_block(p, COARSE.spacing) for p in scenario.pairs]
        wobbles = [wobbled(block, k) for k, block in enumerate(blocks)]
        return (
            [*scenario.chains, *blocks],
            [
                *(c.centreline.points.copy() for c in scenario.chains),
                *(points for points, _ in wobbles),
            ],
            [*(c.fixed for c in scenario.chains), *(fixed for _, fixed in wobbles)],
        )

    @pytest.mark.parametrize(
        "scenario",
        [scenarios.pair_tent(), scenarios.pair_split(), scenarios.pair_corner()],
        ids=lambda s: s.name,
    )
    def test_is_each_members_energy_against_the_others(
        self, scenario: PairsScenario
    ) -> None:
        # pair_split's detours touch its sections' ends: they are their own
        # nets' fan-outs, which the sections ignore and which ignore them.
        members, polylines, masks = self.state(scenario)

        value, _ = layer_energy(members, polylines, masks, self.params(), 0.01)

        expected = reference_energy(
            members, polylines, masks, self.params(), replace(COARSE, spacing=0.1)
        )
        assert value == pytest.approx(expected, rel=1e-12)

    def test_gradient_reaches_through_the_tracks(self) -> None:
        # The bar lies under the tent's tracks: its repulsion of them moves
        # the centreline, and the centreline's of it moves the bar.
        bar = selected("bar", "B", FreeSpace(), (4.0, 2.2), (5.0, 2.4), (6.0, 2.2))
        scenario = replace(scenarios.pair_tent(), chains=(bar,))
        members, polylines, masks = self.state(scenario)
        params = self.params()
        seen = seen_parts(members, polylines)
        _, gradients = layer_energy(members, polylines, masks, params, 0.01, seen)
        apart = layer_energy(members[1:], polylines[1:], masks[1:], params, 0.01)[1]
        assert not np.allclose(apart[0], gradients[1])

        for i in range(len(members)):
            for k in np.flatnonzero(~masks[i]):
                for axis in (0, 1):
                    shifted = [p.copy() for p in polylines]
                    shifted[i][k, axis] += STEP
                    up = layer_energy(members, shifted, masks, params, 0.01, seen)[0]
                    shifted[i][k, axis] -= 2 * STEP
                    down = layer_energy(members, shifted, masks, params, 0.01, seen)[0]
                    numeric = (up - down) / (2 * STEP)
                    assert gradients[i][k, axis] == pytest.approx(
                        numeric, rel=1e-4, abs=1e-6
                    )

    @pytest.mark.parametrize("net", ["P", "N"])
    def test_a_chain_of_one_of_the_diff_pair_s_nets_and_the_section_add_nothing(
        self, net: str
    ) -> None:
        # The bar lies under the tent's tracks, where a bar of another net
        # repels them (`test_gradient_reaches_through_the_tracks`).
        bar = selected("bar", net, FreeSpace(), (4.0, 2.2), (5.0, 2.4), (6.0, 2.2))
        scenario = replace(scenarios.pair_tent(), chains=(bar,))
        members, polylines, masks = self.state(scenario)
        params = self.params()

        value, gradients = layer_energy(members, polylines, masks, params, 0.01)

        for k in range(len(members)):
            alone, alone_gradient = layer_energy(
                [members[k]], [polylines[k]], [masks[k]], params, 0.01
            )
            value -= alone
            np.testing.assert_allclose(gradients[k], alone_gradient[0], atol=1e-12)
        assert value == pytest.approx(0.0, abs=1e-12)


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
        assert_within_extent(pair, given)

    def test_simplified_tracks_stay_parallel_segment_for_segment(
        self, over_pad: RunResult
    ) -> None:
        # As run.py simplifies a section.
        (pair,) = over_pad.pairs
        block = pair.block
        result = ChainResult(
            pair.centreline,
            pair.fixed,
            0,
            pair.before,
            no_contacts(pair.centreline),
        )

        (centreline,) = simplify_chains(
            [block.facing],
            replace(over_pad, chains=(result,), pairs=()),
            scenarios.DP_CHORD_TOLERANCE,
            sections={0: block},
        )
        a, b = block.offset(centreline)

        assert 2 < len(centreline) < len(pair.centreline)
        along_a, along_b = (np.diff(t, axis=0) for t in (a, b))
        cosine = np.einsum("ij,ij->i", along_a, along_b) / (
            np.linalg.norm(along_a, axis=1) * np.linalg.norm(along_b, axis=1)
        )
        assert (1.0 - cosine <= PARALLEL).all()


PARALLEL = 5e-13
"""The most, as 1 - cos of the angle, a segment of one track may turn from
the other's, as test_simplify allows."""


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
        assert length(bar.polyline) < bar.before.length

    def test_keeps_its_gap_and_ends(self, tent: RunResult) -> None:
        (given,) = scenarios.pair_tent().pairs
        (pair,) = tent.pairs
        assert_gap_kept(pair)
        assert_meets_its_ends(pair, given)
        assert_within_extent(pair, given)


FOLD_STEP = 0.2
"""mm from a vertex to the next where `folded` turns back."""


def folded(points: FloatArray, start: int) -> FloatArray:
    """`points` with the two vertices after `start` moved onto the line
    through it parallel to x, the first ahead of it and the second back
    halfway: an exact 180° fold at the first."""
    out = points.copy()
    out[start + 1] = out[start] + (FOLD_STEP, 0.0)
    out[start + 2] = out[start] + (FOLD_STEP / 2, 0.0)
    return out


@pytest.fixture(scope="module")
def corner() -> RunResult:
    return run(scenarios.pair_corner(), CORNER)


class TestTightCorner:
    def test_keeps_the_bend_radius_limit(self, corner: RunResult) -> None:
        (pair,) = corner.pairs
        assert not check_bend_radius(pair.centreline, DP_GAP, PAIR_WIDTH).too_tight
        assert pair.bend is not None and not pair.bend.too_tight
        radius = check_bend_radius(pair.centreline, DP_GAP, PAIR_WIDTH).radius
        assert radius < 1.5 * bend_radius_limit(DP_GAP, PAIR_WIDTH)

    def test_folds_without_the_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def unlimited(given: PairInput, spacing: float) -> PairBlock:
            return replace(pair_block(given, spacing), limit=0.0)

        monkeypatch.setattr(optimize_module, "pair_block", unlimited)
        (pair,) = run(scenarios.pair_corner(), CORNER).pairs
        assert pair.bend is not None and pair.bend.too_tight

    def test_bursts_reject_bends_below_the_limit_without_its_hinge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = members_module.core_member

        def unhinged(member: Member) -> _core.Member:
            return real(member, hinged=False)

        monkeypatch.setattr(members_module, "core_member", unhinged)
        # Routed wider than the limit: length alone would pull it into the
        # corner, below the limit, and only the self check holds it back.
        (pair,) = run(scenarios.pair_round_corner(), CORNER).pairs
        assert length(pair.centreline) <= pair.before.length
        assert pair.bend is not None and not pair.bend.too_tight
        assert pair.rejected_edges > 0

    def test_energy_at_a_fold_is_finite(self) -> None:
        (given,) = scenarios.pair_corner().pairs
        block = pair_block(given, COARSE.spacing)
        points, fixed = resample(
            block.chain.centreline.points, block.chain.fixed, COARSE.spacing
        )
        first = int(np.flatnonzero(~fixed)[0])

        value, (gradient,) = layer_energy(
            [block],
            [folded(points, first)],
            [fixed],
            scenarios.PARAMS,
            COARSE.short_edge,
        )

        assert math.isfinite(value)
        assert np.isfinite(gradient).all()

    def test_a_trial_step_that_folds_the_centreline_ends_no_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # L-BFGS's first trial of the first burst folds the centreline
        # at its second free vertex.
        trials: list[FloatArray] = []

        def folding_first(
            objective: Objective,
            start: FloatArray,
            revert: Check,
            settings: OptimizerSettings,
            stop: Stop | None = None,
        ) -> Burst:
            if not trials:
                trials.append(folded(start.reshape(-1, 2), 0).ravel())
                objective(trials[0])
            return run_burst(objective, start, revert, settings, stop)

        monkeypatch.setattr(optimize_module, "run_burst", folding_first)
        (pair,) = run(scenarios.pair_corner(), CORNER).pairs

        assert trials
        assert pair.bend is not None and not pair.bend.too_tight

    def test_the_self_check_moves_a_fold_back(self) -> None:
        straight = np.array([(float(x), 0.0) for x in range(6)])
        check = SelfCheck.of(straight, bend_radius_limit(DP_GAP, PAIR_WIDTH))

        back, count = check(folded(straight, 1))

        assert count > 0
        assert back[[1, 2, 3]].all()

    def test_inner_track_runs_the_centreline_s_way(self, corner: RunResult) -> None:
        (pair,) = corner.pairs
        along = np.diff(pair.centreline, axis=0)
        for track in (pair.a, pair.b):
            assert np.all(np.einsum("ij,ij->i", np.diff(track, axis=0), along) > 0)
        assert_gap_kept(pair)
        (given,) = scenarios.pair_corner().pairs
        assert_within_extent(pair, given, CORNER)


def test_chamfered_ends_keep_both_tracks_directions() -> None:
    scenario = scenarios.pair_chamfered()
    (given,) = scenario.pairs

    (pair,) = run(scenario).pairs

    assert_meets_its_ends(pair, given)
    assert_gap_kept(pair)
    assert length(pair.centreline) < pair.before.length


def test_track_b_s_end_edge_takes_up_an_uneven_end_s_gap() -> None:
    scenario = scenarios.pair_uneven_end()
    (given,) = scenario.pairs

    (pair,) = run(scenario).pairs

    for track, original in ((pair.a, given.a.points), (pair.b, given.b.points)):
        np.testing.assert_array_equal(track[[0, -1]], original[[0, -1]])
    a = given.a.points
    for points, before in ((pair.a, a), (pair.a[::-1], a[::-1])):
        np.testing.assert_allclose(
            heading(points), heading(before), atol=DIRECTION_TOLERANCE
        )
    assert_gap_kept(pair)
    assert length(pair.centreline) < pair.before.length


@pytest.fixture(scope="module")
def split() -> RunResult:
    return run(scenarios.pair_split())


class TestTwoCoupledSections:
    def test_optimizes_each_section(self, split: RunResult) -> None:
        for given, pair in zip(scenarios.pair_split().pairs, split.pairs, strict=True):
            assert length(pair.centreline) < pair.before.length - 0.05
            assert_gap_kept(pair)
            assert_meets_its_ends(pair, given)

    def test_fan_outs_and_sections_do_not_repel_where_they_touch(
        self, split: RunResult
    ) -> None:
        for chain in split.chains:
            assert length(chain.polyline) < chain.before.length

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


def test_a_pair_turning_back_around_a_pad_keeps_its_gap_and_bend() -> None:
    result = run(scenarios.pair_turning_back(), scenarios.TURNING_BACK)

    (pair,) = result.pairs
    assert_gap_kept(pair)
    assert pair.bend is not None and not pair.bend.too_tight
    assert result.termination is Termination.CONVERGED


def test_a_section_held_stays_as_it_is() -> None:
    (given,) = scenarios.pair_over_pad().pairs

    result = optimize_chains(
        (), scenarios.PARAMS, COARSE, pairs=(replace(given, hold=Hold.GROUP),)
    )

    assert result.termination is Termination.NOTHING_TO_DO
    (pair,) = result.pairs
    np.testing.assert_array_equal(pair.a, given.a.points)
    np.testing.assert_array_equal(pair.b, given.b.points)
    assert pair.block.held
    assert pair.bend is None


def held(scenario: PairsScenario) -> PairsScenario:
    """The scenario with its sections held, as run.py holds one that ended
    too tight."""
    return replace(
        scenario, pairs=tuple(replace(p, hold=Hold.CROWDING) for p in scenario.pairs)
    )


def hausdorff(a: FloatArray, b: FloatArray) -> float:
    return float(LineString(a).hausdorff_distance(LineString(b)))


def hold_every(results: Sequence[PairResult]) -> list[int]:
    """Every section, as if each had ended bent too tightly; the run holds
    those not held yet."""
    return list(range(len(results)))


def hold_once(
    scenario: PairsScenario,
    settings: OptimizerSettings = COARSE,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult:
    """A run of `scenario` that holds its sections when it would end."""
    return optimize_chains(
        scenario.chains,
        scenario.params,
        settings,
        pairs=scenario.pairs,
        clock=clock,
        to_hold=hold_every,
    )


def assert_held_as_drawn(result: RunResult, scenario: PairsScenario) -> None:
    """Each section is its drawn tracks, held, and no chain comes within its
    clearance of them."""
    for pair, given in zip(result.pairs, scenario.pairs, strict=True):
        assert pair.block.hold is Hold.CROWDING
        np.testing.assert_array_equal(pair.a, given.a.points)
        np.testing.assert_array_equal(pair.b, given.b.points)
        for chain in result.chains:
            for track in (given.a.points, given.b.points):
                assert LineString(chain.polyline).distance(LineString(track)) >= (
                    scenarios.PAIR_INFLATION - INSIDE_TOLERANCE
                )


@pytest.fixture(scope="module")
def roofed() -> RunResult:
    return hold_once(scenarios.pair_tent_under_roof())


class TestSectionHeldWhenTheRunWouldEnd:
    def test_ends_held_as_drawn_with_every_chain_clear_of_it(
        self, roofed: RunResult
    ) -> None:
        assert roofed.termination is Termination.CONVERGED
        assert_held_as_drawn(roofed, scenarios.pair_tent_under_roof())

    def test_ends_as_short_as_a_run_that_starts_with_it_held(
        self, roofed: RunResult
    ) -> None:
        over = run(held(scenarios.pair_tent_under_roof()))
        for went_on, started in zip(roofed.chains, over.chains, strict=True):
            assert length(went_on.polyline) <= length(started.polyline) + SETTLED_LENGTH


SETTLED_LENGTH = 0.005
"""mm a chain may end longer when the run goes on past a hold than when it
starts with the section held: two settled runs reach the same shape from
different starts, each within `COARSE.settled_change`."""

HOLD_BUDGET = 4
"""Bursts of `pair_tent_under_roof` at one spacing that the time limit
allows: the pair has come down onto the bar, and the roof has followed it."""


def test_a_section_held_at_the_time_limit_keeps_the_others_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = scenarios.pair_tent_under_roof()
    # Stopped at a coarse level, the roof would go back to its coarse
    # layout, not to the inputs at the final spacing.
    limited = replace(ONE_SPACING, time_limit=HOLD_BUDGET)
    plain = optimize_chains(
        scenario.chains,
        scenario.params,
        limited,
        pairs=scenario.pairs,
        clock=BurstClock(monkeypatch),
    )
    clock = BurstClock(monkeypatch)

    result = hold_once(scenario, limited, clock)

    # No burst runs after the hold, past the limit.
    assert clock.bursts == HOLD_BUDGET
    assert result.termination is Termination.TIME_LIMIT
    assert_held_as_drawn(result, scenario)
    bar, roof = result.chains
    np.testing.assert_array_equal(bar.polyline, plain.chains[0].polyline)
    assert length(bar.polyline) < bar.before.length
    # The roof had come down across where the pair was drawn, so it went back.
    assert (
        LineString(plain.chains[1].polyline).distance(
            LineString(scenario.pairs[0].a.points)
        )
        < scenarios.PAIR_INFLATION
    )
    # Its inputs: its drawn vertices, the edges between them split at the
    # spacing.
    drawn = scenario.chains[1].centreline.points
    np.testing.assert_array_equal(
        roof.polyline,
        resample(drawn, np.ones(len(drawn), dtype=bool), COARSE.spacing)[0],
    )


HELD_COARSELY = 8
"""An iteration cap at which `pair_tent_under_roof` with coarse levels
1.2 and 0.6 mm reaches it at 0.6 mm, so that its section is held there."""


def test_a_hold_at_a_coarse_level_logs_no_stage_again(log: list[str]) -> None:
    settings = replace(COARSE, coarse_spacings=(1.2, 0.6), max_iterations=HELD_COARSELY)

    result = hold_once(scenarios.pair_tent_under_roof(), settings)

    # The cap may also end the run after the hold: the roof, put back, takes
    # more bursts than the cap leaves it.
    assert result.termination in (Termination.CONVERGED, Termination.ITERATION_CAP)
    stages = [
        line.strip()
        for line in log
        if any(word in line for word in ("coarsely", "Refining", "Holding"))
    ]
    assert [line.split(" and going on")[0] for line in stages] == [
        "INFO Optimizing the chains on F.Cu coarsely, at 1.2 mm spacing",
        "INFO Optimizing the chains on F.Cu coarsely, at 0.6 mm spacing",
        "INFO Holding 1 coupled section of the chains on F.Cu as drawn",
        "INFO Refining the chains on F.Cu at 0.3 mm spacing",
    ]


TENT_SHIFT = 12.0
"""mm along x between the two tents of `two_tents`, far enough that neither
sees the other."""


def two_tents() -> PairsScenario:
    """`pair_tent` and a copy of it of other nets beside it, on one layer."""
    (first,) = scenarios.pair_tent().pairs
    (bar,) = scenarios.pair_tent().chains
    shifted = [(x + TENT_SHIFT, y) for x, y in ((0.0, 0.0), (3.0, 3.0), (7.0, 3.0))]
    second = scenarios.coupled(FreeSpace(), *shifted, (10.0 + TENT_SHIFT, 0.0))
    second = replace(
        second,
        a=replace(second.a, description="P2", net="P2"),
        b=replace(second.b, description="N2", net="N2"),
    )
    beside = [(x + TENT_SHIFT, y) for x, y in scenarios.PAIR_BAR]
    bar_two = selected("bar 2", "B2", FreeSpace(), *beside)
    return PairsScenario("two-tents", (first, second), (bar, bar_two))


def hold_first(results: Sequence[PairResult]) -> list[int]:
    """The first section not held yet, as if holding one left the next bent
    too tightly."""
    return [k for k, result in enumerate(results) if not result.block.held][:1]


def test_a_hold_at_the_time_limit_holds_what_it_leaves_too_tight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = two_tents()
    clock = BurstClock(monkeypatch)

    result = optimize_chains(
        scenario.chains,
        scenario.params,
        replace(COARSE, time_limit=1),
        pairs=scenario.pairs,
        clock=clock,
        to_hold=hold_first,
    )

    assert result.termination is Termination.TIME_LIMIT
    assert clock.bursts == 1
    assert_held_as_drawn(result, scenario)


def test_a_hold_after_a_stop_starts_no_burst(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = two_tents()
    clock = BurstClock(monkeypatch)

    result = optimize_chains(
        scenario.chains,
        scenario.params,
        COARSE,
        pairs=scenario.pairs,
        clock=clock,
        stop=lambda: clock.bursts >= 1,
        to_hold=hold_first,
    )

    assert result.termination is Termination.STOPPED
    assert clock.bursts == 1
    assert_held_as_drawn(result, scenario)


SETTLES_AT_ONCE = 100.0
"""A settle threshold, mm, beyond any change of `two_tents`: its first
burst settles it."""


@pytest.mark.parametrize(
    ("time_limit", "stop_after", "termination"),
    [(None, 1, Termination.STOPPED), (1, None, Termination.TIME_LIMIT)],
)
def test_a_hold_after_converging_as_the_run_ends_starts_no_burst(
    monkeypatch: pytest.MonkeyPatch,
    time_limit: float | None,
    stop_after: int | None,
    termination: Termination,
) -> None:
    # The burst that converges the run is the one after which it ends; the
    # hold then undoes the convergence, so the run ends on what ended it.
    scenario = two_tents()
    clock = BurstClock(monkeypatch)
    settings = replace(COARSE, settled_change=SETTLES_AT_ONCE, time_limit=time_limit)

    result = optimize_chains(
        scenario.chains,
        scenario.params,
        settings,
        pairs=scenario.pairs,
        clock=clock,
        stop=None if stop_after is None else lambda: clock.bursts >= stop_after,
        to_hold=hold_first,
    )

    assert result.termination is termination
    assert clock.bursts == 1
    assert_held_as_drawn(result, scenario)


def test_a_second_hold_in_one_problem_goes_on_from_the_first() -> None:
    scenario = two_tents()
    asked: list[list[int]] = []

    def counted(results: Sequence[PairResult]) -> list[int]:
        named = hold_first(results)
        asked.append(named)
        # A hold forgotten would be named again whenever the run settles.
        assert len(asked) <= len(scenario.pairs) + 1
        return named

    result = optimize_chains(
        scenario.chains,
        scenario.params,
        COARSE,
        pairs=scenario.pairs,
        to_hold=counted,
    )

    assert asked == [[0], [1], []]
    assert result.termination is Termination.CONVERGED
    assert_held_as_drawn(result, scenario)
    for chain in result.chains:
        assert length(chain.polyline) < chain.before.length


def test_a_hold_leaves_another_problem_as_it_was() -> None:
    scenario = scenarios.pair_tent()
    back = replace(
        selected("back", "C", FreeSpace(), (0.0, 0.0), (5.0, 3.0), (10.0, 0.0)),
        layer="B.Cu",
    )
    chains = (*scenario.chains, back)
    plain = optimize_chains(chains, scenario.params, COARSE, pairs=scenario.pairs)

    result = optimize_chains(
        chains, scenario.params, COARSE, pairs=scenario.pairs, to_hold=hold_every
    )

    assert result.pairs[0].block.hold is Hold.CROWDING
    np.testing.assert_array_equal(result.chains[1].polyline, plain.chains[1].polyline)


def test_a_section_named_again_once_held_is_not_held_again() -> None:
    scenario = scenarios.pair_tent()
    asked = 0

    def every_section(results: Sequence[PairResult]) -> list[int]:
        nonlocal asked
        asked += 1
        # Holding it again would rebuild the problem every time it settles.
        assert asked <= 2
        return list(range(len(results)))

    result = optimize_chains(
        scenario.chains,
        scenario.params,
        COARSE,
        pairs=scenario.pairs,
        to_hold=every_section,
    )

    assert asked == 2
    assert result.termination is Termination.CONVERGED
    assert result.pairs[0].block.hold is Hold.CROWDING


def test_a_give_up_before_a_hold_outlasts_a_convergence_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = optimize_module.run_burst
    holds: list[int] = []

    def fails_until_held(
        objective: Objective,
        start: FloatArray,
        revert: Check,
        settings: OptimizerSettings,
        stop: Stop | None = None,
    ) -> Burst:
        burst = real(objective, start, revert, settings, stop)
        return replace(burst, abnormal=not holds)

    def counted(results: Sequence[PairResult]) -> list[int]:
        named = hold_every(results)
        holds.extend(named)
        return named

    monkeypatch.setattr(optimize_module, "run_burst", fails_until_held)
    scenario = scenarios.pair_tent_under_roof()

    result = optimize_chains(
        scenario.chains,
        scenario.params,
        COARSE,
        pairs=scenario.pairs,
        to_hold=counted,
    )

    # Named when the run gave up, and again, held, when it converged.
    assert holds == [0, 0]
    every = all_abnormal_bursts(COARSE)
    assert result.gave_up == (("the chains on F.Cu", MAX_ABNORMAL_BURSTS),)
    assert result.termination is Termination.GAVE_UP
    # The abnormal bursts were all before the hold.
    assert result.abnormal == (("the chains on F.Cu", every),)


JOINED_AT = (9.5, 5.0)
"""Where `roof_parts` joins its two parts, right of the pair's arch, mm."""
ROOF_FAR_END = (10.5, 5.0)


def roof_parts(*, via: bool) -> tuple[PairsScenario, tuple[ViaInput, ...]]:
    """`pair_tent` under a roof of two parts joined at `JOINED_AT`: the first
    from beside the arch's left side, over the arch, the second on to the
    right, clear of the arch. Once the pair comes down, the first part pulls
    across where the pair was drawn; the second stays clear of it. Joined by
    a moving via, with the second part on B.Cu, if `via`; else at a
    junction."""
    tent = scenarios.pair_tent()
    eaves = (1.0, scenarios.ROOF_EAVES)
    first = selected("roof", "R", FreeSpace(), eaves, (1.0, 4.5), JOINED_AT)
    second = selected(
        "roof end", "R", FreeSpace(), JOINED_AT, ROOF_FAR_END, layer="B.Cu"
    )
    vias: tuple[ViaInput, ...] = ()
    if via:
        vias = (scenarios.joining(scenarios.through_via("v", "R", JOINED_AT), 1),)
    else:
        loose = first.fixed.copy()
        loose[-1] = False
        first = replace(first, fixed=loose)
        second = replace(second, layer="F.Cu", fixed=np.array([False, True]))
    return replace(tent, chains=(*tent.chains, first, second)), vias


@pytest.mark.parametrize("via", [False, True], ids=["junction", "via"])
def test_what_moves_with_a_part_put_back_goes_back_with_it(
    monkeypatch: pytest.MonkeyPatch, via: bool
) -> None:
    scenario, vias = roof_parts(via=via)

    result = optimize_chains(
        scenario.chains,
        scenario.params,
        replace(COARSE, time_limit=HOLD_BUDGET),
        vias,
        pairs=scenario.pairs,
        clock=BurstClock(monkeypatch),
        to_hold=hold_every,
    )

    assert_held_as_drawn(result, scenario)
    _, first, second = result.chains
    assert tuple(first.polyline[-1]) == JOINED_AT
    assert tuple(second.polyline[0]) == JOINED_AT
    if via:
        assert result.vias[0].position == JOINED_AT


def gap(a: FloatArray, b: FloatArray) -> float:
    return float(LineString(a).distance(LineString(b)))


def y_at(points: FloatArray, x: float) -> float:
    return float(np.interp(x, points[:, 0], points[:, 1]))


def test_tracks_lie_half_a_pitch_off_the_centreline(over_pad: RunResult) -> None:
    (pair,) = over_pad.pairs
    half_pitch = (DP_GAP + PAIR_WIDTH) / 2
    for track in (pair.a, pair.b):
        spread = gap_range(track, pair.centreline, PAIR_WIDTH)
        assert spread == pytest.approx((half_pitch - PAIR_WIDTH,) * 2, abs=GAP_SLACK)


def test_is_deterministic(tent: RunResult) -> None:
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


def test_a_short_chamfer_at_an_end_keeps_its_corner_free() -> None:
    # A direction pin on the corner would keep the turn there from rounding.
    (given,) = scenarios.pair_chamfered().pairs

    block = pair_block(given, COARSE.spacing)

    points, fixed = block.chain.centreline.points, block.chain.fixed
    assert len(points) == len(given.a.points) + 2
    assert fixed[[0, 1, -2, -1]].all()
    assert not fixed[[2, -3]].any()


def test_split_gives_two_sections_and_a_detour_on_each_track() -> None:
    scenario = scenarios.pair_split()
    assert len(scenario.pairs) == 2
    assert [c.centreline.net for c in scenario.chains] == ["P", "N"]


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


INSIDE_DEPTH = 0.02
"""How far the chain starts inside the section's clearance, mm."""


def test_a_chain_starting_inside_a_section_s_clearance_moves_clear_of_it() -> None:
    section = scenarios.coupled(FreeSpace(), (0.0, 0.0), (5.0, 0.0))
    half_extent = pair_block(section, COARSE.spacing).facing.centreline.width / 2
    # The chain runs inside the clearance of the section's nearer track,
    # pinned there at both ends.
    y = half_extent + PAIR_CLEARANCE + PAIR_WIDTH / 2 - INSIDE_DEPTH
    chain = selected("C", "C", FreeSpace(), (0.0, y), (5.0, y))

    result = optimize_chains([chain], scenarios.PARAMS, COARSE, pairs=(section,))

    (moved,), (pair,) = result.chains, result.pairs
    middle = LineString(moved.polyline).interpolate(0.5, normalized=True)
    for track in (pair.a, pair.b):
        assert LineString(track).distance(middle) >= (
            scenarios.PAIR_INFLATION - INSIDE_TOLERANCE
        )


def test_a_junction_where_a_pair_starts_stays_whole() -> None:
    # Each net forks where the pair starts: two branches couple, the third
    # leaves on its own, the pair's two in opposite directions.
    at = {"D+": (10.0, 0.0), "D-": (9.5, 0.5)}

    def branches(net: str, side: float) -> list[Chain]:
        x, y = at[net]
        return [
            Chain(((0.0, y), (x, y)), (True, False), 0.25, "F.Cu", net, ()),
            Chain(((x, y), (20.0, y)), (False, True), 0.25, "F.Cu", net, ()),
            Chain(((x, y), (x, 10.0 * side)), (False, True), 0.25, "F.Cu", net, ()),
        ]

    chains = [*branches("D+", -1.0), *branches("D-", 1.0)]
    whole = [
        ChainInput(
            ChainCentreline(c.net, c.net, c.width, PAIR_CLEARANCE, np.array(c.points)),
            c.layer,
            np.array(c.pinned),
            FreeSpace(),
        )
        for c in chains
    ]

    def free(
        _layer: str, _a: ChainCentreline, _b: ChainCentreline, _gap: float
    ) -> ObstacleField:
        return FreeSpace()

    parts = cut_parts(chains, DEFAULT_CHORD_TOLERANCE)
    pairs = find_pairs([p.chain for p in parts], DEFAULT_CHORD_TOLERANCE)
    layout = lay_out(whole, parts, pairs, free)

    result = optimize_chains(
        layout.inputs, scenarios.PARAMS, COARSE, pairs=layout.sections
    )

    assert layout.sections
    for given, chain in zip(layout.inputs, result.chains, strict=True):
        for end in (0, -1):
            start = tuple(given.centreline.points[end])
            if start in at.values():
                assert tuple(chain.polyline[end]) == start


PINNED_PAIR = ((0.0, 0.0), (3.0, 0.0), (5.0, 2.0), (9.0, 2.0), (11.0, 0.0), (14.0, 0.0))
"""A pair's centreline with two 45° corners on each side of its middle
straight, which runs from x = 5 to x = 9."""
PAD_EDGES = (6.8, 7.2)
"""x where a same-net pad's outline crosses each track of the pair."""
STUB_AT = 7.0
"""x where an unselected same-net stub leaves track P."""
END_PAD_EDGE = 0.055
"""x where the pads at the pair's start cross its tracks, as on the board of
https://github.com/mairas/kibends/issues/42."""
ROUNDED = 0.05
"""mm: how far a rounded corner's track passes from where its corner was, at
least."""
LIKE_UNPINNED = 2e-3
"""mm: how far a pair pinned only in its end pads may end up from the same
pair unpinned, whose direction pin is the pads' pin. The two runs resample
differently and stop where L-BFGS's ftol ends them, about 1 µm apart."""


def pinned_chain(net: str, track: FloatArray, *xs: float) -> Chain:
    """`track` as a chain of `net`, pinned at its ends and at a vertex
    inserted where each of `xs` meets one of its horizontal edges."""
    points = [(float(x), float(y)) for x, y in track]
    pinned = [True, *[False] * (len(points) - 2), True]
    for x in xs:
        k = next(
            k
            for k in range(len(points) - 1)
            if abs(points[k][1] - points[k + 1][1]) < EXTENT_SLACK
            and points[k][0] < x < points[k + 1][0]
        )
        points.insert(k + 1, (x, points[k][1]))
        pinned.insert(k + 1, True)
    return Chain(tuple(points), tuple(pinned), PAIR_WIDTH, "F.Cu", net, ())


def pinned_pair(pins_p: Sequence[float], pins_n: Sequence[float]) -> list[Chain]:
    p, n = scenarios.tracks(*PINNED_PAIR)
    return [pinned_chain("USB_D+", p, *pins_p), pinned_chain("USB_D-", n, *pins_n)]


def free_section(
    _layer: str, _a: ChainCentreline, _b: ChainCentreline, _gap: float
) -> ObstacleField:
    return FreeSpace()


@dataclass(frozen=True)
class Smoothed:
    result: RunResult
    joined: list[FloatArray]
    """One polyline per chain, as the layout joins them."""


def smoothed(chains: Sequence[Chain], settings: OptimizerSettings = COARSE) -> Smoothed:
    """`chains` laid out with their pairs and run in free space."""
    whole = [
        ChainInput(
            ChainCentreline(c.net, c.net, c.width, PAIR_CLEARANCE, np.array(c.points)),
            c.layer,
            np.array(c.pinned),
            FreeSpace(),
        )
        for c in chains
    ]
    parts = cut_parts(chains, DEFAULT_CHORD_TOLERANCE)
    pairs = find_pairs([p.chain for p in parts], DEFAULT_CHORD_TOLERANCE)
    layout = lay_out(whole, parts, pairs, free_section)
    result = optimize_chains(
        layout.inputs, scenarios.PARAMS, settings, pairs=layout.sections
    )
    joined = layout.whole_polylines(
        [c.polyline for c in result.chains], [r.kept() for r in result.pairs]
    )
    return Smoothed(result, joined)


def assert_pins_kept(chains: Sequence[Chain], joined: Sequence[FloatArray]) -> None:
    for chain, line in zip(chains, joined, strict=True):
        for point, pinned in zip(chain.points, chain.pinned, strict=True):
            if pinned:
                assert (line == np.array(point)).all(axis=1).any(), point


def assert_corners_round(chains: Sequence[Chain], joined: Sequence[FloatArray]) -> None:
    """Every corner of each track's routing is rounded off."""
    for chain, line in zip(chains, joined, strict=True):
        points = np.array(chain.points)
        step = np.diff(points, axis=0)
        turning = np.abs(_cross_2d(step[:-1], step[1:])) > DIRECTION_TOLERANCE
        turns = np.flatnonzero(turning) + 1
        assert len(turns) == 4
        for corner in points[turns]:
            assert LineString(line).distance(Point(corner)) >= ROUNDED, corner


def _cross_2d(u: FloatArray, v: FloatArray) -> FloatArray:
    return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]


def assert_moved_sections_keep_their_gap(result: RunResult) -> None:
    moved = [p for p in result.pairs if not p.block.held]
    assert moved
    for pair in moved:
        assert_gap_kept(pair)


def test_a_pair_past_same_net_pads_rounds_its_corners_on_both_sides() -> None:
    chains = pinned_pair(PAD_EDGES, PAD_EDGES)

    out = smoothed(chains)

    assert_pins_kept(chains, out.joined)
    assert_corners_round(chains, out.joined)
    assert_moved_sections_keep_their_gap(out.result)


def test_a_pair_with_a_same_net_stub_rounds_its_corners_on_both_sides() -> None:
    chains = pinned_pair((STUB_AT,), ())

    out = smoothed(chains)

    assert_pins_kept(chains, out.joined)
    assert_corners_round(chains, out.joined)
    assert_moved_sections_keep_their_gap(out.result)


def test_a_pair_pinned_only_in_its_end_pads_moves_as_an_unpinned_one() -> None:
    chains = pinned_pair((END_PAD_EDGE,), (END_PAD_EDGE,))

    out = smoothed(chains)
    unpinned = smoothed(pinned_pair((), ()))

    assert_pins_kept(chains, out.joined)
    for chain, line, free_line in zip(chains, out.joined, unpinned.joined, strict=True):
        assert hausdorff(line, np.array(chain.points)) >= ROUNDED
        assert hausdorff(line, free_line) <= LIKE_UNPINNED


FAN_OUT_SPREAD = 0.2
"""mm each track of a diff pair moves outwards over its 6 mm fan-out: about
2°, so the two stay within each other's clearance for the first 0.1 mm."""


def test_a_diff_pair_fanning_out_at_a_sharp_angle_keeps_its_fan_outs() -> None:
    # Coupled, then straight to pads; nothing but each other could bend them.
    p, n = scenarios.tracks((0.0, 0.0), (6.0, 0.0))

    def fanning(net: str, track: FloatArray) -> Chain:
        y = track[-1, 1]
        points = (
            *((float(a), float(b)) for a, b in track),
            (12.0, float(y + math.copysign(FAN_OUT_SPREAD, y))),
        )
        pinned = (True, *[False] * (len(points) - 2), True)
        return Chain(points, pinned, PAIR_WIDTH, "F.Cu", net, ())

    chains = [fanning("USB_D+", p), fanning("USB_D-", n)]

    out = smoothed(chains)

    assert len(out.result.pairs) == 1
    for chain, line in zip(chains, out.joined, strict=True):
        assert hausdorff(line, np.array(chain.points)) < COARSE.settled_change


SHORT_PART = ((0.0, -5.0), (0.15, -5.0), (0.3, -4.85))
"""A section's centreline whose end edges are shorter than the direction
pins' room, so that its direction pins, halfway along them, leave less
than a spacing between them; clear of `pair_corner`."""


def test_a_section_too_short_to_move_stays_as_routed() -> None:
    short = scenarios.coupled(FreeSpace(), *SHORT_PART)
    # Offsetting its centreline would drop track N's vertex of its own.
    n = short.b.points
    short = replace(
        short,
        b=replace(short.b, points=np.insert(n, 1, (n[0] + n[1]) / 2, axis=0)),
        b_pinned=np.insert(short.b_pinned, 1, False),
    )
    (corner,) = scenarios.pair_corner().pairs

    result = optimize_chains((), scenarios.PARAMS, COARSE, pairs=(short, corner))

    kept, moved = result.pairs
    np.testing.assert_array_equal(kept.a, short.a.points)
    np.testing.assert_array_equal(kept.b, short.b.points)
    assert kept.block.hold is Hold.SHORT
    assert not np.array_equal(moved.a, corner.a.points)


NEAR_CORNER = (0.0, 0.02, -0.02, 0.1, -0.1, 0.2, -0.2, 0.28, -0.28)
"""mm along a track from one of its corners, ahead of it where positive,
all within two pitches."""
CLEAR_OF_CORNER = 0.9
"""mm along a track from one of its corners, beyond two pitches, 0.7 mm, on
both tracks, whose corners lie a pitch·tan(turn/2) apart along the pair."""
CORNERS = (1, 2)
"""Corners of `PINNED_PAIR`: P is inside the first and outside the second.
The other two mirror them, run the other way."""
ACROSS_N_CORNER = 9.0
"""x of a point on P's middle straight whose nearest point on N is N's
corner at the straight's end."""
QUICK_SETTLE = 0.02
"""mm, the settled change of the runs below: four times a real run's, so
that they converge in under a second. They run to convergence, since where
a run cut off unconverged stops depends on the CPU."""
DEFAULT_SPACING = replace(
    COARSE, spacing=OptimizerSettings().spacing, settled_change=QUICK_SETTLE
)
"""The default spacing, at which a section's direction pins and its starting
bend are those of a real run, with short bursts to run fast."""
COARSE_QUICK = replace(COARSE, settled_change=QUICK_SETTLE)
SPACINGS = [
    pytest.param(COARSE_QUICK, id="coarse"),
    pytest.param(DEFAULT_SPACING, id="default"),
]
CONVERGED = {Termination.CONVERGED, Termination.NOTHING_TO_DO}
PINNED_HOLDS = {None, Hold.SHORT, Hold.ROUTING}
"""How a section with pins between its ends may be held: as routed, where
what a pin keeps of its routing bends too tightly or leaves nothing to
move, never after the run."""
RIGHT_ANGLE_PAIR = ((0.0, 0.0), (4.0, 0.0), (4.0, -4.0))
"""A pair's centreline turning left by 90°: P is outside its corner."""


def pinned_along(net: str, track: FloatArray, *arcs: float) -> Chain:
    """`track` as a chain of `net`, pinned at its ends and at the point at
    each of `arcs` along it, a vertex inserted there unless one is."""
    points = [(float(x), float(y)) for x, y in track]
    pinned = [True, *[False] * (len(points) - 2), True]
    for arc in arcs:
        along = arc_lengths(np.array(points))
        k = int(np.searchsorted(along, arc - EXTENT_SLACK))
        if not math.isclose(along[k], arc, abs_tol=EXTENT_SLACK):
            t = (arc - along[k - 1]) / (along[k] - along[k - 1])
            start, end = np.array(points[k - 1]), np.array(points[k])
            x, y = start + t * (end - start)
            points.insert(k, (float(x), float(y)))
            pinned.insert(k, False)
        pinned[k] = True
    return Chain(tuple(points), tuple(pinned), PAIR_WIDTH, "F.Cu", net, ())


def corner_arc(track: FloatArray, corner: int) -> float:
    """Arc length along `track` of its vertex at `corner`."""
    return float(arc_lengths(track)[corner])


def pinned_near(
    tracks: tuple[FloatArray, FloatArray], pinned: str, arcs: Sequence[float]
) -> list[Chain]:
    """Tracks P and N as chains, track `pinned` pinned at `arcs` along it."""
    p, n = tracks
    return [
        pinned_along("USB_D+", p, *(arcs if pinned == "P" else ())),
        pinned_along("USB_D-", n, *(arcs if pinned == "N" else ())),
    ]


def assert_kept_or_held(
    chains: Sequence[Chain], settings: OptimizerSettings, gap: float = DP_GAP
) -> Smoothed:
    """Smooth `chains` to convergence: every pin stays, the joined tracks
    keep the pair's gap, so they neither narrow nor overlap, and no section
    is held after the run for its bend, nor ends it bending too tightly."""
    out = smoothed(chains, settings)

    assert out.result.termination in CONVERGED
    assert_pins_kept(chains, out.joined)
    p, n = out.joined
    low, high = gap_range(p, n, PAIR_WIDTH)
    slack = GAP_TOLERANCE * gap
    assert gap - slack <= low <= high <= gap + slack
    assert {p.block.hold for p in out.result.pairs} <= PINNED_HOLDS
    for pair in out.result.pairs:
        assert pair.block.held or not (pair.bend is not None and pair.bend.too_tight)
    return out


@pytest.mark.parametrize("settings", SPACINGS)
@pytest.mark.parametrize("offset", NEAR_CORNER)
@pytest.mark.parametrize("corner", CORNERS)
@pytest.mark.parametrize("pinned", ["P", "N"])
def test_a_pin_near_a_corner_of_either_track_stays_on_it(
    pinned: str, corner: int, offset: float, settings: OptimizerSettings
) -> None:
    tracks = scenarios.tracks(*PINNED_PAIR)
    track = tracks[0 if pinned == "P" else 1]
    chains = pinned_near(tracks, pinned, [corner_arc(track, corner) + offset])

    out = assert_kept_or_held(chains, settings)

    assert len(out.result.pairs) == 1


@pytest.mark.parametrize(
    ("pinned", "offset"),
    # Its direction pin a spacing short of the corner, which stays free.
    [("P", 0.2), ("P", -0.2), ("N", 0.1), ("N", -0.1)],
)
def test_a_pin_near_a_corner_leaves_the_corner_to_round(
    pinned: str, offset: float
) -> None:
    tracks = scenarios.tracks(*PINNED_PAIR)
    track = tracks[0 if pinned == "P" else 1]
    chains = pinned_near(tracks, pinned, [corner_arc(track, 2) + offset])

    out = assert_kept_or_held(chains, DEFAULT_SPACING)

    assert [p.block.hold for p in out.result.pairs] == [None]


@pytest.mark.parametrize("settings", SPACINGS)
@pytest.mark.parametrize("offset", [0.0, 0.1, -0.1])
@pytest.mark.parametrize("corner", CORNERS)
def test_a_pad_across_both_tracks_near_a_corner_keeps_both_on_it(
    corner: int, offset: float, settings: OptimizerSettings
) -> None:
    # At 0 the pad touches both tracks' corner vertices, across the bend.
    p, n = scenarios.tracks(*PINNED_PAIR)
    chains = [
        pinned_along("USB_D+", p, corner_arc(p, corner) + offset),
        pinned_along("USB_D-", n, corner_arc(n, corner) + offset),
    ]

    out = assert_kept_or_held(chains, settings)

    assert len(out.result.pairs) == 1


@pytest.mark.parametrize("settings", SPACINGS)
@pytest.mark.parametrize(
    ("corner", "offset"),
    # On the bump between the first and last straights, which a pin on
    # either of those would let the pair straighten away in free space.
    [(1, CLEAR_OF_CORNER), (2, -CLEAR_OF_CORNER), (2, CLEAR_OF_CORNER)],
)
@pytest.mark.parametrize("pinned", ["P", "N"])
def test_a_pin_two_pitches_from_every_corner_rounds_them_all(
    pinned: str, corner: int, offset: float, settings: OptimizerSettings
) -> None:
    tracks = scenarios.tracks(*PINNED_PAIR)
    track = tracks[0 if pinned == "P" else 1]
    chains = pinned_near(tracks, pinned, [corner_arc(track, corner) + offset])

    out = assert_kept_or_held(chains, settings)

    assert [p.block.hold for p in out.result.pairs] == [None]
    assert_corners_round(chains, out.joined)


def test_a_pin_across_from_the_other_tracks_corner_holds_its_section() -> None:
    # P is outside the corner: its pin and its corner offset to one point
    # of the centreline, which turns there on the spot.
    out = assert_kept_or_held(pinned_pair((ACROSS_N_CORNER,), ()), COARSE)

    assert [p.block.hold for p in out.result.pairs] == [Hold.ROUTING]


@pytest.mark.parametrize("offset", [0.0, 0.02, -0.02, 0.1, -0.1])
@pytest.mark.parametrize("pinned", ["P", "N"])
def test_a_pin_at_a_right_angle_corner_stays_on_its_track(
    pinned: str, offset: float
) -> None:
    tracks = scenarios.tracks(*RIGHT_ANGLE_PAIR)
    track = tracks[0 if pinned == "P" else 1]
    chains = pinned_near(tracks, pinned, [corner_arc(track, 1) + offset])

    out = assert_kept_or_held(chains, DEFAULT_SPACING)

    assert len(out.result.pairs) == 1


@pytest.mark.parametrize("pinned", ["P", "N"])
def test_a_pin_on_a_straight_lets_a_right_angle_corner_round(pinned: str) -> None:
    tracks = scenarios.tracks(*RIGHT_ANGLE_PAIR)
    track = tracks[0 if pinned == "P" else 1]
    chains = pinned_near(tracks, pinned, [corner_arc(track, 1) - 2.0])

    out = assert_kept_or_held(chains, DEFAULT_SPACING)

    assert [p.block.hold for p in out.result.pairs] == [None]
    for chain, line in zip(chains, out.joined, strict=True):
        corner = Point(np.array(chain.points)[-2])
        assert LineString(line).distance(corner) >= ROUNDED


@pytest.mark.parametrize(
    ("offsets", "holds"),
    # Within about a pitch of the corner, the outer track's pins and its
    # corner offset to centreline points that turn back on each other.
    [((0.1, 0.3), [Hold.ROUTING]), ((0.3, 1.5), [None])],
)
def test_pins_past_an_outer_corner_keep_their_place(
    offsets: tuple[float, float], holds: list[Hold | None]
) -> None:
    # N is outside corner 1: a pad's two outline crossings past it.
    tracks = scenarios.tracks(*PINNED_PAIR)
    at = corner_arc(tracks[1], 1)
    chains = pinned_near(tracks, "N", [at + offset for offset in offsets])

    out = assert_kept_or_held(chains, DEFAULT_SPACING)

    assert [p.block.hold for p in out.result.pairs] == holds


@pytest.mark.parametrize("gap", [0.3, 0.4, 0.5])
def test_a_pin_at_the_outer_corner_of_a_wider_pair_stays_on_its_track(
    gap: float,
) -> None:
    p, n = offset_pair(scenarios.path(*PINNED_PAIR), gap, PAIR_WIDTH)
    # N is outside corner 1.
    chains = pinned_near((p, n), "N", [corner_arc(n, 1)])

    out = assert_kept_or_held(chains, DEFAULT_SPACING, gap)

    assert len(out.result.pairs) == 1


SHARP_SPACING = 1.0
"""A spacing at which the section's corner may turn by 100 degrees within
the bend-radius limit, run alone: the default coarse levels must each be
above the final spacing, and 0.4 mm is not."""
TURN = math.radians(60.0)
SHARPENING = math.radians(40.0)
LEG = 5.0
CORNER_AT = (LEG, 0.0)
BEFORE_CORNER = (LEG - SHARP_SPACING, 0.0)
SHARPENED = (BEFORE_CORNER[0], BEFORE_CORNER[1] + SHARP_SPACING * math.tan(SHARPENING))
AFTER_CORNER = (LEG + SHARP_SPACING * math.cos(TURN), SHARP_SPACING * math.sin(TURN))
POST_WIDTH = 0.1
POST_LENGTH = 0.001


def post_past_the_corner() -> PairsScenario:
    """A pair turning left at `CORNER_AT`, and a short pinned post of
    another net outside it, where track N's mitred corner reaches once the
    vertex before the corner moves to `SHARPENED`: nearer to N's edge after
    the corner than N's clearance, but not to N's corner, before or after,
    nor to N where it starts."""
    end = (LEG + LEG * math.cos(TURN), LEG * math.sin(TURN))
    corners: list[FloatArray] = [
        scenarios.tracks(at, CORNER_AT, AFTER_CORNER)[1][1]
        for at in (BEFORE_CORNER, SHARPENED)
    ]
    before, sharpened = corners
    heading = np.array([math.cos(TURN), math.sin(TURN)])
    outward = np.array([heading[1], -heading[0]])
    # N's corner slides from `before` to `sharpened`; the post faces the
    # middle of that stretch, between N's reach of the edge and of the
    # corners at either end of the stretch.
    reach = (POST_WIDTH + PAIR_WIDTH) / 2 + PAIR_CLEARANCE
    half = float(np.linalg.norm(before - sharpened)) / 2
    beside = (reach + math.sqrt(reach**2 - half**2)) / 2
    middle = (before + sharpened) / 2 + beside * outward
    a, b = (middle + side * POST_LENGTH / 2 * heading for side in (-1, 1))
    post = selected(
        "post",
        "X",
        FreeSpace(),
        (float(a[0]), float(a[1])),
        (float(b[0]), float(b[1])),
        width=POST_WIDTH,
    )
    return PairsScenario(
        "post-past-the-corner",
        (scenarios.coupled(FreeSpace(), (0.0, 0.0), CORNER_AT, end),),
        (replace(post, fixed=np.ones(2, dtype=bool)),),
    )


@pytest.mark.fallback
def test_a_crossing_that_names_nothing_moved_moves_the_burst_back_whole(
    monkeypatch: pytest.MonkeyPatch, log: list[str]
) -> None:
    # N's corner slides back along its edge after the corner into the post,
    # which names that edge's ends, the corner and the vertex after it,
    # neither of which moved; the post is pinned. The section judges its
    # own edges by its centreline's extent, which the post starts inside
    # and gets no deeper into.
    settings = replace(ONE_SPACING, spacing=SHARP_SPACING, max_iterations=1)
    revert, start = first_check(
        monkeypatch, lambda: run(post_past_the_corner(), settings)
    )
    assert not any(FALLBACK_WARNING in line for line in log)
    log.clear()
    candidate = start.copy()
    vertices = candidate.reshape(-1, 2)
    (moved,) = np.flatnonzero((vertices == BEFORE_CORNER).all(axis=1))
    vertices[moved] = SHARPENED

    crossings = revert(candidate)

    assert crossings == 1
    assert revert.reverted.all()
    assert np.array_equal(candidate, start)
    assert [line.strip() for line in log] == [f"WARNING {FALLBACK_WARNING}"]
