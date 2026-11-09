import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pytest
import scenarios
from conftest import Ticking
from distance_fields import BoolArray, Disc, FloatArray, FreeSpace
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
from kibends.bursts import OptimizerSettings, revert_until_clear
from kibends.chain import Chain
from kibends.coords import BOARD_FRAME
from kibends.energy import (
    EnergyParams,
    ObstacleField,
    TermValue,
    obstacle_reach,
    total_energy,
)
from kibends.neighbours import ChainInput, chain_field, neighbour_field
from kibends.obstacles import ChainCentreline
from kibends.optimize import (
    INSIDE_TOLERANCE,
    ChainResult,
    Member,
    PairResult,
    RunResult,
    SelfCheck,
    Termination,
    close_to_itself,
    layer_energy,
    optimize_chains,
    resample,
    seen_parts,
)
from kibends.pair_blocks import Hold, PairBlock, PairInput, pair_block
from kibends.pair_layout import find_pairs, lay_out
from kibends.pairs import (
    BEND_RADIUS_TOLERANCE,
    GAP_TOLERANCE,
    bend_radius_limit,
    check_bend_radius,
    gap_range,
    offset_pair,
)
from kibends.simplify import DEFAULT_CHORD_TOLERANCE, simplify_chains

COARSE = OptimizerSettings(spacing=0.3, settled_change=0.003, burst_maxiter=30)
CORNER = OptimizerSettings(burst_maxiter=30)
"""The default spacing, at which a centreline pulled taut against its
direction pins bends tighter than the limit, with short bursts."""

GAP_SLACK = GAP_TOLERANCE * DP_GAP
DIRECTION_TOLERANCE = 1e-9
EXTENT_SLACK = 1e-9


def run(
    scenario: PairsScenario,
    settings: OptimizerSettings = COARSE,
    start: RunResult | None = None,
) -> RunResult:
    return optimize_chains(
        scenario.chains, scenario.params, settings, pairs=scenario.pairs, start=start
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
    reach = obstacle_reach(params)
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
            near = neighbour_field(
                member.facing, others, reach, points, as_net=member.as_net
            )
            static, limit = member.chain.field, member.limit
        else:
            near = neighbour_field(member, others, reach, points)
            static, limit = member.field, None
        field: ObstacleField = chain_field(static, [near])
        value += total_energy(points, mask, field, params, settings.short_edge, limit)[
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

    # A gap's taper starts at its end's direction pin: at the pinned end
    # edges the gaps have a kink, which no free vertex reaches.
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
        # nets' fan-outs, left out near the touch both ways.
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
        seen = seen_parts(members, polylines, obstacle_reach(params))
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
        # As run.py simplifies a section: its centreline vertices must come
        # back exactly as the run offset them for `offset_kept` to find them.
        (pair,) = over_pad.pairs
        block = pair.block
        result = ChainResult(pair.centreline, pair.fixed, 0, (), pair.before)

        (centreline,) = simplify_chains(
            [block.facing],
            replace(over_pad, chains=(result,), pairs=()),
            scenarios.PARAMS,
            scenarios.DP_CHORD_TOLERANCE,
            sections={0: block},
        )
        a, b = block.offset_kept(pair.centreline, centreline)

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
        def unhinged(
            points: FloatArray,
            fixed: BoolArray,
            limit: float,
            params: EnergyParams,
        ) -> TermValue:
            return 0.0, np.zeros_like(points)

        monkeypatch.setattr(optimize_module, "bend_radius_energy", unhinged)
        # Routed wider than the limit: length alone would pull it into the
        # corner, below the limit, and only the self check holds it back.
        (pair,) = run(scenarios.pair_round_corner(), CORNER).pairs
        assert length(pair.centreline) <= pair.before.length
        assert pair.bend is not None and not pair.bend.too_tight
        assert pair.rejected_edges > pair.too_close == 0

    def test_inner_track_runs_the_centreline_s_way(self, corner: RunResult) -> None:
        (pair,) = corner.pairs
        along = np.diff(pair.centreline, axis=0)
        for track in (pair.a, pair.b):
            assert np.all(np.einsum("ij,ij->i", np.diff(track, axis=0), along) > 0)
        assert_gap_kept(pair)
        (given,) = scenarios.pair_corner().pairs
        assert_within_extent(pair, given, CORNER)


@pytest.mark.parametrize(
    "scenario",
    [scenarios.pair_uneven_end(), scenarios.pair_chamfered()],
    ids=lambda s: s.name,
)
def test_ends_keep_both_tracks_directions(scenario: PairsScenario) -> None:
    (given,) = scenario.pairs

    (pair,) = run(scenario).pairs

    assert_meets_its_ends(pair, given)
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
        for result in (*split.chains, *split.pairs):
            assert result.frozen == ()
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


def test_a_pair_turning_back_around_a_pad_keeps_its_legs_apart() -> None:
    scenario = scenarios.pair_turning_back()
    (given,) = scenario.pairs
    apart = pair_block(given, scenarios.TURNING_BACK.spacing).apart

    result = run(scenario, scenarios.TURNING_BACK)

    (pair,) = result.pairs
    assert_gap_kept(pair)
    # Beyond the turn the centreline stays a pair's outer extent plus
    # clearance from itself, as the self check measures it.
    _, closest = close_to_itself(pair.centreline, apart)
    assert np.all(closest >= apart - INSIDE_TOLERANCE)
    # Any moves the run undid were the section's own parts, which do not
    # keep it from settling.
    assert pair.rejected_edges == pair.too_close
    assert result.termination is Termination.CONVERGED


def test_a_section_held_stays_as_it_is() -> None:
    (given,) = scenarios.pair_over_pad().pairs
    pinned = given.b_pinned.copy()
    pinned[2] = True

    result = optimize_chains(
        (), scenarios.PARAMS, COARSE, pairs=(replace(given, b_pinned=pinned),)
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


@pytest.fixture(scope="module")
def held_tent() -> RunResult:
    """`pair_tent` with its section held, from its inputs."""
    return run(held(scenarios.pair_tent()))


class TestResume:
    def test_a_held_section_stays_and_the_others_go_on_from_the_best_state(
        self, tent: RunResult, held_tent: RunResult
    ) -> None:
        scenario = scenarios.pair_tent()
        first = tent

        again = run(held(scenario), start=first)

        (pair,) = again.pairs
        (given,) = scenario.pairs
        np.testing.assert_array_equal(pair.a, given.a.points)
        np.testing.assert_array_equal(pair.b, given.b.points)
        (bar,), (before,) = again.chains, first.chains
        assert again.iterations <= COARSE.stall_window + 1
        assert hausdorff(bar.polyline, before.polyline) < 10 * COARSE.settled_change
        assert not np.array_equal(bar.polyline, held_tent.chains[0].polyline)

    def test_a_start_that_adds_violations_is_not_taken(
        self, tent: RunResult, held_tent: RunResult
    ) -> None:
        scenario = scenarios.pair_tent()
        first = tent
        (bar,) = first.chains
        # Through the arch's tracks, which the bar never reached.
        crossing = bar.polyline.copy()
        crossing[len(crossing) // 2, 1] = 3.0
        start = replace(first, chains=(replace(bar, polyline=crossing),))

        again = run(held(scenario), start=start)

        np.testing.assert_array_equal(
            again.chains[0].polyline, held_tent.chains[0].polyline
        )

    def test_a_start_across_a_held_section_is_not_taken(self, tent: RunResult) -> None:
        scenario = scenarios.pair_tent()
        (bar,) = scenario.chains
        # The bar starts inside this disc, so a start with one violation is
        # no more than its inputs had.
        scenario = replace(
            scenario, chains=(replace(bar, field=Disc(5.0, 1.695, 0.2)),)
        )
        (best,) = tent.chains
        across = best.polyline.copy()
        across[1] = (3.05, 2.95)
        start = replace(tent, chains=(replace(best, polyline=across),))

        again = run(held(scenario), start=start)

        (pair,) = again.pairs
        bar_after = LineString(again.chains[0].polyline)
        for track in (pair.a, pair.b):
            assert LineString(track).distance(bar_after) >= (
                PAIR_WIDTH + PAIR_CLEARANCE - INSIDE_TOLERANCE
            )

    def test_a_section_resumes_from_its_best_centreline(
        self, over_pad: RunResult
    ) -> None:
        scenario = scenarios.pair_over_pad()
        first = over_pad

        again = run(scenario, start=first)

        (pair,), (before,) = again.pairs, first.pairs
        assert again.iterations <= COARSE.stall_window + 1
        assert (
            hausdorff(pair.centreline, before.centreline) < 10 * COARSE.settled_change
        )
        assert_gap_kept(pair)

    def test_a_resumed_run_gives_every_problem_a_burst_before_its_time_limit(
        self,
    ) -> None:
        scenario = scenarios.pair_tent()
        back = replace(
            selected("back", "C", FreeSpace(), (0.0, 0.0), (5.0, 3.0), (10.0, 0.0)),
            layer="B.Cu",
        )
        chains = (*scenario.chains, back)
        # Stopped early, so that the resumed run's problems have work left.
        first = optimize_chains(
            chains,
            scenario.params,
            replace(COARSE, max_iterations=1),
            pairs=scenario.pairs,
        )
        limited = replace(COARSE, time_limit=0.0)

        again = optimize_chains(
            chains,
            scenario.params,
            limited,
            pairs=held(scenario).pairs,
            clock=Ticking(),
            start=first,
        )

        # The first iteration ran whole; the second stopped after its first
        # burst.
        assert again.termination is Termination.TIME_LIMIT
        assert again.iterations == 2
        cut = optimize_chains(
            chains,
            scenario.params,
            limited,
            pairs=held(scenario).pairs,
            clock=Ticking(),
        )
        # Without a start, the back layer never got its burst.
        assert cut.iterations == 1
        np.testing.assert_array_equal(
            cut.chains[1].polyline,
            resample(back.centreline.points, back.fixed, COARSE.spacing)[0],
        )
        assert not np.array_equal(again.chains[1].polyline, first.chains[1].polyline)


SPARSE = replace(COARSE, spacing=3.0, settled_change=0.03)
BAR_Y = 0.7


@pytest.mark.parametrize("height", [1.5, 3.0, 6.0])
def test_a_chain_cannot_jump_through_a_section_in_one_burst(height: float) -> None:
    # The arch of test_optimize_chains' jump test over a short section, which does
    # not move: at this spacing the arch's one free vertex goes straight to
    # its shortest place beyond the section, clear of it.
    arch = selected("arch", "A", FreeSpace(), (0.0, 0.0), (5.0, height), (10.0, 0.0))
    section = scenarios.coupled(FreeSpace(), (4.0, BAR_Y), (6.0, BAR_Y))

    result = optimize_chains([arch], scenarios.PARAMS, SPARSE, pairs=(section,))

    (chain,), (pair,) = result.chains, result.pairs
    for track in (pair.a, pair.b):
        assert y_at(chain.polyline, 5.0) > y_at(track, 5.0)
        assert gap(chain.polyline, track) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE


SECTION_SPAN = 14.0
"""mm: long enough at SPARSE that each end edge gets a direction pin a
spacing in, and the arch between them free vertices."""


BAR_BELOW_APEX = 2.5
"""mm: the bar starts clear of the arched section, and well above the line
between the section's direction pins."""


@pytest.mark.parametrize("height", [6.0, 8.0])
def test_a_section_cannot_jump_through_a_chain_in_one_burst(height: float) -> None:
    # Pulled straight between its direction pins, the arched section would
    # pass the bar and end clear below it.
    apex = SECTION_SPAN / 2
    section = scenarios.coupled(
        FreeSpace(), (0.0, 0.0), (apex, height), (SECTION_SPAN, 0.0)
    )
    bar_y = height - BAR_BELOW_APEX
    bar = selected("bar", "B", FreeSpace(), (apex - 1.0, bar_y), (apex + 1.0, bar_y))

    result = optimize_chains([bar], scenarios.PARAMS, SPARSE, pairs=(section,))

    (chain,), (pair,) = result.chains, result.pairs
    for track in (pair.a, pair.b):
        assert y_at(track, apex) > bar_y
        assert gap(chain.polyline, track) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE


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


def test_a_short_chamfer_at_an_end_gets_no_sliver() -> None:
    (given,) = scenarios.pair_chamfered().pairs

    block = pair_block(given, COARSE.spacing)

    points, fixed = block.chain.centreline.points, block.chain.fixed
    assert len(points) == len(given.a.points)
    assert fixed[[0, 1, -2, -1]].all()


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
chain along its centreline."""


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
    """Move back, in place and as a burst does, what `SelfCheck` finds in
    `candidate`, a centreline with fixed ends; and how many vertex pairs it
    found too close at first."""
    free = np.ones(len(original), dtype=bool)
    free[[0, -1]] = False
    check = SelfCheck.of(original, apart=APART)
    first: list[int] = []

    def crossings(points: FloatArray) -> tuple[int, BoolArray]:
        back, count, close = check(points)
        assert count == close
        if not first:
            first.append(close)
        return count, back & free

    reverted, _ = revert_until_clear(candidate, original, crossings)
    return reverted, first[0]


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


BUMP = (11, 15)
"""Vertices, from and to, lifted off a straight line into a bump."""
BUMP_HEIGHT = 1.0
"""mm: tall enough that the vertices either side of the bump lie more than a
turn apart along it."""


def test_a_pair_beyond_a_turn_only_as_moved_may_stay_as_close_as_it_started() -> None:
    # The vertices either side of the bump do not move, but the bump puts
    # them more than a turn apart along the line: they may stay as close as
    # they started, or moving back could never clear them, and the whole
    # burst would be undone.
    original = np.stack([np.linspace(0.0, 3.0, 31), np.zeros(31)], axis=1)
    candidate = original.copy()
    candidate[slice(*BUMP), 1] = BUMP_HEIGHT

    reverted, too_close = revert(candidate, original)

    assert too_close == 0
    assert not reverted.any()


TILT = 0.3
"""rad by which a whole hairpin turns."""


def test_a_hairpin_turned_whole_keeps_its_parts_as_far_apart() -> None:
    original = hairpin(LEGS)
    c, s = math.cos(TILT), math.sin(TILT)
    candidate = original @ np.array([[c, s], [-s, c]])

    _, too_close = revert(candidate, original)

    assert too_close == 0


def test_a_section_s_parts_apart_keep_their_tracks_at_clearance() -> None:
    # Two parts of one centreline, as across the legs of a U, `apart` from
    # each other: their tracks keep exactly the clearance.
    block = pair_block(
        scenarios.coupled(FreeSpace(), (0.0, 0.0), (5.0, 0.0)), COARSE.spacing
    )
    out = np.array([(0.0, 0.0), (5.0, 0.0)])
    back = out[::-1] + np.array([0.0, block.apart])
    tracks = [
        LineString(track)
        for part in (out, back)
        for track in offset_pair(part, DP_GAP, PAIR_WIDTH)
    ]

    across = min(t.distance(u) for t in tracks[:2] for u in tracks[2:]) - PAIR_WIDTH

    assert across == pytest.approx(PAIR_CLEARANCE)


REJECTED = 5
STILL_ITERATIONS = COARSE.stall_window - 1
"""Too few bursts for the best energy to stall."""


@pytest.mark.parametrize(
    ("too_close", "settles"), [(REJECTED, True), (REJECTED - 1, False)]
)
def test_a_problem_settles_on_rejections_only_when_they_are_a_section_s_own_parts(
    monkeypatch: pytest.MonkeyPatch, too_close: int, settles: bool
) -> None:
    def still(
        self: optimize_module._Problem,  # pyright: ignore[reportPrivateUsage]
        starts: list[FloatArray],
    ) -> optimize_module._ProblemBurst:  # pyright: ignore[reportPrivateUsage]
        count = len(starts)
        return optimize_module._ProblemBurst(  # pyright: ignore[reportPrivateUsage]
            list(starts),
            list(self.positions),
            [REJECTED] * count,
            [too_close] * count,
            [0] * len(self.vias),
            False,
        )

    monkeypatch.setattr(optimize_module._Problem, "burst", still)  # pyright: ignore[reportPrivateUsage]
    settings = replace(COARSE, max_iterations=STILL_ITERATIONS)

    result = run(scenarios.pair_corner(), settings)

    if settles:
        assert result.termination is Termination.NOTHING_TO_DO
    else:
        assert result.termination is Termination.ITERATION_CAP


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

    layout = lay_out(
        whole, find_pairs(chains, DEFAULT_CHORD_TOLERANCE, frame=BOARD_FRAME), free
    )

    result = optimize_chains(
        layout.inputs, scenarios.PARAMS, COARSE, pairs=layout.sections
    )

    assert layout.sections
    for given, chain in zip(layout.inputs, result.chains, strict=True):
        for end in (0, -1):
            start = tuple(given.centreline.points[end])
            if start in at.values():
                assert tuple(chain.polyline[end]) == start
