"""Coupled sections of differential pairs through
`optimize.optimize_chains`.

The problem's energy is checked against a reference built member by member
(`reference_energy`), and the offset pullback against finite
differences."""

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
    DP_PAD,
    PAIR_CLEARANCE,
    PAIR_WIDTH,
    PairsScenario,
    selected,
)
from shapely.geometry import LineString, Point

import kibends.optimize as optimize_module
from kibends.bursts import (
    Burst,
    Objective,
    OptimizerSettings,
    Revert,
    revert_until_clear,
    run_burst,
)
from kibends.chain import Chain
from kibends.coords import BOARD_FRAME
from kibends.energy import (
    EnergyParams,
    ObstacleField,
    TermValue,
    arc_lengths,
    obstacle_reach,
    total_energy,
)
from kibends.neighbours import ChainInput, chain_field, neighbour_field
from kibends.obstacles import ChainCentreline, ChainField
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
    no_contacts,
    optimize_chains,
    resample,
    seen_parts,
)
from kibends.pair_blocks import Hold, PairBlock, PairInput, pair_block
from kibends.pair_layout import find_pairs, lay_out
from kibends.pairs import (
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

    @pytest.mark.parametrize("net", ["P", "N"])
    def test_a_chain_of_one_of_the_pair_s_nets_repels_the_whole_section(
        self, net: str
    ) -> None:
        # The section cannot tell which track faces the bar, so it keeps its
        # whole extent clear of it, its own net's track included; the bar
        # keeps clear of the other net's track alone.
        bar = selected("bar", net, FreeSpace(), (4.0, 2.2), (5.0, 2.4), (6.0, 2.2))
        scenario = replace(scenarios.pair_tent(), chains=(bar,))
        members, polylines, masks = self.state(scenario)
        block = members[1]
        assert isinstance(block, PairBlock)
        params = self.params()
        points, mask = polylines[1], masks[1]
        bar_points = polylines[0]

        value, gradients = layer_energy(members, polylines, masks, params, 0.01)

        def edges(line: FloatArray, inflation: float) -> ChainField:
            return ChainField(line[:-1], line[1:], np.full(len(line) - 1, inflation))

        facing = block.facing.centreline
        seeing_bar = chain_field(
            block.chain.field,
            [edges(bar_points, (facing.width + PAIR_WIDTH) / 2 + PAIR_CLEARANCE)],
        )
        section = total_energy(points, mask, seeing_bar, params, 0.01, block.limit)[0]
        (other,) = [
            line
            for track, line in zip(block.tracks, block.offset(points), strict=True)
            if track.centreline.net != net
        ]
        seeing_track = chain_field(
            bar.field, [edges(other, PAIR_WIDTH + PAIR_CLEARANCE)]
        )
        own = total_energy(bar_points, bar.fixed, seeing_track, params, 0.01)[0]
        alone = total_energy(
            points, mask, block.chain.field, params, 0.01, block.limit
        )[0]
        assert section > alone
        assert value == pytest.approx(section + own, rel=1e-12)
        apart = layer_energy(members[1:], polylines[1:], masks[1:], params, 0.01)[1]
        assert not np.allclose(apart[0], gradients[1])


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
        result = ChainResult(
            pair.centreline,
            pair.fixed,
            0,
            (),
            pair.before,
            no_contacts(pair.centreline),
        )

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
        # L-BFGS-B's first trial of the first burst folds the centreline
        # at its second free vertex.
        trials: list[FloatArray] = []

        def folding_first(
            objective: Objective,
            start: FloatArray,
            revert: Revert,
            settings: OptimizerSettings,
        ) -> Burst:
            if not trials:
                trials.append(folded(start.reshape(-1, 2), 0).ravel())
                objective(trials[0])
            return run_burst(objective, start, revert, settings)

        monkeypatch.setattr(optimize_module, "run_burst", folding_first)
        (pair,) = run(scenarios.pair_corner(), CORNER).pairs

        assert trials
        assert pair.bend is not None and not pair.bend.too_tight

    def test_the_self_check_moves_a_fold_back(self) -> None:
        straight = np.array([(float(x), 0.0) for x in range(6)])
        check = SelfCheck.of(straight, bend_radius_limit(DP_GAP, PAIR_WIDTH))

        back, count, _ = check(folded(straight, 1))

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
    # Moves the run undid do not keep it from settling.
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


def test_a_chain_starting_inside_a_section_s_clearance_moves_clear_of_it() -> None:
    section = scenarios.coupled(FreeSpace(), (0.0, 0.0), (5.0, 0.0))
    half_extent = pair_block(section, COARSE.spacing).facing.centreline.width / 2
    # The chain runs 2δ inside the clearance of the section's nearer track,
    # pinned there at both ends.
    y = half_extent + PAIR_CLEARANCE + PAIR_WIDTH / 2 - 2 * COARSE.frozen_depth
    chain = selected("C", "C", FreeSpace(), (0.0, y), (5.0, y))

    result = optimize_chains([chain], scenarios.PARAMS, COARSE, pairs=(section,))

    (moved,), (pair,) = result.chains, result.pairs
    middle = LineString(moved.polyline).interpolate(0.5, normalized=True)
    for track in (pair.a, pair.b):
        assert LineString(track).distance(middle) >= (
            scenarios.PAIR_INFLATION - INSIDE_TOLERANCE
        )


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
differently and stop where L-BFGS-B's ftol ends them, about 1 µm apart."""


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
    pairs = find_pairs(chains, DEFAULT_CHORD_TOLERANCE, frame=BOARD_FRAME)
    layout = lay_out(whole, pairs, free_section)
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


SHORT_PART = ((0.0, -5.0), (0.2, -5.0), (0.4, -4.8))
"""A section's centreline whose end edges are shorter than the direction
pins' room, so that every vertex is an end or a direction pin, none more
than a spacing apart; clear of `pair_corner`."""


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
DEFAULT_SPACING = replace(
    COARSE, spacing=OptimizerSettings().spacing, settled_change=0.002, max_iterations=4
)
"""The default spacing, at which a section's direction pins and its starting
bend are those of a real run, with short bursts to run fast."""
SPACINGS = [
    pytest.param(COARSE, id="coarse"),
    pytest.param(DEFAULT_SPACING, id="default"),
]
SPLIT_HOLDS = {None, Hold.COPPER, Hold.SHORT}
"""How a section split at pins, or left whole around one, may be held: as
routed, never for its bend."""
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
    """Smooth `chains`: every pin stays, the joined tracks keep the pair's
    gap, so they neither narrow nor overlap, and no section is held for
    its bend."""
    out = smoothed(chains, settings)

    assert_pins_kept(chains, out.joined)
    p, n = out.joined
    low, high = gap_range(p, n, PAIR_WIDTH)
    slack = GAP_TOLERANCE * gap
    assert gap - slack <= low <= high <= gap + slack
    assert {p.block.hold for p in out.result.pairs} <= SPLIT_HOLDS
    return out


@pytest.mark.parametrize("settings", SPACINGS)
@pytest.mark.parametrize("offset", NEAR_CORNER)
@pytest.mark.parametrize("corner", CORNERS)
@pytest.mark.parametrize("pinned", ["P", "N"])
def test_a_pin_near_a_corner_of_either_track_holds_its_section(
    pinned: str, corner: int, offset: float, settings: OptimizerSettings
) -> None:
    tracks = scenarios.tracks(*PINNED_PAIR)
    track = tracks[0 if pinned == "P" else 1]
    chains = pinned_near(tracks, pinned, [corner_arc(track, corner) + offset])

    out = assert_kept_or_held(chains, settings)

    assert [p.block.hold for p in out.result.pairs] == [Hold.COPPER]


@pytest.mark.parametrize("settings", SPACINGS)
@pytest.mark.parametrize("offset", [0.0, 0.1, -0.1])
@pytest.mark.parametrize("corner", CORNERS)
def test_a_pad_across_both_tracks_near_a_corner_holds_its_section(
    corner: int, offset: float, settings: OptimizerSettings
) -> None:
    # At 0 the pad touches both tracks' corner vertices, across the bend.
    p, n = scenarios.tracks(*PINNED_PAIR)
    chains = [
        pinned_along("USB_D+", p, corner_arc(p, corner) + offset),
        pinned_along("USB_D-", n, corner_arc(n, corner) + offset),
    ]

    out = assert_kept_or_held(chains, settings)

    assert [p.block.hold for p in out.result.pairs] == [Hold.COPPER]


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

    assert [p.block.hold for p in out.result.pairs] == [None, None]
    assert_corners_round(chains, out.joined)


def test_a_pin_across_from_the_other_tracks_corner_holds_its_section() -> None:
    out = assert_kept_or_held(pinned_pair((ACROSS_N_CORNER,), ()), COARSE)

    assert [p.block.hold for p in out.result.pairs] == [Hold.COPPER]


@pytest.mark.parametrize("offset", [0.0, 0.02, -0.02, 0.1, -0.1])
@pytest.mark.parametrize("pinned", ["P", "N"])
def test_a_pin_at_a_right_angle_corner_holds_its_section(
    pinned: str, offset: float
) -> None:
    tracks = scenarios.tracks(*RIGHT_ANGLE_PAIR)
    track = tracks[0 if pinned == "P" else 1]
    chains = pinned_near(tracks, pinned, [corner_arc(track, 1) + offset])

    out = assert_kept_or_held(chains, DEFAULT_SPACING)

    assert [p.block.hold for p in out.result.pairs] == [Hold.COPPER]


@pytest.mark.parametrize("pinned", ["P", "N"])
def test_a_pin_on_a_straight_lets_a_right_angle_corner_round(pinned: str) -> None:
    tracks = scenarios.tracks(*RIGHT_ANGLE_PAIR)
    track = tracks[0 if pinned == "P" else 1]
    chains = pinned_near(tracks, pinned, [corner_arc(track, 1) - 2.0])

    out = assert_kept_or_held(chains, DEFAULT_SPACING)

    assert [p.block.hold for p in out.result.pairs] == [None, None]
    for chain, line in zip(chains, out.joined, strict=True):
        corner = Point(np.array(chain.points)[-2])
        assert LineString(line).distance(corner) >= ROUNDED


@pytest.mark.parametrize(
    ("offsets", "holds"),
    [((0.1, 0.3), [Hold.COPPER]), ((0.3, 1.5), [Hold.COPPER, None])],
)
def test_pins_past_an_outer_corner_hold_only_the_section_around_them(
    offsets: tuple[float, float], holds: list[Hold | None]
) -> None:
    # N is outside corner 1: a pad's two outline crossings past it.
    tracks = scenarios.tracks(*PINNED_PAIR)
    at = corner_arc(tracks[1], 1)
    chains = pinned_near(tracks, "N", [at + offset for offset in offsets])

    out = assert_kept_or_held(chains, DEFAULT_SPACING)

    assert [p.block.hold for p in out.result.pairs] == holds


@pytest.mark.parametrize("gap", [0.3, 0.4, 0.5])
def test_a_pin_at_the_outer_corner_of_a_wider_pair_holds_its_section(
    gap: float,
) -> None:
    p, n = offset_pair(scenarios.path(*PINNED_PAIR), gap, PAIR_WIDTH)
    # N is outside corner 1.
    chains = pinned_near((p, n), "N", [corner_arc(n, 1)])

    out = assert_kept_or_held(chains, DEFAULT_SPACING, gap)

    assert [p.block.hold for p in out.result.pairs] == [Hold.COPPER]


def test_a_vertex_that_does_not_turn_does_not_hide_a_corner() -> None:
    # P's first straight drawn in two clicks, between the pin and its
    # corner.
    p, n = scenarios.tracks(*PINNED_PAIR)
    corner = corner_arc(p, 1)
    clicked = pinned_along("USB_D+", p, corner - 0.15, corner - 0.3)
    pins = list(clicked.pinned)
    pins[2] = False
    chains = [
        replace(clicked, pinned=tuple(pins)),
        pinned_along("USB_D-", n),
    ]

    out = assert_kept_or_held(chains, DEFAULT_SPACING)

    assert [p.block.hold for p in out.result.pairs] == [Hold.COPPER]
