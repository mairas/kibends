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
from distance_fields import BoolArray, FloatArray, FreeSpace
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
from kibends.bursts import (
    DEFAULT_CHORD_TOLERANCE,
    INSIDE_TOLERANCE,
    Burst,
    Objective,
    OptimizerSettings,
    Revert,
    Stop,
    run_burst,
)
from kibends.chain import Chain
from kibends.contacts import no_contacts
from kibends.coords import BOARD_FRAME
from kibends.energy import (
    EnergyParams,
    ObstacleField,
    TermValue,
    arc_lengths,
    total_energy,
)
from kibends.members import Member, layer_energy, seen_parts
from kibends.neighbours import ChainInput, chain_field, neighbour_field
from kibends.obstacles import ChainCentreline
from kibends.optimize import optimize_chains
from kibends.pair_blocks import Hold, PairBlock, PairInput, pair_block
from kibends.pair_layout import cut_parts, find_pairs, lay_out
from kibends.pairs import (
    GAP_TOLERANCE,
    bend_radius_limit,
    check_bend_radius,
    gap_range,
    offset_pair,
)
from kibends.polylines import SelfCheck, resample
from kibends.results import ChainResult, PairResult, RunResult, Termination
from kibends.simplify import simplify_chains

COARSE = OptimizerSettings(spacing=0.3, settled_change=0.003, burst_maxiter=30)
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
            static, limit = member.chain.field, member.limit
        else:
            near = neighbour_field(member, others)
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
        def unhinged(
            points: FloatArray,
            fixed: BoolArray,
            limit: float,
            params: EnergyParams,
        ) -> TermValue:
            return 0.0, np.zeros_like(points)

        monkeypatch.setattr(members_module, "bend_radius_energy", unhinged)
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
        # L-BFGS-B's first trial of the first burst folds the centreline
        # at its second free vertex.
        trials: list[FloatArray] = []

        def folding_first(
            objective: Objective,
            start: FloatArray,
            revert: Revert,
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


def test_a_repeated_run_gives_every_problem_a_burst_before_its_time_limit() -> None:
    scenario = scenarios.pair_tent()
    back = replace(
        selected("back", "C", FreeSpace(), (0.0, 0.0), (5.0, 3.0), (10.0, 0.0)),
        layer="B.Cu",
    )
    chains = (*scenario.chains, back)
    limited = replace(COARSE, time_limit=0.0)
    resampled = resample(back.centreline.points, back.fixed, COARSE.spacing)[0]

    def repeated(again: bool) -> RunResult:
        return optimize_chains(
            chains,
            scenario.params,
            limited,
            pairs=held(scenario).pairs,
            clock=Ticking(),
            again=again,
        )

    # Repeated, the first iteration runs whole, so the back layer gets its
    # burst; a first run stops after the first problem's.
    assert not np.array_equal(repeated(again=True).chains[1].polyline, resampled)
    cut = repeated(again=False)
    assert cut.iterations == 1
    np.testing.assert_array_equal(cut.chains[1].polyline, resampled)


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

    def smooth_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        self.points += len(points)
        return self.field.smooth_distance(points, reach)

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


REJECTED = 5
STILL_ITERATIONS = COARSE.stall_window - 1
"""Too few bursts for the best energy to stall."""


def test_a_problem_whose_bursts_move_back_does_not_settle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def still(
        self: optimize_module._Problem,  # pyright: ignore[reportPrivateUsage]
        starts: list[FloatArray],
        stop: Stop | None = None,
    ) -> optimize_module._ProblemBurst:  # pyright: ignore[reportPrivateUsage]
        count = len(starts)
        return optimize_module._ProblemBurst(  # pyright: ignore[reportPrivateUsage]
            list(starts),
            list(self.positions),
            [REJECTED] * count,
            [0] * len(self.vias),
            False,
        )

    monkeypatch.setattr(optimize_module._Problem, "burst", still)  # pyright: ignore[reportPrivateUsage]
    settings = replace(COARSE, max_iterations=STILL_ITERATIONS)

    result = run(scenarios.pair_corner(), settings)

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

    parts = cut_parts(chains, DEFAULT_CHORD_TOLERANCE)
    pairs = find_pairs(
        [p.chain for p in parts], DEFAULT_CHORD_TOLERANCE, frame=BOARD_FRAME
    )
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
    parts = cut_parts(chains, DEFAULT_CHORD_TOLERANCE)
    pairs = find_pairs(
        [p.chain for p in parts], DEFAULT_CHORD_TOLERANCE, frame=BOARD_FRAME
    )
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
DEFAULT_SPACING = replace(
    COARSE, spacing=OptimizerSettings().spacing, settled_change=0.002, max_iterations=4
)
"""The default spacing, at which a section's direction pins and its starting
bend are those of a real run, with short bursts to run fast."""
COARSE_CAPPED = replace(COARSE, max_iterations=12)
"""`COARSE`, stopped before rounding a corner close to a pin's direction
pin settles fully, about 35 iterations, to run fast."""
SPACINGS = [
    pytest.param(COARSE_CAPPED, id="coarse"),
    pytest.param(DEFAULT_SPACING, id="default"),
]
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
    """Smooth `chains`: every pin stays, the joined tracks keep the pair's
    gap, so they neither narrow nor overlap, and no section is held after
    the run for its bend, nor ends it bending too tightly."""
    out = smoothed(chains, settings)

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
