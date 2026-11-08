import copy
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
import pytest
import scenarios
from conftest import BurstClock, Ticking
from distance_fields import BoolArray, FloatArray, FreeSpace
from scenarios import ChainsScenario, selected
from shapely.geometry import LineString

import kibends.optimize as optimize_module
from kibends.bursts import OptimizerSettings
from kibends.energy import EnergyParams, ObstacleField, obstacle_reach, total_energy
from kibends.neighbours import (
    ChainInput,
    chain_field,
    neighbour_field,
    same_layer_neighbours,
)
from kibends.obstacles import ChainCentreline, CompositeField, chain_obstacles
from kibends.optimize import (
    INSIDE_TOLERANCE,
    MAX_ABNORMAL_BURSTS,
    PROGRESS_INTERVAL,
    PolylineStats,
    RunResult,
    Termination,
    measure,
    optimize,
    optimize_chains,
)

COARSE = OptimizerSettings(spacing=0.3, settled_change=0.003, burst_maxiter=30)


def run(scenario: ChainsScenario, settings: OptimizerSettings = COARSE) -> RunResult:
    return optimize_chains(scenario.chains, scenario.params, settings)


def gap(a: FloatArray, b: FloatArray) -> float:
    """Smallest distance between two polylines' edges, mm."""
    return float(LineString(a).distance(LineString(b)))


def y_at(points: FloatArray, x: float) -> float:
    return float(np.interp(x, points[:, 0], points[:, 1]))


def after(
    chains: Sequence[ChainInput],
    result: RunResult,
    params: EnergyParams,
    settings: OptimizerSettings = COARSE,
) -> list[PolylineStats]:
    """Each chain's statistics with the other chains at their returned shapes."""
    polylines = [c.polyline for c in result.chains]
    reach = obstacle_reach(params)
    neighbours = same_layer_neighbours(chains)
    return [
        measure(
            polylines[i],
            chain_field(
                chain.field,
                (
                    neighbour_field(
                        chain, [(chains[j], polylines[j])], reach, polylines[i]
                    )
                    for j in neighbours[i]
                ),
            ),
            settings.spacing,
        )
        for i, chain in enumerate(chains)
    ]


def run_energy(
    chains: Sequence[ChainInput],
    polylines: Sequence[FloatArray],
    scenario: ChainsScenario,
) -> float:
    """Each chain's energy in its static field plus the others as obstacles."""
    total = 0.0
    for i, chain in enumerate(chains):
        c = chain.centreline
        others = [
            replace(o.centreline, points=polylines[j])
            for j, o in enumerate(chains)
            if j != i
        ]
        field = CompositeField(
            [chain.field, chain_obstacles(c.net, c.width, c.clearance, others)]
        )
        fixed = np.zeros(len(polylines[i]), dtype=bool)
        total += total_energy(polylines[i], fixed, field, scenario.params)[0]
    return total


@pytest.fixture(scope="module")
def channel_pair() -> RunResult:
    return run(scenarios.channel_pair())


class TestChannelPair:
    def test_converges(self, channel_pair: RunResult) -> None:
        assert channel_pair.termination is Termination.CONVERGED

    def test_chains_share_the_channel_with_equal_margins(
        self, channel_pair: RunResult
    ) -> None:
        upper, lower = (c.polyline for c in channel_pair.chains)
        for x in (8.0, 10.0, 12.0):
            a, b = y_at(upper, x), y_at(lower, x)
            to_wall = scenarios.CHANNEL_HALF_GAP - a, scenarios.CHANNEL_HALF_GAP + b
            between = a - b - scenarios.PAIR_INFLATION
            assert to_wall[0] == pytest.approx(between, abs=0.05)
            assert to_wall[1] == pytest.approx(between, abs=0.05)

    def test_chains_never_come_closer_than_their_clearance(
        self, channel_pair: RunResult
    ) -> None:
        upper, lower = (c.polyline for c in channel_pair.chains)
        assert gap(upper, lower) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE


# Both visiting orders give final total lengths this close, mm. Measured:
# 2.5 µm at these settings, 0.5 µm at the defaults. A symmetric pair would
# show no difference at all, as its swapped run is its mirror image.
ORDER_TOLERANCE = 5e-3


def test_visiting_order_barely_changes_the_result() -> None:
    scenario = scenarios.uneven_channel_pair()

    lengths = [
        sum(
            stats.length
            for stats in after(
                chains,
                optimize_chains(chains, scenario.params, COARSE),
                scenario.params,
            )
        )
        for chains in (scenario.chains, scenario.chains[::-1])
    ]

    assert lengths[1] == pytest.approx(lengths[0], abs=ORDER_TOLERANCE)


def test_arch_stays_above_the_bar_it_cannot_cross() -> None:
    scenario = scenarios.tent()
    result = run(scenario)

    arch, bar = (c.polyline for c in result.chains)
    assert result.termination is Termination.CONVERGED
    assert gap(arch, bar) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE
    assert y_at(arch, 5.0) > y_at(bar, 5.0)
    arch_after = after(scenario.chains, result, scenario.params)[0]
    assert arch_after.length < result.chains[0].before.length


def test_chains_of_one_net_are_pulled_together_without_overlapping() -> None:
    result = run(scenarios.same_net_wrap())

    inner, outer = (c.polyline for c in result.chains)
    assert gap(inner, outer) >= scenarios.SAME_NET_INFLATION - INSIDE_TOLERANCE
    assert gap(inner, outer) < scenarios.SAME_NET_INFLATION + 0.05


def test_one_chain_gives_the_single_chain_result() -> None:
    wrap = scenarios.wrap()
    chain = selected("wrap", "N", wrap.field, (0.0, 0.0), (10.0, 0.0), (10.0, 10.0))

    alone = optimize(wrap.polyline, wrap.fixed, wrap.field, wrap.params, COARSE)
    result = optimize_chains([chain], wrap.params, COARSE)

    (only,) = result.chains
    assert np.array_equal(only.polyline, alone.polyline)
    assert np.array_equal(only.fixed, alone.fixed)
    assert result.termination is alone.termination
    assert result.iterations == alone.iterations
    assert only.before == alone.before


def test_vertices_inside_a_neighbours_clearance_are_frozen_and_named() -> None:
    result = run(scenarios.crowded())

    a, b = result.chains
    assert [(v.position, v.neighbours) for v in a.frozen] == [
        (scenarios.CROWDED_VERTEX, ("B",))
    ]
    assert [(v.position, v.neighbours) for v in b.frozen] == [
        (scenarios.CROWDED_NEIGHBOUR_VERTEX, ("A",))
    ]
    assert np.all(a.polyline == scenarios.CROWDED_VERTEX, axis=1).sum() == 1
    assert np.all(b.polyline == scenarios.CROWDED_NEIGHBOUR_VERTEX, axis=1).sum() == 1


def test_chains_inside_each_others_clearance_throughout_stay_put() -> None:
    # Every vertex resampling adds lies deeper than δ inside the other
    # chain's clearance, so every bound of a burst is zero.
    apart = scenarios.PAIR_INFLATION - 2 * COARSE.bubble_tolerance
    a = selected("A", "A", FreeSpace(), (0.0, 0.0), (5.0, 0.0))
    b = selected("B", "B", FreeSpace(), (0.0, apart), (5.0, apart))

    result = optimize_chains([a, b], scenarios.PARAMS, COARSE)

    for given, chain in zip((a, b), result.chains, strict=True):
        assert LineString(chain.polyline).equals(LineString(given.centreline.points))
        assert chain.abnormal_bursts == 0


def test_a_chain_whose_bursts_all_fail_does_not_fail_the_run(
    monkeypatch: pytest.MonkeyPatch, log: list[str]
) -> None:
    misled = selected("misled", "A", FreeSpace(), (0.0, 0.0), (5.0, 0.8), (10.0, 0.0))
    calm = selected("calm", "B", FreeSpace(), (0.0, 5.0), (5.0, 6.0), (10.0, 5.0))
    params = replace(scenarios.PARAMS, repulsion=5.0)
    # How often L-BFGS-B ends abnormally on a misleading field depends on the
    # platform's scipy build, so misled's bursts are failed outright.
    watch_bursts(
        monkeypatch,
        lambda start, burst: abnormal(burst) if start == (0.0, 0.0) else burst,
    )

    result = optimize_chains((calm, misled), params, COARSE)

    calm_after, misled_after = result.chains
    assert result.termination is Termination.CONVERGED
    assert calm_after.abnormal_bursts == 0
    calm_stats = after((calm, misled), result, params)[0]
    assert calm_stats.length < calm_after.before.length
    assert misled_after.abnormal_bursts >= MAX_ABNORMAL_BURSTS
    assert result.gave_up == (("misled", MAX_ABNORMAL_BURSTS),)


@pytest.mark.parametrize(
    "scenario", [scenarios.tent(), scenarios.channel_pair()], ids=lambda s: s.name
)
def test_capped_runs_return_one_state_that_never_gets_worse(
    scenario: ChainsScenario,
) -> None:
    energies: list[float] = []
    for cap in range(1, 5):
        result = run(scenario, replace(COARSE, max_iterations=cap))
        polylines = [c.polyline for c in result.chains]

        assert gap(*polylines) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE
        energies.append(run_energy(scenario.chains, polylines, scenario))
    assert energies == sorted(energies, reverse=True)
    assert energies[-1] < energies[0]


# At spacings 0.25 and 0.3 the drape's first burst once piled two vertices of
# the long arch onto each other at their bubble bounds, 17 µm apart, and
# L-BFGS-B failed on the bending gradient across the edge between them.
@pytest.mark.parametrize(
    ("scenario", "settings"),
    [
        (scenarios.drape(), replace(COARSE, spacing=0.25, settled_change=0.0025)),
        (scenarios.drape(), COARSE),
        (scenarios.squeeze(), COARSE),
        (scenarios.hooks(), COARSE),
    ],
    ids=["drape-0.25", "drape-0.3", "squeeze", "hooks"],
)
def test_moving_chains_tighten_without_crossing(
    scenario: ChainsScenario, settings: OptimizerSettings
) -> None:
    result = run(scenario, settings)

    a, b = (c.polyline for c in result.chains)
    assert result.termination is Termination.CONVERGED
    assert not LineString(a).intersects(LineString(b))
    assert gap(a, b) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE
    for chain, stats in zip(
        result.chains,
        after(scenario.chains, result, scenario.params, settings),
        strict=True,
    ):
        assert stats.length < chain.before.length
        assert chain.abnormal_bursts == 0


# The neck-down's lengths alone and together agree to within this, mm.
JUNCTION_TOLERANCE = 1e-3
JUNCTION_X = 10.0
LEG = 10.0
BOW = 0.5
JUNCTION_SETTINGS = OptimizerSettings(
    spacing=0.2, settled_change=0.002, burst_maxiter=60
)


def test_chains_of_one_net_meeting_end_to_end_leave_the_junction_straight() -> None:
    scenario = scenarios.neck_down()

    result = run(scenario)

    stats = after(scenario.chains, result, scenario.params)
    for chain, given in zip(stats, scenario.chains, strict=True):
        alone = optimize(
            given.centreline.points, given.fixed, given.field, scenario.params, COARSE
        )
        assert chain.length == pytest.approx(alone.after.length, abs=JUNCTION_TOLERANCE)
        assert chain.min_clearance >= 0.0


def test_chains_on_different_layers_do_not_interact() -> None:
    free = FreeSpace()
    across = selected("A", "A", free, (0.0, 0.0), (5.0, 2.0), (10.0, 0.0))
    along = replace(
        selected("B", "B", free, (5.0, -2.0), (7.0, 1.0), (5.0, 4.0)), layer="B.Cu"
    )

    result = optimize_chains([across, along], scenarios.PARAMS, COARSE)

    for chain, given in zip(result.chains, (across, along), strict=True):
        alone = optimize(
            given.centreline.points, given.fixed, free, scenarios.PARAMS, COARSE
        )
        assert np.array_equal(chain.polyline, alone.polyline)
    a, b = (c.polyline for c in result.chains)
    assert LineString(a).intersects(LineString(b))


BurstStart = tuple[float, float]


def watch_bursts(
    monkeypatch: pytest.MonkeyPatch,
    react: Callable[[BurstStart, object], object] = lambda _, burst: burst,
) -> list[BurstStart]:
    """Record each burst by its chain's first point; react may replace its result."""
    real = vars(optimize_module)["_burst"]
    starts: list[BurstStart] = []

    def watched(
        points: FloatArray,
        fixed: BoolArray,
        field: ObstacleField,
        params: EnergyParams,
        settings: OptimizerSettings,
        bend_limit: float | None = None,
        apart: float | None = None,
    ) -> object:
        start = (float(points[0, 0]), float(points[0, 1]))
        starts.append(start)
        burst = real(points, fixed, field, params, settings, bend_limit, apart)
        return react(start, burst)

    monkeypatch.setattr(optimize_module, "_burst", watched)
    return starts


def test_a_settled_chain_rests_while_its_neighbours_stand_still(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = scenarios.uneven_channel_pair()
    quick = replace(
        selected("quick", "Q", FreeSpace(), (0.0, 10.0), (1.0, 10.3), (2.0, 10.0)),
        layer="B.Cu",
    )
    alone = optimize(
        quick.centreline.points, quick.fixed, quick.field, pair.params, COARSE
    )
    starts = watch_bursts(monkeypatch)

    result = optimize_chains([*pair.chains, quick], pair.params, COARSE)

    assert result.termination is Termination.CONVERGED
    assert result.iterations > alone.iterations
    assert starts.count((0.0, 10.0)) == alone.iterations


def test_abnormal_bursts_of_a_woken_chain_are_counted_and_taken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The bar settles in its first burst, before the arch comes near; the
    # arch's repulsion then wakes it, and every later burst of it fails.
    tent = scenarios.tent()
    bar_start = (4.0, scenarios.TENT_HEIGHT)

    def fail(start: BurstStart, burst: object) -> object:
        if start != bar_start or bar_start not in starts[:-1]:
            return burst
        failed.append(start)
        return abnormal(burst)

    failed: list[BurstStart] = []
    starts = watch_bursts(monkeypatch, fail)

    result = optimize_chains(
        tent.chains[::-1], replace(tent.params, repulsion=0.5), COARSE
    )

    bar = result.chains[0]
    assert failed
    assert result.termination is Termination.CONVERGED
    assert bar.abnormal_bursts == len(failed)
    # The arch's push bends the bar: the failed bursts were not discarded.
    assert np.abs(bar.polyline[:, 1] - scenarios.TENT_HEIGHT).max() > 1e-6


def abnormal(burst: object) -> object:
    """The burst as if L-BFGS-B had ended it abnormally (status 2)."""
    failed = copy.copy(burst)
    object.__setattr__(failed, "abnormal", True)
    return failed


def test_an_abnormal_burst_is_taken_like_any_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = scenarios.wrap()
    clean = optimize(
        scenario.polyline, scenario.fixed, scenario.field, scenario.params, COARSE
    )
    first: list[bool] = []

    def fail_first(_: BurstStart, burst: object) -> object:
        first.append(not first)
        return abnormal(burst) if first[-1] else burst

    watch_bursts(monkeypatch, fail_first)

    result = optimize(
        scenario.polyline, scenario.fixed, scenario.field, scenario.params, COARSE
    )

    assert result.termination is Termination.CONVERGED
    assert result.abnormal_bursts == 1
    assert np.array_equal(result.polyline, clean.polyline)


def test_a_chain_stops_after_too_many_abnormal_bursts_in_a_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = scenarios.wrap()
    starts = watch_bursts(monkeypatch, lambda _, burst: abnormal(burst))

    result = optimize(
        scenario.polyline, scenario.fixed, scenario.field, scenario.params, COARSE
    )

    assert result.termination is Termination.CONVERGED
    assert len(starts) == result.iterations == MAX_ABNORMAL_BURSTS
    assert result.abnormal_bursts == MAX_ABNORMAL_BURSTS
    assert result.after.length < result.before.length


def test_a_woken_chain_does_not_stall_on_energies_from_before_it_woke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One chain stalls, the other's moves wake it, and its next burst used to
    # stall it again at once: its history still held the lows from before its
    # energy was raised, so a burst cutting it by 0.2 counted as no progress.
    real = optimize_module.stalled
    stalled_on: list[list[float]] = []

    def watched(lowest_energies: list[float], settings: OptimizerSettings) -> bool:
        assert not any(lowest_energies is history for history in stalled_on)
        if real(lowest_energies, settings):
            stalled_on.append(lowest_energies)
            return True
        return False

    monkeypatch.setattr(optimize_module, "stalled", watched)
    pair = scenarios.uneven_channel_pair()

    result = run(
        pair, OptimizerSettings(spacing=0.2, settled_change=0.002, burst_maxiter=60)
    )

    assert stalled_on
    assert result.termination is Termination.CONVERGED


@pytest.mark.parametrize("angle", [30.0, 10.0])
def test_chains_of_one_net_meeting_at_an_acute_angle_keep_clear(angle: float) -> None:
    # Near the junction each leaves out the other only as far as the other
    # lies within its clearance; at an acute angle that is farther than at a
    # straight neck-down.
    turn = math.radians(angle)
    far = (JUNCTION_X - LEG * math.cos(turn), LEG * math.sin(turn))
    bow = (
        (JUNCTION_X + far[0]) / 2 + BOW * math.sin(turn),
        far[1] / 2 + BOW * math.cos(turn),
    )
    junction = (JUNCTION_X, 0.0)
    a = selected("A", "N", FreeSpace(), (0.0, 0.0), (5.0, -BOW), junction)
    b = selected("B", "N", FreeSpace(), junction, bow, far)

    result = optimize_chains([a, b], scenarios.PARAMS, JUNCTION_SETTINGS)

    assert result.termination is Termination.CONVERGED
    for chain, stats in zip(
        result.chains,
        after([a, b], result, scenarios.PARAMS, JUNCTION_SETTINGS),
        strict=True,
    ):
        assert stats.min_clearance >= -INSIDE_TOLERANCE
        assert stats.length < chain.before.length


TOUCH = (5.0, 0.0)
STUB_TIP = (5.0, -0.1)


def test_chains_of_one_net_touching_mid_chain_do_not_repel() -> None:
    # The stub's end lies in the bar's copper at its pinned vertex TOUCH, and
    # each starts with vertices next to the touch, inside the other's copper.
    bar_points = np.array(
        [
            (0.0, 0.0),
            (2.5, 0.5),
            (4.95, 0.0),
            TOUCH,
            (5.05, 0.0),
            (7.5, 0.5),
            (10.0, 0.0),
        ]
    )
    bar_fixed = np.array([True, False, False, True, False, False, True])
    bar = ChainInput(
        ChainCentreline(
            "bar", "N", scenarios.PAIR_WIDTH, scenarios.PAIR_CLEARANCE, bar_points
        ),
        "F.Cu",
        bar_fixed,
        FreeSpace(),
    )
    stub = selected(
        "stub", "N", FreeSpace(), (5.0, -3.0), (5.5, -1.5), (5.0, -0.15), STUB_TIP
    )

    result = optimize_chains([bar, stub], scenarios.PARAMS, COARSE)

    assert result.termination is Termination.CONVERGED
    for chain, stats in zip(
        result.chains, after([bar, stub], result, scenarios.PARAMS), strict=True
    ):
        assert chain.frozen == ()
        assert stats.min_clearance >= -INSIDE_TOLERANCE
        assert stats.length < chain.before.length
    assert np.all(result.chains[0].polyline == TOUCH, axis=1).sum() == 1


def test_time_limit_stops_after_the_burst_that_passes_it() -> None:
    scenario = scenarios.channel_pair()
    limited = replace(COARSE, time_limit=0.5)

    result = optimize_chains(scenario.chains, scenario.params, limited, clock=Ticking())

    assert result.termination is Termination.TIME_LIMIT
    assert result.iterations == 1
    first, second = (
        LineString(c.polyline).hausdorff_distance(LineString(given.centreline.points))
        for c, given in zip(result.chains, scenario.chains, strict=True)
    )
    assert first > COARSE.settled_change
    # Only the first chain had its burst; the second keeps its shape.
    assert second < INSIDE_TOLERANCE


def test_time_limit_counts_from_when_the_stage_started() -> None:
    scenario = scenarios.uneven_channel_pair()
    # Standing still past the limit: only a deadline counted from `started`
    # has passed.
    clock = Ticking(now=10.0, step=0.0)
    limited = replace(COARSE, time_limit=5.0)

    result = optimize_chains(
        scenario.chains, scenario.params, limited, clock=clock, started=0.0
    )

    assert result.termination is Termination.TIME_LIMIT
    assert result.iterations == 1


PROGRESS_LINE = re.compile(
    r"^(DEBUG|INFO) Iteration (\d+), (\d+\.\d) s: length \d+\.\d{3} mm, "
    r"energy -?\d+\.\d{3} \(best -?\d+\.\d{3}\), \d+ of 2 chains moving$"
)


def progress(log: list[str]) -> list[tuple[str, int, float]]:
    """The progress lines in `log` as (level, iteration, elapsed s)."""
    lines = [line.strip() for line in log if " Iteration " in line]
    matches = [PROGRESS_LINE.match(line) for line in lines]
    assert all(matches), lines
    return [(m[1], int(m[2]), float(m[3])) for m in matches if m]


def test_progress_is_logged_after_every_iteration(log: list[str]) -> None:
    scenario = scenarios.uneven_channel_pair()

    result = optimize_chains(
        scenario.chains, scenario.params, COARSE, clock=Ticking(step=0.0)
    )

    lines = progress(log)
    assert [iteration for _, iteration, _ in lines] == list(
        range(1, result.iterations + 1)
    )
    # A run shorter than the interval says nothing at INFO.
    assert {level for level, _, _ in lines} == {"DEBUG"}


def test_progress_at_info_comes_at_most_once_per_interval(
    log: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = scenarios.uneven_channel_pair()
    # A second per burst, however often the clock is read.
    clock = BurstClock(monkeypatch)

    result = optimize_chains(scenario.chains, scenario.params, COARSE, clock=clock)

    lines = progress(log)
    assert [iteration for _, iteration, _ in lines] == list(
        range(1, result.iterations + 1)
    )
    assert [elapsed for _, _, elapsed in lines][-1] == clock.bursts
    shown = 0.0
    for level, _, elapsed in lines:
        due = elapsed - shown >= PROGRESS_INTERVAL
        assert level == ("INFO" if due else "DEBUG")
        if due:
            shown = elapsed
    assert any(level == "INFO" for level, _, _ in lines)


def test_without_a_time_limit_the_clock_changes_nothing() -> None:
    scenario = scenarios.uneven_channel_pair()

    timed = optimize_chains(scenario.chains, scenario.params, COARSE, clock=Ticking())
    plain = run(scenario)

    assert timed.termination is plain.termination is Termination.CONVERGED
    assert timed.iterations == plain.iterations
    for a, b in zip(timed.chains, plain.chains, strict=True):
        np.testing.assert_array_equal(a.polyline, b.polyline)


def test_a_run_that_settles_on_the_burst_past_the_limit_has_converged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = scenarios.uneven_channel_pair()
    clock = BurstClock(monkeypatch)
    plain = optimize_chains(scenario.chains, scenario.params, COARSE, clock=clock)
    bursts = clock.bursts
    clock.bursts = 0
    # The limit passes during the last burst, after which every chain rests.
    limited = replace(COARSE, time_limit=bursts - 0.5)

    result = optimize_chains(
        scenario.chains, scenario.params, limited, clock=clock, started=0.0
    )

    assert plain.termination is result.termination is Termination.CONVERGED
    assert result.iterations == plain.iterations
