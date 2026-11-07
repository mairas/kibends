import copy
from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
import pytest
import scenarios
from distance_fields import BoolArray, FloatArray, FreeSpace, MisleadingWalls
from scenarios import ChainsScenario, selected
from shapely.geometry import LineString

import kibends.optimize as optimize_module
from kibends.energy import EnergyParams, total_energy
from kibends.obstacles import CompositeField, chain_obstacles
from kibends.optimize import (
    INSIDE_TOLERANCE,
    ChainInput,
    ObstacleField,
    OptimizerSettings,
    RunResult,
    Termination,
    optimize,
    optimize_chains,
)

COARSE = OptimizerSettings(spacing=0.3, burst_maxiter=30)


def run(scenario: ChainsScenario, settings: OptimizerSettings = COARSE) -> RunResult:
    return optimize_chains(scenario.chains, scenario.params, settings)


def gap(a: FloatArray, b: FloatArray) -> float:
    """Smallest distance between two polylines' edges, mm."""
    return float(LineString(a).distance(LineString(b)))


def y_at(points: FloatArray, x: float) -> float:
    return float(np.interp(x, points[:, 0], points[:, 1]))


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
            c.after.length
            for c in optimize_chains(chains, scenario.params, COARSE).chains
        )
        for chains in (scenario.chains, scenario.chains[::-1])
    ]

    assert lengths[1] == pytest.approx(lengths[0], abs=ORDER_TOLERANCE)


def test_arch_stays_above_the_bar_it_cannot_cross() -> None:
    result = run(scenarios.tent())

    arch, bar = (c.polyline for c in result.chains)
    assert result.termination is Termination.CONVERGED
    assert gap(arch, bar) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE
    assert y_at(arch, 5.0) > y_at(bar, 5.0)
    assert result.chains[0].after.length < result.chains[0].before.length


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
    assert only.after == alone.after


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


def test_one_abnormal_chain_ends_the_run_abnormally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    misled = selected(
        "misled", "A", MisleadingWalls(1.0), (0.0, 0.0), (5.0, 0.8), (10.0, 0.0)
    )
    calm = selected("calm", "B", FreeSpace(), (0.0, 5.0), (5.0, 6.0), (10.0, 5.0))
    params = replace(scenarios.PARAMS, repulsion=5.0)
    scenario = ChainsScenario("abnormal", (calm, misled), params)
    outputs: dict[BurstStart, FloatArray] = {}

    def keep(start: BurstStart, burst: object) -> object:
        outputs[start] = vars(burst)["points"]
        return burst

    watch_bursts(monkeypatch, keep)

    result = optimize_chains(scenario.chains, params)

    assert result.termination is Termination.ABNORMAL
    assert result.iterations == 1
    calm_after, misled_after = (c.polyline for c in result.chains)
    assert np.array_equal(calm_after, outputs[(0.0, 5.0)])
    assert result.chains[0].after.length < result.chains[0].before.length
    # The best state holds the calm chain's burst and either the misled
    # chain's start or its failed burst, whichever the guard accepted.
    misled_start = optimize_module.resample(
        misled.centreline.points, misled.fixed, OptimizerSettings().spacing
    )[0]
    assert any(
        np.array_equal(misled_after, candidate)
        for candidate in (misled_start, outputs[(0.0, 0.0)])
    )


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


# At spacings 0.25 and 0.3 the drape's first burst piles two vertices of the
# long arch onto each other at its bubble bounds, and L-BFGS-B fails on the
# bending term's gradient across the 17 µm edge between them.
DRAPE_SETTINGS = replace(COARSE, spacing=0.2)


@pytest.mark.parametrize(
    ("scenario", "settings"),
    [
        (scenarios.drape(), DRAPE_SETTINGS),
        (scenarios.squeeze(), COARSE),
        (scenarios.hooks(), COARSE),
    ],
    ids=["drape", "squeeze", "hooks"],
)
def test_moving_chains_tighten_without_crossing(
    scenario: ChainsScenario, settings: OptimizerSettings
) -> None:
    result = run(scenario, settings)

    a, b = (c.polyline for c in result.chains)
    assert result.termination is Termination.CONVERGED
    assert not LineString(a).intersects(LineString(b))
    assert gap(a, b) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE
    for chain in result.chains:
        assert chain.after.length < chain.before.length


# The neck-down's lengths alone and together agree to within this, mm.
JUNCTION_TOLERANCE = 1e-3


def test_chains_of_one_net_meeting_end_to_end_leave_the_junction_straight() -> None:
    scenario = scenarios.neck_down()

    result = run(scenario)

    for chain, given in zip(result.chains, scenario.chains, strict=True):
        alone = optimize(
            given.centreline.points, given.fixed, given.field, scenario.params, COARSE
        )
        assert chain.after.length == pytest.approx(
            alone.after.length, abs=JUNCTION_TOLERANCE
        )
        assert chain.after.min_clearance >= 0.0


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
    ) -> object:
        start = (float(points[0, 0]), float(points[0, 1]))
        starts.append(start)
        return react(start, real(points, fixed, field, params, settings))

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


def test_an_abnormal_burst_of_a_settled_chain_leaves_it_settled(
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
        # Taken as is, the arch's push would bend the bar.
        abnormal = copy.copy(burst)
        object.__setattr__(abnormal, "abnormal", True)
        return abnormal

    failed: list[BurstStart] = []
    starts = watch_bursts(monkeypatch, fail)

    result = optimize_chains(
        tent.chains[::-1], replace(tent.params, repulsion=0.5), COARSE
    )

    bar = result.chains[0].polyline
    assert failed
    assert result.termination is Termination.CONVERGED
    assert np.abs(bar[:, 1] - scenarios.TENT_HEIGHT).max() < 1e-9
