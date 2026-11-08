import itertools
import math
import re
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest
import scenarios
from conftest import BurstClock, Ticking
from distance_fields import Disc, FloatArray, FreeSpace
from scenarios import ChainsScenario, selected
from shapely.geometry import LineString, Point, Polygon

import kibends.optimize as optimize_module
from kibends.bursts import Burst, Objective, OptimizerSettings, Revert
from kibends.commit import snapped_points
from kibends.energy import (
    EnergyParams,
    ObstacleField,
    obstacle_reach,
    total_energy,
)
from kibends.neighbours import ChainInput, chain_field, neighbour_field
from kibends.obstacles import ChainCentreline
from kibends.optimize import (
    INSIDE_TOLERANCE,
    MAX_ABNORMAL_BURSTS,
    PROGRESS_INTERVAL,
    PolylineStats,
    RunResult,
    Termination,
    layer_energy,
    measure,
    optimize,
    optimize_chains,
    pinned_at_junctions,
    resample,
    seen_parts,
)
from kibends.simplify import simplify_chains

COARSE = OptimizerSettings(spacing=0.3, settled_change=0.003, burst_maxiter=30)


def run(scenario: ChainsScenario, settings: OptimizerSettings = COARSE) -> RunResult:
    return optimize_chains(scenario.chains, scenario.params, settings)


def gap(a: FloatArray, b: FloatArray) -> float:
    """Smallest distance between two polylines' edges, mm."""
    return float(LineString(a).distance(LineString(b)))


def length(points: FloatArray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def y_at(points: FloatArray, x: float) -> float:
    return float(np.interp(x, points[:, 0], points[:, 1]))


def fields(
    chains: Sequence[ChainInput], polylines: Sequence[FloatArray], params: EnergyParams
) -> list[ObstacleField]:
    """Each chain's field with the others on its layer at `polylines`, built
    chain by chain as a reference for the problem's energy."""
    reach = obstacle_reach(params)
    return [
        chain_field(
            chain.field,
            [
                neighbour_field(
                    chain,
                    [
                        (other, polylines[j])
                        for j, other in enumerate(chains)
                        if j != i and other.layer == chain.layer
                    ],
                    reach,
                    polylines[i],
                )
            ],
        )
        for i, chain in enumerate(chains)
    ]


def committed_energy(
    chains: Sequence[ChainInput],
    result: RunResult,
    params: EnergyParams,
    settings: OptimizerSettings = COARSE,
) -> float:
    """The run energy of the returned shapes, each chain's in its field with
    the others."""
    polylines = [c.polyline for c in result.chains]
    return sum(
        total_energy(c.polyline, c.fixed, field, params, settings.short_edge)[0]
        for c, field in zip(
            result.chains, fields(chains, polylines, params), strict=True
        )
    )


def after(
    chains: Sequence[ChainInput],
    result: RunResult,
    params: EnergyParams,
    settings: OptimizerSettings = COARSE,
) -> list[PolylineStats]:
    """Each chain's statistics with the other chains at their returned shapes."""
    polylines = [c.polyline for c in result.chains]
    return [
        measure(p, field, settings.spacing)
        for p, field in zip(polylines, fields(chains, polylines, params), strict=True)
    ]


def min_clearances(
    chains: Sequence[ChainInput], result: RunResult, params: EnergyParams
) -> list[float]:
    return [stats.min_clearance for stats in after(chains, result, params)]


def bumps() -> ChainsScenario:
    """Two chains side by side, each bumped away from the other."""
    free = FreeSpace()
    return ChainsScenario(
        "bumps",
        (
            selected("upper", "A", free, (0.0, 0.25), (5.0, 1.5), (10.0, 0.25)),
            selected("lower", "B", free, (0.0, -0.25), (5.0, -1.5), (10.0, -0.25)),
        ),
    )


class TestEnergy:
    def chains(self) -> tuple[ChainInput, ...]:
        free = FreeSpace()
        return (
            selected("A", "A", free, (0.0, 0.0), (1.0, 0.45), (2.0, 0.6), (3.0, 0.0)),
            selected("B", "B", free, (0.0, 1.0), (1.2, 0.8), (2.0, 1.0), (3.0, 1.0)),
        )

    def params(self) -> EnergyParams:
        return replace(scenarios.PARAMS, repulsion=1.0, falloff_range=0.5)

    def test_is_the_sum_of_each_chains_energy_against_the_others(self) -> None:
        chains = self.chains()
        polylines = [c.centreline.points for c in chains]
        masks = [c.fixed for c in chains]

        value, _ = layer_energy(chains, polylines, masks, self.params(), 0.01)

        expected = sum(
            total_energy(p, m, f, self.params(), 0.01)[0]
            for p, m, f in zip(
                polylines, masks, fields(chains, polylines, self.params()), strict=True
            )
        )
        assert value == pytest.approx(expected, rel=1e-12)

    def test_gradient_reaches_the_other_chains_vertices(self) -> None:
        chains = self.chains()
        polylines = [c.centreline.points.copy() for c in chains]
        masks = [c.fixed for c in chains]
        params = self.params()
        _, gradients = layer_energy(chains, polylines, masks, params, 0.01)
        step = 1e-7

        for i, j in ((0, 1), (0, 2), (1, 1), (1, 2)):
            for axis in (0, 1):
                shifted = [p.copy() for p in polylines]
                shifted[i][j, axis] += step
                up = layer_energy(chains, shifted, masks, params, 0.01)[0]
                shifted[i][j, axis] -= 2 * step
                down = layer_energy(chains, shifted, masks, params, 0.01)[0]
                numeric = (up - down) / (2 * step)
                assert gradients[i][j, axis] == pytest.approx(numeric, rel=1e-5)
        # The chains are near enough for their repulsion to count.
        apart = [
            layer_energy([c], [p], [m], params, 0.01)
            for c, p, m in zip(chains, polylines, masks, strict=True)
        ]
        for (_, gradient), together in zip(apart, gradients, strict=True):
            assert not np.allclose(gradient, together)


def test_two_bumped_chains_shorten_and_stay_apart() -> None:
    scenario = bumps()

    result = run(scenario)

    upper, lower = (c.polyline for c in result.chains)
    assert result.termination is Termination.CONVERGED
    for chain in result.chains:
        assert length(chain.polyline) < chain.before.length - 0.2
        assert chain.rejected_edges == 0
    assert gap(upper, lower) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE
    assert all(
        c >= -INSIDE_TOLERANCE
        for c in min_clearances(scenario.chains, result, scenario.params)
    )


@pytest.mark.parametrize(
    "scenario",
    [scenarios.squeeze(), scenarios.hooks(), scenarios.drape(), scenarios.tent()],
    ids=lambda s: s.name,
)
def test_chains_pulled_into_each_others_space_never_cross(
    scenario: ChainsScenario,
) -> None:
    result = run(scenario)

    a, b = (c.polyline for c in result.chains)
    assert result.termination is Termination.CONVERGED
    assert not LineString(a).intersects(LineString(b))
    assert gap(a, b) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE
    assert sum(length(c.polyline) for c in result.chains) < sum(
        c.before.length for c in result.chains
    )


@pytest.mark.parametrize("apex", [4.0, 10.0])
def test_chains_racing_toward_each_other_head_on_stop_at_their_clearance(
    apex: float,
) -> None:
    # The arch's apex comes down onto the V hanging inside it while the V
    # flattens up toward it, the higher apex from farther and faster.
    free = FreeSpace()
    arch = selected("arch", "A", free, (0.0, 0.0), (5.0, apex), (10.0, 0.0))
    v = selected("V", "B", free, (4.0, 2.5), (5.0, 0.5), (6.0, 2.5))

    result = optimize_chains([arch, v], scenarios.PARAMS, COARSE)

    a, b = (c.polyline for c in result.chains)
    assert not LineString(a).intersects(LineString(b))
    assert gap(a, b) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE
    assert y_at(a, 5.0) > y_at(b, 5.0)


def test_chains_close_together_straighten_together_by_the_step_floor() -> None:
    # Nested arches 20 µm apart flatten together: their bubbles against
    # each other would let the apexes move about 15 µm in a burst, the step
    # floor lets them move half the inflation less δ.
    free = FreeSpace()
    # 20 µm apart square to their legs, which rise 2 in 5.
    apart = (scenarios.PAIR_INFLATION + 0.02) * math.hypot(1.0, 0.4)
    lower = selected("lower", "A", free, (0.0, 0.0), (5.0, 2.0), (10.0, 0.0))
    upper = selected(
        "upper", "B", free, (0.0, apart), (5.0, 2.0 + apart), (10.0, apart)
    )
    floor = 0.5 * (scenarios.PAIR_INFLATION - COARSE.bubble_tolerance)

    result = optimize_chains(
        [lower, upper], scenarios.PARAMS, replace(COARSE, max_iterations=1)
    )

    a, b = (c.polyline for c in result.chains)
    for start, shape in ((2.0, a), (2.0 + apart, b)):
        assert floor / 2 < start - y_at(shape, 5.0) <= floor + INSIDE_TOLERANCE
    assert gap(a, b) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE


SPARSE = replace(COARSE, spacing=3.0, settled_change=0.03)


@pytest.mark.parametrize("height", [1.5, 3.0, 6.0])
def test_a_chain_cannot_jump_through_another_in_one_burst(height: float) -> None:
    # At this spacing the arch has one free vertex, whose straight line to
    # its shortest shape passes the bar and ends clear beyond it: only the
    # bubbles against the other chains keep the burst from taking it, as
    # both shapes at its end are clear of each other.
    free = FreeSpace()
    arch = selected("arch", "A", free, (0.0, 0.0), (5.0, height), (10.0, 0.0))
    bar = selected("bar", "B", free, (4.0, 0.7), (6.0, 0.7))

    result = optimize_chains([arch, bar], scenarios.PARAMS, SPARSE)

    a, b = (c.polyline for c in result.chains)
    assert y_at(a, 5.0) > y_at(b, 5.0)
    assert gap(a, b) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE


# Pulled onto each other against a weak clearance hinge, the chains end
# within this much of their clearance, mm. Without the edge check their best
# state stays where they were before they first overlapped, 5 µm out.
TIGHT = 2e-3


def test_moves_into_another_chains_clearance_are_undone_and_counted() -> None:
    # The hinge is too weak to hold the chains apart: only moving back the
    # edges that newly enter the other chain's clearance does.
    scenario = scenarios.squeeze()
    params = replace(scenario.params, clearance_weight=0.1)

    result = optimize_chains(scenario.chains, params, COARSE)

    a, b = (c.polyline for c in result.chains)
    assert gap(a, b) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE
    assert gap(a, b) < scenarios.PAIR_INFLATION + TIGHT
    assert all(chain.rejected_edges > 0 for chain in result.chains)


def test_a_chain_wrapped_around_an_obstacle_stays_on_its_side() -> None:
    topology = scenarios.topology()
    chain = selected("t", "N", topology.field, (-5.0, 0.0), (0.0, 2.0), (5.0, 0.0))

    (result,) = optimize_chains([chain], topology.params, COARSE).chains

    assert y_at(result.polyline, 0.0) > 0.0
    assert measure(result.polyline, topology.field, COARSE.spacing).min_clearance >= (
        -INSIDE_TOLERANCE
    )
    assert length(result.polyline) < length(chain.centreline.points)


PIN = (5.0, 1.0)


def test_pins_and_ends_stay_where_they_are() -> None:
    free = FreeSpace()
    points = np.array([(0.0, 0.0), (2.5, 2.0), PIN, (7.5, 2.0), (10.0, 0.0)])
    pinned = ChainInput(
        ChainCentreline(
            "pinned", "A", scenarios.PAIR_WIDTH, scenarios.PAIR_CLEARANCE, points
        ),
        "F.Cu",
        np.array([True, False, True, False, True]),
        free,
    )
    other = selected("other", "B", free, (0.0, -1.0), (5.0, -2.0), (10.0, -1.0))

    result = optimize_chains([pinned, other], scenarios.PARAMS, COARSE)

    for chain, given in zip(result.chains, (pinned, other), strict=True):
        kept = given.centreline.points[given.fixed]
        np.testing.assert_array_equal(chain.polyline[chain.fixed], kept)
        np.testing.assert_array_equal(chain.polyline[[0, -1]], kept[[0, -1]])
    assert np.all(result.chains[0].polyline == PIN, axis=1).sum() == 1


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


def on_layer(chain: ChainInput, layer: str) -> ChainInput:
    return replace(chain, layer=layer)


def test_time_limit_stops_after_the_burst_that_passes_it() -> None:
    scenario = bumps()
    back = on_layer(
        selected("back", "C", FreeSpace(), (0.0, 0.0), (5.0, 3.0), (10.0, 0.0)), "B.Cu"
    )
    chains = (*scenario.chains, back)
    limited = replace(COARSE, time_limit=0.5)

    result = optimize_chains(chains, scenario.params, limited, clock=Ticking())

    assert result.termination is Termination.TIME_LIMIT
    assert result.iterations == 1
    front = result.chains[:2]
    for chain in front:
        assert length(chain.polyline) < chain.before.length
    # Only the front layer had its burst; the back one keeps its start.
    np.testing.assert_array_equal(
        result.chains[2].polyline,
        resample(back.centreline.points, back.fixed, COARSE.spacing)[0],
    )


def test_time_limit_counts_from_when_the_stage_started() -> None:
    scenario = bumps()
    limited = replace(COARSE, time_limit=5.0)

    result = optimize_chains(
        scenario.chains,
        scenario.params,
        limited,
        clock=Ticking(now=10.0, step=0.0),
        started=0.0,
    )

    assert result.termination is Termination.TIME_LIMIT
    assert result.iterations == 1


def test_iteration_cap_ends_the_run() -> None:
    result = run(scenarios.hooks(), replace(COARSE, max_iterations=1))

    assert result.termination is Termination.ITERATION_CAP
    assert result.iterations == 1


@pytest.mark.parametrize(
    "scenario", [scenarios.tent(), scenarios.squeeze()], ids=lambda s: s.name
)
def test_capped_runs_return_one_state_that_never_gets_worse(
    scenario: ChainsScenario,
) -> None:
    energies: list[float] = []
    for cap in range(1, 5):
        result = run(scenario, replace(COARSE, max_iterations=cap))
        a, b = (c.polyline for c in result.chains)

        assert gap(a, b) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE
        energies.append(committed_energy(scenario.chains, result, scenario.params))
    assert energies == sorted(energies, reverse=True)
    assert energies[-1] < energies[0]


def test_straight_chains_have_nothing_to_do() -> None:
    free = FreeSpace()
    chains = (
        selected("A", "A", free, (0.0, 0.0), (10.0, 0.0)),
        selected("B", "B", free, (0.0, 1.0), (10.0, 1.0)),
    )

    result = optimize_chains(chains, scenarios.PARAMS, COARSE)

    assert result.termination is Termination.NOTHING_TO_DO
    for chain, given in zip(result.chains, chains, strict=True):
        np.testing.assert_array_equal(chain.polyline, given.centreline.points)


def test_an_optimised_layout_converges_again_almost_unchanged() -> None:
    scenario = scenarios.squeeze()
    first = run(scenario)
    again = tuple(
        replace(
            given,
            centreline=replace(given.centreline, points=chain.polyline),
            fixed=chain.fixed,
        )
        for given, chain in zip(scenario.chains, first.chains, strict=True)
    )

    second = optimize_chains(again, scenario.params, COARSE)

    assert second.termination in (Termination.CONVERGED, Termination.NOTHING_TO_DO)
    assert second.iterations <= COARSE.stall_window + 1
    for before, chain in zip(first.chains, second.chains, strict=True):
        assert (
            LineString(chain.polyline).hausdorff_distance(LineString(before.polyline))
            < 10 * COARSE.settled_change
        )


def test_chains_of_one_net_are_pulled_together_without_overlapping() -> None:
    result = run(scenarios.same_net_wrap())

    inner, outer = (c.polyline for c in result.chains)
    assert result.termination is Termination.CONVERGED
    assert gap(inner, outer) >= scenarios.SAME_NET_INFLATION - INSIDE_TOLERANCE
    assert gap(inner, outer) < scenarios.SAME_NET_INFLATION + 0.05


def test_chains_of_one_net_on_different_layers_are_taken() -> None:
    inner, outer = scenarios.same_net_wrap().chains

    result = optimize_chains([inner, on_layer(outer, "B.Cu")], scenarios.PARAMS, COARSE)

    assert result.termination is Termination.CONVERGED


def test_chains_on_different_layers_do_not_interact() -> None:
    free = FreeSpace()
    across = selected("A", "A", free, (0.0, 0.0), (5.0, 2.0), (10.0, 0.0))
    along = on_layer(
        selected("B", "B", free, (5.0, -2.0), (7.0, 1.0), (5.0, 4.0)), "B.Cu"
    )

    result = optimize_chains([across, along], scenarios.PARAMS, COARSE)

    for chain, given in zip(result.chains, (across, along), strict=True):
        alone = optimize_chains([given], scenarios.PARAMS, COARSE)
        np.testing.assert_array_equal(chain.polyline, alone.chains[0].polyline)


def test_endpoints_must_be_fixed() -> None:
    chain = selected("A", "A", FreeSpace(), (0.0, 0.0), (5.0, 1.0), (10.0, 0.0))
    loose = replace(chain, fixed=np.array([True, False, False]))

    with pytest.raises(ValueError, match="endpoints"):
        optimize_chains([loose], scenarios.PARAMS, COARSE)


def test_a_problem_whose_bursts_all_fail_gives_up_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = optimize_module.bounded_burst
    bursts: list[Burst] = []

    def failing(
        objective: Objective,
        start: FloatArray,
        half_sides: FloatArray,
        revert: Revert,
        settings: OptimizerSettings,
        retry_bounds: FloatArray | None = None,
    ) -> Burst:
        burst = real(objective, start, half_sides, revert, settings, retry_bounds)
        bursts.append(burst)
        return replace(burst, abnormal=True)

    monkeypatch.setattr(optimize_module, "bounded_burst", failing)
    scenario = bumps()

    result = run(scenario)

    assert result.iterations == len(bursts) == MAX_ABNORMAL_BURSTS
    assert result.gave_up == (("the chains on F.Cu", MAX_ABNORMAL_BURSTS),)
    assert result.abnormal == (("the chains on F.Cu", MAX_ABNORMAL_BURSTS),)
    # The bursts were taken: the run lowered the energy, so it converged.
    assert result.termination is Termination.CONVERGED
    for chain in result.chains:
        assert length(chain.polyline) < chain.before.length


PROGRESS_LINE = re.compile(
    r"^(DEBUG|INFO) Iteration (\d+), (\d+\.\d) s: length \d+\.\d{3} mm, "
    r"energy -?\d+\.\d{3} \(best -?\d+\.\d{3}\), largest change \d\.\d{2}e[-+]\d+ mm$"
)


def progress(log: list[str]) -> list[tuple[str, int, float]]:
    lines = [line.strip() for line in log if " Iteration " in line]
    matches = [PROGRESS_LINE.match(line) for line in lines]
    assert all(matches), lines
    return [(m[1], int(m[2]), float(m[3])) for m in matches if m]


def test_progress_is_logged_after_every_iteration(log: list[str]) -> None:
    result = optimize_chains(
        bumps().chains, scenarios.PARAMS, COARSE, clock=Ticking(step=0.0)
    )

    lines = progress(log)
    assert [iteration for _, iteration, _ in lines] == list(
        range(1, result.iterations + 1)
    )
    assert {level for level, _, _ in lines} == {"DEBUG"}


def test_progress_at_info_comes_at_most_once_per_interval(log: list[str]) -> None:
    clock = Ticking(step=PROGRESS_INTERVAL / 2)

    optimize_chains(scenarios.squeeze().chains, scenarios.PARAMS, COARSE, clock=clock)

    levels = [level for level, _, _ in progress(log)]
    assert "INFO" in levels
    assert all(not (a == "INFO" and b == "INFO") for a, b in itertools.pairwise(levels))


@pytest.mark.parametrize(
    "scenario",
    [scenarios.channel_pair(), scenarios.tent(), scenarios.squeeze()],
    ids=lambda s: s.name,
)
def test_chains_end_clear_of_each_other_and_their_obstacles(
    scenario: ChainsScenario,
) -> None:
    result = run(scenario)

    assert all(
        c >= -INSIDE_TOLERANCE
        for c in min_clearances(scenario.chains, result, scenario.params)
    )


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


def test_chains_inside_each_others_clearance_throughout_stay_put() -> None:
    # Every vertex resampling adds lies deeper than δ inside the other
    # chain's clearance, so every bound of a burst is zero.
    apart = scenarios.PAIR_INFLATION - 2 * COARSE.bubble_tolerance
    a = selected("A", "A", FreeSpace(), (0.0, 0.0), (5.0, 0.0))
    b = selected("B", "B", FreeSpace(), (0.0, apart), (5.0, apart))

    result = optimize_chains([a, b], scenarios.PARAMS, COARSE)

    for given, chain in zip((a, b), result.chains, strict=True):
        assert LineString(chain.polyline).equals(LineString(given.centreline.points))


# The neck-down's lengths alone and together agree to within this, mm.
NECK_DOWN_TOLERANCE = 1e-3


def test_chains_of_one_net_meeting_end_to_end_leave_the_junction_straight() -> None:
    scenario = scenarios.neck_down()

    result = run(scenario)

    stats = after(scenario.chains, result, scenario.params)
    for chain, given in zip(stats, scenario.chains, strict=True):
        alone = optimize(
            given.centreline.points, given.fixed, given.field, scenario.params, COARSE
        )
        assert chain.length == pytest.approx(
            alone.after.length, abs=NECK_DOWN_TOLERANCE
        )
        assert chain.min_clearance >= 0.0


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
    # The limit passes during the last burst, after which the problem settles.
    limited = replace(COARSE, time_limit=bursts - 0.5)

    result = optimize_chains(
        scenario.chains, scenario.params, limited, clock=clock, started=0.0
    )

    assert plain.termination is result.termination is Termination.CONVERGED
    assert result.iterations == plain.iterations


JUNCTION_SETTINGS = OptimizerSettings(
    spacing=0.3, settled_change=0.002, burst_maxiter=60
)
PAD = (10.0, 0.0)
LEG = 10.0
BOW = 0.5


def heading(points: FloatArray, at: int, toward: int) -> float:
    """The direction from vertex `at` to vertex `toward`, degrees."""
    dx, dy = points[toward] - points[at]
    return math.degrees(math.atan2(dy, dx))


@pytest.mark.parametrize("angle", [30.0, 10.0])
def test_chains_of_one_net_meeting_at_a_pad_depart_at_their_own_angles(
    angle: float,
) -> None:
    # Near the pad each leaves the other out as far as it lies within their
    # half widths plus the reach; farther at a sharper angle.
    turn = math.radians(angle)
    far = (PAD[0] - LEG * math.cos(turn), LEG * math.sin(turn))
    bow = (
        (PAD[0] + far[0]) / 2 + BOW * math.sin(turn),
        far[1] / 2 + BOW * math.cos(turn),
    )
    a = selected("A", "N", FreeSpace(), (0.0, 0.0), (5.0, -BOW), PAD)
    b = selected("B", "N", FreeSpace(), PAD, bow, far)

    result = optimize_chains([a, b], scenarios.PARAMS, JUNCTION_SETTINGS)

    assert result.termination is Termination.CONVERGED
    for chain, clearance in zip(
        result.chains, min_clearances([a, b], result, scenarios.PARAMS), strict=True
    ):
        assert clearance >= -INSIDE_TOLERANCE
        assert length(chain.polyline) < chain.before.length
    first, second = (c.polyline for c in result.chains)
    np.testing.assert_array_equal(first[-1], PAD)
    np.testing.assert_array_equal(second[0], PAD)
    assert heading(first, -1, 0) == pytest.approx(180.0, abs=1.0)
    assert heading(second, 0, -1) == pytest.approx(180.0 - angle, abs=1.0)


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
    bar = ChainInput(
        ChainCentreline(
            "bar", "N", scenarios.PAIR_WIDTH, scenarios.PAIR_CLEARANCE, bar_points
        ),
        "F.Cu",
        np.array([True, False, False, True, False, False, True]),
        FreeSpace(),
    )
    stub = selected(
        "stub", "N", FreeSpace(), (5.0, -3.0), (5.5, -1.5), (5.0, -0.15), STUB_TIP
    )

    result = optimize_chains([bar, stub], scenarios.PARAMS, COARSE)

    assert result.termination is Termination.CONVERGED
    for chain, clearance in zip(
        result.chains,
        min_clearances([bar, stub], result, scenarios.PARAMS),
        strict=True,
    ):
        assert chain.frozen == ()
        assert clearance >= -INSIDE_TOLERANCE
        assert length(chain.polyline) < chain.before.length
    assert np.all(result.chains[0].polyline == TOUCH, axis=1).sum() == 1


def branch(
    description: str,
    junction: tuple[float, float],
    *vertices: tuple[float, float],
    width: float = scenarios.PAIR_WIDTH,
    field: ObstacleField | None = None,
) -> ChainInput:
    """A chain of net N from a loose end at `junction` through `vertices`."""
    chain = selected(
        description, "N", FreeSpace() if field is None else field, junction, *vertices
    )
    fixed = chain.fixed.copy()
    fixed[0] = False
    return replace(
        chain, centreline=replace(chain.centreline, width=width), fixed=fixed
    )


def junction_of(result: RunResult) -> FloatArray:
    """The point every chain of `result` starts at, asserting they all do."""
    starts = [c.polyline[0] for c in result.chains]
    for start in starts:
        np.testing.assert_array_equal(start, starts[0])
    assert not any(c.fixed[0] for c in result.chains)
    return starts[0]


def angles_at_junction(result: RunResult) -> list[float]:
    """The angles between neighbouring branches where they leave the
    junction, degrees, sorted."""
    headings = sorted(heading(c.polyline, 0, 1) % 360.0 for c in result.chains)
    return sorted(
        (b - a) % 360.0 for a, b in itertools.pairwise([*headings, headings[0]])
    )


def fermat_tee(
    widths: Sequence[float] = (scenarios.PAIR_WIDTH,) * 3,
) -> list[ChainInput]:
    """Three bowed branches from scenarios.FERMAT_START to the corners of
    scenarios.FERMAT_ENDS."""
    start = scenarios.FERMAT_START
    first, second, third = scenarios.FERMAT_ENDS
    return [
        branch("A", start, (0.5, -0.5), first, width=widths[0]),
        branch("B", start, (5.0, -1.0), second, width=widths[1]),
        branch("C", start, (2.0, 4.0), third, width=widths[2]),
    ]


@pytest.mark.parametrize(
    "widths", [(0.2, 0.2, 0.2), (0.2, 0.4, 0.3)], ids=["even", "mixed"]
)
def test_a_tee_moves_its_junction_to_where_the_branches_meet_at_120_degrees(
    widths: tuple[float, float, float],
) -> None:
    chains = fermat_tee(widths)

    result = optimize_chains(chains, scenarios.PARAMS, JUNCTION_SETTINGS)

    assert result.termination is Termination.CONVERGED
    np.testing.assert_allclose(junction_of(result), scenarios.FERMAT_POINT, atol=0.05)
    assert angles_at_junction(result) == pytest.approx([120.0] * 3, abs=2.0)
    assert all(
        c >= -INSIDE_TOLERANCE
        for c in min_clearances(pinned_at_junctions(chains), result, scenarios.PARAMS)
    )
    for chain, given in zip(result.chains, chains, strict=True):
        np.testing.assert_array_equal(chain.polyline[-1], given.centreline.points[-1])


def test_a_star_of_four_branches_meets_in_the_middle() -> None:
    corners = ((-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0))
    start = (1.0, 0.5)
    chains = [
        branch(f"arm {k}", start, (x / 2 + y / 4, y / 2 - x / 4), (x, y))
        for k, (x, y) in enumerate(corners)
    ]

    result = optimize_chains(chains, scenarios.PARAMS, JUNCTION_SETTINGS)

    assert result.termination is Termination.CONVERGED
    np.testing.assert_allclose(junction_of(result), (0.0, 0.0), atol=0.05)
    assert angles_at_junction(result) == pytest.approx([90.0] * 4, abs=2.0)


def test_a_junction_on_a_pad_stays_where_it_is() -> None:
    # Extraction pins a junction that a pad covers: the branches arrive with
    # that end fixed and meet at a pinned touch.
    chains = [
        replace(c, fixed=c.fixed | (np.arange(len(c.fixed)) == 0)) for c in fermat_tee()
    ]

    result = optimize_chains(chains, scenarios.PARAMS, JUNCTION_SETTINGS)

    assert result.termination is Termination.CONVERGED
    for chain, clearance in zip(
        result.chains, min_clearances(chains, result, scenarios.PARAMS), strict=True
    ):
        np.testing.assert_array_equal(chain.polyline[0], scenarios.FERMAT_START)
        assert chain.fixed[0]
        assert clearance >= -INSIDE_TOLERANCE
        assert length(chain.polyline) < chain.before.length


def test_simplified_and_snapped_branches_still_share_the_junction() -> None:
    chains = fermat_tee((0.2, 0.4, 0.3))
    result = optimize_chains(chains, scenarios.PARAMS, JUNCTION_SETTINGS)

    simplified = simplify_chains(pinned_at_junctions(chains), result, scenarios.PARAMS)

    junction = junction_of(result)
    for polyline in simplified:
        np.testing.assert_array_equal(polyline[0], junction)
    assert len({snapped_points(p)[0] for p in simplified}) == 1


class TestJunctionEnergy:
    def params(self) -> EnergyParams:
        return replace(scenarios.PARAMS, repulsion=1.0, falloff_range=0.5)

    def test_straight_branches_pull_the_junction_along_themselves(self) -> None:
        # Straight branches meeting at right angles: only their lengths
        # count, and each pulls the junction straight along itself.
        chains = pinned_at_junctions(
            [
                branch("left", (0.0, 0.0), (-2.0, 0.0), (-4.0, 0.0)),
                branch("right", (0.0, 0.0), (2.0, 0.0), (4.0, 0.0)),
                branch("up", (0.0, 0.0), (0.0, 2.0), (0.0, 4.0)),
            ]
        )
        polylines = [c.centreline.points for c in chains]
        masks = [np.array([False, False, True])] * 3

        value, gradients = layer_energy(chains, polylines, masks, self.params(), 0.01)

        assert value == pytest.approx(12.0, rel=1e-12)
        for g, pull in zip(
            gradients, [(1.0, 0.0), (-1.0, 0.0), (0.0, -1.0)], strict=True
        ):
            np.testing.assert_allclose(g[0], pull, atol=1e-12)

    def test_gradient_of_the_shared_junction_matches_finite_differences(
        self,
    ) -> None:
        # The middle branch swings back toward the first beyond what each
        # leaves out of the other near the junction, so they repel there.
        chains = pinned_at_junctions(
            [
                branch("A", (0.0, 0.0), (3.0, 0.0), (6.0, 0.0)),
                branch("B", (0.0, 0.0), (1.5, 1.5), (3.0, 0.55), (4.5, 1.5)),
                branch("C", (0.0, 0.0), (0.0, -3.0), (0.0, -6.0)),
            ]
        )
        polylines = [c.centreline.points.copy() for c in chains]
        masks = [np.arange(len(p)) == len(p) - 1 for p in polylines]
        params = self.params()
        seen = seen_parts(chains, polylines, obstacle_reach(params))
        value, gradients = layer_energy(chains, polylines, masks, params, 0.01, seen)
        apart = sum(
            layer_energy([c], [p], [m], params, 0.01)[0]
            for c, p, m in zip(chains, polylines, masks, strict=True)
        )
        assert value > apart
        step = 1e-7

        def moved(axis: int, by: float) -> float:
            shifted = [p.copy() for p in polylines]
            for p in shifted:
                p[0, axis] += by
            return layer_energy(chains, shifted, masks, params, 0.01, seen)[0]

        for axis in (0, 1):
            numeric = (moved(axis, step) - moved(axis, -step)) / (2 * step)
            analytic = sum(g[0, axis] for g in gradients)
            assert analytic == pytest.approx(numeric, rel=1e-5)


@pytest.mark.parametrize("height", [1.5, 3.0, 6.0])
def test_a_junction_cannot_jump_through_another_chain_in_one_burst(
    height: float,
) -> None:
    # The arch of test_a_chain_cannot_jump_through_another_in_one_burst,
    # its apex a junction of its two halves.
    free = FreeSpace()
    apex = (5.0, height)
    left = branch("left", apex, (0.0, 0.0))
    right = branch("right", apex, (10.0, 0.0))
    bar = selected("bar", "B", free, (4.0, 0.7), (6.0, 0.7))

    result = optimize_chains([left, right, bar], scenarios.PARAMS, SPARSE)

    junction = junction_of(replace(result, chains=result.chains[:2]))
    b = result.chains[2].polyline
    assert junction[1] > 0.7
    for chain in result.chains[:2]:
        assert gap(chain.polyline, b) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE


BLOCKER_Y = 0.7
BLOCKED = replace(SPARSE, burst_maxiter=10)


@pytest.mark.parametrize("kind", ["chain", "disc"])
def test_a_junction_cannot_pass_what_lies_between_it_and_its_best_place(
    kind: str,
) -> None:
    # The branches would meet at 120 degrees below the blocker, but the
    # junction starts above it, the third branch routed around its right
    # end; at this spacing a burst's straight line there crosses it.
    if kind == "chain":
        field: ObstacleField = FreeSpace()
        others = [selected("bar", "B", field, (4.0, BLOCKER_Y), (6.0, BLOCKER_Y))]
    else:
        field = Disc(5.0, BLOCKER_Y, 0.5)
        others = []
    start = (5.0, 3.0)
    chains = [
        branch("A", start, (2.5, 2.0), (0.0, 0.0), field=field),
        branch("B", start, (7.5, 2.0), (10.0, 0.0), field=field),
        branch("C", start, (6.5, 1.5), (7.0, 0.0), (5.0, -6.0), field=field),
        *others,
    ]

    result = optimize_chains(chains, scenarios.PARAMS, BLOCKED)

    views = pinned_at_junctions(chains)
    assert all(
        c >= -INSIDE_TOLERANCE for c in min_clearances(views, result, scenarios.PARAMS)
    )
    a, b, c = (r.polyline for r in result.chains[:3])
    junction = Point(junction_of(replace(result, chains=result.chains[:3])))
    for first, second in itertools.combinations((a, b, c), 2):
        assert LineString(first).intersection(LineString(second)).equals(junction)
    enclosed = Polygon(np.concatenate([a[::-1], c[1:]]))
    assert enclosed.contains(Point(5.0, BLOCKER_Y))
