import itertools
import re
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest
import scenarios
from conftest import Ticking
from distance_fields import FloatArray, FreeSpace
from scenarios import ChainsScenario, selected
from shapely.geometry import LineString

import kibends.joint as joint_module
from kibends.bursts import Burst, Objective, OptimizerSettings, Revert
from kibends.energy import (
    EnergyParams,
    ObstacleField,
    obstacle_reach,
    total_energy,
)
from kibends.joint import layer_energy, optimize_jointly
from kibends.neighbours import ChainInput, chain_field, neighbour_field
from kibends.obstacles import ChainCentreline
from kibends.optimize import (
    INSIDE_TOLERANCE,
    MAX_ABNORMAL_BURSTS,
    PROGRESS_INTERVAL,
    RunResult,
    Termination,
    measure,
    optimize_chains,
    resample,
)

COARSE = OptimizerSettings(spacing=0.3, settled_change=0.003, burst_maxiter=30)


def run(scenario: ChainsScenario, settings: OptimizerSettings = COARSE) -> RunResult:
    return optimize_jointly(scenario.chains, scenario.params, settings)


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
    """Each chain's field with the others on its layer at `polylines`, as the
    loop builds it."""
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
    """The run energy of the returned shapes, measured as the loop measures it."""
    polylines = [c.polyline for c in result.chains]
    return sum(
        total_energy(c.polyline, c.fixed, field, params, settings.short_edge)[0]
        for c, field in zip(
            result.chains, fields(chains, polylines, params), strict=True
        )
    )


def min_clearances(
    chains: Sequence[ChainInput], result: RunResult, params: EnergyParams
) -> list[float]:
    polylines = [c.polyline for c in result.chains]
    return [
        measure(p, field, COARSE.spacing).min_clearance
        for p, field in zip(polylines, fields(chains, polylines, params), strict=True)
    ]


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

    result = optimize_jointly([arch, v], scenarios.PARAMS, COARSE)

    a, b = (c.polyline for c in result.chains)
    assert not LineString(a).intersects(LineString(b))
    assert gap(a, b) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE
    assert y_at(a, 5.0) > y_at(b, 5.0)


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

    result = optimize_jointly([arch, bar], scenarios.PARAMS, SPARSE)

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

    result = optimize_jointly(scenario.chains, params, COARSE)

    a, b = (c.polyline for c in result.chains)
    assert gap(a, b) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE
    assert gap(a, b) < scenarios.PAIR_INFLATION + TIGHT
    assert all(chain.rejected_edges > 0 for chain in result.chains)


def test_a_chain_wrapped_around_an_obstacle_stays_on_its_side() -> None:
    topology = scenarios.topology()
    chain = selected("t", "N", topology.field, (-5.0, 0.0), (0.0, 2.0), (5.0, 0.0))

    joint = optimize_jointly([chain], topology.params, COARSE)
    loop = optimize_chains([chain], topology.params, COARSE)

    (result,) = joint.chains
    assert y_at(result.polyline, 0.0) > 0.0
    assert measure(result.polyline, topology.field, COARSE.spacing).min_clearance >= (
        -INSIDE_TOLERANCE
    )
    assert length(result.polyline) == pytest.approx(
        length(loop.chains[0].polyline), abs=0.01
    )


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

    result = optimize_jointly([pinned, other], scenarios.PARAMS, COARSE)

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

    result = optimize_jointly(chains, scenario.params, limited, clock=Ticking())

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

    result = optimize_jointly(
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

    result = optimize_jointly(chains, scenarios.PARAMS, COARSE)

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

    second = optimize_jointly(again, scenario.params, COARSE)

    assert second.termination in (Termination.CONVERGED, Termination.NOTHING_TO_DO)
    assert second.iterations <= COARSE.stall_window + 1
    for before, chain in zip(first.chains, second.chains, strict=True):
        assert (
            LineString(chain.polyline).hausdorff_distance(LineString(before.polyline))
            < 10 * COARSE.settled_change
        )


def test_two_chains_of_one_net_on_a_layer_are_refused() -> None:
    with pytest.raises(ValueError, match=r"inner and outer.*net N on F\.Cu"):
        run(scenarios.same_net_wrap())


def test_chains_of_one_net_on_different_layers_are_taken() -> None:
    inner, outer = scenarios.same_net_wrap().chains

    result = optimize_jointly(
        [inner, on_layer(outer, "B.Cu")], scenarios.PARAMS, COARSE
    )

    assert result.termination is Termination.CONVERGED


def test_chains_on_different_layers_do_not_interact() -> None:
    free = FreeSpace()
    across = selected("A", "A", free, (0.0, 0.0), (5.0, 2.0), (10.0, 0.0))
    along = on_layer(
        selected("B", "B", free, (5.0, -2.0), (7.0, 1.0), (5.0, 4.0)), "B.Cu"
    )

    result = optimize_jointly([across, along], scenarios.PARAMS, COARSE)

    for chain, given in zip(result.chains, (across, along), strict=True):
        alone = optimize_jointly([given], scenarios.PARAMS, COARSE)
        np.testing.assert_array_equal(chain.polyline, alone.chains[0].polyline)


def test_endpoints_must_be_fixed() -> None:
    chain = selected("A", "A", FreeSpace(), (0.0, 0.0), (5.0, 1.0), (10.0, 0.0))
    loose = replace(chain, fixed=np.array([True, False, False]))

    with pytest.raises(ValueError, match="endpoints"):
        optimize_jointly([loose], scenarios.PARAMS, COARSE)


def test_a_problem_whose_bursts_all_fail_gives_up_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = joint_module.bounded_burst
    bursts: list[Burst] = []

    def failing(
        objective: Objective,
        start: FloatArray,
        half_sides: FloatArray,
        revert: Revert,
        settings: OptimizerSettings,
    ) -> Burst:
        burst = real(objective, start, half_sides, revert, settings)
        bursts.append(burst)
        return replace(burst, abnormal=True)

    monkeypatch.setattr(joint_module, "bounded_burst", failing)
    scenario = bumps()

    result = run(scenario)

    assert result.iterations == len(bursts) == MAX_ABNORMAL_BURSTS
    assert result.gave_up == (("the chains on F.Cu", MAX_ABNORMAL_BURSTS),)
    assert result.abnormal == (("the chains on F.Cu", MAX_ABNORMAL_BURSTS),)
    assert all(chain.abnormal_bursts == 0 for chain in result.chains)
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
    result = optimize_jointly(
        bumps().chains, scenarios.PARAMS, COARSE, clock=Ticking(step=0.0)
    )

    lines = progress(log)
    assert [iteration for _, iteration, _ in lines] == list(
        range(1, result.iterations + 1)
    )
    assert {level for level, _, _ in lines} == {"DEBUG"}


def test_progress_at_info_comes_at_most_once_per_interval(log: list[str]) -> None:
    clock = Ticking(step=PROGRESS_INTERVAL / 2)

    optimize_jointly(scenarios.squeeze().chains, scenarios.PARAMS, COARSE, clock=clock)

    levels = [level for level, _, _ in progress(log)]
    assert "INFO" in levels
    assert all(not (a == "INFO" and b == "INFO") for a, b in itertools.pairwise(levels))


# Joint may end above the loop's energy by this much, an energy in mm of
# track length: the two discretise the result differently.
EQUIVALENCE_TOLERANCE = 0.01


@pytest.mark.parametrize(
    "scenario",
    [scenarios.channel_pair(), scenarios.tent(), scenarios.squeeze()],
    ids=lambda s: s.name,
)
def test_joint_ends_no_higher_than_the_loop(scenario: ChainsScenario) -> None:
    joint = run(scenario)
    loop = optimize_chains(scenario.chains, scenario.params, COARSE)

    joint_energy = committed_energy(scenario.chains, joint, scenario.params)
    loop_energy = committed_energy(scenario.chains, loop, scenario.params)

    assert joint_energy <= loop_energy + EQUIVALENCE_TOLERANCE
    assert all(
        c >= -INSIDE_TOLERANCE
        for c in min_clearances(scenario.chains, joint, scenario.params)
    )
