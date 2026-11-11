"""Problems of several chains and junctions through
`optimize.optimize_chains`.

The problem's joint energy and gradient are checked against a reference
built chain by chain (`fields`), and the shared junction's gradient against
finite differences."""

import itertools
import math
import re
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest
import scenarios
from conftest import BurstClock, Ticking, clearances
from distance_fields import Disc, FloatArray, FreeSpace
from scenarios import ChainsScenario, selected
from shapely.geometry import LineString, Point

import kibends.optimize as optimize_module
from kibends.bursts import Burst, Objective, OptimizerSettings, Revert, Stop
from kibends.chain import NM_PER_MM, Segment, extract_chains
from kibends.commit import snapped_points
from kibends.coords import BOARD_FRAME
from kibends.energy import (
    EnergyParams,
    ObstacleField,
    total_energy,
)
from kibends.members import layer_energy, seen_parts
from kibends.neighbours import ChainInput, chain_field, neighbour_field
from kibends.obstacles import ChainCentreline
from kibends.optimize import (
    INSIDE_TOLERANCE,
    MAX_ABNORMAL_BURSTS,
    PROGRESS_INTERVAL,
    optimize_chains,
)
from kibends.polylines import resample
from kibends.results import RunResult, Termination
from kibends.simplify import simplify_chains

COARSE = OptimizerSettings(spacing=0.3, settled_change=0.003, burst_maxiter=30)


EQUAL_PITCH = 0.6
"""How far apart the tracks of an equally spaced bus lie, mm, as in issue
106's reproduction."""


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
    chains: Sequence[ChainInput], polylines: Sequence[FloatArray]
) -> list[ObstacleField]:
    """Each chain's field with the others on its layer at `polylines`, built
    chain by chain as a reference for the problem's energy."""
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
        for c, field in zip(result.chains, fields(chains, polylines), strict=True)
    )


def min_clearances(
    chains: Sequence[ChainInput], result: RunResult, params: EnergyParams
) -> list[float]:
    """Each chain's smallest clearance margin, with the other chains at their
    returned shapes."""
    polylines = [c.polyline for c in result.chains]
    return [
        float(clearances(p, field).min())
        for p, field in zip(polylines, fields(chains, polylines), strict=True)
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
            for p, m, f in zip(polylines, masks, fields(chains, polylines), strict=True)
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

    def mixed(self) -> tuple[ChainInput, ...]:
        """`chains` with a chain of A's net crossing A twice and reaching
        into B's clearance, so that it sees B and not A."""
        crossing = selected(
            "A2", "A", FreeSpace(), (0.5, -0.5), (1.5, 0.7), (2.5, -0.5)
        )
        return (*self.chains(), crossing)

    def test_chains_of_one_net_add_nothing_to_each_other(self) -> None:
        a, _, crossing = self.mixed()
        chains = (a, crossing)
        polylines = [c.centreline.points for c in chains]
        masks = [c.fixed for c in chains]

        value, gradients = layer_energy(chains, polylines, masks, self.params(), 0.01)

        for chain, points, mask, gradient in zip(
            chains, polylines, masks, gradients, strict=True
        ):
            alone, alone_gradient = layer_energy(
                [chain], [points], [mask], self.params(), 0.01
            )
            value -= alone
            np.testing.assert_allclose(gradient, alone_gradient[0], atol=1e-12)
        assert value == pytest.approx(0.0, abs=1e-12)

    def test_gradient_of_a_chain_midway_between_two_matches_finite_differences(
        self,
    ) -> None:
        """Three chains exactly EQUAL_PITCH apart: the middle one's points
        are as near to either neighbour, where the nearest alone would give
        the gradient of one side."""
        free = FreeSpace()
        xs = (0.0, 0.5, 1.0, 1.5, 2.0)
        chains = tuple(
            selected(net, net, free, *((x, k * EQUAL_PITCH) for x in xs))
            for k, net in enumerate("ABC")
        )
        polylines = [c.centreline.points.copy() for c in chains]
        masks = [c.fixed for c in chains]
        params = self.params()
        seen = seen_parts(chains, polylines)
        _, gradients = layer_energy(chains, polylines, masks, params, 0.01, seen)
        step = 1e-7

        for i in range(len(polylines)):
            for j in np.flatnonzero(~masks[i]):
                for axis in (0, 1):
                    shifted = [p.copy() for p in polylines]
                    shifted[i][j, axis] += step
                    up = layer_energy(chains, shifted, masks, params, 0.01, seen)[0]
                    shifted[i][j, axis] -= 2 * step
                    down = layer_energy(chains, shifted, masks, params, 0.01, seen)[0]
                    numeric = (up - down) / (2 * step)
                    assert gradients[i][j, axis] == pytest.approx(
                        numeric, rel=1e-5, abs=1e-7
                    )

    def test_gradient_with_chains_of_one_net_and_another(self) -> None:
        chains = self.mixed()
        polylines = [c.centreline.points.copy() for c in chains]
        masks = [c.fixed for c in chains]
        params = self.params()
        seen = seen_parts(chains, polylines)
        value, gradients = layer_energy(chains, polylines, masks, params, 0.01, seen)
        alone = sum(
            layer_energy([c], [p], [m], params, 0.01)[0]
            for c, p, m in zip(chains, polylines, masks, strict=True)
        )
        # B repels both chains of net A.
        assert value > alone
        step = 1e-7

        for i in range(len(polylines)):
            for j in np.flatnonzero(~masks[i]):
                for axis in (0, 1):
                    shifted = [p.copy() for p in polylines]
                    shifted[i][j, axis] += step
                    up = layer_energy(chains, shifted, masks, params, 0.01, seen)[0]
                    shifted[i][j, axis] -= 2 * step
                    down = layer_energy(chains, shifted, masks, params, 0.01, seen)[0]
                    numeric = (up - down) / (2 * step)
                    assert gradients[i][j, axis] == pytest.approx(
                        numeric, rel=1e-5, abs=1e-7
                    )


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


def test_moves_into_another_chains_clearance_are_undone_and_counted() -> None:
    # The hinge is too weak to hold the chains apart: only moving back the
    # edges that newly enter the other chain's clearance does.
    scenario = scenarios.squeeze()
    params = replace(scenario.params, clearance_weight=0.1)

    result = optimize_chains(scenario.chains, params, COARSE)

    a, b = (c.polyline for c in result.chains)
    assert gap(a, b) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE
    assert all(chain.rejected_edges > 0 for chain in result.chains)


PRESSED_SETTINGS = replace(COARSE, spacing=0.1, max_iterations=4)
"""Fine enough that the arches' vertices near their pins lie deep inside
each other's clearance."""


def test_resampling_keeps_vertices_pressed_into_another_chains_clearance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two arches of different nets 0.2 mm apart, half their inflation, are
    # pinned at both ends: near the pins their vertices stay deep inside
    # each other's clearance however the middles move apart, and resampling
    # keeps those where the burst before left them (`_Problem.pressed`).
    real = optimize_module.run_burst
    bursts: list[tuple[FloatArray, FloatArray]] = []

    def capturing(
        objective: Objective,
        start: FloatArray,
        revert: Revert,
        settings: OptimizerSettings,
        stop: Stop | None = None,
    ) -> Burst:
        burst = real(objective, start, revert, settings, stop)
        bursts.append((start.reshape(-1, 2).copy(), burst.points.reshape(-1, 2)))
        return burst

    monkeypatch.setattr(optimize_module, "run_burst", capturing)
    free = FreeSpace()
    chains = [
        selected("A", "A", free, (0.0, 0.0), (5.0, 1.0), (10.0, 0.0)),
        selected("B", "B", free, (0.0, 0.2), (5.0, 1.2), (10.0, 0.2)),
    ]

    optimize_chains(chains, scenarios.PARAMS, PRESSED_SETTINGS)

    kept = [
        point
        for (_, end), (start, _) in itertools.pairwise(bursts)
        for point in end
        if np.any(np.all(start == point, axis=1))
    ]
    assert kept


def test_a_chain_wrapped_around_an_obstacle_stays_on_its_side() -> None:
    topology = scenarios.topology()
    chain = selected("t", "N", topology.field, (-5.0, 0.0), (0.0, 2.0), (5.0, 0.0))

    (result,) = optimize_chains([chain], topology.params, COARSE).chains

    assert y_at(result.polyline, 0.0) > 0.0
    assert clearances(result.polyline, topology.field).min() >= -INSIDE_TOLERANCE
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


def kept_vertices(polyline: FloatArray, vertices: FloatArray) -> int:
    """How many of `vertices` `polyline` still has exactly."""
    return sum(bool(np.all(polyline == v, axis=1).any()) for v in vertices)


def test_vertices_inside_a_neighbours_clearance_move_out() -> None:
    scenario = scenarios.crowded()

    result = run(scenario)

    a, b = result.chains
    assert kept_vertices(a.polyline, np.array([scenarios.CROWDED_VERTEX])) == 0
    assert (
        kept_vertices(b.polyline, np.array([scenarios.CROWDED_NEIGHBOUR_VERTEX])) == 0
    )
    for c in min_clearances(scenario.chains, result, scenario.params):
        assert c >= -INSIDE_TOLERANCE


TIGHT_WIDTH = 0.25
TIGHT_CLEARANCE = 0.2
TIGHT_PITCH = 0.43
"""0.02 mm closer than the tracks' width and clearance ask, mm."""
FAN_PITCH = 0.8
"""How far apart the tracks' pinned ends lie, clear of each other, mm."""


def tight_bus() -> list[ChainInput]:
    """Four tracks of as many nets, in free space, fanning in from their ends
    to run 5 mm side by side TIGHT_PITCH apart."""
    return [
        ChainInput(
            ChainCentreline(
                f"S{i}",
                f"S{i}",
                TIGHT_WIDTH,
                TIGHT_CLEARANCE,
                scenarios.path(
                    (0.0, FAN_PITCH * i),
                    (3.0, TIGHT_PITCH * i),
                    (8.0, TIGHT_PITCH * i),
                    (11.0, FAN_PITCH * i),
                ),
            ),
            "F.Cu",
            scenarios.endpoints_fixed(np.zeros(4)),
            FreeSpace(),
        )
        for i in range(4)
    ]


def test_a_bus_tighter_than_its_clearance_spreads_out() -> None:
    chains = tight_bus()

    result = optimize_chains(chains, scenarios.PARAMS, COARSE)

    for chain, given, c in zip(
        result.chains,
        chains,
        min_clearances(chains, result, scenarios.PARAMS),
        strict=True,
    ):
        assert c >= -INSIDE_TOLERANCE
        corners = given.centreline.points[1:-1]
        assert kept_vertices(chain.polyline, corners) == 0


def on_layer(chain: ChainInput, layer: str) -> ChainInput:
    return replace(chain, layer=layer)


def test_time_limit_stops_in_the_burst_that_passes_it() -> None:
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


def recorded_bursts(monkeypatch: pytest.MonkeyPatch) -> list[Burst]:
    """The bursts the optimizer runs, as `run_burst` returns them."""
    bursts: list[Burst] = []
    real = optimize_module.run_burst

    def recording(
        objective: Objective,
        start: FloatArray,
        revert: Revert,
        settings: OptimizerSettings,
        stop: Stop | None = None,
    ) -> Burst:
        burst = real(objective, start, revert, settings, stop)
        bursts.append(burst)
        return burst

    monkeypatch.setattr(optimize_module, "run_burst", recording)
    return bursts


TIMED_ITERATIONS = 2
"""The L-BFGS-B iterations a burst runs before a clock that ticks a second
a reading, from 0, passes a limit half a second beyond: the run reads it
when setup begins and ends, then after each iteration."""


EVERY_CHANGE = 100.0
"""A settle threshold, mm, beyond any change of the scenarios here."""


def test_time_limit_ends_the_burst_at_the_iteration_past_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = bumps()
    bursts = recorded_bursts(monkeypatch)
    # Any burst would settle by its change, so only being cut keeps the run
    # from converging.
    limited = replace(
        COARSE, time_limit=TIMED_ITERATIONS + 0.5, settled_change=EVERY_CHANGE
    )

    result = optimize_chains(
        scenario.chains, scenario.params, limited, clock=Ticking(), started=0.0
    )

    (burst,) = bursts
    assert burst.cut
    assert burst.iterations == TIMED_ITERATIONS
    assert result.termination is Termination.TIME_LIMIT
    # The cut burst's state passed the guard and is the best.
    for chain in result.chains:
        assert length(chain.polyline) < chain.before.length


def test_a_burst_cut_by_the_time_limit_that_the_guard_rejects_is_not_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = bumps()
    bursts = recorded_bursts(monkeypatch)

    def rejecting(*_: object) -> bool:
        return False

    monkeypatch.setattr(optimize_module._Problem, "guard", rejecting)  # pyright: ignore[reportPrivateUsage]
    limited = replace(COARSE, time_limit=TIMED_ITERATIONS + 0.5)

    result = optimize_chains(
        scenario.chains, scenario.params, limited, clock=Ticking(), started=0.0
    )

    (burst,) = bursts
    assert burst.cut
    assert result.termination is Termination.TIME_LIMIT
    for chain, given in zip(result.chains, scenario.chains, strict=True):
        start = resample(given.centreline.points, given.fixed, COARSE.spacing)[0]
        np.testing.assert_array_equal(chain.polyline, start)


def test_a_repeated_run_past_its_time_limit_runs_its_first_bursts_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = bumps()
    bursts = recorded_bursts(monkeypatch)

    again = optimize_chains(
        scenario.chains,
        scenario.params,
        replace(COARSE, time_limit=0.0),
        clock=Ticking(),
        again=True,
    )

    assert again.termination is Termination.TIME_LIMIT
    assert not bursts[0].cut
    assert bursts[0].iterations > 1


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


FAR_APART = 64.0
"""How far a copy of a scenario lies from it, mm: beyond what either sees."""


def far_copy(chain: ChainInput) -> ChainInput:
    """`chain` FAR_APART above itself, of a net of its own."""
    moved = replace(
        chain.centreline,
        net=f"{chain.centreline.net}'",
        points=chain.centreline.points + np.array([0.0, FAR_APART]),
    )
    return replace(chain, centreline=moved)


def test_the_stall_rule_counts_a_problem_s_moving_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counted: list[tuple[int, int]] = []
    real = optimize_module.stalled

    def spy(
        lowest: Sequence[float],
        lengths: Sequence[float],
        settings: OptimizerSettings,
        members: int,
        vias: int = 0,
    ) -> bool:
        counted.append((members, vias))
        return real(lowest, lengths, settings, members, vias)

    monkeypatch.setattr(optimize_module, "stalled", spy)
    chains = bumps().chains

    optimize_chains(
        [*chains, *(far_copy(c) for c in chains)],
        scenarios.PARAMS,
        replace(COARSE, max_iterations=1),
    )

    assert counted == [(2 * len(chains), 0)]


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


# A rerun on the squeeze's own output settles within these, which vary with
# the platform's floating point. Measured: 3 iterations and 18 µm moved on
# macOS, 4 to 5 iterations and 32 µm on Linux.
REPEAT_ITERATIONS = 8
REPEAT_DRIFT = 0.06


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
    assert second.iterations <= REPEAT_ITERATIONS
    for before, chain in zip(first.chains, second.chains, strict=True):
        assert (
            LineString(chain.polyline).hausdorff_distance(LineString(before.polyline))
            < REPEAT_DRIFT
        )


def test_chains_of_one_net_closer_than_their_range_do_not_push_apart() -> None:
    # Both wrap the disc as each would alone, so they overlap over its top.
    scenario = scenarios.same_net_wrap()

    result = run(scenario)

    assert result.termination is Termination.CONVERGED
    for chain, given in zip(result.chains, scenario.chains, strict=True):
        (alone,) = optimize_chains([given], scenario.params, COARSE).chains
        assert (
            LineString(chain.polyline).hausdorff_distance(LineString(alone.polyline))
            < 10 * COARSE.settled_change
        )
    inner, outer = (c.polyline for c in result.chains)
    assert gap(inner, outer) < scenarios.PAIR_WIDTH / 2


def test_a_chain_passes_through_a_chain_of_its_net() -> None:
    # Of another net, the arch would drape over the bar.
    free = FreeSpace()
    arch = selected("arch", "N", free, (0.0, 0.0), (5.0, 3.0), (10.0, 0.0))
    bar = selected("bar", "N", free, (3.0, 1.0), (7.0, 1.0))

    result = optimize_chains([arch, bar], scenarios.PARAMS, COARSE)

    assert result.termination is Termination.CONVERGED
    straightened = result.chains[0].polyline
    assert length(straightened) < 10.0 + 0.01
    assert abs(y_at(straightened, 5.0)) < 0.01


SIDE_BY_SIDE = scenarios.PAIR_INFLATION - 0.1
"""mm between the centrelines of two straight tracks drawn side by side,
inside the clearance two tracks of different nets keep."""


def side_by_side(net: str, other_net: str) -> RunResult:
    """A straight track of `net` and a shorter one of `other_net` beside its
    middle, `SIDE_BY_SIDE` apart."""
    free = FreeSpace()
    chains = [
        selected("track", net, free, (0.0, 0.0), (5.0, 0.0), (10.0, 0.0)),
        selected("bar", other_net, free, (4.0, SIDE_BY_SIDE), (6.0, SIDE_BY_SIDE)),
    ]
    return optimize_chains(chains, scenarios.PARAMS, COARSE)


def test_tracks_of_a_diff_pair_s_two_nets_are_not_pushed_apart() -> None:
    result = side_by_side("USB_D+", "USB_D-")

    first = result.chains[0].polyline
    assert abs(y_at(first, 5.0)) < COARSE.settled_change


def test_a_diff_pair_s_track_keeps_clear_of_a_third_net() -> None:
    result = side_by_side("USB_D+", "GND")

    first, second = (c.polyline for c in result.chains)
    assert gap(first, second) >= scenarios.PAIR_INFLATION - INSIDE_TOLERANCE


def re_extracted(scenario: ChainsScenario, result: RunResult) -> list[ChainInput]:
    """The chains of `result`, simplified and committed as straight tracks
    and extracted again, as a second run takes them, each with its
    scenario's field."""
    simplified = simplify_chains(scenario.chains, result)
    segments = [
        Segment(
            f"{k}.{i}",
            (round(a[0] * NM_PER_MM), round(a[1] * NM_PER_MM)),
            (round(b[0] * NM_PER_MM), round(b[1] * NM_PER_MM)),
            round(given.centreline.width * NM_PER_MM),
            given.layer,
            given.centreline.net,
        )
        for k, (given, polyline) in enumerate(
            zip(scenario.chains, simplified, strict=True)
        )
        for i, (a, b) in enumerate(itertools.pairwise(snapped_points(polyline)))
    ]
    extracted = sorted(
        extract_chains(segments, frame=BOARD_FRAME),
        key=lambda c: int(c.segment_ids[0].split(".")[0]),
    )
    return [
        ChainInput(
            replace(given.centreline, points=np.array(c.points)),
            given.layer,
            np.array(c.pinned),
            given.field,
        )
        for given, c in zip(scenario.chains, extracted, strict=True)
    ]


def test_chains_of_one_net_left_overlapping_are_extracted_without_pins() -> None:
    scenario = scenarios.same_net_wrap()
    result = run(scenario)

    again = re_extracted(scenario, result)

    inner, outer = (c.centreline.points for c in again)
    assert gap(inner, outer) < scenarios.PAIR_WIDTH / 2
    for chain in again:
        assert chain.fixed[0] and chain.fixed[-1]
        assert not chain.fixed[1:-1].any()


def test_a_crossing_a_run_makes_neither_pins_nor_joins_on_extraction() -> None:
    # The arch comes down through the bar's middle, far from its ends.
    free = FreeSpace()
    scenario = ChainsScenario(
        "arch-through-bar",
        (
            selected("arch", "N", free, (0.0, 0.0), (5.0, 3.0), (10.0, 0.0)),
            selected("bar", "N", free, (5.0, -0.5), (5.0, 1.5)),
        ),
    )
    result = run(scenario)
    arch, bar = (c.polyline for c in result.chains)
    assert LineString(arch).crosses(LineString(bar))

    again = re_extracted(scenario, result)

    assert len(again) == 2
    for chain, given in zip(again, result.chains, strict=True):
        np.testing.assert_allclose(
            chain.centreline.points[[0, -1]], given.polyline[[0, -1]], atol=1e-6
        )
        assert chain.fixed[0] and chain.fixed[-1]
        assert not chain.fixed[1:-1].any()


def test_a_second_run_moves_chains_of_one_net_left_overlapping_apart() -> None:
    # Without the disc each straightens to its own ends, 0.4 mm apart.
    scenario = scenarios.same_net_wrap()
    again = [
        replace(c, field=FreeSpace()) for c in re_extracted(scenario, run(scenario))
    ]

    result = optimize_chains(again, scenario.params, COARSE)

    for chain in result.chains:
        ends = chain.polyline[[0, -1]]
        assert length(chain.polyline) < length(ends) + 0.01


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
    real = optimize_module.run_burst
    bursts: list[Burst] = []

    def failing(
        objective: Objective,
        start: FloatArray,
        revert: Revert,
        settings: OptimizerSettings,
        stop: Stop | None = None,
    ) -> Burst:
        burst = real(objective, start, revert, settings, stop)
        bursts.append(burst)
        return replace(burst, abnormal=True)

    monkeypatch.setattr(optimize_module, "run_burst", failing)
    scenario = bumps()

    result = run(scenario)

    assert result.iterations == len(bursts) == MAX_ABNORMAL_BURSTS
    assert result.gave_up == (("the chains on F.Cu", MAX_ABNORMAL_BURSTS),)
    assert result.abnormal == (("the chains on F.Cu", MAX_ABNORMAL_BURSTS),)
    # The bursts were taken: the run lowered the energy, but did not converge.
    assert result.termination is Termination.GAVE_UP
    for chain in result.chains:
        assert length(chain.polyline) < chain.before.length


def test_a_problem_given_up_on_beside_one_that_converges_gives_up_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = optimize_module.run_burst
    reach = 10.0

    def back_fails(
        objective: Objective,
        start: FloatArray,
        revert: Revert,
        settings: OptimizerSettings,
        stop: Stop | None = None,
    ) -> Burst:
        burst = real(objective, start, revert, settings, stop)
        return replace(burst, abnormal=bool(start.max() > reach))

    monkeypatch.setattr(optimize_module, "run_burst", back_fails)
    free = FreeSpace()
    # Only the back chain's vertices reach past `reach`.
    front = selected("front", "A", free, (0.0, 0.0), (5.0, 1.5), (reach, 0.0))
    back = selected(
        "back", "B", free, (0.0, 0.0), (reach, 1.5), (2 * reach, 0.0), layer="B.Cu"
    )

    result = optimize_chains([front, back], scenarios.PARAMS, COARSE)

    assert result.gave_up == (("the chains on B.Cu", MAX_ABNORMAL_BURSTS),)
    assert result.termination is Termination.GAVE_UP


def test_a_normal_burst_ends_an_abnormal_streak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = optimize_module.run_burst
    bursts: list[Burst] = []

    def second_normal(
        objective: Objective,
        start: FloatArray,
        revert: Revert,
        settings: OptimizerSettings,
        stop: Stop | None = None,
    ) -> Burst:
        burst = real(objective, start, revert, settings, stop)
        bursts.append(burst)
        return replace(burst, abnormal=len(bursts) != 2)

    monkeypatch.setattr(optimize_module, "run_burst", second_normal)

    # The normal burst does not settle the problem at this settled change.
    result = run(bumps(), replace(COARSE, settled_change=3e-4))

    # One abnormal burst, a normal one, then a full streak.
    assert result.iterations == len(bursts) == MAX_ABNORMAL_BURSTS + 2
    assert result.gave_up == (("the chains on F.Cu", MAX_ABNORMAL_BURSTS),)


def test_an_abnormal_burst_s_last_check_is_not_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found: list[int] = []

    def through_the_disc(
        objective: Objective,
        start: FloatArray,
        revert: Revert,
        settings: OptimizerSettings,
        stop: Stop | None = None,
    ) -> Burst:
        candidate = start.copy()
        candidate[1::2] += 1.0
        found.append(revert(candidate))
        return Burst(candidate, abnormal=True)

    monkeypatch.setattr(optimize_module, "run_burst", through_the_disc)
    chain = selected("a", "A", Disc(5.0, 1.0, 0.5), (0.0, 0.0), (5.0, 0.0), (10.0, 0.0))

    result = optimize_chains([chain], scenarios.PARAMS, COARSE)

    assert found and all(crossings > 0 for crossings in found)
    assert result.chains[0].rejected_edges == 0


PROGRESS_LINE = re.compile(
    r"^(DEBUG|INFO) Iteration (\d+), (\d+\.\d) s: length \d+\.\d{3} mm, "
    r"energy -?\d+\.\d{3}, largest change \d\.\d{2}e[-+]\d+ mm$"
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


# Positions of a straight result within this of the ideal line, mm: COARSE
# leaves the width change bowed by 2 µm.
STRAIGHT_TOLERANCE = 0.01
WIDTH_CHANGE_LENGTH = math.dist((0.0, 0.0), scenarios.WIDTH_CHANGE_END)
# mm: the width change's length matches the straight line's 20.396 mm as
# reported to three decimals.
REPORTED = 5e-4


def test_a_width_change_moves_its_junction_onto_the_straight_line() -> None:
    scenario = scenarios.width_change()

    result = run(scenario)

    assert result.termination is Termination.CONVERGED
    wide, narrow = result.chains
    np.testing.assert_array_equal(wide.polyline[-1], narrow.polyline[0])
    assert not wide.fixed[-1] and not narrow.fixed[0]
    line = LineString([(0.0, 0.0), scenarios.WIDTH_CHANGE_END])
    for c in result.chains:
        assert max(line.distance(Point(p)) for p in c.polyline) < STRAIGHT_TOLERANCE
    total = sum(length(c.polyline) for c in result.chains)
    assert total == pytest.approx(WIDTH_CHANGE_LENGTH, abs=REPORTED)


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
        c >= -INSIDE_TOLERANCE for c in min_clearances(chains, result, scenarios.PARAMS)
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

    simplified = simplify_chains(chains, result)

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
        chains = [
            branch("left", (0.0, 0.0), (-2.0, 0.0), (-4.0, 0.0)),
            branch("right", (0.0, 0.0), (2.0, 0.0), (4.0, 0.0)),
            branch("up", (0.0, 0.0), (0.0, 2.0), (0.0, 4.0)),
        ]
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
        # The middle branch swings back toward the first, which it does not
        # see, and passes a chain of another net, which it does.
        branches = [
            branch("A", (0.0, 0.0), (3.0, 0.0), (6.0, 0.0)),
            branch("B", (0.0, 0.0), (1.5, 1.5), (3.0, 0.55), (4.5, 1.5)),
            branch("C", (0.0, 0.0), (0.0, -3.0), (0.0, -6.0)),
        ]
        other = selected("D", "M", FreeSpace(), (1.5, 2.2), (3.0, 1.2), (4.5, 2.2))
        chains = [*branches, other]
        polylines = [c.centreline.points.copy() for c in chains]
        masks = [np.arange(len(p)) == len(p) - 1 for p in polylines[:3]]
        masks.append(other.fixed)
        params = self.params()
        seen = seen_parts(chains, polylines)
        value, gradients = layer_energy(chains, polylines, masks, params, 0.01, seen)

        def apart(members: Sequence[ChainInput]) -> float:
            """The members, the first of `chains`, each alone."""
            n = len(members)
            return sum(
                layer_energy([c], [p], [m], params, 0.01)[0]
                for c, p, m in zip(members, polylines[:n], masks[:n], strict=True)
            )

        assert layer_energy(branches, polylines[:3], masks[:3], params, 0.01)[
            0
        ] == pytest.approx(apart(branches), rel=1e-12)
        assert value > apart(chains)
        step = 1e-7

        def moved(axis: int, by: float) -> float:
            shifted = [p.copy() for p in polylines]
            for p in shifted[:3]:
                p[0, axis] += by
            return layer_energy(chains, shifted, masks, params, 0.01, seen)[0]

        for axis in (0, 1):
            numeric = (moved(axis, step) - moved(axis, -step)) / (2 * step)
            analytic = sum(g[0, axis] for g in gradients)
            assert analytic == pytest.approx(numeric, rel=1e-5)


BUS_TRACKS = 10
BUS_STRAIGHT = 305.941
"""The summed length of straight lines between the bus's ends, mm."""
BUS_SHORTENED = 316.0
"""A summed length the bus falls below once it moves, mm: from 324.853 mm,
where the nearest neighbour alone left it, about a third of the way to
straight."""
BUS_SETTINGS = OptimizerSettings(spacing=1.6, burst_maxiter=30, max_iterations=4)
"""Coarse settings: vertices as far apart as issue 106's faster experiment."""


def test_an_equally_spaced_bus_moves() -> None:
    """Issue 106's bus: ten Z-shaped tracks of as many nets, EQUAL_PITCH
    apart, each as near to one neighbour as to the other. No burst ends
    abnormally and the bus shortens."""
    chains = [
        selected(
            f"S{i}",
            f"S{i}",
            FreeSpace(),
            (0.0, y),
            (10.0 - y, y),
            (16.0 - y, y + 6.0),
            (30.0, y + 6.0),
        )
        for i, y in ((i, i * EQUAL_PITCH) for i in range(BUS_TRACKS))
    ]

    result = optimize_chains(
        chains, replace(scenarios.PARAMS, repulsion=0.5), BUS_SETTINGS
    )

    assert result.abnormal == ()
    length = sum(
        float(np.linalg.norm(np.diff(c.polyline, axis=0), axis=1).sum())
        for c in result.chains
    )
    assert BUS_STRAIGHT < length < BUS_SHORTENED
