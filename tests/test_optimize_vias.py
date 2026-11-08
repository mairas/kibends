# pyright: reportPrivateUsage=false
import math
from dataclasses import replace

import numpy as np
import pytest
import scenarios
from distance_fields import Box, Disc, FloatArray, FreeSpace, Walls
from scenarios import ViasScenario
from shapely.geometry import LineString, Point

import kibends.optimize as optimize_module
import kibends.via_moves as via_moves
from kibends.bursts import OptimizerSettings
from kibends.energy import ObstacleField
from kibends.obstacles import HOLE_CLEARANCE, HOLE_TO_HOLE
from kibends.optimize import INSIDE_TOLERANCE, RunResult, Termination, optimize_chains
from kibends.via_moves import Carried, MovingVias, ViaBurst

COARSE = OptimizerSettings(spacing=0.3, burst_maxiter=30)


def run(scenario: ViasScenario, settings: OptimizerSettings = COARSE) -> RunResult:
    return optimize_chains(scenario.chains, scenario.params, settings, scenario.vias)


def length(points: FloatArray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def route_length(result: RunResult, first: int = 0) -> float:
    return sum(length(result.chains[i].polyline) for i in (first, first + 1))


def to_polyline(at: tuple[float, float], points: FloatArray) -> float:
    return float(Point(at).distance(LineString(points)))


@pytest.fixture(scope="module")
def detour() -> RunResult:
    return run(scenarios.via_detour())


class TestDetour:
    def test_the_via_moves_to_shorten_the_route(self, detour: RunResult) -> None:
        (via,) = detour.vias
        x, y = via.position

        assert detour.termination is Termination.CONVERGED
        # Flat at the optimum: 0.1 mm off the line costs 4 µm of length.
        assert y == pytest.approx(0.0, abs=0.1)
        assert route_length(detour) == pytest.approx(10.0, abs=0.01)
        assert x == pytest.approx(5.0, abs=0.5)

    def test_the_chain_ends_follow_the_via(self, detour: RunResult) -> None:
        (via,) = detour.vias
        top, bottom = (c.polyline for c in detour.chains)

        assert tuple(top[-1]) == via.position
        assert tuple(bottom[0]) == via.position
        assert tuple(top[0]) == scenarios.ROUTE_START
        assert tuple(bottom[-1]) == scenarios.ROUTE_END

    def test_same_inputs_give_the_same_result(self, detour: RunResult) -> None:
        again = run(scenarios.via_detour())

        assert again.vias == detour.vias
        for a, b in zip(again.chains, detour.chains, strict=True):
            assert np.array_equal(a.polyline, b.polyline)


def test_chains_on_one_layer_move_with_their_via_together() -> None:
    result = run(scenarios.via_on_one_layer())

    (via,) = result.vias
    assert result.termination is Termination.CONVERGED
    assert via.position[1] == pytest.approx(0.0, abs=0.1)


def test_a_via_at_its_optimum_stays() -> None:
    result = run(scenarios.via_detour(scenarios.STRAIGHT_VIA))

    (via,) = result.vias
    tolerance = COARSE.displacement_tolerance * COARSE.spacing
    assert math.dist(via.position, scenarios.STRAIGHT_VIA) < tolerance


@pytest.mark.parametrize(
    "clearance_weight",
    [scenarios.PARAMS.clearance_weight, 0.5, 0.0],
    ids=["default", "pulled-harder-than-held", "held-by-the-bubble-alone"],
)
def test_a_track_on_another_layer_hems_the_via_in(clearance_weight: float) -> None:
    track = scenarios.HEMMING_TRACK
    params = replace(scenarios.PARAMS, clearance_weight=clearance_weight)

    result = run(scenarios.via_hemmed(params))

    (via,) = result.vias
    distance, _ = track.signed_distance(np.array([via.position]))
    assert distance[0] >= -INSIDE_TOLERANCE
    assert via.position[1] > track.cy
    assert route_length(result) < route_length(run_unmoved(scenarios.via_hemmed()))


def run_unmoved(scenario: ViasScenario) -> RunResult:
    return optimize_chains(scenario.chains, scenario.params, COARSE)


def test_a_move_cutting_an_edge_across_a_corner_is_undone() -> None:
    box = scenarios.CORNER_BOX

    result = run(scenarios.via_corner())

    (via,) = result.vias
    top = result.chains[0].polyline
    assert via.rejected_edges > 0
    assert via.position[1] < scenarios.DETOUR_VIA[1] - 2.0
    assert box.segments_clear(top[:-1], top[1:]).all()


def test_a_neighbour_keeps_clear_of_where_the_via_is_now() -> None:
    result = run(scenarios.via_beside_chain())

    (via,) = result.vias
    neighbour = result.chains[2].polyline
    gap = to_polyline(via.position, neighbour)
    assert via.position[1] < scenarios.DETOUR_VIA[1] - 2.0
    assert gap >= scenarios.VIA_TO_CHAIN - INSIDE_TOLERANCE
    # Pulled tight against the via where it is: a chain blind to the move
    # cuts through it, and the run keeps an earlier, slack shape instead.
    assert gap < scenarios.VIA_TO_CHAIN + 0.05


# The via and the chain pressing on it settle where the route's pull down
# balances the chain's push up, near y = 0.8 at any spacing. Moved block by
# block, each held the other to the hinge's give, and the pair crept a few
# µm per outer iteration from y = 0.17: 200 iterations, 33 s at default
# settings. Measured: 18 to 21 iterations here, by platform.
CONTACT_ITERATIONS = 30


def test_a_via_and_a_chain_pressing_on_it_settle_together() -> None:
    result = run(scenarios.via_beside_chain())

    (via,) = result.vias
    assert result.termination is Termination.CONVERGED
    assert result.iterations <= CONTACT_ITERATIONS
    assert via.position[1] > 0.7


# Nudges that each stalled the via halfway, near y = 2.1, on one platform or
# another: it settled on lows its block had before the chain joined it.
@pytest.mark.parametrize("nudge", [0.0, 1e-5, -1e-4, 1e-3, -1e-3, 1e-2])
def test_a_via_does_not_stall_when_a_chain_joins_its_burst(nudge: float) -> None:
    result = run(scenarios.via_across_chain(nudge))

    (via,) = result.vias
    assert result.termination is Termination.CONVERGED
    assert via.position[1] < 0.1


def test_three_chains_meet_where_the_via_joins_them_shortest() -> None:
    result = run(scenarios.via_fermat())

    (via,) = result.vias
    assert result.termination is Termination.CONVERGED
    assert math.dist(via.position, scenarios.FERMAT_POINT) < 0.1


def started(scenario: ViasScenario) -> optimize_module._Run:
    """A run of `scenario` before its first step."""
    return optimize_module._Run(
        scenario.chains,
        scenario.params,
        COARSE,
        moving_vias(scenario),
    )


def moving_vias(scenario: ViasScenario) -> MovingVias:
    return MovingVias(
        scenario.chains, scenario.vias, scenario.params, HOLE_CLEARANCE, HOLE_TO_HOLE
    )


def straightened(run: optimize_module._Run, via: tuple[float, float]) -> ViaBurst:
    """A proposed move of the run's via to `via` with both route chains
    straight from their far ends to it."""
    at = np.array(via)
    top, bottom = run.current[0], run.current[1]
    return ViaBurst(
        at,
        0,
        {
            0: np.linspace(top[0], at, len(top)),
            1: np.linspace(at, bottom[-1], len(bottom)),
        },
        {0: run.masks[0], 1: run.masks[1]},
    )


class TestTakingAViaMove:
    def test_a_shorter_route_is_taken(self) -> None:
        run = started(scenarios.via_detour())

        assert run.take_via_move(0, straightened(run, (5.0, 1.0))) > 0.0
        assert tuple(run.positions[0]) == (5.0, 1.0)

    def test_a_move_raising_the_run_energy_is_undone(self) -> None:
        run = started(scenarios.via_detour())
        before = [points.copy() for points in run.current]

        assert run.take_via_move(0, straightened(run, (5.0, 6.0))) == 0.0
        assert tuple(run.positions[0]) == scenarios.DETOUR_VIA
        for points, then in zip(run.current, before, strict=True):
            assert np.array_equal(points, then)

    def test_a_move_adding_a_violation_is_undone(self) -> None:
        # Thin enough that the shorter route's energy falls through it.
        bar = Box(2.0, 0.0, 0.01, 1.0)
        run = started(scenarios.via_detour_over(bar))

        assert run.take_via_move(0, straightened(run, (5.0, 0.0))) == 0.0
        assert tuple(run.positions[0]) == scenarios.DETOUR_VIA


class TestSettlingAfterAViaMove:
    """A via's move settles the chains it moved only while it still carries
    or presses on them; one it left behind keeps its own bursts."""

    @pytest.mark.parametrize("touching", [False, True])
    def test_a_moved_neighbour_settles_only_while_the_via_presses_on_it(
        self, touching: bool
    ) -> None:
        run = started(scenarios.via_beside_chain())
        at = np.array([5.0, 1.0])
        neighbour = run.current[2].copy()
        if touching:
            # Just inside the reach of the via's energy terms.
            neighbour[1:-1, 1] = at[1] - run.vias.ring(0, 2) - 0.015
        burst = replace(
            straightened(run, (5.0, 1.0)),
            polylines={**straightened(run, (5.0, 1.0)).polylines, 2: neighbour},
            anchors={0: run.masks[0], 1: run.masks[1], 2: run.masks[2]},
        )

        assert run.take_via_move(0, burst) > 0.0
        assert run.rests[2].settled is touching


class TestContactsToCarry:
    def test_a_chain_with_edges_as_long_as_the_ring_is_wide_is_left_out(
        self,
    ) -> None:
        scenario = scenarios.via_beside_chain()
        vias = moving_vias(scenario)
        positions = [np.array([5.0, 0.0])]
        ring = vias.ring(0, 2)
        below = -(ring + 0.01)
        fine = np.column_stack([np.linspace(3.0, 7.0, 41), np.full(41, below)])
        coarse = np.array([(3.0, below), (7.0, below)])

        def contacts(neighbour: FloatArray) -> list[int]:
            polylines = [c.centreline.points for c in scenario.chains[:2]]
            return vias.contacts(0, positions, [*polylines, neighbour], {2}, COARSE)

        assert contacts(fine) == [2]
        assert contacts(coarse) == []


class TestContactPassage:
    """A burst leaves a via and a chain pressing on it out of each other's
    bubbles, so a check catches either passing through the other."""

    RING = 0.6
    CHAIN = np.array([(3.0, -0.65), (4.0, -0.62), (5.0, -0.61), (6.0, -0.62)])
    VIA = np.array([5.0, 0.0])

    def kept(self, via_step: tuple[float, float], chain_step: float) -> bool:
        contact = via_moves._Contact.before(self.VIA, self.CHAIN, self.RING)
        moved = self.CHAIN + np.array([0.0, chain_step])
        return contact.kept(self.VIA + np.array(via_step), moved)

    def test_pushing_together_is_kept(self) -> None:
        assert self.kept((0.0, 0.3), 0.3)

    def test_the_via_jumping_the_chain_is_caught(self) -> None:
        assert not self.kept((0.0, -2.0), 0.0)

    def test_the_chain_jumping_the_via_is_caught(self) -> None:
        assert not self.kept((0.0, 0.0), 2.0)

    def test_the_two_swapping_is_caught(self) -> None:
        assert not self.kept((0.0, -1.0), 1.0)

    def test_a_chain_stretched_wider_than_the_ring_is_caught(self) -> None:
        contact = via_moves._Contact.before(self.VIA, self.CHAIN, self.RING)
        stretched = self.CHAIN.copy()
        stretched[2] = (5.0, -3.0)

        assert not contact.kept(self.VIA, stretched)

    def test_a_burst_moves_both_back_when_the_via_jumps_the_chain(self) -> None:
        free = np.array([False, True, True, False])
        nothing = np.empty((0, 2))
        chain = Carried(0, self.CHAIN, free, [], nothing, FreeSpace(), self.RING)
        start = np.concatenate([self.VIA, self.CHAIN[free].ravel()])
        candidate = start.copy()
        candidate[:2] = (5.0, -2.0)

        _, crossings = via_moves._revert_carried_crossings(
            candidate,
            start,
            [chain],
            [slice(2, len(start))],
            [np.ones(len(self.CHAIN) - 1, dtype=bool)],
            [via_moves._Contact.before(self.VIA, self.CHAIN, self.RING)],
        )

        assert crossings > 0
        assert np.array_equal(candidate, start)


class TestPair:
    def test_two_vias_keep_their_copper_clearance(self) -> None:
        result = run(scenarios.via_pair())

        a, b = (v.position for v in result.vias)
        assert math.dist(a, b) >= scenarios.VIA_TO_VIA - INSIDE_TOLERANCE
        assert math.dist(a, b) < scenarios.VIA_TO_VIA + 0.1

    def test_two_vias_keep_hole_to_hole(self) -> None:
        scenario = scenarios.via_pair()
        hole_to_hole = 1.2

        result = optimize_chains(
            scenario.chains,
            scenario.params,
            COARSE,
            scenario.vias,
            hole_to_hole=hole_to_hole,
        )

        a, b = (v.position for v in result.vias)
        apart = scenarios.VIA_DRILL + hole_to_hole
        assert apart > scenarios.VIA_TO_VIA
        assert math.dist(a, b) >= apart - INSIDE_TOLERANCE


def test_a_via_keeps_off_a_chain_of_its_net_that_does_not_end_at_it() -> None:
    result = run(scenarios.via_over_same_net())

    (via,) = result.vias
    inner = result.chains[2].polyline
    touching = scenarios.VIA_DIAMETER / 2 + scenarios.PAIR_WIDTH / 2
    x, y = via.position
    # It may press the chain down, the two moving together, but never cross it.
    assert to_polyline(via.position, inner) >= touching - INSIDE_TOLERANCE
    assert float(np.interp(x, inner[:, 0], inner[:, 1])) < y


def test_a_chain_of_its_net_that_does_not_end_at_it_keeps_off_the_via() -> None:
    result = run(scenarios.via_over_bowed_same_net())

    (via,) = result.vias
    inner = result.chains[2].polyline
    touching = scenarios.VIA_DIAMETER / 2 + scenarios.PAIR_WIDTH / 2
    assert result.termination is Termination.CONVERGED
    assert to_polyline(via.position, inner) >= touching - INSIDE_TOLERANCE
    # Wrapped tight under the via; blind to it, the chain cut through it and
    # the run kept it slack at 5.22 mm.
    assert length(inner) < 4.4


@pytest.mark.parametrize(
    "field",
    [FreeSpace(), Walls(2.1), Walls(2.2)],
    ids=["free", "walls-2.1", "walls-2.2"],
)
def test_chains_straighten_under_a_via_that_stays(field: ObstacleField) -> None:
    result = run(scenarios.via_slack(field))

    (via,) = result.vias
    assert result.termination is Termination.CONVERGED
    # The route is straight, so the via may slide along it.
    assert math.dist(via.position, scenarios.SLACK_VIA) < 0.05
    assert route_length(result) == pytest.approx(10.0, abs=0.01)


def test_chains_straightened_in_one_burst_are_kept() -> None:
    # Two free vertices a chain: its first burst straightens it, and the
    # via's moves it less than the tolerance, so every block rests after the
    # first iteration with the chains far from where they started.
    few = OptimizerSettings(spacing=2.5, burst_maxiter=30)

    result = run(scenarios.via_slack(FreeSpace(), (5.0, 0.01)), few)

    assert result.termination is Termination.CONVERGED
    assert route_length(result) == pytest.approx(10.0, abs=0.01)


def test_a_via_starting_deep_in_its_clearance_stays() -> None:
    scenario = scenarios.via_detour()
    (via,) = scenario.vias
    buried = replace(via, field=Disc(5.0, 3.0, 1.0))

    result = optimize_chains(scenario.chains, scenario.params, COARSE, (buried,))

    (moved,) = result.vias
    assert moved.frozen
    assert moved.position == scenarios.DETOUR_VIA
