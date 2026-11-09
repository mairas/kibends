# pyright: reportPrivateUsage=false
import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest
import scenarios
from distance_fields import Box, Disc, FloatArray, FreeSpace, Walls
from scenarios import ViasScenario, selected
from shapely.geometry import LineString, Point, box
from shapely.geometry.base import BaseGeometry

import kibends.optimize as optimize_module
from kibends.bursts import Burst, Objective, OptimizerSettings, Revert
from kibends.energy import ObstacleField
from kibends.neighbours import ChainInput
from kibends.obstacles import (
    HOLE_CLEARANCE,
    HOLE_TO_HOLE,
    build_via_obstacles,
    own_net_stops,
)
from kibends.optimize import (
    INSIDE_TOLERANCE,
    MAX_ABNORMAL_BURSTS,
    RunResult,
    Termination,
    optimize_chains,
)
from kibends.pair_blocks import PairInput, pair_block
from kibends.snapshot import LayerSnapshot, ViaZone
from kibends.via_moves import MovingVias
from kibends.vias import ChainEnd, ViaInput

COARSE = OptimizerSettings(spacing=0.3, settled_change=0.003, burst_maxiter=30)


def run(
    scenario: ViasScenario,
    settings: OptimizerSettings = COARSE,
    start: RunResult | None = None,
) -> RunResult:
    return optimize_chains(
        scenario.chains, scenario.params, settings, scenario.vias, start=start
    )


def length(points: FloatArray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def route_length(result: RunResult, first: int = 0) -> float:
    return sum(length(result.chains[i].polyline) for i in (first, first + 1))


def to_polyline(at: tuple[float, float], points: FloatArray) -> float:
    return float(Point(at).distance(LineString(points)))


def y_at(points: FloatArray, x: float) -> float:
    return float(np.interp(x, points[:, 0], points[:, 1]))


@pytest.fixture(scope="module")
def detour() -> RunResult:
    return run(scenarios.via_detour())


class TestDetour:
    def test_the_via_moves_to_shorten_the_route(self, detour: RunResult) -> None:
        (via,) = detour.vias
        x, y = via.position

        assert detour.termination is Termination.CONVERGED
        assert not via.frozen
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


TOP_OFFSET = np.array([0.1, -0.05])
BOTTOM_OFFSET = np.array([-0.08, 0.12])


def off_centre() -> ViasScenario:
    """The detour route with each chain ending off the via's centre, within
    its copper."""
    at = np.array(scenarios.DETOUR_VIA)
    top_end = tuple(at + TOP_OFFSET)
    bottom_start = tuple(at + BOTTOM_OFFSET)
    free = FreeSpace()
    return ViasScenario(
        "via-off-centre",
        (
            selected("N top", "N", free, scenarios.ROUTE_START, top_end),
            selected(
                "N bottom", "N", free, bottom_start, scenarios.ROUTE_END, layer="B.Cu"
            ),
        ),
        (scenarios.joining(scenarios.through_via("v", "N", scenarios.DETOUR_VIA), 0),),
    )


def test_ends_off_the_via_s_centre_keep_their_offsets_exactly() -> None:
    result = run(off_centre())

    given = off_centre()
    (via,) = result.vias
    at = np.array(via.position)
    start = np.array(scenarios.DETOUR_VIA)
    top, bottom = (c.polyline for c in result.chains)
    assert at[1] < 1.0
    assert np.array_equal(top[-1], at + (given.chains[0].centreline.points[-1] - start))
    assert np.array_equal(
        bottom[0], at + (given.chains[1].centreline.points[0] - start)
    )


def test_chains_on_one_layer_move_with_their_via_together() -> None:
    result = run(scenarios.via_on_one_layer())

    (via,) = result.vias
    assert result.termination is Termination.CONVERGED
    assert via.position[1] == pytest.approx(0.0, abs=0.1)


def test_a_via_at_its_optimum_stays() -> None:
    result = run(scenarios.via_detour(scenarios.STRAIGHT_VIA))

    (via,) = result.vias
    assert math.dist(via.position, scenarios.STRAIGHT_VIA) < COARSE.settled_change


@pytest.mark.parametrize(
    "clearance_weight",
    [scenarios.PARAMS.clearance_weight, 0.5, 0.0],
    ids=["default", "pulled-harder-than-held", "held-by-the-bubble-alone"],
)
def test_a_track_on_a_layer_without_chains_hems_the_via_in(
    clearance_weight: float,
) -> None:
    track = scenarios.HEMMING_TRACK
    params = replace(scenarios.PARAMS, clearance_weight=clearance_weight)

    result = run(scenarios.via_hemmed(params))

    (via,) = result.vias
    distance, _ = track.signed_distance(np.array([via.position]))
    assert distance[0] >= -INSIDE_TOLERANCE
    assert via.position[1] > track.cy
    assert via.position[1] < scenarios.DETOUR_VIA[1] - 1.0


def in_zone(bottom: float, *others: ViaZone) -> ViasScenario:
    """The detour route with its via in a zone of its net whose region ends at
    y = `bottom`, short of the straight route at y = 0, and in `others`."""
    via = scenarios.through_via("v", "N", scenarios.DETOUR_VIA)
    return ViasScenario(
        "via-in-zone",
        scenarios.route("N", scenarios.DETOUR_VIA),
        (replace(scenarios.joining(via, 0), zones=(plane(bottom), *others)),),
    )


def plane(bottom: float) -> ViaZone:
    """A zone with a minimum width of 0.2 mm whose region ends at y = `bottom`."""
    return ViaZone("plane", {"In1.Cu": box(3, bottom, 7, 5)}, 0.2, 0.0)


MARGIN = plane(0).margin


class TestInsideAZone:
    def test_the_via_moves_within_the_region(self) -> None:
        result = run(in_zone(-1.0))

        (via,) = result.vias
        assert not via.frozen
        assert via.position[1] == pytest.approx(0.0, abs=0.1)

    def test_a_pull_toward_the_region_s_edge_is_held_at_it(self) -> None:
        bottom = 1.5

        result = run(in_zone(bottom))

        (via,) = result.vias
        assert not via.frozen
        # The clearance hinge, which starts `standoff` from the margin, holds
        # it against the pull of the route.
        edge = bottom + MARGIN
        assert edge - INSIDE_TOLERANCE <= via.position[1]
        assert via.position[1] <= edge + scenarios.PARAMS.standoff

    def test_the_via_moves_within_the_union_of_its_zones(self) -> None:
        # `plane` ends short of the straight route, `big` reaches past it.
        big = ViaZone("big", {"In1.Cu": box(3, -1, 7, 5)}, 0.2, 0.0)

        result = run(in_zone(1.5, big))

        (via,) = result.vias
        assert not via.frozen
        assert via.position[1] == pytest.approx(0.0, abs=0.1)

    def test_the_via_crosses_the_seam_of_zones_meeting_edge_to_edge(self) -> None:
        # Neither zone alone holds the via near y = 1.5 by its margin; their
        # union does.
        below = ViaZone("below", {"In1.Cu": box(3, -1, 7, 1.5)}, 0.2, 0.0)

        result = run(in_zone(1.5, below))

        (via,) = result.vias
        assert not via.frozen
        assert via.position[1] == pytest.approx(0.0, abs=0.1)

    def test_a_pull_toward_the_union_s_edge_is_held_at_it(self) -> None:
        bottom = 1.5
        inner = ViaZone("inner", {"In1.Cu": box(3, 2.5, 7, 5)}, 0.2, 0.0)

        result = run(in_zone(bottom, inner))

        (via,) = result.vias
        assert not via.frozen
        edge = bottom + MARGIN
        assert edge - INSIDE_TOLERANCE <= via.position[1]
        assert via.position[1] <= edge + scenarios.PARAMS.standoff

    def test_a_via_whose_centre_starts_shallower_than_the_margin_stays(self) -> None:
        result = run(in_zone(scenarios.DETOUR_VIA[1] - MARGIN / 2))

        (via,) = result.vias
        assert via.frozen
        assert via.position == scenarios.DETOUR_VIA


def layer(name: str, via_keepouts: tuple[BaseGeometry, ...] = ()) -> LayerSnapshot:
    return LayerSnapshot(
        layer=name,
        pads=(),
        tracks=(),
        vias=(),
        graphics=(),
        track_keepouts=(),
        via_keepouts=via_keepouts,
        edge_paths=(),
        clearances={},
        default_clearance=scenarios.PAIR_CLEARANCE,
        diff_pair_gaps={},
    )


def below_keepout(top: float, on: str = "B.Cu") -> ViasScenario:
    """The detour route with a rule area keeping out vias on layer `on`,
    whose top edge is at y = `top`, across the straight route at y = 0."""
    via = scenarios.through_via("v", "N", scenarios.DETOUR_VIA)
    keepout = box(3, -1, 7, top)
    layers = [layer(name, (keepout,) if name == on else ()) for name in via.span]
    return ViasScenario(
        "via-keepout",
        scenarios.route("N", scenarios.DETOUR_VIA),
        (scenarios.joining(via, 0, build_via_obstacles(layers, via)),),
    )


class TestViaKeepout:
    def test_a_pull_toward_it_stops_at_its_edge(self) -> None:
        top = 1.5

        result = run(below_keepout(top))

        (via,) = result.vias
        assert not via.frozen
        # Held where its ring meets the keepout: the clearance hinge, which
        # starts `standoff` from that edge, holds it against the pull of the
        # route.
        edge = top + scenarios.VIA_DIAMETER / 2
        assert edge - INSIDE_TOLERANCE <= via.position[1]
        assert via.position[1] <= edge + scenarios.PARAMS.standoff

    def test_a_via_starting_inside_one_stays(self) -> None:
        result = run(below_keepout(scenarios.DETOUR_VIA[1] + 0.1))

        (via,) = result.vias
        assert via.frozen
        assert via.position == scenarios.DETOUR_VIA


def test_a_move_cutting_an_edge_across_a_corner_is_undone() -> None:
    box = scenarios.CORNER_BOX

    result = run(scenarios.via_corner())

    (via,) = result.vias
    top = result.chains[0].polyline
    # The chain's own vertices wrap the corner as the via pulls it down.
    assert result.chains[0].rejected_edges > 0
    assert via.position[1] < scenarios.DETOUR_VIA[1] - 2.0
    assert box.segments_clear(top[:-1], top[1:]).all()


def first_revert(
    monkeypatch: pytest.MonkeyPatch, scenario: ViasScenario
) -> tuple[Revert, FloatArray]:
    """The check of the first burst of `scenario`'s run, and where the burst
    starts; its one via's centre is the last variable."""
    real = optimize_module.bounded_burst
    captured: list[tuple[Revert, FloatArray]] = []

    def capturing(
        objective: Objective,
        start: FloatArray,
        half_sides: FloatArray,
        revert: Revert,
        settings: OptimizerSettings,
        retry_bounds: FloatArray | None = None,
    ) -> Burst:
        captured.append((revert, start.copy()))
        return real(objective, start, half_sides, revert, settings, retry_bounds)

    monkeypatch.setattr(optimize_module, "bounded_burst", capturing)
    run(scenario, replace(COARSE, max_iterations=1))
    return captured[0]


@pytest.mark.parametrize(
    ("scenario", "to"),
    [
        # Its end edge on F.Cu cuts across the post, both ends clear of it.
        (scenarios.via_detour_over(Box(4.9, 2.0, 0.05, 0.05)), (5.0, 1.5)),
        # Onto the pinned chain of another net on In1.Cu.
        (scenarios.via_over_pinned("P"), (5.0, scenarios.PINNED_LEVEL)),
    ],
    ids=["end-edge-crossing", "newly-inside"],
)
def test_a_burst_moves_the_via_and_its_ends_back(
    monkeypatch: pytest.MonkeyPatch, scenario: ViasScenario, to: tuple[float, float]
) -> None:
    revert, start = first_revert(monkeypatch, scenario)
    candidate = start.copy()
    candidate[-2:] = to

    reverted, crossings = revert(candidate)

    assert crossings > 0
    assert reverted[-2:].all()
    # Not by the fallback that moves everything back.
    assert not reverted.all()
    assert np.array_equal(candidate, start)


@pytest.fixture(scope="module")
def beside() -> RunResult:
    return run(scenarios.via_beside_chain())


class TestBesideAChain:
    def test_the_neighbour_keeps_clear_of_the_via_and_wraps_it(
        self, beside: RunResult
    ) -> None:
        (via,) = beside.vias
        neighbour = beside.chains[2].polyline
        gap = to_polyline(via.position, neighbour)

        assert via.position[1] < scenarios.DETOUR_VIA[1] - 2.0
        assert gap >= scenarios.VIA_TO_CHAIN - INSIDE_TOLERANCE
        assert gap < scenarios.VIA_TO_CHAIN + 0.05

    def test_the_via_and_the_chain_pressing_on_it_settle_together(
        self, beside: RunResult
    ) -> None:
        (via,) = beside.vias

        assert beside.termination is Termination.CONVERGED
        assert via.position[1] > 0.7


@pytest.mark.parametrize("nudge", [0.0, -1e-3])
def test_a_via_is_not_held_back_by_a_chain_it_passes(nudge: float) -> None:
    result = run(scenarios.via_across_chain(nudge))

    (via,) = result.vias
    assert result.termination is Termination.CONVERGED
    assert via.position[1] < 0.1


def test_three_chains_meet_where_the_via_joins_them_shortest() -> None:
    result = run(scenarios.via_fermat())

    (via,) = result.vias
    assert result.termination is Termination.CONVERGED
    assert math.dist(via.position, scenarios.FERMAT_POINT) < 0.1


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


ZONE_CLEARANCE = 0.5
"""More than the scenarios' netclass clearance, so that it is the rule."""


def zoned(scenario: ViasScenario, k: int, layer: str) -> ViasScenario:
    """`scenario` with via k inside a wide zone of its net on `layer` whose
    clearance is ZONE_CLEARANCE."""
    zone = ViaZone("plane", {layer: box(-20, -20, 30, 20)}, 0.2, ZONE_CLEARANCE)
    vias = list(scenario.vias)
    vias[k] = replace(vias[k], zones=(zone,))
    return replace(scenario, vias=tuple(vias))


class TestZoneClearanceToMembers:
    def test_a_chain_on_the_zone_s_layer_keeps_it_from_the_via(self) -> None:
        result = run(zoned(scenarios.via_beside_chain(), 0, "In1.Cu"))

        (via,) = result.vias
        gap = to_polyline(via.position, result.chains[2].polyline)
        expected = scenarios.PAIR_WIDTH / 2 + ZONE_CLEARANCE + MARGIN
        assert expected > scenarios.VIA_TO_CHAIN
        assert gap >= expected - INSIDE_TOLERANCE
        assert gap < expected + 0.05

    def test_a_chain_on_another_layer_keeps_the_netclass_clearance(self) -> None:
        result = run(zoned(scenarios.via_beside_chain(), 0, "In2.Cu"))

        (via,) = result.vias
        gap = to_polyline(via.position, result.chains[2].polyline)
        assert gap < scenarios.VIA_TO_CHAIN + 0.05

    def test_another_moving_via_keeps_it(self) -> None:
        result = run(zoned(scenarios.via_pair(), 0, "In1.Cu"))

        a, b = (v.position for v in result.vias)
        expected = scenarios.VIA_DIAMETER / 2 + ZONE_CLEARANCE + MARGIN
        assert expected > scenarios.VIA_TO_VIA
        assert math.dist(a, b) >= expected - INSIDE_TOLERANCE
        assert math.dist(a, b) < expected + 0.1

    def test_a_via_inside_another_s_zone_distance_holds_only_that_one(
        self,
    ) -> None:
        # 0.85 mm apart: more than the netclass distance, less than the
        # zoned via's. The zoned via stays, as an unselected via, and the
        # other is free to go.
        lower = (5.0, -2.0)
        upper = (5.0, -2.0 + 0.85)
        level = scenarios.PAIR_LEVEL
        scenario = ViasScenario(
            "via-in-zone-distance",
            (
                *scenarios.route("A", lower),
                *scenarios.route("B", upper, (0.0, level), (10.0, level)),
            ),
            (
                scenarios.joining(scenarios.through_via("a", "A", lower), 0),
                scenarios.joining(scenarios.through_via("b", "B", upper), 2),
            ),
        )

        result = run(zoned(scenario, 0, "In1.Cu"))

        a, b = result.vias
        assert a.frozen and a.position == lower
        assert not b.frozen
        assert b.position[1] > upper[1] + 0.5

    def test_two_vias_keep_the_larger_of_their_distances_both_ways(self) -> None:
        vias = moving_vias(zoned(scenarios.via_pair(), 0, "In1.Cu"))

        expected = scenarios.VIA_DIAMETER / 2 + ZONE_CLEARANCE + MARGIN
        assert vias.apart(0, 1) == pytest.approx(expected)
        assert vias.apart(1, 0) == pytest.approx(expected)


VIA_TO_CHAIN = scenarios.VIA_TO_CHAIN


def test_a_via_passes_over_a_chain_of_its_net_that_does_not_end_at_it() -> None:
    # Of another net, the pinned chain would hold it above.
    result = run(scenarios.via_over_pinned("N"))

    (via,) = result.vias
    assert result.termination is Termination.CONVERGED
    assert via.position[1] == pytest.approx(0.0, abs=0.1)
    assert route_length(result) == pytest.approx(10.0, abs=0.01)


# Pulled onto the chain against a weak clearance hinge, the via ends within
# this much of its clearance, mm.
TIGHT = 2e-3


def over_short_pinned() -> ViasScenario:
    """`scenarios.via_over_pinned` of another net with the pinned chain only
    under the via's way, pinned at fewer points."""
    scenario = scenarios.via_over_pinned("P")
    inner = scenario.chains[2]
    count = 25
    points = np.column_stack(
        [np.linspace(2.0, 8.0, count), np.full(count, scenarios.PINNED_LEVEL)]
    )
    short = replace(
        inner,
        centreline=replace(inner.centreline, points=points),
        fixed=np.ones(count, dtype=bool),
    )
    return replace(scenario, chains=(*scenario.chains[:2], short))


def test_a_move_leaving_the_via_inside_its_clearance_is_undone() -> None:
    # The hinge is too weak to hold the via off the pinned chain: only moving
    # it back when it newly enters its clearance does.
    scenario = over_short_pinned()
    params = replace(scenario.params, clearance_weight=0.1)
    shorter = replace(COARSE, burst_maxiter=15)

    result = optimize_chains(scenario.chains, params, shorter, scenario.vias)

    (via,) = result.vias
    gap = to_polyline(via.position, result.chains[2].polyline)
    assert gap >= VIA_TO_CHAIN - INSIDE_TOLERANCE
    assert gap < VIA_TO_CHAIN + TIGHT
    assert via.rejected_edges > 0


# The chain wraps the via and pushes it up as it does. Each held to half
# their gap, the two crept after each other: 81 iterations. Measured: 25.
PRESSING_ITERATIONS = 40


def test_a_chain_of_another_net_keeps_off_the_via() -> None:
    result = run(scenarios.via_over_bowed("P"))

    (via,) = result.vias
    inner = result.chains[2].polyline
    assert result.termination is Termination.CONVERGED
    assert result.iterations <= PRESSING_ITERATIONS
    assert to_polyline(via.position, inner) >= VIA_TO_CHAIN - INSIDE_TOLERANCE


TOUCHING = scenarios.VIA_DIAMETER / 2 + scenarios.PAIR_WIDTH / 2


def test_a_chain_of_its_net_that_does_not_end_at_it_passes_through_the_via() -> None:
    scenario = scenarios.via_over_bowed("N")

    result = run(scenario)

    (via,) = result.vias
    inner = result.chains[2].polyline
    straight = length(scenario.chains[2].centreline.points[[0, -1]])
    assert result.termination is Termination.CONVERGED
    assert length(inner) < straight + 0.01
    assert to_polyline(via.position, inner) < TOUCHING


@pytest.mark.parametrize(
    "field",
    [FreeSpace(), Walls(2.1)],
    ids=["free", "walls-2.1"],
)
def test_chains_straighten_under_a_via_that_stays(field: ObstacleField) -> None:
    result = run(scenarios.via_slack(field))

    (via,) = result.vias
    assert result.termination is Termination.CONVERGED
    assert math.dist(via.position, scenarios.SLACK_VIA) < 0.05
    assert route_length(result) == pytest.approx(10.0, abs=0.01)


def buried() -> ViasScenario:
    """The detour route with its via starting deep in its clearance."""
    scenario = scenarios.via_detour()
    (via,) = scenario.vias
    return replace(scenario, vias=(replace(via, field=Disc(5.0, 3.0, 1.0)),))


def test_a_via_starting_deep_in_its_clearance_stays() -> None:
    result = run(buried())

    (via,) = result.vias
    top, bottom = (c.polyline for c in result.chains)
    assert via.frozen
    assert via.position == scenarios.DETOUR_VIA
    assert tuple(top[-1]) == tuple(bottom[0]) == scenarios.DETOUR_VIA


def test_a_junction_at_a_via_that_stays_stays_on_it() -> None:
    at = (5.0, 3.0)

    def loose(chain: ChainInput, end: int) -> ChainInput:
        fixed = chain.fixed.copy()
        fixed[end] = False
        return replace(chain, fixed=fixed)

    chains = [
        loose(selected("N a", "N", FreeSpace(), (0.0, 0.0), at), -1),
        loose(selected("N b", "N", FreeSpace(), at, (10.0, 0.0)), 0),
        loose(selected("N c", "N", FreeSpace(), at, (5.0, -4.0)), 0),
        selected("N bottom", "N", FreeSpace(), at, (10.0, 5.0), layer="B.Cu"),
    ]
    ends = (ChainEnd(0, True), ChainEnd(1, False), ChainEnd(2, False))
    via = ViaInput(
        scenarios.through_via("v", "N", at),
        (*ends, ChainEnd(3, False)),
        Disc(*at, 1.0),
        scenarios.PAIR_CLEARANCE,
    )

    result = optimize_chains(chains, scenarios.PARAMS, COARSE, [via])

    assert result.vias[0].frozen
    a, b, c, _ = (chain.polyline for chain in result.chains)
    assert tuple(a[-1]) == tuple(b[0]) == tuple(c[0]) == at


def test_layers_without_a_moving_via_stay_independent() -> None:
    scenario = buried()
    params = scenario.params

    result = run(scenario)

    for chain, given in zip(result.chains, scenario.chains, strict=True):
        alone = optimize_chains([given], params, COARSE)
        np.testing.assert_array_equal(chain.polyline, alone.chains[0].polyline)


def test_two_layers_joined_by_a_via_are_one_problem(
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

    result = run(scenarios.via_detour())

    problem = "the chains and vias on F.Cu and B.Cu"
    assert result.iterations == len(bursts) == MAX_ABNORMAL_BURSTS
    assert result.gave_up == ((problem, MAX_ABNORMAL_BURSTS),)
    assert result.abnormal == ((problem, MAX_ABNORMAL_BURSTS),)
    assert result.vias[0].position[1] < scenarios.DETOUR_VIA[1]


@pytest.mark.parametrize(("moved_back", "settles"), [(0, True), (1, False)])
def test_a_via_moved_back_keeps_its_problem_from_settling(
    monkeypatch: pytest.MonkeyPatch, moved_back: int, settles: bool
) -> None:
    def still(
        self: optimize_module._Problem, starts: list[FloatArray]
    ) -> optimize_module._ProblemBurst:
        count = len(starts)
        return optimize_module._ProblemBurst(
            list(starts),
            list(self.positions),
            [0] * count,
            [0] * count,
            [moved_back] * len(self.vias),
            False,
        )

    monkeypatch.setattr(optimize_module._Problem, "burst", still)
    # Too few bursts for the best energy to stall.
    settings = replace(COARSE, max_iterations=COARSE.stall_window - 1)

    result = run(scenarios.via_detour(), settings)

    assert (result.termination is not Termination.ITERATION_CAP) == settles


SPARSE = replace(COARSE, spacing=3.0, settled_change=0.03)
BAR_LEVEL = 0.7


def via_over_bar(height: float) -> ViasScenario:
    """The detour route with its via at `height` over a long bar of another
    net on an inner layer, fixed, that the route's pull would drag it
    through."""
    bar = selected("bar", "B", FreeSpace(), (1.0, BAR_LEVEL), (9.0, BAR_LEVEL))
    via = (5.0, height)
    return ViasScenario(
        "via-over-bar",
        (*scenarios.route("N", via), replace(bar, layer="In1.Cu")),
        (scenarios.joining(scenarios.through_via("v", "N", via), 0),),
    )


@pytest.mark.parametrize("height", [1.5, 3.0, 6.0])
def test_a_via_cannot_jump_through_a_chain_in_one_burst(height: float) -> None:
    # The via's straight line to the route's shortest shape passes the bar
    # and ends clear beyond it: only its bubble against the bar keeps the
    # burst from taking it.
    result = run(via_over_bar(height), SPARSE)

    (via,) = result.vias
    assert via.position[1] > BAR_LEVEL
    gap = to_polyline(via.position, result.chains[2].polyline)
    assert gap >= scenarios.VIA_TO_CHAIN - INSIDE_TOLERANCE


ARCH_FOOT = -1.0


def arch_over_via(height: float) -> ViasScenario:
    """A straight route along y = 0 with its via at (5, 0), and an arch of
    another net on an inner layer from below the via up over it and down
    again, whose shortest shape lies clear below the via."""
    arch = selected(
        "arch",
        "A",
        FreeSpace(),
        (3.0, ARCH_FOOT),
        (5.0, height),
        (7.0, ARCH_FOOT),
        layer="In1.Cu",
    )
    return ViasScenario(
        "arch-over-via",
        (*scenarios.route("N", scenarios.STRAIGHT_VIA), arch),
        (
            scenarios.joining(
                scenarios.through_via("v", "N", scenarios.STRAIGHT_VIA), 0
            ),
        ),
    )


@pytest.mark.parametrize("height", [1.5, 3.0, 6.0])
def test_a_chain_cannot_jump_through_a_via_in_one_burst(height: float) -> None:
    result = run(arch_over_via(height), SPARSE)

    (via,) = result.vias
    arch = result.chains[2].polyline
    x, y = via.position
    assert y_at(arch, x) > y
    assert to_polyline(via.position, arch) >= scenarios.VIA_TO_CHAIN - INSIDE_TOLERANCE


def bump_under(height: float) -> PairInput:
    """A coupled section on In1.Cu along y = 0, bumped up to `height` under
    the detour route's via."""
    return replace(
        scenarios.coupled(
            FreeSpace(),
            (0.0, 0.0),
            (3.0, 0.0),
            (4.0, height),
            (6.0, height),
            (7.0, 0.0),
            (10.0, 0.0),
        ),
        layer="In1.Cu",
    )


def test_a_via_and_a_section_keep_clear_of_each_other() -> None:
    route = scenarios.via_detour((5.0, 2.0))
    capped = replace(COARSE, max_iterations=8)

    result = optimize_chains(
        route.chains, route.params, capped, route.vias, pairs=(bump_under(0.5),)
    )

    (via,) = result.vias
    (section,) = result.pairs
    assert via.position[1] < 1.5
    for track in (section.a, section.b):
        assert to_polyline(via.position, track) >= (
            scenarios.VIA_TO_CHAIN - INSIDE_TOLERANCE
        )
    assert y_at(section.centreline, via.position[0]) < via.position[1]


class TestGradient:
    """The energy's gradient by the via's coordinates takes in its carried
    ends, the chains it sees and that see it, and the other vias."""

    def check(
        self, scenario: ViasScenario, seed: int, pairs: tuple[PairInput, ...] = ()
    ) -> None:
        params = replace(scenario.params, repulsion=0.5, falloff_range=1.5)
        blocks = [pair_block(pair, COARSE.spacing) for pair in pairs]
        everything = [*scenario.chains, *(block.chain for block in blocks)]
        vias = MovingVias(everything, scenario.vias, HOLE_CLEARANCE, HOLE_TO_HOLE)
        (placed,) = optimize_module._problems(
            scenario.chains, blocks, vias, params, COARSE, None
        )
        problem = placed.problem
        layout = problem.layout(problem.points)
        objective = problem.objective(layout)
        rng = np.random.default_rng(seed)
        x = layout.x0 + rng.normal(scale=0.05, size=len(layout.x0))
        _, gradient = objective(x)
        step = 1e-7
        for c in range(len(x)):
            up, down = x.copy(), x.copy()
            up[c] += step
            down[c] -= step
            numeric = (objective(up)[0] - objective(down)[0]) / (2 * step)
            assert gradient[c] == pytest.approx(numeric, rel=1e-4, abs=1e-6)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_with_a_chain_near_the_via(self, seed: int) -> None:
        # Close enough for the via and the chain to repel each other.
        neighbour = selected(
            "C", "C", FreeSpace(), (3.0, 1.0), (5.0, 2.0), (7.0, 1.0), layer="In1.Cu"
        )
        scenario = scenarios.via_detour()
        self.check(replace(scenario, chains=(*scenario.chains, neighbour)), seed)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_with_chains_of_its_net_and_another_near_the_via(self, seed: int) -> None:
        # Of the via's net, crossing the other chain, which sees it.
        mine = selected(
            "N", "N", FreeSpace(), (3.0, 2.5), (5.0, 1.5), (7.0, 2.5), layer="In1.Cu"
        )
        other = selected(
            "C", "C", FreeSpace(), (3.0, 1.0), (5.0, 2.0), (7.0, 1.0), layer="In1.Cu"
        )
        scenario = scenarios.via_detour()
        self.check(replace(scenario, chains=(*scenario.chains, mine, other)), seed)

    def test_with_a_section_near_the_via(self) -> None:
        scenario = scenarios.via_detour()
        section = bump_under(1.0)
        self.check(scenario, 3, (section,))

    def test_with_another_via_near_it(self) -> None:
        lower, upper = (5.0, 1.0), (5.0, 2.2)
        scenario = ViasScenario(
            "vias-near",
            (
                *scenarios.route("A", lower),
                *scenarios.route("B", upper, (0.0, 0.7), (10.0, 0.7)),
            ),
            (
                scenarios.joining(scenarios.through_via("a", "A", lower), 0),
                scenarios.joining(scenarios.through_via("b", "B", upper), 2),
            ),
        )
        self.check(scenario, 2)

    def test_with_a_zoned_via_near_another(self) -> None:
        lower, upper = (5.0, 1.0), (5.0, 2.2)
        scenario = ViasScenario(
            "zoned-vias-near",
            (
                *scenarios.route("A", lower),
                *scenarios.route("B", upper, (0.0, 0.7), (10.0, 0.7)),
            ),
            (
                scenarios.joining(scenarios.through_via("a", "A", lower), 0),
                scenarios.joining(scenarios.through_via("b", "B", upper), 2),
            ),
        )
        self.check(zoned(scenario, 0, "In1.Cu"), 2)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_with_two_vias_of_one_net_at_their_holes_apart(self, seed: int) -> None:
        # Within the hinge's standoff of each other, near the track.
        low = 0.9
        self.check(stitched(low, low + HOLES_APART + 0.005, top=5.0), seed)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_with_a_via_at_its_zone_s_edge(self, seed: int) -> None:
        start = 0.9
        self.check(stitched(start, top=start + STITCH_MARGIN + 0.005), seed)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_with_vias_that_repel_stop_or_stand_still_and_a_zone_edge(
        self, seed: int
    ) -> None:
        # A G via frozen inside the track's clearance; a moving G via at the
        # edge of its zone, near the frozen one and an unselected G via; and
        # a moving H via within range of it and of the track.
        start = 0.9
        scenario = stitched(start, top=start + STITCH_MARGIN + 0.005)
        (moving,) = scenario.vias
        beside = (STITCH_X + HOLES_APART + 0.005, start)
        still = scenarios.through_via("u", "G", beside)
        layers = [replace(layer(name), vias=(still,)) for name in moving.via.span]
        low = (STITCH_X, scenarios.VIA_TO_CHAIN / 2)
        frozen = replace(moving, via=scenarios.through_via("f", "G", low))
        plane = {name: box(0, -5, 10, 10) for name in moving.via.span}
        other = ViaInput(
            scenarios.through_via("h", "H", (STITCH_X + 1.0, 1.3)),
            (),
            FreeSpace(),
            scenarios.PAIR_CLEARANCE,
            (ViaZone("h-plane", plane, 0.2, 0.0),),
        )
        vias = (
            frozen,
            replace(moving, own_net=own_net_stops(layers, moving.via)),
            other,
        )
        self.check(replace(scenario, vias=vias), seed)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_with_a_via_of_its_net_that_stands_still(self, seed: int) -> None:
        start = 0.9
        self.check(beside_still(start, start + HOLES_APART + 0.005), seed)


def moving_vias(scenario: ViasScenario) -> MovingVias:
    return MovingVias(scenario.chains, scenario.vias, HOLE_CLEARANCE, HOLE_TO_HOLE)


class TestResume:
    def test_goes_on_from_the_via_where_the_earlier_run_left_it(self) -> None:
        scenario = scenarios.via_detour()
        once = replace(COARSE, max_iterations=1)
        first = run(scenario, once)

        resumed = run(scenario, once, start=first)
        fresh = run(scenario, once)

        assert resumed.vias[0].position[1] < fresh.vias[0].position[1]

    def test_a_start_with_the_via_newly_inside_its_clearance_is_not_taken(
        self,
    ) -> None:
        free = run(scenarios.via_detour())
        (via,) = free.vias
        # A pad where the earlier run left the via.
        scenario = scenarios.via_detour_over(FreeSpace())
        blocked = replace(
            scenario,
            vias=(replace(scenario.vias[0], field=Disc(*via.position, 0.5)),),
        )

        resumed = run(blocked, replace(COARSE, max_iterations=1), start=free)

        (moved,) = resumed.vias
        assert math.dist(moved.position, via.position) > 0.5 - INSIDE_TOLERANCE
        assert moved.position[1] > 2.0


def test_chains_straightened_in_one_burst_are_kept() -> None:
    # Two free vertices a chain: one burst straightens the route, and the
    # via's move in it is less than the tolerance.
    few = OptimizerSettings(spacing=2.5, settled_change=0.025, burst_maxiter=30)

    result = run(scenarios.via_slack(FreeSpace(), (5.0, 0.01)), few)

    assert result.termination is Termination.CONVERGED
    assert route_length(result) == pytest.approx(10.0, abs=0.01)


STITCH_X = 5.0
STITCH_PARAMS = replace(scenarios.PARAMS, repulsion=0.5)


def stitched(*heights: float, top: float, nets: Sequence[str] = ()) -> ViasScenario:
    """A straight chain of net S on F.Cu along y = 0, pinned at every vertex
    so that it cannot bow away, and a stitching via at each of `heights`
    above it, of net G or the net `nets` gives in turn, inside a zone of its
    net on F.Cu and B.Cu whose region ends at y = `top`."""
    regions = {layer: box(0, -5, 10, top) for layer in ("F.Cu", "B.Cu")}
    zone = ViaZone("plane", regions, 0.2, 0.0)
    vertices = [(float(x), 0.0) for x in np.linspace(0.0, 10.0, 41)]
    chain = selected("S", "S", FreeSpace(), *vertices)
    return ViasScenario(
        "stitching-vias",
        (replace(chain, fixed=np.ones(len(vertices), dtype=bool)),),
        tuple(
            ViaInput(
                scenarios.through_via(f"g{k}", nets[k] if nets else "G", (STITCH_X, y)),
                (),
                FreeSpace(),
                scenarios.PAIR_CLEARANCE,
                (zone,),
            )
            for k, y in enumerate(heights)
        ),
        STITCH_PARAMS,
    )


STITCH_MARGIN = ViaZone("plane", {}, 0.2, 0.0).margin
HOLES_APART = scenarios.VIA_DRILL + HOLE_TO_HOLE
"""How far apart the centres of two vias of one net keep."""


def beside_still(at_y: float, still_y: float) -> ViasScenario:
    """`stitched` with its via at `at_y` and an unselected via of its net at
    `still_y` above it, which stands still."""
    scenario = stitched(at_y, top=still_y + 3.0)
    (given,) = scenario.vias
    still = scenarios.through_via("u", "G", (STITCH_X, still_y))
    layers = [replace(layer(name), vias=(still,)) for name in given.via.span]
    own = own_net_stops(layers, given.via)
    return replace(scenario, vias=(replace(given, own_net=own),))


class TestStitchingVia:
    def test_one_pushed_toward_a_via_of_its_net_that_stands_still_stops_at_it(
        self,
    ) -> None:
        start, still = 0.9, 2.0

        result = run(beside_still(start, still))

        (via,) = result.vias
        gap = still - via.position[1]
        assert via.position[1] > start + 0.1
        # Hole-to-hole, though their rings then overlap.
        assert HOLES_APART < scenarios.VIA_DIAMETER
        assert gap >= HOLES_APART - INSIDE_TOLERANCE
        assert gap <= HOLES_APART + STITCH_PARAMS.standoff

    def test_a_crowded_one_backs_off_to_the_edge_of_its_zone(self) -> None:
        start, top = 0.8, 1.3

        result = run(stitched(start, top=top))

        (via,) = result.vias
        assert not via.frozen
        assert via.position[0] == pytest.approx(STITCH_X, abs=0.01)
        # The zone's edge only stops it, so the track's push takes it there.
        edge = top - STITCH_MARGIN
        assert edge - STITCH_PARAMS.standoff <= via.position[1]
        assert via.position[1] <= edge + INSIDE_TOLERANCE

    def test_one_near_its_zone_s_edge_with_nothing_else_in_range_stays(
        self,
    ) -> None:
        start = scenarios.VIA_TO_CHAIN + STITCH_PARAMS.falloff_range + 0.05

        result = run(stitched(start, top=start + STITCH_MARGIN + 0.3))

        (via,) = result.vias
        assert not via.frozen
        assert via.position == pytest.approx((STITCH_X, start), abs=1e-6)

    def test_two_of_one_net_within_range_of_each_other_stay(self) -> None:
        low = scenarios.VIA_TO_CHAIN + STITCH_PARAMS.falloff_range + 0.05
        high = low + HOLES_APART + STITCH_PARAMS.falloff_range / 2

        result = run(stitched(low, high, top=high + 3.0))

        assert [v.frozen for v in result.vias] == [False, False]
        assert result.vias[0].position == pytest.approx((STITCH_X, low), abs=1e-6)
        assert result.vias[1].position == pytest.approx((STITCH_X, high), abs=1e-6)

    def test_two_of_other_nets_within_range_of_each_other_push_apart(
        self,
    ) -> None:
        low = scenarios.VIA_TO_CHAIN + STITCH_PARAMS.falloff_range + 0.05
        high = low + HOLES_APART + STITCH_PARAMS.falloff_range / 2

        result = run(stitched(low, high, top=high + 3.0, nets=("G", "H")))

        assert [v.frozen for v in result.vias] == [False, False]
        apart = result.vias[1].position[1] - result.vias[0].position[1]
        assert apart > high - low + 0.1

    def test_two_of_one_net_pushed_together_keep_their_holes_apart(self) -> None:
        # The track pushes the lower one up against the upper one, which
        # its zone's edge holds.
        top = 1.9
        result = run(stitched(0.9, 1.75, top=top))

        low, high = (v.position for v in result.vias)
        assert high[1] <= top - STITCH_MARGIN + INSIDE_TOLERANCE
        apart = math.dist(low, high)
        assert apart >= HOLES_APART - INSIDE_TOLERANCE
        assert apart <= HOLES_APART + STITCH_PARAMS.standoff
        assert low[1] > 0.9 + 0.1

    def test_one_beside_a_track_of_its_net_stays(self) -> None:
        # Of another net, the track would push it away.
        start = scenarios.VIA_TO_CHAIN + STITCH_PARAMS.falloff_range / 2

        result = run(stitched(start, top=start + 2.0, nets=("S",)))

        (via,) = result.vias
        assert not via.frozen
        assert via.position == pytest.approx((STITCH_X, start), abs=1e-6)

    def test_one_with_nothing_in_range_stays(self) -> None:
        start = scenarios.VIA_TO_CHAIN + STITCH_PARAMS.falloff_range + 0.05

        result = run(stitched(start, top=start + 2.0))

        (via,) = result.vias
        assert not via.frozen
        assert via.position == pytest.approx((STITCH_X, start), abs=1e-6)

    def test_one_starting_inside_its_clearance_stays_as_unselected(self) -> None:
        start = scenarios.VIA_TO_CHAIN / 2

        result = run(stitched(start, top=1.3))

        (via,) = result.vias
        assert via.frozen
        assert via.position == (STITCH_X, start)


def reference_via_energy(
    problem: optimize_module._Problem,
    polylines: Sequence[FloatArray],
    positions: Sequence[FloatArray],
) -> tuple[float, list[FloatArray], FloatArray]:
    """The moving vias' own terms built via by via: the full integrand for
    the nearest of the via's static field and what `via_repel` finds, and
    the hinge alone for the nearest of its stops and what `via_stop` finds."""
    gradients = [np.zeros_like(points) for points in polylines]
    via_gradients = np.zeros((len(problem.vias), 2))
    value = 0.0
    sights = (
        problem.via_repel.nearest(polylines, positions, problem.reach),
        problem.via_stop.nearest(polylines, positions, problem.reach),
    )
    for n, k in enumerate(problem.free_vias):
        given = problem.vias[k].given
        at = positions[k]
        for static, near, integrand in (
            (given.field, sights[0], problem.integrand),
            (given.stops, sights[1], problem.limit),
        ):
            distance, direction = math.inf, np.zeros(2)
            if static is not None:
                d, g = static.signed_distance(at[None], problem.reach)
                distance, direction = float(d[0]), g[0]
            nearer = float(near.distance[n]) < distance
            if nearer:
                distance, direction = float(near.distance[n]), near.gradient[n]
            term, slope = integrand(np.array([distance]))
            value += given.via.diameter * float(term[0])
            push = given.via.diameter * slope[0] * direction
            via_gradients[k] += push
            if not nearer:
                continue
            j, first = int(near.member[n]), int(near.vertex[n])
            if j < 0:
                via_gradients[first] -= push
                continue
            start, along = near.starts[n], near.ends[n] - near.starts[n]
            squared = float(along @ along)
            t = float(np.clip((at - start) @ along / squared, 0, 1)) if squared else 0
            gradients[j][first] -= (1 - t) * push
            gradients[j][first + 1] -= t * push
    return value, gradients, via_gradients


FROZEN_X = STITCH_X + 2.5
FROZEN_X_ABOVE = 1.2
"""The height of the G via above the frozen one, nearer to it than to the
chain below."""


def stitched_grid() -> ViasScenario:
    """Stitching vias of nets G and H in rows between `stitched`'s chain and
    a pinned chain of net T above its zone, close enough to stop and repel
    each other and to meet their zone's edge, the bottom row seeing the
    chain below and the top row the chain above; an unselected G via
    standing still among them; and, first of all, an H via frozen inside the
    lower chain's clearance, nearer than that chain to a G via above it."""
    heights = [0.9 + 0.55 * row for row in range(4)]
    top = heights[-1] + STITCH_MARGIN + 0.005
    scenario = stitched(*heights, FROZEN_X_ABOVE, top=top)
    still = scenarios.through_via("u", "G", (STITCH_X + HOLES_APART + 0.005, 0.9))
    frozen = scenarios.through_via("f", "H", (FROZEN_X, 0.2))
    vias = [ViaInput(frozen, (), FreeSpace(), scenarios.PAIR_CLEARANCE)]
    for k, given in enumerate(scenario.vias):
        x = STITCH_X + (0.6 if k % 2 else 0.0)
        if given.via.position[1] == FROZEN_X_ABOVE:
            x = FROZEN_X
        net = "H" if k == 2 else "G"
        via = replace(given.via, net=net, position=(x, given.via.position[1]))
        layers = [replace(layer(name), vias=(still,)) for name in via.span]
        vias.append(replace(given, via=via, own_net=own_net_stops(layers, via)))
    vertices = [(float(x), top + 0.5) for x in np.linspace(0.0, 10.0, 41)]
    above = selected("T", "T", FreeSpace(), *vertices)
    chains = (*scenario.chains, replace(above, fixed=np.ones(len(vertices), bool)))
    return replace(scenario, chains=chains, vias=tuple(vias))


VIA_ENERGY_SCENARIOS = {
    "detour": scenarios.via_detour,
    "fermat": scenarios.via_fermat,
    "hemmed": scenarios.via_hemmed,
    "beside-chain": scenarios.via_beside_chain,
    "pair": scenarios.via_pair,
    "over-same-net": lambda: scenarios.via_over_pinned("N"),
    "over-other-net": lambda: scenarios.via_over_pinned("P"),
    "zoned-beside-chain": lambda: zoned(scenarios.via_beside_chain(), 0, "In1.Cu"),
    "in-zone": lambda: in_zone(0.0),
    "below-keepout": lambda: below_keepout(2.0),
    "stitched-two-nets": lambda: stitched(1.5, 2.6, top=5.0, nets=("G", "H")),
    "stitched-holes-apart": lambda: stitched(0.9, 0.9 + HOLES_APART + 0.005, top=5.0),
    "beside-still": lambda: beside_still(0.9, 0.9 + HOLES_APART + 0.005),
    "stitched-grid": stitched_grid,
    "grid": lambda: scenarios.via_grid(3, 4),
}


class TestViaEnergy:
    """The vias' own terms equal those built via by via, at the start and
    with every vertex and via jittered."""

    @pytest.mark.parametrize("jitter", [0.0, 0.05])
    @pytest.mark.parametrize("name", VIA_ENERGY_SCENARIOS)
    def test_matches_the_via_by_via_reference(self, name: str, jitter: float) -> None:
        scenario = VIA_ENERGY_SCENARIOS[name]()
        params = replace(scenario.params, repulsion=0.5, falloff_range=1.5)
        placed = optimize_module._problems(
            scenario.chains, [], moving_vias(scenario), params, COARSE, None
        )
        problems = [p.problem for p in placed if p.problem.free_vias]
        assert problems
        rng = np.random.default_rng(7)
        for problem in problems:
            polylines = [
                p + rng.normal(scale=jitter, size=p.shape) for p in problem.points
            ]
            positions = [
                p + rng.normal(scale=jitter, size=2) for p in problem.positions
            ]

            value, gradients, via_gradients = problem.via_energy(polylines, positions)

            want, want_gradients, want_vias = reference_via_energy(
                problem, polylines, positions
            )
            assert value == pytest.approx(want, rel=1e-12, abs=1e-12)
            for got, expected in zip(gradients, want_gradients, strict=True):
                np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(via_gradients, want_vias, rtol=1e-12, atol=1e-12)
