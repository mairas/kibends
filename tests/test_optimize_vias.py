# pyright: reportPrivateUsage=false
"""Moving vias through `optimize.optimize_chains`: carried chain ends,
stitching vias, zone containment and vias that start inside their
clearance.

The problem's energy and gradient with moving vias are checked against a
reference built via by via (`reference_via_energy`)."""

import itertools
import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest
import scenarios
from conftest import (
    CURVATURE_ATOL,
    CURVATURE_RTOL,
    CURVATURE_STEP,
    build_via_obstacles,
    neutral_distance,
)
from distance_fields import BoolArray, Box, Disc, FloatArray, FreeSpace, Walls
from scenarios import ViasScenario, selected
from shapely.geometry import LineString, Point, box
from shapely.geometry.base import BaseGeometry

import kibends.optimize as optimize_module
from kibends import cli
from kibends.bursts import (
    DEFAULT_SETTINGS,
    FALLBACK_WARNING,
    INSIDE_TOLERANCE,
    Burst,
    Crossings,
    Objective,
    OptimizerSettings,
    Revert,
    Stop,
)
from kibends.energy import (
    SMOOTH_WIDTH,
    ObstacleField,
    attraction_integrand,
    combine_fields,
    obstacle_integrand,
    obstacle_reach,
    repulsion_integrand,
)
from kibends.neighbours import ChainInput
from kibends.obstacles import (
    HOLE_CLEARANCE,
    HOLE_TO_HOLE,
    ChainCentreline,
    Obstacle,
    Obstacles,
    own_net_stops,
    split_via_obstacles,
    via_via_inflation,
)
from kibends.optimize import MAX_ABNORMAL_BURSTS, optimize_chains, stalled
from kibends.pair_blocks import PairInput, pair_block
from kibends.results import RunResult, Termination
from kibends.snapshot import LayerSnapshot, ViaZone
from kibends.via_moves import MovingVias
from kibends.vias import ChainEnd, StaticVia, ViaInput, pairs_with

COARSE = OptimizerSettings(spacing=0.3, settled_change=0.003, burst_maxiter=30)


def run(scenario: ViasScenario, settings: OptimizerSettings = COARSE) -> RunResult:
    return optimize_chains(scenario.chains, scenario.params, settings, scenario.vias)


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


def squeezed(field: ObstacleField, bottom: float) -> ViasScenario:
    """The detour route with its via in `plane(bottom)` and `field` its
    static field."""
    via = scenarios.through_via("v", "N", scenarios.DETOUR_VIA)
    return ViasScenario(
        "via-squeezed",
        scenarios.route("N", scenarios.DETOUR_VIA),
        (replace(scenarios.joining(via, 0, field), zones=(plane(bottom),)),),
    )


WALL_DEPTH = 0.1
"""How far the via starts inside a wall's clearance, mm."""
ZONE_SLACK = 0.05
"""How far inside its zone's margin the via starts, mm."""


SLACK_STRAIGHT = 10.01
"""The slack chain's length once straight, at most, mm."""


def test_a_via_pushed_out_of_a_clearance_stays_inside_its_zone() -> None:
    # The wall above the zone's edge pushes the via toward that edge.
    edge = scenarios.DETOUR_VIA[1] - ZONE_SLACK
    wall = Box(5.0, scenarios.DETOUR_VIA[1] - WALL_DEPTH + 5.0, 20.0, 5.0)

    scenario = squeezed(wall, edge - MARGIN)
    # A slack chain of another net in the same problem, far from the via:
    # moving the via back, not refusing the whole burst, lets it straighten.
    slack = selected("P", "P", FreeSpace(), (0.0, -5.0), (5.0, -3.0), (10.0, -5.0))

    result = run(replace(scenario, chains=(*scenario.chains, slack)))

    (via,) = result.vias
    assert via.position[1] >= edge - INSIDE_TOLERANCE
    assert length(result.chains[-1].polyline) < SLACK_STRAIGHT


WALL_GAP = 0.02
"""How far the via starts clear of the wall's clearance, mm."""


def test_a_via_pushed_out_of_its_own_net_s_stop_keeps_clear_of_a_wall() -> None:
    # The stop below pushes the via up toward the wall of another net.
    x, y = scenarios.DETOUR_VIA
    wall = Box(x, y - WALL_GAP - 5.0, 20.0, 5.0)
    stop = Box(x, y - WALL_DEPTH + 5.0, 20.0, 5.0)
    via = scenarios.through_via("v", "N", scenarios.DETOUR_VIA)
    scenario = ViasScenario(
        "via-stopped",
        scenarios.route("N", scenarios.DETOUR_VIA),
        (replace(scenarios.joining(via, 0, wall), own_net=stop),),
    )

    result = run(scenario)

    (moved,) = result.vias
    at = np.array([moved.position])
    assert float(wall.signed_distance(at)[0][0]) >= -INSIDE_TOLERANCE


GUARD_STEP = 0.02
"""How far the guard test moves the via, mm."""


@pytest.mark.parametrize(
    ("step", "taken"),
    [
        ((-GUARD_STEP, 0.0), False),
        ((GUARD_STEP, 0.0), True),
        ((0.0, -GUARD_STEP), False),
        ((0.0, GUARD_STEP), True),
    ],
    ids=["deeper-in-field", "shallower-in-field", "deeper-in-stops", "shallower"],
)
def test_the_guard_takes_a_via_starting_inside_only_shallower(
    step: tuple[float, float], taken: bool
) -> None:
    # Inside a wall left of it, deeper than it is outside its zone below it:
    # moving along x changes only the one, along y only the other.
    x, y = scenarios.DETOUR_VIA
    wall = Box(x + WALL_DEPTH - 50.0, y, 50.0, 50.0)
    scenario = squeezed(wall, y + ZONE_SLACK - MARGIN)
    (placed,) = optimize_module._problems(
        scenario.chains, [], moving_vias(scenario), scenario.params, COARSE
    )
    problem = placed.problem

    moved = [problem.positions[0] + np.array(step)]

    assert problem.guard(problem.points, problem.masks, moved, problem.seen) is taken


class TestInsideAZone:
    def test_the_via_moves_within_the_region(self) -> None:
        result = run(in_zone(-1.0))

        (via,) = result.vias
        assert via.position[1] == pytest.approx(0.0, abs=0.1)

    def test_a_pull_toward_the_region_s_edge_is_held_at_it(self) -> None:
        bottom = 1.5

        result = run(in_zone(bottom))

        (via,) = result.vias
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
        assert via.position[1] == pytest.approx(0.0, abs=0.1)

    def test_the_via_crosses_the_seam_of_zones_meeting_edge_to_edge(self) -> None:
        # Neither zone alone holds the via near y = 1.5 by its margin; their
        # union does.
        below = ViaZone("below", {"In1.Cu": box(3, -1, 7, 1.5)}, 0.2, 0.0)

        result = run(in_zone(1.5, below))

        (via,) = result.vias
        assert via.position[1] == pytest.approx(0.0, abs=0.1)

    def test_a_pull_toward_the_union_s_edge_is_held_at_it(self) -> None:
        bottom = 1.5
        inner = ViaZone("inner", {"In1.Cu": box(3, 2.5, 7, 5)}, 0.2, 0.0)

        result = run(in_zone(bottom, inner))

        (via,) = result.vias
        edge = bottom + MARGIN
        assert edge - INSIDE_TOLERANCE <= via.position[1]
        assert via.position[1] <= edge + scenarios.PARAMS.standoff

    def test_a_via_whose_centre_starts_shallower_than_the_margin_moves_in(
        self,
    ) -> None:
        bottom = scenarios.DETOUR_VIA[1] - MARGIN / 2

        result = run(in_zone(bottom))

        (via,) = result.vias
        edge = bottom + MARGIN
        assert edge - INSIDE_TOLERANCE <= via.position[1]
        assert via.position[1] <= edge + scenarios.PARAMS.standoff


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
        # Held where its ring meets the keepout: the clearance hinge, which
        # starts `standoff` from that edge, holds it against the pull of the
        # route.
        edge = top + scenarios.VIA_DIAMETER / 2
        assert edge - INSIDE_TOLERANCE <= via.position[1]
        assert via.position[1] <= edge + scenarios.PARAMS.standoff

    def test_a_via_starting_inside_one_ends_outside_it(self) -> None:
        top = scenarios.DETOUR_VIA[1] + 0.1

        result = run(below_keepout(top))

        (via,) = result.vias
        edge = top + scenarios.VIA_DIAMETER / 2
        assert edge - INSIDE_TOLERANCE <= via.position[1]
        assert via.position[1] <= edge + scenarios.PARAMS.standoff


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
    real = optimize_module.run_burst
    captured: list[tuple[Revert, FloatArray]] = []

    def capturing(
        objective: Objective,
        start: FloatArray,
        revert: Revert,
        settings: OptimizerSettings,
        stop: Stop | None = None,
    ) -> Burst:
        captured.append((revert, start.copy()))
        return real(objective, start, revert, settings, stop)

    monkeypatch.setattr(optimize_module, "run_burst", capturing)
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
    masks: list[BoolArray] = []
    real = optimize_module.revert_until_clear

    def spying(
        candidate: FloatArray, start: FloatArray, crossings: Crossings
    ) -> tuple[BoolArray, int]:
        reverted, count = real(candidate, start, crossings)
        masks.append(reverted)
        return reverted, count

    monkeypatch.setattr(optimize_module, "revert_until_clear", spying)

    crossings = revert(candidate)

    (reverted,) = masks
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


REACHED = 0.01
"""How close, mm, a via that nothing holds comes at COARSE settings to
where its chains pull it, with ample room for round-off between machines."""


def spanning(scenario: ViasScenario, *spans: tuple[str, str]) -> ViasScenario:
    """`scenario`, whose via k joins chains 2k and 2k + 1, with via k
    spanning `spans[k]` and those chains on its two layers."""
    chains = list(scenario.chains)
    vias: list[ViaInput] = []
    for k, (given, span) in enumerate(zip(scenario.vias, spans, strict=True)):
        for i, layer in zip((2 * k, 2 * k + 1), span, strict=True):
            chains[i] = replace(chains[i], layer=layer)
        via = replace(given.via, span=span)
        vias.append(replace(given, via=via))
    return replace(scenario, chains=tuple(chains), vias=tuple(vias))


class TestBlindAndBuriedVias:
    def test_two_that_share_no_layer_pass_each_other(self) -> None:
        scenario = spanning(
            scenarios.via_pair(), ("F.Cu", "In1.Cu"), ("In2.Cu", "B.Cu")
        )

        result = run(scenario)

        a, b = (v.position for v in result.vias)
        assert moving_vias(scenario).apart(0, 1) is None
        assert math.dist(a, b) < scenarios.PAIR_LEVEL + REACHED
        assert scenarios.PAIR_LEVEL + REACHED < scenarios.VIA_TO_VIA

    @pytest.mark.parametrize("net", ["B", "A"])
    def test_two_sharing_only_an_end_layer_keep_hole_to_hole(self, net: str) -> None:
        # Of two nets the guard holds it as copper (`via_sight`), of one net
        # as a stop (`via_stop`).
        lower, upper = (5.0, -2.0), (5.0, scenarios.PAIR_LEVEL + 2.0)
        level = scenarios.PAIR_LEVEL
        pair = ViasScenario(
            "via-pair",
            (
                *scenarios.route("A", lower),
                *scenarios.route(net, upper, (0.0, level), (10.0, level)),
            ),
            (
                scenarios.joining(scenarios.through_via("a", "A", lower), 0),
                scenarios.joining(scenarios.through_via("b", net, upper), 2),
            ),
        )
        scenario = spanning(pair, ("F.Cu", "In1.Cu"), ("In1.Cu", "B.Cu"))
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
        assert math.dist(a, b) >= apart - INSIDE_TOLERANCE

    def test_a_chain_on_a_layer_it_does_not_reach_does_not_push_it(self) -> None:
        # The neighbour on In1.Cu holds a through via above y = 0.7.
        scenario = spanning(scenarios.via_beside_chain(), ("In2.Cu", "B.Cu"))

        result = run(scenario)

        (via,) = result.vias
        gap = to_polyline(via.position, result.chains[2].polyline)
        assert via.position[1] < REACHED
        assert gap < scenarios.NEIGHBOUR_LEVEL + REACHED
        assert scenarios.NEIGHBOUR_LEVEL + REACHED < scenarios.VIA_TO_CHAIN

    def test_a_buried_one_pushed_toward_its_zone_s_edge_stays_inside(self) -> None:
        # `stitched`'s crowding chain on In1.Cu, and its zone on In2.Cu alone.
        start, top = 0.8, 1.3
        scenario = stitched(start, top=top)
        (chain,), (given,) = scenario.chains, scenario.vias
        via = replace(given.via, span=("In1.Cu", "In2.Cu"))
        zone = ViaZone("plane", {"In2.Cu": box(0, -5, 10, top)}, 0.2, 0.0)
        buried = replace(
            scenario,
            chains=(replace(chain, layer="In1.Cu"),),
            vias=(replace(given, via=via, zones=(zone,)),),
        )

        result = run(buried)

        (moved,) = result.vias
        edge = top - STITCH_MARGIN
        assert edge - STITCH_PARAMS.standoff <= moved.position[1]
        assert moved.position[1] <= edge + INSIDE_TOLERANCE


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

    def test_two_vias_inside_one_s_zone_distance_move_apart(self) -> None:
        # 0.85 mm apart: more than the netclass distance, less than the
        # zoned via's.
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

        given = zoned(scenario, 0, "In1.Cu")

        result = run(given)

        a, b = (v.position for v in result.vias)
        assert a != lower
        apart = moving_vias(given).apart(0, 1)
        assert apart is not None and apart > math.dist(lower, upper)
        assert math.dist(a, b) >= apart - INSIDE_TOLERANCE

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
    assert via.rejected_edges > 0


PUSHING_DOWN = Disc(5.0, 3.9, 1.0)
"""Static copper whose clearance the detour's via starts 0.1 mm deep in,
from above, so that it pushes the via toward the pinned chain."""


def test_a_via_starting_inside_one_obstacle_is_checked_against_the_others() -> None:
    scenario = over_short_pinned()
    (given,) = scenario.vias
    params = replace(scenario.params, clearance_weight=0.1)
    shorter = replace(COARSE, burst_maxiter=15)

    result = optimize_chains(
        scenario.chains, params, shorter, [replace(given, field=PUSHING_DOWN)]
    )

    (via,) = result.vias
    gap = to_polyline(via.position, result.chains[2].polyline)
    assert via.position != scenarios.DETOUR_VIA
    assert gap >= VIA_TO_CHAIN - INSIDE_TOLERANCE
    assert via.rejected_edges > 0


INSIDE_PINNED = 0.3
"""How deep the via starts inside the clearance of the chain pinned above
it, mm."""
INTO_WALL = 0.1
"""How far the sideways move takes the via into the wall's clearance, mm;
less than INSIDE_PINNED."""


def beside_a_wall_under_pinned() -> ViasScenario:
    """The detour's via INSIDE_PINNED inside the clearance of a pinned chain
    of another net running along x above it, and a wall of static copper
    left of it that it starts clear of."""
    x, y = scenarios.DETOUR_VIA
    scenario = over_short_pinned()
    inner = scenario.chains[2]
    level = y + VIA_TO_CHAIN - INSIDE_PINNED
    points = inner.centreline.points.copy()
    points[:, 1] = level
    wall = Box(x - 0.1 - 5.0, y, 5.0, 50.0)
    (given,) = scenario.vias
    return replace(
        scenario,
        chains=(
            *scenario.chains[:2],
            replace(inner, centreline=replace(inner.centreline, points=points)),
        ),
        vias=(replace(given, field=wall),),
    )


SIDEWAYS = (scenarios.DETOUR_VIA[0] - 0.1 - INTO_WALL, scenarios.DETOUR_VIA[1])
"""Into the wall's clearance, as deep inside the pinned chain's as before."""


def test_a_burst_moves_back_a_via_newly_inside_one_obstacle_while_inside_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revert, start = first_revert(monkeypatch, beside_a_wall_under_pinned())
    candidate = start.copy()
    candidate[-2:] = SIDEWAYS

    crossings = revert(candidate)

    assert crossings > 0
    assert np.array_equal(candidate[-2:], start[-2:])


def test_the_guard_refuses_a_via_newly_inside_one_obstacle_while_inside_another() -> (
    None
):
    scenario = beside_a_wall_under_pinned()
    (placed,) = optimize_module._problems(
        scenario.chains, [], moving_vias(scenario), scenario.params, COARSE
    )
    problem = placed.problem

    moved = [np.array(SIDEWAYS)]

    assert not problem.guard(problem.points, problem.masks, moved, problem.seen)


def inside_one_item_beside_another(stops: bool = False) -> ViasScenario:
    """`beside_a_wall_under_pinned` with the pinned chain and the wall two
    items of the via's static copper, or of its own net's stops."""
    x, y = scenarios.DETOUR_VIA
    level = y + VIA_TO_CHAIN - INSIDE_PINNED
    items = Obstacles(
        [
            Obstacle("track", LineString([(0.0, level), (10.0, level)]), VIA_TO_CHAIN),
            Obstacle("wall", box(x - 0.1 - 10.0, y - 10.0, x - 0.1, y + 10.0), 0.0),
        ]
    )
    scenario = scenarios.via_detour()
    (given,) = scenario.vias
    placed = replace(given, own_net=items) if stops else replace(given, field=items)
    return replace(scenario, vias=(placed,))


STEM = 1.0
"""How far along the track above the via a branch of it leaves, mm: the
via starts clear of the branch."""
STEM_EXTENT = scenarios.PAIR_WIDTH / 2
"""How far the copper of the track and its branch reaches beyond their
centrelines, mm."""


def inside_a_track_beside_its_branch(
    branch_net: str, branch_layer: str = "F.Cu"
) -> ViasScenario:
    """The detour's via INSIDE_PINNED inside the clearance of a static track
    of net T on F.Cu running along x above it, and a branch of net
    `branch_net` on `branch_layer` leaving the track STEM along it, away
    from the via: of net T on F.Cu, a T-junction, which the energy keeps
    two obstacles."""
    x, y = scenarios.DETOUR_VIA
    level = y + VIA_TO_CHAIN - INSIDE_PINNED
    items = Obstacles(
        [
            Obstacle(
                "track",
                LineString([(0.0, level), (10.0, level)]),
                VIA_TO_CHAIN,
                "T",
                STEM_EXTENT,
                "F.Cu",
            ),
            Obstacle(
                "branch",
                LineString([(x + STEM, level), (x + STEM, level + 5.0)]),
                VIA_TO_CHAIN,
                branch_net,
                STEM_EXTENT,
                branch_layer,
            ),
        ]
    )
    scenario = scenarios.via_detour()
    (given,) = scenario.vias
    return replace(scenario, vias=(replace(given, field=items),))


UNDER_THE_BRANCH = (scenarios.DETOUR_VIA[0] + STEM, scenarios.DETOUR_VIA[1])
"""Along the track under where its branch leaves, as deep inside the
track's clearance as before and so inside the branch's."""


BRANCHES = [("T", "F.Cu", False), ("U", "F.Cu", True), ("T", "B.Cu", True)]
"""Each branch's net and layer, and whether a via moving under it from
inside the track's clearance enters new copper's."""
BRANCH_IDS = ["same-net", "other-net", "other-layer"]


@pytest.mark.parametrize(
    ("branch_net", "branch_layer", "back"), BRANCHES, ids=BRANCH_IDS
)
def test_a_burst_judges_a_via_against_a_track_and_its_branch_as_one(
    monkeypatch: pytest.MonkeyPatch,
    log: list[str],
    branch_net: str,
    branch_layer: str,
    back: bool,
) -> None:
    revert, start = first_revert(
        monkeypatch, inside_a_track_beside_its_branch(branch_net, branch_layer)
    )
    candidate = start.copy()
    candidate[-2:] = UNDER_THE_BRANCH

    crossings = revert(candidate)

    assert (crossings > 0) is back
    assert np.array_equal(candidate[-2:], start[-2:]) is back
    assert not any(FALLBACK_WARNING in line for line in log)


def test_a_burst_moves_back_a_via_newly_inside_one_static_item_inside_another(
    monkeypatch: pytest.MonkeyPatch, log: list[str]
) -> None:
    revert, start = first_revert(monkeypatch, inside_one_item_beside_another())
    candidate = start.copy()
    candidate[-2:] = SIDEWAYS

    crossings = revert(candidate)

    assert crossings > 0
    assert np.array_equal(candidate[-2:], start[-2:])
    assert not any(FALLBACK_WARNING in line for line in log)


@pytest.mark.parametrize("stops", [False, True], ids=["copper", "own-net"])
def test_the_guard_refuses_a_via_newly_inside_one_static_item_inside_another(
    stops: bool,
) -> None:
    scenario = inside_one_item_beside_another(stops)
    (placed,) = optimize_module._problems(
        scenario.chains, [], moving_vias(scenario), scenario.params, COARSE
    )
    problem = placed.problem

    moved = [np.array(SIDEWAYS)]

    assert not problem.guard(problem.points, problem.masks, moved, problem.seen)


def stitching_inside(chain: ChainInput) -> ViasScenario:
    """A via that carries no chain at the detour's via, and `chain`, of
    another net, running along x INSIDE_PINNED inside its clearance."""
    return ViasScenario(
        "stitching-inside",
        (chain,),
        (
            ViaInput(
                scenarios.through_via("s", "N", scenarios.DETOUR_VIA),
                (),
                FreeSpace(),
                scenarios.PAIR_CLEARANCE,
            ),
        ),
    )


INSIDE_LEVEL = scenarios.DETOUR_VIA[1] + VIA_TO_CHAIN - INSIDE_PINNED


def reverted_by(
    monkeypatch: pytest.MonkeyPatch, revert: Revert, candidate: FloatArray
) -> BoolArray:
    """What `revert` moves back of `candidate`."""
    masks: list[BoolArray] = []
    real = optimize_module.revert_until_clear

    def spying(
        candidate: FloatArray, start: FloatArray, crossings: Crossings
    ) -> tuple[BoolArray, int]:
        reverted, count = real(candidate, start, crossings)
        masks.append(reverted)
        return reverted, count

    monkeypatch.setattr(optimize_module, "revert_until_clear", spying)
    revert(candidate)
    (reverted,) = masks
    return reverted


def test_a_via_whose_ring_a_pinned_edge_newly_enters_moves_back(
    monkeypatch: pytest.MonkeyPatch, log: list[str]
) -> None:
    # Sliding along the chain leaves the via no deeper inside it, so only
    # the chain's edges newly entering its ring find the move.
    count = 25
    points = np.column_stack(
        [np.linspace(2.0, 8.0, count), np.full(count, INSIDE_LEVEL)]
    )
    pinned = ChainInput(
        ChainCentreline(
            "inner", "P", scenarios.PAIR_WIDTH, scenarios.PAIR_CLEARANCE, points
        ),
        "In1.Cu",
        np.ones(count, dtype=bool),
        FreeSpace(),
    )
    revert, start = first_revert(monkeypatch, stitching_inside(pinned))
    candidate = start.copy()
    candidate[-2:] += (0.5, 0.0)

    reverted = reverted_by(monkeypatch, revert, candidate)

    assert reverted[-2:].all()
    assert np.array_equal(candidate, start)
    assert not any(FALLBACK_WARNING in line for line in log)


ACROSS_X = 5.0
"""Where a free chain of net A runs up across a pinned chain of net B."""
BESIDE_X = 6.5
"""Where a pinned chain of net C runs up, clear of both."""
ALONG_X = 4.0
"""Along B from where A crosses it, where B's edges start clear of A."""


def pinned(chain: ChainInput) -> ChainInput:
    return replace(chain, fixed=np.ones(len(chain.centreline.points), dtype=bool))


def across_and_beside() -> ViasScenario:
    """A free chain of net A crossing a pinned chain of net B, ending short
    of A's other side, and a pinned chain of net C beside them."""
    free = FreeSpace()
    along = [(float(x), 0.0) for x in np.linspace(3.0, 5.5, 11)]
    return ViasScenario(
        "across-and-beside",
        (
            selected("A", "A", free, (ACROSS_X, -2.0), (ACROSS_X, 2.0)),
            pinned(selected("B", "B", free, *along)),
            pinned(selected("C", "C", free, (BESIDE_X, -2.0), (BESIDE_X, 2.0))),
        ),
        (),
    )


def test_an_edge_across_one_member_moved_into_another_moves_back(
    monkeypatch: pytest.MonkeyPatch, log: list[str]
) -> None:
    # A's edges at the crossing start across B, so only judging each member
    # apart finds them entering C; C, pinned, cannot move back.
    revert, start = first_revert(monkeypatch, across_and_beside())
    candidate = start.copy()
    vertices = candidate.reshape(-1, 2)
    nearest = int(np.argmin(np.abs(vertices[:, 1])))
    vertices[nearest, 0] = BESIDE_X

    reverted = reverted_by(monkeypatch, revert, candidate)

    assert reverted.reshape(-1, 2)[nearest].all()
    assert np.array_equal(candidate, start)
    assert not any(FALLBACK_WARNING in line for line in log)


def test_an_edge_slid_along_a_pinned_member_it_crosses_moves_back(
    monkeypatch: pytest.MonkeyPatch, log: list[str]
) -> None:
    # A still crosses B, so only B's edges, which cannot move, find the move:
    # the crossing moves back A's edge, the other side.
    revert, start = first_revert(monkeypatch, across_and_beside())
    candidate = start.copy()
    vertices = candidate.reshape(-1, 2)
    nearest = int(np.argmin(np.abs(vertices[:, 1])))
    vertices[nearest, 0] = ALONG_X

    reverted = reverted_by(monkeypatch, revert, candidate)

    assert reverted.reshape(-1, 2)[nearest].all()
    assert np.array_equal(candidate, start)
    assert not any(FALLBACK_WARNING in line for line in log)


ALONGSIDE = 0.2
"""How far from a pinned chain of net B a free chain of net A runs, mm:
inside their clearance of each other, 0.4 mm between centrelines."""


def alongside(slope: float = 0.0) -> ViasScenario:
    """A free chain of net A ALONGSIDE a pinned chain of net B, both
    rising by `slope` along x."""
    free = FreeSpace()
    return ViasScenario(
        "alongside",
        (
            selected("A", "A", free, (0.0, ALONGSIDE), (10.0, ALONGSIDE + 10 * slope)),
            pinned(selected("B", "B", free, (0.0, 0.0), (10.0, 10 * slope))),
        ),
        (),
    )


def test_an_edge_moved_deeper_into_a_member_it_started_inside_moves_back(
    monkeypatch: pytest.MonkeyPatch, log: list[str]
) -> None:
    # A's edges start inside B's clearance, so only their depth finds the
    # move; B, pinned, cannot move back.
    revert, start = first_revert(monkeypatch, alongside())
    candidate = start.copy()
    vertices = candidate.reshape(-1, 2)
    middle = int(np.argmin(np.abs(vertices[:, 0] - 5.0)))
    vertices[middle, 1] = ALONGSIDE / 2

    reverted = reverted_by(monkeypatch, revert, candidate)

    assert reverted.reshape(-1, 2)[middle].all()
    assert np.array_equal(candidate, start)
    assert not any(FALLBACK_WARNING in line for line in log)


SLID = 0.1
"""How far along the chains the tangential move slides A's vertex, mm."""


@pytest.mark.parametrize("slope", [0.0, 1.7], ids=["along-x", "sloping"])
def test_an_edge_slid_along_inside_a_member_s_clearance_stays(
    monkeypatch: pytest.MonkeyPatch, log: list[str], slope: float
) -> None:
    # A's vertex slides along B at the same depth: only round-off in its
    # edges' depth changes, which sloping chains show.
    revert, start = first_revert(monkeypatch, alongside(slope))
    candidate = start.copy()
    vertices = candidate.reshape(-1, 2)
    middle = int(np.argmin(np.abs(vertices[:, 0] - 5.0)))
    vertices[middle] += SLID * np.array([1.0, slope]) / math.hypot(1.0, slope)
    moved = candidate.copy()

    reverted = reverted_by(monkeypatch, revert, candidate)

    assert not reverted.any()
    assert np.array_equal(candidate, moved)
    assert not any(FALLBACK_WARNING in line for line in log)


DIVERGING = 0.05
"""How much a free chain of net A rises along x away from a pinned chain
of net B that runs along x, starting inside their clearance."""


@pytest.mark.parametrize("slide", [-SLID, SLID], ids=["back", "forward"])
def test_an_edge_slid_along_an_oblique_chain_inside_a_member_s_clearance_stays(
    monkeypatch: pytest.MonkeyPatch, log: list[str], slide: float
) -> None:
    # A's vertex slides along A, which runs at an angle to B: the edge it
    # lengthens takes over part of its neighbour at that part's depth, and
    # A's deepest inside B stays where it was.
    free = FreeSpace()
    scenario = ViasScenario(
        "diverging",
        (
            selected(
                "A",
                "A",
                free,
                (0.0, ALONGSIDE / 2),
                (10.0, ALONGSIDE / 2 + 10 * DIVERGING),
            ),
            pinned(selected("B", "B", free, (0.0, 0.0), (10.0, 0.0))),
        ),
        (),
    )
    revert, start = first_revert(monkeypatch, scenario)
    candidate = start.copy()
    vertices = candidate.reshape(-1, 2)
    middle = int(np.argmin(np.abs(vertices[:, 0] - 3.0)))
    vertices[middle] += slide * np.array([1.0, DIVERGING]) / math.hypot(1.0, DIVERGING)
    moved = candidate.copy()

    reverted = reverted_by(monkeypatch, revert, candidate)

    assert not reverted.any()
    assert np.array_equal(candidate, moved)
    assert not any(FALLBACK_WARNING in line for line in log)


def test_a_member_edge_beside_the_nearest_moved_deeper_into_a_via_s_ring_stays(
    monkeypatch: pytest.MonkeyPatch, log: list[str]
) -> None:
    # The chain's vertex beside the via, slid toward it along the chain,
    # takes the next edge deeper into the via's ring, but the edge under
    # the via stays the deepest at the same depth: the chain's copper in
    # the ring is where it was.
    chain = selected(
        "P",
        "P",
        FreeSpace(),
        (0.0, INSIDE_LEVEL),
        (10.0, INSIDE_LEVEL),
        layer="In1.Cu",
    )
    revert, start = first_revert(monkeypatch, stitching_inside(chain))
    candidate = start.copy()
    vertices = candidate[:-2].reshape(-1, 2)
    x = scenarios.DETOUR_VIA[0]
    beside = int(np.flatnonzero(vertices[:, 0] > x)[0])
    assert vertices[beside - 1, 0] <= x
    vertices[beside, 0] = (vertices[beside, 0] + x) / 2
    moved = candidate.copy()

    reverted = reverted_by(monkeypatch, revert, candidate)

    assert not reverted.any()
    assert np.array_equal(candidate, moved)
    assert not any(FALLBACK_WARNING in line for line in log)


@pytest.mark.parametrize(
    "away", [0.0, INSIDE_PINNED / 4], ids=["at-the-same-depth", "shallower"]
)
def test_a_via_slid_along_inside_a_pinned_chain_s_clearance_stays(
    monkeypatch: pytest.MonkeyPatch, log: list[str], away: float
) -> None:
    # The via slides along the chain no deeper inside it, so that the chain's
    # edges beside the one under it go deeper into its ring while the
    # chain's deepest stays as deep.
    count = 25
    points = np.column_stack(
        [np.linspace(2.0, 8.0, count), np.full(count, INSIDE_LEVEL)]
    )
    chain = ChainInput(
        ChainCentreline(
            "inner", "P", scenarios.PAIR_WIDTH, scenarios.PAIR_CLEARANCE, points
        ),
        "In1.Cu",
        np.ones(count, dtype=bool),
        FreeSpace(),
    )
    revert, start = first_revert(monkeypatch, stitching_inside(chain))
    candidate = start.copy()
    candidate[-2:] += (SLID, -away)
    moved = candidate.copy()

    reverted = reverted_by(monkeypatch, revert, candidate)

    assert not reverted.any()
    assert np.array_equal(candidate, moved)
    assert not any(FALLBACK_WARNING in line for line in log)


def test_a_member_edge_moved_deeper_into_a_via_s_ring_moves_back(
    monkeypatch: pytest.MonkeyPatch, log: list[str]
) -> None:
    # The chain's edges near the via start inside its ring, so only the
    # via's distance to the chain finds the move.
    chain = selected(
        "P",
        "P",
        FreeSpace(),
        (0.0, INSIDE_LEVEL),
        (10.0, INSIDE_LEVEL),
        layer="In1.Cu",
    )
    revert, start = first_revert(monkeypatch, stitching_inside(chain))
    candidate = start.copy()
    vertices = candidate[:-2].reshape(-1, 2)
    near = np.abs(vertices[:, 0] - scenarios.DETOUR_VIA[0]) < COARSE.spacing
    assert near.any()
    vertices[near, 1] -= INSIDE_PINNED / 2

    reverted = reverted_by(monkeypatch, revert, candidate)

    assert reverted[:-2].reshape(-1, 2)[near].all()
    assert np.array_equal(candidate, start)
    assert not any(FALLBACK_WARNING in line for line in log)


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


BURIED_IN = Disc(5.0, 3.5, 1.0)
"""Static copper whose clearance the detour's via starts 0.5 mm deep in."""


def test_a_via_starting_deep_in_its_clearance_moves_out_with_its_ends() -> None:
    scenario = scenarios.via_detour()
    (given,) = scenario.vias

    result = run(replace(scenario, vias=(replace(given, field=BURIED_IN),)))

    (via,) = result.vias
    top, bottom = (c.polyline for c in result.chains)
    assert math.dist(via.position, (BURIED_IN.cx, BURIED_IN.cy)) >= (
        BURIED_IN.radius - INSIDE_TOLERANCE
    )
    assert tuple(top[-1]) == tuple(bottom[0]) == via.position


def test_a_junction_at_a_via_starting_inside_its_clearance_moves_with_it() -> None:
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
        BURIED_IN,
        scenarios.PAIR_CLEARANCE,
    )

    result = optimize_chains(chains, scenarios.PARAMS, COARSE, [via])

    (moved,) = result.vias
    assert moved.position != at
    a, b, c, d = (chain.polyline for chain in result.chains)
    assert tuple(a[-1]) == tuple(b[0]) == tuple(c[0]) == tuple(d[0])
    assert tuple(d[0]) == moved.position


def static_vias() -> tuple[StaticVia, ...]:
    """A GND via and a buried via of another net, standing still."""
    buried_via = replace(
        scenarios.through_via("b", "P", (2.0, 2.0)),
        span=("In1.Cu", "In2.Cu"),
    )
    return (
        StaticVia(
            scenarios.through_via("g", "GND", (7.0, 3.0)),
            ground=True,
            clearance=scenarios.PAIR_CLEARANCE,
        ),
        StaticVia(buried_via, ground=False, clearance=scenarios.PAIR_CLEARANCE),
    )


def test_a_problem_keeps_the_static_vias_its_moving_vias_meet() -> None:
    given = scenarios.via_detour()

    placed = optimize_module._problems(
        given.chains,
        [],
        moving_vias(given),
        given.params,
        COARSE,
        static_vias(),
    )

    vias = [v for p in placed for v in p.problem.vias]
    assert {v.given.via.id for v in vias} == {"v"}
    assert {s.via.id for p in placed for s in p.problem.static_vias} == {"g", "b"}


def test_static_vias_it_does_not_pair_with_leave_the_result_as_it_is() -> None:
    given = scenarios.via_detour()
    regular = [s for s in static_vias() if not s.ground]

    alone = run(given)
    among = optimize_chains(
        given.chains, given.params, COARSE, given.vias, static_vias=regular
    )

    for chain, expected in zip(among.chains, alone.chains, strict=True):
        np.testing.assert_array_equal(chain.polyline, expected.polyline)
    assert among.vias == alone.vias


def test_layers_without_a_moving_via_stay_independent() -> None:
    # A buried stitching via ties In1.Cu and In2.Cu, which no chain is on.
    buried_via = replace(
        scenarios.through_via("s", "G", (5.0, 6.0)),
        span=("In1.Cu", "In2.Cu"),
    )
    stitching = ViaInput(buried_via, (), FreeSpace(), scenarios.PAIR_CLEARANCE)
    scenario = replace(scenarios.via_detour(), vias=(stitching,))
    params = scenario.params

    result = run(scenario)

    placed = optimize_module._problems(
        scenario.chains, [], moving_vias(scenario), params, COARSE
    )
    assert sorted(len(p.problem.vias) for p in placed) == [0, 0, 1]

    for chain, given in zip(result.chains, scenario.chains, strict=True):
        alone = optimize_chains([given], params, COARSE)
        np.testing.assert_array_equal(chain.polyline, alone.chains[0].polyline)


def test_two_layers_joined_by_a_via_are_one_problem(
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
        self: optimize_module._Problem,
        starts: list[FloatArray],
        stop: Stop | None = None,
    ) -> optimize_module._ProblemBurst:
        count = len(starts)
        return optimize_module._ProblemBurst(
            list(starts),
            list(self.positions),
            [0] * count,
            [moved_back] * len(self.vias),
            False,
        )

    monkeypatch.setattr(optimize_module._Problem, "burst", still)
    # Too few bursts for the best energy to stall.
    settings = replace(COARSE, max_iterations=COARSE.stall_window - 1)

    result = run(scenarios.via_detour(), settings)

    assert (result.termination is not Termination.ITERATION_CAP) == settles


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


RETURN_PARAMS = replace(
    scenarios.PARAMS, repulsion=0.5, attraction=0.25, attraction_range=4.0
)
"""Attraction strong enough to move a via from 3 mm away in a few bursts."""


def grounded(via_id: str, at: tuple[float, float], net: str = "GND") -> ViaInput:
    """A stitching via of a ground net in free space."""
    return ViaInput(
        scenarios.through_via(via_id, net, at),
        (),
        FreeSpace(),
        scenarios.PAIR_CLEARANCE,
        ground=True,
    )


def static_ground(via_id: str, at: tuple[float, float], net: str = "GND") -> StaticVia:
    return StaticVia(
        scenarios.through_via(via_id, net, at), True, scenarios.PAIR_CLEARANCE
    )


def returned(*ground: ViaInput) -> ViasScenario:
    """The detour route through its via of net N, with the `ground` vias
    moving, and attraction."""
    scenario = scenarios.via_detour()
    return replace(scenario, vias=(*scenario.vias, *ground), params=RETURN_PARAMS)


class TestGradient:
    """The energy's gradient by the via's coordinates takes in its carried
    ends, the chains it sees and that see it, and the other vias."""

    def check(
        self,
        scenario: ViasScenario,
        seed: int,
        pairs: tuple[PairInput, ...] = (),
        static: Sequence[StaticVia] = (),
    ) -> None:
        params = replace(scenario.params, repulsion=0.5, falloff_range=1.5)
        blocks = [pair_block(pair, COARSE.spacing) for pair in pairs]
        everything = [*scenario.chains, *(block.chain for block in blocks)]
        vias = MovingVias(everything, scenario.vias, HOLE_CLEARANCE, HOLE_TO_HOLE)
        (placed,) = optimize_module._problems(
            scenario.chains, blocks, vias, params, COARSE, static
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

    def test_with_a_via_at_every_vertex_of_a_grid(self) -> None:
        """Issue 100's grid, at its start, where a chain's vertex beside a
        via can lie as near to the vias on either side of it: the slope
        along random directions is the gradient's, which checks every
        element at once."""
        scenario = scenarios.via_grid(3, 3, via_every=1)
        params = replace(scenario.params, repulsion=0.5, falloff_range=1.5)
        vias = MovingVias(scenario.chains, scenario.vias, HOLE_CLEARANCE, HOLE_TO_HOLE)
        (placed,) = optimize_module._problems(scenario.chains, [], vias, params, COARSE)
        layout = placed.problem.layout(placed.problem.points)
        objective = placed.problem.objective(layout)
        x = layout.x0
        _, gradient = objective(x)
        step = 1e-7
        for direction in np.random.default_rng(0).normal(size=(3, len(x))):
            numeric = (
                objective(x + step * direction)[0] - objective(x - step * direction)[0]
            ) / (2 * step)
            assert gradient @ direction == pytest.approx(numeric, rel=1e-4, abs=1e-6)

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
        # A G via starting inside the track's clearance; a G via at the edge
        # of its zone, near that one and an unselected G via; and an H via
        # within range of it and of the track.
        start = 0.9
        scenario = stitched(start, top=start + STITCH_MARGIN + 0.005)
        (moving,) = scenario.vias
        beside = (STITCH_X + HOLES_APART + 0.005, start)
        still = scenarios.through_via("u", "G", beside)
        layers = [replace(layer(name), vias=(still,)) for name in moving.via.span]
        low = (STITCH_X, scenarios.VIA_TO_CHAIN / 2)
        inside = replace(moving, via=scenarios.through_via("f", "G", low))
        plane = {name: box(0, -5, 10, 10) for name in moving.via.span}
        other = ViaInput(
            scenarios.through_via("h", "H", (STITCH_X + 1.0, 1.3)),
            (),
            FreeSpace(),
            scenarios.PAIR_CLEARANCE,
            (ViaZone("h-plane", plane, 0.2, 0.0),),
        )
        vias = (
            inside,
            replace(moving, own_net=own_net_stops(layers, moving.via)),
            other,
        )
        self.check(replace(scenario, vias=vias), seed)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_with_a_via_of_its_net_that_stands_still(self, seed: int) -> None:
        start = 0.9
        self.check(beside_still(start, start + HOLES_APART + 0.005), seed)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_with_a_ground_via_moving_beside_it(self, seed: int) -> None:
        self.check(returned(grounded("g", (5.0, 4.2))), seed)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_with_a_ground_via_standing_beside_it(self, seed: int) -> None:
        self.check(returned(), seed, static=(static_ground("s", (5.9, 3.7)),))

    @pytest.mark.parametrize("seed", [0, 1])
    def test_between_two_ground_vias(self, seed: int) -> None:
        # One standing still and one moving, either side of the via.
        moving = grounded("g", (6.1, 3.2))
        self.check(returned(moving), seed, static=(static_ground("s", (3.9, 2.9)),))

    @pytest.mark.parametrize("seed", [0, 1])
    def test_with_a_ground_via_at_the_attraction_s_range(self, seed: int) -> None:
        cutoff = RETURN_PARAMS.attraction_range
        moving = grounded("g", (5.0, scenarios.DETOUR_VIA[1] + cutoff - 0.01))
        self.check(returned(moving), seed)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_with_ground_vias_of_one_net(self, seed: int) -> None:
        # One moving within range, one at hole-to-hole and one standing.
        near = grounded("n", (5.0 + HOLES_APART + 0.005, 4.2))
        self.check(
            returned(grounded("g", (5.0, 4.2)), near, grounded("m", (6.2, 4.6))),
            seed,
            static=(static_ground("s", (4.0, 4.4)),),
        )

    @pytest.mark.parametrize("seed", [0, 1])
    def test_with_vias_of_two_ground_nets(self, seed: int) -> None:
        agnd = replace(
            grounded("a", (6.0, 2.4)),
            via=scenarios.through_via("a", "AGND", (6.0, 2.4)),
        )
        self.check(
            returned(grounded("g", (5.0, 4.2)), agnd),
            seed,
            static=(static_ground("s", (4.2, 2.3), net="AGND"),),
        )


def moving_vias(scenario: ViasScenario) -> MovingVias:
    return MovingVias(scenario.chains, scenario.vias, HOLE_CLEARANCE, HOLE_TO_HOLE)


def test_chains_straightened_in_one_burst_are_kept() -> None:
    # Two free vertices a chain: one burst straightens the route, and the
    # via's move in it is less than the tolerance.
    few = OptimizerSettings(spacing=2.5, settled_change=0.025, burst_maxiter=30)

    result = run(scenarios.via_slack(FreeSpace(), (5.0, 0.01)), few)

    assert result.termination is Termination.CONVERGED
    assert route_length(result) == pytest.approx(10.0, abs=0.01)


STITCH_X = 5.0
STITCH_PARAMS = replace(scenarios.PARAMS, repulsion=0.5)
STITCH_ZONE_LAYERS = ("F.Cu", "B.Cu")


def stitched(*heights: float, top: float, nets: Sequence[str] = ()) -> ViasScenario:
    """A straight chain of net S on F.Cu along y = 0, pinned at every vertex
    so that it cannot bow away, and a stitching via at each of `heights`
    above it, of net G or the net `nets` gives in turn, inside a zone of its
    net on F.Cu and B.Cu whose region ends at y = `top`."""
    regions = {layer: box(0, -5, 10, top) for layer in STITCH_ZONE_LAYERS}
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


OUTSIDE_ZONE = 0.5
"""How far above its zone's region the stitching vias start, mm: deeper
inside their containment than they come inside each other's hole-to-hole."""
HOLES_SPARE = 0.05
"""How far beyond hole-to-hole of each other the stitching vias start, mm."""
TOWARD = 0.15
"""How far a move takes one via toward the other, mm; more than HOLES_SPARE
and less than OUTSIDE_ZONE."""


def outside_their_zone() -> ViasScenario:
    """Two stitching vias of one net side by side OUTSIDE_ZONE above their
    zone's region, HOLES_SPARE beyond hole-to-hole of each other."""
    level = 1.5
    scenario = stitched(level, level, top=level - OUTSIDE_ZONE)
    left, right = scenario.vias
    moved = replace(right.via, position=(STITCH_X + HOLES_APART + HOLES_SPARE, level))
    return replace(scenario, vias=(left, replace(right, via=moved)))


def test_a_burst_moves_back_a_via_deep_outside_its_zone_entering_another_s_holes(
    monkeypatch: pytest.MonkeyPatch, log: list[str]
) -> None:
    # The move keeps the left via as deep outside its zone, so only judging
    # each stop apart finds it; both vias move back, not the whole burst.
    revert, start = first_revert(monkeypatch, outside_their_zone())
    candidate = start.copy()
    candidate[-4] += TOWARD

    reverted = reverted_by(monkeypatch, revert, candidate)

    assert reverted[-4:].all()
    assert np.array_equal(candidate, start)
    assert not any(FALLBACK_WARNING in line for line in log)


def test_the_guard_refuses_a_via_deep_outside_its_zone_entering_another_s_holes() -> (
    None
):
    scenario = outside_their_zone()
    (placed,) = optimize_module._problems(
        scenario.chains, [], moving_vias(scenario), scenario.params, COARSE
    )
    problem = placed.problem
    left, right = problem.positions

    moved = [left + np.array((TOWARD, 0.0)), right]

    assert not problem.guard(problem.points, problem.masks, moved, problem.seen)


ALONE_Y = 3.0
"""Where a lone stitching via stands, mm: beyond every term's reach of the
pinned chain at y = 0."""
OPEN_TOP = ALONE_Y + 5.0
"""Where its zone's region ends, mm: beyond every term's reach of it."""
COPPER_RADIUS = 1.0
"""The radius of the static copper beside it, inflated for it, mm."""
WITHIN_STANDOFF = STITCH_PARAMS.standoff / 2
WITHIN_FALLOFF = STITCH_PARAMS.falloff_range / 2
"""Beyond the standoff: only the repulsion is curved there."""
PRESSED_VIA_SCALE = 0.5


def alone(*, gap: float | None = None, top: float = OPEN_TOP) -> ViasScenario:
    """A stitching via at ALONE_Y in `stitched`'s zone ending at `top`, with
    static copper `gap` mm right of its clearance unless None."""
    scenario = stitched(ALONE_Y, top=top)
    if gap is None:
        return scenario
    (given,) = scenario.vias
    copper = Disc(STITCH_X + COPPER_RADIUS + gap, ALONE_Y, COPPER_RADIUS)
    return replace(scenario, vias=(replace(given, field=copper),))


def alone_problem(scenario: ViasScenario) -> optimize_module._Problem:
    (placed,) = optimize_module._problems(
        scenario.chains, [], moving_vias(scenario), scenario.params, DEFAULT_SETTINGS
    )
    return placed.problem


def via_scale(scenario: ViasScenario) -> float:
    """The scale of the lone via's variable in the first burst."""
    problem = alone_problem(scenario)
    layout = problem.layout(problem.points)
    seen = problem.seeing(layout.starts)
    stiffness = problem.stiffness(layout, seen, problem.lookups(seen))
    scale = optimize_module._scale(stiffness, DEFAULT_SETTINGS.spacing)
    (variable,) = layout.via_index
    return float(scale[2 * variable])


class TestViaStiffness:
    def test_a_free_via_scales_by_one(self) -> None:
        assert via_scale(alone()) == 1.0

    def test_a_via_pressed_at_clearance_against_static_copper_scales_down(
        self,
    ) -> None:
        assert via_scale(alone(gap=0.0)) < PRESSED_VIA_SCALE

    @pytest.mark.parametrize("gap", [WITHIN_STANDOFF, WITHIN_FALLOFF])
    @pytest.mark.parametrize("pushing", ["copper", "zone"])
    def test_is_the_curvature_along_the_push(self, pushing: str, gap: float) -> None:
        # One obstacle with a flat face toward the via, so that the push has
        # one direction and the diagonal curvature along it is exactly the
        # integrand's f'' (static copper, the repulsion's term) or the
        # hinge's alone (the zone's edge, a stop).
        if pushing == "copper":
            scenario, axis = alone(gap=gap), 0
        else:
            scenario, axis = alone(top=ALONE_Y + STITCH_MARGIN + gap), 1
        problem = alone_problem(scenario)
        polylines, (at,) = problem.points, problem.positions

        (stiffness,) = problem.via_stiffness(polylines, [at])

        step = np.zeros(2)
        step[axis] = CURVATURE_STEP
        up = problem.via_energy(polylines, [at + step])[2][0, axis]
        down = problem.via_energy(polylines, [at - step])[2][0, axis]
        numeric = (up - down) / (2 * CURVATURE_STEP)
        assert stiffness == pytest.approx(
            numeric, rel=CURVATURE_RTOL, abs=CURVATURE_ATOL
        )


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
        assert via.position == pytest.approx((STITCH_X, start), abs=1e-6)

    def test_two_of_one_net_within_range_of_each_other_stay(self) -> None:
        low = scenarios.VIA_TO_CHAIN + STITCH_PARAMS.falloff_range + 0.05
        high = low + HOLES_APART + STITCH_PARAMS.falloff_range / 2

        result = run(stitched(low, high, top=high + 3.0))

        assert result.vias[0].position == pytest.approx((STITCH_X, low), abs=1e-6)
        assert result.vias[1].position == pytest.approx((STITCH_X, high), abs=1e-6)

    def test_two_of_other_nets_within_range_of_each_other_push_apart(
        self,
    ) -> None:
        low = scenarios.VIA_TO_CHAIN + STITCH_PARAMS.falloff_range + 0.05
        high = low + HOLES_APART + STITCH_PARAMS.falloff_range / 2

        result = run(stitched(low, high, top=high + 3.0, nets=("G", "H")))

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
        # The upper via's two stops, its zone's region, one on both layers,
        # and the lower via, are about equally near it: the energy sees them
        # up to τ·ln 2 nearer (`energy.smooth_minimum`).
        seen_nearer = SMOOTH_WIDTH * math.log(2)
        assert apart <= HOLES_APART + STITCH_PARAMS.standoff + seen_nearer
        assert low[1] > 0.9 + 0.1

    def test_one_beside_a_track_of_its_net_stays(self) -> None:
        # Of another net, the track would push it away.
        start = scenarios.VIA_TO_CHAIN + STITCH_PARAMS.falloff_range / 2

        result = run(stitched(start, top=start + 2.0, nets=("S",)))

        (via,) = result.vias
        assert via.position == pytest.approx((STITCH_X, start), abs=1e-6)

    def test_one_with_nothing_in_range_stays(self) -> None:
        start = scenarios.VIA_TO_CHAIN + STITCH_PARAMS.falloff_range + 0.05

        result = run(stitched(start, top=start + 2.0))

        (via,) = result.vias
        assert via.position == pytest.approx((STITCH_X, start), abs=1e-6)

    def test_one_starting_inside_its_clearance_backs_out_of_it(self) -> None:
        start = scenarios.VIA_TO_CHAIN / 2

        result = run(stitched(start, top=1.3))

        (via,) = result.vias
        assert via.position[1] >= scenarios.VIA_TO_CHAIN - INSIDE_TOLERANCE


def reference_via_energy(
    problem: optimize_module._Problem,
    polylines: Sequence[FloatArray],
    positions: Sequence[FloatArray],
) -> tuple[float, list[FloatArray], FloatArray]:
    """The moving vias' own terms built via by via: the full integrand for
    the smooth minimum over what stands still and repels the via and what
    `via_repel` finds, and the hinge alone for that over its stops and
    what `via_stop` finds, each answer pulled by its share."""
    gradients = [np.zeros_like(points) for points in polylines]
    via_gradients = np.zeros((len(problem.vias), 2))
    value = 0.0
    sights = (
        problem.via_repel.smooth_answers(polylines, positions, problem.reach),
        problem.via_stop.smooth_answers(polylines, positions, problem.reach),
    )
    for k, via in enumerate(problem.vias):
        given = via.given
        at = positions[k]
        for static, near, integrand in (
            (given.field, sights[0], problem.integrand),
            (given.stops, sights[1], problem.limit),
        ):
            d, g = np.array([math.inf]), np.zeros((1, 2))
            if static is not None:
                d, g = static.smooth_distance(at[None], problem.reach)
            distance, (static_share, seen_share) = combine_fields(
                d, near.distance[k : k + 1]
            )
            direction = static_share[0] * g[0] + seen_share[0] * near.gradient[k]
            term, slope = integrand(distance)
            value += given.via.diameter * float(term[0])
            weight = given.via.diameter * slope[0]
            via_gradients[k] += weight * direction
            for n in np.flatnonzero(near.point == k).tolist():
                push = weight * seen_share[0] * near.share[n] * near.direction[n]
                j, first = int(near.member[n]), int(near.vertex[n])
                if j < 0:
                    via_gradients[first] -= push
                    continue
                start, along = near.starts[n], near.ends[n] - near.starts[n]
                squared = float(along @ along)
                t = (
                    float(np.clip((at - start) @ along / squared, 0, 1))
                    if squared
                    else 0
                )
                gradients[j][first] -= (1 - t) * push
                gradients[j][first + 1] -= t * push
    return value, gradients, via_gradients


INSIDE_X = STITCH_X + 2.5
INSIDE_X_ABOVE = 1.2
"""The height of the G via above the one starting inside the chain's
clearance, nearer to it than to the chain below."""


def stitched_grid() -> ViasScenario:
    """Stitching vias of nets G and H in rows between `stitched`'s chain and
    a pinned chain of net T above its zone, close enough to stop and repel
    each other and to meet their zone's edge, the bottom row seeing the
    chain below and the top row the chain above; an unselected G via
    standing still among them; and, first of all, an H via starting inside
    the lower chain's clearance, nearer than that chain to a G via above it."""
    heights = [0.9 + 0.55 * row for row in range(4)]
    top = heights[-1] + STITCH_MARGIN + 0.005
    scenario = stitched(*heights, INSIDE_X_ABOVE, top=top)
    still = scenarios.through_via("u", "G", (STITCH_X + HOLES_APART + 0.005, 0.9))
    inside = scenarios.through_via("f", "H", (INSIDE_X, 0.2))
    vias = [ViaInput(inside, (), FreeSpace(), scenarios.PAIR_CLEARANCE)]
    for k, given in enumerate(scenario.vias):
        x = STITCH_X + (0.6 if k % 2 else 0.0)
        if given.via.position[1] == INSIDE_X_ABOVE:
            x = INSIDE_X
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
            scenario.chains, [], moving_vias(scenario), params, COARSE
        )
        problems = [p.problem for p in placed if p.problem.vias]
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


def reference_pair_energy(
    problem: optimize_module._Problem,
    positions: Sequence[FloatArray],
    static: Sequence[StaticVia],
) -> tuple[float, FloatArray]:
    """The pairwise via terms summed pair by pair, from each moving via's
    side, weighted by its diameter: a ground via and a regular one attract
    and keep clear of each other, two ground vias of different nets only
    keep clear, with the other vias of the problem at their `apart` and
    with the `static` vias at the distance `via_via_inflation` gives with
    the moving via's zones on their layers; two ground vias of one net
    repel each other beyond their rings and clearance."""
    params = problem.params
    obstacle = obstacle_integrand(params)
    repulsion = repulsion_integrand(params)
    attraction = attraction_integrand(params)
    gradient = np.zeros((len(problem.vias), 2))
    value = 0.0
    for k in range(len(problem.vias)):
        given = problem.vias[k].given
        others: list[tuple[int | None, FloatArray, StaticVia, float]] = [
            (u, positions[u], static_of(v.given), problem.vias[k].apart[u])
            for u, v in enumerate(problem.vias)
            if u != k and u in problem.vias[k].apart
        ]
        for s in static:
            clearances = {given.via.net: given.clearance, s.via.net: s.clearance}
            zones = [z for z in given.zones if any(map(s.via.reaches, z.layers))]
            apart = via_via_inflation(
                given.via, s.via, clearances.__getitem__, zones=zones
            )
            assert apart is not None
            at = np.array(s.via.position, dtype=float)
            others.append((None, at, s, apart))
        for u, at, other, apart in others:
            delta = positions[k] - at
            r = float(np.linalg.norm(delta))
            if other.via.net == given.via.net:
                if not given.ground:
                    continue
                rings = (given.via.diameter + other.via.diameter) / 2
                kept = rings + max(given.clearance, other.clearance)
                term, slope = repulsion(np.array([r - kept]))
            elif other.ground or given.ground:
                term, slope = obstacle(np.array([r - apart]))
                if other.ground != given.ground:
                    pull, pull_slope = attraction(np.array([r]), np.array([apart]))
                    term, slope = term + pull, slope + pull_slope
            else:
                continue
            weight = given.via.diameter
            value += weight * float(term[0])
            push = weight * float(slope[0]) * delta / r
            gradient[k] += push
            if u is not None:
                gradient[u] -= push
    return value, gradient


def two_grounds() -> ViasScenario:
    """`returned` with a GND via and an AGND via moving beside the via."""
    agnd = replace(
        grounded("a", (6.0, 2.4)), via=scenarios.through_via("a", "AGND", (6.0, 2.4))
    )
    return returned(grounded("g", (5.0, 4.2)), agnd)


def close_grounds() -> ViasScenario:
    """`returned` with a GND via and an AGND via moving above the via,
    inside the clearance hinge of their pair."""
    gnd = (5.0, 4.2)
    agnd = (5.0, gnd[1] + scenarios.VIA_TO_VIA + RETURN_PARAMS.standoff / 2)
    return returned(
        grounded("g", gnd),
        replace(grounded("a", agnd), via=scenarios.through_via("a", "AGND", agnd)),
    )


SHORT_PULL = 0.9
"""An attraction range shorter than the distance the vias keep plus the
obstacle reach, so that standing vias beyond it still repel."""


def short_pull() -> ViasScenario:
    """`returned` with a GND via moving and attraction of `SHORT_PULL`
    range: the standing P via and GND via near the moving ones lie beyond
    it, inside their repulsion."""
    scenario = returned(grounded("g", (5.0, 4.2)))
    return replace(
        scenario, params=replace(scenario.params, attraction_range=SHORT_PULL)
    )


EDGE_OF_REACH = 0.95
"""Where a pair at the edge of its reach lies, as a fraction of it: near
enough to give energy, past a query radius only a little too short."""


def far_return() -> ViasScenario:
    """`returned` with a GND via moving at EDGE_OF_REACH of the attraction
    range from the via."""
    x, y = scenarios.DETOUR_VIA
    return returned(
        grounded("g", (x, y + EDGE_OF_REACH * RETURN_PARAMS.attraction_range))
    )


def far_spread() -> ViasScenario:
    """`short_pull` with a second GND via above its moving one, at
    EDGE_OF_REACH of the obstacle reach beyond their rings and clearance."""
    scenario = short_pull()
    (_, g) = scenario.vias
    rings = g.via.diameter + g.clearance
    gap = rings + EDGE_OF_REACH * obstacle_reach(scenario.params)
    x, y = g.via.position
    return replace(scenario, vias=(*scenario.vias, grounded("h", (x, y + gap))))


def grounded_grid() -> ViasScenario:
    """`stitched_grid` with its G vias on a ground net and attraction."""
    scenario = stitched_grid()
    vias = tuple(replace(v, ground=v.via.net == "G") for v in scenario.vias)
    return replace(scenario, vias=vias, params=RETURN_PARAMS)


def inside_ground() -> ViasScenario:
    """`returned` with a GND via that starts deep inside the via's
    clearance."""
    return returned(grounded("f", (5.3, 3.0)), grounded("g", (5.0, 4.2)))


GROUND_STATIC = (
    static_ground("s", (5.9, 3.7)),
    static_ground("t", (3.0, 3.1), net="AGND"),
    StaticVia(scenarios.through_via("r", "P", (4.0, 4.0)), False, 0.3),
)
PAIR_ENERGY_SCENARIOS = {
    "moving": lambda: returned(grounded("g", (5.0, 4.2))),
    "standing": returned,
    "two-grounds": two_grounds,
    "close-grounds": close_grounds,
    "short-pull": short_pull,
    "far-return": far_return,
    "far-spread": far_spread,
    "zoned": lambda: zoned(returned(grounded("g", (5.0, 4.2))), 0, "In1.Cu"),
    "starts-inside": inside_ground,
    "grid": grounded_grid,
}


class TestPairEnergy:
    """The pairwise via terms equal those summed pair by pair, at the start
    and with every via jittered."""

    @pytest.mark.parametrize("jitter", [0.0, 0.05])
    @pytest.mark.parametrize("name", PAIR_ENERGY_SCENARIOS)
    def test_matches_the_pair_by_pair_reference(self, name: str, jitter: float) -> None:
        scenario = PAIR_ENERGY_SCENARIOS[name]()
        placed = optimize_module._problems(
            scenario.chains,
            [],
            moving_vias(scenario),
            scenario.params,
            COARSE,
            GROUND_STATIC,
        )
        problems = [p.problem for p in placed if p.problem.vias]
        assert problems
        rng = np.random.default_rng(5)
        for problem in problems:
            positions = [
                p + rng.normal(scale=jitter, size=2) for p in problem.positions
            ]

            assert problem.pairs is not None
            value, gradient = problem.pairs.energy(positions)

            want, want_gradient = reference_pair_energy(
                problem, positions, problem.static_vias
            )
            assert want != 0.0
            assert value == pytest.approx(want, rel=1e-12, abs=1e-12)
            np.testing.assert_allclose(gradient, want_gradient, rtol=1e-10, atol=1e-12)


NEUTRAL = neutral_distance(RETURN_PARAMS, scenarios.VIA_TO_VIA)
# The pairs here settle within 1e-7 mm of NEUTRAL; the rest is platform slack.
SETTLED = 0.02
"""How near its neutral distance a pair settles under COARSE, mm."""
FAR_CHAIN = -10.0
"""The height of a pinned chain of another net, out of every via's reach,
which gives the vias' layers a problem."""


def far_chain() -> ChainInput:
    vertices = [(float(x), FAR_CHAIN) for x in np.linspace(-5.0, 5.0, 5)]
    chain = selected("S", "S", FreeSpace(), *vertices)
    return replace(chain, fixed=np.ones(len(vertices), dtype=bool))


ABOVE = box(-5.0, 0.4, 5.0, 8.0)
"""The region of the GND zone of `stitching`'s vias: above y = 0, so that
one never moves past the SIG via at the origin."""
BELOW = box(-5.0, -8.0, 5.0, 2.6)
"""The region of the SIG zone of `stitching`'s vias."""


def stitching(via_id: str, net: str, at: tuple[float, float]) -> ViaInput:
    """A stitching via in free space, inside a zone of its net, on a ground
    net if `net` is GND."""
    region = ABOVE if net == "GND" else BELOW
    zone = ViaZone(net, {layer: region for layer in ("F.Cu", "B.Cu")}, 0.2, 0.0)
    return replace(grounded(via_id, at, net), ground=net == "GND", zones=(zone,))


def static_of(given: ViaInput) -> StaticVia:
    return StaticVia(given.via, given.ground, given.clearance)


def split_from(given: ViaInput, static: Sequence[StaticVia]) -> ViaInput:
    """`given` with its field and `static_vias` built as `run` builds them
    over layers that hold the `static` vias."""
    layers = [
        replace(layer(name), vias=tuple(s.via for s in static))
        for name in given.via.span
    ]
    by_id = {s.via.id: s for s in static}
    field, static_vias = split_via_obstacles(
        layers,
        given.via,
        lambda other: pairs_with(given.via, given.ground, by_id[other.id]),
        zones=given.zones,
    )
    return replace(given, field=field, static_vias=static_vias)


def pair_run(vias: Sequence[ViaInput], static: Sequence[StaticVia] = ()) -> RunResult:
    return optimize_chains(
        (far_chain(),), RETURN_PARAMS, COARSE, vias, static_vias=static
    )


class TestReturnVia:
    """A ground via and a regular one draw together to their neutral
    distance, as the pair's repulsion and attraction balance it."""

    def test_the_neutral_distance_lies_between_the_limit_and_the_range(
        self,
    ) -> None:
        assert scenarios.VIA_TO_VIA + RETURN_PARAMS.falloff_range / 2 < NEUTRAL
        assert RETURN_PARAMS.attraction_range > NEUTRAL

    @pytest.mark.parametrize("still", [None, "SIG", "GND"])
    def test_a_pair_3_mm_apart_settles_at_it(self, still: str | None) -> None:
        signal = stitching("s", "SIG", (0.0, 0.0))
        ground = stitching("g", "GND", (0.0, 3.0))
        static = [static_of(v) for v in (signal, ground) if v.via.net == still]
        moving = [split_from(v, static) for v in (signal, ground) if v.via.net != still]
        if static:
            # A fixed attractor stands in the static_vias field alone, so
            # that its repulsion is counted once, by the pairwise term.
            assert all(v.static_vias is not None for v in moving)

        result = pair_run(moving, static)

        ends = {v.via.id: v.via.position for v in moving} | {
            s.via.id: s.via.position for s in static
        }
        for v, given in zip(result.vias, moving, strict=True):
            ends[given.via.id] = v.position
        distance = math.dist(ends["s"], ends["g"])
        assert distance == pytest.approx(NEUTRAL, abs=SETTLED)
        assert distance > scenarios.VIA_TO_VIA + RETURN_PARAMS.standoff

    def test_ground_vias_nearer_than_the_signal_via_leave_it_there(self) -> None:
        # Of the stitching via's net, around where it settles, each nearer
        # to it there than the signal via, and each pushing it equally.
        signal = static_of(stitching("s", "SIG", (0.0, 0.0)))
        end = np.array([0.0, NEUTRAL])
        around = [
            static_ground(
                f"n{k}", tuple(end + 1.2 * np.array([math.cos(a), math.sin(a)]))
            )
            for k, a in enumerate(np.radians([30.0, 150.0, 270.0]))
        ]
        assert all(math.dist(n.via.position, end) < NEUTRAL for n in around)

        # Off the end among them; from outside, their push would hold it
        # where the pull is too weak to bring it in.
        start = (0.3, NEUTRAL + 0.4)

        result = pair_run([stitching("g", "GND", start)], [signal, *around])

        (via,) = result.vias
        assert via.position == pytest.approx(tuple(end), abs=SETTLED)
        for n in around:
            assert math.dist(via.position, n.via.position) > HOLES_APART

    def test_a_signal_via_pulled_by_its_chains_pushes_a_stitching_via_aside(
        self,
    ) -> None:
        start = (5.3, 1.5)

        result = run(returned(grounded("g", start)))

        signal, ground = result.vias
        top, bottom = (c.polyline for c in result.chains)
        assert signal.position[1] < 0.5
        assert math.dist(ground.position, start) > 0.5
        assert math.dist(signal.position, ground.position) >= (
            scenarios.VIA_TO_VIA - INSIDE_TOLERANCE
        )
        for chain in (top, bottom):
            assert to_polyline(ground.position, chain) >= (
                scenarios.VIA_TO_CHAIN - INSIDE_TOLERANCE
            )

    def test_a_via_pulled_at_one_by_its_chains_stops_short_of_the_hinge(
        self,
    ) -> None:
        under = static_ground("u", scenarios.STRAIGHT_VIA)
        scenario = returned()

        result = optimize_chains(
            scenario.chains, scenario.params, COARSE, scenario.vias, static_vias=[under]
        )

        (via,) = result.vias
        assert via.position[1] < scenarios.DETOUR_VIA[1] - 1.0
        distance = math.dist(via.position, under.via.position)
        assert distance > scenarios.VIA_TO_VIA + RETURN_PARAMS.standoff

    def test_a_stitching_via_drawn_to_its_zone_s_edge_settles_there(self) -> None:
        start, top = 1.6, 2.2
        scenario = stitched(start, top=top)
        (given,) = scenario.vias
        edge = top - STITCH_MARGIN
        # Farther beyond the edge than the neutral distance, so it pulls.
        beyond = static_of(stitching("s", "SIG", (STITCH_X, edge + NEUTRAL + 0.5)))

        result = optimize_chains(
            scenario.chains,
            RETURN_PARAMS,
            COARSE,
            [replace(given, ground=True)],
            static_vias=[beyond],
        )

        (via,) = result.vias
        assert result.termination is Termination.CONVERGED
        assert edge - RETURN_PARAMS.standoff <= via.position[1]
        assert via.position[1] <= edge + INSIDE_TOLERANCE


GROUND_RINGS = scenarios.VIA_TO_VIA
"""Ring to ring plus the netclass clearance: beyond it two ground vias of
one net only repel."""


def ground_run(vias: Sequence[ViaInput], static: Sequence[StaticVia] = ()) -> RunResult:
    return optimize_chains(
        (far_chain(),),
        RETURN_PARAMS,
        replace(COARSE, max_iterations=100),
        vias,
        static_vias=static,
    )


def pairwise_distances(points: Sequence[tuple[float, float]]) -> list[float]:
    return [math.dist(a, b) for i, a in enumerate(points) for b in points[i + 1 :]]


class TestGroundVias:
    """Ground vias of one net repel each other beyond their rings and
    clearance, and keep only hole-to-hole as their limit."""

    def test_two_at_hole_to_hole_move_apart_beyond_their_rings(self) -> None:
        # Beyond the hinge's standoff, which would push them otherwise.
        low, high = (0.0, 2.0), (0.0, 2.0 + HOLES_APART + 0.05)

        result = ground_run([stitching("a", "GND", low), stitching("b", "GND", high)])

        a, b = (v.position for v in result.vias)
        assert math.dist(a, b) > GROUND_RINGS

    def test_three_around_one_signal_via_spread_out(self) -> None:
        signal = static_of(stitching("s", "SIG", (0.0, 0.0)))
        starts = [(-0.6, 2.2), (0.0, 2.2), (0.6, 2.2)]

        result = ground_run(
            [stitching(f"g{k}", "GND", at) for k, at in enumerate(starts)], [signal]
        )

        ends = [v.position for v in result.vias]
        assert min(pairwise_distances(ends)) > GROUND_RINGS
        assert min(pairwise_distances(ends)) > min(pairwise_distances(starts)) + 0.3
        for end in ends:
            assert math.dist(end, (0.0, 0.0)) < RETURN_PARAMS.attraction_range

    def test_a_grid_of_them_converges(self) -> None:
        pitch = 1.0
        grid = [
            stitching(f"g{r}{c}", "GND", (pitch * (c - 1), 2.0 + pitch * r))
            for r in range(3)
            for c in range(3)
        ]

        result = ground_run(grid)

        assert result.termination is Termination.CONVERGED
        ends = [v.position for v in result.vias]
        assert min(pairwise_distances(ends)) > pitch

    def test_one_standing_still_repels_a_moving_one(self) -> None:
        still = static_ground("s", (0.0, 2.0))
        start = (0.0, 2.0 + HOLES_APART + 0.1)

        result = ground_run([stitching("g", "GND", start)], [still])

        (via,) = result.vias
        assert math.dist(via.position, still.via.position) > GROUND_RINGS

    def test_two_of_one_signal_net_keep_only_hole_to_hole(self) -> None:
        # Within range of each other, with a ground via near.
        low = (0.0, 0.0)
        high = (0.0, HOLES_APART + 0.1)
        ground = static_ground("g", (3.0, 0.0))

        result = ground_run(
            [stitching("a", "SIG", low), stitching("b", "SIG", high)], [ground]
        )

        # The ground via draws both; they do not push each other apart.
        a, b = (v.position for v in result.vias)
        assert math.dist(a, b) <= math.dist(low, high) + SETTLED
        assert math.dist(a, b) >= HOLES_APART - INSIDE_TOLERANCE

    def test_two_of_different_ground_nets_keep_copper_clearance(self) -> None:
        start = GROUND_RINGS + 0.05
        agnd = replace(
            stitching("a", "GND", (0.0, 2.0)),
            via=scenarios.through_via("a", "AGND", (0.0, 2.0)),
        )

        result = ground_run([stitching("g", "GND", (0.0, 2.0 + start)), agnd])

        a, b = (v.position for v in result.vias)
        assert math.dist(a, b) > start

    def test_one_pulled_at_another_ground_net_s_keeps_copper_clearance(
        self,
    ) -> None:
        # Its chains pull it straight at the AGND via. Without repulsion and
        # with nothing but free space in its field, only the clearance hinge
        # of their pair can hold it.
        net = "GND"
        via = scenarios.through_via("v", net, scenarios.DETOUR_VIA)
        under = static_ground("u", scenarios.STRAIGHT_VIA, net="AGND")

        result = optimize_chains(
            scenarios.route(net, scenarios.DETOUR_VIA),
            replace(RETURN_PARAMS, repulsion=0.0),
            COARSE,
            [replace(scenarios.joining(via, 0), ground=True)],
            static_vias=[under],
        )

        (moved,) = result.vias
        assert moved.position[1] < scenarios.DETOUR_VIA[1] - 1.0
        distance = math.dist(moved.position, under.via.position)
        assert distance >= scenarios.VIA_TO_VIA - INSIDE_TOLERANCE


STITCHING_PITCH = 1.27
STITCHING_COUNT = 3
SIGNAL_KEEPOUT = 0.85
"""How near the signal via no stitching via is placed, mm."""
ROUTE_SPAN = 2.0
"""How far the route starts and ends beyond the grid, mm."""
ROUTE_RISE = 1.0
"""How far the route rises to each side of its via, mm."""


def crossed_stitching() -> tuple[list[ChainInput], list[ViaInput]]:
    """A route through a signal via amid a selected grid of GND stitching
    vias in a GND zone on both layers, which spread and creep once the
    route has its shape."""
    size = (STITCHING_COUNT - 1) * STITCHING_PITCH
    region = box(-1.0, -1.0, size + 1.0, size + 1.0)
    zone = ViaZone("GND", dict.fromkeys(("F.Cu", "B.Cu"), region), 0.2, 0.0)
    middle = size / 2 - STITCHING_PITCH / 2
    at = (middle, middle)
    signal = scenarios.through_via("x", "S", at)
    start = (-ROUTE_SPAN, middle - ROUTE_RISE)
    end = (size + ROUTE_SPAN, middle + ROUTE_RISE)
    chains = list(scenarios.route("S", at, start, end))
    vias = [scenarios.joining(signal, 0)]
    for i in range(STITCHING_COUNT):
        for j in range(STITCHING_COUNT):
            where = (i * STITCHING_PITCH, j * STITCHING_PITCH)
            if math.dist(where, at) > SIGNAL_KEEPOUT:
                via = scenarios.through_via(f"g{i}_{j}", "GND", where)
                vias.append(
                    ViaInput(
                        via,
                        (),
                        FreeSpace(),
                        scenarios.PAIR_CLEARANCE,
                        zones=(zone,),
                        ground=True,
                    )
                )
    return chains, vias


DEFAULT_PARAMS = cli.smooth_options(cli.parse_args(["smooth"])).params
STITCHED_SETTINGS = replace(DEFAULT_SETTINGS, burst_maxiter=20, max_iterations=12)
"""Short bursts keep the stitched grid's runs fast; the cap ends a run
that the stall rule does not."""

Stalls = list[tuple[list[float], list[float], int, int, bool]]


def spy_on_stalls(monkeypatch: pytest.MonkeyPatch) -> Stalls:
    """Record what each call of `optimize.stalled` was given and answered."""
    calls: Stalls = []
    real = optimize_module.stalled

    def spy(
        lowest: Sequence[float],
        lengths: Sequence[float],
        settings: OptimizerSettings,
        members: int,
        vias: int = 0,
    ) -> bool:
        answer = real(lowest, lengths, settings, members, vias)
        calls.append((list(lowest), list(lengths), members, vias, answer))
        return answer

    monkeypatch.setattr(optimize_module, "stalled", spy)
    return calls


def test_moving_vias_end_a_stitched_grid_s_creep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chains, vias = crossed_stitching()
    calls = spy_on_stalls(monkeypatch)

    result = optimize_chains(chains, DEFAULT_PARAMS, STITCHED_SETTINGS, vias)

    lowest, lengths, members, counted, answer = calls[-1]
    assert result.termination is Termination.CONVERGED
    assert answer
    assert (members, counted) == (len(chains), len(vias))
    # Without the vias' share the problem would have gone on.
    assert not stalled(lowest, lengths, STITCHED_SETTINGS, members)


def test_shortening_tracks_keep_a_stitched_grid_going(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chains, vias = crossed_stitching()
    calls = spy_on_stalls(monkeypatch)
    shrinking = itertools.count()
    # More than a tolerance over the window.
    shortening = DEFAULT_SETTINGS.stall_tolerance

    def length(_: object) -> float:
        return 10.0 - shortening * next(shrinking)

    monkeypatch.setattr(optimize_module._Problem, "length", length)

    optimize_chains(chains, DEFAULT_PARAMS, STITCHED_SETTINGS, vias)

    assert not any(answer for *_, answer in calls)
    # The energy alone would have stalled it.
    assert any(
        stalled(lowest, [0.0] * len(lengths), STITCHED_SETTINGS, members, counted)
        for lowest, lengths, members, counted, _ in calls
    )
