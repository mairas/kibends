# pyright: reportPrivateUsage=false
import math
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
    CompositeField,
    build_via_obstacles,
    zone_containment,
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


def in_zone(bottom: float) -> ViasScenario:
    """The detour route with its via in a zone of its net whose region ends at
    y = `bottom`, short of the straight route at y = 0."""
    via = scenarios.through_via("v", "N", scenarios.DETOUR_VIA)
    return ViasScenario(
        "via-in-zone",
        scenarios.route("N", scenarios.DETOUR_VIA),
        (scenarios.joining(via, 0, CompositeField(zone_containment(plane(bottom)))),),
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
        # Onto the pinned chain of its net on In1.Cu.
        (scenarios.via_over_same_net(), (5.0, scenarios.SAME_NET_LEVEL)),
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


TOUCHING = scenarios.VIA_DIAMETER / 2 + scenarios.PAIR_WIDTH / 2


def test_a_via_keeps_off_a_chain_of_its_net_that_does_not_end_at_it() -> None:
    result = run(scenarios.via_over_same_net())

    (via,) = result.vias
    inner = result.chains[2].polyline
    x, y = via.position
    assert to_polyline(via.position, inner) >= TOUCHING - INSIDE_TOLERANCE
    assert y_at(inner, x) < y


# Pulled onto the chain against a weak clearance hinge, the via ends within
# this much of touching it, mm.
TIGHT = 2e-3


def over_short_same_net() -> ViasScenario:
    """`scenarios.via_over_same_net` with the pinned chain only under the
    via's way, pinned at fewer points."""
    scenario = scenarios.via_over_same_net()
    inner = scenario.chains[2]
    count = 25
    points = np.column_stack(
        [np.linspace(2.0, 8.0, count), np.full(count, scenarios.SAME_NET_LEVEL)]
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
    scenario = over_short_same_net()
    params = replace(scenario.params, clearance_weight=0.1)
    shorter = replace(COARSE, burst_maxiter=15)

    result = optimize_chains(scenario.chains, params, shorter, scenario.vias)

    (via,) = result.vias
    gap = to_polyline(via.position, result.chains[2].polyline)
    assert gap >= TOUCHING - INSIDE_TOLERANCE
    assert gap < TOUCHING + TIGHT
    assert via.rejected_edges > 0


# The chain wraps the via and pushes it up as it does. Each held to half
# their gap, the two crept after each other: 81 iterations. Measured: 25.
PRESSING_ITERATIONS = 40


def test_a_chain_of_its_net_that_does_not_end_at_it_keeps_off_the_via() -> None:
    result = run(scenarios.via_over_bowed_same_net())

    (via,) = result.vias
    inner = result.chains[2].polyline
    assert result.termination is Termination.CONVERGED
    assert result.iterations <= PRESSING_ITERATIONS
    assert to_polyline(via.position, inner) >= TOUCHING - INSIDE_TOLERANCE
    assert length(inner) < 4.4


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
