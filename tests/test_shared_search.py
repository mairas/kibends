"""The energy and the vias' free distances found through search trees
shared between members, against a reference that builds a field of its own
for each member and each via. The reference decides itself what each
member and each via sees, by net and layer, with copper of one net left
out, so that copper of one net interacting in what the members and the
moving vias see of each other fails it; only the distances, the vias'
rings and a section's leave-out near its fan-outs (`kept_spans`, which
test_neighbours checks) come from the code under test.

Where two pieces answer a point at exactly the same distance, which one
does depends on the order a tree lists them in, so the gradient may go to
either: the states here have no such point. A grid of routes whose vias
stand in line has them at the start, where a member's end lies midway
between the vias above and below it."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

import numpy as np
import pytest
import scenarios
from distance_fields import FloatArray, FreeSpace
from reference_fields import chain_obstacles_for_via, via_obstacles_for_via
from scenarios import ChainsScenario, PairsScenario, ViasScenario, selected
from shapely.geometry import box

import kibends.optimize as optimize_module
from kibends.bursts import DEFAULT_SETTINGS, point_distance
from kibends.energy import (
    BoolArray,
    EnergyParams,
    IntArray,
    ObstacleField,
    bend_radius_energy,
    edge_integral,
    obstacle_integrand,
    obstacle_reach,
    quadrature_points,
    shape_energy,
)
from kibends.members import Member, members_energy, seen_parts, shown
from kibends.neighbours import ChainInput, kept_spans
from kibends.obstacles import (
    HOLE_CLEARANCE,
    HOLE_TO_HOLE,
    ChainField,
    CompositeField,
    zones_on,
)
from kibends.optimize import LIST_SKIN
from kibends.pair_blocks import PairBlock, pair_block
from kibends.seen import NeighbourList, SeenParts
from kibends.snapshot import ViaZone
from kibends.via_moves import MovingVias
from kibends.vias import ViaInput

Scenario = ChainsScenario | PairsScenario | ViasScenario

SCENARIOS: list[Callable[[], Scenario]] = [
    scenarios.centreline_over_pad,
    scenarios.channel_pair,
    scenarios.crowded,
    scenarios.drape,
    scenarios.hooks,
    scenarios.width_change,
    scenarios.same_net_wrap,
    scenarios.squeeze,
    scenarios.tent,
    scenarios.uneven_channel_pair,
    scenarios.pair_chamfered,
    scenarios.pair_corner,
    scenarios.pair_over_pad,
    scenarios.pair_round_corner,
    scenarios.pair_split,
    scenarios.pair_tent,
    scenarios.pair_turning_back,
    scenarios.pair_uneven_end,
    scenarios.via_across_chain,
    scenarios.via_beside_chain,
    scenarios.via_corner,
    scenarios.via_detour,
    scenarios.via_fermat,
    scenarios.via_hemmed,
    scenarios.via_on_one_layer,
    scenarios.via_pair,
    lambda: _pair_tent_over("P"),
    lambda: _pair_tent_over("Q"),
    lambda: _pair_tent_fanning_out(),
    lambda: scenarios.via_over_bowed("N"),
    lambda: scenarios.via_over_bowed("P"),
    lambda: scenarios.via_over_pinned("N"),
    lambda: scenarios.via_over_pinned("P"),
    lambda: scenarios.via_detour_over(FreeSpace()),
    lambda: scenarios.via_slack(FreeSpace()),
    lambda: scenarios.via_grid(3, 3),
    lambda: _detour_among_ground_vias(),
]
IDS = [
    *(make.__name__ for make in SCENARIOS[:-11]),
    "pair_tent_over_its_own_net",
    "pair_tent_over_another_net",
    "pair_tent_fanning_out",
    "via_over_bowed_same_net",
    "via_over_bowed_other_net",
    "via_over_pinned_same_net",
    "via_over_pinned_other_net",
    "via_detour_over",
    "via_slack",
    "via_grid",
    "via_detour_among_ground_vias",
]


def _detour_among_ground_vias() -> ViasScenario:
    """`scenarios.via_detour` with two GND vias, an AGND via and a regular
    via of another net moving near its via and each other, so that ground
    and regular vias, ground vias of one net and of two nets, and two
    regular vias meet."""
    scenario = scenarios.via_detour()

    def stitching(via_id: str, net: str, at: tuple[float, float]) -> ViaInput:
        return ViaInput(
            scenarios.through_via(via_id, net, at),
            (),
            FreeSpace(),
            scenarios.PAIR_CLEARANCE,
            ground=net.endswith("GND"),
        )

    return replace(
        scenario,
        vias=(
            *scenario.vias,
            stitching("g", "GND", (5.0, 4.0)),
            stitching("h", "GND", (5.6, 4.3)),
            stitching("a", "AGND", (6.0, 2.4)),
            stitching("p", "P", (4.0, 2.5)),
        ),
    )


def _pair_tent_over(net: str) -> PairsScenario:
    """`scenarios.pair_tent` with a bar of `net` under the section's tracks,
    near enough for the two to see each other, away from its fan-outs."""
    bar = selected("bar", net, FreeSpace(), (4.0, 2.2), (5.0, 2.4), (6.0, 2.2))
    return replace(scenarios.pair_tent(), chains=(bar,))


def _pair_tent_fanning_out() -> PairsScenario:
    """`scenarios.pair_tent` with a fan-out of each net leaving its track's
    end, so that the section leaves each out near where it touches it."""
    scenario = scenarios.pair_tent()
    (section,) = scenario.pairs
    fans = (
        _fan_out("P", section.a.points[-1], -1.0),
        _fan_out("N", section.b.points[-1], 0.5),
    )
    return replace(scenario, chains=(*scenario.chains, *fans))


def _fan_out(net: str, end: FloatArray, drop: float) -> ChainInput:
    """A chain of `net` from `end`, falling by `drop` over its first mm and
    then running level."""
    x, y = float(end[0]), float(end[1])
    return selected(
        f"fan {net}", net, FreeSpace(), (x, y), (x + 1.0, y + drop), (x + 2.0, y + drop)
    )


JITTER = 0.03
"""How far each variable moves from the start, at most, mm: past the
neighbour list's skin, so that it finds its pairs again."""
WITHIN_SKIN = 0.2 * LIST_SKIN
"""How far each variable moves from the start, at most, mm, for a state the
neighbour list answers with the pairs it found at the start."""


def _problems(
    scenario: Scenario, params: EnergyParams
) -> list[optimize_module._Problem]:  # pyright: ignore[reportPrivateUsage]
    chains = list(scenario.chains)
    pairs = list(getattr(scenario, "pairs", ()))
    blocks = [pair_block(pair, DEFAULT_SETTINGS.spacing) for pair in pairs]
    everything = [*chains, *(block.chain for block in blocks)]
    vias = MovingVias(
        everything, list(getattr(scenario, "vias", ())), HOLE_CLEARANCE, HOLE_TO_HOLE
    )
    placed = optimize_module._problems(  # pyright: ignore[reportPrivateUsage]
        chains, blocks, vias, params, DEFAULT_SETTINGS, None
    )
    return [p.problem for p in placed if p.problem.moving]


def _parameters(points: FloatArray, starts: FloatArray, ends: FloatArray) -> FloatArray:
    along = ends - starts
    squared = np.einsum("ij,ij->i", along, along)
    projection = np.einsum("ij,ij->i", points - starts, along)
    return np.clip(projection / np.where(squared > 0.0, squared, 1.0), 0.0, 1.0)


def _covered(
    points: FloatArray, spans: Sequence[tuple[float, float]] | None
) -> tuple[IntArray, FloatArray, FloatArray]:
    """Each edge of `points` and where along it, from 0 to 1, the arc-length
    `spans` cover it, by default all of it."""
    count = len(points) - 1
    if spans is None:
        return np.arange(count, dtype=np.intp), np.zeros(count), np.ones(count)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    edges: list[int] = []
    starts: list[float] = []
    ends: list[float] = []
    for left, right in spans:
        for e in range(count):
            a, b = float(arc[e]), float(arc[e + 1])
            if a < right and b > left:
                span = b - a if b > a else 1.0
                edges.append(e)
                starts.append(min(max((left - a) / span, 0.0), 1.0))
                ends.append(min(max((right - a) / span, 0.0), 1.0))
    return (
        np.array(edges, dtype=np.intp),
        np.array(starts, dtype=float),
        np.array(ends, dtype=float),
    )


def _reference_parts(
    problem: optimize_module._Problem,  # pyright: ignore[reportPrivateUsage]
    polylines: Sequence[FloatArray],
) -> list[SeenParts]:
    """What each member sees, as copper of one net does not interact: every
    track of another net on its layer, inflated by both half widths and the
    larger clearance, and every via spanning its layer of another net,
    inflated by its ring. A chain sees each track whole; a section sees as
    its centreline of the pair's outer extent, and leaves a track of either
    of its nets out near where it touches it (`kept_spans`)."""
    members = problem.members
    lines = [shown(m, p) for m, p in zip(members, polylines, strict=True)]
    lengths = [len(points) for parts in lines for _, points in parts]
    firsts = np.cumsum([0, *lengths])
    vias_from = int(firsts[-1])
    parts: list[SeenParts] = []
    for i, member in enumerate(members):
        own = member.facing if isinstance(member, PairBlock) else member
        nets: Mapping[str, ChainInput] = (
            member.as_net if isinstance(member, PairBlock) else {}
        )
        me = own.centreline
        vertex = [np.empty(0, dtype=np.intp)]
        start = [np.empty(0)]
        end = [np.empty(0)]
        inflation = [np.empty(0)]
        track = 0
        for j, tracks in enumerate(lines):
            for other, points in tracks:
                them = other.centreline
                if j != i and other.layer == own.layer and them.net != me.net:
                    margin = max(me.clearance, them.clearance)
                    viewer = nets.get(them.net)
                    spans = (
                        None
                        if viewer is None
                        else kept_spans(
                            viewer,
                            other,
                            points,
                            problem.reach,
                            polylines[i],
                            margin=margin,
                        )
                    )
                    edge, a, b = _covered(points, spans)
                    vertex.append(firsts[track] + edge)
                    start.append(a)
                    end.append(b)
                    inflation.append(
                        np.full(len(edge), (me.width + them.width) / 2 + margin)
                    )
                track += 1
        body = problem.bodies[i]
        for k, via in enumerate(problem.vias):
            given = via.given.via
            if given.reaches(body.layer) and given.net != body.centreline.net:
                vertex.append(np.array([vias_from + 2 * k]))
                start.append(np.zeros(1))
                end.append(np.ones(1))
                inflation.append(np.array([via.rings[i]]))
        parts.append(
            SeenParts(
                np.concatenate(vertex),
                np.concatenate(start),
                np.concatenate(end),
                np.concatenate(inflation),
            )
        )
    return parts


def _reference_members(
    members: Sequence[Member],
    polylines: Sequence[FloatArray],
    masks: Sequence[BoolArray],
    params: EnergyParams,
    short_edge: float,
    seen: Sequence[SeenParts],
    positions: Sequence[FloatArray],
) -> tuple[float, list[FloatArray], FloatArray]:
    """`members_energy` with a field of each member's own pieces."""
    reach = obstacle_reach(params)
    integrand = obstacle_integrand(params)
    k = params.points_per_edge
    lines = [shown(m, p) for m, p in zip(members, polylines, strict=True)]
    flat = np.concatenate(
        [
            *(points for parts in lines for _, points in parts),
            *(np.array([at, at]) for at in positions),
        ]
    )
    at_shown = np.zeros_like(flat)
    gradients: list[FloatArray] = []
    value = 0.0
    for i, (member, points) in enumerate(zip(members, polylines, strict=True)):
        body = member.chain if isinstance(member, PairBlock) else member
        queries = quadrature_points(points, k)
        static, static_direction = body.field.signed_distance(queries, reach)
        part = seen[i]
        a, b = flat[part.vertex], flat[part.vertex + 1]
        starts = a + part.start[:, None] * (b - a)
        ends = a + part.end[:, None] * (b - a)
        near = ChainField(starts, ends, part.inflation).nearest(queries, reach)
        nearer = near.distance < static
        distance = np.where(nearer, near.distance, static)
        direction = np.where(nearer[:, None], near.gradient, static_direction)
        obstacle, obstacle_gradient, slopes = edge_integral(
            points, distance, direction, integrand, k
        )
        shape, shape_gradient = shape_energy(points, masks[i], short_edge)
        value += shape + obstacle
        gradient = shape_gradient + obstacle_gradient
        if isinstance(member, PairBlock):
            bend, bend_gradient = bend_radius_energy(
                points, masks[i], member.limit, params
            )
            value += bend
            gradient += bend_gradient
        gradients.append(gradient)
        pulled = np.flatnonzero(nearer & (slopes != 0.0))
        edge = near.edge[pulled]
        start = part.vertex[edge]
        along = _parameters(queries[pulled], starts[edge], ends[edge])
        t = part.start[edge] + along * (part.end[edge] - part.start[edge])
        push = slopes[pulled, None] * near.gradient[pulled]
        np.add.at(at_shown, start, -push * (1 - t)[:, None])
        np.add.at(at_shown, start + 1, -push * t[:, None])
    first = 0
    for member, points, gradient, parts in zip(
        members, polylines, gradients, lines, strict=True
    ):
        pieces: list[FloatArray] = []
        for _, track in parts:
            pieces.append(at_shown[first : first + len(track)])
            first += len(track)
        if isinstance(member, PairBlock):
            gradient += member.offset_pullback(points, *pieces)
        else:
            gradient += pieces[0]
    for g, mask in zip(gradients, masks, strict=True):
        g[mask] = 0.0
    return value, gradients, at_shown[first:].reshape(-1, 2, 2).sum(axis=1)


def _reference_view(
    problem: optimize_module._Problem,  # pyright: ignore[reportPrivateUsage]
    k: int,
    polylines: Sequence[FloatArray],
    positions: Sequence[FloatArray],
    *,
    unpaired: bool = False,
) -> tuple[ChainField, FloatArray, FloatArray, list[tuple[int, int]]]:
    """What via k sees of the members and the other vias, as a field of its
    own, its edges' starts and ends, and each edge's member and first
    vertex, or -1 and the via; `unpaired` leaves out the vias that only stop
    it, of its net, and those the pairwise term takes instead, of other
    nets where either is a ground via."""
    via = problem.vias[k]
    starts = [np.empty((0, 2))]
    ends = [np.empty((0, 2))]
    inflations = [np.empty(0)]
    owner: list[tuple[int, int]] = []
    for j, body in enumerate(problem.bodies):
        given = via.given.via
        if not given.reaches(body.layer) or body.centreline.net == given.net:
            # Copper of one net does not interact.
            continue
        ring = via.rings[j]
        points = polylines[j]
        starts.append(points[:-1])
        ends.append(points[1:])
        inflations.append(np.full(len(points) - 1, ring))
        owner += [(j, e) for e in range(len(points) - 1)]
    for u, apart in via.apart.items():
        other = problem.vias[u].given
        if unpaired and (
            other.via.net == via.given.via.net or other.ground or via.given.ground
        ):
            continue
        starts.append(positions[u][None])
        ends.append(positions[u][None])
        inflations.append(np.array([apart]))
        owner.append((-1, u))
    a, b = np.concatenate(starts), np.concatenate(ends)
    return ChainField(a, b, np.concatenate(inflations)), a, b, owner


def _reference_distances(
    problem: optimize_module._Problem,  # pyright: ignore[reportPrivateUsage]
    polylines: Sequence[FloatArray],
    positions: Sequence[FloatArray],
) -> list[float]:
    """`_Problem.via_distances` with a field of each via's own view."""
    return [
        min(
            float(
                problem.vias[k].given.field.signed_distance(positions[k][None])[0][0]
            ),
            float(
                _reference_view(problem, k, polylines, positions)[0].signed_distance(
                    positions[k][None]
                )[0][0]
            ),
        )
        for k in problem.free_vias
    ]


def _reference_vias(
    problem: optimize_module._Problem,  # pyright: ignore[reportPrivateUsage]
    polylines: Sequence[FloatArray],
    positions: Sequence[FloatArray],
) -> tuple[float, list[FloatArray], FloatArray]:
    """`_Problem.via_energy` with a field of each via's own view."""
    gradients = [np.zeros_like(points) for points in polylines]
    via_gradients = np.zeros((len(problem.vias), 2))
    value = 0.0
    for k in problem.free_vias:
        via = problem.vias[k]
        at = positions[k][None]
        static, static_direction = via.given.field.signed_distance(at, problem.reach)
        field, a, b, owner = _reference_view(
            problem, k, polylines, positions, unpaired=True
        )
        near = field.nearest(at, problem.reach)
        nearer = bool(near.distance[0] < static[0])
        distance = near.distance if nearer else static
        direction = near.gradient[0] if nearer else static_direction[0]
        term, slope = problem.integrand(distance)
        value += via.given.via.diameter * float(term[0])
        push = via.given.via.diameter * slope[0] * direction
        via_gradients[k] += push
        if not nearer:
            continue
        e = int(near.edge[0])
        j, first = owner[e]
        if j < 0:
            via_gradients[first] -= push
            continue
        t = _parameters(at, a[e : e + 1], b[e : e + 1])[0]
        gradients[j][first] -= (1 - t) * push
        gradients[j][first + 1] -= t * push
    return value, gradients, via_gradients


def _assert_same(
    got: tuple[float, list[FloatArray], FloatArray],
    expected: tuple[float, list[FloatArray], FloatArray],
) -> None:
    assert got[0] == pytest.approx(expected[0], rel=1e-10, abs=1e-12)
    for g, e in zip(got[1], expected[1], strict=True):
        np.testing.assert_allclose(g, e, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(got[2], expected[2], rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("repulsion", [None, 0.5])
@pytest.mark.parametrize("make", SCENARIOS, ids=IDS)
def test_the_energy_is_the_one_each_members_own_field_gives(
    make: Callable[[], Scenario], repulsion: float | None
) -> None:
    """At the start, moved within the skin and moved past it, the members'
    energy and the vias' own, searched and through a neighbour list, and
    the vias' free distances are what fields built for each member and via
    alone give, with copper of one net left out of what a chain or a via
    sees."""
    scenario = make()
    params = (
        scenario.params
        if repulsion is None
        else replace(scenario.params, repulsion=repulsion)
    )
    problems = _problems(scenario, params)
    assert problems
    rng = np.random.default_rng(0)
    for problem in problems:
        layout = problem.layout(problem.points)
        seen = seen_parts(problem.members, layout.starts, problem.reach, problem.rings)
        masks = problem.energy_masks(problem.masks)
        listed = NeighbourList(seen, problem.reach, LIST_SKIN)
        for jitter in (0.0, WITHIN_SKIN, JITTER):
            x = layout.x0 + jitter * rng.uniform(-1.0, 1.0, len(layout.x0))
            polylines, positions = layout.unflatten(x)
            arguments = (
                problem.members,
                polylines,
                masks,
                params,
                DEFAULT_SETTINGS.short_edge,
                seen,
                positions,
            )
            expected = _reference_members(
                *arguments[:5],
                _reference_parts(problem, layout.starts),
                positions,
            )
            _assert_same(members_energy(*arguments), expected)
            _assert_same(members_energy(*arguments, listed), expected)
            _assert_same(
                problem.via_energy(polylines, positions),
                _reference_vias(problem, polylines, positions),
            )
            np.testing.assert_allclose(
                problem.via_distances(polylines, positions),
                _reference_distances(problem, polylines, positions),
                rtol=1e-12,
                atol=1e-12,
            )


@pytest.mark.parametrize("make", SCENARIOS, ids=IDS)
def test_the_neighbour_field_is_what_seen_parts_sees(
    make: Callable[[], Scenario],
) -> None:
    """What frozen detection and the best-state guard judge against
    (`_Problem.neighbour`) is what the energy and the edge check
    see of the other members (`seen_parts`): the same distances, at each
    member's own points and around them, at the start and moved, a
    section's leave-out near its own nets' chains included."""
    scenario = make()
    rng = np.random.default_rng(0)
    for problem in _problems(scenario, scenario.params):
        layout = problem.layout(problem.points)
        for jitter in (0.0, JITTER):
            x = layout.x0 + jitter * rng.uniform(-1.0, 1.0, len(layout.x0))
            polylines, _ = layout.unflatten(x)
            seen = seen_parts(problem.members, polylines, problem.reach)
            flat = np.concatenate(
                [
                    points
                    for member, line in zip(problem.members, polylines, strict=True)
                    for _, points in shown(member, line)
                ]
            )
            everywhere = np.concatenate(polylines)
            queries = np.concatenate(
                [
                    everywhere,
                    everywhere + rng.uniform(-0.5, 0.5, everywhere.shape),
                ]
            )
            for i in range(len(problem.members)):
                part = seen[i]
                a, b = flat[part.vertex], flat[part.vertex + 1]
                expected, _ = ChainField(
                    a + part.start[:, None] * (b - a),
                    a + part.end[:, None] * (b - a),
                    part.inflation,
                ).signed_distance(queries)
                field = problem.neighbour(i, problem.neighbours[i], polylines)
                got = (
                    np.full(len(queries), np.inf)
                    if field is None
                    else field.signed_distance(queries)[0]
                )
                np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


def _reference_free_distance(
    vias: MovingVias,
    v: int,
    positions: Sequence[FloatArray],
    polylines: Sequence[FloatArray],
) -> float:
    """Via v's free distance as the frozen judgement takes it, from fields
    of its own: its limits, the chains of other nets on each layer it spans
    with its zones there, and the other vias by its own zones alone."""
    given = vias.vias[v]
    via = given.via
    fields: list[ObstacleField] = [given.limits]
    for layer in dict.fromkeys(chain.layer for chain in vias.chains):
        near = [
            replace(chain.centreline, points=points)
            for chain, points in zip(vias.chains, polylines, strict=True)
            if chain.layer == layer
            and via.reaches(layer)
            and chain.centreline.net != via.net
        ]
        fields.append(
            chain_obstacles_for_via(
                via,
                near,
                vias.clearance_of,
                hole_clearance=vias.hole_clearance,
                zones=zones_on(given.zones, [layer]),
            )
        )
    others = via_obstacles_for_via(
        via,
        [vias.at(u, positions) for u in range(len(vias.vias)) if u != v],
        vias.clearance_of,
        hole_clearance=vias.hole_clearance,
        hole_to_hole=vias.hole_to_hole,
        zones=given.zones,
    )
    if others is not None:
        fields.append(others)
    return point_distance(CompositeField(fields), positions[v])


VIA_SCENARIOS = [
    (make, name)
    for make, name in zip(SCENARIOS, IDS, strict=True)
    if name.startswith("via_")
]


ZONE_CLEARANCE = 0.4
"""Wider than the scenarios' netclass clearance, so that a zone's clearance
sets how far its via keeps from other-net copper, mm."""
PLANE = box(-50.0, -50.0, 50.0, 50.0)
"""A zone region reaching every via of the scenarios, far from all of them."""


def _zoned(
    scenario: ViasScenario, zones: Callable[[int], tuple[ViaZone, ...]]
) -> ViasScenario:
    """`scenario` with each via k staying inside `zones(k)`."""
    return replace(
        scenario,
        vias=tuple(replace(via, zones=zones(k)) for k, via in enumerate(scenario.vias)),
    )


def _plane(layer: str, min_width: float = 0.2) -> ViaZone:
    return ViaZone(f"plane {layer}", {layer: PLANE}, min_width, ZONE_CLEARANCE)


ZONED_VIA_SCENARIOS: list[tuple[Callable[[], Scenario], str]] = [
    (
        lambda: _zoned(scenarios.via_beside_chain(), lambda _: (_plane("In1.Cu"),)),
        "via_beside_chain_zoned_on_its_layer",
    ),
    (
        lambda: _zoned(scenarios.via_beside_chain(), lambda _: (_plane("In2.Cu"),)),
        "via_beside_chain_zoned_on_another_layer",
    ),
    (
        lambda: _zoned(
            scenarios.via_grid(3, 2, via_every=1),
            lambda k: (_plane("B.Cu" if k % 2 else "F.Cu"),),
        ),
        "via_grid_zoned_on_alternating_layers",
    ),
    (
        lambda: _zoned(
            scenarios.via_pair(), lambda k: (_plane("In1.Cu", 0.2 + 0.6 * k),)
        ),
        "via_pair_zoned_with_different_margins",
    ),
]


@pytest.mark.parametrize(
    "make",
    [make for make, _ in (*VIA_SCENARIOS, *ZONED_VIA_SCENARIOS)],
    ids=[n for _, n in (*VIA_SCENARIOS, *ZONED_VIA_SCENARIOS)],
)
def test_the_free_distance_judging_vias_frozen_is_their_own_fields(
    make: Callable[[], Scenario],
) -> None:
    """The free distance `_problems` freezes vias by (`MovingVias.distances`)
    is what a field of each via's own view gives, at the start and moved,
    with each via's zones on the layers they lie on, and by its own zones
    alone against another via."""
    scenario = make()
    assert isinstance(scenario, ViasScenario)
    vias = MovingVias(
        list(scenario.chains), list(scenario.vias), HOLE_CLEARANCE, HOLE_TO_HOLE
    )
    rng = np.random.default_rng(0)
    originals = [chain.centreline.points for chain in vias.chains]
    for jitter in (0.0, JITTER):
        positions = [
            at + jitter * rng.uniform(-1.0, 1.0, 2) for at in vias.starting_positions()
        ]
        polylines = [
            points + jitter * rng.uniform(-1.0, 1.0, points.shape)
            for points in originals
        ]
        np.testing.assert_allclose(
            vias.distances(positions, polylines),
            [
                _reference_free_distance(vias, v, positions, polylines)
                for v in range(len(vias.vias))
            ],
            rtol=1e-12,
            atol=1e-12,
        )
