"""The energy and the vias' free distances found through search trees
shared between members, and the checks on what the members see of each
other, against a reference that builds a field of its own for each member
and each via. The reference decides itself what each
member and each via sees, by net and layer, with copper of one net left
out, so that copper of one net interacting in what the members and the
moving vias see of each other fails it; only the distances, the vias'
rings and a section's leave-out near its fan-outs (`kept_spans`, which
test_neighbours checks) come from the code under test.

The energy takes the smooth minimum over obstacles, so two obstacles
equally near a point share its gradient whatever order a tree lists them
in. Within one obstacle, where two of its pieces answer a point at exactly
the same distance, which one does depends on that order, so the gradient
may go to either: the states here have no such point."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

import numpy as np
import pytest
import scenarios
from distance_fields import FloatArray, FreeSpace
from reference_fields import chain_obstacles_for_via
from scenarios import (
    PAIR_INFLATION,
    ChainsScenario,
    PairsScenario,
    ViasScenario,
    selected,
)
from scipy.spatial import cKDTree
from shapely.geometry import box

import kibends.obstacles as obstacles_module
import kibends.optimize as optimize_module
import kibends.seen as seen_module
from kibends.bursts import DEFAULT_SETTINGS, point_distance
from kibends.energy import (
    BoolArray,
    EnergyParams,
    IntArray,
    bend_radius_energy,
    combine_fields,
    edge_integral,
    obstacle_integrand,
    obstacle_reach,
    quadrature_points,
    shape_energy,
)
from kibends.members import Member, members_energy, seen_parts, shown, shown_flat
from kibends.neighbours import ChainInput, kept_spans
from kibends.obstacles import (
    HOLE_CLEARANCE,
    HOLE_TO_HOLE,
    ChainField,
    CompositeField,
    complex_points,
    pairs_inside,
    via_via_inflation,
    zones_on,
)
from kibends.optimize import PRESSED_DEPTH
from kibends.pair_blocks import PairBlock, pair_block
from kibends.polylines import measure, violations
from kibends.seen import SKIN_LEAST, NeighbourList, Pieces, Seen, Track, Viewer
from kibends.snapshot import ViaZone
from kibends.via_moves import FreeDistances, MovingVias
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
    lambda: scenarios.via_grid(3, 3, via_every=1),
    lambda: _detour_among_ground_vias(),
    lambda: _one_per_layer(),
    lambda: _stitching_alone(),
    lambda: _mixed_netclasses(),
]
IDS = [
    *(make.__name__ for make in SCENARIOS[:-15]),
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
    "via_grid_with_a_via_at_every_vertex",
    "via_detour_among_ground_vias",
    "one_per_layer",
    "stitching_alone",
    "mixed_netclasses",
]


def _detour_among_ground_vias() -> ViasScenario:
    """`scenarios.via_detour` with two GND vias, an AGND via and a regular
    via of another net moving near its via and each other, so that ground
    and regular vias, ground vias of one net and of two nets, and two
    regular vias meet."""
    scenario = scenarios.via_detour()
    return replace(scenario, vias=(*scenario.vias, *_stitching_alone().vias))


def _stitching(via_id: str, net: str, at: tuple[float, float]) -> ViaInput:
    return ViaInput(
        scenarios.through_via(via_id, net, at),
        (),
        FreeSpace(),
        scenarios.PAIR_CLEARANCE,
        ground=net.endswith("GND"),
    )


def _one_per_layer() -> ChainsScenario:
    """`scenarios.tent` with its bar on the other side, so that each layer
    holds one member."""
    arch, bar = scenarios.tent().chains
    return ChainsScenario("one-per-layer", (arch, replace(bar, layer="B.Cu")))


WIDE_WIDTH = 0.3
WIDE_CLEARANCE = 0.25
"""A second netclass's track width and clearance, mm, both wider than the
scenarios'."""


def _mixed_netclasses() -> ViasScenario:
    """`scenarios.via_detour`, whose net runs on two layers, with a chain of
    a second netclass above its via and two chains of no net below its
    route, which keep clear of the others but not of each other."""
    scenario = scenarios.via_detour()
    wide = selected("wide", "W", FreeSpace(), (2.0, 4.0), (5.0, 4.2), (8.0, 4.0))
    wide = replace(
        wide,
        centreline=replace(wide.centreline, width=WIDE_WIDTH, clearance=WIDE_CLEARANCE),
    )
    unnamed = [
        selected(f"unnamed {k}", "", FreeSpace(), (1.0, y), (5.0, y + 0.4), (9.0, y))
        for k, y in enumerate((-1.0, -1.6))
    ]
    return replace(scenario, chains=(*scenario.chains, wide, *unnamed))


def _stitching_alone() -> ViasScenario:
    """The stitching vias of `_detour_among_ground_vias` without a chain."""
    return ViasScenario(
        "stitching-alone",
        (),
        (
            _stitching("g", "GND", (5.0, 4.0)),
            _stitching("h", "GND", (5.6, 4.3)),
            _stitching("a", "AGND", (6.0, 2.4)),
            _stitching("p", "P", (4.0, 2.5)),
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


DISTANCE_ROUNDING = 1e-12
"""How far a distance the shared search gives may part from a member's
own field's, which takes the same differences, mm."""
JITTER = SKIN_LEAST
"""How far each variable moves from the start, at most, mm: up to
SKIN_LEAST times the square root of two for a vertex, past the skin a
neighbour list starts with, so that it finds its pairs again with a
wider skin."""
WITHIN_SKIN = 0.2 * SKIN_LEAST
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
) -> list[tuple[Pieces, IntArray]]:
    """What each member sees, as copper of one net does not interact: every
    track of another net on its layer, inflated by both half widths and the
    larger clearance, and every via spanning its layer of another net,
    inflated by its ring. A chain sees each track whole; a section sees as
    its centreline of the pair's outer extent, and leaves a track of either
    of its nets out near where it touches it (`kept_spans`). With them, each
    piece's obstacle: each track shown, a section's two apart, and each via
    is one."""
    members = problem.members
    lines = [shown(m, p) for m, p in zip(members, polylines, strict=True)]
    lengths = [len(points) for parts in lines for _, points in parts]
    firsts = np.cumsum([0, *lengths])
    vias_from = int(firsts[-1])
    parts: list[tuple[Pieces, IntArray]] = []
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
        obstacle = [np.empty(0, dtype=np.intp)]
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
                    obstacle.append(np.full(len(edge), track, dtype=np.intp))
                track += 1
        body = problem.bodies[i]
        for k, via in enumerate(problem.vias):
            given = via.given.via
            if given.reaches(body.layer) and given.net != body.centreline.net:
                vertex.append(np.array([vias_from + 2 * k]))
                start.append(np.zeros(1))
                end.append(np.ones(1))
                inflation.append(np.array([via.rings[i]]))
                obstacle.append(np.array([len(lengths) + k], dtype=np.intp))
        parts.append(
            (
                Pieces(
                    np.concatenate(vertex),
                    np.concatenate(start),
                    np.concatenate(end),
                    np.concatenate(inflation),
                ),
                np.concatenate(obstacle),
            )
        )
    return parts


def _reference_members(
    members: Sequence[Member],
    polylines: Sequence[FloatArray],
    masks: Sequence[BoolArray],
    params: EnergyParams,
    short_edge: float,
    seen: Sequence[tuple[Pieces, IntArray]],
    positions: Sequence[FloatArray],
) -> tuple[float, list[FloatArray], FloatArray]:
    """`members_energy` with a field of each member's own pieces, the
    smooth minimum taken over them and the static field together."""
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
        static, static_direction = body.field.smooth_distance(queries, reach)
        part, obstacles = seen[i]
        a, b = flat[part.vertex], flat[part.vertex + 1]
        starts = a + part.start[:, None] * (b - a)
        ends = a + part.end[:, None] * (b - a)
        near = ChainField(starts, ends, part.inflation, obstacles).smooth_answers(
            queries, reach
        )
        distance, (static_share, seen_share) = combine_fields(static, near.distance)
        direction = (
            static_share[:, None] * static_direction
            + seen_share[:, None] * near.gradient
        )
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
        point = near.point
        edge = near.edge
        start = part.vertex[edge]
        along = _parameters(queries[point], starts[edge], ends[edge])
        t = part.start[edge] + along * (part.end[edge] - part.start[edge])
        push = (slopes * seen_share)[point, None] * near.share[:, None] * near.direction
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
    other_nets: bool = False,
) -> tuple[ChainField, FloatArray, FloatArray, list[tuple[int, int]]]:
    """What via k sees of the members and the other vias, as a field of its
    own, each member and each via an obstacle, its edges' starts and ends,
    and each edge's member and first vertex, or -1 and the via; `unpaired`
    leaves out the vias that only stop it, of its net, and those the
    pairwise term takes instead, of other nets where either is a ground
    via; `other_nets` leaves out only the vias of its net."""
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
        if other_nets and other.via.net == via.given.via.net:
            continue
        starts.append(positions[u][None])
        ends.append(positions[u][None])
        inflations.append(np.array([apart]))
        owner.append((-1, u))
    a, b = np.concatenate(starts), np.concatenate(ends)
    obstacles = np.array(
        [j if j >= 0 else len(problem.bodies) + u for j, u in owner], dtype=np.intp
    ).reshape(-1)
    return ChainField(a, b, np.concatenate(inflations), obstacles), a, b, owner


def _reference_distances(
    problem: optimize_module._Problem,  # pyright: ignore[reportPrivateUsage]
    polylines: Sequence[FloatArray],
    positions: Sequence[FloatArray],
) -> tuple[list[float], dict[tuple[int, int], float]]:
    """`_Problem.via_distances` however far, each via's to its static copper
    and, by (via, obstacle), to each member and via of another net, with a
    field of each of those alone from the via's own view."""
    static = [
        point_distance(problem.vias[k].given.copper, positions[k])
        for k in range(len(problem.vias))
    ]
    seen: dict[tuple[int, int], float] = {}
    for k, via in enumerate(problem.vias):
        _, a, b, owner = _reference_view(
            problem, k, polylines, positions, other_nets=True
        )
        inflations = np.array(
            [via.rings[j] if j >= 0 else via.apart[u] for j, u in owner]
        )
        obstacles = np.array(
            [j if j >= 0 else len(problem.bodies) + u for j, u in owner],
            dtype=np.intp,
        )
        for obstacle in np.unique(obstacles):
            alone = obstacles == obstacle
            field = ChainField(a[alone], b[alone], inflations[alone], obstacles[alone])
            seen[k, int(obstacle)] = point_distance(field, positions[k])
    return static, seen


def _free_by_obstacle(
    free: FreeDistances,
) -> tuple[list[float], dict[tuple[int, int], float]]:
    """`free` as `_reference_distances` gives it."""
    pairs = free.by_obstacle
    return [float(d) for d in free.static], {
        (int(k), int(o)): float(d)
        for k, o, d in zip(pairs.via, pairs.obstacle, pairs.distance, strict=True)
    }


def _reference_vias(
    problem: optimize_module._Problem,  # pyright: ignore[reportPrivateUsage]
    polylines: Sequence[FloatArray],
    positions: Sequence[FloatArray],
) -> tuple[float, list[FloatArray], FloatArray]:
    """`_Problem.via_energy` with a field of each via's own view, its
    repelling term alone, the smooth minimum taken over it and the via's
    static field together. The scenarios' vias have no stops."""
    gradients = [np.zeros_like(points) for points in polylines]
    via_gradients = np.zeros((len(problem.vias), 2))
    value = 0.0
    for k in range(len(problem.vias)):
        via = problem.vias[k]
        assert via.given.stops is None
        at = positions[k][None]
        static, static_direction = via.given.field.smooth_distance(at, problem.reach)
        field, a, b, owner = _reference_view(
            problem, k, polylines, positions, unpaired=True
        )
        near = field.smooth_answers(at, problem.reach)
        distance, (static_share, seen_share) = combine_fields(static, near.distance)
        direction = (
            static_share[0] * static_direction[0] + seen_share[0] * (near.gradient[0])
        )
        term, slope = problem.integrand(distance)
        weight = via.given.via.diameter * slope[0]
        value += via.given.via.diameter * float(term[0])
        via_gradients[k] += weight * direction
        for e, share, toward in zip(near.edge, near.share, near.direction, strict=True):
            push = weight * seen_share[0] * share * toward
            j, first = owner[int(e)]
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
        listed = NeighbourList(seen, problem.reach)
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
            static, by_obstacle = _free_by_obstacle(
                problem.via_distances(polylines, positions, math.inf)
            )
            expected_static, expected_seen = _reference_distances(
                problem, polylines, positions
            )
            np.testing.assert_allclose(static, expected_static, rtol=0, atol=0)
            assert by_obstacle.keys() == expected_seen.keys()
            np.testing.assert_allclose(
                [by_obstacle[key] for key in expected_seen],
                list(expected_seen.values()),
                rtol=1e-12,
                atol=1e-12,
            )


def _exact_field(
    starts: FloatArray, ends: FloatArray, inflations: FloatArray
) -> ChainField:
    """Edges as a field for exact distances alone, which do not depend on
    what the obstacles are: each edge one."""
    return ChainField(
        starts, ends, inflations, np.arange(len(inflations), dtype=np.intp)
    )


def _reference_fields(
    problem: optimize_module._Problem,  # pyright: ignore[reportPrivateUsage]
    polylines: Sequence[FloatArray],
    positions: Sequence[FloatArray],
) -> list[ChainField]:
    """What each member sees of the others and the vias, as a field of its
    own pieces."""
    flat = shown_flat(problem.members, polylines, positions)
    fields: list[ChainField] = []
    for part, _ in _reference_parts(problem, polylines):
        a, b = flat[part.vertex], flat[part.vertex + 1]
        fields.append(
            _exact_field(
                a + part.start[:, None] * (b - a),
                a + part.end[:, None] * (b - a),
                part.inflation,
            )
        )
    return fields


def _reference_entered(
    problem: optimize_module._Problem,  # pyright: ignore[reportPrivateUsage]
    polylines: Sequence[FloatArray],
    flat: FloatArray,
) -> list[set[tuple[int, int]]]:
    """Per member, the (edge, vertex) pairs of each of its edges and the
    first vertex of each of its own pieces that the edge enters, judged
    for every edge and piece."""
    entered: list[set[tuple[int, int]]] = []
    for (part, _), points in zip(
        _reference_parts(problem, polylines), polylines, strict=True
    ):
        a, b = flat[part.vertex], flat[part.vertex + 1]
        starts = complex_points(a + part.start[:, None] * (b - a))
        along = complex_points(a + part.end[:, None] * (b - a)) - starts
        ca, cb = complex_points(points[:-1]), complex_points(points[1:])
        edge = np.repeat(np.arange(len(ca)), len(starts))
        piece = np.tile(np.arange(len(starts)), len(ca))
        inside = pairs_inside(
            ca[edge],
            (cb - ca)[edge],
            starts[piece],
            along[piece],
            part.inflation[piece],
            0.0,
        )
        entered.append(
            set(
                zip(
                    edge[inside].tolist(),
                    part.vertex[piece[inside]].tolist(),
                    strict=True,
                )
            )
        )
    return entered


@pytest.mark.parametrize("make", SCENARIOS, ids=IDS)
def test_the_checks_are_those_each_members_own_field_gives(
    make: Callable[[], Scenario],
) -> None:
    """What the guard, resampling and the edge checks judge, searched for
    every member at once, is what a field of each member's own pieces
    gives, with copper of one net left out: the distance from each member's
    points and from points around them, whether its edges stay clear, its
    violations with its static copper, its pressed vertices and its edges
    across the others, at the start and moved, a section's leave-out near
    its own nets' chains included."""
    scenario = make()
    rng = np.random.default_rng(0)
    problems = _problems(scenario, scenario.params)
    assert problems
    for problem in problems:
        layout = problem.layout(problem.points)
        for jitter in (0.0, JITTER):
            x = layout.x0 + jitter * rng.uniform(-1.0, 1.0, len(layout.x0))
            polylines, positions = layout.unflatten(x)
            seen = problem.seeing(polylines)
            flat = shown_flat(problem.members, polylines, positions)
            fields = _reference_fields(problem, polylines, positions)
            queries = [
                np.concatenate([points, points + rng.uniform(-0.5, 0.5, points.shape)])
                for points in polylines
            ]
            for got, field, at in zip(
                seen.distances(queries, flat), fields, queries, strict=True
            ):
                np.testing.assert_allclose(
                    got,
                    field.signed_distance(at)[0],
                    rtol=DISTANCE_ROUNDING,
                    atol=DISTANCE_ROUNDING,
                )
            clear = [
                field.segments_clear(points[:-1], points[1:])
                for field, points in zip(fields, polylines, strict=True)
            ]
            for got, expected, entered in zip(
                seen.segments_clear(polylines, flat),
                clear,
                _reference_entered(problem, polylines, flat),
                strict=True,
            ):
                np.testing.assert_array_equal(got.clear, expected)
                pairs = zip(got.edge.tolist(), got.vertex.tolist(), strict=True)
                assert set(pairs) == entered
            assert problem.edges_across(polylines, positions, seen) == [
                int(np.count_nonzero(~c)) for c in clear
            ]
            spacing = DEFAULT_SETTINGS.spacing
            for got, expected in zip(
                problem.start_statistics(polylines, positions),
                (
                    measure(points, CompositeField([body.field, field]), spacing)
                    for body, points, field in zip(
                        problem.bodies, polylines, fields, strict=True
                    )
                ),
                strict=True,
            ):
                assert (got.length, got.segment_count, got.tightest_bend_radius) == (
                    expected.length,
                    expected.segment_count,
                    expected.tightest_bend_radius,
                )
                assert (got.min_clearance, got.mean_clearance) == pytest.approx(
                    (expected.min_clearance, expected.mean_clearance),
                    rel=DISTANCE_ROUNDING,
                    abs=DISTANCE_ROUNDING,
                )
            masks = problem.masks
            assert problem.violations(polylines, masks, positions, seen) == [
                violations(points, mask, CompositeField([body.field, field]))
                for body, points, mask, field in zip(
                    problem.bodies, polylines, masks, fields, strict=True
                )
            ]
            for got, field, points in zip(
                problem.pressed(polylines, positions, seen),
                fields,
                polylines,
                strict=True,
            ):
                np.testing.assert_array_equal(
                    got, field.signed_distance(points)[0] < -PRESSED_DEPTH
                )


ONE_NET_CHAINS = 60
ONE_NET_VERTICES = 40
ONE_NET_PITCH = 0.3
"""How far apart the chains of one net lie, mm: each sees none of them."""
FAR_AWAY = 50.0
"""How far the chain of another net lies from them, mm."""
PAIRS_PER_POINT = 100
"""How many (point, edge) pairs per point the trees may give: a search that
looked at every edge of the one net would take thousands."""


def test_the_exact_distance_looks_only_at_what_each_member_sees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Many chains of one net and one of another net far from them: the
    exact distance from each chain is the far one's, or theirs for the far
    one, found without looking at the near chains of one net for each
    other's points."""
    xs = np.linspace(0.0, 12.0, ONE_NET_VERTICES)
    chains = [
        selected(f"g{k}", "GND", FreeSpace(), *((x, k * ONE_NET_PITCH) for x in xs))
        for k in range(ONE_NET_CHAINS)
    ]
    chains.append(selected("far", "X", FreeSpace(), *((x, -FAR_AWAY) for x in xs)))
    polylines = [chain.centreline.points for chain in chains]
    seen = seen_parts(chains, polylines, obstacle_reach(scenarios.PARAMS))
    looked = [0]
    near_segments = seen_module.near_segments
    each_within = obstacles_module._pairs_each_within  # pyright: ignore[reportPrivateUsage]

    def near(
        starts: FloatArray, ends: FloatArray, points: FloatArray, radius: float
    ) -> tuple[IntArray, IntArray]:
        found = near_segments(starts, ends, points, radius)
        looked[0] += len(found[0])
        return found

    def each(
        tree: cKDTree, centres: FloatArray, radii: FloatArray
    ) -> tuple[IntArray, IntArray]:
        found = each_within(tree, centres, radii)
        looked[0] += len(found[0])
        return found

    # Every pair the search looks at comes out of one of the two trees.
    monkeypatch.setattr(seen_module, "near_segments", near)
    monkeypatch.setattr(obstacles_module, "_pairs_each_within", each)
    got = seen.distances(polylines, np.concatenate(polylines))
    monkeypatch.undo()

    points = sum(len(line) for line in polylines)
    assert looked[0] <= PAIRS_PER_POINT * points
    for chain, distance in zip(chains, got, strict=True):
        others = [c for c in chains if c.centreline.net != chain.centreline.net]
        field = _exact_field(
            np.concatenate([c.centreline.points[:-1] for c in others]),
            np.concatenate([c.centreline.points[1:] for c in others]),
            np.full(sum(len(c.centreline.points) - 1 for c in others), PAIR_INFLATION),
        )
        np.testing.assert_allclose(
            distance,
            field.signed_distance(chain.centreline.points)[0],
            rtol=DISTANCE_ROUNDING,
            atol=DISTANCE_ROUNDING,
        )


NUMBERED_INFLATION = 0.3
"""How far every two tracks of the numbered nets keep apart, mm."""
TRACK_NETS = (0, 1, 3, 4, 6, 9)
"""The numbered nets' tracks, one each, a row apart: net 9's is the chain's
own, which every member leaves out."""
ROW = 2.0
"""How far apart the tracks lie, mm: farther than twice the inflation, so
that each point's nearest piece is found beyond the first search. A point
1 mm past the end of net 1's track has it nearest."""


EXCEPTION_RING = 0.5
"""How far a chain keeps from the via it sees, mm: the widest inflation."""
SPAN_INFLATION = 0.4
"""How far a section keeps from the stretch of its own net's track it sees."""


def test_the_exact_distance_finds_exceptions_beyond_the_first_search() -> None:
    """A chain whose nearest piece is a via, and a section whose nearest is
    a stretch of a track of its own net, both much farther than twice the
    widest inflation: only each member's own exceptions answer them, the
    stretch as itself rather than its whole edge."""
    flat = np.array(
        [[0.0, 0.0], [5.0, 0.0], [0.0, 8.0], [10.0, 8.0], [20.0, 0.0], [20.0, 0.0]]
    )
    tracks = [Track(0, 2, "F.Cu", 0, 0), Track(2, 2, "F.Cu", 1, 0)]
    viewers = [Viewer("F.Cu", 0, 0, ()), Viewer("F.Cu", 5, 0, (1, 2))]
    via = Pieces(np.array([4]), np.zeros(1), np.ones(1), np.array([EXCEPTION_RING]))
    stretch = Pieces(
        np.array([2]), np.array([0.25]), np.array([0.5]), np.array([SPAN_INFLATION])
    )
    seen = Seen(tracks, viewers, np.array([[NUMBERED_INFLATION]]), [via, stretch])
    queries = [np.array([[30.0, 0.0]]), np.array([[9.0, 20.0]])]
    fields = [
        _exact_field(
            flat[[2, 4]],
            flat[[3, 5]],
            np.array([NUMBERED_INFLATION, EXCEPTION_RING]),
        ),
        _exact_field(
            flat[[0, 2]] + np.array([[0.0, 0.0], [2.5, 0.0]]),
            np.array([[5.0, 0.0], [5.0, 8.0]]),
            np.array([NUMBERED_INFLATION, SPAN_INFLATION]),
        ),
    ]

    got = seen.distances(queries, flat)

    widest = max(EXCEPTION_RING, SPAN_INFLATION)
    for distance, field, at in zip(got, fields, queries, strict=True):
        expected = field.signed_distance(at)[0]
        assert np.all(expected > 2 * widest)
        np.testing.assert_allclose(
            distance, expected, rtol=DISTANCE_ROUNDING, atol=DISTANCE_ROUNDING
        )


def test_the_exact_distance_asks_every_net_a_chain_sees() -> None:
    """A chain of net 9 beside a section that leaves out nets 6 and 9, so
    that no edge of net 9 is among the layer's, and the largest net among
    them is 6: in the three bits that numbers, net 9 agrees with net 1, and
    the chain must still see net 1. A second section leaves out nets 3 and
    9. Each member's exact distance is what a field of the tracks it sees
    gives."""
    tracks: list[Track] = []
    rows: list[FloatArray] = []
    for k, net in enumerate(TRACK_NETS):
        y = k * ROW
        rows.append(np.array([[0.0, y], [5.0, y]]))
        tracks.append(Track(2 * k, 2, "F.Cu", net, 0))
    flat = np.concatenate(rows)
    viewers = [
        Viewer("F.Cu", 9, 0, ()),
        Viewer("F.Cu", 8, 0, (6, 9)),
        Viewer("F.Cu", 10, 0, (3, 9)),
    ]
    none = Pieces(np.empty(0, dtype=np.intp), np.empty(0), np.empty(0), np.empty(0))
    seen = Seen(tracks, viewers, np.array([[NUMBERED_INFLATION]]), [none] * 3)
    queries = [np.array([[2.5, -1.0], [6.0, ROW], [2.5, 20.0]]) for _ in viewers]

    for viewer, at, got in zip(
        viewers, queries, seen.distances(queries, flat), strict=True
    ):
        left_out = {viewer.net, *viewer.exempt}
        shown = [k for k, t in enumerate(tracks) if t.net not in left_out]
        field = _exact_field(
            flat[[2 * k for k in shown]],
            flat[[2 * k + 1 for k in shown]],
            np.full(len(shown), NUMBERED_INFLATION),
        )
        np.testing.assert_allclose(
            got,
            field.signed_distance(at)[0],
            rtol=DISTANCE_ROUNDING,
            atol=DISTANCE_ROUNDING,
        )


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


ZONED_VIA_SCENARIOS: list[tuple[Callable[[], ViasScenario], str]] = [
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


def _reference_free_distance(
    problem: optimize_module._Problem,  # pyright: ignore[reportPrivateUsage]
    k: int,
    polylines: Sequence[FloatArray],
    positions: Sequence[FloatArray],
) -> float:
    """Via k's free distance from fields of its own: its static copper, the
    members of other nets on each layer it spans with its zones there, and
    the vias of other nets at the larger of both vias' zone distances."""
    given = problem.vias[k].given
    via = given.via
    clearances = {b.centreline.net: b.centreline.clearance for b in problem.bodies}
    clearances |= {v.given.via.net: v.given.clearance for v in problem.vias}
    clearance_of = clearances.__getitem__
    distance = point_distance(given.copper, positions[k])
    for layer in dict.fromkeys(body.layer for body in problem.bodies):
        near = [
            replace(body.centreline, points=points)
            for body, points in zip(problem.bodies, polylines, strict=True)
            if body.layer == layer
            and via.reaches(layer)
            and body.centreline.net != via.net
        ]
        if near:
            field = chain_obstacles_for_via(
                via,
                near,
                clearance_of,
                hole_clearance=HOLE_CLEARANCE,
                zones=zones_on(given.zones, [layer]),
            )
            distance = min(distance, point_distance(field, positions[k]))
    for u, other in enumerate(problem.vias):
        kept = (
            None
            if other.given.via.net == via.net
            else via_via_inflation(
                via,
                other.given.via,
                clearance_of,
                hole_clearance=HOLE_CLEARANCE,
                hole_to_hole=HOLE_TO_HOLE,
                zones=given.zones,
                other_zones=other.given.zones,
            )
        )
        if kept is not None:
            distance = min(distance, math.dist(positions[k], positions[u]) - kept)
    return distance


@pytest.mark.parametrize(
    "make",
    [make for make, _ in ZONED_VIA_SCENARIOS],
    ids=[name for _, name in ZONED_VIA_SCENARIOS],
)
def test_the_free_distance_of_zoned_vias_is_their_own_fields(
    make: Callable[[], ViasScenario],
) -> None:
    """The free distance the guard judges vias by (`_Problem.via_distances`)
    is what fields of each via's own view give, at the start and moved, with
    each via's zones on the layers they lie on and the larger of two vias'
    zone distances between them."""
    scenario = make()
    rng = np.random.default_rng(0)
    problems = _problems(scenario, scenario.params)
    assert problems
    for problem in problems:
        layout = problem.layout(problem.points)
        for jitter in (0.0, JITTER):
            x = layout.x0 + jitter * rng.uniform(-1.0, 1.0, len(layout.x0))
            polylines, positions = layout.unflatten(x)
            static, seen = _free_by_obstacle(
                problem.via_distances(polylines, positions, math.inf)
            )
            nearest = [
                min([d, *(s for (k, _), s in seen.items() if k == via)])
                for via, d in enumerate(static)
            ]
            np.testing.assert_allclose(
                nearest,
                [
                    _reference_free_distance(problem, k, polylines, positions)
                    for k in range(len(problem.vias))
                ],
                rtol=1e-12,
                atol=1e-12,
            )
