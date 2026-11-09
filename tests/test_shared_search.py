"""The energy and the vias' free distances found through search trees
shared between members, against a reference that builds a field of its own
for each member and each via.

Where two pieces answer a point at exactly the same distance, which one
does depends on the order a tree lists them in, so the gradient may go to
either: the states here have no such point. A grid of routes whose vias
stand in line has them at the start, where a member's end lies midway
between the vias above and below it."""

from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
import pytest
import scenarios
from distance_fields import FloatArray, FreeSpace
from scenarios import ChainsScenario, PairsScenario, ViasScenario

import kibends.optimize as optimize_module
from kibends.bursts import DEFAULT_SETTINGS
from kibends.energy import (
    BoolArray,
    EnergyParams,
    bend_radius_energy,
    edge_integral,
    obstacle_integrand,
    obstacle_reach,
    quadrature_points,
    shape_energy,
)
from kibends.obstacles import HOLE_CLEARANCE, HOLE_TO_HOLE, ChainField
from kibends.optimize import (
    LIST_SKIN,
    Member,
    NeighbourList,
    SeenParts,
    members_energy,
    seen_parts,
    shown,
)
from kibends.pair_blocks import PairBlock, pair_block
from kibends.via_moves import MovingVias

Scenario = ChainsScenario | PairsScenario | ViasScenario

SCENARIOS: list[Callable[[], Scenario]] = [
    scenarios.centreline_over_pad,
    scenarios.channel_pair,
    scenarios.crowded,
    scenarios.drape,
    scenarios.hooks,
    scenarios.neck_down,
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
    scenarios.via_over_bowed_same_net,
    scenarios.via_over_same_net,
    scenarios.via_pair,
    lambda: scenarios.via_detour_over(FreeSpace()),
    lambda: scenarios.via_slack(FreeSpace()),
    lambda: scenarios.via_grid(3, 3),
]
IDS = [
    *(make.__name__ for make in SCENARIOS[:-3]),
    "via_detour_over",
    "via_slack",
    "via_grid",
]

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
) -> tuple[ChainField, FloatArray, FloatArray, list[tuple[int, int]]]:
    """What via k sees of the members and the other vias, as a field of its
    own, its edges' starts and ends, and each edge's member and first
    vertex, or -1 and the via."""
    via = problem.vias[k]
    starts = [np.empty((0, 2))]
    ends = [np.empty((0, 2))]
    inflations = [np.empty(0)]
    owner: list[tuple[int, int]] = []
    for j, ring in via.rings.items():
        points = polylines[j]
        starts.append(points[:-1])
        ends.append(points[1:])
        inflations.append(np.full(len(points) - 1, ring))
        owner += [(j, e) for e in range(len(points) - 1)]
    for u, apart in via.apart.items():
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
        field, a, b, owner = _reference_view(problem, k, polylines, positions)
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
    alone give."""
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
            expected = _reference_members(*arguments)
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
