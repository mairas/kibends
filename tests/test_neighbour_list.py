from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
import scenarios
from distance_fields import FloatArray, FreeSpace
from scenarios import ChainsScenario, PairsScenario, ViasScenario

import kibends.optimize as optimize_module
from kibends.bursts import DEFAULT_SETTINGS
from kibends.energy import quadrature_points
from kibends.members import seen_parts
from kibends.neighbours import ChainInput
from kibends.obstacles import HOLE_CLEARANCE, HOLE_TO_HOLE
from kibends.optimize import LIST_SKIN
from kibends.pair_blocks import pair_block
from kibends.seen import NeighbourList, Pairs, Seen
from kibends.via_moves import MovingVias

Scenario = ChainsScenario | PairsScenario | ViasScenario

SCENARIOS: list[Callable[[], Scenario]] = [
    scenarios.channel_pair,
    scenarios.same_net_wrap,
    scenarios.crowded,
    scenarios.pair_tent,
    scenarios.pair_split,
    scenarios.via_beside_chain,
    scenarios.via_pair,
]


@pytest.mark.parametrize("repulsion", [0.0, 0.5])
@pytest.mark.parametrize("make", SCENARIOS, ids=lambda make: make.__name__)
def test_the_listed_energy_is_the_searched_one(
    make: Callable[[], Scenario], repulsion: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through moves within the skin, which keep to the pairs found, and
    beyond it, which find them again, the energy and its gradient are the
    same with the neighbour list as with a search over every piece."""
    scenario = make()
    params = replace(scenario.params, repulsion=repulsion, falloff_range=1.0)
    chains = list(scenario.chains)
    pairs = list(getattr(scenario, "pairs", ()))
    blocks = [pair_block(pair, DEFAULT_SETTINGS.spacing) for pair in pairs]
    everything = [*chains, *(block.chain for block in blocks)]
    vias = MovingVias(
        everything, list(getattr(scenario, "vias", ())), HOLE_CLEARANCE, HOLE_TO_HOLE
    )
    (placed,) = optimize_module._problems(  # pyright: ignore[reportPrivateUsage]
        chains, blocks, vias, params, DEFAULT_SETTINGS, None
    )
    problem = placed.problem
    layout = problem.layout(problem.points)
    seen = seen_parts(problem.members, layout.starts, problem.reach, problem.rings)
    searched = problem.objective(layout, seen, listed=False)
    listed = problem.objective(layout, seen)
    searches = 0
    real = Seen.candidates

    def counted(
        self: Seen,
        n: int,
        points: FloatArray,
        owner: np.ndarray,
        flat: FloatArray,
        reach: float,
        slack: float,
    ) -> Pairs:
        nonlocal searches
        searches += 1
        return real(self, n, points, owner, flat, reach, slack)

    monkeypatch.setattr(Seen, "candidates", counted)
    rng = np.random.default_rng(0)
    x = layout.x0.copy()
    steps = [0.0, 0.2, 0.2, 0.2, 2.0, 0.2, 0.2, 6.0, 0.2]
    for step in steps:
        x = x + step * LIST_SKIN * rng.uniform(-1.0, 1.0, len(x)) / np.sqrt(2)
        value, gradient = searched(x)
        listed_value, listed_gradient = listed(x)
        assert listed_value == pytest.approx(value, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(listed_gradient, gradient, rtol=1e-10, atol=1e-12)
    layers = len(seen.layers)
    # The first evaluation and the two steps past the skin search each layer.
    # Random steps swing a section's mitred tracks farther than its
    # centreline, so a layer with a section searches more often, though
    # not at every step.
    if pairs:
        assert searches < len(steps) * layers
    else:
        assert searches == 3 * layers


REACH = 1.0
WITHIN = REACH + scenarios.PAIR_INFLATION
"""How far from another net's centreline a point is still within reach."""
APPROACH = 0.95 * LIST_SKIN
"""How far a point and an edge each move toward each other, within the
skin, so that the list does not search again."""


def _level(net: str, y: float) -> ChainInput:
    """A chain of `net` along y, pinned at its ends."""
    return scenarios.selected(
        net, net, FreeSpace(), *((x, y) for x in (0.0, 1.0, 2.0, 3.0, 4.0))
    )


def _counted(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """How many times a layer finds its pairs, as a one-element list."""
    searches = [0]
    real = Seen.candidates

    def counted(
        self: Seen,
        n: int,
        points: FloatArray,
        owner: np.ndarray,
        flat: FloatArray,
        reach: float,
        slack: float,
    ) -> Pairs:
        searches[0] += 1
        return real(self, n, points, owner, flat, reach, slack)

    monkeypatch.setattr(Seen, "candidates", counted)
    return searches


def _assert_listed_is_searched(
    chains: list[ChainInput], moves: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With each chain moved along y by its move, the list made at the
    start, without searching again, answers every point as a search does."""
    searches = _counted(monkeypatch)
    starts = [chain.centreline.points for chain in chains]
    seen = seen_parts(chains, starts, REACH)
    listed = NeighbourList(seen, REACH, LIST_SKIN)

    def answers(
        polylines: list[FloatArray], neighbours: NeighbourList | None
    ) -> list[tuple[FloatArray, np.ndarray]]:
        queries = [quadrature_points(points, 3) for points in polylines]
        flat = np.concatenate(polylines)
        return [
            (near.distance, near.edge)
            for near in seen.nearest(queries, flat, REACH, neighbours)
        ]

    answers(starts, listed)
    moved = [
        points + np.array([0.0, move])
        for points, move in zip(starts, moves, strict=True)
    ]
    found = answers(moved, listed)
    assert searches[0] == 1
    for (distance, edge), (expected, expected_edge) in zip(
        found, answers(moved, None), strict=True
    ):
        np.testing.assert_allclose(distance, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(edge, expected_edge)


def test_an_edge_listed_beyond_reach_answers_once_it_comes_within_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chain beyond reach by less than twice the skin, and another moving
    toward it by almost a skin each, which brings it within reach."""
    far = WITHIN + 1.8 * LIST_SKIN
    chains = [_level("A", 0.0), _level("B", far)]
    _assert_listed_is_searched(chains, [APPROACH, -APPROACH], monkeypatch)
    assert far - 2 * APPROACH < WITHIN


def test_an_edge_listed_behind_the_nearest_answers_once_it_comes_nearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chain between two others, one surely within reach and the other
    farther by less than twice the skin, moving toward the farther one by
    almost a skin as it moves closer by as much and the nearer one away."""
    near, far = 0.6, 0.6 + 3 * LIST_SKIN
    chains = [_level("A", 0.0), _level("B", near), _level("C", -far)]
    _assert_listed_is_searched(chains, [-APPROACH, APPROACH, APPROACH], monkeypatch)
    assert far - 2 * APPROACH < near + 2 * APPROACH
