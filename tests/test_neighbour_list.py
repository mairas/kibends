from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
import scenarios
from distance_fields import FloatArray, FreeSpace
from scenarios import ChainsScenario, PairsScenario, ViasScenario

import kibends.optimize as optimize_module
from kibends.bursts import DEFAULT_SETTINGS
from kibends.energy import SMOOTH_REACH, quadrature_points
from kibends.members import seen_parts
from kibends.neighbours import ChainInput
from kibends.obstacles import HOLE_CLEARANCE, HOLE_TO_HOLE, SmoothAnswer
from kibends.pair_blocks import pair_block
from kibends.seen import SKIN_LEAST, SKIN_MOST, NeighbourList, Pairs, Seen
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
        x = x + step * SKIN_LEAST * rng.uniform(-1.0, 1.0, len(x)) / np.sqrt(2)
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
WITHIN = REACH + SMOOTH_REACH + scenarios.PAIR_INFLATION
"""How far from another net's centreline a point is still within reach of
the smooth minimum."""
APPROACH = 0.95 * SKIN_LEAST
"""How far a point and an edge each move toward each other, within the
skin a list starts with, so that it does not search again."""
FAR_MOVE = 1.0
"""A move far beyond any skin, mm."""


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
    chains: list[ChainInput],
    steps: list[list[float]],
    searches_after: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With each chain moved along y by its move in each of `steps` in turn,
    the list made at the start has searched as many times as
    `searches_after` gives after each step, once at the start included, and
    answers every point as a search does."""
    searches = _counted(monkeypatch)
    polylines = [chain.centreline.points for chain in chains]
    seen = seen_parts(chains, polylines, REACH)
    listed = NeighbourList(seen, REACH)

    def answers(
        shapes: list[FloatArray], neighbours: NeighbourList | None
    ) -> list[SmoothAnswer]:
        queries = [quadrature_points(points, 3) for points in shapes]
        flat = np.concatenate(shapes)
        return seen.smooth_answers(queries, flat, REACH, neighbours)

    answers(polylines, listed)
    for moves, count in zip(steps, searches_after, strict=True):
        polylines = [
            points + np.array([0.0, move])
            for points, move in zip(polylines, moves, strict=True)
        ]
        found = answers(polylines, listed)
        for got, expected in zip(found, answers(polylines, None), strict=True):
            np.testing.assert_array_equal(got.distance, expected.distance)
            np.testing.assert_array_equal(got.point, expected.point)
            np.testing.assert_array_equal(got.edge, expected.edge)
            np.testing.assert_array_equal(got.share, expected.share)
        assert searches[0] == count


def test_an_edge_listed_beyond_reach_answers_once_it_comes_within_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chain beyond reach by less than twice the skin, and another moving
    toward it by almost a skin each, which brings it within reach."""
    far = WITHIN + 1.8 * SKIN_LEAST
    chains = [_level("A", 0.0), _level("B", far)]
    _assert_listed_is_searched(chains, [[APPROACH, -APPROACH]], [1], monkeypatch)
    assert far - 2 * APPROACH < WITHIN


def test_an_edge_listed_behind_the_nearest_answers_once_it_comes_nearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chain between two others, one surely within reach and the other
    farther than SMOOTH_REACH behind it, by less than twice the skin,
    moving toward the farther one by almost a skin as it moves closer by as
    much and the nearer one away."""
    near, far = 0.6, 0.6 + SMOOTH_REACH + 1.8 * SKIN_LEAST
    chains = [_level("A", 0.0), _level("B", near), _level("C", -far)]
    _assert_listed_is_searched(
        chains, [[-APPROACH, APPROACH, APPROACH]], [1], monkeypatch
    )
    assert far - 2 * APPROACH < near + 2 * APPROACH


def test_a_list_that_searched_again_keeps_its_pairs_through_as_far_a_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chain moving SKIN_MOST toward another, which makes the list search
    again with it beyond reach by less than twice that, then both moving
    toward each other by almost as much, which brings it within reach
    without another search."""
    far = WITHIN + 2.8 * SKIN_MOST
    chains = [_level("A", 0.0), _level("B", far)]
    closer = 0.95 * SKIN_MOST
    _assert_listed_is_searched(
        chains, [[0.0, -SKIN_MOST], [closer, -closer]], [2, 2], monkeypatch
    )
    assert far - SKIN_MOST - 2 * closer < WITHIN


def test_a_list_widens_its_skin_to_SKIN_MOST_at_most(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a move far beyond SKIN_MOST, which makes the list search
    again, a move of almost SKIN_MOST keeps to the pairs found and one a
    little farther makes it search once more."""
    chains = [_level("A", 0.0), _level("B", 0.6)]
    steps = [[0.0, FAR_MOVE], [0.0, 0.95 * SKIN_MOST], [0.0, 0.1 * SKIN_MOST]]
    _assert_listed_is_searched(chains, steps, [2, 2, 3], monkeypatch)
