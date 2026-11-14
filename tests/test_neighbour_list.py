from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
import scenarios
from conftest import SUM_ORDER_ROUNDING
from distance_fields import FloatArray, FreeSpace
from scenarios import ChainsScenario, PairsScenario, ViasScenario

import kibends.optimize as optimize_module
from kibends import _core
from kibends.bursts import DEFAULT_SETTINGS
from kibends.energy import SMOOTH_REACH, quadrature_points
from kibends.members import seen_parts
from kibends.neighbours import ChainInput
from kibends.obstacles import HOLE_CLEARANCE, HOLE_TO_HOLE
from kibends.pair_blocks import pair_block
from kibends.seen import SKIN_LEAST, SKIN_MOST, NeighbourList, SmoothAnswer
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
    make: Callable[[], Scenario], repulsion: float
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
        chains, blocks, vias, params, DEFAULT_SETTINGS
    )
    problem = placed.problem
    layout = problem.layout(problem.points)
    seen = seen_parts(problem.members, layout.starts, problem.rings)
    searched = problem.objective(layout, seen, None)
    listed = problem.objective(layout, seen, problem.lookups(seen))
    searches = _searches()
    rng = np.random.default_rng(0)
    x = layout.x0.copy()
    steps = [0.0, 0.2, 0.2, 0.2, 2.0, 0.2, 0.2, 6.0, 0.2]
    for step in steps:
        x = x + step * SKIN_LEAST * rng.uniform(-1.0, 1.0, len(x)) / np.sqrt(2)
        value, gradient = searched(x)
        listed_value, listed_gradient = listed(x)
        assert listed_value == pytest.approx(
            value, rel=SUM_ORDER_ROUNDING, abs=SUM_ORDER_ROUNDING
        )
        np.testing.assert_allclose(
            listed_gradient, gradient, rtol=SUM_ORDER_ROUNDING, atol=SUM_ORDER_ROUNDING
        )
    # Else every step searched afresh and the list answered nothing itself.
    assert searches() < len(steps) * seen.layer_count


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


def _searches() -> Callable[[], int]:
    """How many times a layer has found its pairs since the call
    (`kibends._core.statistics`)."""
    start = _core.statistics().list_searches
    return lambda: _core.statistics().list_searches - start


def _assert_listed_is_searched(
    chains: list[ChainInput],
    steps: list[list[float]],
    searches_after: list[int] | None = None,
) -> None:
    """With each chain moved along y by its move in each of `steps` in turn,
    the list made at the start answers every point as a search does.

    `searches_after`, where given, is how many times it has searched after
    each step, once at the start included: the premise that the steps it
    did not search at were answered from the pairs it found before, where a
    search would answer rightly whatever the list kept."""
    searches = _searches()
    polylines = [chain.centreline.points for chain in chains]
    seen = seen_parts(chains, polylines)
    listed = NeighbourList(seen, REACH)

    def answers(
        shapes: list[FloatArray], neighbours: NeighbourList | None
    ) -> list[SmoothAnswer]:
        queries = [quadrature_points(points, 3) for points in shapes]
        flat = np.concatenate(shapes)
        return seen.smooth_answers(queries, flat, REACH, neighbours)

    answers(polylines, listed)
    counts = searches_after or [None] * len(steps)
    for moves, count in zip(steps, counts, strict=True):
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
        assert count is None or searches() == count


def test_an_edge_listed_beyond_reach_answers_once_it_comes_within_it() -> None:
    """A chain beyond reach by less than twice the skin, and another moving
    toward it by almost a skin each, which brings it within reach."""
    far = WITHIN + 1.8 * SKIN_LEAST
    chains = [_level("A", 0.0), _level("B", far)]
    _assert_listed_is_searched(chains, [[APPROACH, -APPROACH]], [1])
    assert far - 2 * APPROACH < WITHIN


def test_an_edge_listed_behind_the_nearest_answers_once_it_comes_nearer() -> None:
    """A chain between two others, one surely within reach and the other
    farther than SMOOTH_REACH behind it, by less than twice the skin,
    moving toward the farther one by almost a skin as it moves closer by as
    much and the nearer one away."""
    near, far = 0.6, 0.6 + SMOOTH_REACH + 1.8 * SKIN_LEAST
    chains = [_level("A", 0.0), _level("B", near), _level("C", -far)]
    _assert_listed_is_searched(chains, [[-APPROACH, APPROACH, APPROACH]], [1])
    assert far - 2 * APPROACH < near + 2 * APPROACH


def test_a_list_that_searched_again_keeps_its_pairs_through_as_far_a_move() -> None:
    """A chain moving SKIN_MOST toward another, which makes the list search
    again with it beyond reach by less than twice that, then both moving
    toward each other by almost as much, which brings it within reach
    without another search."""
    far = WITHIN + 2.8 * SKIN_MOST
    chains = [_level("A", 0.0), _level("B", far)]
    closer = 0.95 * SKIN_MOST
    _assert_listed_is_searched(chains, [[0.0, -SKIN_MOST], [closer, -closer]], [2, 2])
    assert far - SKIN_MOST - 2 * closer < WITHIN


def test_a_list_that_searched_after_a_far_move_answers_moves_past_its_skin() -> None:
    """A chain moving far beyond SKIN_MOST toward another, to beyond the
    reach it would list with twice that, then both moving toward each other
    by almost SKIN_MOST, and then by enough more to bring it within reach:
    a list that kept its pairs past SKIN_MOST would miss it."""
    after_far = WITHIN + 2.05 * SKIN_MOST
    chains = [_level("A", 0.0), _level("B", after_far + FAR_MOVE)]
    closer, past = 0.95 * SKIN_MOST, 0.2 * SKIN_MOST
    steps = [[0.0, -FAR_MOVE], [closer, -closer], [past, -past]]
    _assert_listed_is_searched(chains, steps)
    assert after_far - 2 * closer > WITHIN
    assert after_far - 2 * (closer + past) < WITHIN
