from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
import scenarios
from distance_fields import FloatArray
from scenarios import ChainsScenario, PairsScenario, ViasScenario

import kibends.optimize as optimize_module
from kibends.bursts import DEFAULT_SETTINGS
from kibends.obstacles import HOLE_CLEARANCE, HOLE_TO_HOLE, ChainField
from kibends.optimize import LIST_SKIN, seen_parts
from kibends.pair_blocks import pair_block
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
    real = ChainField.candidates

    def counted(
        self: ChainField, points: FloatArray, reach: float, slack: float
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal searches
        searches += 1
        return real(self, points, reach, slack)

    monkeypatch.setattr(ChainField, "candidates", counted)
    rng = np.random.default_rng(0)
    x = layout.x0.copy()
    steps = [0.0, 0.2, 0.2, 0.2, 2.0, 0.2, 0.2, 6.0, 0.2]
    for step in steps:
        x = x + step * LIST_SKIN * rng.uniform(-1.0, 1.0, len(x)) / np.sqrt(2)
        value, gradient = searched(x)
        listed_value, listed_gradient = listed(x)
        assert listed_value == pytest.approx(value, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(listed_gradient, gradient, rtol=1e-10, atol=1e-12)
    members = len(problem.members)
    # Random steps swing a section's mitred tracks farther than its centreline,
    # so only plain chains keep to their pairs between the large steps.
    assert searches < len(steps) * members
