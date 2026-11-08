import numpy as np
import pytest
import scenarios
from distance_fields import BoolArray, FloatArray

from kibends.bursts import OptimizerSettings
from kibends.optimize import resample
from kibends.pair_blocks import PairBlock, pair_block

COARSE = OptimizerSettings(spacing=0.3, settled_change=0.003, burst_maxiter=30)

STEP = 1e-7
WOBBLE = 0.05
"""mm by which free centreline vertices are moved off their resampled
places, so that every vertex turns."""


def wobbled(block: PairBlock, seed: int) -> tuple[FloatArray, BoolArray]:
    points, fixed = resample(
        block.chain.centreline.points, block.chain.fixed, COARSE.spacing
    )
    rng = np.random.default_rng(seed)
    points = points.copy()
    points[~fixed] += rng.uniform(-WOBBLE, WOBBLE, (int((~fixed).sum()), 2))
    return points, fixed


@pytest.mark.parametrize(
    "scenario",
    [scenarios.pair_uneven_end(), scenarios.pair_corner()],
    ids=lambda s: s.name,
)
def test_offset_pullback_matches_finite_differences(
    scenario: scenarios.PairsScenario,
) -> None:
    (given,) = scenario.pairs
    block = pair_block(given, COARSE.spacing)
    points, fixed = wobbled(block, 1)
    rng = np.random.default_rng(2)
    at_a, at_b = rng.normal(size=points.shape), rng.normal(size=points.shape)

    def value(centreline: FloatArray) -> float:
        a, b = block.offset(centreline)
        return float(np.sum(at_a * a) + np.sum(at_b * b))

    analytic = block.offset_pullback(points, at_a, at_b)

    # A gap's taper starts at its end's direction pin: at the pinned end
    # edges the gaps have a kink, which no free vertex reaches.
    free = np.flatnonzero(~fixed)
    numeric = np.zeros_like(points)
    for k in free:
        for axis in (0, 1):
            up, down = points.copy(), points.copy()
            up[k, axis] += STEP
            down[k, axis] -= STEP
            numeric[k, axis] = (value(up) - value(down)) / (2 * STEP)
    np.testing.assert_allclose(analytic[free], numeric[free], rtol=1e-5, atol=1e-6)
