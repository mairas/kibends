"""Fields built from chains' centrelines by the optimizer's inflation
rules, for tests that check those rules against the board's obstacles or
use them as a reference."""

from collections.abc import Sequence

from kibends.obstacles import ChainCentreline, ChainField, chain_inflation, chains_field


def chain_obstacles(
    net: str, width: float, clearance: float, others: Sequence[ChainCentreline]
) -> ChainField:
    """`others` as obstacles to a track of `net`, `width` and netclass
    `clearance` (mm), each inflated as `chain_inflation` gives it."""
    return chains_field((o, chain_inflation(net, width, clearance, o)) for o in others)
