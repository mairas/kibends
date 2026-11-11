"""Selected chains as obstacles to each other.

Each chain has a static field of everything on the board but the selected
chains. The other selected chains on its layer are added to it as obstacles
in one field (`neighbour_field`), at whatever shape the caller holds them
in; simplification gives it only those near what it asks about
(`simplify`). The optimizer asks the same of all its members at once
instead (`members.seen_parts`), with no field per member. A chain ignores
the chains of its net and of the other net of its diff pair
(`pairs.diff_pair_nets`), and a coupled section those of its two nets:
they may come close, overlap or cross. KiCad's DRC reports what that lets
through between the two nets of a diff pair.
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace

from kibends.energy import BoolArray, DistanceField, FloatArray, ObstacleField
from kibends.obstacles import (
    ChainCentreline,
    CompositeField,
    chain_inflation,
    chains_field,
)
from kibends.pairs import diff_pair_nets


@dataclass(frozen=True, eq=False)
class ChainInput:
    """One chain of a run, with the field of everything but the selected chains."""

    centreline: ChainCentreline
    """Its description, net, width, clearance and starting polyline."""
    layer: str
    """Only chains on the same layer see each other."""
    fixed: BoolArray
    field: ObstacleField
    """Obstacles for its net and width, without any selected chain."""
    contacts: tuple[tuple[int, DistanceField], ...] = ()
    """The vertices that keep a pad or via, by index, each with the outside
    of the region it may lie in, held as the board edge is: positive inside.
    Such a vertex is free, but a stop keeps it inside and a burst never
    leaves it outside. A fixed vertex keeps its contact by staying."""
    held: bool = False
    """It stays as drawn: every vertex is fixed, and none is added between
    them, as resampling a chain does."""


def chain_field(
    static: ObstacleField, neighbours: ObstacleField | None
) -> ObstacleField:
    """A chain's field: `static` alone, or with `neighbours` if not None."""
    return static if neighbours is None else CompositeField([static, neighbours])


def neighbour_field(
    own: ChainInput,
    shown: Iterable[tuple[ChainInput, FloatArray]],
    *,
    ignored: frozenset[str] | None = None,
) -> ObstacleField | None:
    """The chains of `shown`, each at its points and an obstacle of its own,
    as obstacles to `own` in one field, inflated as `obstacles.chain_inflation`
    gives it; None if nothing is left. Chains of the nets `ignored`, by
    default `own`'s and the other net of its diff pair, are no obstacles."""
    me = own.centreline
    nets = diff_pair_nets(me.net) if ignored is None else ignored
    parts: list[tuple[list[ChainCentreline], float]] = []
    for other, points in shown:
        them = other.centreline
        inflation = chain_inflation(nets, me.width, me.clearance, them)
        if inflation is not None:
            parts.append(([replace(them, points=points)], inflation))
    if not parts:
        return None
    return chains_field(parts)
