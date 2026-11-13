"""Selected chains as obstacles to each other.

Each chain has a static field of everything on the board but the selected
chains. The other selected chains on its layer are its obstacles too, at
whatever shape the caller holds them in: the optimizer asks what each
member sees of the others at once (`members.seen_parts`, through
`kibends_core::members`), and simplification builds each member's field
from the same rule (`kibends_core::simplify`). A chain ignores
the chains of its net and of the other net of its diff pair
(`pairs.diff_pair_nets`), and a coupled section those of its two nets:
they may come close, overlap or cross. KiCad's DRC reports what that lets
through between the two nets of a diff pair.
"""

from dataclasses import dataclass

from kibends.energy import BoolArray, DistanceField, ObstacleField
from kibends.obstacles import ChainCentreline


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
    them, as laying a chain out does."""
