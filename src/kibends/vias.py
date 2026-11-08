"""Decide which selected vias may move with the chains they join.

A selected via moves only when it is an unlocked through via, every copper
item touching it on any layer is the end of a selected chain, and at least
two such ends meet it. Every other selected via stays fixed, with a warning
naming the reason, and pins the chains it touches like an unselected via.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from loguru import logger

from kibends.chain import Chain
from kibends.energy import ObstacleField
from kibends.snapshot import Via, ViaContact, ViaKind

MIN_JOINED_ENDS = 2


class ChainEnd(NamedTuple):
    chain: int
    """Index into the chains given to `movable_vias`."""
    last: bool
    """Whether it is the chain's last point rather than its first."""


@dataclass(frozen=True)
class MovableVia:
    via: Via
    ends: tuple[ChainEnd, ...]
    """The chain ends whose copper overlaps the via, in chain order."""


@dataclass(frozen=True, eq=False)
class ViaInput:
    """A moving via of a run, with the field of everything that stands still."""

    via: Via
    ends: tuple[ChainEnd, ...]
    """The chain ends that move with it, each keeping its offset from the
    via's centre."""
    field: ObstacleField
    """Obstacles for the via over every layer it spans, as
    `build_via_obstacles` gives them: without the selected chains and the
    moving vias, which the optimizer adds at their current shapes."""


def movable_vias(
    vias: Sequence[Via],
    contacts: Mapping[str, Sequence[ViaContact]],
    chains: Sequence[Chain],
) -> list[MovableVia]:
    """The vias among `vias` that may move, warning once about each other one.

    `contacts` gives the same-net copper touching each via, as
    `BoardSnapshot.read` finds it.
    """
    chain_of = {s: k for k, c in enumerate(chains) for s in c.segment_ids}
    movable: list[MovableVia] = []
    for via in vias:
        ends = tuple(
            ChainEnd(k, last)
            for k, chain in enumerate(chains)
            for last, point in ((False, chain.points[0]), (True, chain.points[-1]))
            if chain.net == via.net
            and math.dist(point, via.position) <= (via.diameter + chain.width) / 2
        )
        ended = {end.chain for end in ends}
        strangers = [
            c.description
            for c in contacts.get(via.id, ())
            if c.track_id is None or chain_of.get(c.track_id) not in ended
        ]
        reason = _fixed_reason(via, strangers, len(ends))
        if reason is None:
            movable.append(MovableVia(via, ends))
        else:
            x, y = via.position
            logger.warning(f"Via {via.id} at ({x:.3f}, {y:.3f}) stays fixed: {reason}")
    return movable


def _fixed_reason(via: Via, strangers: Sequence[str], ends: int) -> str | None:
    if via.locked:
        return "it is locked"
    if via.kind is not ViaKind.THROUGH:
        return f"it is a {via.kind.value} via; only through vias move"
    if strangers:
        return f"copper that does not move with it touches it: {', '.join(strangers)}"
    if ends == 0:
        return "no selected chain ends at it"
    if ends < MIN_JOINED_ENDS:
        return (
            f"only {ends} selected chain end meets it, "
            f"and moving it needs {MIN_JOINED_ENDS}"
        )
    return None
