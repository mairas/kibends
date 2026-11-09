"""Decide which selected vias may move with the chains they join.

A selected via moves only when it is an unlocked through via, every copper
item touching it on any layer is the end of a selected chain or a zone of
its net, no other copper touches those ends, and either at least two such
ends meet it or none does and a zone of its net reaches it: a stitching
via, which carries no chain and moves only where other copper crowds it. A
via that moves stays inside the region of every zone of its net whose
region reaches its ring, on each layer where it does, with its centre at
least `ViaZone.margin` inside, so that the next refill reconnects it; one
whose centre does not start that deep inside, and may be connected only
now, stays fixed, with a debug line naming the zone and layer. One that
starts closer to other nets' copper than the zone's clearance and the
margin is left to the optimizer, which keeps it where it is as it keeps any
via starting inside its clearance. Every other selected via stays fixed,
with a debug line naming the reason, and pins the chains it touches like an
unselected via.
"""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from loguru import logger
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from kibends.bursts import INSIDE_TOLERANCE
from kibends.chain import Chain
from kibends.coords import UserFrame
from kibends.energy import ObstacleField
from kibends.snapshot import Disc, Via, ViaContact, ViaKind, ViaZone


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
    zones: tuple[ViaZone, ...]
    """The zones of its net whose region reaches its ring, which it stays
    inside."""


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
    clearance: float
    """Its net's netclass clearance, mm."""
    zones: tuple[ViaZone, ...] = ()
    """The zones it stays inside, whose clearance it keeps from the chains
    and vias of other nets on their layers."""


def may_move(via: Via) -> bool:
    """Whether `via` is of a kind that moves: an unlocked through via."""
    return not via.locked and via.kind is ViaKind.THROUGH


def carried_ends(via: Via, chains: Sequence[Chain]) -> tuple[ChainEnd, ...]:
    """The ends of `chains` whose copper overlaps `via`'s, in chain order."""
    return tuple(
        ChainEnd(k, last)
        for k, chain in enumerate(chains)
        for last, point in ((False, chain.points[0]), (True, chain.points[-1]))
        if chain.net == via.net
        and math.dist(point, via.position) <= (via.diameter + chain.width) / 2
    )


def carried_copper(
    vias: Sequence[Via], chains: Sequence[Chain]
) -> dict[str, tuple[Disc, ...]]:
    """By via id, the copper of the chain ends each of `vias` would carry,
    for `BoardSnapshot.read` to find what touches it."""
    return {
        via.id: tuple(
            Disc(chain.points[-1 if end.last else 0], chain.width / 2)
            for end in carried_ends(via, chains)
            for chain in (chains[end.chain],)
        )
        for via in vias
    }


def movable_vias(
    vias: Sequence[Via],
    contacts: Mapping[str, Sequence[ViaContact]],
    chains: Sequence[Chain],
    *,
    frame: UserFrame,
    zones: Mapping[str, Sequence[ViaZone]],
) -> list[MovableVia]:
    """The vias among `vias` that may move, logging once at debug level why
    each other one stays, at its position in `frame`.

    `contacts` gives the same-net copper touching each via or the chain ends
    it carries, as `BoardSnapshot.read` finds it from `carried_copper`, and
    `zones` the zones of its net whose region reaches each one's ring,
    which keep it from moving only when its centre does not start
    `ViaZone.margin` inside their regions.
    """
    chains_of: dict[str, set[int]] = defaultdict(set)
    for k, c in enumerate(chains):
        for s in c.segment_ids:
            chains_of[s].add(k)
    movable: list[MovableVia] = []
    for via in vias:
        ends = carried_ends(via, chains)
        ended = {end.chain for end in ends}
        strangers = [
            c.description
            for c in contacts.get(via.id, ())
            if c.track_id is None or chains_of[c.track_id].isdisjoint(ended)
        ]
        inside = tuple(zones.get(via.id, ()))
        reason = _fixed_reason(via, strangers, len(ends), bool(inside)) or (
            _edge_reason(via, inside)
        )
        if reason is None:
            for zone in inside:
                logger.debug(f"Via {via.id} stays inside {zone.description}")
            movable.append(MovableVia(via, ends, inside))
        else:
            logger.debug(
                f"Via {via.id} at {frame.point(via.position)} stays fixed: {reason}"
            )
    return movable


def _fixed_reason(
    via: Via, strangers: Sequence[str], ends: int, zoned: bool
) -> str | None:
    if via.locked:
        return "it is locked"
    if via.kind is not ViaKind.THROUGH:
        return f"it is a {via.kind.value} via; only through vias move"
    if strangers:
        verb = "connects" if len(strangers) == 1 else "connect"
        return f"{', '.join(strangers)} {verb} to it"
    if ends == 0:
        # A stitching via: its zones keep it connected wherever it moves.
        if zoned:
            return None
        return "no selected track ends at it and no zone of its net reaches it"
    if ends == 1:
        return "only one selected track ends at it"
    return None


def _edge_reason(via: Via, zones: Sequence[ViaZone]) -> str | None:
    """Why `via` cannot stay inside the regions of `zones`: on some layer,
    it does not start inside the region by the zone's margin."""
    centre = Point(via.position)
    edges = [
        f"zone {zone.name} on {layer}"
        for zone in zones
        for layer, region in zone.regions.items()
        if _depth(region, centre) < zone.margin - INSIDE_TOLERANCE
    ]
    if not edges:
        return None
    return f"it sits at the edge of {' and '.join(edges)}"


def _depth(area: BaseGeometry, point: Point) -> float:
    """How far `point` lies inside `area`, negative outside it."""
    if area.is_empty:
        return -math.inf
    to_edge = area.boundary.distance(point)
    return to_edge if area.covers(point) else -to_edge
