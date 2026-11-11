"""Decide which selected vias may move with the chains they join.

A selected via moves only when it is unlocked, not on a net of a diff pair
(`pairs.in_diff_pair`: KiCad's naming rule, with the other net on the
board, whether or not the selection pairs its tracks), every copper item touching
it on a layer it spans is the end of a selected chain or a zone of its
net, no other copper touches those ends, and either such an end meets it
or a zone of its net reaches it. One end is enough: a pad's fan-out via
moves along its track toward the pad. One with no end is a stitching via,
which carries no chain and moves only where other copper crowds it.
Through, blind, buried and micro vias move alike, each on the layers of its
span: copper on a layer it does not reach neither moves with it nor holds
it (`carried_ends`, `snapshot.Via.reaches`). A chain that runs through a
via that may move, with a vertex on its centre, is cut there into two
chains that end at it (`chain_cuts`), which it then carries; one that
passes it with no vertex there is other copper touching it. A via that
moves stays inside the zones of its net whose region reaches its ring,
taken together on each layer where one does (`zone_unions`): its centre
stays inside the region of one of them by that zone's margin
(`ViaZone.margin`), or inside the union of their regions by the largest of
their margins, so that the next refill reconnects it through one of them.
One whose centre does not start so, and may be connected only now, stays
fixed, with a debug line naming the zones and layers. One that starts
closer to other nets' copper than the zone's clearance and the
margin is left to the optimizer, which moves it out as it moves any via
starting inside its clearance. Every other selected via stays fixed,
with a debug line naming the reason, and holds the chains it touches like
an unselected via: a chain vertex on its centre stays there, and a chain
with none there keeps a vertex inside it.

Every via is a ground via, on one of the run's ground nets, or a regular
one, on any other net (`ViaInput.ground`, `StaticVia`). One rule,
`pair_kind`, says how two vias whose spans overlap interact in the
optimizer's pairwise term (`via_pairs`). A ground via and a regular one, or
two ground vias of different ground nets, are a pair of two nets
(`pairs_with`); the vias standing still that a moving via so pairs with
are held apart from its field (`ViaInput.static_vias`). Ground vias of one
net stay in its stops, which keep hole-to-hole, and the pairwise term also
has them repel each other.
"""

import enum
import math
from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np
from loguru import logger
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from kibends import _core
from kibends.bursts import INSIDE_TOLERANCE
from kibends.chain import Chain, Point2
from kibends.coords import UserFrame
from kibends.energy import ObstacleField
from kibends.obstacles import CompositeField, containment, plain_shapes
from kibends.pairs import in_diff_pair
from kibends.snapshot import (
    Disc,
    Via,
    ViaContact,
    ViaZone,
    ZoneUnion,
    zone_unions,
)


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
    inside taken together on each layer (`zone_unions`)."""


@dataclass(frozen=True, eq=False)
class ViaInput:
    """A moving via of a run, with the field of everything that stands still."""

    via: Via
    ends: tuple[ChainEnd, ...]
    """The chain ends that move with it, each keeping its offset from the
    via's centre."""
    field: ObstacleField
    """Obstacles for the via over every layer it spans, as
    `obstacles.split_via_obstacles` gives them: without the selected chains and the
    moving vias, which the optimizer adds at their current shapes."""
    clearance: float
    """Its net's netclass clearance, mm."""
    zones: tuple[ViaZone, ...] = ()
    """The zones of its net whose region reaches its ring, which it stays
    inside taken together on each layer (`containment`), and whose
    clearance it keeps from the chains and vias of other nets on their
    layers."""

    own_net: ObstacleField | None = None
    """What of its net stands still and only stops it, as `own_net_stops`
    gives it: the holes of its net's vias and plated pads, and its net's
    pads."""
    ground: bool = False
    """Whether it is on a ground net."""
    static_vias: ObstacleField | None = None
    """The vias standing still that it pairs with (`pairs_with`), split out of
    `field` as `static_via_obstacles` gives them: they limit where it may
    go, and the optimizer's pairwise term (`via_pairs`) gives their
    energy."""

    @cached_property
    def stop_parts(self) -> tuple[ObstacleField | None, ObstacleField | None]:
        """`stops` in its two parts, each None where it has none: the
        outside of where its zones connect it, and `own_net`. The checks
        judge them apart, so that a via inside one is still kept out of the
        other."""
        return containment(self.zones), self.own_net

    @cached_property
    def stops(self) -> ObstacleField | None:
        """What stands still and only stops it, without repelling it: the
        outside of where its zones, taken together on each layer, connect it
        (`containment`) and `own_net`."""
        parts = [f for f in self.stop_parts if f is not None]
        return CompositeField(parts) if parts else None

    @cached_property
    def copper(self) -> ObstacleField:
        """What stands still that it keeps its clearance from: `field` and
        `static_vias`, as `field` unsplit; what only stops it is `stops`."""
        if self.static_vias is None:
            return self.field
        return CompositeField([self.field, self.static_vias])


class StaticVia(NamedTuple):
    """A via of the board that is none of a run's moving vias, standing
    still, and its class."""

    via: Via
    ground: bool
    """Whether it is on a ground net."""
    clearance: float
    """Its net's netclass clearance, mm."""


class PairKind(enum.IntEnum):
    """How two vias interact in the optimizer's pairwise term
    (`via_pairs`)."""

    RETURN = 0
    """A ground via and a regular one: attraction and the full obstacle
    term."""
    GROUNDS = 1
    """Ground vias of two ground nets: the full obstacle term."""
    SPREAD = 2
    """Ground vias of one ground net: the repulsion alone, beyond their
    rings and clearance."""


def pair_kind(
    via: Via, ground: bool, other: Via, other_ground: bool
) -> PairKind | None:
    """How `via` and `other`, each a ground via or not, interact in the
    pairwise term; None if they do not."""
    if not via.shares_layer(other):
        return None
    if via.net == other.net:
        return PairKind.SPREAD if ground else None
    if ground and other_ground:
        return PairKind.GROUNDS
    if ground or other_ground:
        return PairKind.RETURN
    return None


def pairs_with(via: Via, ground: bool, other: StaticVia) -> bool:
    """Whether `via`, a ground via or not, pairs with `other`, standing
    still, as two vias of other nets (`pair_kind`): the pairwise term then
    gives their whole obstacle term, so `other` leaves `via`'s field."""
    kind = pair_kind(via, ground, other.via, other.ground)
    return kind is not None and kind is not PairKind.SPREAD


def may_move(via: Via, nets: Collection[str]) -> bool:
    """Whether `via` may move: it is unlocked and not on a net of a diff
    pair among `nets`, the board's."""
    return not via.locked and not in_diff_pair(via.net, nets)


def chain_cuts(vias: Sequence[Via]) -> Callable[[str, str], list[Point2]]:
    """Where `vias` cut the chains of a layer and net (`split_chains`): at
    the centre of each of them of that net that reaches that layer, so that a
    chain with a vertex there ends at it, for it to carry."""

    def cuts(layer: str, net: str) -> list[Point2]:
        return [v.position for v in vias if v.net == net and v.reaches(layer)]

    return cuts


def carried_ends(via: Via, chains: Sequence[Chain]) -> tuple[ChainEnd, ...]:
    """The ends of `chains` on layers `via` spans whose copper overlaps
    `via`'s, in chain order."""
    return tuple(
        ChainEnd(k, last)
        for k, chain in enumerate(chains)
        for last, point in ((False, chain.points[0]), (True, chain.points[-1]))
        if chain.net == via.net
        and via.reaches(chain.layer)
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
    nets: Collection[str],
) -> list[MovableVia]:
    """The vias among `vias` that may move, logging once at debug level why
    each other one stays, at its position in `frame`.

    `contacts` gives the same-net copper touching each via or the chain ends
    it carries, as `BoardSnapshot.read` finds it from `carried_copper`, and
    `zones` the zones of its net whose region reaches each one's ring,
    which keep it from moving only when, on some layer, its centre does not
    start where they connect it taken together (`ZoneUnion.centres`).
    `nets` are the board's net names, which tell its diff pairs.
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
        unions = zone_unions(inside)
        held = any(chains[end.chain].held for end in ends)
        reason = _fixed_reason(
            via, nets, strangers, bool(ends), bool(inside), held=held
        ) or (_edge_reason(via, unions))
        if reason is None:
            for layer, union in unions.items():
                logger.debug(
                    f"Via {via.id} stays inside {_zones(union.names)} on {layer}"
                )
            movable.append(MovableVia(via, ends, inside))
        else:
            logger.debug(
                f"Via {via.id} at {frame.point(via.position)} stays fixed: {reason}"
            )
    return movable


def _fixed_reason(
    via: Via,
    nets: Collection[str],
    strangers: Sequence[str],
    ended: bool,
    zoned: bool,
    *,
    held: bool,
) -> str | None:
    if via.locked:
        return "it is locked"
    if in_diff_pair(via.net, nets):
        return f"it is on {via.net}, a net of a diff pair"
    if held:
        return "a track of a selected group ends at it"
    if strangers:
        verb = "connects" if len(strangers) == 1 else "connect"
        return f"{', '.join(strangers)} {verb} to it"
    if not ended and not zoned:
        return "no selected track ends at it and no zone of its net reaches it"
    return None


def _edge_reason(via: Via, unions: Mapping[str, ZoneUnion]) -> str | None:
    """Why `via` cannot stay inside its zones: on some layer, its centre
    does not start where they connect it taken together, by layer in
    `unions`."""
    centre = Point(via.position)
    edges: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for layer, union in unions.items():
        if _depth(union.centres, centre) < -INSIDE_TOLERANCE:
            edges[union.names].append(layer)
    if not edges:
        return None
    named = (
        f"{_zones(names)} on {' and '.join(layers)}" for names, layers in edges.items()
    )
    return f"it sits at the edge of {'; '.join(named)}"


def _zones(names: Sequence[str]) -> str:
    return f"{'zone' if len(names) == 1 else 'zones'} {' and '.join(names)}"


def _depth(area: BaseGeometry, point: Point) -> float:
    """How far `point` lies inside `area`, negative outside it, as the
    containment field judges it (`_core.zone_depth`)."""
    at = np.array([[point.x, point.y]])
    return float(_core.zone_depth(plain_shapes([area]), at)[0])
