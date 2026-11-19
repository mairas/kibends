"""Refusing a run whose selected copper starts inside a clearance, by the
rules a user can see.

A burst never brings copper newly inside a clearance, nor deeper inside
another selected chain's or moving via's, than its start had it
(`kibends_core::burst`, `kibends_core::optimize::checks`), so a run that
starts clean ends clean. Before the first burst the optimizer judges the
problem's inputs, the copper as drawn with its long edges split, by the
measure its guard counts violations with
(`_core.ProblemState.start_inside`): a member's run of vertices inside the
clearance of its static copper or of another member or via, a run that
holds a pinned vertex included, and each edge that cuts in between two
vertices that are clear; a
moving via inside the clearance of any obstacle or outside one of its
stops. A pinned end on a pad or via of its own net is none of these, since
a member's field holds none of its net's copper.

That measure keeps more than the rules in places, so each place is judged
again by the rules a user can see, as kibends reads them: the netclass
clearances, the hole and edge clearances and hole-to-hole (`violations`).
A member's place inside its static copper is judged against every
obstacle it is inside, the board edge and each item not of its net, each
by its own depth, and the worst is named. Copper inside only what the
optimizer adds, a section's stand-in for its two tracks or a via's zone
clearance and margin, starts, and the bursts move it out, as they move
any copper starting inside the optimizer's clearances.

A selected arc reaches the optimizer as chords (`snapshot.Arc.segments`),
which sit inside the arc by up to their sagitta, about 1 µm on a 1 mm
radius, and the start is judged by those chords. Where copper lies on an
arc's concave side, the chord nearest it is nearer than the arc, so
copper's deepest point on a chord is allowed the arc's offset from the
chord there (`Chord.allowance`). That offset is taken square to the
chord, while the arc's distance runs toward its centre, so off a chord's
middle it allows a little more than the arc is from the copper: under
0.5 µm on 5 degree chords for copper 0.6 mm off them. On the convex side
the chord is farther than the arc and is judged as it is: an arc inside a
clearance only between chords that are clear starts, since the run
replaces it with straight tracks, and KiCad's DRC sees the drawn arc
where the chain is left unchanged. An unselected arc, which the fields
hold as its chords (`Obstacle.arc_centre`), is judged by the same rule:
copper on its concave side is allowed the arc's offset from the chord
nearest it (`_static_allowance`), on its convex side none. A hole that is
not round, such as a plated slot, which the fields hold as a circle of its
long axis (`Obstacle.oval`), is skipped: nothing beside it is judged
against it, nor a via of its net, and KiCad's DRC judges it. Nor is a
place
refused that is inside by no more than KiCad's grid rounds copper at
exactly a clearance (`START_TOLERANCE`).

`optimize.optimize_chains` raises `InsideClearanceError` naming each place
(`Violation`); `run.run_smooth` gives it the run's frame, in which its
message prints positions.
"""

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType

import numpy as np
import shapely
import shapely.ops
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from kibends import _core
from kibends.chain import NM_PER_MM, Point2
from kibends.coords import UserFrame
from kibends.energy import BoolArray, FloatArray, ObstacleField
from kibends.neighbours import ChainInput
from kibends.obstacles import (
    ChainCentreline,
    CompositeField,
    Obstacle,
    ObstacleKind,
    Obstacles,
    via_chain_inflation,
    via_via_inflation,
)
from kibends.pair_blocks import PairBlock
from kibends.snapshot import Arc, Via, circle_centre
from kibends.vias import ViaInput

INSIDE_MESSAGE = (
    "The selection starts inside the clearance it keeps; nothing committed. "
    "Move the copper apart or relax the rules, then run again:"
)

START_TOLERANCE = math.sqrt(2) * 1e-6
"""How far inside a clearance the start check lets copper be, mm: KiCad
keeps coordinates in whole nanometres, so copper drawn or committed at
exactly a clearance, each of its two sides rounded to the grid by up to
half a diagonal nanometre, may measure up to √2 nm inside it."""

EDGE_SAMPLES = 17
"""Points along an edge at which a depth no shapely distance measures is
sampled, the board edge's, an area's inside or a distance field's, ends
included: a chord's deepest point is its middle, which an odd count
holds."""
SIDE_REACH = 1e4
"""mm, past any board: how far the half planes reach that split an obstacle
by the line of an arc chord."""

BOARD_EDGE = "the board edge"
STATIC_COPPER = "static copper"
"""What a place inside no named item is inside: the board edge in a field
of items, else the field as a whole."""

Member = ChainInput | PairBlock

BOARD_FRAME = UserFrame()
"""Positions as the board holds them, until the run gives its frame."""


@dataclass(frozen=True)
class Violation:
    """A place where the selected copper as drawn is inside a clearance."""

    subject: str
    """The chain, section or via whose copper is inside."""
    obstacle: str
    """What it is too close to."""
    at: Point2
    """Where, in the board frame, mm."""
    depth: float
    """How far inside the clearance, mm."""
    required: float | None
    """The clearance required there, between copper of two nets, mm; None
    where no single clearance applies, as for a hole, the board edge or
    two vias of one net."""

    @property
    def distance(self) -> float | None:
        """How far apart the copper is, mm, negative where it overlaps;
        None where no single clearance applies."""
        return None if self.required is None else self.required - self.depth

    def describe(self, frame: UserFrame) -> str:
        """The place as one line, positions in `frame`, led by where it is,
        since the subject may name another point of the chain."""
        where = f"at {frame.point(self.at)} mm:"
        distance = self.distance
        if self.required is None or distance is None:
            return (
                f"{where} {self.subject} is {self.depth:.3f} mm inside its "
                f"clearance of {self.obstacle}"
            )
        required = f"{self.required:.3f} mm required"
        if distance < 0.0:
            return (
                f"{where} {self.subject} overlaps {self.obstacle} by "
                f"{-distance:.3f} mm, {required}"
            )
        return (
            f"{where} {self.subject} is {distance:.3f} mm from {self.obstacle}, "
            f"{required}"
        )


class InsideClearanceError(RuntimeError):
    """The selected copper starts inside a clearance; nothing optimized or
    committed. Its message names each place, positions in `frame`."""

    def __init__(
        self, violations: Sequence[Violation], frame: UserFrame = BOARD_FRAME
    ) -> None:
        self.violations = tuple(violations)
        self.frame = frame
        lines = (f"  {v.describe(frame)}" for v in self.violations)
        super().__init__("\n".join([INSIDE_MESSAGE, *lines]))

    def in_frame(self, frame: UserFrame) -> InsideClearanceError:
        """The same places, printed in `frame`."""
        return InsideClearanceError(self.violations, frame)


@dataclass(frozen=True)
class Chord:
    """A chord of a selected arc."""

    sagitta: float
    """How far it sits inside its arc, mm."""
    centre: Point2
    """The arc's centre, on its concave side."""

    def allowance(self, a: Point2, b: Point2, near: Point2, obstacle: Point2) -> float:
        """How much nearer than the arc this chord from `a` to `b` is to
        `obstacle` at `near`, a place along it: where the obstacle lies on
        the arc's concave side, the centre's side of the chord, the arc's
        offset from the chord there, sqrt(R² - u²) - (R - s) at u from the
        chord's middle, zero at its ends; nothing on the convex side, where
        the chord is the farther."""
        (ax, ay), (bx, by) = a, b
        dx, dy = bx - ax, by - ay

        def side(p: Point2) -> float:
            return dx * (p[1] - ay) - dy * (p[0] - ax)

        if self.sagitta <= 0.0 or side(self.centre) * side(obstacle) <= 0.0:
            return 0.0
        half = math.hypot(dx, dy) / 2
        radius = (half * half + self.sagitta * self.sagitta) / (2 * self.sagitta)
        along = ((near[0] - ax) * dx + (near[1] - ay) * dy) / (2 * half) - half
        u = min(abs(along), half)
        return max(0.0, math.sqrt(radius * radius - u * u) - (radius - self.sagitta))


ArcChords = Mapping[tuple[Point2, Point2], Chord]
"""The chords of the selected arcs, by their ends."""


NO_CHORDS: ArcChords = MappingProxyType({})


def sagitta(radius: float, chord: float) -> float:
    """How far a chord of length `chord` sits inside its arc of `radius`."""
    half = min(chord / 2, radius)
    return radius - math.sqrt(radius * radius - half * half)


def arc_chords(arcs: Iterable[Arc]) -> ArcChords:
    """The chords the selected `arcs` reach the optimizer as, in mm."""
    chords: dict[tuple[Point2, Point2], Chord] = {}
    for arc in arcs:
        start, mid, end = (
            (x / NM_PER_MM, y / NM_PER_MM) for x, y in (arc.start, arc.mid, arc.end)
        )
        centre = circle_centre(start, mid, end)
        if centre is None:
            continue
        radius = math.dist(start, centre)
        for segment in arc.segments():
            a = (segment.start[0] / NM_PER_MM, segment.start[1] / NM_PER_MM)
            b = (segment.end[0] / NM_PER_MM, segment.end[1] / NM_PER_MM)
            chords[(a, b)] = Chord(sagitta(radius, math.dist(a, b)), centre)
    return chords


@dataclass(frozen=True)
class Rules:
    """What a start is judged by beside the fields and the netclass
    clearances: the hole clearance and hole-to-hole the run reads, and the
    selected arcs' chords."""

    hole_clearance: float
    hole_to_hole: float
    chords: ArcChords = NO_CHORDS


@dataclass(frozen=True)
class _Static:
    """The obstacles of a field, as shapely measures them."""

    items: tuple[Obstacle, ...]
    geometries: FloatArray


def _statics(field: ObstacleField | None) -> list[_Static]:
    if field is None:
        return []
    if isinstance(field, CompositeField):
        return [s for f in field.fields for s in _statics(f)]
    if isinstance(field, Obstacles) and field.items:
        geometries = np.empty(len(field.items), dtype=object)
        geometries[:] = [item.geometry for item in field.items]
        return [_Static(field.items, geometries)]
    return []


def _distances(static: _Static, geometry: BaseGeometry) -> FloatArray:
    return np.asarray(
        shapely.distance(static.geometries, geometry),  # pyright: ignore[reportUnknownMemberType]
        dtype=float,
    )


def _nearest(
    statics: Iterable[_Static], geometry: BaseGeometry
) -> tuple[Obstacle, float] | None:
    """The item `geometry` is deepest inside the inflation of, and how far
    inside it is, mm, signed: None for no item."""
    best: tuple[Obstacle, float] | None = None
    for static in statics:
        gaps = _distances(static, geometry) - np.array(
            [item.inflation for item in static.items]
        )
        k = int(np.argmin(gaps))
        if best is None or gaps[k] < best[1]:
            best = (static.items[k], float(gaps[k]))
    return best


def _unnamed(field: ObstacleField | None, statics: Sequence[_Static]) -> str:
    return BOARD_EDGE if statics or isinstance(field, Obstacles) else STATIC_COPPER


def _subject(start: Point2, end: Point2) -> BaseGeometry:
    return Point(start) if start == end else LineString([start, end])


def _point(p: BaseGeometry) -> Point2:
    assert isinstance(p, Point)
    return (float(p.x), float(p.y))


def _member_name(member: Member) -> str:
    if isinstance(member, PairBlock):
        return (
            f"the section of {member.given.a.description} and "
            f"{member.given.b.description}"
        )
    return member.centreline.description


def _via_name(via: Via) -> str:
    return f"via {via.id}"


def _tracks(member: Member) -> tuple[ChainCentreline, ...]:
    """A member's copper as drawn: a chain itself, a section its two
    tracks."""
    if isinstance(member, PairBlock):
        return member.given.a, member.given.b
    return (member.centreline,)


def _vertices(drawn: FloatArray) -> list[Point2]:
    return [(float(x), float(y)) for x, y in drawn]


def _chord(chords: ArcChords, a: Point2, b: Point2) -> Chord | None:
    return chords.get((a, b)) or chords.get((b, a))


def _edge_allowance(
    chords: ArcChords, a: Point2, b: Point2, at: Point2, obstacle: Point2
) -> float:
    """How much a place at `at` on the edge of a member's copper from `a`
    to `b` is let off toward `obstacle`: where the edge is an arc chord, its
    allowance (`Chord.allowance`); else nothing."""
    chord = _chord(chords, a, b)
    return 0.0 if chord is None else chord.allowance(a, b, at, obstacle)


def _nearest_allowance(
    drawn: FloatArray, at: Point2, chords: ArcChords, obstacle: Point2
) -> float:
    """The allowance (`_edge_allowance`) at `at` toward `obstacle` of the
    edge of `drawn` nearest `at`."""
    edges = list(pairwise(_vertices(drawn)))
    if not edges:
        return 0.0
    a, b = min(edges, key=lambda e: LineString(e).distance(Point(at)))
    return _edge_allowance(chords, a, b, at, obstacle)


def _sub_edges(
    track: ChainCentreline, copper: BaseGeometry
) -> list[tuple[Point2, Point2, Point2, Point2]]:
    """The edges of `copper`, a stretch of `track`, each with the edge of
    the track it lies on: that edge's ends, then its own."""
    edges = list(pairwise(_vertices(track.points)))
    found: list[tuple[Point2, Point2, Point2, Point2]] = []
    for part in shapely.get_parts(copper):  # pyright: ignore[reportUnknownMemberType]
        coords = _vertices(np.asarray(shapely.get_coordinates(part), dtype=float))  # pyright: ignore[reportUnknownMemberType]
        if len(coords) == 1:
            coords = coords * 2
        for start, end in pairwise(coords):
            middle = Point((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            a, b = min(edges, key=lambda e: LineString(e).distance(middle))
            found.append((a, b, start, end))
    return found


def _segment(start: Point2, end: Point2) -> BaseGeometry:
    return Point(start) if start == end else LineString([start, end])


def violations(
    insides: Sequence[_core.Inside],
    members: Sequence[Member],
    vias: Sequence[ViaInput],
    rules: Rules,
) -> list[Violation]:
    """Each of a problem's `insides` (`_core.Inside`) judged by the rules a
    user can see, and named: of its `members` and its moving `vias`.

    The optimizer's own measure keeps more than a clearance in places: a
    section stands in for its two tracks as one track as wide as the pair,
    of a net of its own, and a via on a zone's layers keeps the zone's
    clearance and margin, so that the refill reconnects it. So a place
    involving a via or a section is judged again with each section as its
    two drawn tracks, each of its own net, by the netclass, hole and
    hole-to-hole rules without the via's zones; one only the optimizer's
    measure finds is left to the bursts. So are a via outside where its
    zones connect it and one inside its net's pads, which no clearance rule
    asks. A place on an arc's concave side is allowed the arc's offset
    from the chord it lies along (`Chord.allowance`), and one inside by no
    more than START_TOLERANCE is left out."""
    clearance_of = {v.via.net: v.clearance for v in vias}
    named = (
        _violation(i, members, vias, rules, clearance_of.__getitem__) for i in insides
    )
    # Two members, or a via and a member, are judged by their whole copper,
    # which gives the same place for each place the optimizer finds between
    # them.
    unique: dict[tuple[str, str, Point2], Violation] = {}
    for v in named:
        if v is not None:
            unique.setdefault((v.subject, v.obstacle, v.at), v)
    return list(unique.values())


def _violation(
    inside: _core.Inside,
    members: Sequence[Member],
    vias: Sequence[ViaInput],
    rules: Rules,
    clearance_of: Callable[[str], float],
) -> Violation | None:
    member = None if inside.member is None else members[inside.member]
    via = None if inside.via is None else vias[inside.via]
    other = inside.other
    other_member = None
    other_via = None
    if other is not None and inside.against is _core.InsideOf.MEMBER:
        other_member = members[other]
    if other is not None and inside.against is _core.InsideOf.VIA:
        other_via = vias[other]
    run = (inside.run_start, inside.run_end)
    if via is not None and other_member is not None:
        return _via_and_member(via, other_member, None, rules, clearance_of)
    if member is not None and other_via is not None:
        return _via_and_member(other_via, member, run, rules, clearance_of)
    if via is not None and other_via is not None:
        return _two_vias(via, other_via, rules, clearance_of)
    if via is not None and inside.against is _core.InsideOf.STOPS:
        return _via_and_stops(via, rules, clearance_of)
    if via is not None:
        return _via_and_static(via)
    assert member is not None
    if other_member is not None:
        return _two_members(member, other_member, run, rules)
    return _member_and_static(inside, member, rules)


def _via_and_member(
    via: ViaInput,
    member: Member,
    run: Sequence[Point2] | None,
    rules: Rules,
    clearance_of: Callable[[str], float],
) -> Violation | None:
    """A via inside a member's clearance, judged by each drawn track's
    netclass and hole rules without the via's zones, at each edge of the
    track, or of the stretch of it along `run`, by its depth less its own
    allowance (`_edge_allowance`): the worst."""
    centre = Point(via.via.position)
    worst: tuple[float, float, ChainCentreline] | None = None
    for track in _tracks(member):
        keep = via_chain_inflation(
            via.via, track, clearance_of, hole_clearance=rules.hole_clearance
        )
        if keep is None:
            continue
        line = LineString(track.points)
        copper = line if run is None else _along(track, run)
        for a, b, start, end in _sub_edges(track, copper):
            near = _point(shapely.ops.nearest_points(_segment(start, end), centre)[0])
            depth = keep - math.dist(near, via.via.position)
            depth -= _edge_allowance(rules.chords, a, b, near, via.via.position)
            if worst is None or depth > worst[0]:
                worst = (depth, keep, track)
    if worst is None or worst[0] <= START_TOLERANCE:
        return None
    depth, keep, track = worst
    required = keep - track.width / 2 - via.via.diameter / 2
    return Violation(
        _via_name(via.via), _member_name(member), via.via.position, depth, required
    )


def _two_vias(
    via: ViaInput,
    other: ViaInput,
    rules: Rules,
    clearance_of: Callable[[str], float],
) -> Violation | None:
    """Two vias inside each other's clearance or hole-to-hole, judged
    without their zones."""
    keep = via_via_inflation(
        via.via,
        other.via,
        clearance_of,
        hole_clearance=rules.hole_clearance,
        hole_to_hole=rules.hole_to_hole,
    )
    if keep is None:
        return None
    depth = keep - math.dist(via.via.position, other.via.position)
    if depth <= START_TOLERANCE:
        return None
    # Of one net only hole-to-hole applies, which keeps no copper gap.
    required = (
        None
        if via.via.net == other.via.net
        else keep - via.via.diameter / 2 - other.via.diameter / 2
    )
    return Violation(
        _via_name(via.via), _via_name(other.via), via.via.position, depth, required
    )


def _via_and_stops(
    via: ViaInput, rules: Rules, clearance_of: Callable[[str], float]
) -> Violation | None:
    """A via outside a stop, of which only its net's holes keep a rule:
    each hole of its net's pads that `own_net` holds, and each via of its
    net standing still (`ViaInput.own_vias`), judged for hole-to-hole; its
    net's pads only stop it."""
    centre = Point(via.via.position)
    worst: tuple[float, str] | None = None
    for static in _statics(via.own_net):
        for item in static.items:
            if item.kind is not ObstacleKind.HOLE or item.oval:
                continue
            depth = item.inflation - float(item.geometry.distance(centre))
            if worst is None or depth > worst[0]:
                worst = (depth, item.description)
    for other in via.own_vias:
        keep = via_via_inflation(
            via.via,
            other,
            clearance_of,
            hole_clearance=rules.hole_clearance,
            hole_to_hole=rules.hole_to_hole,
        )
        if keep is None:
            continue
        depth = keep - math.dist(via.via.position, other.position)
        if worst is None or depth > worst[0]:
            worst = (depth, _via_name(other))
    if worst is None or worst[0] <= START_TOLERANCE:
        return None
    depth, obstacle = worst
    return Violation(_via_name(via.via), obstacle, via.via.position, depth, None)


def _via_and_static(via: ViaInput) -> Violation | None:
    """A via inside static copper's clearance, judged by its field without
    its zones."""
    centre = Point(via.via.position)
    field = via.rules_copper
    distance, _ = field.signed_distance(np.array([via.via.position], dtype=float))
    depth = -float(distance[0])
    if depth <= START_TOLERANCE:
        return None
    statics = _statics(field)
    found = _nearest(statics, centre)
    if (
        found is not None
        and (_skipped(found[0]) or found[0].arc_centre is not None)
        and -found[1] >= depth - START_TOLERANCE
    ):
        # What the start check skips, or an unselected arc's chords, which
        # it allows the arc's offset, explains the depth: judge them so, and
        # what else there is.
        plain = [
            _Static(
                tuple(i for i in s.items if not _skipped(i) and i.arc_centre is None),
                np.array(
                    [
                        i.geometry
                        for i in s.items
                        if not _skipped(i) and i.arc_centre is None
                    ],
                    dtype=object,
                ),
            )
            for s in statics
        ]
        found = _nearest([s for s in plain if s.items], centre)
        depth = max(
            -found[1] if found is not None else -math.inf,
            _edge_depth_at(field, via.via.position),
        )
        for s in statics:
            for item in s.items:
                if item.arc_centre is None:
                    continue
                there = _point(shapely.ops.nearest_points(item.geometry, centre)[0])
                allowed = (
                    item.inflation
                    - math.dist(there, via.via.position)
                    - _static_allowance(item, there, via.via.position)
                )
                if allowed > depth:
                    depth, found = allowed, (item, -allowed)
        if depth <= START_TOLERANCE:
            return None
    obstacle = _unnamed(field, statics)
    required: float | None = None
    if found is not None and found[1] < -START_TOLERANCE:
        item, _ = found
        obstacle = item.description
        if item.net is not None:
            required = item.inflation - via.via.diameter / 2 - item.extent
    return Violation(_via_name(via.via), obstacle, via.via.position, depth, required)


def _skipped(item: Obstacle) -> bool:
    """Whether the start check leaves `item` to KiCad's DRC: a hole that is
    not round, which the fields hold as a circle of its long axis."""
    return item.oval


def _static_allowance(item: Obstacle, at: Point2, toward: Point2) -> float:
    """How much nearer than its arc an unselected arc's chord is to `toward`
    at `at`, the item's point nearest it (`Chord.allowance`): the chord of
    the item's edge nearest `at`, about the arc's centre. Nothing for any
    other item."""
    centre = item.arc_centre
    if centre is None:
        return 0.0
    coords = _vertices(np.asarray(shapely.get_coordinates(item.geometry), dtype=float))  # pyright: ignore[reportUnknownMemberType]
    edges = list(pairwise(coords))
    if not edges:
        return 0.0
    a, b = min(edges, key=lambda e: LineString(e).distance(Point(at)))
    chord = Chord(sagitta(math.dist(a, centre), math.dist(a, b)), centre)
    return chord.allowance(a, b, at, toward)


def _edge_depth_at(field: ObstacleField | None, at: Point2) -> float:
    """How deep `at` is inside the clearance of the board edges of
    `field`'s parts; -inf where it has none."""
    if isinstance(field, CompositeField):
        return max((_edge_depth_at(f, at) for f in field.fields), default=-math.inf)
    if not isinstance(field, Obstacles) or field.edge is None:
        return -math.inf
    outline, inflation = field.edge
    distance = float(outline.boundary.distance(Point(at)))
    inside = bool(outline.contains(Point(at)))
    return inflation - distance if inside else inflation + distance


def _two_members(
    member: Member, other: Member, run: Sequence[Point2], rules: Rules
) -> Violation | None:
    """Two members inside each other's clearance, judged by their drawn
    tracks and the larger netclass clearance: the member's along `run`, at
    each of its samples and its nearest point to the other, each by its
    depth less its own allowance (`_edge_allowance`) and the other's there:
    the worst."""
    along = np.linspace(0.0, 1.0, EDGE_SAMPLES)
    worst: tuple[float, float, Point2] | None = None
    for track in _tracks(member):
        copper = _along(track, run)
        for against in _tracks(other):
            far = LineString(against.points)
            keep = max(track.clearance, against.clearance)
            for a, b, start, end in _sub_edges(track, copper):
                nearest = _point(
                    shapely.ops.nearest_points(_segment(start, end), far)[0]
                )
                places = [
                    nearest,
                    *(
                        (
                            start[0] + t * (end[0] - start[0]),
                            start[1] + t * (end[1] - start[1]),
                        )
                        for t in along
                    ),
                ]
                for here in places:
                    there = _point(shapely.ops.nearest_points(far, Point(here))[0])
                    gap = math.dist(here, there) - track.width / 2 - against.width / 2
                    allowance = max(
                        _edge_allowance(rules.chords, a, b, here, there),
                        _nearest_allowance(against.points, there, rules.chords, here),
                    )
                    depth = keep - gap - allowance
                    if worst is None or depth > worst[0]:
                        worst = (depth, keep, here)
    if worst is None or worst[0] <= START_TOLERANCE:
        return None
    depth, keep, at = worst
    return Violation(_member_name(member), _member_name(other), at, depth, keep)


def _same_net(item: Obstacle, track: ChainCentreline) -> bool:
    """Whether `item` is copper of `track`'s net or a plated hole drilled
    through it, which keeps no clearance from the track."""
    return track.net in (item.net, item.owner)


Sample = tuple[float, Point2, Point2]
"""How deep copper is inside an obstacle's clearance at a point, mm, before
any chord's allowance; the point; and a point on the obstacle's side of it,
for the allowance."""


@dataclass(frozen=True)
class _Inside:
    """Where a place's copper is inside one obstacle's clearance."""

    obstacle: str
    samples: tuple[Sample, ...]
    """Each deeper than START_TOLERANCE."""
    required: float | None


def _samples(copper: BaseGeometry) -> FloatArray:
    """The points of `copper`, a point or polylines, at which a depth is
    sampled: its vertices and EDGE_SAMPLES along each of its edges."""
    along = np.linspace(0.0, 1.0, EDGE_SAMPLES)[:, None]
    parts: list[FloatArray] = []
    for part in shapely.get_parts(copper):  # pyright: ignore[reportUnknownMemberType]
        coords = np.asarray(shapely.get_coordinates(part), dtype=float)  # pyright: ignore[reportUnknownMemberType]
        if len(coords) == 1:
            parts.append(coords)
        for a, b in pairwise(coords):
            parts.append(a + along * (b - a))
    return np.vstack(parts) if parts else np.empty((0, 2))


def _chord_at(
    drawn: FloatArray, at: Point2, chords: ArcChords
) -> tuple[Point2, Point2, Chord] | None:
    """The arc chord of `drawn` nearest `at`, None where that edge is
    none."""
    edges = list(pairwise(_vertices(drawn)))
    if not edges:
        return None
    a, b = min(edges, key=lambda e: LineString(e).distance(Point(at)))
    chord = _chord(chords, a, b)
    return None if chord is None else (a, b, chord)


def _sides(
    target: BaseGeometry, a: Point2, b: Point2, chord: Chord
) -> tuple[BaseGeometry, ...]:
    """`target` split by the line through the chord from `a` to `b`: its
    part on the arc's concave side, the centre's, and on its convex side,
    each where it has one."""
    ux, uy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(ux, uy)
    ux, uy = ux / length, uy / length
    nx, ny = -uy, ux
    if nx * (chord.centre[0] - a[0]) + ny * (chord.centre[1] - a[1]) < 0.0:
        nx, ny = -nx, -ny
    start = (a[0] - ux * SIDE_REACH, a[1] - uy * SIDE_REACH)
    end = (b[0] + ux * SIDE_REACH, b[1] + uy * SIDE_REACH)
    parts = [
        target.intersection(
            Polygon(
                [
                    start,
                    end,
                    (end[0] + s * nx * SIDE_REACH, end[1] + s * ny * SIDE_REACH),
                    (start[0] + s * nx * SIDE_REACH, start[1] + s * ny * SIDE_REACH),
                ]
            )
        )
        for s in (1.0, -1.0)
    ]
    return tuple(p for p in parts if not p.is_empty)


def _measured(
    target: BaseGeometry,
    keep: float,
    samples: FloatArray,
    within: BoolArray,
    track: ChainCentreline,
    chords: ArcChords,
    item: Obstacle | None = None,
) -> list[Sample]:
    """How deep each of `samples` is inside the clearance `keep` of
    `target`, the boundary of an obstacle, with its nearest point on it.
    Inside the obstacle's area (`within`) the depth runs on into it. A
    sample on an arc chord of `track` is measured against the target's
    part on each side of the chord on its own, so that the allowance
    (`Chord.allowance`) of the concave side does not cover the convex. An
    unselected arc `item`'s chords are allowed their own toward the sample
    (`_static_allowance`)."""
    distances = np.asarray(
        shapely.distance(shapely.points(samples), target),  # pyright: ignore[reportUnknownMemberType]
        dtype=float,
    )
    split: dict[tuple[Point2, Point2], tuple[BaseGeometry, ...]] = {}
    found: list[Sample] = []
    for (x, y), distance, inside in zip(samples, distances, within, strict=True):
        at = (float(x), float(y))
        if inside:
            found.append((keep + float(distance), at, at))
            continue
        if keep - distance <= START_TOLERANCE:
            continue
        on = _chord_at(track.points, at, chords)
        parts: tuple[BaseGeometry, ...] = (target,)
        if on is not None:
            a, b, chord = on
            if (a, b) not in split:
                split[(a, b)] = _sides(target, a, b, chord)
            parts = split[(a, b)]
        for part in parts:
            there = _point(shapely.ops.nearest_points(Point(at), part)[1])
            allowance = 0.0 if item is None else _static_allowance(item, there, at)
            depth = keep - float(part.distance(Point(at))) - allowance
            found.append((depth, at, there))
    return [s for s in found if s[0] > START_TOLERANCE]


def _item_samples(
    item: Obstacle,
    keep: float,
    copper: BaseGeometry,
    samples: FloatArray,
    track: ChainCentreline,
    chords: ArcChords,
) -> list[Sample]:
    """Where `copper` of `track` is inside `item`'s clearance `keep`: its
    nearest point to the item, and each of its `samples` inside."""
    here, there = (_point(p) for p in shapely.ops.nearest_points(copper, item.geometry))
    allowance = _static_allowance(item, there, here)
    nearest: Sample = (keep - math.dist(here, there) - allowance, here, there)
    area = item.geometry.area > 0.0
    within = (
        np.asarray(shapely.contains_xy(item.geometry, samples[:, 0], samples[:, 1]))  # pyright: ignore[reportUnknownMemberType]
        if area
        else np.zeros(len(samples), dtype=bool)
    )
    target = item.geometry.boundary if area else item.geometry
    found = _measured(target, keep, samples, within, track, chords, item)
    if nearest[0] > START_TOLERANCE and not within.any():
        found.append(nearest)
    return found


def _insides(
    field: ObstacleField | None,
    copper: BaseGeometry,
    track: ChainCentreline,
    offset: float,
    chords: ArcChords,
    *,
    own_net_skipped: bool,
) -> list[_Inside]:
    """Each obstacle of `field` that `copper` of `track` is inside the
    clearance of by more than START_TOLERANCE: each item, and the board
    edge, each measured on its own; with `own_net_skipped`, as a section's
    field holds them, nothing of the track's net. The field's clearances
    reach `offset` beyond the track's, as a section's reach its outer
    extent."""
    if field is None:
        return []
    if isinstance(field, CompositeField):
        return [
            i
            for f in field.fields
            for i in _insides(
                f, copper, track, offset, chords, own_net_skipped=own_net_skipped
            )
        ]
    if not isinstance(field, Obstacles):
        return []
    samples = _samples(copper)
    found: list[_Inside] = []
    for static in _statics(field):
        keeps = np.array([item.inflation for item in static.items]) - offset
        depths = keeps - _distances(static, copper)
        for k in np.flatnonzero(depths > START_TOLERANCE):
            item = static.items[int(k)]
            if _skipped(item) or (own_net_skipped and _same_net(item, track)):
                continue
            required = (
                None
                if item.net is None
                else item.inflation - offset - track.width / 2 - item.extent
            )
            inside = _item_samples(
                item, float(keeps[k]), copper, samples, track, chords
            )
            found.append(_Inside(item.description, tuple(inside), required))
    edge = field.edge
    if edge is not None and len(samples):
        outline, inflation = edge
        outside = ~np.asarray(
            shapely.contains_xy(outline, samples[:, 0], samples[:, 1]),  # pyright: ignore[reportUnknownMemberType]
            dtype=bool,
        )
        inside = _measured(
            outline.boundary, inflation - offset, samples, outside, track, chords
        )
        if inside:
            found.append(_Inside(BOARD_EDGE, tuple(inside), None))
    return found


def _unmeasured(field: ObstacleField | None, samples: FloatArray) -> list[Sample]:
    """Where `samples` are inside the parts of `field` that hold no items
    to name, such as a distance field, each with a point toward what is
    nearest."""
    if field is None or isinstance(field, Obstacles) or not len(samples):
        return []
    if isinstance(field, CompositeField):
        return [s for f in field.fields for s in _unmeasured(f, samples)]
    distances, gradients = field.signed_distance(samples)
    return [
        (-float(d), (float(x), float(y)), (float(x - gx), float(y - gy)))
        for (x, y), d, (gx, gy) in zip(samples, distances, gradients, strict=True)
        if -d > START_TOLERANCE
    ]


def _along(track: ChainCentreline, run: Sequence[Point2]) -> BaseGeometry:
    """`track` along itself between where the ends of `run`, a stretch of
    its member's copper, project onto it."""
    line = LineString(track.points)
    along = [line.project(Point(p)) for p in run]
    return shapely.ops.substring(line, min(along), max(along))


def _worst(
    samples: Iterable[Sample], drawn: FloatArray, chords: ArcChords
) -> tuple[float, Point2] | None:
    """The deepest of `samples` less the allowance of the arc chord of
    `drawn` nearest each toward its own obstacle point, and where."""
    judged = [
        (depth - _nearest_allowance(drawn, at, chords, toward), at)
        for depth, at, toward in samples
    ]
    return max(judged, key=lambda j: j[0]) if judged else None


def _centreline_allowance(
    member: Member, at: Point2, toward: Point2, chords: ArcChords
) -> float:
    """The allowance at `at` toward `toward` of the edge of the member's
    centreline nearest `at`, as the optimizer measures it: a chain's own
    chord, a section's centreline as the chord of the arc concentric with
    its tracks' chords there."""
    if not isinstance(member, PairBlock):
        return _nearest_allowance(member.centreline.points, at, chords, toward)
    edges = list(pairwise(_vertices(member.chain.centreline.points)))
    if not edges:
        return 0.0
    a, b = min(edges, key=lambda e: LineString(e).distance(Point(at)))
    for track in _tracks(member):
        line = LineString(track.points)
        near = shapely.ops.nearest_points(line, Point(at))[0]
        track_edges = list(pairwise(_vertices(track.points)))
        ta, tb = min(track_edges, key=lambda e: LineString(e).distance(near))
        chord = _chord(chords, ta, tb)
        if chord is not None:
            radius = math.dist(a, chord.centre)
            arc = Chord(sagitta(radius, math.dist(a, b)), chord.centre)
            return arc.allowance(a, b, at, toward)
    return 0.0


def _member_and_static(
    inside: _core.Inside, member: Member, rules: Rules
) -> Violation | None:
    """A member inside its static field's clearance, judged at the place
    against each obstacle it is inside, the board edge and every item, each
    at every sample by its depth less the allowance there of the arc chord
    nearest toward it (`Chord.allowance`): the worst. A chain is judged as
    the stretch of it the optimizer's place stands for, its run inside
    (`_core.Inside.run_start`), a section as its two drawn tracks along
    that run, clear of their nets' copper and plated holes. What no item
    names, as a distance field, is judged by the optimizer's measure of the
    place."""
    place = _subject(inside.start, inside.end)
    run = (inside.run_start, inside.run_end)
    tracks = _tracks(member)
    coppers = [(track, _along(track, run)) for track in tracks]
    if isinstance(member, PairBlock):
        field = member.chain.field
        offset = (member.given.gap + member.given.a.width) / 2
    else:
        field = member.field
        offset = 0.0
    section = isinstance(member, PairBlock)
    worst: tuple[float, str, Point2, float | None] | None = None
    for track, copper in coppers:
        if copper.is_empty:
            continue
        for found in _insides(
            field, copper, track, offset, rules.chords, own_net_skipped=section
        ):
            judged = _worst(found.samples, track.points, rules.chords)
            if judged is not None and (worst is None or judged[0] > worst[0]):
                worst = (judged[0], found.obstacle, judged[1], found.required)
    for depth, at, toward in _unmeasured(field, _samples(place)):
        judged = depth - _centreline_allowance(member, at, toward, rules.chords)
        if worst is None or judged > worst[0]:
            worst = (judged, STATIC_COPPER, at, None)
    if worst is None or worst[0] <= START_TOLERANCE:
        return None
    depth, obstacle, at, required = worst
    return Violation(_member_name(member), obstacle, at, depth, required)
