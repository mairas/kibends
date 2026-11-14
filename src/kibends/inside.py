"""Refusing a run whose selected copper starts inside a clearance, by the
rules a user can see.

A burst never brings copper newly inside a clearance, nor deeper inside
another selected chain's or moving via's, than its start had it
(`kibends_core::burst`, `kibends_core::optimize::checks`), so a run that
starts clean ends clean. Before the first burst the optimizer judges each
problem's inputs, the copper as drawn with its long edges split, by the
measure its guard counts violations with (`_core.ProblemState.start_inside`):
a member's run of vertices inside the clearance of its static copper or of
another member or via, but a run that holds a pinned vertex, which cannot
move, and each edge that cuts in between two vertices that are clear; a
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
where the chain is left unchanged. Nor is a place refused that is inside
by no more than KiCad's grid rounds copper at exactly a clearance
(`START_TOLERANCE`).

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
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from kibends import _core
from kibends.chain import NM_PER_MM, Point2
from kibends.coords import UserFrame
from kibends.energy import FloatArray, ObstacleField
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
from kibends.snapshot import Arc, Via
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
EDGE_STEP = 1e-4
"""mm, the step across which the board edge's distance tells which side of
a place the edge lies on."""

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
        """The place as one line, positions in `frame`."""
        where = f"at {frame.point(self.at)} mm"
        distance = self.distance
        if self.required is None or distance is None:
            return (
                f"{self.subject} is {self.depth:.3f} mm inside its clearance of "
                f"{self.obstacle} {where}"
            )
        required = f"{self.required:.3f} mm required"
        if distance < 0.0:
            return (
                f"{self.subject} overlaps {self.obstacle} by {-distance:.3f} mm "
                f"{where}, {required}"
            )
        return (
            f"{self.subject} is {distance:.3f} mm from {self.obstacle} {where}, "
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


def circle_centre(a: Point2, b: Point2, c: Point2) -> Point2 | None:
    """The centre of the circle through `a`, `b` and `c`, None where they
    are collinear."""
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if d == 0.0:
        return None
    sa, sb, sc = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    return (
        (sa * (by - cy) + sb * (cy - ay) + sc * (ay - by)) / d,
        (sa * (cx - bx) + sb * (ax - cx) + sc * (bx - ax)) / d,
    )


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


def _nearest_allowance(
    drawn: FloatArray, at: Point2, chords: ArcChords, obstacle: Point2
) -> float:
    """The allowance at `at` toward `obstacle` of the edge of `drawn`
    nearest `at`, 0 where that edge is no arc chord."""
    edges = list(pairwise(_vertices(drawn)))
    if not edges:
        return 0.0
    a, b = min(edges, key=lambda e: LineString(e).distance(Point(at)))
    chord = _chord(chords, a, b)
    return 0.0 if chord is None else chord.allowance(a, b, at, obstacle)


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
    if via is not None and other_member is not None:
        return _via_and_member(via, other_member, rules, clearance_of)
    if member is not None and other_via is not None:
        return _via_and_member(other_via, member, rules, clearance_of)
    if via is not None and other_via is not None:
        return _two_vias(via, other_via, rules, clearance_of)
    if via is not None:
        return _via_and_static(via, stops=inside.against is _core.InsideOf.STOPS)
    assert member is not None
    if other_member is not None:
        return _two_members(member, other_member, rules)
    return _member_and_static(inside, member, rules)


def _via_and_member(
    via: ViaInput,
    member: Member,
    rules: Rules,
    clearance_of: Callable[[str], float],
) -> Violation | None:
    """A via inside a member's clearance, judged by each drawn track's
    netclass and hole rules without the via's zones."""
    centre = Point(via.via.position)
    worst: tuple[float, float, ChainCentreline] | None = None
    for track in _tracks(member):
        keep = via_chain_inflation(
            via.via, track, clearance_of, hole_clearance=rules.hole_clearance
        )
        if keep is None:
            continue
        depth = keep - LineString(track.points).distance(centre)
        depth -= _nearest_allowance(
            track.points, via.via.position, rules.chords, via.via.position
        )
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


def _via_and_static(via: ViaInput, *, stops: bool) -> Violation | None:
    """A via inside static copper's clearance, judged by its field without
    its zones, or outside a stop, of which only its net's holes keep a rule,
    hole-to-hole."""
    centre = Point(via.via.position)
    if stops:
        # A hole is a point, whose distance shapely measures exactly.
        found = _nearest(_statics(via.own_net), centre)
        if found is None or found[1] >= -START_TOLERANCE:
            return None
        item, gap = found
        if item.kind is not ObstacleKind.HOLE:
            return None
        return Violation(
            _via_name(via.via), item.description, via.via.position, -gap, None
        )
    field = via.rules_copper or via.copper
    distance, _ = field.signed_distance(np.array([via.via.position], dtype=float))
    depth = -float(distance[0])
    if depth <= START_TOLERANCE:
        return None
    statics = _statics(field)
    found = _nearest(statics, centre)
    obstacle = _unnamed(field, statics)
    required: float | None = None
    if found is not None and found[1] < -START_TOLERANCE:
        item, _ = found
        obstacle = item.description
        if item.net is not None:
            required = item.inflation - via.via.diameter / 2 - item.extent
    return Violation(_via_name(via.via), obstacle, via.via.position, depth, required)


def _two_members(member: Member, other: Member, rules: Rules) -> Violation | None:
    """Two members inside each other's clearance, judged by their drawn
    tracks and the larger netclass clearance."""
    worst: tuple[float, float, Point2] | None = None
    for track in _tracks(member):
        line = LineString(track.points)
        for against in _tracks(other):
            far = LineString(against.points)
            keep = max(track.clearance, against.clearance)
            here, there = (_point(p) for p in shapely.ops.nearest_points(line, far))
            gap = math.dist(here, there) - track.width / 2 - against.width / 2
            allowance = max(
                _nearest_allowance(track.points, here, rules.chords, there),
                _nearest_allowance(against.points, there, rules.chords, here),
            )
            depth = keep - gap - allowance
            if worst is None or depth > worst[0]:
                worst = (depth, keep, here)
    if worst is None or worst[0] <= START_TOLERANCE:
        return None
    depth, keep, at = worst
    return Violation(_member_name(member), _member_name(other), at, depth, keep)


def _toward_field(field: ObstacleField, at: Point2) -> Point2:
    """A point toward what `field` holds nearest `at`, against the field's
    gradient there."""
    _, gradient = field.signed_distance(np.array([at], dtype=float))
    return (at[0] - float(gradient[0][0]), at[1] - float(gradient[0][1]))


def _same_net(item: Obstacle, track: ChainCentreline) -> bool:
    """Whether `item` is copper of `track`'s net or a hole drilled in it,
    which keeps no clearance from the track."""
    return track.net in (item.net, item.owner)


@dataclass(frozen=True)
class _Inside:
    """How deep a place's copper is inside one obstacle's clearance."""

    obstacle: str
    depth: float
    """mm, before any chord's allowance."""
    at: Point2
    """The copper's deepest point."""
    toward: Point2
    """A point on the obstacle's side of `at`, for the allowance."""
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


def _item_depth(
    item: Obstacle, keep: float, copper: BaseGeometry, samples: FloatArray
) -> tuple[float, Point2, Point2]:
    """How deep `copper` is inside `item`'s clearance `keep`, its deepest
    point and the item's nearest point to it. Inside an area, as a pad's,
    the depth runs on into it: shapely's distance there is zero."""
    here, there = shapely.ops.nearest_points(copper, item.geometry)
    depth = keep - float(here.distance(there))
    at, toward = _point(here), _point(there)
    if depth >= keep and item.geometry.area > 0.0:
        boundary = item.geometry.boundary
        within = [p for p in samples if item.geometry.contains(Point(p))]
        if within:
            into = [float(boundary.distance(Point(p))) for p in within]
            k = int(np.argmax(into))
            depth = keep + into[k]
            at = toward = (float(within[k][0]), float(within[k][1]))
    return depth, at, toward


def _edge_depth(
    field: Obstacles, samples: FloatArray, offset: float
) -> tuple[float, Point2, Point2] | None:
    """How deep `samples` reach into the clearance of `field`'s board edge,
    less `offset`, the deepest of them and a point toward the edge; None
    without an edge or where none is inside it."""
    edges = field.static_edges
    if len(edges) <= len(field.items) or not len(samples):
        return None
    edge = int(edges.obstacles[len(field.items)])
    points, obstacles, distances = edges.distances(samples, -offset)
    mine = obstacles == edge
    if not mine.any():
        return None
    k = int(np.argmin(distances[mine]))
    x, y = samples[int(points[mine][k])]
    at = (float(x), float(y))
    # Which side the edge lies on, from how its distance changes nearby.
    step = np.array([[x + EDGE_STEP, y], [x, y + EDGE_STEP], [x, y]])
    near_points, near_obstacles, near = edges.distances(step, math.inf)
    by_point = {
        int(p): float(d)
        for p, o, d in zip(near_points, near_obstacles, near, strict=True)
        if o == edge
    }
    gx = by_point.get(0, 0.0) - by_point.get(2, 0.0)
    gy = by_point.get(1, 0.0) - by_point.get(2, 0.0)
    norm = math.hypot(gx, gy) or 1.0
    return -float(distances[mine][k]) - offset, at, (x - gx / norm, y - gy / norm)


def _insides(
    field: ObstacleField | None,
    copper: BaseGeometry,
    track: ChainCentreline,
    offset: float,
) -> list[_Inside]:
    """Each obstacle of `field` that `copper` of `track` is inside the
    clearance of by more than START_TOLERANCE, nothing of the track's net:
    each item, and the board edge, each measured on its own. The field's
    clearances reach `offset` beyond the track's, as a section's reach its
    outer extent."""
    if field is None:
        return []
    if isinstance(field, CompositeField):
        return [i for f in field.fields for i in _insides(f, copper, track, offset)]
    if not isinstance(field, Obstacles):
        return []
    samples = _samples(copper)
    found: list[_Inside] = []
    for static in _statics(field):
        keeps = np.array([item.inflation for item in static.items]) - offset
        depths = keeps - _distances(static, copper)
        for k in np.flatnonzero(depths > START_TOLERANCE):
            item = static.items[int(k)]
            if _same_net(item, track):
                continue
            depth, at, toward = _item_depth(item, float(keeps[k]), copper, samples)
            required = (
                None
                if item.net is None
                else item.inflation - offset - track.width / 2 - item.extent
            )
            found.append(_Inside(item.description, depth, at, toward, required))
    edge = _edge_depth(field, samples, offset)
    if edge is not None and edge[0] > START_TOLERANCE:
        depth, at, toward = edge
        found.append(_Inside(BOARD_EDGE, depth, at, toward, None))
    return found


def _unmeasured(
    field: ObstacleField | None, samples: FloatArray
) -> tuple[float, Point2, Point2] | None:
    """How deep `samples` reach into the parts of `field` that hold no
    items to name, such as a distance field, the deepest of them and a point
    toward it; None where it has none."""
    if field is None or isinstance(field, Obstacles) or not len(samples):
        return None
    if isinstance(field, CompositeField):
        parts = [_unmeasured(f, samples) for f in field.fields]
        found = [p for p in parts if p is not None]
        return max(found, key=lambda p: p[0]) if found else None
    distance, _ = field.signed_distance(samples)
    k = int(np.argmin(distance))
    at = (float(samples[k][0]), float(samples[k][1]))
    return -float(distance[k]), at, _toward_field(field, at)


def _member_and_static(
    inside: _core.Inside, member: Member, rules: Rules
) -> Violation | None:
    """A member inside its static field's clearance, judged at the place
    against each obstacle it is inside, the board edge and every item not
    of its net, each by its own depth less the allowance of the arc chord
    nearest toward it (`Chord.allowance`): the worst. A chain is judged as
    the place itself, a section as its two drawn tracks near it. What no
    item names, as a distance field, is judged by the optimizer's measure
    of the place."""
    place = _subject(inside.start, inside.end)
    tracks = _tracks(member)
    if isinstance(member, PairBlock):
        field = member.chain.field
        extent = (
            member.given.gap + member.given.a.width
        ) / 2 + member.given.a.width / 2
        coppers = [
            (track, LineString(track.points).intersection(place.buffer(2 * extent)))
            for track in tracks
        ]
        offset = extent - member.given.a.width / 2
    else:
        field = member.field
        coppers = [(member.centreline, place)]
        offset = 0.0
    worst: _Inside | None = None
    worst_depth = START_TOLERANCE
    for track, copper in coppers:
        if copper.is_empty:
            continue
        for found in _insides(field, copper, track, offset):
            depth = found.depth - _nearest_allowance(
                track.points, found.at, rules.chords, found.toward
            )
            if depth > worst_depth:
                worst, worst_depth = found, depth
    unnamed = _unmeasured(field, _samples(place))
    if unnamed is not None:
        depth, at, toward = unnamed
        allowance = max(
            _nearest_allowance(
                track.points,
                _point(
                    shapely.ops.nearest_points(LineString(track.points), Point(at))[0]
                ),
                rules.chords,
                toward,
            )
            for track in tracks
        )
        if depth - allowance > worst_depth:
            worst = _Inside(STATIC_COPPER, depth, at, toward, None)
            worst_depth = depth - allowance
    if worst is None:
        return None
    return Violation(
        _member_name(member), worst.obstacle, worst.at, worst_depth, worst.required
    )
