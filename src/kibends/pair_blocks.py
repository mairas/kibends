"""Coupled sections of differential pairs as blocks of the optimizer.

A coupled section moves as its centreline: a chain as wide as the pair's
outer extent, two track widths plus the gap, that keeps the larger of the
two nets' clearances, in a static field from
`obstacles.build_pair_obstacles`. The bend-radius hinge of
`energy.bend_radius_energy` keeps its inner track from folding; a section
whose own routing already bends tighter is held as it is. Its two ends are
pinned, and so is a vertex a spacing in from each, so that the pair meets
its fan-outs in the direction it did. Same-net copper that touches a track
between the ends pins the centreline where that puts the track through the
point it touches, at the section's gap, with a vertex a spacing to either
side, so that the track passes there in the direction it did; the rest of
the section moves. Both tracks are the centreline offset by half the pitch,
through their own ends and pins; where an end's own gap differs from the
pair's, the gap tapers between the two beyond that end's direction pin. The
centreline is a little wider than the pair, by as far as the outer track's
mitred corners reach beyond half the pitch at the bend-radius limit, so
that its clearance covers them.

The other blocks see the two tracks at the centreline's current shape. The
centreline sees every other chain, and every track of another diff pair,
as a track of the diff pair's outer extent would. The section and the chains of its
two nets ignore each other, as chains of one net and the two nets of a diff
pair do (`neighbours.neighbour_field`), so its fan-outs leave it freely.
"""

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import NamedTuple

import numpy as np

from kibends.bursts import INSIDE_TOLERANCE
from kibends.energy import BoolArray, FloatArray, IntArray, ObstacleField, arc_lengths
from kibends.neighbours import ChainInput
from kibends.obstacles import ChainCentreline
from kibends.pairs import (
    VERTEX_SNAP,
    BendCheck,
    Track,
    bend_radius_limit,
    centreline,
    check_bend_radius,
    offset_pair,
    offset_pair_pullback,
    with_nearest_vertices,
)

STUB_SPACINGS = 2
"""An edge leaving a pin this many spacings long or longer gets a direction
pin a spacing along it; an edge between two pins needs one more."""

TAPER_SLOPE = 0.01
"""Where an end's gap differs from the pair's, each track turns by at most
this much, as a tangent, to take up the difference beyond the end's
direction pin; less where the section is too short."""

MIN_TAPER = 1e-6
"""mm: the shortest taper, to keep it finite where the gaps agree."""


class Hold(Enum):
    """Why a coupled section stays as it is."""

    GROUP = "a track of it is in a selected group, held as drawn"
    SHORT = "it is too short between the copper at its ends to move"
    ROUTING = "its routing bends too tightly for the inner track"
    CROWDING = (
        "the copper around it kept it from rounding a corner too tight for "
        "the inner track"
    )
    FOLDING = (
        "rounding a corner too tight for the inner track would bring its "
        "parts beyond the corner closer together than they are routed"
    )


@dataclass(frozen=True, eq=False)
class PairInput:
    """One coupled section of a differential pair in a run."""

    a: ChainCentreline
    """Track a's section, with its net, width and clearance."""
    b: ChainCentreline
    """Track b's section, running the same way as `a`, of the same width."""
    a_pinned: BoolArray
    b_pinned: BoolArray
    """Where same-net copper touches each track. The centreline is fixed
    where a pin between the ends needs it, with its direction there
    (`_moving_start`), and free elsewhere."""
    gap: float
    """Edge to edge between the two tracks, mm."""
    layer: str
    field: ObstacleField
    """Obstacles for the centreline, from `obstacles.build_pair_obstacles`."""
    hold: Hold | None = None
    """Why it stays as it is regardless of its own routing."""


@dataclass(frozen=True)
class _Gaps:
    """The gap along a section: the pair's, turning into each end's own gap
    over a taper beyond the end's direction pin, so that both tracks keep
    their end directions where the routing left them a little apart."""

    middle: float
    start: float
    end: float
    held_start: float
    held_end: float
    """Arc length from each end to its direction pin, mm."""
    taper_start: float
    taper_end: float

    @staticmethod
    def uniform(gap: float) -> _Gaps:
        """`gap` all along."""
        return _Gaps(gap, gap, gap, 0.0, 0.0, MIN_TAPER, MIN_TAPER)

    @staticmethod
    def of(a: FloatArray, b: FloatArray, gap: float, width: float) -> _Gaps:
        """For track `b` beside `a`, with `a`'s direction pins at its
        second and second-to-last vertices."""
        arc = arc_lengths(a)
        held_start, held_end = float(arc[1]), float(arc[-1] - arc[-2])
        start, end = end_gaps(a, b, width)
        room = max((float(arc[-1]) - held_start - held_end) / 2, MIN_TAPER)

        def taper(other: float) -> float:
            return min(max(abs(other - gap) / 2 / TAPER_SLOPE, MIN_TAPER), room)

        return _Gaps(gap, start, end, held_start, held_end, taper(start), taper(end))

    @property
    def widest(self) -> float:
        return max(self.middle, self.start, self.end)

    def at(self, points: FloatArray) -> FloatArray:
        """The gap at each vertex of a centreline."""
        arc = arc_lengths(points)
        near_start = _ramp(arc - self.held_start, self.taper_start)
        near_end = _ramp(arc[-1] - arc - self.held_end, self.taper_end)
        return (
            self.middle
            + (self.start - self.middle) * near_start
            + (self.end - self.middle) * near_end
        )

    def pullback(self, points: FloatArray, gradient: FloatArray) -> FloatArray:
        """The gradient of a function of `at(points)`, `gradient` at each
        vertex, taken back to the vertices: a gap depends on how far along
        the centreline its vertex lies from each end."""
        arc = arc_lengths(points)
        from_start = (self.start - self.middle) * _ramp_slope(
            arc - self.held_start, self.taper_start
        )
        from_end = (self.end - self.middle) * _ramp_slope(
            arc[-1] - arc - self.held_end, self.taper_end
        )
        # Edge e counts in the arc length from the start of every vertex
        # after it, and in that from the end of every vertex up to it.
        after = np.cumsum((gradient * from_start)[::-1])[::-1]
        upto = np.cumsum(gradient * from_end)
        at_length = after[1:] + upto[:-1]
        delta = np.diff(points, axis=0)
        along = delta / np.hypot(delta[:, 0], delta[:, 1])[:, None]
        result = np.zeros_like(points)
        result[1:] += at_length[:, None] * along
        result[:-1] -= at_length[:, None] * along
        return result


def end_gaps(a: FloatArray, b: FloatArray, width: float) -> tuple[float, float]:
    """The gap at each end of track `b` beside `a`, across `a`'s end edges."""
    start = _separation(a[0], a[1], b[0]) - width
    end = _separation(a[-1], a[-2], b[-1]) - width
    return start, end


def _ramp(beyond: FloatArray, length: float) -> FloatArray:
    """1 up to 0, falling to 0 at `length`."""
    return np.clip(1.0 - beyond / length, 0.0, 1.0)


def _ramp_slope(beyond: FloatArray, length: float) -> FloatArray:
    """The slope of `_ramp` in `beyond`."""
    return np.where((beyond > 0.0) & (beyond < length), -1.0 / length, 0.0)


def _separation(end: FloatArray, toward: FloatArray, point: FloatArray) -> float:
    """Distance from `point` to the line through `end` and `toward`."""
    step = toward - end
    offset = point - end
    cross = step[0] * offset[1] - step[1] * offset[0]
    return abs(float(cross)) / float(np.hypot(step[0], step[1]))


class _Pins(NamedTuple):
    """The centreline's vertices where a track keeps a point of its own,
    the section's ends and the pins between them, and those points."""

    a_at: FloatArray
    a: FloatArray
    b_at: FloatArray
    b: FloatArray

    def held(self, points: FloatArray) -> tuple[BoolArray, BoolArray]:
        """Which vertices of the centreline at `points` hold a point of
        track a, and which one of track b."""
        return _at(points, self.a_at).any(axis=1), _at(points, self.b_at).any(axis=1)

    def keep(self, points: FloatArray, a: FloatArray, b: FloatArray) -> None:
        """Put the points the centreline at `points` holds into tracks `a`
        and `b`, its offsets, which miss them by round-off."""
        for at, own, track in ((self.a_at, self.a, a), (self.b_at, self.b, b)):
            vertex, pin = np.nonzero(_at(points, at))
            track[vertex] = own[pin]


def _at(points: FloatArray, at: FloatArray) -> BoolArray:
    """Whether each of `points` is each of `at`: fixed vertices stay where
    they started, exactly."""
    return (points[:, None] == at[None]).all(axis=2)


@dataclass(frozen=True)
class PairBlock:
    """A section's centreline as a chain of the run, and its two tracks as
    the other chains see them."""

    given: PairInput
    chain: ChainInput
    """Of a net of its own, so that no chain counts as its net's, and wide
    enough to cover the outer track's mitred corners, as the vias see it."""
    tracks: tuple[ChainInput, ChainInput]
    """Tracks a and b at the starting centreline, pinned at their ends."""
    facing: ChainInput
    """The centreline as wide as the pair's outer extent, as it keeps clear
    of the other selected chains. Their own clearance from its tracks, and
    the run's guard on their violations, cover the corners."""
    directions: FloatArray
    """Its direction pins (`_pin_directions`), which stay where they started
    while it moves."""
    pins: _Pins
    a_on_right: bool
    hold: Hold | None
    """Why it is pinned throughout and stays as it is, if it is."""
    tight_start: BendCheck | None
    """The starting centreline's bend, when it is too tight: then the
    section is held, since no result could be checked against it."""
    gaps: _Gaps
    limit: float
    """The centreline's smallest bend radius, mm."""

    @property
    def nets(self) -> frozenset[str]:
        """Its two nets, whose chains it ignores."""
        return frozenset({self.given.a.net, self.given.b.net})

    @property
    def held(self) -> bool:
        return self.hold is not None

    @property
    def apart(self) -> float:
        """How far apart, mm, the centreline keeps its own parts beyond a
        turn: the pair's outer extent plus the larger of its clearances, so
        that one part's tracks keep clear of the other's."""
        facing = self.facing.centreline
        return facing.width + facing.clearance

    def offset(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Tracks a and b along the centreline at `points`, through the
        section's own ends and pins."""
        return self._offset(self.gaps.at(points), points)

    def offset_pullback(
        self, points: FloatArray, at_a: FloatArray, at_b: FloatArray
    ) -> FloatArray:
        """The gradient of a function of `offset(points)`, `at_a` and `at_b`
        at the vertices of tracks a and b, taken back to the centreline's
        vertices. The tracks' ends and pins are the section's own, and a
        held section's tracks are its own throughout, so neither depends on
        the centreline."""
        if self.held:
            return np.zeros_like(points)
        held_a, held_b = self.pins.held(points)
        at_a, at_b = (
            np.where(held_a[:, None], 0.0, at_a),
            np.where(held_b[:, None], 0.0, at_b),
        )
        right, left = (at_a, at_b) if self.a_on_right else (at_b, at_a)
        gaps = self.gaps.at(points)
        gradient, at_gaps = offset_pair_pullback(
            points, gaps, self.given.a.width, right, left
        )
        return gradient + self.gaps.pullback(points, at_gaps)

    def offset_kept(
        self, optimised: FloatArray, kept: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Tracks a and b along `kept`, vertices of the centreline at
        `optimised`, each at its gap there, as the run offset them."""
        gaps = self.gaps.at(optimised)
        index = {(float(x), float(y)): k for k, (x, y) in enumerate(optimised)}
        at = gaps[[index[float(x), float(y)] for x, y in kept]]
        return self._offset(at, kept)

    def _offset(
        self, gaps: FloatArray, points: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        return _offset(self.given, self.a_on_right, self.pins, self.held, gaps, points)

    def taper_bounds(self, points: FloatArray) -> BoolArray:
        """Which vertices of the centreline at `points` bound the stretch
        between the tapers of ends whose gaps differ from the pair's: there
        and beyond, the gap is the pair's."""
        arc = arc_lengths(points)
        bounds = np.zeros(len(points), dtype=bool)
        gaps = self.gaps
        if abs(gaps.start - gaps.middle) > INSIDE_TOLERANCE:
            flat = np.flatnonzero(arc >= gaps.held_start + gaps.taper_start)
            bounds[flat[:1]] = True
        if abs(gaps.end - gaps.middle) > INSIDE_TOLERANCE:
            flat = np.flatnonzero(arc[-1] - arc >= gaps.held_end + gaps.taper_end)
            bounds[flat[-1:]] = True
        return bounds

    def bend(self, points: FloatArray) -> BendCheck:
        """The centreline at `points` against the bend-radius limit."""
        return check_bend_radius(points, self.gaps.widest, self.given.a.width)

    def direction_pins(self, points: FloatArray) -> BoolArray:
        """Which vertices of the centreline at `points` are its direction
        pins, which stay where they started."""
        return (points[:, None] == self.directions[None]).all(axis=2).any(axis=1)


def pair_block(given: PairInput, spacing: float) -> PairBlock:
    """The block of a coupled section, its direction pins a `spacing` from
    its ends and pins, or at the end of a shorter edge."""
    width = given.a.width
    a_on_right = centreline(given.a.points, given.b.points, width).a_on_right
    (fixed, pins, gaps, line), hold, tight_start = _start(given, spacing, a_on_right)
    held = hold is not None
    limit = bend_radius_limit(gaps.widest, width)
    half_pitch = (gaps.widest + width) / 2
    # A mitred outer corner lies half_pitch / cos(turn / 2) from the
    # centreline's vertex. At the bend-radius limit a vertex turns at most
    # 2·atan(spacing / limit) between edges as long as the spacing.
    overshoot = half_pitch * (math.hypot(1.0, spacing / limit) - 1.0)
    pseudo = ChainCentreline(
        f"{given.a.description} and {given.b.description}, coupled",
        f"{given.a.net} & {given.b.net}",
        2 * (half_pitch + overshoot) + width,
        max(given.a.clearance, given.b.clearance),
        line,
    )
    chain = ChainInput(pseudo, given.layer, fixed, given.field, held=held)
    facing = replace(chain, centreline=replace(pseudo, width=2 * half_pitch + width))
    start_a, start_b = _offset(given, a_on_right, pins, held, gaps.at(line), line)
    track_a, track_b = (
        ChainInput(
            replace(track, points=points),
            given.layer,
            pinned | _fixed_ends(len(points)) if held else _fixed_ends(len(points)),
            given.field,
        )
        for track, points, pinned in (
            (given.a, start_a, given.a_pinned),
            (given.b, start_b, given.b_pinned),
        )
    )
    held_a, held_b = pins.held(line)
    return PairBlock(
        given,
        chain,
        (track_a, track_b),
        facing,
        directions=line[fixed & ~held_a & ~held_b],
        pins=pins,
        a_on_right=a_on_right,
        hold=hold,
        tight_start=tight_start,
        gaps=gaps,
        limit=limit,
    )


class _Start(NamedTuple):
    """A section's starting centreline as the run takes it."""

    fixed: BoolArray
    pins: _Pins
    """Its ends and pins; the rest of its fixed vertices are direction
    pins."""
    gaps: _Gaps
    line: FloatArray


def _start(
    given: PairInput, spacing: float, a_on_right: bool
) -> tuple[_Start, Hold | None, BendCheck | None]:
    """The section's starting centreline, why it is held if it is, and its
    starting bend if that is why."""
    if given.hold is not None:
        return _held_start(given, a_on_right), given.hold, None
    start = _moving_start(given, spacing, a_on_right)
    if not _has_room(given, start, spacing):
        return _held_start(given, a_on_right), Hold.SHORT, None
    bend = check_bend_radius(start.line, start.gaps.widest, given.a.width)
    if not bend.too_tight:
        return start, None, None
    return _held_start(given, a_on_right), Hold.ROUTING, bend


def _has_room(given: PairInput, start: _Start, spacing: float) -> bool:
    """Whether resampling at `spacing` leaves the centreline a vertex that
    may move, some stretch between its fixed vertices, its ends and their
    direction pins, longer than a spacing, and each track is longer than a
    spacing. Pins close to each other, or to its ends, may leave no room."""
    arc = arc_lengths(start.line)[start.fixed]
    tracks = (given.a.points, given.b.points)
    return bool((np.diff(arc) > spacing).any()) and all(
        arc_lengths(track)[-1] > spacing for track in tracks
    )


def _moving_start(given: PairInput, spacing: float, a_on_right: bool) -> _Start:
    """The centreline beside track a, with a vertex across from each pin of
    track b where that puts b through the pin at the section's gap, fixed
    at the ends and pins of either track and at the direction pins beside
    each."""
    width = given.a.width
    b = given.b.points
    on_b = np.flatnonzero(given.b_pinned[1:-1]) + 1
    # Where track a lies across from each pin of b, at the section's gap.
    centre_b = _line(b, _Gaps.uniform(given.gap), width, not a_on_right)
    track, across = with_nearest_vertices(
        Track(given.a.points, given.a_pinned),
        2 * centre_b[on_b] - b[on_b],
        VERTEX_SNAP,
    )
    last = len(track.points) - 1
    own_a = track.pinned.copy()
    own_a[[0, last]] = True
    # One across from an end is the end, which holds b's own end.
    inner = [(k, j) for k, j in zip(across, on_b.tolist(), strict=True) if 0 < k < last]
    across_b = np.array([k for k, _ in inner], dtype=np.intp)
    pins_b = np.array([j for _, j in inner], dtype=np.intp)
    pins = own_a.copy()
    pins[across_b] = True
    a, fixed, kept = _pin_directions(track.points, pins, spacing)
    gaps = _Gaps.of(a, b, given.gap, width)
    line = _line(a, gaps, width, a_on_right)
    from_b = _line(b, gaps, width, not a_on_right)
    b_only = ~own_a[across_b]
    line[kept[across_b[b_only]]] = from_b[pins_b[b_only]]
    at_b = np.concatenate([[0], kept[across_b], [len(line) - 1]])
    own_b = b[np.concatenate([[0], pins_b, [len(b) - 1]])]
    held = _Pins(line[kept[own_a]], track.points[own_a], line[at_b], own_b)
    return _Start(fixed, held, gaps, line)


def _held_start(given: PairInput, a_on_right: bool) -> _Start:
    a = given.a.points
    gaps = _Gaps.uniform(given.gap)
    fixed = np.ones(len(a), dtype=bool)
    line = _line(a, gaps, given.a.width, a_on_right)
    ends = given.b.points[[0, -1]]
    return _Start(fixed, _Pins(line, a, line[[0, -1]], ends), gaps, line)


def _line(a: FloatArray, gaps: _Gaps, width: float, a_on_right: bool) -> FloatArray:
    """The centreline beside track `a`."""
    right, left = offset_pair(a, gaps.at(a), width)
    return left if a_on_right else right


def _offset(
    given: PairInput,
    a_on_right: bool,
    pins: _Pins,
    held: bool,
    gaps: FloatArray,
    points: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """The tracks along the centreline at `points`, each vertex at its
    gap in `gaps`, through the points it holds (`pins`)."""
    if held:
        return given.a.points, given.b.points
    right, left = offset_pair(points, gaps, given.a.width)
    a, b = (right, left) if a_on_right else (left, right)
    pins.keep(points, a, b)
    return a, b


def _fixed_ends(count: int) -> BoolArray:
    """A mask of a polyline's `count` vertices fixing its two ends."""
    mask = np.zeros(count, dtype=bool)
    mask[[0, -1]] = True
    return mask


def _pin_directions(
    line: FloatArray, pins: BoolArray, spacing: float
) -> tuple[FloatArray, BoolArray, IntArray]:
    """`line` with its `pins` fixed, and a direction pin on each edge
    leaving a pin, so that the pair keeps its direction through the pin;
    and the index of each vertex of `line` in the result.

    A direction pin is inserted a `spacing` along the edge where that
    leaves at least a spacing of the edge beyond it, and beyond the
    direction pin from its other end if that is a pin too; on a shorter
    edge, such as a chamfer, the edge's far vertex is the direction pin, so
    that no sliver edge is left.
    """
    points: list[FloatArray] = [line[0]]
    fixed = [bool(pins[0])]
    kept = [0]
    for k in range(len(line) - 1):
        start, end = line[k], line[k + 1]
        ends = (bool(pins[k]), bool(pins[k + 1]))
        length = float(np.hypot(*(end - start)))
        room = length >= (STUB_SPACINGS + ends.count(True) - 1) * spacing
        inserted = [
            at + (toward - at) * (spacing / length)
            for at, toward, pin in ((start, end, ends[0]), (end, start, ends[1]))
            if pin and room
        ]
        points += inserted
        fixed += [True] * len(inserted)
        if ends[1] and not room:
            fixed[-1] = True
        points.append(end)
        fixed.append(ends[1] or (ends[0] and not room))
        kept.append(len(points) - 1)
    return np.array(points), np.array(fixed), np.array(kept, dtype=np.intp)
