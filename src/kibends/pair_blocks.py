"""Coupled sections of differential pairs as blocks of the optimizer.

A coupled section moves as its centreline: a chain as wide as the pair's
outer extent, two track widths plus the gap, that keeps the larger of the
two nets' clearances, in a static field from
`obstacles.build_pair_obstacles`. The bend-radius hinge of
`energy.bend_radius_energy` keeps its inner track from folding; a section
whose own routing already bends tighter is held as it is. Its two
ends are pinned, and so is a vertex a spacing in from each, so that the
pair meets its fan-outs in the direction it did. Both tracks are the
centreline offset by half the pitch, ending where the section's tracks did;
where an end's own gap differs from the pair's, the gap tapers between the
two beyond that end's direction pin. The centreline is a little wider than
the pair, by as far as the outer track's mitred corners reach beyond half
the pitch at the bend-radius limit, so that its clearance covers them.

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
from kibends.energy import BoolArray, FloatArray, ObstacleField, arc_lengths
from kibends.neighbours import ChainInput
from kibends.obstacles import ChainCentreline
from kibends.pairs import (
    BendCheck,
    bend_radius_limit,
    centreline,
    check_bend_radius,
    offset_pair,
    offset_pair_pullback,
)

STUB_SPACINGS = 2
"""An end edge this many spacings long or longer gets a direction pin a
spacing in; one edge alone needs one more, for the pin at its other end."""

TAPER_SLOPE = 0.01
"""Where an end's gap differs from the pair's, each track turns by at most
this much, as a tangent, to take up the difference beyond the end's
direction pin; less where the section is too short."""

MIN_TAPER = 1e-6
"""mm: the shortest taper, to keep it finite where the gaps agree."""


class Hold(Enum):
    """Why a coupled section stays as it is."""

    COPPER = "same-net copper touches it between its ends"
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
    """Where same-net copper touches each track. `pairs.find_pair` ends a
    section at every pin where both tracks run straight; one left with a
    pin between its ends stays as it is."""
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
        """Tracks a and b along the centreline at `points`, with the
        section's own end points."""
        return _offset(
            self.given, self.a_on_right, self.held, self.gaps.at(points), points
        )

    def offset_pullback(
        self, points: FloatArray, at_a: FloatArray, at_b: FloatArray
    ) -> FloatArray:
        """The gradient of a function of `offset(points)`, `at_a` and `at_b`
        at the vertices of tracks a and b, taken back to the centreline's
        vertices. The tracks' end points are the section's own, and a held
        section's tracks are its own throughout, so neither depends on the
        centreline."""
        if self.held:
            return np.zeros_like(points)
        right, left = (at_a, at_b) if self.a_on_right else (at_b, at_a)
        right, left = right.copy(), left.copy()
        right[[0, -1]] = left[[0, -1]] = 0.0
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
        return _offset(self.given, self.a_on_right, self.held, at, kept)

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
        pins = self.chain.centreline.points[[1, -2]]
        at = (points[:, None] == pins[None]).all(axis=2).any(axis=1)
        at[[0, -1]] = False
        return at


def pair_block(given: PairInput, spacing: float) -> PairBlock:
    """The block of a coupled section, its direction pins a `spacing` in
    from its ends, or at the end of a shorter end edge."""
    width = given.a.width
    a_on_right = centreline(given.a.points, given.b.points, width).a_on_right
    (fixed, gaps, line), hold, tight_start = _start(given, spacing, a_on_right)
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
    start_a, start_b = _offset(given, a_on_right, held, gaps.at(line), line)
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
    return PairBlock(
        given,
        chain,
        (track_a, track_b),
        facing,
        a_on_right=a_on_right,
        hold=hold,
        tight_start=tight_start,
        gaps=gaps,
        limit=limit,
    )


class _Start(NamedTuple):
    """A section's starting centreline as the run takes it."""

    fixed: BoolArray
    gaps: _Gaps
    line: FloatArray


def _start(
    given: PairInput, spacing: float, a_on_right: bool
) -> tuple[_Start, Hold | None, BendCheck | None]:
    """The section's starting centreline, why it is held if it is, and its
    starting bend if that is why."""
    pinned = given.a_pinned[1:-1].any() or given.b_pinned[1:-1].any()
    hold = given.hold or (Hold.COPPER if pinned else None)
    if hold is not None:
        return _held_start(given, a_on_right), hold, None
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
    spacing. A section split at a pin close to another, or to its end, may
    have no room."""
    arc = arc_lengths(start.line)[start.fixed]
    tracks = (given.a.points, given.b.points)
    return bool((np.diff(arc) > spacing).any()) and all(
        arc_lengths(track)[-1] > spacing for track in tracks
    )


def _moving_start(given: PairInput, spacing: float, a_on_right: bool) -> _Start:
    a, fixed = _pin_directions(given.a.points, spacing)
    gaps = _Gaps.of(a, given.b.points, given.gap, given.a.width)
    return _Start(fixed, gaps, _line(a, gaps, given.a.width, a_on_right))


def _held_start(given: PairInput, a_on_right: bool) -> _Start:
    a = given.a.points
    gaps = _Gaps.uniform(given.gap)
    fixed = np.ones(len(a), dtype=bool)
    return _Start(fixed, gaps, _line(a, gaps, given.a.width, a_on_right))


def _line(a: FloatArray, gaps: _Gaps, width: float, a_on_right: bool) -> FloatArray:
    """The centreline beside track `a`."""
    right, left = offset_pair(a, gaps.at(a), width)
    return left if a_on_right else right


def _offset(
    given: PairInput,
    a_on_right: bool,
    held: bool,
    gaps: FloatArray,
    points: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """The tracks along the centreline at `points`, each vertex at its
    gap in `gaps`."""
    if held:
        return given.a.points, given.b.points
    right, left = offset_pair(points, gaps, given.a.width)
    a, b = (right, left) if a_on_right else (left, right)
    a[[0, -1]] = given.a.points[[0, -1]]
    b[[0, -1]] = given.b.points[[0, -1]]
    return a, b


def _fixed_ends(count: int) -> BoolArray:
    """A mask of a polyline's `count` vertices fixing its two ends."""
    mask = np.zeros(count, dtype=bool)
    mask[[0, -1]] = True
    return mask


def _pin_directions(line: FloatArray, spacing: float) -> tuple[FloatArray, BoolArray]:
    """`line` with its ends and their direction pins, its second and
    second-to-last vertices, pinned.

    A pin is inserted a `spacing` in from the end where that leaves at
    least a spacing of the end edge beyond it; on a shorter end edge, such
    as a chamfer, the edge's far vertex is the pin, so that no sliver edge
    is left.
    """
    single = len(line) == 2
    room = (STUB_SPACINGS + 1 if single else STUB_SPACINGS) * spacing
    first = [_along(line[0], line[1], spacing)] if _length(line[:2]) >= room else []
    last = [_along(line[-1], line[-2], spacing)] if _length(line[-2:]) >= room else []
    points = np.concatenate([line[:1], *first, line[1:-1], *last[::-1], line[-1:]])
    fixed = np.zeros(len(points), dtype=bool)
    fixed[[0, 1, -2, -1]] = True
    return points, fixed


def _length(edge: FloatArray) -> float:
    step = edge[1] - edge[0]
    return float(np.hypot(step[0], step[1]))


def _along(start: FloatArray, toward: FloatArray, spacing: float) -> FloatArray:
    step = toward - start
    return (start + step * (spacing / _length(np.stack([start, toward]))))[None]
