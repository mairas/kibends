"""Coupled sections of differential pairs as blocks of the optimizer.

A coupled section moves as its centreline: a chain as wide as the pair's
outer extent, two track widths plus the gap, that keeps the larger of the
two nets' clearances, in a static field from
`obstacles.build_pair_obstacles`. The bend-radius hinge of
`energy.bend_radius_energy` keeps its inner track from folding. Each
track's routing is first merged within a micrometre (`ROUTING_TOLERANCE`),
through its ends and pins, as `simplify` merges a result, so that a jog or
sliver too short to see bends nothing. Its two ends are pinned, and so is
a vertex a spacing in from each, so that the pair meets its fan-outs in the
direction it did. Same-net copper that touches a track between the ends
pins the centreline where that puts the track through the point it
touches, at the section's gap, with a vertex a spacing to either side, so
that the track passes there in the direction it did; the rest of the
section moves. Such a direction pin stays on its pin's own straight,
halfway along it where the straight is shorter, so that a corner beyond
stays free to round. A section is held as it is where its routing, without
its direction pins, already bends tighter than the limit, or where the
turns its fixed vertices keep do. Both tracks are the centreline offset by
half the pitch at the section's routed gap all along, through their own
ends and pins: where an end's own gap differs from it, the track's end edge
takes up the difference. The centreline is a little wider than the pair,
by as far as the outer track's mitred corners reach beyond half the pitch
at the bend-radius limit, so that its clearance covers them.

The other blocks see the two tracks at the centreline's current shape. The
centreline sees every other chain, and every track of another diff pair,
as a track of the diff pair's outer extent would. The section and the chains of its
two nets ignore each other, as chains of one net and the two nets of a diff
pair do (`kibends_core::members`), so its fan-outs leave it freely.
"""

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import NamedTuple

import numpy as np

from kibends import _core
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
    with_nearest_vertices,
)

STUB_SPACINGS = 2
"""An edge leaving a pin this many spacings long or longer gets a direction
pin a spacing along it; an edge between two pins needs one more."""

ROUTING_TOLERANCE = 1e-3
"""mm a track's routing may be simplified by before its section moves:
well above the jogs and slivers KiCad's nanometre grid leaves, well below
the micrometres a pair's end gap may differ from its gap by, which its end
edges take up, and below the fold a short chamfer leaves in the inner track."""

_NOWHERE = _core.CompositeField([])
"""No obstacles, for merging a track's routing."""


class Hold(Enum):
    """Why a coupled section stays as it is."""

    GROUP = "a track of it is in a selected group, held as drawn"
    SHORT = "it is too short between the copper at its ends to move"
    ROUTING = "its routing bends too tightly for the inner track"
    CROWDING = (
        "the copper around it kept it from rounding a corner too tight for "
        "the inner track"
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


def _separation(end: FloatArray, toward: FloatArray, point: FloatArray) -> float:
    """Distance from `point` to the line through `end` and `toward`."""
    step = toward - end
    offset = point - end
    cross = step[0] * offset[1] - step[1] * offset[0]
    return abs(float(cross)) / math.hypot(step[0], step[1])


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
    limit: float
    """The centreline's smallest bend radius, mm."""
    uneven: FloatArray
    """The direction pins bounding an end edge whose own gap differs from
    the section's, so that track b's end edge, from the end to the pin,
    takes up the difference."""
    core: _core.PairOffset
    """Its tracks as the offset of the centreline, as the Rust energy takes
    them (`kibends_core::pair_blocks`)."""

    @property
    def nets(self) -> frozenset[str]:
        """Its two nets, whose chains it ignores."""
        return frozenset({self.given.a.net, self.given.b.net})

    @property
    def held(self) -> bool:
        return self.hold is not None

    def offset(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Tracks a and b along the centreline at `points`, through the
        section's own ends and pins."""
        return self.core.offset(_points(points))

    def offset_pullback(
        self, points: FloatArray, at_a: FloatArray, at_b: FloatArray
    ) -> FloatArray:
        """The gradient of a function of `offset(points)`, `at_a` and `at_b`
        at the vertices of tracks a and b, taken back to the centreline's
        vertices. The tracks' ends and pins are the section's own, and a
        held section's tracks are its own throughout, so neither depends on
        the centreline."""
        return self.core.offset_pullback(_points(points), _points(at_a), _points(at_b))

    def bend(self, points: FloatArray) -> BendCheck:
        """The centreline at `points` against the bend-radius limit."""
        return check_bend_radius(points, self.given.gap, self.given.a.width)

    def direction_pins(self, points: FloatArray) -> BoolArray:
        """Which vertices of the centreline at `points` are its direction
        pins, which stay where they started."""
        return _at(points, self.directions).any(axis=1)

    def mergeable_pins(self, points: FloatArray) -> BoolArray:
        """The direction pins of the centreline at `points` that
        simplification may merge across: all but those bounding an uneven
        end's edge (`uneven`), where merging would slant track b along the
        merged edge."""
        return self.direction_pins(points) & ~_at(points, self.uneven).any(axis=1)


def pair_block(given: PairInput, spacing: float) -> PairBlock:
    """The block of a coupled section, its direction pins a `spacing` from
    its ends and pins, or at the end of a shorter edge."""
    width = given.a.width
    a_on_right = centreline(given.a.points, given.b.points, width).a_on_right
    (fixed, pins, line), hold, tight_start = _start(given, spacing, a_on_right)
    held = hold is not None
    limit = bend_radius_limit(given.gap, width)
    half_pitch = (given.gap + width) / 2
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
    core = _core.PairOffset(
        given.gap,
        width,
        a_on_right,
        held,
        _points(pins.a_at),
        _points(pins.a),
        _points(pins.b_at),
        _points(pins.b),
        _points(given.a.points),
        _points(given.b.points),
    )
    start_a, start_b = core.offset(_points(line))
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
    directions = fixed & ~held_a & ~held_b
    return PairBlock(
        given,
        chain,
        (track_a, track_b),
        facing,
        directions=line[directions],
        pins=pins,
        a_on_right=a_on_right,
        hold=hold,
        tight_start=tight_start,
        limit=limit,
        uneven=line[[1, -2]][directions[[1, -2]] & _uneven_ends(given)],
        core=core,
    )


def _uneven_ends(given: PairInput) -> BoolArray:
    """Whether the gap at each end of the section, across track a's end
    edges, differs from its routed gap."""
    a, b = given.a.points, given.b.points
    width = given.a.width
    start = _separation(a[0], a[1], b[0]) - width
    end = _separation(a[-1], a[-2], b[-1]) - width
    return np.abs(np.array([start, end]) - given.gap) > INSIDE_TOLERANCE


class _Start(NamedTuple):
    """A section's starting centreline as the run takes it."""

    fixed: BoolArray
    pins: _Pins
    """Its ends and pins; the rest of its fixed vertices are direction
    pins."""
    line: FloatArray


def _start(
    given: PairInput, spacing: float, a_on_right: bool
) -> tuple[_Start, Hold | None, BendCheck | None]:
    """The section's starting centreline, why it is held if it is, and its
    starting bend if that is why."""
    if given.hold is not None:
        return _held_start(given, a_on_right), given.hold, None
    routed = _routed(given)
    start = _moving_start(routed, spacing, a_on_right)
    if not _has_room(routed, start, spacing):
        return _held_start(given, a_on_right), Hold.SHORT, None
    held_a, held_b = start.pins.held(start.line)
    # Its routing as drawn, whose straights its direction pins lie on, and
    # the turns its fixed vertices keep, which nothing rounds.
    routing = start.line[~start.fixed | held_a | held_b]
    gap, width = given.gap, given.a.width
    bend = min(
        check_bend_radius(routing, gap, width),
        check_bend_radius(start.line, gap, width, start.fixed),
        key=lambda check: check.radius,
    )
    if not bend.too_tight:
        return start, None, None
    return _held_start(given, a_on_right), Hold.ROUTING, bend


def _routed(given: PairInput) -> PairInput:
    """The section with each track's routing simplified within
    `ROUTING_TOLERANCE`, through the track's ends and pins: a jog or sliver
    too short to see is no bend. A track merges as it is, not its offset,
    whose mitres may fold across a corner. Its end edges merge only along
    their lines, so that the section keeps its ends' directions and gaps."""
    simplified: list[tuple[ChainCentreline, BoolArray]] = []
    for track, pinned in ((given.a, given.a_pinned), (given.b, given.b_pinned)):
        fixed = pinned | _fixed_ends(len(pinned))
        tolerances = np.full(len(pinned), ROUTING_TOLERANCE)
        tolerances[[1, -2]] = INSIDE_TOLERANCE
        kept = _core.kept_vertices(track.points, fixed, _NOWHERE, tolerances)
        simplified.append((replace(track, points=track.points[kept]), pinned[kept]))
    (a, a_pinned), (b, b_pinned) = simplified
    return replace(given, a=a, b=b, a_pinned=a_pinned, b_pinned=b_pinned)


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
    centre_b = _line(b, given.gap, width, not a_on_right)
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
    line = _line(a, given.gap, width, a_on_right)
    from_b = _line(b, given.gap, width, not a_on_right)
    b_only = ~own_a[across_b]
    line[kept[across_b[b_only]]] = from_b[pins_b[b_only]]
    at_b = np.concatenate([[0], kept[across_b], [len(line) - 1]])
    own_b = b[np.concatenate([[0], pins_b, [len(b) - 1]])]
    held = _Pins(line[kept[own_a]], track.points[own_a], line[at_b], own_b)
    return _Start(fixed, held, line)


def _held_start(given: PairInput, a_on_right: bool) -> _Start:
    a = given.a.points
    fixed = np.ones(len(a), dtype=bool)
    line = _line(a, given.gap, given.a.width, a_on_right)
    ends = given.b.points[[0, -1]]
    return _Start(fixed, _Pins(line, a, line[[0, -1]], ends), line)


def _line(a: FloatArray, gap: float, width: float, a_on_right: bool) -> FloatArray:
    """The centreline beside track `a`, at `gap`."""
    right, left = offset_pair(a, gap, width)
    return left if a_on_right else right


def _points(points: FloatArray) -> FloatArray:
    """`points` as an (n, 2) array of floats."""
    return np.asarray(points, dtype=float).reshape(-1, 2)


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
    direction pin from its other end if that is a pin too. On a shorter
    edge, such as one ending at a corner close to the pin, it is inserted
    halfway, so that it stays on the pin's own straight and the corner
    stays free to round; an edge between two pins too short for either
    gets none, as both its ends are fixed.
    """
    points: list[FloatArray] = [line[0]]
    fixed = [bool(pins[0])]
    kept = [0]
    for k in range(len(line) - 1):
        start, end = line[k], line[k + 1]
        ends = (bool(pins[k]), bool(pins[k + 1]))
        length = math.hypot(*(end - start))
        if length >= (STUB_SPACINGS + ends.count(True) - 1) * spacing:
            reach = spacing
        else:
            reach = 0.0 if all(ends) else length / 2
        inserted = [
            at + (toward - at) * (reach / length)
            for at, toward, pin in ((start, end, ends[0]), (end, start, ends[1]))
            if pin and reach > 0.0
        ]
        points += inserted
        fixed += [True] * len(inserted)
        points.append(end)
        fixed.append(ends[1])
        kept.append(len(points) - 1)
    return np.array(points), np.array(fixed), np.array(kept, dtype=np.intp)
