"""Differential pairs: pairing nets by name, finding where two chains run
coupled, and the geometry of a pair's centreline and its two tracks.

Two tracks run coupled wherever they run parallel at about the pair's
pitch, however short the stretch, as in KiCad's coupling check; either may
be drawn in any number of segments.

Each coupled stretch is one section, whatever same-net copper pins either
track between its ends: the section's centreline is fixed where a pin
needs it (`pair_blocks`), and a section left with no vertex to move is
held as routed (`pair_blocks.Hold.SHORT`), without a warning.

Units are mm floats in KiCad's board frame, with y pointing down.
"""

import math
from collections.abc import Collection
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import NamedTuple

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import LineString

from kibends import _core
from kibends.chain import Chain
from kibends.energy import (
    MAX_SMOOTH_TURN,
    BoolArray,
    FloatArray,
    IntArray,
)

GAP_TOLERANCE = 0.1
"""Fraction of the gap by which the tracks' spacing may differ from the pair's
pitch and still count as coupled, on top of the chord tolerance. Hand-routed
pairs keep their gap to the nanometre; this absorbs uneven routing."""

MAX_GAP_TO_WIDTH = 4.0
"""Tracks further apart than this many track widths are not coupled. Real
pairs use a gap of about one to three widths."""

VERTEX_SNAP = 1e-5
"""mm: a coupled section's end ends at a chain vertex this close or within the
chord tolerance, whichever is larger, so that neither nanometre rounding of
45° routing nor tracks simplified one by one leave a sliver segment."""

REACHED_PITCHES = 1.0
"""A chain end this many pitches or less along the chain from a coupled
section's end counts as reached by it (`reach`). A section ends at an
in-band vertex, and near a 45° corner the nearest such vertex can lie up to
about a pitch short of a stub's junction."""

SAMPLES_PER_WIDTH = 4
"""Both tracks are sampled at this many points per track width."""

SQUARE_SLACK = 1e-9
"""mm: a point or projection this close to an edge's end counts as at it, so
that a point on a vertex lies on both edges there and edges that end square
across each other still face each other."""

MITER_FLOOR = 1.0 + math.cos(MAX_SMOOTH_TURN)
"""The least 1 + cos(turn) a miter divides by: that of MAX_SMOOTH_TURN, past
which `energy` continues the curvature linearly. Past it a miter shrinks
back to nothing at a fold instead of growing without bound: a trial step
that folds a centreline offsets it to finite tracks, and the bend-radius
hinge and the self check keep the optimizer away from it."""


class PairError(ValueError):
    """Two chains cannot be treated as one differential pair."""


complementary_net = _core.complementary_net
"""The name the other net of a diff pair would have, matching KiCad's
router (`MatchDpSuffix`), or None (`kibends_core::pairs`). A net is of a
diff pair only when that net exists (`in_diff_pair`)."""


def in_diff_pair(net: str, nets: Collection[str]) -> bool:
    """Whether `net` is a net of a diff pair, as KiCad treats it
    (`DRC_ENGINE::IsNetADiffPair`): its complementary net is among `nets`,
    the board's."""
    other = complementary_net(net)
    return other is not None and other in nets


def nets_pair(a: str, b: str) -> bool:
    return complementary_net(a) == b


diff_pair_nets = _core.diff_pair_nets
"""`net` and the other net of its diff pair, if it has one. Callers match
it only against nets of selected chains, so a complement that is not on
the board matches nothing, as `in_diff_pair` would have it."""


@dataclass(frozen=True)
class Track:
    """A chain's points with vertices inserted where coupled sections end.

    Inserted vertices are not pinned.
    """

    points: FloatArray
    pinned: BoolArray

    def between(self, start: int, end: int) -> FloatArray:
        """A copy of the points from vertex `start` to vertex `end`, inclusive."""
        return self.points[start : end + 1].copy()


class Section(NamedTuple):
    """One coupled stretch: inclusive vertex indices into `Pair.a.points`
    and `Pair.b.points`, with start < end on each track. `a_start` lies
    opposite `b_start` unless `b_reversed`, when b runs against a and
    `a_start` lies opposite `b_end`. `gap` is the section's median
    edge-to-edge gap."""

    a_start: int
    a_end: int
    b_start: int
    b_end: int
    b_reversed: bool
    gap: float


class Piece(NamedTuple):
    """An uncoupled stretch of one track: inclusive vertex indices into its
    `Track`, and the indices into `Pair.sections` of the sections it meets at
    its start (`after`) and end (`before`), None at a chain end. A section
    that reaches a chain end leaves a one-vertex piece there."""

    start: int
    end: int
    after: int | None
    before: int | None


@dataclass(frozen=True)
class Pair:
    """Two chains of a differential pair and their coupled sections, ordered
    along `a`. Between and around the sections the tracks are uncoupled.

    `gap` is the median edge-to-edge gap over all coupled sections together.
    """

    a: Track
    b: Track
    sections: tuple[Section, ...]
    width: float
    gap: float

    def coupled(self, section: Section) -> tuple[FloatArray, FloatArray]:
        return (
            self.a.between(section.a_start, section.a_end),
            self.b.between(section.b_start, section.b_end),
        )

    def uncoupled(self) -> tuple[list[Piece], list[Piece]]:
        """The pieces of each track outside the sections, in track order."""
        spans_a = [(s.a_start, s.a_end, k) for k, s in enumerate(self.sections)]
        spans_b = [(s.b_start, s.b_end, k) for k, s in enumerate(self.sections)]
        return (
            _pieces(len(self.a.points), spans_a),
            _pieces(len(self.b.points), spans_b),
        )


def _pieces(count: int, spans: list[tuple[int, int, int]]) -> list[Piece]:
    pieces: list[Piece] = []
    start, after = 0, None
    for first, last, section in sorted(spans):
        pieces.append(Piece(start, first, after, section))
        start, after = last, section
    pieces.append(Piece(start, count - 1, after, None))
    return pieces


def find_pair(a: Chain, b: Chain, chord_tolerance: float) -> Pair | None:
    """Find the coupled sections of two chains on one layer whose nets pair.

    A coupled section is a stretch, however short, where each track runs
    parallel to the other within a band around the pair's pitch, edge
    against facing edge as KiCad's coupling check has it, so that tracks
    that only pass through the band at an angle do not couple.
    `chord_tolerance` widens the band, since tracks simplified on their own
    each stray from the true curve by up to that. The ends of a section lie
    opposite each other, so a track that runs on past the other's fan-out
    gets a vertex there. None when no stretch qualifies.

    Pins between a stretch's ends stay inside its section.
    """
    if not nets_pair(a.net, b.net):
        raise PairError(f"nets {a.net} and {b.net} do not form a differential pair")
    if a.width != b.width:
        raise PairError(
            f"differential pair {a.net}/{b.net} has two track widths: "
            f"{a.width:g} mm and {b.width:g} mm"
        )
    points_a = np.array(a.points, dtype=float)
    points_b = np.array(b.points, dtype=float)
    spans = _coupled_spans(points_a, points_b, a.width, chord_tolerance)
    if not spans:
        return None
    snap = vertex_snap(chord_tolerance)
    # Each span's two ends, in turn, on each track.
    track_a, index_a = _with_vertices(
        points_a, np.array(a.pinned), [at for span in spans for at in span.a], snap
    )
    track_b, index_b = _with_vertices(
        points_b, np.array(b.pinned), [at for span in spans for at in span.b], snap
    )
    sections: list[Section] = []
    distances: list[FloatArray] = []
    for a_start, a_end, b_first, b_last in zip(
        index_a[::2], index_a[1::2], index_b[::2], index_b[1::2], strict=True
    ):
        b_start, b_end = min(b_first, b_last), max(b_first, b_last)
        section_distances = _centre_distances(
            track_a.points[a_start : a_end + 1],
            track_b.points[b_start : b_end + 1],
            a.width,
        )
        distances.append(section_distances)
        gap = float(np.median(section_distances)) - a.width
        sections.append(Section(a_start, a_end, b_start, b_end, b_first > b_last, gap))
    gap = float(np.median(np.concatenate(distances))) - a.width
    return Pair(track_a, track_b, tuple(sections), a.width, gap)


def vertex_snap(chord_tolerance: float) -> float:
    """How near, mm, a point must lie to a chain vertex to count as on it."""
    return max(VERTEX_SNAP, chord_tolerance)


class Opposite(NamedTuple):
    """An end of one chain of a pair that a coupled section reaches, and
    where that section ends on the other chain."""

    at: float
    """Arc length along the other chain, mm."""
    end: int
    """Which end: 0 for its start, -1 for its end."""


class Reach(NamedTuple):
    """What the coupled sections of a pair reach on one of its chains."""

    spans: list[tuple[float, float]]
    """Each section's arc lengths along the chain, mm."""
    opposite: list[Opposite]
    """The ends of the other chain that a section reaches, opposite this one."""


def reach(pair: Pair) -> tuple[Reach, Reach]:
    """What the sections of `pair` reach on chain a and on chain b.

    A chain end counts as reached when a section ends within `reached` of
    it along the chain."""
    arcs = [_polyline(track.points).arc for track in (pair.a, pair.b)]
    spans: tuple[list[tuple[float, float]], list[tuple[float, float]]] = ([], [])
    opposite: tuple[list[Opposite], list[Opposite]] = ([], [])
    for s in pair.sections:
        spans[0].append((float(arcs[0][s.a_start]), float(arcs[0][s.a_end])))
        spans[1].append((float(arcs[1][s.b_start]), float(arcs[1][s.b_end])))
        # (vertex of a, vertex of b) at each end of the section.
        across = [(s.a_start, s.b_start), (s.a_end, s.b_end)]
        if s.b_reversed:
            across = [(s.a_start, s.b_end), (s.a_end, s.b_start)]
        for vertices in across:
            for side, other in ((0, 1), (1, 0)):
                at = arcs[side][vertices[side]]
                for end, chain_end in ((0, 0.0), (-1, float(arcs[side][-1]))):
                    if abs(at - chain_end) <= reached(pair):
                        there = float(arcs[other][vertices[other]])
                        opposite[other].append(Opposite(there, end))
    return Reach(spans[0], opposite[0]), Reach(spans[1], opposite[1])


def reached(pair: Pair) -> float:
    """How near, mm, a coupled section must end to a point along a chain to
    reach it: `REACHED_PITCHES` pitches."""
    return REACHED_PITCHES * (pair.width + pair.gap)


def between_sections(
    at: float, spans: list[tuple[float, float]], tolerance: float
) -> bool:
    """Whether arc length `at` lies between two of the `spans`, outside each,
    within `tolerance`."""
    return (
        any(end <= at + tolerance for _, end in spans)
        and any(start >= at - tolerance for start, _ in spans)
        and not any(start + tolerance < at < end - tolerance for start, end in spans)
    )


def cut_chain(chain: Chain, at: list[float], snap: float) -> list[Chain]:
    """`chain` cut into parts, in order along it, at each arc length of `at`
    more than `snap` from its ends, at a vertex pinned there: one within
    `snap` of the arc length, or else one inserted."""
    track, cuts = _with_vertices(
        np.array(chain.points, dtype=float), np.array(chain.pinned), at, snap
    )
    last = len(track.points) - 1
    inner = sorted({k for k in cuts if 0 < k < last})
    if not inner:
        return [chain]
    points = [(float(x), float(y)) for x, y in track.points]
    pinned = [bool(p) or k in inner for k, p in enumerate(track.pinned)]
    return [
        replace(
            chain,
            points=tuple(points[start : end + 1]),
            pinned=tuple(pinned[start : end + 1]),
        )
        for start, end in pairwise([0, *inner, last])
    ]


class _SectionEnds(NamedTuple):
    """Arc lengths of a section's two ends along each chain; `a[k]` lies
    opposite `b[k]`."""

    a: tuple[float, float]
    b: tuple[float, float]


class _Polyline(NamedTuple):
    starts: FloatArray
    directions: FloatArray
    lengths: FloatArray
    arc: FloatArray


def _polyline(points: FloatArray) -> _Polyline:
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(lengths)])
    return _Polyline(points[:-1], delta / lengths[:, None], lengths, arc)


def _cross(u: FloatArray, v: FloatArray) -> FloatArray:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _point_at(points: FloatArray, distances: FloatArray) -> FloatArray:
    arc = _polyline(points).arc
    return np.stack(
        [
            _core.interp(distances, arc, points[:, 0]),
            _core.interp(distances, arc, points[:, 1]),
        ],
        axis=-1,
    )


def _coupled_spans(
    a: FloatArray, b: FloatArray, width: float, chord_tolerance: float
) -> list[_SectionEnds]:
    """Sample both tracks, measure each sample's distance to the other track,
    and keep the runs of samples in the band around the dominant pitch whose
    edge runs parallel to the facing one (`_parallel`), however short.

    Runs on the two tracks belong to one section when either maps onto the
    other: at a corner the outer track leaves the band around its vertex, but
    the inner track runs on unbroken and joins the outer track's two runs.
    """
    step = width / SAMPLES_PER_WIDTH
    at_a, at_b = _sample_arcs(a, step), _sample_arcs(b, step)
    near_a = _nearest(_point_at(a, at_a), b)
    near_b = _nearest(_point_at(b, at_b), a)
    pitch = _dominant_pitch(
        np.concatenate([near_a.distance, near_b.distance]), width, chord_tolerance
    )
    if pitch is None:
        return []
    band = _band(pitch, width, chord_tolerance)
    runs_a = _runs(
        (np.abs(near_a.distance - pitch) <= band)
        & _parallel(a, at_a, b, near_a, pitch, band)
    )
    runs_b = _runs(
        (np.abs(near_b.distance - pitch) <= band)
        & _parallel(b, at_b, a, near_b, pitch, band)
    )
    reach_a, reach_b = at_a[runs_a], at_b[runs_b]
    links = _overlapping(_mapped(runs_a, near_a.along), reach_b, step)

    vertex_a, vertex_b = _nearest(a, b), _nearest(b, a)
    # In-band vertices of both tracks as (arc length along a, along b).
    candidates = np.concatenate(
        [
            np.stack([_polyline(a).arc, vertex_a.along], axis=1)[
                np.abs(vertex_a.distance - pitch) <= band
            ],
            np.stack([vertex_b.along, _polyline(b).arc], axis=1)[
                np.abs(vertex_b.distance - pitch) <= band
            ],
        ]
    )
    spans = [
        _ends(
            a,
            b,
            (float(reach_a[rows].min()), float(reach_a[rows].max())),
            (float(reach_b[columns].min()), float(reach_b[columns].max())),
            candidates,
            pitch,
            step,
        )
        for rows, columns in _components(links)
    ]
    # `_ends` places an end only to within a step, or to a vertex within a
    # snap; a span no longer than that on either track has no extent. A span
    # runs along a; b may run either way.
    resolution = max(step, vertex_snap(chord_tolerance))
    return [
        span
        for span in sorted(spans, key=lambda s: s.a[0])
        if min(span.a[1] - span.a[0], abs(span.b[1] - span.b[0])) > resolution
    ]


def _sample_arcs(points: FloatArray, step: float) -> FloatArray:
    """Arc lengths of evenly spaced samples, at most `step` apart."""
    length = float(_polyline(points).arc[-1])
    return np.linspace(0.0, length, math.ceil(length / step) + 1)


def _band(pitch: float, width: float, chord_tolerance: float) -> float:
    return GAP_TOLERANCE * (pitch - width) + chord_tolerance


def _dominant_pitch(
    distances: FloatArray, width: float, chord_tolerance: float
) -> float | None:
    """The spacing shared by the most samples, so that a shorter stretch at
    another spacing, such as a fan-out, does not set it. A longer one does:
    without the design rules there is no telling which spacing is the pair's.
    None when no sample is close enough to couple."""
    gaps = distances - width
    usable = np.sort(distances[(gaps > 0.0) & (gaps <= MAX_GAP_TO_WIDTH * width)])
    if len(usable) == 0:
        return None
    window = 2 * (GAP_TOLERANCE * (usable - width) + chord_tolerance)
    upper = np.searchsorted(usable, usable + window, side="right")
    best = int(np.argmax(upper - np.arange(len(usable))))
    return float(np.median(usable[best : upper[best]]))


def _parallel(
    line: FloatArray,
    at: FloatArray,
    other: FloatArray,
    near: _Nearest,
    pitch: float,
    band: float,
) -> BoolArray:
    """Whether the edge of `line` at each arc length of `at` and the edge of
    `other` that its nearest point `near` lies on run parallel, as KiCad's
    coupling check has it (`SEG::ApproxParallel` over a common projection):
    over the stretch where the two edges face each other, the spacing stays
    within `band` of `pitch`. Edges that only meet end to end do not face
    each other. A point on a vertex lies on both edges there, and either
    one will do."""
    own, across = _polyline(line), _polyline(other)
    edge = np.clip(np.searchsorted(own.arc, at, "right") - 1, 0, len(own.lengths) - 1)
    result = np.zeros(len(at), dtype=bool)
    for mine in _edges_at(own, at - own.arc[edge], edge):
        for theirs in _edges_at(across, near.along - across.arc[near.edge], near.edge):
            result |= _facing_in_band(line, mine, across, theirs, pitch, band)
    return result


def _edges_at(line: _Polyline, t: FloatArray, edge: IntArray) -> list[IntArray]:
    """The edge each point lies on, given `t` along it, and the neighbouring
    edge for a point on the vertex they share."""
    last = len(line.lengths) - 1
    before = (t <= SQUARE_SLACK) & (edge > 0)
    after = (t >= line.lengths[edge] - SQUARE_SLACK) & (edge < last)
    return [edge, np.where(before, edge - 1, np.where(after, edge + 1, edge))]


def _facing_in_band(
    line: FloatArray,
    edge: IntArray,
    across: _Polyline,
    facing: IntArray,
    pitch: float,
    band: float,
) -> BoolArray:
    """Whether each `edge` of `line` overlaps edge `facing` of the other
    polyline square across, and the spacing stays within `band` of `pitch`
    at both ends of the overlap, so all along it."""
    start, end = line[edge], line[edge + 1]
    origin, direction = across.starts[facing], across.directions[facing]
    length = across.lengths[facing]
    # Where the edge's ends project along `facing`, from its start.
    t_start = np.sum((start - origin) * direction, axis=1)
    rate = np.sum((end - origin) * direction, axis=1) - t_start
    perpendicular = np.abs(rate) <= SQUARE_SLACK
    rate = np.where(perpendicular, 1.0, rate)
    # The fractions of the edge whose projection lies inside `facing`.
    enter, leave = -t_start / rate, (length - t_start) / rate
    low = np.where(perpendicular, 0.0, np.maximum(np.minimum(enter, leave), 0.0))
    high = np.where(perpendicular, 1.0, np.minimum(np.maximum(enter, leave), 1.0))
    within = (t_start >= -SQUARE_SLACK) & (t_start <= length + SQUARE_SLACK)
    overlaps = np.where(perpendicular, within, low <= high)
    in_band = [
        np.abs(
            np.abs(_cross(direction, start + f[:, None] * (end - start) - origin))
            - pitch
        )
        <= band
        for f in (low, high)
    ]
    return overlaps & in_band[0] & in_band[1]


def _runs(inside: BoolArray) -> IntArray:
    """Inclusive (first, last) index pairs of the True runs, one row each."""
    edges = np.diff(np.concatenate([[0], inside.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    return np.stack([starts, np.flatnonzero(edges == -1) - 1], axis=1)


def _mapped(runs: IntArray, along: FloatArray) -> FloatArray:
    """The interval of the other track that each run's samples are nearest to."""
    return np.array(
        [(along[i : j + 1].min(), along[i : j + 1].max()) for i, j in runs.tolist()]
    ).reshape(-1, 2)


def _overlapping(first: FloatArray, second: FloatArray, slack: float) -> BoolArray:
    """Whether interval i of `first` overlaps interval j of `second`."""
    return (first[:, None, 0] <= second[None, :, 1] + slack) & (
        first[:, None, 1] >= second[None, :, 0] - slack
    )


def _components(links: BoolArray) -> list[tuple[IntArray, IntArray]]:
    """Connected groups of a bipartite graph given by its biadjacency matrix,
    as (rows, columns); a group without both is dropped."""
    rows = links.shape[0]
    parent = list(range(rows + links.shape[1]))

    def root(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, j in np.argwhere(links).tolist():
        parent[root(i)] = root(rows + j)
    label = np.array([root(k) for k in range(len(parent))])
    groups: list[tuple[IntArray, IntArray]] = []
    for g in np.unique(label).tolist():
        members = np.flatnonzero(label == g)
        in_rows = members[members < rows]
        in_columns = members[members >= rows] - rows
        if len(in_rows) and len(in_columns):
            groups.append((in_rows, in_columns))
    return groups


def _ends(
    a: FloatArray,
    b: FloatArray,
    hull_a: tuple[float, float],
    hull_b: tuple[float, float],
    candidates: FloatArray,
    pitch: float,
    step: float,
) -> _SectionEnds:
    """Place a section's ends opposite each other.

    Sampled runs overshoot a fan-out a little on the track that runs on, and
    end up to a sample `step` short of where a track leaves the band, so each
    end moves to the outermost in-band vertex of either track from a step
    outward to a pitch inward, given as `candidates` (arc length along a,
    along b), or else stays at the inner of the two run ends. Neither end
    moves inward past the other, so a run shorter than a pitch keeps its
    ends in order.
    """
    along_a = _nearest(_point_at(b, np.array(hull_b)), a).along
    start = max(hull_a[0], float(along_a.min()))
    end = min(hull_a[1], float(along_a.max()))

    def snap(target: float, inward: float, limit: float) -> tuple[float, float]:
        reach = inward * (candidates[:, 0] - target)
        close = candidates[
            (reach >= -step) & (reach <= min(pitch, inward * (limit - target)))
        ]
        if len(close):
            chosen = close[np.argmin(inward * close[:, 0])]
            return float(chosen[0]), float(chosen[1])
        foot = _nearest(_point_at(a, np.array([target])), b).along
        return target, float(foot[0])

    # The start snaps first and bounds the end; bounding each by the other's
    # unsnapped target would let them cross again.
    first = snap(start, 1.0, end)
    last = snap(end, -1.0, first[0])
    return _SectionEnds((first[0], last[0]), (first[1], last[1]))


def with_nearest_vertices(
    track: Track, targets: FloatArray, snap: float
) -> tuple[Track, list[int]]:
    """`track` with a vertex at its point nearest each of `targets`, unless
    one lies within `snap` of it, and the vertex index of each target."""
    if len(targets) == 0:
        return track, []
    along = _nearest(targets, track.points).along
    return _with_vertices(track.points, track.pinned, along.tolist(), snap)


def _with_vertices(
    points: FloatArray, pinned: BoolArray, distances: list[float], snap: float
) -> tuple[Track, list[int]]:
    """Insert a vertex at each arc length unless one lies within `snap`, and
    return the vertex index of each arc length."""
    for distance in distances:
        arc = _polyline(points).arc
        if np.abs(arc - distance).min() > snap:
            index = int(np.searchsorted(arc, distance))
            vertex = _point_at(points, np.array([distance]))[0]
            points = np.insert(points, index, vertex, axis=0)
            pinned = np.insert(pinned, index, False)
    arc = _polyline(points).arc
    indices = [int(np.argmin(np.abs(arc - d))) for d in distances]
    return Track(points, pinned), indices


class Centreline(NamedTuple):
    """A pair's centreline, and whether track `a` lies to the right of it,
    the first track `offset_pair` returns."""

    points: FloatArray
    a_on_right: bool


def centreline(a: FloatArray, b: FloatArray, width: float) -> Centreline:
    """The centreline of two coupled tracks: `a` offset toward `b` by half
    their median spacing, so it follows `a`'s direction and vertices.

    Built from one track only, because tracks simplified one by one put their
    vertices in different places: midpoints between the two make the
    centreline kink wherever a vertex of one track faces the other. The mitred
    offset keeps `a`'s bends and reproduces a corner of a hand-routed pair
    exactly. `width` sets the spacing of the samples that measure the pitch.
    """
    samples = _point_at(a, _sample_arcs(a, width / SAMPLES_PER_WIDTH))
    half_pitch = float(np.median(_nearest(samples, b).distance)) / 2
    miter = _miters(a)
    toward_b = np.sum(miter * (_nearest(a, b).foot - a), axis=1)
    shift = math.copysign(half_pitch, float(np.median(toward_b)))
    return Centreline(a + shift * miter, a_on_right=shift < 0.0)


def _centre_distances(a: FloatArray, b: FloatArray, width: float) -> FloatArray:
    """Distances from samples along each track to the other track's centre."""
    step = width / SAMPLES_PER_WIDTH
    samples_a = shapely.get_coordinates(shapely.segmentize(LineString(a), step))
    samples_b = shapely.get_coordinates(shapely.segmentize(LineString(b), step))
    return np.concatenate(
        [_nearest(samples_a, b).distance, _nearest(samples_b, a).distance]
    )


class _Nearest(NamedTuple):
    """For each query point: its nearest point on a polyline, the arc length
    there, and the distance to it."""

    foot: FloatArray
    along: FloatArray
    distance: FloatArray
    edge: IntArray
    """The index of the polyline's edge the foot lies on."""


def _nearest(points: FloatArray, polyline: FloatArray) -> _Nearest:
    """Vectorised nearest points; an R-tree over the segments keeps dense
    tracks from costing points times segments."""
    line = _polyline(polyline)
    segments = shapely.linestrings(np.stack([polyline[:-1], polyline[1:]], axis=1))
    tree = STRtree(np.asarray(segments, dtype=object))
    query, segment = tree.query_nearest(shapely.points(points), all_matches=False)
    index = np.empty(len(points), dtype=np.intp)
    index[query] = segment
    projection = np.sum((points - line.starts[index]) * line.directions[index], axis=1)
    t = np.clip(projection, 0.0, line.lengths[index])
    foot = line.starts[index] + t[:, None] * line.directions[index]
    offset = points - foot
    return _Nearest(foot, line.arc[index] + t, np.linalg.norm(offset, axis=1), index)


def offset_pair(
    centreline: FloatArray, gap: float, width: float
) -> tuple[FloatArray, FloatArray]:
    """The two tracks of a pair at ±(gap + width)/2 from `centreline`
    (`kibends_core::pair_blocks::offset_pair`).

    The first track lies to the right of the direction of travel as seen with
    y down, the second to the left. Vertices are mitred, so every offset
    segment is parallel to its centreline segment at exactly half the pitch.
    The centreline must have no repeated points, and neither track folds
    while `check_bend_radius` passes.
    """
    return _core.offset_pair(_floats(centreline), gap, width)


def offset_pair_pullback(
    centreline: FloatArray,
    gap: float,
    width: float,
    right: FloatArray,
    left: FloatArray,
) -> FloatArray:
    """The gradient of a function of `offset_pair`'s two tracks, `right` and
    `left` at each of their vertices, taken back to the centreline's
    vertices (`kibends_core::pair_blocks::offset_pair_pullback`)."""
    return _core.offset_pair_pullback(
        _floats(centreline), gap, width, _floats(right), _floats(left)
    )


def _floats(points: FloatArray) -> FloatArray:
    """`points` as an (n, 2) array of floats."""
    return np.asarray(points, dtype=float).reshape(-1, 2)


def _miters(polyline: FloatArray) -> FloatArray:
    """Per vertex, the shift that moves both adjacent segments one unit to
    the right of travel (y down): the bisector, scaled by 1/cos(turn/2), at
    most as much as at MAX_SMOOTH_TURN (`MITER_FLOOR`)."""
    delta = np.diff(polyline, axis=0)
    direction = delta / np.linalg.norm(delta, axis=1)[:, None]
    normal = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
    before = np.concatenate([normal[:1], normal])
    after = np.concatenate([normal, normal[-1:]])
    scale = np.maximum(1.0 + np.sum(before * after, axis=1), MITER_FLOOR)
    return (before + after) / scale[:, None]


bend_radius_limit = _core.bend_radius_limit
"""The smallest centreline bend radius at which the inner track's inner
edge still has a radius of at least zero."""


class BendCheck(NamedTuple):
    radius: float
    too_tight: bool


def check_bend_radius(
    centreline: FloatArray,
    gap: float,
    width: float,
    turning: BoolArray | None = None,
) -> BendCheck:
    """The centreline's smallest bend radius and whether it is below the
    limit, counting the turns at the vertices `turning` (`bend_radii`)."""
    return BendCheck(*_core.check_bend_radius(centreline, gap, width, turning))


bend_radii = _core.bend_radii
"""Each edge's bend radius, as `energy.edge_bend_radii` defines it, 0
beside a fold and for an edge of no length, where the centreline turns on
the spot (`kibends_core::pairs::bend_radii`). Only the turns and folds at
the vertices `turning` count, every vertex by default."""

too_tight = _core.too_tight
"""Whether each bend radius is below `limit`, beyond round-off."""
