"""Differential pairs: pairing nets by name, finding where two chains run
coupled, and the geometry of a pair's centreline and its two tracks.

Units are mm floats in KiCad's board frame, with y pointing down.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import LineString

from kibends.chain import Chain
from kibends.energy import (
    MAX_SMOOTH_TURN,
    BoolArray,
    FloatArray,
    IntArray,
    edge_bend_radii,
)

_SUFFIX_COMPLEMENTS = {"+": "-", "-": "+", "P": "N", "N": "P"}
_IGNORED_TRAILING = "0123456789_"

GAP_TOLERANCE = 0.1
"""Fraction of the gap by which the tracks' spacing may differ from the pair's
pitch and still count as coupled, on top of the chord tolerance. Hand-routed
pairs keep their gap to the nanometre; this absorbs uneven routing."""

MAX_GAP_TO_WIDTH = 4.0
"""Tracks further apart than this many track widths are not coupled. Real
pairs use a gap of about one to three widths."""

MIN_COUPLED_PITCHES = 5.0
MIN_COUPLED_LENGTH = 1.0
"""mm: a coupled stretch must be at least `MIN_COUPLED_PITCHES` pitches and at
least this long; shorter ones are coincidences in fan-outs, not coupling."""

VERTEX_SNAP = 1e-5
"""mm: a coupled section's end ends at a chain vertex this close or within the
chord tolerance, whichever is larger, so that neither nanometre rounding of
45° routing nor tracks simplified one by one leave a sliver segment."""

SAMPLES_PER_WIDTH = 4
"""Both tracks are sampled at this many points per track width."""

SQUARE_SLACK = 1e-9
"""mm: a foot this close to an edge's end still counts as inside it, so that
a point opposite a straight vertex does not drop out."""

BEND_RADIUS_TOLERANCE = 1e-3
"""Relative: a dense polyline drawn at the radius limit measures a hair below
it, R·cos(Δ/2) for a turn Δ per vertex, and must still pass."""

MITER_FLOOR = 1.0 + math.cos(MAX_SMOOTH_TURN)
"""The least 1 + cos(turn) a miter divides by: that of MAX_SMOOTH_TURN, past
which `energy` continues the curvature linearly. Past it a miter shrinks
back to nothing at a fold instead of growing without bound: a trial step
that folds a centreline offsets it to finite tracks, and the bend-radius
hinge and the self check keep the optimizer away from it."""

FOLD_SLACK = 1e-12
"""1 + cos(turn) at or below which a vertex is a fold: off the axes,
rounding leaves an exact fold's 1 + cos a hair above 0, where sin / (1 + cos)
is noise. A turn this close to a fold has a bend radius of well under a
nanometre either way."""


class PairError(ValueError):
    """Two chains cannot be treated as one differential pair."""


def complementary_net(name: str) -> str | None:
    """The other net of a pair, matching KiCad's router (`MatchDpSuffix`):
    trailing digits and underscores are skipped, and the character before
    them must be `+`, `-`, `P` or `N` (upper case only)."""
    stem = name.rstrip(_IGNORED_TRAILING)
    complement = _SUFFIX_COMPLEMENTS.get(stem[-1:])
    if complement is None:
        return None
    return stem[:-1] + complement + name[len(stem) :]


def nets_pair(a: str, b: str) -> bool:
    return complementary_net(a) == b


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
    """One coupled stretch: inclusive vertex indices into `Pair.a.points` and
    `Pair.b.points`, with start < end on each track. `a_start` lies opposite
    `b_start` unless `b_reversed`, when b runs against a and `a_start` lies
    opposite `b_end`. `gap` is the section's median edge-to-edge gap."""

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


def runs_coupled(a: Chain, b: Chain, chord_tolerance: float) -> bool:
    """Whether two chains of one width have a stretch that `find_pair` would
    take as coupled, whatever their nets."""
    return bool(
        _coupled_spans(
            np.array(a.points, dtype=float),
            np.array(b.points, dtype=float),
            a.width,
            chord_tolerance,
        )
    )


def find_pair(a: Chain, b: Chain, chord_tolerance: float) -> Pair | None:
    """Find the coupled sections of two chains whose nets pair.

    A coupled section is a stretch where each track stays within a band
    around the pair's pitch from the other, at least `MIN_COUPLED_PITCHES`
    pitches and `MIN_COUPLED_LENGTH` long. `chord_tolerance` widens the band,
    since tracks simplified on their own each stray from the true curve by up
    to that. The ends of a section lie opposite each other, so a track that
    runs on past the other's fan-out gets a vertex there. None when no
    stretch qualifies.
    """
    if not nets_pair(a.net, b.net):
        raise PairError(f"nets {a.net} and {b.net} do not form a differential pair")
    if a.layer != b.layer:
        raise PairError(
            f"differential pair {a.net}/{b.net} is on two layers: "
            f"{a.layer} and {b.layer}"
        )
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
    snap = max(VERTEX_SNAP, chord_tolerance)
    track_a, index_a = _with_vertices(
        points_a, np.array(a.pinned), [d for s in spans for d in s.a], snap
    )
    track_b, index_b = _with_vertices(
        points_b, np.array(b.pinned), [d for s in spans for d in s.b], snap
    )
    sections: list[Section] = []
    distances: list[FloatArray] = []
    for k in range(len(spans)):
        a_start, a_end = index_a[2 * k : 2 * k + 2]
        b_first, b_last = index_b[2 * k : 2 * k + 2]
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
    lengths = np.hypot(delta[:, 0], delta[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(lengths)])
    return _Polyline(points[:-1], delta / lengths[:, None], lengths, arc)


def _cross(u: FloatArray, v: FloatArray) -> FloatArray:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _point_at(points: FloatArray, distances: FloatArray) -> FloatArray:
    arc = _polyline(points).arc
    return np.stack(
        [
            np.interp(distances, arc, points[:, 0]),
            np.interp(distances, arc, points[:, 1]),
        ],
        axis=-1,
    )


def _coupled_spans(
    a: FloatArray, b: FloatArray, width: float, chord_tolerance: float
) -> list[_SectionEnds]:
    """Sample both tracks, measure each sample's distance to the other track,
    and keep the runs of samples in the band around the dominant pitch.

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
    runs_a = _runs(np.abs(near_a.distance - pitch) <= band)
    runs_b = _runs(np.abs(near_b.distance - pitch) <= band)
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
    min_length = max(MIN_COUPLED_PITCHES * pitch, MIN_COUPLED_LENGTH)
    return [
        span
        for span in sorted(spans, key=lambda s: s.a[0])
        if min(abs(span.a[1] - span.a[0]), abs(span.b[1] - span.b[0])) >= min_length
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
    along b), or else stays at the inner of the two run ends.
    """
    along_a = _nearest(_point_at(b, np.array(hull_b)), a).along
    start = max(hull_a[0], float(along_a.min()))
    end = min(hull_a[1], float(along_a.max()))

    def snap(target: float, inward: float) -> tuple[float, float]:
        reach = inward * (candidates[:, 0] - target)
        close = candidates[(reach >= -step) & (reach <= pitch)]
        if len(close):
            chosen = close[np.argmin(inward * close[:, 0])]
            return float(chosen[0]), float(chosen[1])
        foot = _nearest(_point_at(a, np.array([target])), b).along
        return target, float(foot[0])

    first, last = snap(start, 1.0), snap(end, -1.0)
    return _SectionEnds((first[0], last[0]), (first[1], last[1]))


def _with_vertices(
    points: FloatArray, pinned: BoolArray, distances: list[float], snap: float
) -> tuple[Track, list[int]]:
    """Insert a vertex at each arc length unless one lies within `snap`, and
    return the vertex index of each arc length."""
    for distance in distances:
        arc = _polyline(points).arc
        if np.abs(arc - distance).min() > snap:
            index = int(np.searchsorted(arc, distance))
            vertex = _point_at(points, np.array(distance))
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


def gap_range(a: FloatArray, b: FloatArray, width: float) -> tuple[float, float]:
    """The smallest and largest edge-to-edge gap between two tracks.

    The smallest is exact: between polylines it occurs at a vertex. The
    largest is sampled, square across: from points whose nearest point on
    the other track lies inside an edge. A mitred pair's outer corner lies
    farther from the inner track than either of its edges, by
    pitch·(1/cos(turn/2) - 1), but the gap across the edges stays the pair's.
    """
    near = _centre_nearest(a, b, width)
    square = near.distance[near.square] if near.square.any() else near.distance
    return float(near.distance.min()) - width, float(square.max()) - width


def gaps_across(
    a: FloatArray, b: FloatArray, width: float, turn: float, reach: float
) -> tuple[float, float]:
    """The smallest edge-to-edge gap between a pair's tracks a and b where
    they lie within `turn` of each other along the pair, side by side, and
    where they lie farther apart along it, as across the legs of a U or the
    laps of a spiral; infinite where no centres come within `reach`.

    A point's place along the pair is that of its nearest point on the
    pair's centreline. Each smallest gap is that of samples of either track,
    vertices included, to the other's edges.
    """
    middle = centreline(a, b, width).points
    beside, beyond = [math.inf], [math.inf]
    for points, other in ((a, b), (b, a)):
        samples = shapely.get_coordinates(
            shapely.segmentize(LineString(points), width / SAMPLES_PER_WIDTH)
        )
        line = _polyline(other)
        segments = shapely.linestrings(np.stack([other[:-1], other[1:]], axis=1))
        tree = STRtree(np.asarray(segments, dtype=object))
        query, index = tree.query(
            shapely.points(samples), predicate="dwithin", distance=reach
        )
        start = samples[query] - line.starts[index]
        t = np.clip(
            np.sum(start * line.directions[index], axis=1), 0.0, line.lengths[index]
        )
        foot = line.starts[index] + t[:, None] * line.directions[index]
        offset = samples[query] - foot
        distance = np.hypot(offset[:, 0], offset[:, 1])
        apart = np.abs(
            _nearest(samples[query], middle).along - _nearest(foot, middle).along
        )
        beside.extend(distance[apart <= turn])
        beyond.extend(distance[apart > turn])
    return min(beside) - width, min(beyond) - width


def _centre_distances(a: FloatArray, b: FloatArray, width: float) -> FloatArray:
    """Distances from samples along each track to the other track's centre."""
    return _centre_nearest(a, b, width).distance


class _Across(NamedTuple):
    distance: FloatArray
    square: BoolArray
    """Whether the nearest point lies inside an edge, not at its end."""


def _centre_nearest(a: FloatArray, b: FloatArray, width: float) -> _Across:
    step = width / SAMPLES_PER_WIDTH
    samples_a = shapely.get_coordinates(shapely.segmentize(LineString(a), step))
    samples_b = shapely.get_coordinates(shapely.segmentize(LineString(b), step))
    near_a, near_b = _nearest(samples_a, b), _nearest(samples_b, a)
    return _Across(
        np.concatenate([near_a.distance, near_b.distance]),
        np.concatenate([near_a.square, near_b.square]),
    )


class _Nearest(NamedTuple):
    """For each query point: its nearest point on a polyline, the arc length
    there, and the distance to it."""

    foot: FloatArray
    along: FloatArray
    distance: FloatArray
    square: BoolArray
    """Whether the foot lies inside its edge rather than at one of its ends."""


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
    square = (projection > -SQUARE_SLACK) & (
        projection < line.lengths[index] + SQUARE_SLACK
    )
    return _Nearest(
        foot, line.arc[index] + t, np.hypot(offset[:, 0], offset[:, 1]), square
    )


def offset_pair(
    centreline: FloatArray, gap: float | FloatArray, width: float
) -> tuple[FloatArray, FloatArray]:
    """The two tracks of a pair at ±(gap + width)/2 from `centreline`, with
    `gap` one for the whole pair or one per vertex.

    The first track lies to the right of the direction of travel as seen with
    y down, the second to the left. Vertices are mitred, so every offset
    segment is parallel to its centreline segment at exactly half the pitch.
    The centreline must have no repeated points, and neither track folds
    while `check_bend_radius` passes.
    """
    half_pitch = np.broadcast_to((gap + width) / 2, len(centreline))
    shift = half_pitch[:, None] * _miters(centreline)
    return centreline + shift, centreline - shift


def offset_pair_pullback(
    centreline: FloatArray,
    gap: FloatArray,
    width: float,
    right: FloatArray,
    left: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """The gradient of a function of `offset_pair`'s two tracks, `right` and
    `left` at each of their vertices, taken back to the centreline's
    vertices and to each vertex's gap."""
    half_pitch = (gap + width) / 2
    delta = np.diff(centreline, axis=0)
    lengths = np.hypot(delta[:, 0], delta[:, 1])
    along = delta / lengths[:, None]
    normal = np.stack([-along[:, 1], along[:, 0]], axis=1)
    before = np.concatenate([normal[:1], normal])
    after = np.concatenate([normal, normal[-1:]])
    total = before + after
    one_plus_cos = 1.0 + np.sum(before * after, axis=1)
    scale = np.maximum(one_plus_cos, MITER_FLOOR)
    apart = right - left
    miter = total / scale[:, None]
    at_miter = half_pitch[:, None] * apart
    # The miter (b + a) / max(1 + b·a, MITER_FLOOR) of the normals b before
    # and a after a vertex, differentiated in each.
    weight = np.where(
        one_plus_cos > MITER_FLOOR,
        np.sum(total * at_miter, axis=1) / scale**2,
        0.0,
    )
    at_before = at_miter / scale[:, None] - after * weight[:, None]
    at_after = at_miter / scale[:, None] - before * weight[:, None]
    at_normal = np.zeros_like(normal)
    at_normal[0] += at_before[0]
    at_normal += at_before[1:]
    at_normal += at_after[:-1]
    at_normal[-1] += at_after[-1]
    # normal = R·along with R a quarter turn; along = delta / |delta|.
    turned = np.stack([at_normal[:, 1], -at_normal[:, 0]], axis=1)
    across = np.sum(normal * at_normal, axis=1)[:, None]
    at_delta = (turned - along * across) / lengths[:, None]
    gradient = right + left
    gradient[1:] += at_delta
    gradient[:-1] -= at_delta
    return gradient, np.sum(apart * miter, axis=1) / 2


def _miters(polyline: FloatArray) -> FloatArray:
    """Per vertex, the shift that moves both adjacent segments one unit to
    the right of travel (y down): the bisector, scaled by 1/cos(turn/2), at
    most as much as at MAX_SMOOTH_TURN (`MITER_FLOOR`)."""
    delta = np.diff(polyline, axis=0)
    direction = delta / np.hypot(delta[:, 0], delta[:, 1])[:, None]
    normal = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
    before = np.concatenate([normal[:1], normal])
    after = np.concatenate([normal, normal[-1:]])
    scale = np.maximum(1.0 + np.sum(before * after, axis=1), MITER_FLOOR)
    return (before + after) / scale[:, None]


def bend_radius_limit(gap: float, width: float) -> float:
    """The smallest centreline bend radius at which the inner track's inner
    edge still has a radius of at least zero."""
    return (gap + width) / 2 + width / 2


class BendCheck(NamedTuple):
    radius: float
    too_tight: bool


def check_bend_radius(centreline: FloatArray, gap: float, width: float) -> BendCheck:
    """The centreline's smallest bend radius and whether it is below the limit."""
    radius = float(bend_radii(centreline).min())
    return BendCheck(radius, bool(too_tight(radius, bend_radius_limit(gap, width))))


def bend_radii(centreline: FloatArray) -> FloatArray:
    """Each edge's bend radius, as `energy.edge_bend_radii` defines it, 0
    beside a fold. For a dense arc of radius R with turn Δ per vertex it is
    R·cos(Δ/2)."""
    delta = np.diff(centreline, axis=0)
    lengths = np.hypot(delta[:, 0], delta[:, 1])
    direction = delta / lengths[:, None]
    sine = _cross(direction[:-1], direction[1:])
    one_plus_cos = 1.0 + np.sum(direction[:-1] * direction[1:], axis=1)
    folds = one_plus_cos <= FOLD_SLACK
    half_turns = np.divide(sine, one_plus_cos, out=np.zeros_like(sine), where=~folds)
    radii, _ = edge_bend_radii(lengths, half_turns)
    radii[:-1][folds] = 0.0
    radii[1:][folds] = 0.0
    return radii


def too_tight(radius: FloatArray | float, limit: float) -> BoolArray:
    """Whether each bend radius is below `limit`, beyond round-off."""
    return np.asarray(radius) < limit * (1.0 - BEND_RADIUS_TOLERANCE)
