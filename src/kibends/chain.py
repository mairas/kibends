"""Order a selection of track segments into chains with pinned vertices.

Units: segment endpoints and widths are integer nanometres, as KiCad stores
them, so that endpoint matching and deduplication are exact. Same-net copper
is shapely geometry in millimetres. The resulting chain is in millimetre
floats. Every coordinate is in KiCad's board frame, with y pointing down.
"""

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import accumulate, pairwise
from typing import NamedTuple

import shapely
from loguru import logger
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

Point2 = tuple[float, float]
NmPoint = tuple[int, int]

NM_PER_MM = 1_000_000
# Below KiCad's 1 nm resolution, so two distinct chain vertices never merge.
PIN_MERGE_TOLERANCE_MM = 0.5 / NM_PER_MM
# Far below any real geometry, yet above the rounding of a point placed on a
# diagonal segment.
JUNCTION_TOLERANCE_NM = 10
MAX_CHAIN_DEGREE = 2
# Widens the float candidate search so a point exactly at the tolerance still
# reaches the exact integer test.
CANDIDATE_MARGIN_NM = 1
CANDIDATE_MARGIN_MM = CANDIDATE_MARGIN_NM / NM_PER_MM


class ChainError(ValueError):
    """The selection cannot be split into open, unbranched chains."""


@dataclass(frozen=True)
class Segment:
    """One selected straight track segment, in integer nanometres."""

    id: str
    start: NmPoint
    end: NmPoint
    width: int
    layer: str
    net: str
    locked: bool = False


Adjacency = dict[NmPoint, list[Segment]]


class SameNetTrack(NamedTuple):
    """A same-net track that is not part of the chain; `width` in mm."""

    centreline: LineString
    width: float


SameNetCopper = BaseGeometry | SameNetTrack


@dataclass(frozen=True)
class Chain:
    """An ordered polyline in mm that replaces the segments in `segment_ids`.

    `pinned[i]` tells whether `points[i]` must stay where it is: the two
    endpoints and every point where same-net copper touches the chain.
    `segment_ids` includes duplicate and zero-length segments, which the
    chain also replaces.
    """

    points: tuple[Point2, ...]
    pinned: tuple[bool, ...]
    width: float
    layer: str
    net: str
    segment_ids: tuple[str, ...]


def extract_chains(
    segments: Sequence[Segment],
    same_net_copper: Callable[[str], Iterable[SameNetCopper]] = lambda _net: (),
) -> list[Chain]:
    """Split `segments` into open chains, one per connected piece, or raise.

    Selected segments are grouped by net and width, and each group is split
    into connected pieces. Within one net, selected tracks of all widths must
    not branch: two widths may meet only end to end, where both chains end and
    so are pinned. `same_net_copper(net)` returns the copper of `net` on the
    layer that pins its chains, as for `extract_chain`. Locked segments stay
    out of every chain, with a warning, and pin the chains they touch.

    Chains are ordered by net, then by first point, then by width; each
    starts at its end with the smaller (x, y).
    """
    kept, locked = _split_locked(segments)
    return _chains(kept, locked, same_net_copper)


def _split_locked(
    segments: Sequence[Segment],
) -> tuple[list[Segment], list[Segment]]:
    """Unlocked and locked segments, warning about the locked ones."""
    locked = [s for s in segments if s.locked]
    if locked:
        logger.warning(
            "Leaving locked segments untouched: {}", ", ".join(s.id for s in locked)
        )
    return [s for s in segments if not s.locked], locked


def _chains(
    kept: Sequence[Segment],
    locked: Sequence[Segment],
    same_net_copper: Callable[[str], Iterable[SameNetCopper]],
) -> list[Chain]:
    if not kept:
        raise ChainError("selection has no tracks that can be modified")
    layer = _single_layer(kept)
    if all(s.start == s.end for s in kept):
        raise ChainError("selection has no tracks of non-zero length")

    chains: list[Chain] = []
    for net, of_net in _group(kept, lambda s: s.net).items():
        copper = _CopperIndex(
            [
                *same_net_copper(net),
                *(
                    SameNetTrack(
                        LineString([_mm(s.start), _mm(s.end)]), s.width / NM_PER_MM
                    )
                    for s in _locked_on(locked, layer, net)
                ),
            ]
        )
        by_width = {
            width: (of_width, *_unique_edges(of_width))
            for width, of_width in _group(of_net, lambda s: s.width).items()
        }
        _check_junctions([e for _, edges, _ in by_width.values() for e in edges])
        for width, (of_width, edges, dots) in by_width.items():
            adjacency = _adjacency(edges, dots)
            pieces = _pieces(adjacency)
            piece_of = {p: i for i, piece in enumerate(pieces) for p in piece}
            ids: list[list[str]] = [[] for _ in pieces]
            for s in of_width:
                ids[piece_of[s.start]].append(s.id)
            chains += [
                _chain(
                    {p: adjacency[p] for p in piece},
                    copper,
                    width=width,
                    layer=layer,
                    net=net,
                    segment_ids=piece_ids,
                )
                for piece, piece_ids in zip(pieces, ids, strict=True)
            ]
    return sorted(chains, key=lambda c: (c.net, c.points[0], c.width))


def extract_chain(
    segments: Sequence[Segment], same_net_copper: Iterable[SameNetCopper] = ()
) -> Chain:
    """Build the one open chain that `segments` form, or raise `ChainError`.

    `same_net_copper` holds the copper of the chain's net on its layer: pad,
    via and copper shape outlines (polygons) and unselected tracks. A track
    touches the chain where its copper reaches the chain's copper.
    Zones must not be passed: they reflow after the edit, so they never pin.
    Locked segments stay out of the chain, with a warning; where one touches
    the chain, it pins it like unselected copper.

    The chain starts at the end with the smaller (x, y).
    """
    kept, locked = _split_locked(segments)
    if kept:
        _, layer, net = _uniform_attributes(kept)
        edges, dots = _unique_edges(kept)
        if edges:
            _check_junctions(edges)
            _check_one_piece(_adjacency(edges, dots), _locked_on(locked, layer, net))
    copper = tuple(same_net_copper)
    (chain,) = _chains(kept, locked, lambda _net: copper)
    return chain


def _locked_on(locked: Iterable[Segment], layer: str, net: str) -> list[Segment]:
    """The distinct locked edges of `net` on `layer`, which pin like copper."""
    edges, _ = _unique_edges([s for s in locked if s.layer == layer and s.net == net])
    return edges


class _CopperIndex:
    """One net's pinning copper, searchable by distance from a chain."""

    def __init__(self, items: Sequence[SameNetCopper]) -> None:
        self._items = items
        self._tree = shapely.STRtree(
            [i.centreline if isinstance(i, SameNetTrack) else i for i in items]
        )
        self._track_reach = max(
            (i.width / 2 for i in items if isinstance(i, SameNetTrack)), default=0.0
        )

    def near(self, line: LineString, half_width: float) -> list[SameNetCopper]:
        """Every item that can touch `line`, in the order given."""
        reach = half_width + self._track_reach + CANDIDATE_MARGIN_MM
        indices = self._tree.query(line, predicate="dwithin", distance=reach)
        return [self._items[i] for i in sorted(map(int, indices))]


def _chain(
    adjacency: Adjacency,
    copper: _CopperIndex,
    *,
    width: int,
    layer: str,
    net: str,
    segment_ids: Sequence[str],
) -> Chain:
    if not any(adjacency.values()):
        raise ChainError(
            f"selected zero-length tracks {', '.join(segment_ids)} at "
            f"{_fmt(min(adjacency))} touch no other selected track of their width"
        )
    _check_open(adjacency)
    half_width = width / NM_PER_MM / 2
    walked = [_mm(p) for p in _walk(adjacency)]
    points, pinned = _pin_touches(
        walked, copper.near(LineString(walked), half_width), half_width
    )
    return Chain(
        points=tuple(points),
        pinned=tuple(pinned),
        width=width / NM_PER_MM,
        layer=layer,
        net=net,
        segment_ids=tuple(segment_ids),
    )


def _group[Key](
    segments: Iterable[Segment], key: Callable[[Segment], Key]
) -> dict[Key, list[Segment]]:
    groups: dict[Key, list[Segment]] = defaultdict(list)
    for s in segments:
        groups[key(s)].append(s)
    return groups


def _check_junctions(edges: Sequence[Segment]) -> None:
    """Refuse branches among the selected edges of one net, of any widths.

    `edges` must already be free of duplicates within each width, so a span
    that appears twice is two tracks of different widths.
    """
    first_by_ends: dict[frozenset[NmPoint], Segment] = {}
    for s in edges:
        other = first_by_ends.setdefault(frozenset((s.start, s.end)), s)
        if other is not s:
            raise ChainError(
                f"selected tracks {other.id} and {s.id} overlap from "
                f"{_fmt(s.start)} to {_fmt(s.end)} with different widths"
            )
    _check_no_mid_segment_junctions(edges)
    _check_degrees(_adjacency(edges, ()))


def _mm(p: NmPoint) -> Point2:
    return p[0] / NM_PER_MM, p[1] / NM_PER_MM


def _fmt(p: NmPoint) -> str:
    x, y = _mm(p)
    return f"({x:.3f}, {y:.3f}) mm"


def _fmt_width(width_nm: int) -> str:
    return f"{width_nm / NM_PER_MM:g} mm"


def _uniform_attributes(segments: Sequence[Segment]) -> tuple[int, str, str]:
    width = _single_value(segments, "widths", lambda s: s.width, _fmt_width)
    layer = _single_layer(segments)
    net = _single_value(segments, "nets", lambda s: s.net, str)
    return width, layer, net


def _single_layer(segments: Sequence[Segment]) -> str:
    return _single_value(
        segments, "layers", lambda s: s.layer, str, "; run kibends once per layer"
    )


def _single_value[Attribute: (int, str)](
    segments: Sequence[Segment],
    what: str,
    key: Callable[[Segment], Attribute],
    fmt: Callable[[Attribute], str],
    hint: str = "",
) -> Attribute:
    ids_by_value: dict[Attribute, list[str]] = defaultdict(list)
    for s in segments:
        ids_by_value[key(s)].append(s.id)
    if len(ids_by_value) == 1:
        return next(iter(ids_by_value))
    listing = ", ".join(
        f"{fmt(value)} ({', '.join(ids)})"
        for value, ids in sorted(ids_by_value.items())
    )
    raise ChainError(f"selected tracks have different {what}: {listing}{hint}")


def _unique_edges(segments: Sequence[Segment]) -> tuple[list[Segment], set[NmPoint]]:
    """Split segments into edges without duplicates, and zero-length dots."""
    seen: set[frozenset[NmPoint]] = set()
    edges: list[Segment] = []
    dots: set[NmPoint] = set()
    for s in segments:
        if s.start == s.end:
            dots.add(s.start)
            continue
        key = frozenset((s.start, s.end))
        if key not in seen:
            seen.add(key)
            edges.append(s)
    return edges, dots


def _check_no_mid_segment_junctions(edges: Sequence[Segment]) -> None:
    """Refuse an edge end inside another edge, the first in input order.

    An STR tree narrows the candidates, so the check is O(n log n) for n
    edges rather than comparing every pair; the exact test stays integer.
    """
    if not edges:
        return
    tree = shapely.STRtree([LineString([s.start, s.end]) for s in edges])
    ends = [p for s in edges for p in (s.start, s.end)]
    end_index, edge_index = tree.query(
        shapely.points(ends),
        predicate="dwithin",
        distance=JUNCTION_TOLERANCE_NM + CANDIDATE_MARGIN_NM,
    )
    candidates = zip(map(int, end_index), map(int, edge_index), strict=True)
    hits = sorted(
        (i, j) for i, j in candidates if i // 2 != j and _lies_inside(ends[i], edges[j])
    )
    if hits:
        i, j = hits[0]
        raise ChainError(
            f"selected tracks branch at {_fmt(ends[i])}: "
            f"{edges[i // 2].id} ends in the middle of {edges[j].id}"
        )


def _lies_inside(p: NmPoint, s: Segment) -> bool:
    """Whether `p` is on the interior of `s`, in exact integer arithmetic."""
    if p in (s.start, s.end):
        return False
    (ax, ay), (bx, by), (px, py) = s.start, s.end, p
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    along = (px - ax) * dx + (py - ay) * dy
    if not 0 < along < length_sq:
        return False
    cross = dx * (py - ay) - dy * (px - ax)
    return cross * cross <= JUNCTION_TOLERANCE_NM**2 * length_sq


def _adjacency(edges: Sequence[Segment], dots: Iterable[NmPoint]) -> Adjacency:
    adjacency: Adjacency = {p: [] for p in dots}
    for s in edges:
        adjacency.setdefault(s.start, []).append(s)
        adjacency.setdefault(s.end, []).append(s)
    return adjacency


def _other_end(s: Segment, p: NmPoint) -> NmPoint:
    return s.end if p == s.start else s.start


def _check_degrees(adjacency: Adjacency) -> None:
    for p, segs in sorted(adjacency.items()):
        if len(segs) > MAX_CHAIN_DEGREE:
            ids = ", ".join(s.id for s in segs)
            raise ChainError(f"selected tracks branch at {_fmt(p)}: {ids} meet there")


def _pieces(adjacency: Adjacency) -> list[set[NmPoint]]:
    visited: set[NmPoint] = set()
    pieces: list[set[NmPoint]] = []
    for start in sorted(adjacency):
        if start in visited:
            continue
        piece = {start}
        stack = [start]
        while stack:
            p = stack.pop()
            for s in adjacency[p]:
                q = _other_end(s, p)
                if q not in piece:
                    piece.add(q)
                    stack.append(q)
        visited |= piece
        pieces.append(piece)
    return pieces


def _check_one_piece(adjacency: Adjacency, locked: Sequence[Segment]) -> None:
    pieces = _pieces(adjacency)
    if len(pieces) == 1:
        return
    piece_of = {p: i for i, piece in enumerate(pieces) for p in piece}
    locked_adjacency = _adjacency(locked, ())
    splitters = sorted(
        {
            s.id
            for locked_piece in _pieces(locked_adjacency)
            if len({piece_of[p] for p in locked_piece if p in piece_of}) > 1
            for p in locked_piece
            for s in locked_adjacency[p]
        }
    )
    if splitters:
        raise ChainError(
            f"locked segments {', '.join(splitters)} split the selected chain; "
            "unlock them or select one side only"
        )
    described = "; ".join(_describe(piece, adjacency) for piece in pieces)
    raise ChainError(f"selected tracks form {len(pieces)} separate pieces: {described}")


def _describe(piece: set[NmPoint], adjacency: Adjacency) -> str:
    ends = sorted(p for p in piece if len(adjacency[p]) < MAX_CHAIN_DEGREE)
    if not ends:
        return f"loop through {_fmt(min(piece))}"
    return " to ".join(_fmt(p) for p in ends)


def _check_open(adjacency: Adjacency) -> None:
    if all(len(segs) == MAX_CHAIN_DEGREE for segs in adjacency.values()):
        raise ChainError(
            f"selected tracks form a closed loop through {_fmt(min(adjacency))}; "
            "the chain must have two ends"
        )


def _walk(adjacency: Adjacency) -> list[NmPoint]:
    current = min(p for p, segs in adjacency.items() if len(segs) == 1)
    ordered = [current]
    came_by: Segment | None = None
    while onward := [s for s in adjacency[current] if s is not came_by]:
        came_by = onward[0]
        current = _other_end(came_by, current)
        ordered.append(current)
    return ordered


class _Station(NamedTuple):
    along: float
    point: Point2
    pinned: bool
    original: bool


def _pin_touches(
    points: Sequence[Point2], copper: Iterable[SameNetCopper], half_width: float
) -> tuple[list[Point2], list[bool]]:
    """Insert and pin a vertex wherever `copper` touches the chain."""
    line = LineString(points)
    last = len(points) - 1
    lengths = (math.dist(a, b) for a, b in pairwise(points))
    stations = [
        _Station(along, p, i in (0, last), original=True)
        for i, (along, p) in enumerate(
            zip(accumulate(lengths, initial=0.0), points, strict=True)
        )
    ]
    for item in copper:
        if isinstance(item, SameNetTrack):
            geometry, reach = item.centreline, half_width + item.width / 2
        else:
            geometry, reach = item, half_width
        for touch in _touch_points(line, geometry, reach):
            along = line.project(Point(touch))
            on_line = line.interpolate(along)
            stations.append(
                _Station(along, (on_line.x, on_line.y), True, original=False)
            )
    stations.sort(key=lambda st: (st.along, not st.original))

    merged: list[_Station] = []
    for station in stations:
        prev = merged[-1] if merged else None
        if (
            prev is None
            or station.along - prev.along > PIN_MERGE_TOLERANCE_MM
            or (station.original and prev.original)
        ):
            merged.append(station)
            continue
        keep = prev if prev.original else station
        merged[-1] = keep._replace(pinned=prev.pinned or station.pinned)
    return [st.point for st in merged], [st.pinned for st in merged]


def _touch_points(line: LineString, copper: BaseGeometry, reach: float) -> list[Point2]:
    """Where `copper` meets the chain: outline crossings, else the nearest point
    if `copper` is within `reach` of the centreline without crossing it."""
    outline = copper.boundary if isinstance(copper, Polygon | MultiPolygon) else copper
    crossings = _points_of(line.intersection(outline))
    if crossings:
        return crossings
    if not copper.intersects(line) and copper.distance(line) <= reach:
        nearest, _ = nearest_points(line, copper)
        return [(nearest.x, nearest.y)]
    return []


def _points_of(geometry: BaseGeometry) -> list[Point2]:
    points: list[Point2] = []
    for part in shapely.get_parts(geometry):
        if part.is_empty:
            continue
        if isinstance(part, Point):
            points.append((part.x, part.y))
        elif isinstance(part, LineString):
            (x0, y0), *_, (x1, y1) = part.coords
            points += [(x0, y0), (x1, y1)]
    return points
