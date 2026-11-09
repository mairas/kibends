"""Order a selection of track segments into chains with pinned vertices.

A chain is one connected piece of one layer, net and width
(`extract_chains`). Same-net copper holds it where it touches: unselected
tracks and copper shapes pin it, and a pad or a via that stays is a region
it keeps a vertex inside (`Contact`), an end sliding within it where it may
(`_slides`), unless a vertex of the chain lies on its anchor, which pins
that vertex there. Where three or more selected tracks of one net end, the
selection splits into chains that end there, loose unless copper pins them;
an end that reaches into another selected track may join it there
(`_join_mid_track_ends`, `_may_join`). Pins never change a chain's order or
its ends.

Where selected chains of one net overlap or cross away from their ends,
nothing pins either: copper of one net does not interact, so a run may move
them apart even where that overlap was their only connection. A stub drawn
past the track it crosses, its end beyond that track's copper, is not
joined and may come off it without a warning; the README tells users to
end stubs on the track or check the ratsnest.

A route through a via is one chain per layer. The copper of a selected via
that may move does not pin, so that it does not tether the chains it
carries; `vias` then decides which do move, and `run` extracts the chains
again with the fixed ones' copper pinning.

Units: segment endpoints and widths are integer nanometres, as KiCad stores
them, so that endpoint matching and deduplication are exact. Same-net copper
is shapely geometry in millimetres. The resulting chain is in millimetre
floats. Every coordinate is in KiCad's board frame, with y pointing down.
"""

import math
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cached_property
from itertools import accumulate, pairwise
from typing import NamedTuple

import shapely
from loguru import logger
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

from kibends.coords import UserFrame

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
# Marks copper that belongs to no selected piece.
NO_PIECE = -1


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
    """A same-net track that is not part of the chain, or the end of a
    selected one, a point; `width` in mm."""

    centreline: LineString | Point
    width: float


@dataclass(frozen=True, eq=False)
class Contact:
    """A pad or a via that stays: a region a chain need only keep one vertex
    inside, anywhere, to stay connected to it.

    KiCad connects two items of one net when their copper overlaps on a
    layer they share (`CN_VISITOR` in its connectivity code), so a vertex
    inside the pad or via connects both tracks that end there. `margin`,
    mm, is how far inside its outline that vertex stays.

    `anchor` is the pad's position or the via's centre. KiCad's drag of a
    via or a footprint carries only the track ends there, so a chain vertex
    on it stays on it. It is converted from integer nanometres as the
    chain's vertices are, so float equality is nanometre equality."""

    shape: BaseGeometry
    margin: float
    anchor: Point2

    @cached_property
    def inside(self) -> BaseGeometry:
        """Where a vertex that keeps the contact may lie, shrunk once, when
        a chain first touches it."""
        return self.shape.buffer(-self.margin)


SameNetCopper = BaseGeometry | SameNetTrack | Contact


@dataclass(frozen=True)
class Chain:
    """An ordered polyline in mm that replaces the segments in `segment_ids`.

    `pinned[i]` tells whether `points[i]` must stay where it is: the two
    endpoints and every point where same-net copper touches the chain.
    `slides` lists, by index, the vertices that keep a `Contact` instead:
    each may move anywhere inside its region, and is not pinned.
    `segment_ids` includes duplicate and zero-length segments, which the
    chain also replaces. A track split where another ends inside it is in
    the `segment_ids` of every chain that has a part of it
    (`shared_track_groups`).
    """

    points: tuple[Point2, ...]
    pinned: tuple[bool, ...]
    width: float
    layer: str
    net: str
    segment_ids: tuple[str, ...]
    slides: tuple[tuple[int, BaseGeometry], ...] = ()

    def with_slides_pinned(self) -> Chain:
        """The chain with each sliding vertex pinned where it is."""
        sliding = {i for i, _ in self.slides}
        pinned = tuple(pin or i in sliding for i, pin in enumerate(self.pinned))
        return replace(self, pinned=pinned, slides=())


def extract_chains(
    segments: Sequence[Segment],
    same_net_copper: Callable[[str, str], Iterable[SameNetCopper]] = (
        lambda _layer, _net: ()
    ),
    *,
    frame: UserFrame,
) -> list[Chain]:
    """Split `segments` into open chains, one per connected piece, or raise.

    Selected segments are grouped by layer, net and width, and each group is
    split into connected pieces. Two widths that meet end to end both end
    there and so are pinned; two that overlap along a span stay a chain
    each. A point where three or more
    selected tracks of one net on one layer meet end is a junction: the
    pieces are split there into chains that end at it, loose, unless other
    copper pins any of them there, which pins them all. A stub whose end
    reaches the copper of another track inside it meets it at a junction:
    the other is split at the point of its centreline nearest the end, and
    the end moves there (`_join_mid_track_ends`). `same_net_copper(layer,
    net)` returns the copper of `net` on `layer` that pins its chains, as
    for `extract_chain`; a via joining chains on two layers pins both, and a
    net's chain on one layer pins another where its end's copper reaches the
    other's. Where they overlap or cross away from their ends, nothing pins
    either: copper of one net does not interact, and the run may move them
    apart.
    Locked segments stay out of every chain, with a warning, and pin the
    chains they touch on their layer.

    Chains are ordered by net, then by layer, then by first point, then by
    width, then by their points; each starts at its end with the smaller
    (x, y). Errors give positions in `frame`.
    """
    kept, locked = split_locked(segments)
    return split_chains(kept, locked, same_net_copper, frame=frame)


def split_locked(
    segments: Sequence[Segment],
) -> tuple[list[Segment], list[Segment]]:
    """Unlocked and locked segments, warning about the locked ones."""
    locked = [s for s in segments if s.locked]
    if locked:
        logger.warning(
            "Leaving locked segments untouched: {}", ", ".join(s.id for s in locked)
        )
    return [s for s in segments if not s.locked], locked


def split_chains(
    kept: Sequence[Segment],
    locked: Sequence[Segment],
    same_net_copper: Callable[[str, str], Iterable[SameNetCopper]],
    *,
    frame: UserFrame,
) -> list[Chain]:
    """`extract_chains` for segments `split_locked` has already split."""
    if not kept:
        raise ChainError("selection has no tracks that can be modified")
    if all(s.start == s.end for s in kept):
        raise ChainError("selection has no tracks of non-zero length")

    chains: list[Chain] = []
    for (layer, net), given in _group(kept, lambda s: (s.layer, s.net)).items():
        fixed = _fixed_copper(same_net_copper(layer, net), locked, layer, net)
        of_net, moved = _join_mid_track_ends(given, fixed)
        moved_from: dict[Point2, list[Point2]] = defaultdict(list)
        for end, at in moved.items():
            moved_from[_mm(at)].append(_mm(end))
        by_width = {
            width: (of_width, *_unique_edges(of_width))
            for width, of_width in _group(of_net, lambda s: s.width).items()
        }
        all_edges = [e for _, edges, _ in by_width.values() for e in edges]
        junctions = {
            p
            for p, segs in _adjacency(all_edges, ()).items()
            if len(segs) > MAX_CHAIN_DEGREE
        }
        pieces: list[tuple[int, Adjacency, list[str]]] = []
        for width, (of_width, edges, dots) in by_width.items():
            of_width_pieces = _split(_adjacency(edges, dots), junctions)
            by_edge = {
                frozenset((s.start, s.end)): k
                for k, piece in enumerate(of_width_pieces)
                for s in _edges_of(piece)
            }
            by_point: dict[NmPoint, int] = {}
            for k, piece in enumerate(of_width_pieces):
                for p in piece:
                    by_point.setdefault(p, k)
            ids: list[list[str]] = [[] for _ in of_width_pieces]
            for s in of_width:
                k = (
                    by_point[s.start]
                    if s.start == s.end
                    else by_edge[frozenset((s.start, s.end))]
                )
                ids[k].append(s.id)
            pieces += [
                (width, piece, list(dict.fromkeys(piece_ids)))
                for piece, piece_ids in zip(of_width_pieces, ids, strict=True)
            ]
        # Copper of one net does not interact, so selected chains pin each
        # other only where one ends in the other's copper: there the end's
        # copper pins like unselected copper. Where they overlap or cross
        # away from their ends, nothing pins either.
        ends = [
            (k, SameNetTrack(Point(_mm(p)), width / NM_PER_MM))
            for k, (width, adjacency, _) in enumerate(pieces)
            for p, at in adjacency.items()
            if len(at) == 1
        ]
        copper = _CopperIndex(
            [*fixed, *(t for _, t in ends)],
            [*(NO_PIECE for _ in fixed), *(k for k, _ in ends)],
        )
        # An end on another piece's track holds it there, though it does not
        # pin it, so it must not slide off it.
        tracks = [
            (k, _same_net_track(s))
            for k, (_, adjacency, _) in enumerate(pieces)
            for s in dict.fromkeys(_edges_of(adjacency))
        ]
        bodies = _CopperIndex([t for _, t in tracks], [k for k, _ in tracks])
        net_chains = [
            _chain(
                adjacency,
                copper,
                bodies,
                piece=k,
                width=width,
                layer=layer,
                net=net,
                segment_ids=piece_ids,
                frame=frame,
                junctions=junctions,
                moved_from=moved_from,
            )
            for k, (width, adjacency, piece_ids) in enumerate(pieces)
        ]
        chains += _pin_junctions_whole(net_chains, {_mm(p) for p in junctions})
    return sorted(
        chains, key=lambda c: (c.net, c.layer, c.points[0], c.width, c.points)
    )


def shared_track_groups(segment_ids: Sequence[Collection[str]]) -> list[int]:
    """For each chain, given by its `segment_ids`, the smallest index of the
    chains it shares a track with, directly or through others.

    A track split at a junction belongs to every chain that meets there, so
    those chains are replaced together and count as one.
    """
    group = list(range(len(segment_ids)))
    first_with: dict[str, int] = {}
    for k, ids in enumerate(segment_ids):
        for i in ids:
            other = group[first_with.setdefault(i, k)]
            if other != group[k]:
                low, high = sorted((other, group[k]))
                group = [low if g == high else g for g in group]
    return group


def _pin_junctions_whole(
    chains: Sequence[Chain], junctions: Collection[Point2]
) -> list[Chain]:
    """Pin every branch end at a junction where copper pins any of them.

    Copper that reaches the widest branch's end alone still connects every
    branch there, so the junction must stay put for all of them.
    """
    pinned_at = {
        c.points[i]
        for c in chains
        for i in (0, -1)
        if c.pinned[i] and c.points[i] in junctions
    }
    return [
        replace(
            c,
            pinned=tuple(
                pin or (i in (0, len(c.points) - 1) and c.points[i] in pinned_at)
                for i, pin in enumerate(c.pinned)
            ),
        )
        for c in chains
    ]


def _same_net_track(s: Segment) -> SameNetTrack:
    return SameNetTrack(LineString([_mm(s.start), _mm(s.end)]), s.width / NM_PER_MM)


def _edges_of(adjacency: Adjacency) -> list[Segment]:
    return [s for segs in adjacency.values() for s in segs]


def extract_chain(
    segments: Sequence[Segment],
    same_net_copper: Iterable[SameNetCopper] = (),
    *,
    frame: UserFrame,
) -> Chain:
    """Build the one open chain that `segments` form, or raise `ChainError`.

    `same_net_copper` holds the copper of the chain's net on its layer: pad,
    via and copper shape outlines (polygons) and unselected tracks. A track
    touches the chain where its copper reaches the chain's copper. Pads and
    vias given as `Contact`s hold the chain by one sliding vertex instead
    (`_slides`).
    Zones must not be passed: they reflow after the edit, so they never pin.
    Locked segments stay out of the chain, with a warning; where one touches
    the chain, it pins it like unselected copper.

    The chain starts at the end with the smaller (x, y).
    """
    kept, locked = split_locked(segments)
    copper = tuple(same_net_copper)
    if kept:
        _, layer, net = _uniform_attributes(kept)
        joined, _ = _join_mid_track_ends(
            kept, _fixed_copper(copper, locked, layer, net)
        )
        edges, dots = _unique_edges(joined)
        if edges:
            _check_degrees(_adjacency(edges, ()), frame)
            _check_one_piece(
                _adjacency(edges, dots), _locked_on(locked, layer, net), frame
            )
    (chain,) = split_chains(kept, locked, lambda _layer, _net: copper, frame=frame)
    return chain


def _fixed_copper(
    copper: Iterable[SameNetCopper], locked: Iterable[Segment], layer: str, net: str
) -> list[SameNetCopper]:
    """The copper that pins `net`'s chains on `layer`: `copper` and the
    locked tracks there."""
    return [*copper, *map(_same_net_track, _locked_on(locked, layer, net))]


def _locked_on(locked: Iterable[Segment], layer: str, net: str) -> list[Segment]:
    """The distinct locked edges of `net` on `layer`, which pin like copper."""
    edges, _ = _unique_edges([s for s in locked if s.layer == layer and s.net == net])
    return edges


class _CopperIndex:
    """One net's pinning copper, searchable by distance from a chain.

    `pieces[i]` names the selected piece that `items[i]` belongs to, or is
    `NO_PIECE`, so that a piece's search can skip its own tracks.
    """

    def __init__(self, items: Sequence[SameNetCopper], pieces: Sequence[int]) -> None:
        self._items = items
        self._pieces = pieces
        self._tree = shapely.STRtree([_outline(i) for i in items])
        self._track_reach = max(
            (i.width / 2 for i in items if isinstance(i, SameNetTrack)), default=0.0
        )

    def near(
        self, line: BaseGeometry, half_width: float, piece: int
    ) -> list[tuple[SameNetCopper, bool]]:
        """Every item but `piece`'s own that can touch `line`, in the order
        given, each with whether it is a selected track."""
        reach = half_width + self._track_reach + CANDIDATE_MARGIN_MM
        indices = self._tree.query(line, predicate="dwithin", distance=reach)
        return [
            (self._items[i], self._pieces[i] != NO_PIECE)
            for i in sorted(map(int, indices))
            if self._pieces[i] != piece
        ]


def _chain(
    adjacency: Adjacency,
    copper: _CopperIndex,
    bodies: _CopperIndex,
    *,
    piece: int,
    width: int,
    layer: str,
    net: str,
    segment_ids: Sequence[str],
    frame: UserFrame,
    junctions: Collection[NmPoint],
    moved_from: Mapping[Point2, Sequence[Point2]],
) -> Chain:
    if not any(adjacency.values()):
        raise ChainError(
            f"selected zero-length tracks {', '.join(segment_ids)} at "
            f"{_fmt(min(adjacency), frame)} touch no other selected track of "
            "their width"
        )
    _check_open(adjacency, frame)
    half_width = width / NM_PER_MM / 2
    walk = _walk(adjacency)
    walked = [_mm(p) for p in walk]
    loose_ends = [_mm(p) for p in (walk[0], walk[-1]) if p in junctions]
    loose = {p: moved_from.get(p, ()) for p in loose_ends}
    reached = shapely.union_all(
        [LineString(walked), *(_reached_from(p, o) for p, o in loose.items())]
    )
    held = {
        p
        for p in (walked[0], walked[-1])
        if _reaches(
            [t for t, _ in bodies.near(Point(p), half_width, piece)], p, half_width
        )
    }
    near = copper.near(reached, half_width, piece)
    slides = _slides(walked, near, half_width, loose, {*loose, *held})
    points, pinned, within = _pin_touches(walked, slides, half_width, loose)
    return Chain(
        points=tuple(points),
        pinned=tuple(pinned),
        width=width / NM_PER_MM,
        layer=layer,
        net=net,
        segment_ids=tuple(segment_ids),
        slides=tuple(
            (i, region) for i, region in enumerate(within) if region is not None
        ),
    )


def _group[Key](
    segments: Iterable[Segment], key: Callable[[Segment], Key]
) -> dict[Key, list[Segment]]:
    groups: dict[Key, list[Segment]] = defaultdict(list)
    for s in segments:
        groups[key(s)].append(s)
    return groups


def _mm(p: NmPoint) -> Point2:
    return p[0] / NM_PER_MM, p[1] / NM_PER_MM


def _fmt(p: NmPoint, frame: UserFrame) -> str:
    return f"{frame.point(_mm(p))} mm"


def _fmt_width(width_nm: int) -> str:
    return f"{width_nm / NM_PER_MM:g} mm"


def _uniform_attributes(segments: Sequence[Segment]) -> tuple[int, str, str]:
    width = _single_value(segments, "widths", lambda s: s.width, _fmt_width)
    layer = _single_value(
        segments,
        "layers",
        lambda s: s.layer,
        str,
        "; tracks on several layers do not form one chain",
    )
    net = _single_value(segments, "nets", lambda s: s.net, str)
    return width, layer, net


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


class _Reach(NamedTuple):
    """A track `track` whose copper an end reaches nearest `at` inside it."""

    distance_sq: Fraction
    at: NmPoint
    track: Segment


def _join_mid_track_ends(
    segments: Sequence[Segment], fixed: Sequence[SameNetCopper]
) -> tuple[list[Segment], dict[NmPoint, NmPoint]]:
    """`segments` of one net and layer, split where a stub meets a track,
    and where each moved end came from.

    KiCad connects two tracks wherever their copper overlaps. An end `p`
    joins a track `r` that its copper reaches nearest a point inside `r`
    when every selected track ending at `p` may join `r` (`_may_join`). `r`
    is split at the nearest such point, rounded to the nanometre, and `p`
    moves there, so the point joins them, unless `p` is off that point and
    `fixed` copper, which pins, reaches `p`'s copper, which the move would
    take away from it.
    Other ends stay where they are, and copper overlap pins them. An STR
    tree narrows the candidates; the distances are exact.
    """
    edges, _ = _unique_edges(segments)
    if not edges:
        return list(segments), {}
    adjacency = _adjacency(edges, ())
    piece_of = {p: k for k, piece in enumerate(_pieces(adjacency)) for p in piece}
    ends = sorted(adjacency)
    tree = shapely.STRtree([LineString([s.start, s.end]) for s in edges])
    end_index, edge_index = tree.query(
        shapely.points(ends),
        predicate="dwithin",
        distance=max(s.width for s in edges) + CANDIDATE_MARGIN_NM,
    )
    near: dict[NmPoint, list[Segment]] = defaultdict(list)
    for i, j in zip(map(int, end_index), map(int, edge_index), strict=True):
        near[ends[i]].append(edges[j])
    moved: dict[NmPoint, NmPoint] = {}
    cuts: dict[frozenset[NmPoint], set[NmPoint]] = defaultdict(set)
    for p, candidates in near.items():
        widest = max(s.width for s in adjacency[p])
        reaching = [
            _Reach(distance_sq, at, r)
            for r in candidates
            for distance_sq, at in [_nearest_on(r, p)]
            if at is not None
            and 4 * distance_sq <= (widest + r.width) ** 2
            and all(
                _may_join(s, p, r, connected=piece_of[r.start] == piece_of[p])
                for s in adjacency[p]
            )
        ]
        if not reaching:
            continue
        # Ties go to the smaller point, so the input order does not matter.
        best = min(reaching, key=lambda r: (r.distance_sq, r.at))
        if best.at == p or not _reaches(fixed, _mm(p), widest / NM_PER_MM / 2):
            moved[p] = best.at
            cuts[frozenset((best.track.start, best.track.end))].add(best.at)
    if not moved:
        return list(segments), {}
    joined: list[Segment] = []
    for s in segments:
        inside = sorted(
            cuts.get(frozenset((s.start, s.end)), ()),
            key=lambda q: math.dist(s.start, q),
        )
        path = [moved.get(s.start, s.start), *inside, moved.get(s.end, s.end)]
        joined += [replace(s, start=a, end=b) for a, b in pairwise(path)]
    return joined, moved


def _reaches(copper: Iterable[SameNetCopper], p: Point2, half_width: float) -> bool:
    """Whether any of `copper` reaches the disc of `half_width` around `p`."""
    return any(
        item.centreline.distance(Point(p)) <= half_width + item.width / 2
        if isinstance(item, SameNetTrack)
        else _outline(item).distance(Point(p)) <= half_width
        for item in copper
    )


def _outline(item: SameNetCopper) -> BaseGeometry:
    """A track's centreline, or the copper of anything else."""
    if isinstance(item, SameNetTrack):
        return item.centreline
    if isinstance(item, Contact):
        return item.shape
    return item


def _may_join(s: Segment, p: NmPoint, r: Segment, *, connected: bool) -> bool:
    """Whether `s`'s end `p`, which reaches `r`, may join `r`.

    `s` must not cross `r`'s centreline by more than `r`'s half width,
    so that only copper inside `r` goes when `p` moves onto it. Then it
    joins where `s` lies on `r` and has its width, so that the two merge,
    or where it meets `r` as a stub: `connected`, whether the selection
    already connects `s` and `r`, is false, and `s` leaves `r`'s reach, its
    centreline widened by both half widths, through `r`'s side rather than
    running along `r` past its end.
    """
    q = _other_end(s, p)
    length_sq = _along(r, r.end)
    across_p, across_q = _across(r, p), _across(r, q)
    on_line = JUNCTION_TOLERANCE_NM**2 * length_sq
    if (
        across_p * across_q < 0
        and across_q**2 > on_line
        and 4 * across_p**2 > r.width**2 * length_sq
    ):
        return False
    on_r = (2 * JUNCTION_TOLERANCE_NM) ** 2
    if s.width == r.width and all(4 * _nearest_on(r, x)[0] <= on_r for x in (p, q)):
        return True
    if connected:
        return False
    side = 1 if across_q > 0 else -1
    from_p, from_q = side * across_p, side * across_q
    reach = (s.width + r.width) / 2 * math.sqrt(length_sq)
    if from_q <= reach:
        return False
    toward_q = (reach - from_p) / (from_q - from_p)
    exit_at = _along(r, p) + toward_q * (_along(r, q) - _along(r, p))
    return 0 < exit_at < length_sq


def _across(r: Segment, x: NmPoint) -> int:
    """`x`'s distance from `r`'s line, signed, times `r`'s length."""
    (ax, ay), (bx, by) = r.start, r.end
    return (bx - ax) * (x[1] - ay) - (by - ay) * (x[0] - ax)


def _along(r: Segment, x: NmPoint) -> int:
    """`x`'s position along `r` from its start, times `r`'s length."""
    (ax, ay), (bx, by) = r.start, r.end
    return (x[0] - ax) * (bx - ax) + (x[1] - ay) * (by - ay)


def _nearest_on(s: Segment, p: NmPoint) -> tuple[Fraction, NmPoint | None]:
    """The squared distance from `p` to `s`, and the nearest point if it is
    inside `s` on the nanometre grid, in exact arithmetic."""
    (ax, ay), (bx, by), (px, py) = s.start, s.end, p
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    along = (px - ax) * dx + (py - ay) * dy
    if along <= 0:
        return Fraction((px - ax) ** 2 + (py - ay) ** 2), None
    if along >= length_sq:
        return Fraction((px - bx) ** 2 + (py - by) ** 2), None
    cross = dx * (py - ay) - dy * (px - ax)
    at = (
        ax + round(Fraction(along * dx, length_sq)),
        ay + round(Fraction(along * dy, length_sq)),
    )
    inside = at not in (s.start, s.end)
    return Fraction(cross * cross, length_sq), at if inside else None


def _adjacency(edges: Sequence[Segment], dots: Iterable[NmPoint]) -> Adjacency:
    adjacency: Adjacency = {p: [] for p in dots}
    for s in edges:
        adjacency.setdefault(s.start, []).append(s)
        adjacency.setdefault(s.end, []).append(s)
    return adjacency


def _other_end(s: Segment, p: NmPoint) -> NmPoint:
    return s.end if p == s.start else s.start


def _check_degrees(adjacency: Adjacency, frame: UserFrame) -> None:
    for p, segs in sorted(adjacency.items()):
        if len(segs) > MAX_CHAIN_DEGREE:
            ids = ", ".join(dict.fromkeys(s.id for s in segs))
            raise ChainError(
                f"selected tracks branch at {_fmt(p, frame)}: {ids} meet there"
            )


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


def _split(adjacency: Adjacency, junctions: Collection[NmPoint]) -> list[Adjacency]:
    """The connected pieces of `adjacency`, cut at `junctions`: each edge at
    a junction ends its piece there."""
    pieces: list[Adjacency] = []
    taken: set[Segment] = set()
    for start in sorted(adjacency):
        if not adjacency[start]:
            pieces.append({start: []})
        for first in adjacency[start]:
            if first in taken:
                continue
            taken.add(first)
            piece: Adjacency = {}
            stack = [first]
            while stack:
                s = stack.pop()
                for p in (s.start, s.end):
                    piece.setdefault(p, []).append(s)
                    if p in junctions:
                        continue
                    for onward in adjacency[p]:
                        if onward not in taken:
                            taken.add(onward)
                            stack.append(onward)
            pieces.append(piece)
    return pieces


def _check_one_piece(
    adjacency: Adjacency, locked: Sequence[Segment], frame: UserFrame
) -> None:
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
    described = "; ".join(_describe(piece, adjacency, frame) for piece in pieces)
    raise ChainError(f"selected tracks form {len(pieces)} separate pieces: {described}")


def _describe(piece: set[NmPoint], adjacency: Adjacency, frame: UserFrame) -> str:
    ends = sorted(p for p in piece if len(adjacency[p]) < MAX_CHAIN_DEGREE)
    if not ends:
        return f"loop through {_fmt(min(piece), frame)}"
    return " to ".join(_fmt(p, frame) for p in ends)


def _check_open(adjacency: Adjacency, frame: UserFrame) -> None:
    if all(len(segs) == MAX_CHAIN_DEGREE for segs in adjacency.values()):
        raise ChainError(
            "selected tracks form a closed loop through "
            f"{_fmt(min(adjacency), frame)}; "
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


class _Slides(NamedTuple):
    """How the copper near a chain holds it."""

    pinning: list[tuple[SameNetCopper, bool]]
    """Copper that pins wherever it touches, each item flagged with whether
    it is a selected track."""
    ends: dict[int, BaseGeometry]
    """The ends, 0 or -1, that slide, each within its region."""
    passing: list[tuple[Point2, BaseGeometry]]
    """A vertex to insert at each point, sliding within its region."""


def _slides(
    points: Sequence[Point2],
    near: Sequence[tuple[SameNetCopper, bool]],
    half_width: float,
    loose: Mapping[Point2, Collection[Point2]],
    held: Collection[Point2],
) -> _Slides:
    """Which vertex keeps each `Contact` in `near` that the chain touches.

    A vertex on a contact's anchor is pinned there. Every contact that
    reaches such a vertex is kept by it and gets no sliding vertex.
    Otherwise, an end that is not `held`, at a junction (`loose`) or on
    another selected track, and that no other copper reaches slides when it
    lies inside every contact that reaches it, within all of them. A contact
    that no sliding end keeps and that reaches no other end gets one vertex
    instead, at the middle of the longest stretch of the chain inside it.
    Any other contact pins as other copper does.
    """
    pinning: list[tuple[SameNetCopper, bool]] = [
        (item, sel) for item, sel in near if not isinstance(item, Contact)
    ]
    contacts = [item for item, _ in near if isinstance(item, Contact)]
    if not contacts:
        return _Slides(pinning, {}, [])
    vertices = set(points)
    # Exact float equality is nanometre equality: see `Contact.anchor`.
    anchored = {c.anchor for c in contacts if c.anchor in vertices}
    pinning += [(Point(a), False) for a in anchored]
    others = [item for item, _ in pinning]
    ends: dict[int, BaseGeometry] = {}
    kept = {
        k
        for k, c in enumerate(contacts)
        if any(_reaches([c], a, half_width) for a in anchored)
    }
    for end in (0, -1):
        p = points[end]
        reaching = [k for k, c in enumerate(contacts) if _reaches([c], p, half_width)]
        insides = [contacts[k].inside for k in reaching]
        if (
            p in held
            or not reaching
            or _reaches(others, p, half_width)
            or not all(inside.covers(Point(p)) for inside in insides)
        ):
            continue
        ends[end] = shapely.intersection_all(insides)
        kept |= set(reaching)
    still = [
        _reached_from(p, loose.get(p, ()))
        for end, p in ((0, points[0]), (-1, points[-1]))
        if end not in ends
    ]
    line = LineString(points)
    passing: list[tuple[Point2, BaseGeometry]] = []
    for k, c in enumerate(contacts):
        if k in kept:
            continue
        stretch = _longest_line(line.intersection(c.inside))
        if stretch is None or any(c.shape.distance(at) <= half_width for at in still):
            pinning.append((c.shape, False))
            continue
        middle = stretch.interpolate(0.5, normalized=True)
        passing.append(((middle.x, middle.y), c.inside))
    return _Slides(pinning, ends, passing)


def _longest_line(geometry: BaseGeometry) -> LineString | None:
    """The longest line of `geometry`, if it has one of some length."""
    lines = [
        part
        for part in shapely.get_parts(geometry)
        if isinstance(part, LineString) and part.length > 0.0
    ]
    return max(lines, key=lambda line: line.length, default=None)


class _Station(NamedTuple):
    along: float
    point: Point2
    pinned: bool
    original: bool
    within: BaseGeometry | None = None


def _pin_touches(
    points: Sequence[Point2],
    slides: _Slides,
    half_width: float,
    loose: Mapping[Point2, Collection[Point2]],
) -> tuple[list[Point2], list[bool], list[BaseGeometry | None]]:
    """Insert and pin a vertex wherever the pinning copper of `slides`
    touches the chain, and insert or free the vertices that slide, each
    with its region; a pin merged with a sliding vertex pins it.

    An end at a `loose` point, a junction, is pinned only by copper that
    is not a selected track, as those are the junction's branches, and
    reaches it or the line to any point the end moved from, its copper
    before the move.
    """
    items = [
        (
            (item.centreline, half_width + item.width / 2)
            if isinstance(item, SameNetTrack)
            else (_outline(item), half_width),
            selected,
        )
        for item, selected in slides.pinning
    ]
    still_loose = {
        p
        for p, origins in loose.items()
        if not any(
            not selected and geometry.distance(_reached_from(p, origins)) <= reach
            for (geometry, reach), selected in items
        )
    }
    line = LineString(points)
    last = len(points) - 1
    ends = {0 if end == 0 else last: region for end, region in slides.ends.items()}
    lengths = (math.dist(a, b) for a, b in pairwise(points))
    stations = [
        _Station(
            along,
            p,
            i in (0, last) and p not in still_loose and i not in ends,
            original=True,
            within=ends.get(i),
        )
        for i, (along, p) in enumerate(
            zip(accumulate(lengths, initial=0.0), points, strict=True)
        )
    ]
    for p, region in slides.passing:
        along = line.project(Point(p))
        stations.append(_Station(along, p, False, original=False, within=region))
    for (geometry, reach), selected in items:
        for touch in _touch_points(line, geometry, reach):
            if selected and any(
                math.dist(touch, p) <= PIN_MERGE_TOLERANCE_MM for p in loose
            ):
                continue
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
        pinned = prev.pinned or station.pinned
        merged[-1] = keep._replace(
            pinned=pinned,
            within=None if pinned else _within_both(prev.within, station.within),
        )
    return (
        [st.point for st in merged],
        [st.pinned for st in merged],
        [st.within for st in merged],
    )


def _within_both(a: BaseGeometry | None, b: BaseGeometry | None) -> BaseGeometry | None:
    """Where a vertex keeping both regions may lie; None for neither."""
    if a is None:
        return b
    if b is None:
        return a
    return a.intersection(b)


def _reached_from(p: Point2, origins: Collection[Point2]) -> BaseGeometry:
    """`p` and the lines from it to each of `origins`."""
    return shapely.union_all([Point(p), *(LineString([p, o]) for o in origins)])


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
