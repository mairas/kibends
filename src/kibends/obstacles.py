"""Obstacles on the chain's layer, or on every layer a moving via spans,
read from the board's snapshot; the static field itself is Rust's.

Each obstacle is raw geometry (a copper polygon, a track centreline, a via or
hole centre) plus an inflation: the distance from that geometry at which the
chain's centreline, or the via's centre, reaches the required clearance. This
module reads the snapshot's copper, holes, keepouts and board outline into
such items (`Obstacle`), with the inflations the clearance rules give
(`_Keeper`), and hands them to the Rust field as plain coordinates
(`plain_shapes`), which measures distances and searches (`kibends._core`; its
design is in `kibends_core::field`). Python keeps every shapely construction:
buffers, unions, the board outline, the zones' regions.

A chain's static field is one per (layer, net, width), from other nets'
copper, holes, keepouts and the board edge, with every selected track and
moving via removed (`build_obstacles`); no copper of the chain's net is an
obstacle. A moving via's field spans every layer it spans, including rule
areas keeping out vias (`split_via_obstacles`). What only limits a via
without repelling it is held apart, as the board edge is: the outside of
where its zones connect it (`containment`) and the fabrication rules
within its net, hole-to-hole and pad overlap (`own_net_stops`). On a
zone's layers the via also keeps the zone's clearance and margin from
other-net copper where that exceeds the netclass rule (`_zone_inflation`),
so the refill reconnects it wherever the other copper ends up. Two vias
keep their distance, and a via's zones count against another, only where
their spans overlap (`via_via_inflation`).

With ground nets, the vias standing still that a moving via pairs with,
ground with regular or of two ground nets (`vias.pairs_with`), leave its
field and are built apart, each as the field would count it
(`split_via_obstacles`): `vias.ViaInput.copper` takes them back in, so
the distances it gives are the unsplit field's, and the optimizer's
pairwise term (`via_pairs`) gives their energy.

The other selected chains are obstacles of their own kind (`ChainField`),
rebuilt at their current shapes; its searches and the energy's go through
the Rust pair helpers (`nearest_of_pairs`, `smooth_of_pairs` and the others
here), which take numpy arrays.
"""

import itertools
import math
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
import shapely
from loguru import logger
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from kibends import _core
from kibends.chain import Point2
from kibends.energy import (
    SMOOTH_REACH,
    BoolArray,
    DistanceField,
    FloatArray,
    IntArray,
    ObstacleField,
    combine_fields,
    smooth_reach,
)
from kibends.snapshot import (
    LayerSnapshot,
    Pad,
    PadKind,
    Via,
    ViaZone,
    edge_paths_closed,
    outline_from_edge_paths,
    zone_unions,
)

ObjectArray = NDArray[np.object_]
ComplexArray = NDArray[np.complex128]

# The API cannot report the board's copper-to-edge and copper-to-hole rules.
# 0.3 mm to the edge meets common fabs' minimum for routed outlines, and
# 0.25 mm to holes and between holes are the hole clearance and hole-to-hole
# minimum KiCad's own project templates set. KiCad's DRC, which the user runs,
# enforces the board's real rules.
EDGE_CLEARANCE = 0.3
HOLE_CLEARANCE = 0.25
HOLE_TO_HOLE = 0.25

# Closer than this to the nearest feature, the direction to it is noise; the
# boundary's normal is used instead. Far below KiCad's 1 nm resolution.
DEGENERATE_DISTANCE = 1e-9
# Copper this much farther from a chain than both half widths still touches
# it: 1 nm, KiCad's coordinate resolution.
TOUCH_TOLERANCE = 1e-6
# Other chains' edges are split to at most this length, so that a point near an
# edge is also near its midpoint, where the k-d tree finds it. Chains resampled
# at the default 0.1 mm `OptimizerSettings.spacing` are never split.
MAX_EDGE_LENGTH = 0.5

POLYGON_TYPE = 3
"""shapely's type id of a Polygon."""


def plain_shapes(geometries: Iterable[BaseGeometry]) -> _core.Shapes:
    """shapely geometries as the plain coordinates the Rust field takes:
    each geometry's parts, a polygon's rings exterior first, a line's or a
    point's coordinates as one ring."""
    shapes: ObjectArray = np.empty(0, dtype=object)
    listed = list(geometries)
    if listed:
        shapes = np.empty(len(listed), dtype=object)
        shapes[:] = listed
    parts, part_of = shapely.get_parts(shapes, return_index=True)
    # Untyped **kwargs in types-shapely, as in Obstacles.__init__.
    types: NDArray[np.int32] = shapely.get_type_id(parts)  # pyright: ignore[reportUnknownMemberType]
    polygon = types == POLYGON_TYPE
    polygons = np.flatnonzero(polygon)
    rings, ring_of = shapely.get_rings(parts[polygons], return_index=True)
    ring_part = np.concatenate([polygons[ring_of], np.flatnonzero(~polygon)])
    order = np.argsort(ring_part, kind="stable")
    every_ring = np.concatenate([rings, parts[~polygon]])[order]
    coords, coord_ring = shapely.get_coordinates(every_ring, return_index=True)
    return _core.Shapes(
        np.asarray(coords, dtype=float).reshape(-1, 2),
        _ends(coord_ring, len(every_ring)),
        _ends(ring_part[order], len(parts)),
        polygon,
        _ends(part_of, len(shapes)),
        shapely.get_type_id(shapes).astype(np.int32),  # pyright: ignore[reportUnknownMemberType]
    )


def _ends(owner: IntArray, count: int) -> IntArray:
    """Where each of `count` runs ends, of items each of the run `owner`
    gives, ascending."""
    return np.cumsum(np.bincount(owner, minlength=count)).astype(np.intp)


@dataclass(frozen=True)
class Obstacle:
    description: str
    geometry: BaseGeometry
    """Polygonal copper, or a centreline or centre point to be inflated."""
    inflation: float
    """Distance from `geometry` at which the chain's centreline is at clearance."""
    net: str | None = None
    """The net of copper, None for what is not copper."""
    extent: float = 0.0
    """How far copper reaches beyond `geometry`, mm: half a track's width or
    a via's diameter."""
    layer: str | None = None
    """The layer of copper, None for what is not copper of one layer."""


class Obstacles:
    """Signed distance to the nearest inflated obstacle, and edge checks:
    `items`, and the outside of `board_outline` inflated by
    `edge_inflation`, as a Rust field (`_core.Obstacles`).

    Positive in free space, zero exactly at the required clearance, negative
    inside an inflated obstacle or outside the board. Where obstacles overlap,
    the deepest one counts. `smooth_distance` is the smooth minimum over the
    obstacles: the items of one geometry, every connected run of
    centrelines of one inflation that meet end to end, and the board edge.
    """

    def __init__(
        self,
        items: Sequence[Obstacle],
        board_outline: BaseGeometry | None = None,
        edge_inflation: float = 0.0,
    ) -> None:
        self.items = tuple(items)
        self.board_outline = board_outline
        self.edge_inflation = edge_inflation
        self._field = _core.Obstacles(
            plain_shapes(o.geometry for o in self.items),
            np.array([o.inflation for o in self.items], dtype=float),
            [o.net for o in self.items],
            np.array([o.extent for o in self.items], dtype=float),
            [o.layer for o in self.items],
            None if board_outline is None else plain_shapes([board_outline]),
            edge_inflation,
        )

    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """Distance (n,) and its unit gradient (n, 2) at (n, 2) points in mm.

        With no obstacles and no board outline the distance is +inf and the
        gradient zero. Points farther than `reach` from every inflated
        obstacle and from the inflated board edge get +inf and a zero
        gradient; points outside the board are always reported.
        """
        return self._field.signed_distance(_points(points), reach)

    def smooth_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """`signed_distance` as the energy takes it: the smooth minimum over
        the obstacles, from the field's edges (`static_edges`)."""
        return self._field.smooth_distance(_points(points), reach)

    @property
    def static_edges(self) -> StaticEdges:
        """The field as edges, its items in order, then the board edge."""
        return self._field.static_edges

    def segments_clear(
        self, starts: FloatArray, ends: FloatArray, tolerance: float = 0.0
    ) -> BoolArray:
        """Whether each segment from starts[i] to ends[i] stays out of every
        inflated obstacle and inside the inflated board edge.

        Touching an inflated boundary, which is exactly at clearance, is clear,
        and so is entering it by at most `tolerance` mm. The tolerance does not
        extend the board outline itself, which only matters with an edge
        clearance below it.
        """
        return self._field.segments_clear(_points(starts), _points(ends), tolerance)


def _points(points: FloatArray) -> FloatArray:
    """`points` as an (n, 2) array of floats."""
    return np.asarray(points, dtype=float).reshape(-1, 2)


class CompositeField:
    """The union of several fields: the nearest of them answers each point,
    and the smooth distance is over all their obstacles.

    A point beyond `reach` of every field is +inf with a zero gradient. A
    segment is clear only if it is clear in every field.
    """

    def __init__(self, fields: Sequence[ObstacleField]) -> None:
        self.fields = tuple(fields)

    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """Distance (n,) and its unit gradient (n, 2) at (n, 2) points in mm."""
        if not self.fields:
            n = len(np.asarray(points).reshape(-1, 2))
            return np.full(n, np.inf), np.zeros((n, 2))
        distance, gradient = self.fields[0].signed_distance(points, reach)
        for field in self.fields[1:]:
            d, g = field.signed_distance(points, reach)
            nearer = d < distance
            if nearer.any():
                distance = np.where(nearer, d, distance)
                gradient = np.where(nearer[:, None], g, gradient)
        return distance, gradient

    def smooth_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """The smooth minimum of the fields' smooth distances
        (`energy.combine_fields`). It is the one over all their obstacles
        but for those SMOOTH_REACH leaves out within each field: up to
        τ·e⁻¹⁰ per obstacle apart from the one `static_parts` gives."""
        n = len(np.asarray(points).reshape(-1, 2))
        gradient = np.zeros((n, 2))
        if not self.fields:
            return np.full(n, np.inf), gradient
        answers = [field.smooth_distance(points, reach) for field in self.fields]
        smooth, shares = combine_fields(*(d for d, _ in answers))
        for share, (_, g) in zip(shares, answers, strict=True):
            gradient += share[:, None] * g
        return smooth, gradient

    def segments_clear(
        self, starts: FloatArray, ends: FloatArray, tolerance: float = 0.0
    ) -> BoolArray:
        """Whether each segment from starts[i] to ends[i] is clear in every
        field, each allowing `tolerance` inside its clearance."""
        clear = np.ones(len(np.asarray(starts).reshape(-1, 2)), dtype=np.bool_)
        for field in self.fields:
            clear &= field.segments_clear(starts, ends, tolerance)
        return clear


LINE = _core.LINE
"""A `StaticEdges` item measured to its own geometry: a centreline or a
centre."""
AREA = _core.AREA
"""A `StaticEdges` item measured to the boundary of its area, negative
inside it: copper, a keepout."""
OUTSIDE = _core.OUTSIDE
"""A `StaticEdges` item measured to the boundary of its area, negative
outside it: the board edge, the region a via's zones connect it in."""

INDEXED_EDGES = _core.INDEXED_EDGES
"""A `StaticEdges` item with more edges than this, such as copper text, a
logo or a board outline, finds its edges near a point with a tree over
them rather than measuring every one."""

StaticEdges = _core.StaticEdges
StaticItems = _core.StaticItems


class Found(NamedTuple):
    """(point, edge) pairs of static fields that can answer points, as
    `static_smooth` takes them: by point, then obstacle, then edge, so
    that the pairs of one item and those of one obstacle at a point lie
    together. Each pair's edge, where it starts and the way to its end, its
    inflation, how far from the point it counts unless the point is inside
    its item, its obstacle and item; whether the point was inside the item
    when found, and whether it may have crossed the item's boundary since,
    so that it is tested again."""

    point: IntArray
    starts: ComplexArray
    ends: ComplexArray
    inflation: FloatArray
    limit: FloatArray
    obstacle: IntArray
    item: IntArray
    inside: BoolArray
    unsure: BoolArray


NO_FOUND = Found(
    np.empty(0, dtype=np.intp),
    np.empty(0, dtype=complex),
    np.empty(0, dtype=complex),
    np.empty(0),
    np.empty(0),
    np.empty(0, dtype=np.intp),
    np.empty(0, dtype=np.intp),
    np.empty(0, dtype=bool),
    np.empty(0, dtype=bool),
)


class StaticDistances(NamedTuple):
    """(point, obstacle) pairs of a static field, by point then obstacle,
    and each point's exact signed distance to its obstacle."""

    point: IntArray
    obstacle: IntArray
    distance: FloatArray


NO_DISTANCES = StaticDistances(
    np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp), np.empty(0)
)


def join_found(founds: Sequence[Found]) -> Found:
    """`founds` one after another."""
    if len(founds) == 1:
        return founds[0]
    every = [NO_FOUND, *founds]
    return Found(
        np.concatenate([f.point for f in every]),
        np.concatenate([f.starts for f in every]),
        np.concatenate([f.ends for f in every]),
        np.concatenate([f.inflation for f in every]),
        np.concatenate([f.limit for f in every]),
        np.concatenate([f.obstacle for f in every]),
        np.concatenate([f.item for f in every]),
        np.concatenate([f.inside for f in every]),
        np.concatenate([f.unsure for f in every]),
    )


def static_parts(field: DistanceField) -> list[StaticEdges] | None:
    """`field` as `StaticEdges`, one per field it joins, or None if it is
    of a kind that gives none."""
    if isinstance(field, Obstacles | ChainField):
        return [field.static_edges]
    if not isinstance(field, CompositeField):
        return None
    parts: list[StaticEdges] = []
    for f in field.fields:
        found = static_parts(f)
        if found is None:
            return None
        parts += found
    return parts


def search_parts(
    parts: Sequence[tuple[StaticEdges, int]],
    coords: FloatArray,
    reach: float,
    slack: float,
) -> Found:
    """`StaticEdges.search` over several parts, each with the number its
    items start at, their obstacles numbered one part after another, so
    that the smooth minimum is over all of them."""
    return Found(*_core.search_parts(list(parts), _points(coords), reach, slack))


def static_distances(
    parts: Sequence[StaticEdges], coords: FloatArray, below: float
) -> StaticDistances:
    """`StaticEdges.distances` over several parts, their obstacles numbered
    one part after another, as `search_parts` numbers them."""
    return StaticDistances(*_core.static_distances(list(parts), _points(coords), below))


def parts_smooth_distance(
    parts: Sequence[StaticEdges], points: FloatArray, reach: float = math.inf
) -> tuple[FloatArray, FloatArray]:
    """`smooth_distance` over the obstacles of `parts`, from one search."""
    return _core.parts_smooth_distance(list(parts), _points(points), reach)


def static_smooth(
    query: ComplexArray, found: Found, items: StaticItems
) -> tuple[FloatArray, FloatArray]:
    """The smooth distance at each point of `query` and its gradient over
    the pairs `found` gives, its items numbered as `items` numbers them:
    of each item the nearest edge to a point, the exact signed distance,
    then of each obstacle the nearest item, then the smooth minimum over
    the obstacles. A point inside an area is at the negative of its
    distance to the area's boundary, and a point on an item's edge takes
    its normal."""
    return _core.static_smooth(query, *found, items)


@dataclass(frozen=True, eq=False)
class ChainCentreline:
    """A selected chain's identity and polyline.

    It is a chain's own input to the optimizer, and, at the chain's current
    shape, an obstacle to the other selected chains on its layer.
    """

    description: str
    net: str
    width: float
    clearance: float
    """Its net's netclass clearance in mm."""
    points: FloatArray
    """Its polyline, (n, 2) in mm."""


class NearestEdge(NamedTuple):
    distance: FloatArray
    gradient: FloatArray
    edge: IntArray
    """The edge nearest each point once inflated, -1 where none is in reach."""


class ChainField:
    """Signed distance to other chains' inflated centrelines, and edge checks.

    The same answers as `Obstacles` with each centreline as an obstacle,
    except that zero-length edges and zero-length query segments count like
    any other, where shapely's tree query skips them. Done in numpy instead
    of shapely because the optimizer rebuilds fields like this for every
    burst and check. A k-d tree over the edge midpoints finds the edges near
    each query, and the exact distances to those edges are computed in bulk
    (`nearest_of_pairs`), as `kibends.seen` computes them for many members
    at once. Points are complex numbers inside, which keeps every gather
    one-dimensional.

    Its smooth distance takes the edges of one obstacle, a chain or a via,
    as one: the nearest of them, then the smooth minimum over the obstacles
    (`smooth_of_pairs`).
    """

    def __init__(
        self,
        starts: FloatArray,
        ends: FloatArray,
        inflations: FloatArray,
        obstacles: IntArray,
    ) -> None:
        """Edges from starts[i] to ends[i], (m, 2) in mm, each inflated by
        inflations[i] and part of the obstacle obstacles[i]."""
        self._starts = complex_points(starts)
        self._along = complex_points(ends) - self._starts
        self._inflations = inflations
        self._obstacles = obstacles
        self._max_inflation = float(inflations.max(initial=0.0))
        self._max_half_length = float(np.abs(self._along).max(initial=0.0)) / 2
        self._middles = (starts + ends) / 2

    @cached_property
    def _tree(self) -> cKDTree | None:
        # Built on first use: a field whose checks all end early needs none.
        return cKDTree(self._middles) if len(self._middles) else None

    @cached_property
    def static_edges(self) -> StaticEdges:
        """The field as `StaticEdges`, each edge an item of its own, for a
        field that stands still. Its distances differ from
        `smooth_distance`'s by round-off, since `static_smooth` measures
        by GEOS's point-to-segment formula."""
        ends = self._starts + self._along
        return StaticEdges.from_segments(
            np.column_stack([self._starts.real, self._starts.imag]),
            np.column_stack([ends.real, ends.imag]),
            self._inflations,
            self._obstacles,
        )

    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """Distance (n,) and its unit gradient (n, 2) at (n, 2) points in mm.

        With no edges the distance is +inf and the gradient zero everywhere.
        Points farther than `reach` from every inflated edge get +inf and a
        zero gradient.
        """
        nearest = self.nearest(points, reach)
        return nearest.distance, nearest.gradient

    def nearest(self, points: FloatArray, reach: float = math.inf) -> NearestEdge:
        """`signed_distance`, and the index of the edge that answers each
        point, -1 where none does."""
        coords = np.asarray(points, dtype=float).reshape(-1, 2)
        distance = np.full(len(coords), np.inf)
        gradient = np.zeros((len(coords), 2))
        edges = np.full(len(coords), -1, dtype=np.intp)
        if self._tree is None or not len(coords):
            return NearestEdge(distance, gradient, edges)
        query = complex_points(coords)
        point, edge, point_limit = self._search(coords, query, reach, 0.0)
        return self._nearest_of(query, point, edge, point_limit)

    def smooth_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """`signed_distance` as the energy takes it: the smooth minimum over
        the obstacles of each one's nearest edge (`smooth_of_pairs`)."""
        found = self.smooth_answers(points, reach)
        return found.distance, found.gradient

    def smooth_answers(
        self, points: FloatArray, reach: float = math.inf
    ) -> SmoothAnswer:
        """`smooth_distance`, and the edges that answer each point, by
        their index, with their shares."""
        coords = np.asarray(points, dtype=float).reshape(-1, 2)
        if self._tree is None or not len(coords):
            return _no_answers(len(coords))
        query = complex_points(coords)
        point, edge, limit = self._search(
            coords, query, smooth_reach(reach), SMOOTH_REACH
        )
        order = obstacle_order(point, self._obstacles[edge], edge)
        point, edge = point[order], edge[order]
        found = smooth_of_pairs(
            query,
            point,
            self._starts[edge],
            self._along[edge],
            self._inflations[edge],
            limit[order],
            self._obstacles[edge],
        )
        return found._replace(edge=edge[found.edge])

    def _search(
        self, coords: FloatArray, query: ComplexArray, reach: float, beyond: float
    ) -> tuple[IntArray, IntArray, FloatArray]:
        """(point, edge) pairs among which are the edges within `reach` of
        each point once inflated, or without a finite reach those within
        `beyond` of its nearest; and how far from its point each pair is
        looked at."""
        assert self._tree is not None
        if math.isfinite(reach):
            # An edge is within reach once inflated only if it is within reach
            # plus the largest inflation.
            limit = reach + self._max_inflation
            point, edge = _pairs_within(
                self._tree, coords, limit + self._max_half_length + DEGENERATE_DISTANCE
            )
            return point, edge, np.full(len(point), limit)
        _, nearest = self._tree.query(coords)
        nearest_raw = np.abs(
            _offsets(query, self._starts[nearest], self._along[nearest])
        )
        # An edge farther than the nearest one can still be nearer once
        # inflated, by at most the spread of the inflations.
        limits = nearest_raw - self._inflations[nearest] + self._max_inflation + beyond
        point, edge = _pairs_each_within(
            self._tree, coords, limits + self._max_half_length + DEGENERATE_DISTANCE
        )
        return point, edge, limits[point]

    def _nearest_of(
        self,
        query: ComplexArray,
        point: IntArray,
        edge: IntArray,
        point_limit: FloatArray | float,
    ) -> NearestEdge:
        """`nearest` at `query` over the (point, edge) pairs given, each
        looked at within `point_limit` of its point."""
        found = nearest_of_pairs(
            query,
            point,
            self._starts[edge],
            self._along[edge],
            self._inflations[edge],
            point_limit,
        )
        answered = found.edge >= 0
        edges = np.full(len(query), -1, dtype=np.intp)
        edges[answered] = edge[found.edge[answered]]
        return NearestEdge(found.distance, found.gradient, edges)

    def segments_clear(
        self, starts: FloatArray, ends: FloatArray, tolerance: float = 0.0
    ) -> BoolArray:
        """Whether each segment from starts[i] to ends[i] stays out of every
        inflated edge.

        Touching an inflated boundary, which is exactly at clearance, is clear,
        and so is entering it by at most `tolerance` mm.
        """
        a = complex_points(np.asarray(starts, dtype=float).reshape(-1, 2))
        b = complex_points(np.asarray(ends, dtype=float).reshape(-1, 2))
        clear = np.ones(len(a), dtype=np.bool_)
        reach = self._max_inflation - tolerance
        if self._tree is None or reach <= 0.0 or not len(a):
            return clear
        middle = (a + b) / 2
        segment, edge = _pairs_each_within(
            self._tree,
            np.column_stack([middle.real, middle.imag]),
            reach + np.abs(b - a) / 2 + self._max_half_length + DEGENERATE_DISTANCE,
        )
        if not len(segment):
            return clear
        inside = pairs_inside(
            a[segment],
            b[segment] - a[segment],
            self._starts[edge],
            self._along[edge],
            self._inflations[edge],
            tolerance,
        )
        clear[segment[inside]] = False
        return clear


def pairs_inside(
    a: ComplexArray,
    ab: ComplexArray,
    starts: ComplexArray,
    along: ComplexArray,
    inflations: FloatArray,
    tolerance: float,
) -> BoolArray:
    """`ChainField.segments_clear` over (segment, edge) pairs: whether the
    segment from a[k] along ab[k] enters the edge from starts[k] along
    along[k], inflated by inflations[k], by more than `tolerance`."""
    return _core.pairs_inside(a, ab, starts, along, inflations, tolerance)


def pair_depths(
    a: ComplexArray,
    ab: ComplexArray,
    starts: ComplexArray,
    along: ComplexArray,
    inflations: FloatArray,
) -> FloatArray:
    """The signed distance between each pair of `pairs_inside`: the gap
    between the segment and the edge, less the edge's inflation; negative
    inside, and least, -inflation, where the two cross."""
    return _core.pair_depths(a, ab, starts, along, inflations)


def nearest_of_pairs(
    query: ComplexArray,
    point: IntArray,
    starts: ComplexArray,
    along: ComplexArray,
    inflations: FloatArray,
    limit: FloatArray | float,
) -> NearestEdge:
    """`ChainField.nearest` at the points `query` over (point, segment)
    pairs: pair k is the segment from starts[k] along along[k], inflated by
    inflations[k], looked at within `limit`, one or per pair, of
    query[point[k]]. The edge that answers a point is its pair's index."""
    return NearestEdge(
        *_core.nearest_of_pairs(
            query, point, starts, along, inflations, _per_pair(limit, point)
        )
    )


class SmoothAnswer(NamedTuple):
    """The smooth distance at each point and its gradient, and what answers
    it: per obstacle that has a share at a point, the point, the edge
    nearest it of that obstacle, its share and its unit gradient."""

    distance: FloatArray
    gradient: FloatArray
    point: IntArray
    edge: IntArray
    share: FloatArray
    direction: FloatArray


def _no_answers(count: int) -> SmoothAnswer:
    """`SmoothAnswer` at `count` points that nothing answers."""
    return SmoothAnswer(
        np.full(count, np.inf),
        np.zeros((count, 2)),
        np.empty(0, dtype=np.intp),
        np.empty(0, dtype=np.intp),
        np.empty(0),
        np.empty((0, 2)),
    )


def smooth_of_pairs(
    query: ComplexArray,
    point: IntArray,
    starts: ComplexArray,
    along: ComplexArray,
    inflations: FloatArray,
    limit: FloatArray | float,
    obstacle: IntArray,
) -> SmoothAnswer:
    """`ChainField.smooth_answers` at the points `query` over (point, segment)
    pairs, taken as `nearest_of_pairs` takes them, pair k's segment part of
    obstacle[k]: of each obstacle the nearest segment to a point, of equal
    ones the first in that order, then the smooth minimum over the obstacles
    (`energy.smooth_minimum`). The pairs come in the order `obstacle_order`
    gives, so that the pairs of a point and an obstacle lie together. The
    edge that answers is its pair's index; the answers come in the order
    of their points."""
    return SmoothAnswer(
        *_core.smooth_of_pairs(
            query, point, starts, along, inflations, _per_pair(limit, point), obstacle
        )
    )


def obstacle_order(point: IntArray, obstacle: IntArray, tiebreak: IntArray) -> IntArray:
    """The order of (point, segment) pairs, pair k's segment part of
    obstacle[k], that `smooth_of_pairs` takes them in: by point, then by
    obstacle, then by tiebreak[k], unique among a point's pairs, such as
    the segment's index, so that of equally near segments of an obstacle
    the lowest tiebreak answers."""
    return _core.obstacle_order(point, obstacle, tiebreak)


def candidate_pairs(
    query: ComplexArray,
    point: IntArray,
    starts: ComplexArray,
    along: ComplexArray,
    inflations: FloatArray,
    within: FloatArray | float,
    slack: float,
) -> BoolArray:
    """Which (point, segment) pairs, as `smooth_of_pairs` takes them, can
    answer their point within `within`, one or per pair, once the distance
    from each point to each segment has changed by at most `slack`, mm."""
    return _core.candidate_pairs(
        query, point, starts, along, inflations, _per_pair(within, point), slack
    )


def near_segments(
    starts: FloatArray, ends: FloatArray, points: FloatArray, radius: float
) -> tuple[IntArray, IntArray]:
    """(point, segment) index pairs, by point then segment, among them every
    pair whose segment from starts[k] to ends[k] comes within `radius` of
    points[i], found with one tree over the segments' middles."""
    return _core.near_segments(_points(starts), _points(ends), _points(points), radius)


def _per_pair(value: FloatArray | float, point: IntArray) -> FloatArray:
    """`value`, one or per pair, as one per pair of `point`."""
    return np.broadcast_to(np.asarray(value, dtype=float), point.shape).copy()


def complex_points(points: FloatArray) -> ComplexArray:
    """(n, 2) points as n complex numbers."""
    return points[:, 0] + 1j * points[:, 1]


def _pairs_within(
    tree: cKDTree, centres: FloatArray, radius: float
) -> tuple[IntArray, IntArray]:
    """(centre, point) index pairs with the tree's point within `radius`."""
    # Built for one query, the tree skips the balancing and node shrinking
    # that only pay off over many queries.
    once = cKDTree(centres, balanced_tree=False, compact_nodes=False)
    found = once.sparse_distance_matrix(tree, radius, output_type="ndarray")
    return found["i"].astype(np.intp), found["j"].astype(np.intp)


def _pairs_each_within(
    tree: cKDTree, centres: FloatArray, radii: FloatArray
) -> tuple[IntArray, IntArray]:
    """(centre, point) index pairs with the tree's point within radii[i] of
    centres[i]."""
    found = tree.query_ball_point(centres, radii, return_sorted=False)
    counts = np.fromiter((len(f) for f in found), np.intp, len(found))
    flat = itertools.chain.from_iterable(found)
    points = np.fromiter(flat, np.intp, int(counts.sum()))
    return np.repeat(np.arange(len(centres)), counts), points


def _offsets(
    points: ComplexArray, starts: ComplexArray, along: ComplexArray
) -> ComplexArray:
    """The vector to each point from the nearest point of the segment from
    starts[i] along along[i]."""
    relative = points - starts
    projection = relative.real * along.real + relative.imag * along.imag
    squared = along.real**2 + along.imag**2
    t = np.clip(projection / np.where(squared > 0.0, squared, 1.0), 0.0, 1.0)
    return relative - t * along


def _split_edges(vertices: FloatArray) -> tuple[FloatArray, FloatArray]:
    """The polyline's edges, each split evenly into parts of at most
    MAX_EDGE_LENGTH."""
    a, b = vertices[:-1], vertices[1:]
    lengths = np.hypot(*(b - a).T)
    parts = np.maximum(np.ceil(lengths / MAX_EDGE_LENGTH), 1).astype(np.intp)
    edge = np.repeat(np.arange(len(a)), parts)
    part = np.arange(len(edge)) - np.repeat(np.cumsum(parts) - parts, parts)
    along = (b - a)[edge]
    first = (part / parts[edge])[:, None]
    last = ((part + 1) / parts[edge])[:, None]
    return a[edge] + first * along, a[edge] + last * along


def chain_inflation(
    ignored: frozenset[str], width: float, clearance: float, other: ChainCentreline
) -> float | None:
    """The distance a track of `width` and netclass `clearance` (mm) keeps
    its centreline from `other`'s, mm: both half widths and the larger of
    both clearances, as from an unselected track; None for a chain of a net
    in `ignored`, the track's own and the other net of its diff pair, which
    may come close, overlap or cross."""
    if other.net in ignored:
        return None
    return track_inflation(width, clearance, other.width, other.clearance)


def track_inflation(
    width: float, clearance: float, other_width: float, other_clearance: float
) -> float:
    """`chain_inflation` between tracks of two nets, of `width` and
    `clearance` and of `other_width` and `other_clearance`, mm."""
    # Tracks have no holes, so the keeper's hole clearance is never asked.
    keeper = _Keeper(None, width / 2, clearance, HOLE_CLEARANCE)
    inflation = keeper.copper("", other_width / 2, other_clearance)
    # A keeper of no net keeps clear of every net.
    assert inflation is not None
    return inflation


def chains_field(
    inflated: Iterable[
        tuple[ChainCentreline | Sequence[ChainCentreline], float | None]
    ],
) -> ChainField:
    """The centrelines as a field, each inflated by its own amount and an
    obstacle of its own, or several as the pieces of one; those with None
    are left out."""
    starts: list[FloatArray] = [np.empty((0, 2))]
    ends: list[FloatArray] = [np.empty((0, 2))]
    inflations: list[FloatArray] = [np.empty(0)]
    obstacles: list[IntArray] = [np.empty(0, dtype=np.intp)]
    count = 0
    for o, inflation in inflated:
        if inflation is None:
            continue
        for piece in [o] if isinstance(o, ChainCentreline) else o:
            a, b = _split_edges(np.asarray(piece.points, dtype=float).reshape(-1, 2))
            starts.append(a)
            ends.append(b)
            inflations.append(np.full(len(a), inflation))
            obstacles.append(np.full(len(a), count, dtype=np.intp))
        count += 1
    return ChainField(
        np.concatenate(starts),
        np.concatenate(ends),
        np.concatenate(inflations),
        np.concatenate(obstacles),
    )


@dataclass(frozen=True)
class _Keeper:
    """What keeps clear of the board's items: the copper disc around each
    point of a track's centreline, or a via's ring and hole at its centre.

    Each method gives the inflation of another item's raw geometry, or None
    where no rule applies between the two.
    """

    net: str | None
    """None for a pair's centreline, which keeps clear of every net."""
    radius: float
    """Half the track width, or the via's ring radius, mm."""
    clearance: float
    """Its net's netclass clearance, mm."""
    hole_clearance: float
    hole_radius: float | None = None
    """A via's drill radius; None for a track."""
    hole_to_hole: float = HOLE_TO_HOLE
    zones: tuple[ViaZone, ...] = ()
    """For a via, the zones it stays inside on the layers at hand."""

    def copper(self, net: str, extent: float, clearance: float) -> float | None:
        """For copper of `net` reaching `extent` mm beyond its geometry, with
        `clearance` its own clearance; None for copper of its own net."""
        if net == self.net:
            return None
        inflation = max(
            extent + max(self.clearance, clearance) + self.radius,
            _zone_inflation(self.zones, extent, self.clearance, clearance),
        )
        if self.hole_radius is None:
            return inflation
        return max(inflation, extent + self.hole_radius + self.hole_clearance)

    def hole(self, net: str, radius: float, *, plated: bool) -> float | None:
        """For a hole of `radius` mm of `net`. A same-net plated hole is part
        of the connection, so only hole-to-hole applies to it."""
        inflations: list[float] = []
        if net != self.net or not plated:
            inflations.append(radius + self.hole_clearance + self.radius)
        if self.hole_radius is not None:
            inflations.append(radius + self.hole_radius + self.hole_to_hole)
        return max(inflations, default=None)


def _zone_inflation(
    zones: Iterable[ViaZone], extent: float, *clearances: float
) -> float:
    """How far a via's centre keeps from other-net copper reaching `extent`
    mm beyond its geometry for the next refill of `zones` to connect it: the
    refill keeps the larger of the zone's clearance and `clearances` from
    that copper, and the fill reaches the via's centre by the zone's margin.
    -inf without zones."""
    return max(
        (extent + max(*clearances, z.clearance) + z.margin for z in zones),
        default=-math.inf,
    )


def zones_on(zones: Iterable[ViaZone], layers: Iterable[str]) -> tuple[ViaZone, ...]:
    """The zones among `zones` a via stays inside on any of `layers`."""
    wanted = set(layers)
    return tuple(z for z in zones if wanted.intersection(z.layers))


def _copper_items(
    snapshot: LayerSnapshot,
    keeper: _Keeper,
    vias: Iterable[Via],
    via_keeper: Callable[[Via], _Keeper] | None = None,
) -> list[Obstacle]:
    """The snapshot layer's copper and track keepouts, with `vias` in place
    of the layer's vias, holes left out; `via_keeper` gives the keeper of
    each via's ring, `keeper` by default."""
    keeper_of: Callable[[Via], _Keeper] = via_keeper or (lambda _: keeper)
    pads = [
        _kept(
            keeper,
            snapshot.layer,
            f"pad {pad.name}",
            pad.shape,
            pad.net,
            0.0,
            pad.clearance_override
            if pad.clearance_override is not None
            else snapshot.clearance(pad.net),
        )
        for pad in snapshot.pads
        if pad.shape is not None
    ]
    tracks = [
        _kept(
            keeper,
            snapshot.layer,
            f"track {t.id}",
            t.centreline,
            t.net,
            t.width / 2,
            snapshot.clearance(t.net),
        )
        for t in snapshot.tracks
    ]
    graphics = [
        _kept(
            keeper,
            snapshot.layer,
            g.description,
            g.geometry,
            g.net,
            0.0,
            snapshot.clearance(g.net),
        )
        for g in snapshot.graphics
    ]
    items = [
        *pads,
        *tracks,
        *_via_rings(snapshot, vias, keeper_of),
        *graphics,
        *(Obstacle("track keepout", k, keeper.radius) for k in snapshot.track_keepouts),
    ]
    return [item for item in items if item is not None]


def _kept(
    keeper: _Keeper,
    layer: str,
    description: str,
    geometry: BaseGeometry,
    net: str,
    extent: float,
    clearance: float,
) -> Obstacle | None:
    """Copper of `net` on `layer` reaching `extent` mm beyond `geometry`,
    with `clearance` its own, as an obstacle to what `keeper` keeps, or
    None for copper of its own net."""
    inflation = keeper.copper(net, extent, clearance)
    if inflation is None:
        return None
    return Obstacle(description, geometry, inflation, net, extent, layer)


def _via_rings(
    snapshot: LayerSnapshot, vias: Iterable[Via], keeper_of: Callable[[Via], _Keeper]
) -> list[Obstacle | None]:
    """The rings of `vias`, each kept from by the keeper `keeper_of` gives,
    at the clearances `snapshot` gives their nets, None for one of the
    keeper's net."""
    return [
        _kept(
            keeper_of(v),
            snapshot.layer,
            f"via {v.id}",
            Point(v.position),
            v.net,
            v.diameter / 2,
            snapshot.clearance(v.net),
        )
        for v in vias
    ]


def _hole_items(
    pads: Iterable[Pad], vias: Iterable[Via], keeper: _Keeper
) -> list[Obstacle]:
    candidates = [
        (
            f"hole of pad {pad.name}",
            Point(pad.position),
            keeper.hole(pad.net, pad.drill / 2, plated=pad.kind is not PadKind.NPTH),
        )
        for pad in pads
        if pad.drill > 0
    ]
    candidates += [
        (
            f"hole of via {v.id}",
            Point(v.position),
            keeper.hole(v.net, v.drill / 2, plated=True),
        )
        for v in vias
    ]
    return [Obstacle(d, g, i) for d, g, i in candidates if i is not None]


def _board_outline(edge_paths: Sequence[Sequence[Point2]]) -> BaseGeometry | None:
    """The board area, or None if Edge.Cuts is not closed."""
    if edge_paths_closed(edge_paths):
        return outline_from_edge_paths(edge_paths)
    return None


def build_obstacles(
    snapshot: LayerSnapshot,
    net: str,
    width: float,
    *,
    edge_clearance: float = EDGE_CLEARANCE,
    hole_clearance: float = HOLE_CLEARANCE,
) -> Obstacles:
    """Obstacles for a track of `net` and `width` mm on the snapshot's layer.

    Copper clearance is the larger of both items' netclass clearances, with a
    pad's own override replacing its netclass. Copper of `net` is never an
    obstacle; its holes are not either, except unplated ones.
    """
    keeper = _Keeper(net, width / 2, snapshot.clearance(net), hole_clearance)
    return _layer_obstacles(snapshot, keeper, edge_clearance)


def build_pair_obstacles(
    snapshot: LayerSnapshot,
    a: ChainCentreline,
    b: ChainCentreline,
    gap: float,
    *,
    edge_clearance: float = EDGE_CLEARANCE,
    hole_clearance: float = HOLE_CLEARANCE,
) -> Obstacles:
    """Obstacles for the centreline of a coupled section of a differential
    pair, its tracks `a` and `b` `gap` mm apart, on the snapshot's layer.

    The centreline keeps clear as one track as wide as the pair would: half
    its outer extent, a track width and half the gap, plus the larger of
    the two nets' clearances and the obstacle's own. That is exact beside
    the pair and errs on the safe side ahead of it.

    Copper of the pair's own nets is an obstacle like any other's, since the
    other track must keep clear of it and the centreline cannot tell its two
    sides apart. Copper that overlaps the section's track of its own net is
    left out: what the section connects to, at its pinned ends or between
    them, where the centreline cannot keep clear of it. Copper that a fan-out
    beyond the section connects to stays, as the other track keeps clear of
    it in its own fan-out's place.
    """
    connected = {track.net: LineString(track.points) for track in (a, b)}

    def free(net: str, geometry: BaseGeometry, extent: float) -> bool:
        line = connected.get(net)
        return line is None or geometry.distance(line) > (
            extent + a.width / 2 + TOUCH_TOLERANCE
        )

    pair_snapshot = replace(
        snapshot,
        pads=tuple(
            p for p in snapshot.pads if p.shape is None or free(p.net, p.shape, 0.0)
        ),
        tracks=tuple(
            t for t in snapshot.tracks if free(t.net, t.centreline, t.width / 2)
        ),
        vias=tuple(
            v for v in snapshot.vias if free(v.net, Point(v.position), v.diameter / 2)
        ),
        graphics=tuple(g for g in snapshot.graphics if free(g.net, g.geometry, 0.0)),
    )
    clearance = max(snapshot.clearance(a.net), snapshot.clearance(b.net))
    keeper = _Keeper(None, a.width + gap / 2, clearance, hole_clearance)
    return _layer_obstacles(pair_snapshot, keeper, edge_clearance)


def _layer_obstacles(
    snapshot: LayerSnapshot, keeper: _Keeper, edge_clearance: float
) -> Obstacles:
    items = [
        *_copper_items(snapshot, keeper, snapshot.vias),
        *_hole_items(snapshot.pads, snapshot.vias, keeper),
    ]
    outline = _board_outline(snapshot.edge_paths)
    if outline is None:
        logger.warning(
            "Edge.Cuts has no closed outline; the board edge is not kept clear "
            "while smoothing, only KiCad's DRC covers it"
        )
        return Obstacles(items)
    return Obstacles(items, outline, edge_clearance + keeper.radius)


def split_via_obstacles(
    layers: Sequence[LayerSnapshot],
    via: Via,
    split: Collection[str],
    *,
    moving: Collection[str] = frozenset(),
    edge_clearance: float = EDGE_CLEARANCE,
    hole_clearance: float = HOLE_CLEARANCE,
    hole_to_hole: float = HOLE_TO_HOLE,
    zones: Sequence[ViaZone] = (),
) -> tuple[ObstacleField, Obstacles | None]:
    """Obstacles for `via` at any position, over every layer of `layers` it
    spans. The outside of the regions of `zones` is not among them, nor is
    anything of its own net: those only limit where the via may go and do
    not repel it (`containment`, `own_net_stops`).

    Rule areas keeping out vias on those layers keep its ring out, as track
    keepouts keep out a track's copper.

    Its ring keeps the copper clearance, as a track of its diameter would,
    and its hole keeps the hole clearance, to other nets' copper on each of
    those layers. On a layer of one of `zones`, its centre also keeps the
    zone's clearance and margin from that copper (`_zone_inflation`), so
    that the refill reaches it. Its hole keeps hole-to-hole from every other
    hole of another net the layers hold, and an unplated hole of its own net
    keeps the hole clearance. One field over all the layers' items is their
    minimum. The via itself and the vias in `moving`, which the caller adds
    at their current positions, are left out; so is the copper of zones,
    which KiCad refills.

    The vias of other nets whose ids are in `split` are left out of the
    field and come back apart, as obstacles to `via`, each as the field
    would count it, or None if it picks none; both from one reading of the
    layers.
    """
    found = _ViaSurroundings.of(layers, via, moving, hole_clearance, hole_to_hole)
    picked = [v for v in found.others if v.id in split]
    kept = [v for v in found.others if v.id not in split]
    field = _via_field(found, via, kept, edge_clearance, zones)
    if not picked:
        return field, None
    rings = _via_rings(found.spanned[0], picked, found.via_keeper(zones))
    return field, Obstacles(
        [
            *(ring for ring in rings if ring is not None),
            *_hole_items((), picked, found.keeper),
        ]
    )


def _via_field(
    found: _ViaSurroundings,
    via: Via,
    others: Sequence[Via],
    edge_clearance: float,
    zones: Sequence[ViaZone],
) -> ObstacleField:
    """`split_via_obstacles`' field of `via` in `found`, with only `others`
    of the vias of other nets there."""
    spanned, keeper = found.spanned, found.keeper
    drilled = [p for p in spanned[0].pads if not _plated_of(p, via)]

    def on(layers: Iterable[str]) -> _Keeper:
        return replace(keeper, zones=zones_on(zones, layers))

    # The via spans every layer here, so another via's ring counts once, as
    # do pads' holes, which are the same on every layer.
    items = [
        # The copper items include track keepouts, which hold the via back as
        # well, since the chain ends it joins sit at its centre.
        *_copper_items(
            spanned[0], on([spanned[0].layer]), others, found.via_keeper(zones)
        ),
        *(o for s in spanned[1:] for o in _copper_items(s, on([s.layer]), ())),
        *_hole_items(drilled, others, keeper),
        *(
            Obstacle("via keepout", k, keeper.radius)
            for s in spanned
            for k in s.via_keepouts
        ),
    ]
    outline = _board_outline(spanned[0].edge_paths)
    return Obstacles(items, outline, edge_clearance + keeper.radius)


@dataclass(frozen=True)
class _ViaSurroundings:
    """The layers a moving via spans, its net's clearance and its keeper
    there, and the vias there that stand still, of other nets and of its
    own."""

    spanned: Sequence[LayerSnapshot]
    clearance: float
    keeper: _Keeper
    others: Sequence[Via]
    own: Sequence[Via]

    @classmethod
    def of(
        cls,
        layers: Sequence[LayerSnapshot],
        via: Via,
        moving: Collection[str],
        hole_clearance: float,
        hole_to_hole: float,
    ) -> _ViaSurroundings:
        spanned = [s for s in layers if via.reaches(s.layer)]
        if not spanned:
            raise ValueError(f"no layer that via {via.id} spans was given")
        left_out = {via.id, *moving}
        clearance = spanned[0].clearance(via.net)
        keeper = _via_keeper(via, clearance, hole_clearance, hole_to_hole)
        standing = {
            v.id: v for s in spanned for v in s.vias if v.id not in left_out
        }.values()
        return cls(
            spanned,
            clearance,
            keeper,
            tuple(v for v in standing if v.net != via.net),
            tuple(v for v in standing if v.net == via.net),
        )

    def via_keeper(self, zones: Sequence[ViaZone]) -> Callable[[Via], _Keeper]:
        """The keeper of another via's ring: with `zones` on the layers both
        span."""

        def of(other: Via) -> _Keeper:
            layers = (s.layer for s in self.spanned if other.reaches(s.layer))
            return replace(self.keeper, zones=zones_on(zones, layers))

        return of


def own_net_stops(
    layers: Sequence[LayerSnapshot],
    via: Via,
    *,
    moving: Collection[str] = frozenset(),
    hole_clearance: float = HOLE_CLEARANCE,
    hole_to_hole: float = HOLE_TO_HOLE,
    allow_via_in_pad: bool = False,
) -> ObstacleField | None:
    """What of `via`'s net on the layers of `layers` it spans limits it, as
    obstacles that stop it without repelling it, or None if nothing does.

    Copper of one net does not interact, but two drilling rules hold: the
    holes of its net's vias, but those in `moving`, and of its net's plated
    pads keep hole-to-hole from its hole, as two moving vias of one net keep
    (`via_via_inflation`); and its ring may touch its net's pads but not
    overlap them, since a via in a pad needs the fabricator to fill and cap
    it, unless `allow_via_in_pad`, which KiCad's DRC does not check either.
    """
    found = _ViaSurroundings.of(layers, via, moving, hole_clearance, hole_to_hole)
    spanned, keeper, clearance = found.spanned, found.keeper, found.clearance
    plated = [p for p in spanned[0].pads if _plated_of(p, via)]
    items = _hole_items(plated, (), keeper)
    if not allow_via_in_pad:
        items += [
            Obstacle(f"pad {pad.name}", pad.shape, keeper.radius)
            for s in spanned
            for pad in s.pads
            if pad.net == via.net and pad.shape is not None
        ]
    parts: list[ObstacleField] = [Obstacles(items)] if items else []
    vias = _vias_field(keeper, found.own, lambda _: clearance)
    if vias is not None:
        parts.append(vias)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else CompositeField(parts)


def _plated_of(pad: Pad, via: Via) -> bool:
    """Whether `pad` has a plated hole of `via`'s net, which only stops the
    via (`own_net_stops`) and is left out of its field."""
    return pad.net == via.net and pad.kind is not PadKind.NPTH


def containment(zones: Iterable[ViaZone]) -> ObstacleField | None:
    """The outside of where `zones` connect a via, as one field for its
    centre, or None without zones: on each layer where they reach the via's
    ring, the outside of where its centre may lie for the next refill to
    connect it again through one of them (`ZoneUnion.centres`), held as the
    board edge is.

    The regions are built from outlines alone; kibends never reads a fill.
    The clearance the refill keeps from other nets' copper in a zone is the
    via's to keep instead, at the copper's current place
    (`_zone_inflation`).
    """
    # A region the same on several layers is one obstacle, not one per layer.
    regions = {u.centres.wkb: u.centres for u in zone_unions(zones).values()}
    parts = [Obstacles((), region) for region in regions.values()]
    return CompositeField(parts) if parts else None


def moving_via_obstacles(
    layer: str,
    net: str,
    width: float,
    clearance: float,
    vias: Iterable[Via],
    clearance_of: Callable[[str], float],
    *,
    hole_clearance: float = HOLE_CLEARANCE,
    zones: Mapping[str, Sequence[ViaZone]] = MappingProxyType({}),
) -> ChainField | None:
    """`vias`, at their positions, as obstacles to a track of `net`, `width`
    and netclass `clearance` (mm) on `layer`, or None if none is one.

    Each counts as `build_obstacles` counts a via on the board: its ring and
    hole, kept at the clearances of both nets, with `clearance_of` giving a
    via's netclass clearance. A via of `net` is no obstacle, nor is one not
    spanning `layer`. A via of another net in one of the zones `zones` gives by via
    id, on `layer`, keeps the zone's clearance and margin from the track, as
    `via_chain_inflation` keeps it.
    """
    keeper = _Keeper(net, width / 2, clearance, hole_clearance)
    return _vias_field(
        keeper,
        (v for v in vias if v.reaches(layer)),
        clearance_of,
        zones_of=lambda v: zones_on(zones.get(v.id, ()), [layer]),
    )


def via_via_inflation(
    via: Via,
    other: Via,
    clearance_of: Callable[[str], float],
    *,
    hole_clearance: float = HOLE_CLEARANCE,
    hole_to_hole: float = HOLE_TO_HOLE,
    zones: Sequence[ViaZone] = (),
    other_zones: Sequence[ViaZone] = (),
) -> float | None:
    """The distance `via`'s centre keeps from `other`'s, mm, as
    `split_via_obstacles` counts a via on the board; None if no rule
    applies, as for two vias whose spans do not overlap.

    Rings and holes of another net keep the copper and hole clearances,
    holes of any net keep hole-to-hole, and rings of another net the
    clearance and margin of `zones`, the zones `via` stays inside, and of
    `other_zones`, those `other` stays inside, each on the layers both
    vias span: given both, the distance is the same both ways."""
    if not via.shares_layer(other):
        return None
    keeper = _via_keeper(
        via,
        clearance_of(via.net),
        hole_clearance,
        hole_to_hole,
        _zones_reaching(zones, other),
    )
    return _via_inflation(
        keeper, other, clearance_of, zones=_zones_reaching(other_zones, via)
    )


def _zones_reaching(zones: Iterable[ViaZone], via: Via) -> tuple[ViaZone, ...]:
    """The zones among `zones` on a layer `via` spans. A zone's `layers`
    hold only layers its own via spans, so these are the zones on the
    layers both vias span, as `zones_on` gives them for those layers."""
    return tuple(z for z in zones if any(map(via.reaches, z.layers)))


def via_chain_inflation(
    via: Via,
    chain: ChainCentreline,
    clearance_of: Callable[[str], float],
    *,
    hole_clearance: float = HOLE_CLEARANCE,
    zones: Sequence[ViaZone] = (),
) -> float | None:
    """The distance `via`'s centre keeps from `chain`'s centreline, mm:
    the larger of the copper clearance from the via's ring and the hole
    clearance from its hole, and the clearance and margin of `zones`, the
    zones the via stays inside on the chain's layer; None for a chain of
    its net, which is no obstacle to it."""
    if chain.net == via.net:
        return None
    # Chains have no holes, so the keeper's hole-to-hole is never asked.
    keeper = _Keeper(
        via.net,
        via.diameter / 2,
        clearance_of(via.net),
        hole_clearance,
        via.drill / 2,
        zones=tuple(zones),
    )
    return keeper.copper(chain.net, chain.width / 2, chain.clearance)


def _via_keeper(
    via: Via,
    clearance: float,
    hole_clearance: float,
    hole_to_hole: float,
    zones: Sequence[ViaZone] = (),
) -> _Keeper:
    return _Keeper(
        via.net,
        via.diameter / 2,
        clearance,
        hole_clearance,
        via.drill / 2,
        hole_to_hole,
        zones=tuple(zones),
    )


def _vias_field(
    keeper: _Keeper,
    vias: Iterable[Via],
    clearance_of: Callable[[str], float],
    *,
    zones_of: Callable[[Via], Sequence[ViaZone]] = lambda _: (),
) -> ChainField | None:
    """The vias' rings and holes as obstacles to `keeper`, or None if none
    is one. `zones_of` gives the zones each via stays inside where the
    keeper is."""
    centres: list[Point2] = []
    inflations: list[float] = []
    for v in vias:
        inflation = _via_inflation(keeper, v, clearance_of, zones=zones_of(v))
        if inflation is not None:
            centres.append(v.position)
            inflations.append(inflation)
    if not centres:
        return None
    # Zero-length edges: ChainField measures to a point exactly as to an
    # edge. Each via is an obstacle of its own.
    at = np.array(centres, dtype=float)
    return ChainField(
        at,
        at,
        np.array(inflations, dtype=float),
        np.arange(len(centres), dtype=np.intp),
    )


def _via_inflation(
    keeper: _Keeper,
    via: Via,
    clearance_of: Callable[[str], float],
    *,
    zones: Sequence[ViaZone] = (),
) -> float | None:
    """How far `keeper` keeps from `via`'s centre, by its ring and hole, and
    by the zones `via` stays inside where the keeper is, as the via keeps
    from it (`_Keeper.copper`)."""
    ring = keeper.copper(via.net, via.diameter / 2, clearance_of(via.net))
    if ring is not None:
        ring = max(
            ring,
            _zone_inflation(
                zones, keeper.radius, keeper.clearance, clearance_of(via.net)
            ),
        )
    hole = keeper.hole(via.net, via.drill / 2, plated=True)
    return max((i for i in (ring, hole) if i is not None), default=None)
