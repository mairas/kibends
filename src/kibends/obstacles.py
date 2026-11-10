"""Inflated obstacles on the chain's layer, or on every layer a moving via
spans, answering distance queries.

Each obstacle is raw geometry (a copper polygon, a track centreline, a via or
hole centre) plus an inflation: the distance from that geometry at which the
chain's centreline, or the via's centre, reaches the required clearance.
Distances are measured to the raw geometry and the inflation subtracted,
which is the exact signed distance to the inflated shape without buffering
it. The board edge is the outside of the Edge.Cuts outline, inflated the
same way.

Every field answers two questions. `signed_distance` is the exact
distance to the nearest inflated obstacle, which every clearance
judgement takes. `smooth_distance` is what the energy takes: the smooth
minimum over the field's obstacles (`energy.smooth_minimum`), each
obstacle's own distance the exact one to the nearest of its parts. What
one obstacle is depends on the field: in `Obstacles`, the items of one
geometry, every connected run of centrelines of one inflation that meet
end to end, and the board edge (`_obstacle_ids`); in `ChainField`, the edges its caller
groups, a chain or a via (`chains_field`); in `CompositeField`, each
field's own.

`Obstacles` answers `signed_distance` and every segment check with
shapely, and `smooth_distance` from its items as edges (`StaticEdges`):
a search with the field's tree finds the items near the points and the
edges of them that can answer, and numpy measures those by GEOS's
point-to-segment formula (`static_smooth`). A moving via's clearance
checks take each obstacle's exact distance from the same edges
(`static_distances`), so that they judge static copper obstacle by
obstacle; to them an obstacle of `Obstacles` is its connected copper of
one net on one layer (`Obstacles._clearance_ids`), since the energy's
split at a width
change or a T-junction is no gap in the copper. An `Obstacles` field's
distances are then shapely's to round-off, and bit for bit with the GEOS
it was tested on (shapely 2.2.0, GEOS 3.14.1). A static `ChainField` and
the joined fields of a `CompositeField` take the same path in the energy
(`static_parts`): the first differs from `ChainField.smooth_distance` by
round-off, and the second takes one smooth minimum over every part's
obstacles, where
`CompositeField.smooth_distance` combines each field's own. The energy
keeps each member's and moving via's search between evaluations
(`seen.StaticList`), so that static copper costs no tree query and no
shapely call per evaluation.

A chain's static field is one per (layer, net, width), from other nets'
copper, holes, keepouts and the board edge, with every selected track and
moving via removed (`build_obstacles`); no copper of the chain's net is an
obstacle. A moving via's field spans every layer it spans, including rule
areas keeping out vias (`build_via_obstacles`). What only limits a via
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
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from kibends.chain import Point2
from kibends.energy import (
    SMOOTH_REACH,
    BoolArray,
    DistanceField,
    FloatArray,
    IntArray,
    ObstacleField,
    combine_fields,
    smooth_minimum,
    smooth_reach,
    weighted_sum,
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
# Step along a boundary to find its tangent, and off it to tell inside from out.
NORMAL_PROBE = 1e-4
FALLBACK_DIRECTION = (1.0, 0.0)
AREA_DIMENSION = 2
# Copper this much farther from a chain than both half widths still touches
# it: 1 nm, KiCad's coordinate resolution.
TOUCH_TOLERANCE = 1e-6
# A segment that touches an inflated chain may measure this far inside it, from
# round-off in splitting its edges and in the distance; it still touches.
ROUND_OFF = 1e-9
# Other chains' edges are split to at most this length, so that a point near an
# edge is also near its midpoint, where the k-d tree finds it. Chains resampled
# at the default 0.1 mm `OptimizerSettings.spacing` are never split.
MAX_EDGE_LENGTH = 0.5
# Points this much deeper than the reach inside the inflated board edge skip
# its exact distance; far more than a buffer's chord error at 16 segments per
# quarter circle on any realistic reach.
DEEP_MARGIN = 0.01


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
    """Signed distance to the nearest inflated obstacle, and edge checks.

    Positive in free space, zero exactly at the required clearance, negative
    inside an inflated obstacle or outside the board. Where obstacles overlap,
    the deepest one counts.
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
        self._geometries: ObjectArray = np.array(
            [o.geometry for o in self.items], dtype=object
        )
        self._inflations: FloatArray = np.array(
            [o.inflation for o in self.items], dtype=float
        )
        # types-shapely leaves **kwargs untyped on the vectorised functions
        # used below, which pyright strict reports wherever they are named.
        self._is_area: BoolArray = (
            shapely.get_dimensions(self._geometries)  # pyright: ignore[reportUnknownMemberType]
            == AREA_DIMENSION
        )
        # Inside a polygon, the depth is the distance to its boundary.
        self._targets: ObjectArray = np.where(
            self._is_area, shapely.boundary(self._geometries), self._geometries
        )
        self._tree = STRtree(self._geometries)
        self._max_inflation = float(self._inflations.max(initial=0.0))
        self._deep: dict[float, BaseGeometry] = {}
        self._obstacle = _obstacle_ids(self.items)
        self._edge_obstacle = int(self._obstacle.max(initial=-1)) + 1

    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """Distance (n,) and its unit gradient (n, 2) at (n, 2) points in mm.

        With no obstacles and no board outline the distance is +inf and the
        gradient zero. Points farther than `reach` from every inflated
        obstacle and from the inflated board edge get +inf and a zero
        gradient; points outside the board are always reported.
        """
        coords = np.asarray(points, dtype=float).reshape(-1, 2)
        point, _, distance, gradient = self._candidates(coords, reach)
        return _nearest(point, distance, gradient, len(coords))

    def smooth_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """`signed_distance` as the energy takes it: the smooth minimum over
        the obstacles (`_obstacle_ids`), from the field's edges
        (`static_edges`)."""
        return parts_smooth_distance([self.static_edges], points, reach)

    @cached_property
    def static_edges(self) -> StaticEdges:
        """The field as edges, its items in order, then the board edge."""
        geometries = list(self._geometries)
        kinds = np.where(self._is_area, AREA, LINE)
        inflations = self._inflations
        obstacles = self._obstacle
        if self.board_outline is not None:
            geometries.append(self.board_outline)
            kinds = np.append(kinds, OUTSIDE)
            inflations = np.append(inflations, self.edge_inflation)
            obstacles = np.append(obstacles, self._edge_obstacle)
        return StaticEdges(
            np.array(geometries, dtype=object),
            kinds.astype(np.intp),
            inflations,
            obstacles.astype(np.intp),
            self._tree,
            self._clearance_ids,
        )

    def _clearance_ids(self) -> IntArray:
        """Each item's obstacle as a clearance check judges it, then the
        board edge's: the obstacles (`_obstacle_ids`) joined where copper of
        one net overlaps on one layer, as at a track's width change or a
        T-junction, each numbered as its lowest obstacle. Connected copper
        is one piece of copper, whatever the energy splits it into; copper
        on two layers is two, even where a via joins them, since a via
        crossing from one to the other's clearance enters new copper."""
        ids = self._obstacle
        count = int(ids.max(initial=-1)) + 1
        net = np.array([o.net for o in self.items], dtype=object)
        layer = np.array([o.layer for o in self.items], dtype=object)
        is_copper = np.array(
            [o.net is not None and o.layer is not None for o in self.items],
            dtype=bool,
        )
        extent = np.array([o.extent for o in self.items], dtype=float)
        copper = np.flatnonzero(is_copper)
        joined = ids
        if len(copper) > 1:
            # Two pieces overlap within twice the wider one's extent, so the
            # wider finds the pair; one wide track does not widen every query.
            source, found = self._tree.query(
                self._geometries[copper],
                predicate="dwithin",
                distance=2 * extent[copper] + TOUCH_TOLERANCE,
            )
            a = np.minimum(copper[source], found)
            b = np.maximum(copper[source], found)
            pair = np.unique(a * len(self.items) + b)
            a, b = pair // len(self.items), pair % len(self.items)
            same = (
                (a != b)
                & is_copper[a]
                & is_copper[b]
                & (net[a] == net[b])
                & (layer[a] == layer[b])
            )
            a, b = a[same], b[same]
            gaps: FloatArray = shapely.distance(  # pyright: ignore[reportUnknownMemberType]
                self._geometries[a], self._geometries[b]
            )
            touch = gaps <= extent[a] + extent[b] + TOUCH_TOLERANCE
            graph = coo_array(
                (np.ones(int(touch.sum())), (ids[a[touch]], ids[b[touch]])),
                shape=(count, count),
            )
            _, label = connected_components(graph, directed=False)
            lowest = np.full(count, count, dtype=np.intp)
            np.minimum.at(lowest, label, np.arange(count, dtype=np.intp))
            joined = lowest[label][ids]
        if self.board_outline is None:
            return joined
        return np.append(joined, self._edge_obstacle).astype(np.intp)

    def _candidates(
        self, coords: FloatArray, reach: float
    ) -> tuple[IntArray, IntArray, FloatArray, FloatArray]:
        """Each item's and the board edge's obstacle, signed distance and
        unit gradient at the points they answer, by the point's index: every
        one within `reach`, or without a finite reach every one as near as
        the nearest, and the board edge."""
        if not self.items and self.board_outline is None:
            return _NO_CANDIDATES
        found: list[tuple[IntArray, IntArray, FloatArray, FloatArray]] = []
        query: ObjectArray = np.asarray(shapely.points(coords), dtype=object)
        if self.items:
            point, item, distance, gradient = self._item_candidates(
                coords, query, reach
            )
            found.append((point, self._obstacle[item], distance, gradient))
        if self.board_outline is not None:
            near = np.ones(len(coords), dtype=bool)
            if math.isfinite(reach):
                near = ~shapely.contains_xy(
                    self._deep_inside(reach), coords[:, 0], coords[:, 1]
                )
                if near.any():
                    outside = ~shapely.covered_by(query[near], self.board_outline)
                    to_edge = self.board_outline.boundary.distance(query[near])
                    near[near] = outside | (to_edge - self.edge_inflation <= reach)
            if near.any():
                inside_board, toward_inside = _signed_to(
                    coords[near],
                    query[near],
                    np.full(near.sum(), self.board_outline, dtype=object),
                    np.full(near.sum(), self.board_outline.boundary, dtype=object),
                    np.ones(near.sum(), dtype=bool),
                )
                at = np.flatnonzero(near)
                found.append(
                    (
                        at,
                        np.full(len(at), self._edge_obstacle, dtype=np.intp),
                        -inside_board - self.edge_inflation,
                        -toward_inside,
                    )
                )
        if len(found) <= 1:
            return found[0] if found else _NO_CANDIDATES
        return (
            np.concatenate([p for p, _, _, _ in found]),
            np.concatenate([o for _, o, _, _ in found]),
            np.concatenate([d for _, _, d, _ in found]),
            np.concatenate([g for _, _, _, g in found]),
        )

    def _deep_inside(self, reach: float) -> BaseGeometry:
        """The part of the board farther than `reach` from the inflated
        edge, prepared, with a margin for the buffer's arc approximation."""
        deep = self._deep.get(reach)
        if deep is None:
            assert self.board_outline is not None
            margin = self.edge_inflation + reach + DEEP_MARGIN
            deep = self._deep[reach] = self.board_outline.buffer(-margin)
            shapely.prepare(deep)
        return deep

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
        segments: ObjectArray = np.asarray(
            shapely.linestrings(
                np.stack([np.asarray(starts, float), np.asarray(ends, float)], axis=1)
            ),
            dtype=object,
        )
        clear = np.ones(len(segments), dtype=np.bool_)
        if self.items:
            segment_index, item_index = self._tree.query(
                segments, predicate="dwithin", distance=self._max_inflation
            )
            # Untyped **kwargs in types-shapely, as in __init__.
            gaps = shapely.distance(  # pyright: ignore[reportUnknownMemberType]
                segments[segment_index], self._geometries[item_index]
            )
            allowed = self._inflations[item_index] - tolerance
            clear[segment_index[gaps < allowed]] = False
        if self.board_outline is not None:
            clear &= shapely.covered_by(segments, self.board_outline)
            clear &= (
                self.board_outline.boundary.distance(segments)
                >= self.edge_inflation - tolerance
            )
        return clear

    def _item_candidates(
        self, coords: FloatArray, query: ObjectArray, reach: float
    ) -> tuple[IntArray, IntArray, FloatArray, FloatArray]:
        if math.isfinite(reach):
            # An item is within reach once inflated only if its raw geometry
            # is within reach plus the largest inflation.
            search = np.full(len(coords), reach + self._max_inflation)
        else:
            (point_index, nearest_index), nearest = self._tree.query_nearest(
                query, return_distance=True, all_matches=False
            )
            # An item farther than the nearest raw geometry can still be nearer
            # once inflated, by at most the spread of the inflations.
            search = np.empty(len(coords))
            search[point_index] = (
                nearest - self._inflations[nearest_index] + self._max_inflation
            )
        pairs_point, pairs_item = self._tree.query(
            query, predicate="dwithin", distance=search + DEGENERATE_DISTANCE
        )
        raw, raw_gradient = _signed_to(
            coords[pairs_point],
            query[pairs_point],
            self._geometries[pairs_item],
            self._targets[pairs_item],
            self._is_area[pairs_item],
        )
        return pairs_point, pairs_item, raw - self._inflations[pairs_item], raw_gradient


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


LINE = 0
"""A `StaticEdges` item measured to its own geometry: a centreline or a
centre."""
AREA = 1
"""A `StaticEdges` item measured to the boundary of its area, negative
inside it: copper, a keepout."""
OUTSIDE = 2
"""A `StaticEdges` item measured to the boundary of its area, negative
outside it: the board edge, the region a via's zones connect it in."""

INDEXED_EDGES = 128
"""A `StaticEdges` item with more edges than this, such as copper text, a
logo or a board outline, finds its edges near a point with a tree over
them rather than measuring every one; twice a default buffered circle's."""


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

    def taken(self, pairs: IntArray) -> Found:
        """The pairs `pairs` indexes, in that order."""
        return Found(
            self.point[pairs],
            self.starts[pairs],
            self.ends[pairs],
            self.inflation[pairs],
            self.limit[pairs],
            self.obstacle[pairs],
            self.item[pairs],
            self.inside[pairs],
            self.unsure[pairs],
        )


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


class StaticEdges:
    """A static field's obstacles as edges, with the search that finds the
    edges near points.

    Each item is a geometry of a kind (`LINE`, `AREA`, `OUTSIDE`), with its
    inflation and its obstacle. It is looked for as far beyond the reach as
    the largest inflation of the items but the `OUTSIDE` ones, as
    `Obstacles` and `ChainField` look, or for the outside of an area as far
    as its own. Its edges are its geometry's, or its boundary's, and an
    item's are numbered together. They are taken from shapely only for items
    a search finds, so that a field over a whole board costs what lies near
    the points that ask it: eight bytes for each item never found, as the
    index of where a found item's edges and bounding box are kept.

    A tree over the items finds those near a point; the `OUTSIDE` items
    come last and are few, and a point is near one unless it lies deep
    inside its area. Whether a point is inside an area is tested exactly,
    and only where the point is inside the area's bounding box. Of an item
    with more than INDEXED_EDGES edges, a tree over pieces of its edges
    finds those near a point (`_indexed_pairs`).
    """

    def __init__(
        self,
        geometries: ObjectArray,
        kinds: IntArray,
        inflations: FloatArray,
        obstacles: IntArray,
        tree: STRtree | None = None,
        clearance: Callable[[], IntArray] | None = None,
    ) -> None:
        self.geometries = geometries
        self.kinds = kinds
        self.inflations = inflations
        self.obstacles = obstacles
        self.indexed = int(np.count_nonzero(kinds != OUTSIDE))
        """How many items the tree over the items holds, all but the
        `OUTSIDE` ones, which come after them."""
        assert (kinds[self.indexed :] == OUTSIDE).all()
        if tree is None and self.indexed:
            tree = STRtree(geometries[: self.indexed])
        self._tree = tree if self.indexed else None
        self.obstacle_count = int(obstacles.max(initial=-1)) + 1
        self._clearance = clearance
        self._inflation = float(inflations[: self.indexed].max(initial=0.0))
        self._slot = np.full(len(geometries), -1, dtype=np.intp)
        self._first: IntArray = np.empty(0, dtype=np.intp)
        self._count: IntArray = np.empty(0, dtype=np.intp)
        self._bounds: FloatArray = np.empty((0, 4))
        self._starts: ComplexArray = np.empty(0, dtype=complex)
        self._ends: ComplexArray = np.empty(0, dtype=complex)
        self._pieces: dict[int, tuple[cKDTree, IntArray, float]] = {}
        self._deep: dict[tuple[int, float, float], BaseGeometry] = {}

    def search(self, coords: FloatArray, reach: float, slack: float) -> Found:
        """The pairs that can answer the points `coords` within `reach` once
        each has moved at most `slack` mm: of an item's edges those that
        can be its nearest, of the items those that can come within
        SMOOTH_REACH of the point's nearest, as `static_smooth` measures
        them. Without a finite reach, those within SMOOTH_REACH of the
        nearest. Items and obstacles are numbered as given."""
        point, item = self._near(coords, reach, slack)
        self._split(item)
        slot = self._slot[item]
        some = self._count[slot] > 0
        point, item, slot = point[some], item[some], slot[some]
        if not len(point):
            return NO_FOUND
        inside = self.inside(item, coords[point])
        query = complex_points(coords)
        pair, edge = self._pair_edges(query[point], item, slot, slack)
        per_pair = np.bincount(pair, minlength=len(point))
        firsts = np.cumsum(per_pair) - per_pair
        starts = self._starts[edge]
        raw = np.abs(_offsets(query[point[pair]], starts, self._ends[edge] - starts))
        nearest = np.minimum.reduceat(raw, firsts)
        limit = reach + np.where(
            self.kinds[item] == OUTSIDE, self.inflations[item], self._inflation
        )
        distance = np.where(inside, -nearest, nearest) - self.inflations[item]
        unsure = (
            (slack > 0.0)
            & (self.kinds[item] != LINE)
            & (nearest <= slack + DEGENERATE_DISTANCE)
        )
        # Distance changes no faster than the point moves, so an item that
        # stays in reach bounds how far the point's nearest can lie.
        stays = (inside & ~unsure) | (nearest + slack <= limit)
        bound = np.full(len(coords), np.inf)
        np.minimum.at(bound, point[stays], distance[stays] + slack)
        tolerance = 2 * DEGENERATE_DISTANCE
        reachable = inside | unsure | (nearest - slack <= limit + DEGENERATE_DISTANCE)
        keep = reachable & (distance - slack <= bound[point] + SMOOTH_REACH + tolerance)
        kept = keep[pair] & (raw <= nearest[pair] + 2 * slack + tolerance)
        pair, edge = pair[kept], edge[kept]
        order = obstacle_order(point[pair], self.obstacles[item[pair]], edge)
        pair, edge = pair[order], edge[order]
        return Found(
            point[pair],
            self._starts[edge],
            self._ends[edge],
            self.inflations[item[pair]],
            limit[pair],
            self.obstacles[item[pair]],
            item[pair],
            inside[pair],
            unsure[pair],
        )

    @cached_property
    def clearance_obstacles(self) -> IntArray:
        """Each item's obstacle as a clearance check judges it, numbered
        below `obstacle_count`: its obstacle, or for an `Obstacles` field
        its connected copper (`Obstacles._clearance_ids`). Taken only when
        a moving via's clearance asks."""
        return self.obstacles if self._clearance is None else self._clearance()

    def distances(self, coords: FloatArray, below: float) -> StaticDistances:
        """Each point's exact signed distance to each obstacle nearer than
        `below`, as a clearance check judges obstacles
        (`clearance_obstacles`), measured as `static_smooth` measures: its
        nearest item's nearest edge. Without a finite `below`, to at least
        each obstacle within SMOOTH_REACH of the point's nearest."""
        point, item = self._near(coords, max(below, 0.0), 0.0)
        self._split(item)
        slot = self._slot[item]
        some = self._count[slot] > 0
        point, item, slot = point[some], item[some], slot[some]
        if not len(point):
            return NO_DISTANCES
        inside = self.inside(item, coords[point])
        query = complex_points(coords)
        pair, edge = self._pair_edges(query[point], item, slot, 0.0)
        at = query[point[pair]]
        starts, ends = self._starts[edge], self._ends[edge]
        best, _, length = _measured(
            at, starts, ends, _segment_distances(at, starts, ends), *_groups(pair, pair)
        )
        measured = pair[best]
        of = item[measured]
        distance = np.where(inside[measured], -length, length) - self.inflations[of]
        obstacle = self.clearance_obstacles[of]
        point = point[measured]
        order = np.lexsort((distance, obstacle, point))
        point, obstacle, distance = point[order], obstacle[order], distance[order]
        first = np.ones(len(point), dtype=bool)
        first[1:] = (point[1:] != point[:-1]) | (obstacle[1:] != obstacle[:-1])
        kept = first & (distance < below)
        return StaticDistances(point[kept], obstacle[kept], distance[kept])

    def inside(self, items: IntArray, coords: FloatArray) -> BoolArray:
        """Whether each point lies inside its item, items[i] of coords[i]:
        in an `AREA`'s area or outside an `OUTSIDE`'s, never for a `LINE`.
        Only items a search found are asked."""
        kind = self.kinds[items]
        bounds = self._bounds[self._slot[items]]
        x, y = coords[:, 0], coords[:, 1]
        boxed = (
            (x >= bounds[:, 0])
            & (y >= bounds[:, 1])
            & (x <= bounds[:, 2])
            & (y <= bounds[:, 3])
        )
        test = boxed & (kind != LINE)
        within = np.zeros(len(items), dtype=bool)
        within[test] = shapely.contains_xy(
            self.geometries[items[test]], x[test], y[test]
        )
        return np.where(kind == OUTSIDE, ~within, within)

    def normal(self, item: int, point: FloatArray) -> tuple[float, float]:
        """The unit gradient at a point lying on the item's edges, where
        the direction to them is noise, as `Obstacles` takes it."""
        geometry = self.geometries[item]
        kind = int(self.kinds[item])
        target = geometry if kind == LINE else geometry.boundary
        nx, ny = _outward_normal(point, geometry, target, kind != LINE)
        return (-nx, -ny) if kind == OUTSIDE else (nx, ny)

    def _near(
        self, coords: FloatArray, reach: float, slack: float
    ) -> tuple[IntArray, IntArray]:
        """(point, item) pairs, among them every item that can come within
        `reach` of a point once inflated and once the point has moved
        `slack`, or without a finite reach within SMOOTH_REACH of its
        nearest."""
        points: list[IntArray] = [np.empty(0, dtype=np.intp)]
        items: list[IntArray] = [np.empty(0, dtype=np.intp)]
        if self._tree is not None and len(coords):
            query: ObjectArray = np.asarray(shapely.points(coords), dtype=object)
            if math.isfinite(reach):
                radius = np.full(len(coords), reach + self._inflation + slack)
            else:
                (at, nearest_item), nearest = self._tree.query_nearest(
                    query, return_distance=True, all_matches=False
                )
                # As in `Obstacles._item_candidates`, once the nearest has
                # come `slack` nearer and another gone `slack` farther.
                radius = np.empty(len(coords))
                radius[at] = (
                    nearest
                    - self.inflations[nearest_item]
                    + self._inflation
                    + SMOOTH_REACH
                    + 2 * slack
                )
            point, item = self._tree.query(
                query, predicate="dwithin", distance=radius + DEGENERATE_DISTANCE
            )
            points.append(point.astype(np.intp))
            items.append(item.astype(np.intp))
        for k in range(self.indexed, len(self.geometries)):
            near = np.ones(len(coords), dtype=bool)
            if math.isfinite(reach):
                near = ~shapely.contains_xy(
                    self._deep_inside(k, reach, slack), coords[:, 0], coords[:, 1]
                )
            at = np.flatnonzero(near)
            points.append(at)
            items.append(np.full(len(at), k, dtype=np.intp))
        return np.concatenate(points), np.concatenate(items)

    def _deep_inside(self, k: int, reach: float, slack: float) -> BaseGeometry:
        """The part of `OUTSIDE` item k's area that no point within `slack`
        of it can see the item from within `reach`, prepared, with a margin
        for the buffer's arc approximation."""
        deep = self._deep.get((k, reach, slack))
        if deep is None:
            margin = float(self.inflations[k]) + reach + slack + DEEP_MARGIN
            deep = self._deep[(k, reach, slack)] = self.geometries[k].buffer(-margin)
            shapely.prepare(deep)
        return deep

    def _pair_edges(
        self, at: ComplexArray, item: IntArray, slot: IntArray, slack: float
    ) -> tuple[IntArray, IntArray]:
        """(pair, edge) of the pairs of points `at` and their items `item`,
        by pair: every edge of an item, or of one with more than
        INDEXED_EDGES those that `_indexed_pairs` finds."""
        counts = self._count[slot]
        few = np.flatnonzero(counts <= INDEXED_EDGES)
        each = counts[few]
        pairs = [np.repeat(few, each)]
        firsts = np.repeat(self._first[slot[few]] - (np.cumsum(each) - each), each)
        edges = [firsts + np.arange(len(pairs[0]))]
        many = np.flatnonzero(counts > INDEXED_EDGES)
        by_item = many[np.argsort(item[many], kind="stable")]
        cuts = np.flatnonzero(np.diff(item[by_item])) + 1
        for group in np.split(by_item, cuts) if len(by_item) else ():
            k = int(item[group[0]])
            local, edge = self._indexed_pairs(k, at[group], slack)
            pairs.append(group[local])
            edges.append(self._first[self._slot[k]] + edge)
        pair, edge = np.concatenate(pairs), np.concatenate(edges)
        order = np.argsort(pair, kind="stable")
        return pair[order], edge[order]

    def _indexed_pairs(
        self, k: int, at: ComplexArray, slack: float
    ) -> tuple[IntArray, IntArray]:
        """(point, edge) pairs of points `at` and item k's edges, numbered
        from its first, among them every edge `search` can keep: each edge
        within 2·slack of the point's nearest. The nearest piece's middle
        is no nearer than the nearest edge, and every edge has a piece whose
        middle is within half the longest piece of the edge's nearest point,
        so the pieces within that bound, 2·slack and half a piece hold every
        such edge, the nearest among them."""
        tree, edge_of, half_piece = self._pieces_of(k)
        points = np.column_stack([at.real, at.imag])
        nearest, _ = tree.query(points)
        bound = np.asarray(nearest, dtype=float)
        # Round-off in the distances, far below the nanometre allowed.
        radius = bound + 2 * slack + half_piece + 3 * DEGENERATE_DISTANCE
        point, piece = _pairs_each_within(tree, points, radius)
        key = np.unique(point * len(edge_of) + edge_of[piece])
        return key // len(edge_of), key % len(edge_of)

    def _pieces_of(self, k: int) -> tuple[cKDTree, IntArray, float]:
        """A tree over the middles of item k's edges split into pieces of at
        most MAX_EDGE_LENGTH, each piece's edge, numbered from the item's
        first, and half the longest piece."""
        pieces = self._pieces.get(k)
        if pieces is None:
            slot = int(self._slot[k])
            first, count = int(self._first[slot]), int(self._count[slot])
            a = self._starts[first : first + count]
            along = self._ends[first : first + count] - a
            lengths = np.abs(along)
            parts = np.maximum(np.ceil(lengths / MAX_EDGE_LENGTH), 1).astype(np.intp)
            edge = np.repeat(np.arange(count), parts)
            part = np.arange(len(edge)) - np.repeat(np.cumsum(parts) - parts, parts)
            middles = a[edge] + (part + 0.5) / parts[edge] * along[edge]
            tree = cKDTree(np.column_stack([middles.real, middles.imag]))
            half_piece = float((lengths / parts).max()) / 2
            pieces = self._pieces[k] = (tree, edge, half_piece)
        return pieces

    def _split(self, items: IntArray) -> None:
        """Take the edges and bounds of those of `items` not yet split."""
        new = np.unique(items[self._slot[items] < 0])
        if not len(new):
            return
        geometries = self.geometries[new]
        area = self.kinds[new] != LINE
        targets = geometries.copy()
        targets[area] = shapely.boundary(geometries[area])
        starts, ends, of = _edges_of(targets)
        counts = np.bincount(of, minlength=len(new))
        self._slot[new] = len(self._count) + np.arange(len(new))
        self._first = np.concatenate(
            [self._first, len(self._starts) + np.cumsum(counts) - counts]
        )
        self._count = np.concatenate([self._count, counts])
        a, b = complex_points(starts), complex_points(ends)
        self._starts = np.concatenate([self._starts, a])
        self._ends = np.concatenate([self._ends, b])
        # Untyped **kwargs in types-shapely, as in Obstacles.__init__.
        bounds: FloatArray = shapely.bounds(geometries)  # pyright: ignore[reportUnknownMemberType]
        self._bounds = np.concatenate([self._bounds, bounds])


def _edges_of(targets: ObjectArray) -> tuple[FloatArray, FloatArray, IntArray]:
    """The edges of each linear geometry or point in `targets`, a point as
    an edge of no length: where each starts and ends, (m, 2), and its
    geometry's index, ascending."""
    parts, part_of = shapely.get_parts(targets, return_index=True)
    coords, coord_part = shapely.get_coordinates(parts, return_index=True)
    sizes = np.bincount(coord_part, minlength=len(parts))
    linked = np.flatnonzero(coord_part[1:] == coord_part[:-1])
    single = (np.cumsum(sizes) - sizes)[sizes == 1]
    first = np.concatenate([linked, single])
    last = np.concatenate([linked + 1, single])
    order = np.argsort(first, kind="stable")
    first, last = first[order], last[order]
    return coords[first], coords[last], part_of[coord_part[first]].astype(np.intp)


class StaticItems:
    """The items of several `StaticEdges`, numbered one part after
    another, for what `static_smooth` asks of an item."""

    def __init__(self, parts: Sequence[StaticEdges]) -> None:
        self.parts = tuple(parts)
        self.starts: IntArray = np.cumsum(
            [0, *(len(p.geometries) for p in parts)], dtype=np.intp
        )
        """Where each part's items start, and then their count."""

    def inside(self, items: IntArray, coords: FloatArray) -> BoolArray:
        """`StaticEdges.inside` of items numbered across the parts."""
        part = np.searchsorted(self.starts, items, side="right") - 1
        within = np.zeros(len(items), dtype=bool)
        for p in map(int, np.unique(part)):
            at = part == p
            within[at] = self.parts[p].inside(items[at] - self.starts[p], coords[at])
        return within

    def normal(self, item: int, point: FloatArray) -> tuple[float, float]:
        """`StaticEdges.normal` of an item numbered across the parts."""
        p = int(np.searchsorted(self.starts, item, side="right")) - 1
        return self.parts[p].normal(item - int(self.starts[p]), point)


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
    founds: list[Found] = []
    obstacles = 0
    for part, first in parts:
        found = part.search(coords, reach, slack)
        founds.append(
            found._replace(obstacle=found.obstacle + obstacles, item=found.item + first)
        )
        obstacles += part.obstacle_count
    if len(founds) <= 1:
        return founds[0] if founds else NO_FOUND
    joined = join_found(founds)
    order = obstacle_order(
        joined.point, joined.obstacle, np.arange(len(joined.point), dtype=np.intp)
    )
    return joined.taken(order)


def static_distances(
    parts: Sequence[StaticEdges], coords: FloatArray, below: float
) -> StaticDistances:
    """`StaticEdges.distances` over several parts, their obstacles numbered
    one part after another, as `search_parts` numbers them."""
    points: list[IntArray] = [NO_DISTANCES.point]
    obstacles: list[IntArray] = [NO_DISTANCES.obstacle]
    distances: list[FloatArray] = [NO_DISTANCES.distance]
    first = 0
    for part in parts:
        found = part.distances(coords, below)
        points.append(found.point)
        obstacles.append(found.obstacle + first)
        distances.append(found.distance)
        first += part.obstacle_count
    point, obstacle = np.concatenate(points), np.concatenate(obstacles)
    order = np.lexsort((obstacle, point))
    return StaticDistances(
        point[order], obstacle[order], np.concatenate(distances)[order]
    )


def parts_smooth_distance(
    parts: Sequence[StaticEdges], points: FloatArray, reach: float = math.inf
) -> tuple[FloatArray, FloatArray]:
    """`smooth_distance` over the obstacles of `parts`, from one search."""
    coords = np.asarray(points, dtype=float).reshape(-1, 2)
    items = StaticItems(parts)
    found = search_parts(
        list(zip(parts, items.starts, strict=False)), coords, smooth_reach(reach), 0.0
    )
    return static_smooth(complex_points(coords), found, items)


def static_smooth(
    query: ComplexArray, found: Found, items: StaticItems
) -> tuple[FloatArray, FloatArray]:
    """The smooth distance at each point of `query` and its gradient over
    the pairs `found` gives, its items numbered as `items` numbers them:
    of each item the nearest edge to a point, the exact signed distance,
    then of each obstacle the nearest item, then the smooth minimum over
    the obstacles (`energy.smooth_minimum`). A point inside an area is at
    the negative of its distance to the area's boundary, and a point on
    an item's edge takes its normal (`StaticEdges.normal`)."""
    count = len(query)
    if not len(found.point):
        return np.full(count, np.inf), np.zeros((count, 2))
    at = query[found.point]
    key = _segment_distances(at, found.starts, found.ends)
    inside = found.inside
    unsure = np.flatnonzero(found.unsure)
    if len(unsure):
        inside = inside.copy()
        where = at[unsure]
        inside[unsure] = items.inside(
            found.item[unsure], np.column_stack([where.real, where.imag])
        )
    counted = np.where(inside | (key <= found.limit + DEGENERATE_DISTANCE), key, np.inf)
    best, offset, length = _measured(
        at, found.starts, found.ends, counted, *_groups(found.point, found.item)
    )
    sign = np.where(inside[best], -1.0, 1.0)
    reached = np.isfinite(counted[best])
    distance = np.where(reached, sign * length, np.inf) - found.inflation[best]
    point = found.point[best]
    chosen, smooth, share = _nearest_then_smooth(
        point, found.obstacle[best], distance, count
    )
    length = length[chosen]
    degenerate = length < DEGENERATE_DISTANCE
    safe = np.where(degenerate, 1.0, length)
    direction = sign[chosen, None] * offset[chosen] / safe[:, None]
    for k in np.flatnonzero(degenerate):
        c = query[point[chosen[k]]]
        direction[k] = items.normal(
            int(found.item[best[chosen[k]]]), np.array([c.real, c.imag])
        )
    return smooth, weighted_sum(point[chosen], share, direction, count)


# GEOS's point-to-segment formula, which `Obstacles.signed_distance` gets
# through shapely: equal to shapely's distances to round-off, and bit for
# bit with shapely 2.2.0 on GEOS 3.14.1.


def _measured(
    at: ComplexArray,
    starts: ComplexArray,
    ends: ComplexArray,
    key: FloatArray,
    first: IntArray,
    group: IntArray,
) -> tuple[IntArray, FloatArray, FloatArray]:
    """Of each run of (point, edge) pairs `_groups` gives, the pair at[k]
    and the edge from starts[k] to ends[k] of least key[k], as GEOS chooses
    a geometry's nearest segment; the offset to the point from that edge's
    nearest point, (n, 2), and its length, as GEOS measures it."""
    best = _nearest_in_groups(key, first, group)
    nearest = _closest_points(at[best], starts[best], ends[best])
    offset = np.column_stack([at[best].real, at[best].imag]) - nearest
    return best, offset, np.linalg.norm(offset, axis=1)


def _segment_distances(
    points: ComplexArray, starts: ComplexArray, ends: ComplexArray
) -> FloatArray:
    """The distance from each point to the segment from starts[i] to
    ends[i], as GEOS measures it to choose a geometry's nearest segment."""
    px, py, ax, ay = points.real, points.imag, starts.real, starts.imag
    bx, by = ends.real, ends.imag
    dx, dy = bx - ax, by - ay
    squared = dx * dx + dy * dy
    single = squared == 0.0
    scale = np.where(single, 1.0, squared)
    r = ((px - ax) * dx + (py - ay) * dy) / scale
    across = ((ay - py) * dx - (ax - px) * dy) / scale
    to_start = np.sqrt((px - ax) * (px - ax) + (py - ay) * (py - ay))
    to_end = np.sqrt((px - bx) * (px - bx) + (py - by) * (py - by))
    return np.where(
        single | (r <= 0.0),
        to_start,
        np.where(r >= 1.0, to_end, np.abs(across) * np.sqrt(scale)),
    )


def _closest_points(
    points: ComplexArray, starts: ComplexArray, ends: ComplexArray
) -> FloatArray:
    """The point of each segment from starts[i] to ends[i] nearest
    points[i], (n, 2), as GEOS finds it."""
    px, py, ax, ay = points.real, points.imag, starts.real, starts.imag
    bx, by = ends.real, ends.imag
    dx, dy = bx - ax, by - ay
    squared = dx * dx + dy * dy
    on_start = (px == ax) & (py == ay)
    on_end = (px == bx) & (py == by)
    r = ((px - ax) * dx + (py - ay) * dy) / np.where(squared == 0.0, 1.0, squared)
    factor = np.where(on_start | (squared == 0.0), 0.0, np.where(on_end, 1.0, r))
    inner = (factor > 0.0) & (factor < 1.0)
    to_start = np.sqrt((px - ax) * (px - ax) + (py - ay) * (py - ay))
    to_end = np.sqrt((px - bx) * (px - bx) + (py - by) * (py - by))
    first = to_start < to_end
    x = np.where(inner, ax + factor * dx, np.where(first, ax, bx))
    y = np.where(inner, ay + factor * dy, np.where(first, ay, by))
    return np.column_stack([x, y])


def _obstacle_ids(items: Sequence[Obstacle]) -> IntArray:
    """Each item's obstacle, numbered from 0. Items of one geometry are one
    obstacle: a pad or keepout on each of the layers a via spans, a via's
    ring and its hole. So are centrelines of one inflation that meet end to
    end, as the segments of a track do, so that its joints and bends add
    nothing; the joining is transitive, so a connected run of them is one
    obstacle however it branches.

    Centrelines of two inflations stay two obstacles where they meet, as
    at a width change, and so does a centreline whose end lands on the
    middle of another, a T-junction: opposite its stem a point reads up to
    τ·ln 2 nearer than it is. A clearance check joins them again
    (`_clearance_ids`)."""
    parent = list(range(len(items)))

    def root(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    first: dict[bytes | tuple[tuple[float, ...], float], int] = {}
    for k, item in enumerate(items):
        geometry, inflation = item.geometry, item.inflation
        keys: list[bytes | tuple[tuple[float, ...], float]] = [geometry.wkb]
        if isinstance(geometry, LineString) and not geometry.is_empty:
            ends = geometry.coords
            keys += [(tuple(ends[0]), inflation), (tuple(ends[-1]), inflation)]
        for key in keys:
            parent[root(k)] = root(first.setdefault(key, k))
    ids: dict[int, int] = {}
    return np.array(
        [ids.setdefault(root(k), len(ids)) for k in range(len(items))], dtype=np.intp
    ).reshape(-1)


_NO_CANDIDATES = (
    np.empty(0, dtype=np.intp),
    np.empty(0, dtype=np.intp),
    np.empty(0),
    np.empty((0, 2)),
)


def _nearest(
    point: IntArray, distance: FloatArray, gradient: FloatArray, count: int
) -> tuple[FloatArray, FloatArray]:
    """At each of `count` points, the least of the candidate distances at
    it, candidate k's distance[k] at point[k], and that candidate's
    gradient; +inf and zero where there is none. Of equal ones, the first."""
    nearest = np.full(count, np.inf)
    toward = np.zeros((count, 2))
    order = np.lexsort((distance, point))
    _, first = np.unique(point[order], return_index=True)
    best = order[first]
    nearest[point[best]] = distance[best]
    toward[point[best]] = gradient[best]
    return nearest, toward


def _signed_to(
    coords: FloatArray,
    query: ObjectArray,
    geometries: ObjectArray,
    targets: ObjectArray,
    is_area: BoolArray,
) -> tuple[FloatArray, FloatArray]:
    """Signed distance from each point to its geometry, negative inside an
    area, and the unit gradient, pairing coords[i] with geometries[i]."""
    inside = is_area & shapely.intersects(geometries, query)
    # Untyped **kwargs in types-shapely, as in Obstacles.__init__.
    lines = shapely.shortest_line(query, targets)  # pyright: ignore[reportUnknownMemberType]
    nearest = shapely.get_coordinates(lines).reshape(-1, 2, 2)[:, 1]
    offset = coords - nearest
    length = np.linalg.norm(offset, axis=1)
    sign = np.where(inside, -1.0, 1.0)
    degenerate = length < DEGENERATE_DISTANCE
    safe = np.where(degenerate, 1.0, length)
    gradient = sign[:, None] * offset / safe[:, None]
    for i in np.flatnonzero(degenerate):
        gradient[i] = _outward_normal(
            coords[i], geometries[i], targets[i], bool(is_area[i])
        )
    return sign * length, gradient


def _outward_normal(
    point: FloatArray, geometry: BaseGeometry, target: BaseGeometry, is_area: bool
) -> tuple[float, float]:
    """The unit normal of `target` at a point lying on it, outward for areas."""
    if not isinstance(target, LineString | MultiLineString):
        return FALLBACK_DIRECTION
    along = target.project(Point(point))
    before = target.interpolate(max(along - NORMAL_PROBE, 0.0))
    after = target.interpolate(along + NORMAL_PROBE)
    tx, ty = after.x - before.x, after.y - before.y
    norm = float(np.hypot(tx, ty))
    if norm == 0.0:
        return FALLBACK_DIRECTION
    nx, ny = -ty / norm, tx / norm
    probe = Point(point[0] + NORMAL_PROBE * nx, point[1] + NORMAL_PROBE * ny)
    if is_area and geometry.contains(probe):
        return -nx, -ny
    return nx, ny


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
        lines: ObjectArray = np.empty(0, dtype=object)
        if len(ends):
            lines = np.asarray(
                shapely.linestrings(
                    np.stack(
                        [
                            np.column_stack([self._starts.real, self._starts.imag]),
                            np.column_stack([ends.real, ends.imag]),
                        ],
                        axis=1,
                    )
                ),
                dtype=object,
            )
        return StaticEdges(
            lines,
            np.full(len(self._inflations), LINE, dtype=np.intp),
            self._inflations,
            self._obstacles,
        )

    @staticmethod
    def merged(fields: Sequence[ChainField]) -> ChainField:
        """One field answering as the union of `fields` does, with one tree."""
        starts = np.concatenate([f._starts for f in fields])
        along = np.concatenate([f._along for f in fields])
        counts = [int(f._obstacles.max(initial=-1)) + 1 for f in fields]
        offsets = np.cumsum([0, *counts[:-1]])
        return ChainField(
            np.column_stack([starts.real, starts.imag]),
            np.column_stack([(starts + along).real, (starts + along).imag]),
            np.concatenate([f._inflations for f in fields]),
            np.concatenate(
                [
                    np.empty(0, dtype=np.intp),
                    *(f._obstacles + o for f, o in zip(fields, offsets, strict=True)),
                ]
            ),
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
    gap = _segment_gaps(a, ab, starts, along)
    return gap < inflations - tolerance - ROUND_OFF


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
    distance = np.full(len(query), np.inf)
    gradient = np.zeros((len(query), 2))
    edges = np.full(len(query), -1, dtype=np.intp)
    if not len(point):
        return NearestEdge(distance, gradient, edges)
    offset = _offsets(query[point], starts, along)
    raw = np.abs(offset)
    candidate = np.where(raw <= limit + DEGENERATE_DISTANCE, raw - inflations, np.inf)
    np.minimum.at(distance, point, candidate)
    best = np.flatnonzero((candidate == distance[point]) & (candidate < np.inf))
    gradient[point[best]] = _unit_gradient(offset[best], raw[best], along[best])
    edges[point[best]] = best
    return NearestEdge(distance, gradient, edges)


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
    count = len(query)
    if not len(point):
        return _no_answers(count)
    offset = _offsets(query[point], starts, along)
    raw = np.abs(offset)
    distance = np.where(raw <= limit + DEGENERATE_DISTANCE, raw - inflations, np.inf)
    best, smooth, share = _nearest_then_smooth(point, obstacle, distance, count)
    at = point[best]
    direction = _unit_gradient(offset[best], raw[best], along[best])
    return SmoothAnswer(
        smooth, weighted_sum(at, share, direction, count), at, best, share, direction
    )


def _nearest_then_smooth(
    point: IntArray, obstacle: IntArray, distance: FloatArray, count: int
) -> tuple[IntArray, FloatArray, FloatArray]:
    """Of candidates in the order `obstacle_order` gives, candidate k at
    distance[k] from point[k] and part of obstacle[k]: each obstacle's
    nearest at each point, of equal ones the first, by index, those that
    have a share; the smooth minimum over them at each of `count` points
    (`energy.smooth_minimum`); and their shares."""
    best = _nearest_in_groups(distance, *_groups(point, obstacle))
    smooth, share = smooth_minimum(point[best], distance[best], count)
    counted = np.flatnonzero(share > 0.0)
    return best[counted], smooth, share[counted]


def obstacle_order(point: IntArray, obstacle: IntArray, tiebreak: IntArray) -> IntArray:
    """The order of (point, segment) pairs, pair k's segment part of
    obstacle[k], that `smooth_of_pairs` takes them in: by point, then by
    obstacle, then by tiebreak[k], unique among a point's pairs, such as
    the segment's index. Every pair's key differs, so the order is the
    same whatever order the pairs come in, and of equally near segments of
    an obstacle the lowest tiebreak answers."""
    tiebreaks = int(tiebreak.max(initial=0)) + 1
    obstacles = int(obstacle.max(initial=0)) + 1
    key = (point.astype(np.int64) * obstacles + obstacle) * tiebreaks + tiebreak
    # Unique keys need no stable sort, which costs twice as much.
    return np.argsort(key)


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
    raw = np.abs(_offsets(query[point], starts, along))
    keep = raw <= within + slack + 2 * DEGENERATE_DISTANCE
    # Only obstacles within SMOOTH_REACH of the nearest answer a point. An
    # edge that stays within reach however its distance changes bounds how
    # far away the nearest is, and an edge that cannot come within
    # SMOOTH_REACH of the bound never answers.
    distance = raw - inflations
    reached = np.where(raw + slack <= within, distance + slack, np.inf)
    bound = np.full(len(query), np.inf)
    np.minimum.at(bound, point, reached)
    keep &= distance - slack <= bound[point] + SMOOTH_REACH + 2 * DEGENERATE_DISTANCE
    return keep


def _nearest_in_groups(
    distance: FloatArray, first: IntArray, group: IntArray
) -> IntArray:
    """Of each run of pairs `_groups` gives, the first of the nearest, by
    index."""
    lowest = np.minimum.reduceat(distance, first)
    nearest = np.flatnonzero(distance == lowest[group])
    return nearest[np.flatnonzero(np.diff(group[nearest], prepend=-1) != 0)]


def _groups(point: IntArray, obstacle: IntArray) -> tuple[IntArray, IntArray]:
    """Where each run of pairs of one point and one obstacle starts, of
    pairs in the order `obstacle_order` gives, and each pair's run."""
    starting = np.ones(len(point), dtype=bool)
    starting[1:] = (point[1:] != point[:-1]) | (obstacle[1:] != obstacle[:-1])
    return np.flatnonzero(starting), np.cumsum(starting) - 1


def near_segments(
    starts: FloatArray, ends: FloatArray, points: FloatArray, radius: float
) -> tuple[IntArray, IntArray]:
    """(point, segment) index pairs, among them every pair whose segment
    from starts[k] to ends[k] comes within `radius` of points[i], found with
    one tree over the segments' middles."""
    empty = np.empty(0, dtype=np.intp)
    if not len(starts) or not len(points):
        return empty, empty
    half_length = float(np.max(np.hypot(*(ends - starts).T))) / 2
    tree = cKDTree((starts + ends) / 2)
    return _pairs_within(tree, points, radius + half_length + DEGENERATE_DISTANCE)


def _unit_gradient(
    offset: ComplexArray, length: FloatArray, along: ComplexArray
) -> FloatArray:
    """The direction away from each segment along `along`; its normal on
    the segment."""
    degenerate = length < DEGENERATE_DISTANCE
    direction = offset / np.where(degenerate, 1.0, length)
    if degenerate.any():
        along = along[degenerate]
        norm = np.abs(along)
        # A quarter turn of the tangent, as `_outward_normal` turns it.
        direction[degenerate] = np.where(
            norm > 0.0,
            1j * along / np.where(norm > 0.0, norm, 1.0),
            complex(*FALLBACK_DIRECTION),
        )
    return np.column_stack([direction.real, direction.imag])


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


def _segment_gaps(
    a: ComplexArray, ab: ComplexArray, c: ComplexArray, cd: ComplexArray
) -> FloatArray:
    """Distance between each segment from a[i] along ab[i] and the segment
    from c[i] along cd[i]."""

    def cross(u: ComplexArray, v: ComplexArray) -> FloatArray:
        return u.real * v.imag - u.imag * v.real

    b, d = a + ab, c + cd
    crossing = (cross(ab, c - a) * cross(ab, d - a) < 0.0) & (
        cross(cd, a - c) * cross(cd, b - c) < 0.0
    )
    # Segments that do not cross are nearest at an endpoint of one of them.
    nearest_end = np.minimum.reduce(
        [
            np.abs(_offsets(c, a, ab)),
            np.abs(_offsets(d, a, ab)),
            np.abs(_offsets(a, c, cd)),
            np.abs(_offsets(b, c, cd)),
        ]
    )
    return np.where(crossing, 0.0, nearest_end)


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
    net: str, width: float, clearance: float, other: ChainCentreline
) -> float | None:
    """The distance a track of `net`, `width` and netclass `clearance` (mm)
    keeps its centreline from `other`'s, mm: both half widths and the larger
    of both clearances, as from an unselected track; None for a chain of its
    net, since copper of one net may come close, overlap or cross."""
    if other.net == net:
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


def build_via_obstacles(
    layers: Sequence[LayerSnapshot],
    via: Via,
    *,
    moving: Collection[str] = frozenset(),
    edge_clearance: float = EDGE_CLEARANCE,
    hole_clearance: float = HOLE_CLEARANCE,
    hole_to_hole: float = HOLE_TO_HOLE,
    zones: Sequence[ViaZone] = (),
) -> ObstacleField:
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
    """
    found = _ViaSurroundings.of(layers, via, moving, hole_clearance, hole_to_hole)
    return _via_field(found, via, found.others, edge_clearance, zones)


def split_via_obstacles(
    layers: Sequence[LayerSnapshot],
    via: Via,
    split: Callable[[Via], bool],
    *,
    moving: Collection[str] = frozenset(),
    edge_clearance: float = EDGE_CLEARANCE,
    hole_clearance: float = HOLE_CLEARANCE,
    hole_to_hole: float = HOLE_TO_HOLE,
    zones: Sequence[ViaZone] = (),
) -> tuple[ObstacleField, Obstacles | None]:
    """`build_via_obstacles` without the vias of other nets that `split`
    picks, and those vias as obstacles to `via`, each as the field would
    count it, or None if it picks none; both from one reading of the
    layers."""
    found = _ViaSurroundings.of(layers, via, moving, hole_clearance, hole_to_hole)
    picked = [v for v in found.others if split(v)]
    kept = [v for v in found.others if not split(v)]
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
    """`build_via_obstacles` of `via` in `found`, with only `others` of the
    vias of other nets there."""
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
    if outline is None:
        return Obstacles(items)
    return Obstacles(items, outline, edge_clearance + keeper.radius)


@dataclass(frozen=True)
class _ViaSurroundings:
    """The layers a moving via spans, its keeper, and the other-net vias
    there that stand still."""

    spanned: Sequence[LayerSnapshot]
    keeper: _Keeper
    others: Sequence[Via]

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
        keeper = _via_keeper(
            via, spanned[0].clearance(via.net), hole_clearance, hole_to_hole
        )
        others = {
            v.id: v
            for s in spanned
            for v in s.vias
            if v.id not in left_out and v.net != via.net
        }
        return cls(spanned, keeper, tuple(others.values()))

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
    spanned = [s for s in layers if via.reaches(s.layer)]
    if not spanned:
        raise ValueError(f"no layer that via {via.id} spans was given")
    left_out = {via.id, *moving}
    own = {
        v.id: v
        for s in spanned
        for v in s.vias
        if v.id not in left_out and v.net == via.net
    }.values()
    clearance = spanned[0].clearance(via.net)
    keeper = _via_keeper(via, clearance, hole_clearance, hole_to_hole)
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
    vias = _vias_field(keeper, own, lambda _: clearance)
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


def contained(field: ObstacleField, zones: Iterable[ViaZone]) -> ObstacleField:
    """`field` with the outside of the regions of `zones`: every limit on
    where a via may go."""
    inside = containment(zones)
    return field if inside is None else CompositeField([field, inside])


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
    `build_via_obstacles` counts a via on the board; None if no rule
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
