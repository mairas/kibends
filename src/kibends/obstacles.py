"""Inflated obstacles on the chain's layer, answering distance queries.

Each obstacle is raw geometry (a copper polygon, a track centreline, a via or
hole centre) plus an inflation: the distance from that geometry at which the
chain's centreline reaches the required clearance. Distances are measured to
the raw geometry and the inflation subtracted, which is the exact signed
distance to the inflated shape without buffering it. The board edge is the
outside of the Edge.Cuts outline, inflated the same way.
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import shapely
from loguru import logger
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from kibends.energy import BoolArray, FloatArray, ObstacleField
from kibends.snapshot import (
    LayerSnapshot,
    PadKind,
    edge_paths_closed,
    outline_from_edge_paths,
)

ObjectArray = NDArray[np.object_]
IntArray = NDArray[np.intp]
ComplexArray = NDArray[np.complex128]

# The API cannot report the board's copper-to-edge and copper-to-hole rules.
# 0.3 mm to the edge meets common fabs' minimum for routed outlines, and
# 0.25 mm to holes is the hole clearance KiCad's own project templates set.
# The kicad-cli DRC check enforces the board's real rules either way.
EDGE_CLEARANCE = 0.3
HOLE_CLEARANCE = 0.25

# Closer than this to the nearest feature, the direction to it is noise; the
# boundary's normal is used instead. Far below KiCad's 1 nm resolution.
DEGENERATE_DISTANCE = 1e-9
# Step along a boundary to find its tangent, and off it to tell inside from out.
NORMAL_PROBE = 1e-4
FALLBACK_DIRECTION = (1.0, 0.0)
AREA_DIMENSION = 2
# Other chains' edges are split to at most this length, so that a point near an
# edge is also near its midpoint, where the k-d tree finds it. Chains resampled
# at the default 0.1 mm `OptimizerSettings.spacing` are never split.
MAX_EDGE_LENGTH = 0.5


@dataclass(frozen=True)
class Obstacle:
    description: str
    geometry: BaseGeometry
    """Polygonal copper, or a centreline or centre point to be inflated."""
    inflation: float
    """Distance from `geometry` at which the chain's centreline is at clearance."""


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
        distance = np.full(len(coords), np.inf)
        gradient = np.zeros((len(coords), 2))
        if not self.items and self.board_outline is None:
            return distance, gradient
        query: ObjectArray = np.asarray(shapely.points(coords), dtype=object)
        if self.items:
            distance, gradient = self._obstacle_distance(coords, query, reach)
        if self.board_outline is not None:
            near = np.ones(len(coords), dtype=bool)
            if math.isfinite(reach):
                outside = ~shapely.covered_by(query, self.board_outline)
                to_edge = self.board_outline.boundary.distance(query)
                near = outside | (to_edge - self.edge_inflation <= reach)
            inside_board, toward_inside = _signed_to(
                coords[near],
                query[near],
                np.full(near.sum(), self.board_outline, dtype=object),
                np.full(near.sum(), self.board_outline.boundary, dtype=object),
                np.ones(near.sum(), dtype=bool),
            )
            edge = np.full(len(coords), np.inf)
            edge[near] = -inside_board - self.edge_inflation
            edge_gradient = np.zeros((len(coords), 2))
            edge_gradient[near] = -toward_inside
            nearer = edge < distance
            distance = np.where(nearer, edge, distance)
            gradient = np.where(nearer[:, None], edge_gradient, gradient)
        return distance, gradient

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

    def _obstacle_distance(
        self, coords: FloatArray, query: ObjectArray, reach: float
    ) -> tuple[FloatArray, FloatArray]:
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
        candidate = raw - self._inflations[pairs_item]
        order = np.lexsort((candidate, pairs_point))
        _, first = np.unique(pairs_point[order], return_index=True)
        best = order[first]
        distance = np.full(len(coords), np.inf)
        gradient = np.zeros((len(coords), 2))
        distance[pairs_point[best]] = candidate[best]
        gradient[pairs_point[best]] = raw_gradient[best]
        return distance, gradient


class CompositeField:
    """The union of several fields: the nearest of them answers each point.

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

    def segments_clear(
        self, starts: FloatArray, ends: FloatArray, tolerance: float = 0.0
    ) -> BoolArray:
        """Whether each segment from starts[i] to ends[i] is clear in every
        field, each allowing `tolerance` inside its clearance."""
        clear = np.ones(len(np.asarray(starts).reshape(-1, 2)), dtype=np.bool_)
        for field in self.fields:
            clear &= field.segments_clear(starts, ends, tolerance)
        return clear


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


class ChainField:
    """Signed distance to other chains' inflated centrelines, and edge checks.

    The same answers as `Obstacles` with each centreline as an obstacle,
    except that zero-length edges and zero-length query segments count like
    any other, where shapely's tree query skips them. Done in numpy instead
    of shapely because the optimizer rebuilds this field for every burst and
    queries it on every energy evaluation. A k-d tree over the edge midpoints
    finds the edges near each query, and the exact distances to those edges
    are computed in bulk. Points are complex numbers inside, which keeps
    every gather one-dimensional.
    """

    def __init__(
        self, starts: FloatArray, ends: FloatArray, inflations: FloatArray
    ) -> None:
        """Edges from starts[i] to ends[i], (m, 2) in mm, each inflated by
        inflations[i]."""
        self._starts = _complex(starts)
        self._along = _complex(ends) - self._starts
        self._inflations = inflations
        self._max_inflation = float(inflations.max(initial=0.0))
        self._max_half_length = float(np.abs(self._along).max(initial=0.0)) / 2
        self._tree = cKDTree((starts + ends) / 2) if len(starts) else None

    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """Distance (n,) and its unit gradient (n, 2) at (n, 2) points in mm.

        With no edges the distance is +inf and the gradient zero everywhere.
        Points farther than `reach` from every inflated edge get +inf and a
        zero gradient.
        """
        coords = np.asarray(points, dtype=float).reshape(-1, 2)
        distance = np.full(len(coords), np.inf)
        gradient = np.zeros((len(coords), 2))
        if self._tree is None or not len(coords):
            return distance, gradient
        query = _complex(coords)
        if math.isfinite(reach):
            # An edge is within reach once inflated only if it is within reach
            # plus the largest inflation.
            limit = reach + self._max_inflation
            point, edge = _pairs_within(
                self._tree, coords, limit + self._max_half_length + DEGENERATE_DISTANCE
            )
            point_limit: FloatArray | float = limit
        else:
            _, nearest = self._tree.query(coords)
            nearest_raw = np.abs(
                _offsets(query, self._starts[nearest], self._along[nearest])
            )
            # An edge farther than the nearest one can still be nearer once
            # inflated, by at most the spread of the inflations.
            limits = nearest_raw - self._inflations[nearest] + self._max_inflation
            point, edge = _pairs_each_within(
                self._tree, coords, limits + self._max_half_length + DEGENERATE_DISTANCE
            )
            point_limit = limits[point]
        if not len(point):
            return distance, gradient
        offset = _offsets(query[point], self._starts[edge], self._along[edge])
        raw = np.abs(offset)
        candidate = np.where(
            raw <= point_limit + DEGENERATE_DISTANCE,
            raw - self._inflations[edge],
            np.inf,
        )
        np.minimum.at(distance, point, candidate)
        best = np.flatnonzero((candidate == distance[point]) & (candidate < np.inf))
        gradient[point[best]] = self._unit_gradient(offset[best], raw[best], edge[best])
        return distance, gradient

    def segments_clear(
        self, starts: FloatArray, ends: FloatArray, tolerance: float = 0.0
    ) -> BoolArray:
        """Whether each segment from starts[i] to ends[i] stays out of every
        inflated edge.

        Touching an inflated boundary, which is exactly at clearance, is clear,
        and so is entering it by at most `tolerance` mm.
        """
        a = _complex(np.asarray(starts, dtype=float).reshape(-1, 2))
        b = _complex(np.asarray(ends, dtype=float).reshape(-1, 2))
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
        gap = _segment_gaps(
            a[segment], b[segment] - a[segment], self._starts[edge], self._along[edge]
        )
        clear[segment[gap < self._inflations[edge] - tolerance]] = False
        return clear

    def _unit_gradient(
        self, offset: ComplexArray, length: FloatArray, edge: IntArray
    ) -> FloatArray:
        """The direction away from each edge; its normal on the centreline."""
        degenerate = length < DEGENERATE_DISTANCE
        direction = offset / np.where(degenerate, 1.0, length)
        if degenerate.any():
            along = self._along[edge[degenerate]]
            norm = np.abs(along)
            # A quarter turn of the tangent, as `_outward_normal` turns it.
            direction[degenerate] = np.where(
                norm > 0.0,
                1j * along / np.where(norm > 0.0, norm, 1.0),
                complex(*FALLBACK_DIRECTION),
            )
        return np.column_stack([direction.real, direction.imag])


def _complex(points: FloatArray) -> ComplexArray:
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


def chain_obstacles(
    net: str, width: float, clearance: float, others: Sequence[ChainCentreline]
) -> ChainField:
    """The field for a track of `net`, `width` and netclass `clearance` (mm)
    made of other chains' centrelines.

    Another net's chain is kept at the larger of both clearances. A chain of
    the same net may touch this one but not overlap it, since DRC does not
    flag same-net overlap and both would otherwise be pulled onto each other.
    """
    half_width = width / 2
    starts: list[FloatArray] = [np.empty((0, 2))]
    ends: list[FloatArray] = [np.empty((0, 2))]
    inflations: list[FloatArray] = [np.empty(0)]
    for o in others:
        inflation = (
            half_width
            + o.width / 2
            + (max(clearance, o.clearance) if o.net != net else 0.0)
        )
        a, b = _split_edges(np.asarray(o.points, dtype=float).reshape(-1, 2))
        starts.append(a)
        ends.append(b)
        inflations.append(np.full(len(a), inflation))
    return ChainField(
        np.concatenate(starts), np.concatenate(ends), np.concatenate(inflations)
    )


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
    half_width = width / 2
    own = snapshot.clearance(net)

    def copper(other_clearance: float) -> float:
        return max(own, other_clearance) + half_width

    items: list[Obstacle] = []
    for pad in snapshot.pads:
        other_net = pad.net != net
        if other_net and pad.shape is not None:
            clearance = (
                pad.clearance_override
                if pad.clearance_override is not None
                else snapshot.clearance(pad.net)
            )
            items.append(Obstacle(f"pad {pad.name}", pad.shape, copper(clearance)))
        if pad.drill > 0 and (other_net or pad.kind is PadKind.NPTH):
            items.append(
                Obstacle(
                    f"hole of pad {pad.name}",
                    Point(pad.position),
                    pad.drill / 2 + hole_clearance + half_width,
                )
            )
    for t in snapshot.tracks:
        if t.net != net:
            inflation = t.width / 2 + copper(snapshot.clearance(t.net))
            items.append(Obstacle(f"track {t.id}", t.centreline, inflation))
    for v in snapshot.vias:
        if v.net != net:
            centre = Point(v.position)
            inflation = v.diameter / 2 + copper(snapshot.clearance(v.net))
            items.append(Obstacle(f"via {v.id}", centre, inflation))
            items.append(
                Obstacle(
                    f"hole of via {v.id}",
                    centre,
                    v.drill / 2 + hole_clearance + half_width,
                )
            )
    for g in snapshot.graphics:
        if g.net != net:
            items.append(
                Obstacle(g.description, g.geometry, copper(snapshot.clearance(g.net)))
            )
    items.extend(Obstacle("track keepout", k, half_width) for k in snapshot.keepouts)

    if not edge_paths_closed(snapshot.edge_paths):
        logger.warning(
            "Edge.Cuts has no closed outline; the board edge is not kept clear "
            "while smoothing, only the DRC check covers it"
        )
        return Obstacles(items)
    return Obstacles(
        items,
        outline_from_edge_paths(snapshot.edge_paths),
        edge_clearance + half_width,
    )
