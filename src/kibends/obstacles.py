"""Inflated obstacles on the chain's layer, answering distance queries.

Each obstacle is raw geometry (a copper polygon, a track centreline, a via or
hole centre) plus an inflation: the distance from that geometry at which the
chain's centreline reaches the required clearance. Distances are measured to
the raw geometry and the inflation subtracted, which is the exact signed
distance to the inflated shape without buffering it. The board edge is the
outside of the Edge.Cuts outline, inflated the same way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import shapely
from loguru import logger
from numpy.typing import NDArray
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from kibends.energy import BoolArray, FloatArray
from kibends.snapshot import (
    LayerSnapshot,
    PadKind,
    edge_paths_closed,
    outline_from_edge_paths,
)

ObjectArray = NDArray[np.object_]

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
        self._is_area: BoolArray = (
            shapely.get_dimensions(self._geometries) == AREA_DIMENSION
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
        query: ObjectArray = np.asarray(shapely.points(coords), dtype=object)
        distance = np.full(len(coords), np.inf)
        gradient = np.zeros((len(coords), 2))
        if self.items:
            distance, gradient = self._obstacle_distance(coords, query, reach)
        if self.board_outline is not None:
            near = np.ones(len(coords), dtype=bool)
            if math.isfinite(reach):
                outside = ~shapely.covered_by(query, self.board_outline)
                to_edge = shapely.distance(query, self.board_outline.boundary)
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

    def segments_clear(self, starts: FloatArray, ends: FloatArray) -> BoolArray:
        """Whether each segment from starts[i] to ends[i] stays out of every
        inflated obstacle and inside the inflated board edge.

        Touching an inflated boundary, which is exactly at clearance, is clear.
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
            gaps = shapely.distance(
                segments[segment_index], self._geometries[item_index]
            )
            clear[segment_index[gaps < self._inflations[item_index]]] = False
        if self.board_outline is not None:
            clear &= shapely.covered_by(segments, self.board_outline)
            clear &= (
                shapely.distance(segments, self.board_outline.boundary)
                >= self.edge_inflation
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
    lines = shapely.shortest_line(query, targets)
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
    along = shapely.line_locate_point(target, Point(point))
    before = shapely.line_interpolate_point(target, max(along - NORMAL_PROBE, 0.0))
    after = shapely.line_interpolate_point(target, along + NORMAL_PROBE)
    tx, ty = after.x - before.x, after.y - before.y
    norm = float(np.hypot(tx, ty))
    if norm == 0.0:
        return FALLBACK_DIRECTION
    nx, ny = -ty / norm, tx / norm
    probe = Point(point[0] + NORMAL_PROBE * nx, point[1] + NORMAL_PROBE * ny)
    if is_area and geometry.contains(probe):
        return -nx, -ny
    return nx, ny


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
