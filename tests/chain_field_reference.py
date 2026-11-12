"""The Python ChainField and chains_field as they were before the port, for
the parity test; deleted with it."""

import itertools
import math
from collections.abc import Iterable, Sequence
from functools import cached_property

import numpy as np
from energy_reference import SMOOTH_REACH, BoolArray, FloatArray, IntArray, smooth_reach
from scipy.spatial import cKDTree

from kibends.obstacles import (
    DEGENERATE_DISTANCE,
    MAX_EDGE_LENGTH,
    ChainCentreline,
    ComplexArray,
    NearestEdge,
    SmoothAnswer,
    StaticEdges,
    complex_points,
    nearest_of_pairs,
    obstacle_order,
    pairs_inside,
    smooth_of_pairs,
)


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
