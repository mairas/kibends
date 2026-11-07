"""Analytic DistanceField and ObstacleField implementations for tests."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from kibends.energy import BoolArray, FloatArray

FAR = 1e3
"""Distance reported by FreeSpace, beyond any falloff range used in tests, mm."""


def _point_segment_distance(
    points: FloatArray, starts: FloatArray, ends: FloatArray
) -> FloatArray:
    edge = ends - starts
    squared = np.einsum("ij,ij->i", edge, edge)
    t = np.einsum("ij,ij->i", points - starts, edge) / np.where(
        squared > 0.0, squared, 1.0
    )
    closest: FloatArray = starts + np.clip(t, 0.0, 1.0)[:, None] * edge
    return np.asarray(np.linalg.norm(points - closest, axis=1))


@dataclass(frozen=True)
class FreeSpace:
    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        return np.full(len(points), FAR), np.zeros_like(points)

    def segments_clear(self, starts: FloatArray, ends: FloatArray) -> BoolArray:
        return np.ones(len(starts), dtype=bool)


@dataclass(frozen=True)
class Disc:
    cx: float
    cy: float
    radius: float

    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        offset = points - np.array([self.cx, self.cy])
        norm = np.linalg.norm(offset, axis=1)
        safe = np.where(norm > 0.0, norm, 1.0)
        direction = np.where(norm[:, None] > 0.0, offset / safe[:, None], [1.0, 0.0])
        return norm - self.radius, direction

    def segments_clear(self, starts: FloatArray, ends: FloatArray) -> BoolArray:
        centre = np.broadcast_to([self.cx, self.cy], starts.shape)
        return _point_segment_distance(centre, starts, ends) >= self.radius


@dataclass(frozen=True)
class Box:
    """Axis-aligned box given by its centre and half-sizes.

    A corner radius rounds the corners, as inflating a sharp box by that
    radius would; the half-sizes include the rounding.
    """

    cx: float
    cy: float
    half_x: float
    half_y: float
    corner_radius: float = 0.0

    def _core(self) -> FloatArray:
        return np.array([self.half_x, self.half_y]) - self.corner_radius

    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        offset = points - np.array([self.cx, self.cy])
        sign = np.where(offset >= 0.0, 1.0, -1.0)
        q = np.abs(offset) - self._core()
        outside = np.maximum(q, 0.0)
        outside_norm = np.linalg.norm(outside, axis=1)
        is_outside = outside_norm > 0.0
        safe = np.where(is_outside, outside_norm, 1.0)
        outside_grad = sign * outside / safe[:, None]
        inside_axis = np.argmax(q, axis=1)
        inside_grad = np.zeros_like(points)
        rows = np.arange(len(points))
        inside_grad[rows, inside_axis] = sign[rows, inside_axis]
        distance = np.where(is_outside, outside_norm, np.max(q, axis=1))
        gradient = np.where(is_outside[:, None], outside_grad, inside_grad)
        return distance - self.corner_radius, gradient

    def segments_clear(self, starts: FloatArray, ends: FloatArray) -> BoolArray:
        core = self._core()
        centre = np.array([self.cx, self.cy])
        low, high = centre - core, centre + core
        enter, leave = _clip_to_rectangle(starts, ends, low, high)
        meets_core = enter <= leave
        if self.corner_radius == 0.0:
            t = np.where(meets_core, (enter + leave) / 2, 0.0)
            middle = starts + t[:, None] * (ends - starts)
            strictly_inside = np.all((middle > low) & (middle < high), axis=1)
            return np.asarray(~(meets_core & strictly_inside))
        corners = (low, np.array([low[0], high[1]]), high, np.array([high[0], low[1]]))
        corner_distance = np.min(
            [
                _point_segment_distance(
                    np.broadcast_to(corner, starts.shape), starts, ends
                )
                for corner in corners
            ],
            axis=0,
        )
        endpoint_distance = np.minimum(
            self._outside_core_distance(starts), self._outside_core_distance(ends)
        )
        distance = np.minimum(corner_distance, endpoint_distance)
        return np.asarray(~meets_core & (distance >= self.corner_radius))

    def _outside_core_distance(self, points: FloatArray) -> FloatArray:
        offset = np.abs(points - np.array([self.cx, self.cy])) - self._core()
        return np.asarray(np.linalg.norm(np.maximum(offset, 0.0), axis=1))


def _clip_to_rectangle(
    starts: FloatArray, ends: FloatArray, low: FloatArray, high: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Liang-Barsky: parameter interval of each segment inside the closed box.

    The segment misses the box where the returned enter exceeds leave.
    """
    delta = ends - starts
    enter = np.zeros(len(starts))
    leave = np.ones(len(starts))
    for axis in range(2):
        for p, q in (
            (-delta[:, axis], starts[:, axis] - low[axis]),
            (delta[:, axis], high[axis] - starts[:, axis]),
        ):
            parallel = p == 0.0
            t = q / np.where(parallel, 1.0, p)
            enter = np.where(~parallel & (p < 0.0), np.maximum(enter, t), enter)
            leave = np.where(~parallel & (p > 0.0), np.minimum(leave, t), leave)
            leave = np.where(parallel & (q < 0.0), -np.inf, leave)
    return enter, leave


@dataclass(frozen=True)
class Walls:
    """Two walls parallel to the x axis at y = ±half_gap; free space between."""

    half_gap: float

    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        y = points[:, 1]
        toward_centre = np.where(y >= 0.0, -1.0, 1.0)
        gradient = np.stack([np.zeros_like(y), toward_centre], axis=1)
        return self.half_gap - np.abs(y), gradient

    def segments_clear(self, starts: FloatArray, ends: FloatArray) -> BoolArray:
        inside = np.abs(np.stack([starts[:, 1], ends[:, 1]])) <= self.half_gap
        return np.asarray(np.all(inside, axis=0))


@dataclass(frozen=True)
class Union:
    """Nearest of several obstacles; exact in free space."""

    shapes: Sequence[Disc | Box | Walls]

    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        results = [shape.signed_distance(points) for shape in self.shapes]
        distances = np.stack([distance for distance, _ in results])
        gradients = np.stack([gradient for _, gradient in results])
        nearest = np.argmin(distances, axis=0)
        rows = np.arange(len(points))
        return distances[nearest, rows], gradients[nearest, rows]

    def segments_clear(self, starts: FloatArray, ends: FloatArray) -> BoolArray:
        clear = [shape.segments_clear(starts, ends) for shape in self.shapes]
        return np.asarray(np.all(clear, axis=0))
