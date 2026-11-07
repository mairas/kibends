"""Analytic DistanceField implementations for energy tests."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Disc:
    cx: float
    cy: float
    radius: float

    def signed_distance(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        offset = points - np.array([self.cx, self.cy])
        norm = np.linalg.norm(offset, axis=1)
        safe = np.where(norm > 0.0, norm, 1.0)
        direction = np.where(norm[:, None] > 0.0, offset / safe[:, None], [1.0, 0.0])
        return norm - self.radius, direction


@dataclass(frozen=True)
class Box:
    """Axis-aligned box given by its centre and half-sizes."""

    cx: float
    cy: float
    half_x: float
    half_y: float

    def signed_distance(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        offset = points - np.array([self.cx, self.cy])
        sign = np.where(offset >= 0.0, 1.0, -1.0)
        q = np.abs(offset) - np.array([self.half_x, self.half_y])
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
        return distance, gradient


@dataclass(frozen=True)
class Walls:
    """Two walls parallel to the x axis at y = ±half_gap; free space between."""

    half_gap: float

    def signed_distance(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        y = points[:, 1]
        toward_centre = np.where(y >= 0.0, -1.0, 1.0)
        gradient = np.stack([np.zeros_like(y), toward_centre], axis=1)
        return self.half_gap - np.abs(y), gradient
