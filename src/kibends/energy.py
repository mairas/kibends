"""Energy terms of a track polyline, each returning its value and gradient;
the terms themselves are Rust's (`kibends._core`; their design is in
`kibends_core::energy`).

Polylines are (n, 2) float arrays in mm. Every term returns the gradient with
respect to all vertex coordinates as an (n, 2) array, with the rows of fixed
vertices zeroed. The obstacle terms take a smooth minimum over the obstacles
near a point (`smooth_minimum`), and every clearance judgement the exact
distance (`DistanceField.signed_distance`).

This module keeps the fields' Protocols and the energy's parameters, which
Python code builds and changes with `dataclasses.replace`.
"""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from kibends import _core

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.intp]
TermValue = tuple[float, FloatArray]
Integrand = _core.Integrand
"""A function of the distance, called with distances: its value and slope."""
Curvature = _core.Curvature
"""An integrand's second derivative by the distance."""
PairIntegrand = _core.PairIntegrand

DEFAULT_ATTRACTION_RANGE = _core.DEFAULT_ATTRACTION_RANGE
"""Centre distance at which a ground via and a regular one stop attracting
each other, mm."""
MIN_EDGE_LENGTH = _core.MIN_EDGE_LENGTH
"""1 nm, KiCad's coordinate resolution: shorter edges are treated as
degenerate."""
MAX_SMOOTH_TURN = _core.MAX_SMOOTH_TURN
"""Beyond this turning angle the curvature 2·tan(φ/2) is continued
linearly in φ."""
SMOOTH_WIDTH = _core.SMOOTH_WIDTH
"""τ of the smooth minimum over obstacles (`smooth_minimum`), mm: two
equally near obstacles look τ·ln 2, 3.5 µm, nearer than either."""
SMOOTH_REACH = _core.SMOOTH_REACH
"""How much farther than the nearest an obstacle still counts in the smooth
minimum, and how far beyond the energy's reach obstacles are looked for,
mm."""


class DistanceField(Protocol):
    """Signed distance to the nearest inflated obstacle boundary."""

    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """Return the distance (n,) and its unit gradient (n, 2) at (n, 2) points.

        The distance is positive in free space and negative inside an inflated
        obstacle; the gradient points toward increasing distance. Distances
        beyond `reach` may be returned as +inf with a zero gradient.
        """
        ...

    def smooth_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """`signed_distance` as the energy takes it: the smooth minimum over
        the field's obstacles (`smooth_minimum`) of each one's exact
        distance, and its gradient, the obstacles' unit gradients weighted
        by their shares. Obstacles farther than `reach` plus SMOOTH_REACH
        may be left out, and a point with none gets +inf."""
        ...


class ObstacleField(DistanceField, Protocol):
    """Distance field that can also check segments exactly."""

    def segments_clear(
        self, starts: FloatArray, ends: FloatArray, tolerance: float = 0.0
    ) -> BoolArray:
        """Return per segment whether it stays out of every inflated obstacle.

        starts and ends are (n, 2) arrays in mm. A segment is clear when it
        does not enter the region where the signed distance is below
        -tolerance; touching the boundary is clear.
        """
        ...


@dataclass(frozen=True)
class EnergyParams:
    """Weights and shape parameters of the total energy; length has weight 1."""

    repulsion: float
    falloff_range: float
    """Distance beyond the clearance boundary at which repulsion reaches 0, mm."""
    clearance_weight: float
    standoff: float
    """Distance from the clearance boundary at which the hinge starts, mm."""
    points_per_edge: int
    """Interior quadrature points per edge, in addition to its endpoints."""
    attraction: float = 0.0
    """Weight of a ground via and a regular one pulling together; 0 turns
    it off (`attraction_integrand`)."""
    attraction_range: float = DEFAULT_ATTRACTION_RANGE
    """Centre distance at which that pull reaches 0, mm."""


arc_lengths = _core.arc_lengths
attraction_integrand = _core.attraction_integrand
bend_radius_energy = _core.bend_radius_energy
combine_fields = _core.combine_fields
edge_bend_radii = _core.edge_bend_radii
edge_curvature = _core.edge_curvature
edge_integral = _core.edge_integral
hinge_curvature = _core.hinge_curvature
hinge_integrand = _core.hinge_integrand
obstacle_curvature = _core.obstacle_curvature
obstacle_integrand = _core.obstacle_integrand
obstacle_reach = _core.obstacle_reach
point_stiffness = _core.point_stiffness
quadrature_points = _core.quadrature_points
repulsion_integrand = _core.repulsion_integrand
shape_energy = _core.shape_energy
smooth_minimum = _core.smooth_minimum
smooth_reach = _core.smooth_reach
weighted_sum = _core.weighted_sum
