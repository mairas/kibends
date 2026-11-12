"""The terms that keep a chain on its contacts (`neighbours.ChainInput.contacts`).

The terms are Rust's (`kibends_core::contacts`): a vertex that keeps a
pad or a via that stays is held inside its region by a stop that limits
without repelling (`contact_stops`), and after a burst the optimizer
(`optimize`) moves back a vertex that newly ends outside (`contacts_left`).
`no_contacts` is the per-vertex record (`results.ChainResult.contact_of`)
of a polyline that keeps none.
"""

from collections.abc import Sequence

import numpy as np

from kibends import _core
from kibends.bursts import INSIDE_TOLERANCE
from kibends.energy import BoolArray, DistanceField, FloatArray, IntArray, Integrand
from kibends.obstacles import core_field


def _regions(within: Sequence[DistanceField]) -> list[_core.Field]:
    return [core_field(field) for field in within]


def contact_stops(
    points: FloatArray,
    contact_of: IntArray,
    within: Sequence[DistanceField],
    width: float,
    limit: Integrand,
) -> tuple[float, FloatArray]:
    """The stops that keep each vertex v of `points` that keeps a contact
    inside its region, and their gradient at every vertex: `limit`, the
    clearance hinge, at its distance to the edge of `within[contact_of[v]]`,
    weighted by the chain's `width` (`kibends_core::contacts`)."""
    return _core.contact_stops(points, contact_of, _regions(within), width, limit)


def contacts_left(
    points: FloatArray, contact_of: IntArray, within: Sequence[DistanceField]
) -> BoolArray:
    """The vertices of `points` that keep a contact and lie outside its
    region, beyond INSIDE_TOLERANCE."""
    return _core.contacts_left(points, contact_of, _regions(within), INSIDE_TOLERANCE)


def no_contacts(points: FloatArray) -> IntArray:
    """`ChainResult.contact_of` for a polyline that keeps no contact."""
    return np.full(len(points), -1, dtype=np.intp)
