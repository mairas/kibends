"""The terms that keep a chain on its contacts (`neighbours.ChainInput.contacts`).

A vertex that keeps a pad or a via that stays is free, held inside the
region where it keeps contact by a stop, the clearance hinge at its
distance to the region's edge, which limits without repelling
(`contact_stops`). After a burst, the optimizer (`optimize`) moves back a
vertex that newly ends outside (`contacts_left`). `no_contacts` is the
per-vertex record (`results.ChainResult.contact_of`) of a polyline that
keeps none.
"""

from collections.abc import Sequence

import numpy as np

from kibends.bursts import INSIDE_TOLERANCE
from kibends.energy import BoolArray, DistanceField, FloatArray, IntArray, Integrand


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
    weighted by the chain's `width`, as a via's ring stands for a track as
    long as it is wide. They limit without repelling: the hinge rises only
    within its standoff of the edge."""
    gradient = np.zeros_like(points)
    value = 0.0
    for v in np.flatnonzero(contact_of >= 0):
        field = within[int(contact_of[v])]
        distance, direction = field.signed_distance(points[v][None])
        term, slope = limit(distance)
        value += width * float(term[0])
        gradient[v] += width * slope[0] * direction[0]
    return value, gradient


def contacts_left(
    points: FloatArray, contact_of: IntArray, within: Sequence[DistanceField]
) -> BoolArray:
    """The vertices of `points` that keep a contact and lie outside its
    region, beyond INSIDE_TOLERANCE."""
    outside = np.zeros(len(points), dtype=bool)
    for v in np.flatnonzero(contact_of >= 0):
        distance, _ = within[int(contact_of[v])].signed_distance(points[v][None])
        outside[v] = distance[0] < -INSIDE_TOLERANCE
    return outside


def no_contacts(points: FloatArray) -> IntArray:
    """`ChainResult.contact_of` for a polyline that keeps no contact."""
    return np.full(len(points), -1, dtype=np.intp)
