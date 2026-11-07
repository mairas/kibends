"""Synthetic boards for optimizer tests and plots, in inflated (clearance) space.

Distances are to the clearance boundary: a centreline at distance 0 runs at
exactly the required clearance from the copper.
"""

from dataclasses import dataclass, replace

import numpy as np
from distance_fields import BoolArray, Box, Disc, FloatArray, FreeSpace, Union, Walls

from kibends.energy import EnergyParams

ENVELOPE = 0.35
"""Clearance plus half track width, mm: the corner radius of inflated boxes."""

PARAMS = EnergyParams(
    stiffness=0.3,
    repulsion=0.0,
    falloff_range=1.0,
    clearance_weight=50.0,
    standoff=0.02,
    points_per_edge=3,
)

Field = FreeSpace | Disc | Box | Walls | Union


@dataclass(frozen=True)
class Scenario:
    name: str
    polyline: FloatArray
    fixed: BoolArray
    field: Field
    params: EnergyParams = PARAMS


def endpoints_fixed(polyline: FloatArray) -> BoolArray:
    fixed = np.zeros(len(polyline), dtype=bool)
    fixed[[0, -1]] = True
    return fixed


def path(*vertices: tuple[float, float]) -> FloatArray:
    return np.array(vertices, dtype=float)


def chain(*vertices: tuple[float, float]) -> tuple[FloatArray, BoolArray]:
    polyline = path(*vertices)
    return polyline, endpoints_fixed(polyline)


WRAP_BOX = Box(6.5, 3.5, 2.0, 2.0, corner_radius=ENVELOPE)


def wrap(stiffness: float = PARAMS.stiffness) -> Scenario:
    """L-shaped path around a box sitting in the inner corner of the L."""
    polyline, fixed = chain((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))
    return Scenario(
        f"wrap-stiffness-{stiffness}",
        polyline,
        fixed,
        WRAP_BOX,
        replace(PARAMS, stiffness=stiffness),
    )


CHANNEL_HALF_GAP = 1.0
CHANNEL_HUG = -0.8


def channel(repulsion: float) -> Scenario:
    """Straight path hugging one wall of a channel between pinned endpoints."""
    polyline, fixed = chain((0.0, CHANNEL_HUG), (20.0, CHANNEL_HUG))
    return Scenario(
        f"channel-repulsion-{repulsion}",
        polyline,
        fixed,
        Walls(CHANNEL_HALF_GAP),
        replace(PARAMS, repulsion=repulsion, falloff_range=1.5),
    )


TOPOLOGY_DISC = Disc(0.0, 0.0, 1.0)


def topology() -> Scenario:
    """Path left of a disc that lies on the straight line between its endpoints."""
    polyline, fixed = chain((-5.0, 0.0), (0.0, 2.0), (5.0, 0.0))
    return Scenario("topology", polyline, fixed, TOPOLOGY_DISC)


def pad_row() -> Scenario:
    """Path at exactly the clearance along the flat bottoms of a row of boxes."""
    boxes = [
        Box(x, 0.5, 0.6, 0.5, corner_radius=ENVELOPE) for x in (2.0, 4.0, 6.0, 8.0)
    ]
    polyline, fixed = chain((0.0, 0.0), (10.0, 0.0))
    return Scenario(
        "pad-row",
        polyline,
        fixed,
        Union(boxes),
        replace(PARAMS, repulsion=0.5),
    )


FROZEN_DISC = Disc(5.0, 0.0, 1.0)
FROZEN_VERTEX = (5.0, -0.5)


def frozen() -> Scenario:
    """Path with a kink to pull out and a vertex deep inside a disc."""
    polyline, fixed = chain((0.0, -2.0), (2.5, -4.0), FROZEN_VERTEX, (10.0, -2.0))
    return Scenario("frozen", polyline, fixed, FROZEN_DISC)


DETOUR_HALF_WIDTH = ENVELOPE
"""Half-width of the channel's free band: the copper gap is four envelopes."""
DETOUR_LENGTH = 20.0
DETOUR_POCKET_X = 10.0
DETOUR_RISE = 2.5
"""Height of the excursion: 5 mm up and back."""
DETOUR_LEG_OFFSET = 0.15
FAR_SIDE = 50.0


def detour() -> Scenario:
    """Straight channel with a dead-end side pocket the path detours 5 mm into."""
    w, x, rise = DETOUR_HALF_WIDTH, DETOUR_POCKET_X, DETOUR_RISE
    pocket_top = rise + DETOUR_LEG_OFFSET
    left_edge, right_edge = x - w, x + w
    obstacles = Union(
        [
            Box(x, -w - FAR_SIDE, FAR_SIDE, FAR_SIDE),
            Box(left_edge - FAR_SIDE, w + FAR_SIDE, FAR_SIDE, FAR_SIDE),
            Box(right_edge + FAR_SIDE, w + FAR_SIDE, FAR_SIDE, FAR_SIDE),
            Box(x, pocket_top + FAR_SIDE, FAR_SIDE, FAR_SIDE),
        ]
    )
    polyline, fixed = chain(
        (0.0, 0.0),
        (x - DETOUR_LEG_OFFSET, 0.0),
        (x - DETOUR_LEG_OFFSET, rise),
        (x + DETOUR_LEG_OFFSET, rise),
        (x + DETOUR_LEG_OFFSET, 0.0),
        (DETOUR_LENGTH, 0.0),
    )
    return Scenario("detour", polyline, fixed, obstacles)


GAP_DISCS = (Disc(5.0, 1.0, 0.8), Disc(5.0, -1.0, 0.8))


def narrow_gap() -> Scenario:
    """Path threading a 0.4 mm gap that the straight line would pass above."""
    polyline, fixed = chain((0.0, 2.5), (5.0, 0.0), (10.0, 2.5))
    return Scenario("narrow-gap", polyline, fixed, Union(GAP_DISCS))
