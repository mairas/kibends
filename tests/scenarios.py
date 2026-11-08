"""Synthetic boards for optimizer tests and plots, in inflated (clearance) space.

Distances are to the clearance boundary: a centreline at distance 0 runs at
exactly the required clearance from the copper.
"""

from dataclasses import dataclass, replace

import numpy as np
from distance_fields import BoolArray, Box, Disc, FloatArray, FreeSpace, Union, Walls

from kibends.energy import EnergyParams
from kibends.neighbours import ChainInput
from kibends.obstacles import ChainCentreline

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


PAIR_WIDTH = 0.2
PAIR_CLEARANCE = 0.2
PAIR_INFLATION = PAIR_WIDTH + PAIR_CLEARANCE
"""Distance two chains of different nets keep between their centrelines."""
SAME_NET_INFLATION = PAIR_WIDTH
"""Distance two chains of one net keep: they may touch but not overlap."""


@dataclass(frozen=True)
class ChainsScenario:
    """Several chains on one layer, each with its own static field."""

    name: str
    chains: tuple[ChainInput, ...]
    params: EnergyParams = PARAMS


def selected(
    description: str,
    net: str,
    field: Field,
    *vertices: tuple[float, float],
) -> ChainInput:
    polyline, fixed = chain(*vertices)
    return ChainInput(
        ChainCentreline(description, net, PAIR_WIDTH, PAIR_CLEARANCE, polyline),
        "F.Cu",
        fixed,
        field,
    )


def channel_pair() -> ChainsScenario:
    """Two chains hugging opposite walls of a channel, each the other's obstacle."""
    walls = Walls(CHANNEL_HALF_GAP)
    hug = -CHANNEL_HUG
    return ChainsScenario(
        "channel-pair",
        (
            selected("A", "A", walls, (0.0, hug), (20.0, hug)),
            selected("B", "B", walls, (0.0, -hug), (20.0, -hug)),
        ),
        replace(PARAMS, repulsion=1.0, falloff_range=1.5),
    )


def uneven_channel_pair() -> ChainsScenario:
    """The channel pair with one chain starting nearer the middle, kinked."""
    pair = channel_pair()
    walls = Walls(CHANNEL_HALF_GAP)
    return replace(
        pair,
        name="uneven-channel-pair",
        chains=(
            pair.chains[0],
            selected("B", "B", walls, (0.0, -0.3), (8.0, -0.6), (20.0, -0.3)),
        ),
    )


TENT_HEIGHT = 1.0


def tent() -> ChainsScenario:
    """A chain arching over a short straight one that blocks its shortcut."""
    return ChainsScenario(
        "tent",
        (
            selected(
                "arch",
                "A",
                FreeSpace(),
                (0.0, 0.0),
                (3.0, 3.0),
                (7.0, 3.0),
                (10.0, 0.0),
            ),
            selected("bar", "B", FreeSpace(), (4.0, TENT_HEIGHT), (6.0, TENT_HEIGHT)),
        ),
    )


SAME_NET_DISC = Disc(5.0, 0.0, 1.0)


def same_net_wrap() -> ChainsScenario:
    """Two chains of one net arching over a disc, the outer one above the inner."""
    return ChainsScenario(
        "same-net-wrap",
        (
            selected("inner", "N", SAME_NET_DISC, (0.0, 0.0), (5.0, 1.5), (10.0, 0.0)),
            selected("outer", "N", SAME_NET_DISC, (0.0, 0.4), (5.0, 2.5), (10.0, 0.4)),
        ),
    )


CROWDED_VERTEX = (5.0, 0.0)
CROWDED_NEIGHBOUR_VERTEX = (5.0, 0.1)


def crowded() -> ChainsScenario:
    """Two chains hand-routed 0.1 mm apart at one vertex each."""
    return ChainsScenario(
        "crowded",
        (
            selected("A", "A", FreeSpace(), (0.0, 0.0), CROWDED_VERTEX, (10.0, 0.0)),
            selected(
                "B", "B", FreeSpace(), (0.0, 1.0), CROWDED_NEIGHBOUR_VERTEX, (10.0, 1.0)
            ),
        ),
    )


def drape() -> ChainsScenario:
    """A long arch above a short one, both pulled flat onto each other."""
    free = FreeSpace()
    return ChainsScenario(
        "drape",
        (
            selected("A", "A", free, (0.0, 0.0), (5.0, 3.0), (10.0, 0.0)),
            selected("B", "B", free, (3.0, 0.8), (5.0, 2.0), (7.0, 0.8)),
        ),
    )


def squeeze() -> ChainsScenario:
    """A V hanging inside an arch whose shortcut it blocks."""
    free = FreeSpace()
    return ChainsScenario(
        "squeeze",
        (
            selected("A", "A", free, (0.0, 0.0), (5.0, 4.0), (10.0, 0.0)),
            selected("B", "B", free, (4.0, 2.5), (5.0, 0.5), (6.0, 2.5)),
        ),
    )


def hooks() -> ChainsScenario:
    """Two interlocked hooks, each pulled tight around the other's end."""
    free = FreeSpace()
    return ChainsScenario(
        "hooks",
        (
            selected("A", "A", free, (0.0, 0.0), (0.0, 4.0), (6.0, 4.0), (6.0, 1.0)),
            selected("B", "B", free, (8.0, 5.0), (8.0, -1.0), (2.0, -1.0), (2.0, 2.5)),
        ),
    )


JUNCTION = (10.0, 0.0)


def neck_down() -> ChainsScenario:
    """Two chains of one net meeting end to end, each bowed on its own side."""
    free = FreeSpace()
    return ChainsScenario(
        "neck-down",
        (
            selected("A", "N", free, (0.0, 0.0), (5.0, 2.0), JUNCTION),
            selected("B", "N", free, JUNCTION, (15.0, 2.0), (20.0, 0.0)),
        ),
    )
