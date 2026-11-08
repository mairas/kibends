"""Synthetic boards for optimizer tests and plots, in inflated (clearance) space.

Distances are to the clearance boundary: a centreline at distance 0 runs at
exactly the required clearance from the copper.
"""

from dataclasses import dataclass, replace

import numpy as np
from distance_fields import BoolArray, Box, Disc, FloatArray, FreeSpace, Union, Walls

from kibends.energy import EnergyParams, ObstacleField
from kibends.neighbours import ChainInput
from kibends.obstacles import ChainCentreline
from kibends.snapshot import Via, ViaKind
from kibends.vias import ChainEnd, ViaInput

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
    field: ObstacleField,
    *vertices: tuple[float, float],
    layer: str = "F.Cu",
) -> ChainInput:
    polyline, fixed = chain(*vertices)
    return ChainInput(
        ChainCentreline(description, net, PAIR_WIDTH, PAIR_CLEARANCE, polyline),
        layer,
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


VIA_DIAMETER = 0.6
VIA_DRILL = 0.3
VIA_TO_CHAIN = VIA_DIAMETER / 2 + PAIR_CLEARANCE + PAIR_WIDTH / 2
"""Distance a via keeps from another net's chain, centre to centreline."""
VIA_TO_VIA = VIA_DIAMETER + PAIR_CLEARANCE
"""Distance two vias of different nets keep between their centres, by copper."""
ROUTE_START = (0.0, 0.0)
ROUTE_END = (10.0, 0.0)
DETOUR_VIA = (5.0, 3.0)
STRAIGHT_VIA = (5.0, 0.0)


@dataclass(frozen=True)
class ViasScenario:
    """Chains joined by moving vias, each chain and via with its static field."""

    name: str
    chains: tuple[ChainInput, ...]
    vias: tuple[ViaInput, ...]
    params: EnergyParams = PARAMS


def through_via(via_id: str, net: str, at: tuple[float, float]) -> Via:
    return Via(
        via_id, net, at, VIA_DIAMETER, VIA_DRILL, ViaKind.THROUGH, ("F.Cu", "B.Cu")
    )


def route(
    net: str,
    via: tuple[float, float],
    start: tuple[float, float] = ROUTE_START,
    end: tuple[float, float] = ROUTE_END,
    *,
    top: ObstacleField | None = None,
) -> tuple[ChainInput, ChainInput]:
    """An F.Cu chain from `start` to a via at `via` and a B.Cu chain on to
    `end`; `top` is the F.Cu chain's static field."""
    return (
        selected(f"{net} top", net, top or FreeSpace(), start, via),
        selected(f"{net} bottom", net, FreeSpace(), via, end, layer="B.Cu"),
    )


def joining(via: Via, first: int, field: ObstacleField | None = None) -> ViaInput:
    """`via` joining the last end of chain `first` to the first of the next."""
    return ViaInput(
        via, (ChainEnd(first, True), ChainEnd(first + 1, False)), field or FreeSpace()
    )


def via_detour(at: tuple[float, float] = DETOUR_VIA) -> ViasScenario:
    """An F.Cu - via - B.Cu route whose via sits off the straight line."""
    return ViasScenario(
        "via-detour", route("N", at), (joining(through_via("v", "N", at), 0),)
    )


def via_on_one_layer(at: tuple[float, float] = DETOUR_VIA) -> ViasScenario:
    """The detour route with both chains on F.Cu, each the other's neighbour."""
    via = through_via("v", "N", at)
    return ViasScenario(
        "via-on-one-layer",
        (
            selected("N left", "N", FreeSpace(), ROUTE_START, at),
            selected("N right", "N", FreeSpace(), at, ROUTE_END),
        ),
        (joining(via, 0),),
    )


FERMAT_START = (1.0, 0.2)
FERMAT_ENDS = ((0.0, 0.0), (10.0, 0.0), (5.0, 5.0))
FERMAT_POINT = (5.0, 5.0 / 3**0.5)
"""Where the three chains meet at 120 degrees, the shortest junction."""


def via_fermat() -> ViasScenario:
    """Three chains of one net meeting at a via near the first one's far end,
    two of them on F.Cu at an angle, so that each is the other's same-net
    neighbour, and one on B.Cu. The via travels along the first chain to
    where all three meet at 120 degrees, and the first chain then runs past
    where the junction was."""
    first, second, third = FERMAT_ENDS
    via = through_via("v", "N", FERMAT_START)
    return ViasScenario(
        "via-fermat",
        (
            selected("N first", "N", FreeSpace(), first, FERMAT_START),
            selected("N second", "N", FreeSpace(), FERMAT_START, second),
            selected("N third", "N", FreeSpace(), FERMAT_START, third, layer="B.Cu"),
        ),
        (
            ViaInput(
                via,
                (ChainEnd(0, True), ChainEnd(1, False), ChainEnd(2, False)),
                FreeSpace(),
            ),
        ),
    )


def via_detour_over(top: ObstacleField) -> ViasScenario:
    """The detour route with `top` the F.Cu chain's static field."""
    return ViasScenario(
        "via-detour-over",
        route("N", DETOUR_VIA, top=top),
        (joining(through_via("v", "N", DETOUR_VIA), 0),),
    )


HEMMING_TRACK = Box(5.0, 1.5, 3.0, 0.1)
"""A track on an inner layer, inflated for the via, across its way down: thin
enough to jump, too long to slide around."""


def via_hemmed(params: EnergyParams = PARAMS) -> ViasScenario:
    """The detour route with a track on an inner layer under the via."""
    return ViasScenario(
        "via-hemmed",
        route("N", DETOUR_VIA),
        (joining(through_via("v", "N", DETOUR_VIA), 0, HEMMING_TRACK),),
        params,
    )


CORNER_BOX = Box(2.5, -2.1, 1.0, 2.9)
"""Sharp-cornered, its top at y = 0.8 just under the top chain's start."""


def via_corner() -> ViasScenario:
    """The detour route with a box on F.Cu that the top chain must wrap as
    the via pulls it down."""
    return ViasScenario(
        "via-corner",
        route("N", DETOUR_VIA, top=CORNER_BOX),
        (joining(through_via("v", "N", DETOUR_VIA), 0),),
    )


NEIGHBOUR_LEVEL = 0.5


def via_beside_chain() -> ViasScenario:
    """The detour route and a chain of another net on an inner layer, bowed
    down out of the via's way along a wall that slows its straightening, so
    that the via has moved before the chain reaches it."""
    neighbour = selected(
        "C",
        "C",
        Walls(CHANNEL_HALF_GAP),
        (3.0, NEIGHBOUR_LEVEL),
        (5.0, -0.95),
        (7.0, NEIGHBOUR_LEVEL),
        layer="In1.Cu",
    )
    return ViasScenario(
        "via-beside-chain",
        (*route("N", DETOUR_VIA), neighbour),
        (joining(through_via("v", "N", DETOUR_VIA), 0),),
    )


PAIR_LEVEL = 0.7
"""Height of the second route's ends: its via cannot reach its line."""


def via_pair() -> ViasScenario:
    """Two routes of different nets whose vias are pulled toward each other."""
    lower, upper = (5.0, -2.0), (5.0, PAIR_LEVEL + 2.0)
    return ViasScenario(
        "via-pair",
        (
            *route("A", lower),
            *route("B", upper, (0.0, PAIR_LEVEL), (10.0, PAIR_LEVEL)),
        ),
        (
            joining(through_via("a", "A", lower), 0),
            joining(through_via("b", "B", upper), 2),
        ),
    )


SAME_NET_LEVEL = 1.5


PINNED_STEP = 0.05
"""Shorter than any spacing used, so a chain pinned at every vertex this far
apart has no free vertex and stands still."""


def via_over_same_net() -> ViasScenario:
    """The detour route above a chain of its own net on an inner layer,
    pinned throughout, so that only the via can keep the two apart."""
    count = round(10.0 / PINNED_STEP) + 1
    points = np.column_stack(
        [np.linspace(0.0, 10.0, count), np.full(count, SAME_NET_LEVEL)]
    )
    inner = ChainInput(
        ChainCentreline("inner", "N", PAIR_WIDTH, PAIR_CLEARANCE, points),
        "In1.Cu",
        np.ones(count, dtype=bool),
        FreeSpace(),
    )
    return ViasScenario(
        "via-over-same-net",
        (*route("N", DETOUR_VIA), inner),
        (joining(through_via("v", "N", DETOUR_VIA), 0),),
    )


SLACK_VIA = (5.0, 0.0)


def via_slack(
    field: ObstacleField, at: tuple[float, float] = SLACK_VIA
) -> ViasScenario:
    """A via at or near where it belongs, carrying two kinked chains, F.Cu
    and B.Cu, that straighten while it stays."""
    return ViasScenario(
        "via-slack",
        (
            selected("top", "N", field, ROUTE_START, (2.5, 2.0), at),
            selected("bottom", "N", field, at, (7.5, -2.0), ROUTE_END, layer="B.Cu"),
        ),
        (joining(through_via("v", "N", at), 0),),
    )


ROUTE_LEVEL = 3.0
BOWED_VIA = (5.0, ROUTE_LEVEL)


def via_over_bowed_same_net() -> ViasScenario:
    """A straight route along y = 3 with its via over a chain of its own net
    on an inner layer, bowed down under the via and pulled up past it."""
    inner = selected(
        "inner",
        "N",
        FreeSpace(),
        (3.0, ROUTE_LEVEL + 0.2),
        (5.0, 0.5),
        (7.0, ROUTE_LEVEL + 0.2),
        layer="In1.Cu",
    )
    return ViasScenario(
        "via-over-bowed-same-net",
        (
            *route("N", BOWED_VIA, (0.0, ROUTE_LEVEL), (10.0, ROUTE_LEVEL)),
            inner,
        ),
        (joining(through_via("v", "N", BOWED_VIA), 0),),
    )


CROSSING_VIA = (4.0, 4.0)


def via_across_chain(nudge: float = 0.0) -> ViasScenario:
    """An L route, (0, 0) to (4, 0) down to a via at `CROSSING_VIA` and on
    by (8, 4) to (8, 0), and a chain of another net on an inner layer
    around the via's lower left, straight a line through the via. With
    repulsion, the chain straightens into the via and presses on it as the
    via starts up, then falls behind. `nudge` moves the chain's start along
    x, mm: the run is chaotic enough that a few nm decide when the chain
    joins and leaves the via's burst."""
    inner = selected(
        "C",
        "C",
        FreeSpace(),
        (2.0 + nudge, 2.0),
        (2.0, 6.0),
        (6.0, 6.0),
        layer="In1.Cu",
    )
    return ViasScenario(
        "via-across-chain",
        (
            selected("N top", "N", FreeSpace(), ROUTE_START, (4.0, 0.0), CROSSING_VIA),
            selected(
                "N bottom",
                "N",
                FreeSpace(),
                CROSSING_VIA,
                (8.0, 4.0),
                (8.0, 0.0),
                layer="B.Cu",
            ),
            inner,
        ),
        (joining(through_via("v", "N", CROSSING_VIA), 0),),
        replace(PARAMS, repulsion=0.5),
    )
