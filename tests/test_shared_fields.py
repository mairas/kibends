"""Static fields built from a board's snapshots against fields built from
their own named items, on a board of three copper layers drawn from exact
coordinates only, so that every input is the same on every platform: pads
of one shape on every layer, tracks meeting end to end, at a width change
and at a T-junction, blind and through vias, moving and split vias, a
graphic of more than INDEXED_EDGES edges, keepouts, a zoned moving via and
a coupled section, placed symmetrically so that distances tie exactly.

`TestGolden` pins every field's answers to the digests below, recorded
from the fields as built before they shared their items; run this file
as a script to record them again.
"""

import hashlib
import json
import math
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from shapely.geometry import LineString, Polygon, box

from kibends.energy import FloatArray, ObstacleField
from kibends.obstacles import (
    INDEXED_EDGES,
    ChainCentreline,
    CompositeField,
    Obstacles,
    build_obstacles,
    build_pair_obstacles,
    own_net_stops,
    split_via_obstacles,
)
from kibends.snapshot import (
    Graphic,
    LayerSnapshot,
    Pad,
    PadKind,
    Track,
    Via,
    ViaZone,
)

LAYERS = ("F.Cu", "In1.Cu", "B.Cu")
SIDE = 10.0
"""The board is the square from the origin to (SIDE, SIDE), mm."""
PAD_HALF = 0.5
CLEARANCES = {"HV": 0.4}
DEFAULT_CLEARANCE = 0.2
WIDTHS = (0.25, 0.4)
NETS = ("SIG", "GND", "HV", "P", "")
MOVING = frozenset({"m-gnd", "m-sig", "m-hv"})
PAIR_GAP = 0.15
PAIR_WIDTH = 0.25
STAIRS = 80
"""Steps of the graphic's staircase edge: more than INDEXED_EDGES edges."""
SEED = 195
RANDOM_POINTS = 200
LATTICE_STEP = 0.5
"""Points on this lattice lie where symmetric copper is equally near."""
REACH = 0.5
TOLERANCES = (0.0, 0.01, 0.1)


def square(x: float, y: float, half: float = PAD_HALF) -> Polygon:
    return box(x - half, y - half, x + half, y + half)


def tht(pad_id: str, net: str, x: float, y: float, *, oval: bool = False) -> Pad:
    return Pad(
        pad_id, pad_id, net, PadKind.PTH, (x, y), 0.8, square(x, y), oval_drill=oval
    )


def staircase() -> Polygon:
    """A polygon with a staircase edge of STAIRS steps."""
    x0, y0, step = 1.0, 4.0, 1 / 32
    ring = [(x0, y0)]
    for k in range(STAIRS):
        ring += [(x0 + k * step, y0 + (k + 1) * step / 2)]
        ring += [(x0 + (k + 1) * step, y0 + (k + 1) * step / 2)]
    ring += [(x0 + STAIRS * step, y0)]
    return Polygon(ring)


def via(
    via_id: str, net: str, x: float, y: float, span: tuple[str, str] = ("F.Cu", "B.Cu")
) -> Via:
    return Via(via_id, net, (x, y), 0.6, 0.3, span)


VIAS = (
    via("v-gnd", "GND", 6, 2),
    via("v-sig", "SIG", 6, 8),
    via("v-blind", "HV", 7, 4, ("In1.Cu", "B.Cu")),
    via("v-twin", "GND", 0.5 + 6, 6.5),
    via("m-gnd", "GND", 5, 3.5),
    via("m-sig", "SIG", 3, 7),
    via("m-hv", "HV", 8.5, 5, ("In1.Cu", "B.Cu")),
)


def layer(name: str) -> LayerSnapshot:
    """The board's layer `name`."""
    pads = [
        tht("J1-1", "GND", 2, 2),
        tht("J1-2", "SIG", 4, 2),
        tht("J2-1", "GND", 8, 8, oval=True),
        Pad("MH", "MH", "", PadKind.NPTH, (8.5, 1.5), 1.0, None),
        Pad(
            "U1-1",
            "1",
            "SIG",
            PadKind.SMD,
            (2, 8.5),
            0.0,
            square(2, 8.5) if name == "F.Cu" else None,
        ),
        Pad(
            "U1-2",
            "2",
            "HV",
            PadKind.SMD,
            (4, 8.5),
            0.0,
            square(4, 8.5) if name == "F.Cu" else None,
            clearance_override=0.3,
        ),
        Pad(
            "P1",
            "P1",
            "P",
            PadKind.SMD,
            (1, 0.8),
            0.0,
            square(1, 0.8, 0.2) if name == "F.Cu" else None,
        ),
    ]
    tracks: list[Track] = []
    graphics: list[Graphic] = []
    track_keepouts = []
    via_keepouts = []
    if name == "F.Cu":
        tracks = [
            Track("a1", "GND", 0.25, LineString([(0.5, 5), (3, 5)])),
            Track("a2", "GND", 0.25, LineString([(3, 5), (5, 5)])),
            Track("a3", "GND", 0.4, LineString([(5, 5), (6, 5)])),
            Track("a4", "GND", 0.25, LineString([(4, 5), (4, 6)])),
            Track("h1", "HV", 0.25, LineString([(1, 9.5), (6, 9.5)])),
            Track("p-fan", "P", 0.25, LineString([(8, 0.8), (9, 0.3)])),
        ]
        graphics = [Graphic("graphic logo", "HV", staircase())]
        track_keepouts = [box(8, 3, 9, 4)]
    elif name == "In1.Cu":
        tracks = [Track("i1", "SIG", 0.25, LineString([(1, 3), (9, 3)]))]
        via_keepouts = [box(1, 6, 2, 7)]
    else:
        tracks = [
            Track("b1", "GND", 0.25, LineString([(1, 6), (9, 6)])),
            Track("b2", "SIG", 0.25, LineString([(1, 4), (9, 4)])),
        ]
        track_keepouts = [box(8, 3, 9, 4)]
    vias = [v for v in VIAS if v.reaches(name)]
    edge = ((0, 0), (SIDE, 0), (SIDE, SIDE), (0, SIDE), (0, 0))
    return LayerSnapshot(
        name,
        tuple(pads),
        tuple(tracks),
        tuple(vias),
        tuple(graphics),
        tuple(track_keepouts),
        tuple(via_keepouts),
        (tuple((float(x), float(y)) for x, y in edge),),
        CLEARANCES,
        DEFAULT_CLEARANCE,
    )


ZONE = ViaZone(
    "plane", {"In1.Cu": box(3, 2.5, 7, 6), "B.Cu": box(3, 2.5, 7, 6)}, 0.3, 0.6
)
"""The zone the moving ground via stays inside, its clearance wider than
the netclass rule so that it inflates copper on its layers."""
ZONED = "m-gnd"


@dataclass(frozen=True)
class Built:
    name: str
    field: ObstacleField


def snapshots() -> list[LayerSnapshot]:
    return [layer(name) for name in LAYERS]


def the_via(snaps: Sequence[LayerSnapshot], via_id: str) -> Via:
    return next(v for s in snaps for v in s.vias if v.id == via_id)


def built_fields() -> Iterator[Built]:
    """Every kind of static field a run builds, on the board."""
    snaps = snapshots()
    for snap in snaps:
        for net in NETS:
            for width in WIDTHS:
                yield Built(
                    f"chain {snap.layer} {net} {width}",
                    build_obstacles(snap, net, width, moving=MOVING),
                )
    a = ChainCentreline("P", "P", PAIR_WIDTH, 0.2, np.array([[1, 0.8], [8, 0.8]]))
    pitch = PAIR_WIDTH + PAIR_GAP
    b = ChainCentreline(
        "N", "N", PAIR_WIDTH, 0.2, np.array([[1, 0.8 + pitch], [8, 0.8 + pitch]])
    )
    yield Built(
        "section", build_pair_obstacles(snaps[0], a, b, PAIR_GAP, moving=MOVING)
    )
    splits: dict[str, frozenset[str]] = {
        "m-gnd": frozenset({"v-sig", "v-blind"}),
        "m-sig": frozenset({"v-gnd", "v-twin"}),
        "m-hv": frozenset({"v-gnd"}),
    }
    for via_id, split in splits.items():
        moving_via = the_via(snaps, via_id)
        for zones in ((), (ZONE,)) if via_id == ZONED else ((),):
            zoned = bool(zones)
            for apart in (frozenset[str](), split):
                field, static = split_via_obstacles(
                    snaps, moving_via, apart, moving=MOVING, zones=zones
                )
                name = f"via {via_id} zoned={zoned} split={sorted(apart)}"
                yield Built(name, field)
                if static is not None:
                    yield Built(f"{name} apart", static)
        for allow in (False, True):
            stops = own_net_stops(
                snaps, moving_via, moving=MOVING, allow_via_in_pad=allow
            )
            if stops is not None:
                yield Built(f"stops {via_id} allow={allow}", stops)


def obstacles_of(field: ObstacleField) -> list[Obstacles]:
    """The `Obstacles` parts of `field`."""
    if isinstance(field, Obstacles):
        return [field]
    if isinstance(field, CompositeField):
        return [o for f in field.fields for o in obstacles_of(f)]
    return []


def query_points() -> FloatArray:
    rng = np.random.default_rng(SEED)
    scattered = rng.uniform(-0.5, SIDE + 0.5, size=(RANDOM_POINTS, 2))
    ticks = np.arange(0.0, SIDE + LATTICE_STEP, LATTICE_STEP)
    lattice = np.stack(np.meshgrid(ticks, ticks), axis=-1).reshape(-1, 2)
    return np.concatenate([scattered, lattice])


def segments() -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(SEED + 1)
    starts = rng.uniform(0, SIDE, size=(RANDOM_POINTS, 2))
    return starts, starts + rng.uniform(-1, 1, size=(RANDOM_POINTS, 2))


def described(field: Obstacles) -> list[tuple[object, ...]]:
    return [
        (
            o.description,
            o.inflation.hex(),
            o.geometry.wkb_hex,
            o.net,
            o.extent.hex(),
            o.layer,
            o.kind.name,
            o.owner,
            o.oval,
        )
        for o in field.items
    ]


def answers(field: ObstacleField) -> list[NDArray[np.generic]]:
    """What `field` answers at the query points and segments."""
    at = query_points()
    starts, ends = segments()
    found: list[NDArray[np.generic]] = []
    for reach in (math.inf, REACH):
        found.extend(field.signed_distance(at, reach))
        found.extend(field.smooth_distance(at, reach))
    for tolerance in TOLERANCES:
        found.append(field.segments_clear(starts, ends, tolerance))
    for part in obstacles_of(field):
        edges = part.static_edges
        found += [edges.obstacles, edges.clearance_obstacles]
        for below in (math.inf, REACH):
            found.extend(edges.distances(at, below))
    return found


def digest(field: ObstacleField) -> str:
    h = hashlib.sha256()
    for part in obstacles_of(field):
        h.update(repr(described(part)).encode())
        edge = part.edge
        h.update(repr(None if edge is None else edge[1].hex()).encode())
    for array in answers(field):
        a = np.ascontiguousarray(array)
        h.update(struct.pack("<q", a.size))
        h.update(a.astype(a.dtype.newbyteorder("<")).tobytes())
    return h.hexdigest()


def owned(field: Obstacles) -> Obstacles:
    """`field` built again from its own named items."""
    edge = field.edge
    if edge is None:
        return Obstacles(field.items)
    return Obstacles(field.items, *edge)


def test_the_board_holds_what_the_fields_must_tell_apart() -> None:
    snaps = snapshots()
    assert len(staircase().exterior.coords) - 1 > INDEXED_EDGES
    assert [v.id for v in snaps[0].vias] == [
        "v-gnd",
        "v-sig",
        "v-twin",
        "m-gnd",
        "m-sig",
    ]
    assert "v-blind" in {v.id for v in snaps[1].vias}
    # The pads' copper is the same geometry on every layer it is on.
    assert snaps[0].pads[0].shape == snaps[2].pads[0].shape


@pytest.mark.parametrize(
    "built", [b for b in built_fields() if obstacles_of(b.field)], ids=lambda b: b.name
)
def test_a_field_answers_as_one_built_from_its_own_items(built: Built) -> None:
    for field in obstacles_of(built.field):
        again = owned(field)
        assert described(field) == described(again)
        assert field.edge == again.edge
        for got, want in zip(answers(field), answers(again), strict=True):
            np.testing.assert_array_equal(got, want)


DIGESTS = Path(__file__).parent / "data" / "shared_fields_digests.json"
"""Each field's digest, recorded from the fields as built before they
shared their items."""


def recorded() -> dict[str, str]:
    return cast(dict[str, str], json.loads(DIGESTS.read_text()))


class TestGolden:
    @pytest.mark.parametrize("built", list(built_fields()), ids=lambda b: b.name)
    def test_a_field_answers_as_recorded(self, built: Built) -> None:
        assert digest(built.field) == recorded()[built.name]


if __name__ == "__main__":
    digests = {b.name: digest(b.field) for b in built_fields()}
    DIGESTS.write_text(json.dumps(digests, indent=2) + "\n")
