"""Shared fixtures and kipy fakes.

FakeBoard and the kipy builders are adapted from kistitch's
tests/test_snapshot.py (MIT, Copyright (c) 2025 Hat Labs Oy).
"""

import itertools
import math
from collections import Counter
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Iterable,
    Iterator,
    Sequence,
)
from dataclasses import dataclass, field
from typing import cast

import kipy.board
import kipy.board_types as kbt
import numpy as np
import pytest
import shapely
from google.protobuf.message import Message
from kipy.common_types import Commit, Text, TextBox
from kipy.geometry import ArcStartMidEnd, PolygonWithHoles, PolyLineNode, Vector2
from kipy.project_types import NetClass
from kipy.proto.board.board_commands_pb2 import BoardOriginType
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.board.board_types_pb2 import PadType, ViaType, ZoneType
from kipy.proto.common.commands import base_commands_pb2
from kipy.proto.common.commands.editor_commands_pb2 import (
    AddToSelection,
    CreateItems,
    CreateItemsResponse,
    DeleteItems,
    DeleteItemsResponse,
    ItemDeletionStatus,
    ItemStatusCode,
    SelectionResponse,
)
from kipy.proto.common.types import base_types_pb2, project_settings_pb2
from kipy.proto.common.types.base_types_pb2 import KIID, DocumentSpecifier
from kipy.util.proto import pack_any
from kipy.wrapper import Wrapper
from loguru import logger

import kibends.optimize as optimize_module
from kibends import _core
from kibends.bursts import FALLBACK_WARNING, INSIDE_TOLERANCE, OptimizerSettings
from kibends.chain import (
    NM_PER_MM,
    Chain,
    SameNetCopper,
    Segment,
    split_chains,
    split_locked,
)
from kibends.coords import UserFrame
from kibends.energy import (
    BoolArray,
    DistanceField,
    EnergyParams,
    FloatArray,
    ObstacleField,
    TermValue,
    attraction_integrand,
    bend_radius_energy,
    edge_integral,
    obstacle_integrand,
    obstacle_reach,
    quadrature_points,
)
from kibends.members import Member, members_energy, seen_parts
from kibends.neighbours import ChainInput
from kibends.obstacles import (
    EDGE_CLEARANCE,
    HOLE_CLEARANCE,
    HOLE_TO_HOLE,
    ChainCentreline,
    CompositeField,
    Obstacles,
    containment,
    core_field,
    split_via_obstacles,
)
from kibends.polylines import count_violations
from kibends.seen import Seen
from kibends.snapshot import BoardSnapshot, Disc, LayerSnapshot, Via, ViaZone

PAD_HALF_SIZE = 0.25
TEXT_HALF_WIDTH = 1.0
TEXT_HALF_HEIGHT = 0.5


def all_abnormal_bursts(settings: OptimizerSettings) -> int:
    """The bursts a problem whose every burst ends abnormally takes under
    `settings`: a streak of `optimize.MAX_ABNORMAL_BURSTS` at each level,
    the coarse ones and the final one, since a level that gives up hands
    over."""
    levels = len(settings.coarse_spacings)
    return (levels + 1) * optimize_module.MAX_ABNORMAL_BURSTS


class Ticking:
    """A clock that reads `now` and then advances by `step` seconds."""

    def __init__(self, now: float = 0.0, step: float = 1.0) -> None:
        self.now = now
        self.step = step

    def __call__(self) -> float:
        now = self.now
        self.now += self.step
        return now


def in_pool[T](threads: int, work: Callable[[], T]) -> T:
    """What `work` returns, run on a rayon pool of `threads` threads of its
    own, over which its evaluations spread."""
    answers: list[T] = []
    _core.testing.in_pool(threads, lambda: answers.append(work()))
    return answers[0]


class BurstClock:
    """A clock that reads how many bursts have ended since it was made,
    however often it is read: one per problem and outer iteration. It does
    not move during a burst, so a time limit never ends one early."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.bursts = 0
        real = cast(Callable[..., object], optimize_module.run_burst)

        def counted(*args: object, **kwargs: object) -> object:
            burst = real(*args, **kwargs)
            self.bursts += 1
            return burst

        monkeypatch.setattr(optimize_module, "run_burst", counted)

    def __call__(self) -> float:
        return float(self.bursts)


def first_check(
    monkeypatch: pytest.MonkeyPatch, run: Callable[[], object]
) -> tuple[_core.Revert, FloatArray]:
    """The check of the first burst that `run` starts, and where the burst
    starts."""
    real = cast(Callable[..., object], optimize_module.run_burst)
    captured: list[tuple[_core.Revert, FloatArray]] = []

    def capturing(
        objective: object, start: FloatArray, revert: object, *args: object
    ) -> object:
        assert isinstance(revert, _core.Revert)
        captured.append((revert, start.copy()))
        return real(objective, start, revert, *args)

    monkeypatch.setattr(optimize_module, "run_burst", capturing)
    run()
    return captured[0]


def _fallback_sink(warned: list[str]) -> int:
    """A loguru sink that adds each `FALLBACK_WARNING` to `warned`."""
    return logger.add(
        lambda m: warned.append(str(m)),
        level="WARNING",
        filter=lambda record: record["message"] == FALLBACK_WARNING,
    )


_FALLBACK_FIXTURES: list[tuple[str, object]] = []
"""Each fixture value whose setup reached the fallback, by its fixture's
name; held, so that its identity names the setup it came from: of which
parameter, or of which of fixtures that share a name."""


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(
    fixturedef: pytest.FixtureDef[object],
) -> Generator[None, object, object]:
    """Records a fixture whose setup reaches the fallback, which a fixture
    shared by several tests sets up before `no_burst_moved_back_whole`
    listens."""
    warned: list[str] = []
    sink = _fallback_sink(warned)
    try:
        value = yield
    finally:
        logger.remove(sink)
    if warned:
        _FALLBACK_FIXTURES.append((fixturedef.argname, value))
    return value


def _uses_a_fallback_fixture(node: pytest.Item | pytest.Collector) -> bool:
    """Whether the test `node` got a fixture value whose setup reached the
    fallback."""
    if not isinstance(node, pytest.Function):
        return False
    return any(
        name in node.funcargs and node.funcargs[name] is value
        for name, value in _FALLBACK_FIXTURES
    )


@pytest.fixture(autouse=True)
def no_burst_moved_back_whole(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fails any test in which a burst's check reaches the fallback that
    moves it back whole (`FALLBACK_WARNING`), or that uses a fixture whose
    setup did, unless marked `fallback`."""
    warned: list[str] = []
    sink = _fallback_sink(warned)
    yield
    logger.remove(sink)
    reached = warned or _uses_a_fallback_fixture(request.node)
    if reached and request.node.get_closest_marker("fallback") is None:
        pytest.fail("a burst reached the fallback that moves it all back")


@pytest.fixture
def log() -> Iterator[list[str]]:
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{level} {message}")
    yield lines
    logger.remove(sink)


def nm(v: float) -> int:
    return round(v * NM_PER_MM)


BOARD_FRAME = UserFrame()
"""The board frame itself, as a grid origin at the page origin gives it."""

CURVATURE_STEP = 1e-7
"""mm of a central difference of a gradient, for a curvature."""
CURVATURE_RTOL = 1e-5
CURVATURE_ATOL = 1e-6
"""How near an analytic curvature lies to its central difference."""

CLEARANCE_STEP = 0.01
"""mm between the points along a polyline's edges where `clearances` looks."""


def clearances(polyline: FloatArray, field: DistanceField) -> FloatArray:
    """The signed distance from `field` along each edge of `polyline`, in
    steps of at most CLEARANCE_STEP, both ends included."""
    points = [
        start + np.linspace(0.0, 1.0, steps + 1)[:, None] * (end - start)
        for start, end in itertools.pairwise(polyline)
        for steps in [max(1, math.ceil(math.dist(start, end) / CLEARANCE_STEP))]
    ]
    distance, _ = field.signed_distance(np.concatenate(points))
    return distance


GAP_SAMPLES_PER_WIDTH = 4
"""Points per track width at which `gap_range` samples each track."""
CORNER_SLACK = 1e-9
"""mm: a nearest point this close to a vertex lies at it, for `gap_range`."""


def gap_range(a: FloatArray, b: FloatArray, width: float) -> tuple[float, float]:
    """The smallest and largest edge-to-edge gap between two tracks of
    `width`.

    The smallest is exact: between polylines it occurs at a vertex. The
    largest is sampled, square across: from points whose nearest point on
    the other track lies inside an edge, not at a vertex. A mitred pair's
    outer corner lies farther from the inner track than either of its
    edges, but the gap across the edges stays the pair's."""
    smallest = shapely.LineString(a).distance(shapely.LineString(b))
    step = width / GAP_SAMPLES_PER_WIDTH
    across: list[FloatArray] = []
    for here, there in ((a, b), (b, a)):
        samples = shapely.points(
            shapely.get_coordinates(shapely.segmentize(shapely.LineString(here), step))
        )
        other = shapely.LineString(there)
        # types-shapely leaves the vectorised functions' kwargs untyped.
        at: FloatArray = np.asarray(
            shapely.line_locate_point(other, samples),  # pyright: ignore[reportUnknownMemberType]
            dtype=float,
        )
        distance: FloatArray = np.asarray(
            shapely.distance(samples, other),  # pyright: ignore[reportUnknownMemberType]
            dtype=float,
        )
        vertices = np.concatenate(
            [[0.0], np.cumsum(np.hypot(*np.diff(there, axis=0).T))]
        )
        square = np.abs(at[:, None] - vertices[None]).min(axis=1) > CORNER_SLACK
        across.append(distance[square])
    square_distances = np.concatenate(across)
    largest = (
        float(square_distances.max()) if len(square_distances) else float(smallest)
    )
    return float(smallest) - width, largest - width


def obstacle_energy(
    points: FloatArray, field: DistanceField, params: EnergyParams
) -> TermValue:
    """The obstacle terms of one polyline's energy in `field` alone, with
    the gradient of every vertex, fixed ones included."""
    distance, direction = field.smooth_distance(
        quadrature_points(points, params.points_per_edge),
        reach=obstacle_reach(params),
    )
    value, gradient, _ = edge_integral(
        points, distance, direction, obstacle_integrand(params), params.points_per_edge
    )
    return value, gradient


def total_energy(
    points: FloatArray,
    fixed: BoolArray,
    field: DistanceField,
    params: EnergyParams,
    short_edge: float = 0.0,
    bend_limit: float | None = None,
) -> TermValue:
    """One polyline's energy in `field` alone, as the optimizer sums its
    terms (`members.members_energy`), with the bend-radius hinge at
    `bend_limit` if given; raises ValueError if it is not finite."""
    points = np.asarray(points, dtype=float)
    alone = ChainInput(
        ChainCentreline("alone", "", 0.0, 0.0, points), "", fixed, core_field(field)
    )
    value, (gradient,), _ = members_energy(
        [alone],
        [points],
        [fixed],
        params,
        short_edge,
        seen_parts([alone], [points]),
        (),
    )
    if bend_limit is not None:
        bend_value, bend_gradient = bend_radius_energy(
            points, fixed, bend_limit, params
        )
        value += bend_value
        gradient = gradient + bend_gradient
        if not (np.isfinite(value) and np.all(np.isfinite(gradient))):
            raise ValueError(f"energy is not finite: {value}")
        gradient[fixed] = 0.0
    return value, gradient


def layer_energy(
    chains: Sequence[Member],
    polylines: Sequence[FloatArray],
    masks: Sequence[BoolArray],
    params: EnergyParams,
    short_edge: float,
    seen: Seen | None = None,
) -> tuple[float, list[FloatArray]]:
    """`members.members_energy` of chains and sections of one layer without
    vias, by default seeing each other as `members.seen_parts` finds them."""
    if seen is None:
        seen = seen_parts(chains, polylines)
    value, gradients, _ = members_energy(
        chains, polylines, masks, params, short_edge, seen, ()
    )
    return value, gradients


def build_via_obstacles(
    layers: Sequence[LayerSnapshot],
    via: Via,
    *,
    moving: Collection[str] = frozenset(),
    edge_clearance: float = EDGE_CLEARANCE,
    hole_clearance: float = HOLE_CLEARANCE,
    hole_to_hole: float = HOLE_TO_HOLE,
    zones: Sequence[ViaZone] = (),
) -> ObstacleField:
    """`obstacles.split_via_obstacles`' field with every via of another net
    in it."""
    field, _ = split_via_obstacles(
        layers,
        via,
        (),
        moving=moving,
        edge_clearance=edge_clearance,
        hole_clearance=hole_clearance,
        hole_to_hole=hole_to_hole,
        zones=zones,
    )
    return field


def contained(field: ObstacleField, zones: Iterable[ViaZone]) -> ObstacleField:
    """`field` with the outside of the regions of `zones`: every limit on
    where a via may go."""
    inside = containment(zones)
    return field if inside is None else CompositeField([field, inside])


def violations(points: FloatArray, anchors: BoolArray, field: ObstacleField) -> int:
    """`polylines.count_violations` of a polyline against one field: the
    definition the optimizer's count over every member at once
    (`optimize._Problem.violations`) is held to."""
    distance, _ = field.signed_distance(points)
    inside = distance < -INSIDE_TOLERANCE
    ends_clear = ~inside[:-1] & ~inside[1:]
    cut = np.zeros(len(ends_clear), dtype=bool)
    cut[ends_clear] = ~field.segments_clear(
        points[:-1][ends_clear], points[1:][ends_clear], INSIDE_TOLERANCE
    )
    return count_violations(inside, cut, anchors)


def extract_chains(
    segments: Sequence[Segment],
    same_net_copper: Callable[[str, str], Iterable[SameNetCopper]] = (
        lambda _layer, _net: ()
    ),
    *,
    frame: UserFrame,
    cuts: Callable[[str, str], Collection[Disc]] = lambda _layer, _net: (),
) -> list[Chain]:
    """The chains of a selection, its locked segments split off as the run
    splits them (`chain.split_chains`)."""
    kept, locked = split_locked(segments)
    return split_chains(kept, locked, same_net_copper, frame=frame, cuts=cuts)


def one_chain(
    segments: Sequence[Segment],
    same_net_copper: Iterable[SameNetCopper] = (),
    *,
    frame: UserFrame,
) -> Chain:
    """The one chain `extract_chains` gives `segments`, with `same_net_copper`
    on every layer and net."""
    copper = tuple(same_net_copper)
    (chain,) = extract_chains(segments, lambda _layer, _net: copper, frame=frame)
    return chain


def vec(x: float, y: float) -> Vector2:
    return Vector2.from_xy(nm(x), nm(y))


def kipy_polygon(points: Sequence[tuple[float, float]]) -> PolygonWithHoles:
    pwh = PolygonWithHoles()
    for x, y in points:
        pwh.outline.append(PolyLineNode.from_xy(nm(x), nm(y)))
    pwh.outline.closed = True
    return pwh


def kipy_circular_outline(
    centre: tuple[float, float], radius: float
) -> PolygonWithHoles:
    """A circular zone's outline as KiCad gives it: one arc that ends where it
    starts, with mid opposite."""
    cx, cy = centre
    arc = ArcStartMidEnd()
    arc.start = vec(cx + radius, cy)
    arc.mid = vec(cx - radius, cy)
    arc.end = vec(cx + radius, cy)
    node = PolyLineNode()
    node.arc = arc
    pwh = PolygonWithHoles()
    pwh.outline.append(node)
    pwh.outline.closed = True
    return pwh


def kipy_rect(x0: float, y0: float, x1: float, y1: float) -> PolygonWithHoles:
    return kipy_polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))


def kipy_track(
    track_id: str,
    net: str,
    start: tuple[float, float],
    end: tuple[float, float],
    width: float = 0.25,
    layer: BL.ValueType = BL.BL_F_Cu,
    locked: bool = False,
) -> kbt.Track:
    t = kbt.Track()
    t.proto.id.value = track_id
    t.net = kbt.Net(name=net)
    t.start = vec(*start)
    t.end = vec(*end)
    t.width = nm(width)
    t.layer = layer
    t.locked = locked
    return t


def kipy_arc(
    track_id: str,
    net: str = "SIG",
    start: tuple[float, float] = (0, 0),
    mid: tuple[float, float] = (1, 1),
    end: tuple[float, float] = (2, 0),
    width: float = 0.25,
    layer: BL.ValueType = BL.BL_F_Cu,
    locked: bool = False,
) -> kbt.ArcTrack:
    a = kbt.ArcTrack()
    a.proto.id.value = track_id
    a.net = kbt.Net(name=net)
    a.start = vec(*start)
    a.mid = vec(*mid)
    a.end = vec(*end)
    a.width = nm(width)
    a.layer = layer
    a.locked = locked
    return a


def kipy_group(name: str, items: Sequence[kbt.BoardItem]) -> kbt.Group:
    """A group of `items` as KiCad's selection hands it over: its `items` is
    empty and only its proto lists their ids."""
    g = kbt.Group()
    g.proto.id.value = f"group-{name}"
    g.proto.name = name
    g.proto.items.extend(KIID(value=i.id.value) for i in items)
    return g


def kipy_pad(
    pad_id: str,
    net: str,
    x: float,
    y: float,
    layers: Sequence[int] = (BL.BL_F_Cu,),
    override: float | None = None,
    parent: str | None = None,
    pad_type: int = PadType.PT_SMD,
    drill: float = 0.0,
    number: str = "1",
) -> kbt.Pad:
    p = kbt.Pad()
    p.proto.id.value = pad_id
    p.proto.net.name = net
    p.proto.type = pad_type
    p.proto.number = number
    p.position = vec(x, y)
    p.proto.pad_stack.layers.extend(layers)
    if drill:
        p.padstack.drill.diameter = vec(drill, drill)
    if override is not None:
        p.proto.copper_clearance_override.value_nm = nm(override)
    if parent is not None:
        p.proto.parent.value = parent
    return p


def kipy_via(
    via_id: str,
    net: str,
    x: float,
    y: float,
    diameter: float = 0.6,
    drill: float = 0.3,
    *,
    locked: bool = False,
    via_type: ViaType.ValueType = ViaType.VT_THROUGH,
    span: tuple[BL.ValueType, BL.ValueType] = (BL.BL_F_Cu, BL.BL_B_Cu),
) -> kbt.Via:
    v = kbt.Via()
    v.proto.id.value = via_id
    v.net = kbt.Net(name=net)
    v.position = vec(x, y)
    v.padstack.proto.copper_layers.add()
    v.diameter = nm(diameter)
    v.drill_diameter = nm(drill)
    v.locked = locked
    v.type = via_type
    v.padstack.drill.start_layer, v.padstack.drill.end_layer = span
    return v


def kipy_footprint(
    fp_id: str,
    ref: str = "U1",
    override: float | None = None,
    items: Sequence[Wrapper] = (),
) -> kbt.FootprintInstance:
    f = kbt.FootprintInstance()
    f.proto.id.value = fp_id
    f.reference_field.text.value = ref
    if override is not None:
        f.proto.overrides.copper_clearance.value_nm = nm(override)
    for item in items:
        f.definition.add_item(item)
    return f


def _drawn[ShapeT: kbt.BoardShape](
    shape: ShapeT,
    layer: BL.ValueType,
    width: float,
    net: str,
    shape_id: str,
    filled: bool = False,
) -> ShapeT:
    shape.proto.id.value = shape_id
    shape.layer = layer
    shape.attributes.stroke.width = nm(width)
    shape.attributes.fill.filled = filled
    shape.net = kbt.Net(name=net)
    return shape


def kipy_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    layer: BL.ValueType = BL.BL_F_Cu,
    width: float = 0.1,
    net: str = "",
    shape_id: str = "shape",
) -> kbt.BoardSegment:
    s = kbt.BoardSegment()
    s.start = vec(*start)
    s.end = vec(*end)
    return _drawn(s, layer, width, net, shape_id)


def kipy_board_arc(
    start: tuple[float, float],
    mid: tuple[float, float],
    end: tuple[float, float],
    layer: BL.ValueType = BL.BL_F_Cu,
    width: float = 0.1,
    shape_id: str = "arc",
) -> kbt.BoardArc:
    a = kbt.BoardArc()
    a.start = vec(*start)
    a.mid = vec(*mid)
    a.end = vec(*end)
    return _drawn(a, layer, width, "", shape_id)


def kipy_circle(
    centre: tuple[float, float],
    radius: float,
    filled: bool = False,
    layer: BL.ValueType = BL.BL_F_Cu,
    width: float = 0.1,
    shape_id: str = "circle",
) -> kbt.BoardCircle:
    c = kbt.BoardCircle()
    c.center = vec(*centre)
    c.radius_point = vec(centre[0] + radius, centre[1])
    return _drawn(c, layer, width, "", shape_id, filled)


def kipy_rectangle(
    top_left: tuple[float, float],
    bottom_right: tuple[float, float],
    filled: bool = False,
    layer: BL.ValueType = BL.BL_F_Cu,
    width: float = 0.1,
    shape_id: str = "rect",
    corner_radius: float = 0.0,
) -> kbt.BoardRectangle:
    r = kbt.BoardRectangle()
    r.top_left = vec(*top_left)
    r.bottom_right = vec(*bottom_right)
    r.proto.shape.rectangle.corner_radius.value_nm = nm(corner_radius)
    return _drawn(r, layer, width, "", shape_id, filled)


def kipy_bezier(shape_id: str = "bezier") -> kbt.BoardBezier:
    b = kbt.BoardBezier()
    b.start = vec(0, 0)
    b.control1 = vec(1, 1)
    b.control2 = vec(2, 1)
    b.end = vec(3, 0)
    return _drawn(b, BL.BL_F_Cu, 0.1, "", shape_id)


def kipy_text(
    value: str, x: float, y: float, layer: BL.ValueType = BL.BL_F_Cu
) -> kbt.BoardText:
    t = kbt.BoardText()
    t.proto.id.value = f"text-{value}"
    t.value = value
    t.position = vec(x, y)
    t.layer = layer
    return t


def kipy_field(
    value: str,
    x: float,
    y: float,
    layer: BL.ValueType = BL.BL_F_Cu,
    visible: bool = True,
) -> kbt.Field:
    f = kbt.Field()
    f.name = "Reference"
    f.text.value = value
    f.text.position = vec(x, y)
    f.layer = layer
    f.visible = visible
    return f


def kipy_zone(
    outline: PolygonWithHoles,
    layers: Sequence[int],
    keepout_tracks: bool = True,
    rule_area: bool = True,
    keepout_vias: bool = False,
    zone_id: str = "zone",
    net: str = "GND",
) -> kbt.Zone:
    z = kbt.Zone()
    z.proto.id.value = zone_id
    z.proto.type = ZoneType.ZT_RULE_AREA if rule_area else ZoneType.ZT_COPPER
    z.proto.layers.extend(layers)
    z.proto.outline.polygons.add().CopyFrom(outline.proto)
    if rule_area:
        z.proto.rule_area_settings.keepout_tracks = keepout_tracks
        z.proto.rule_area_settings.keepout_vias = keepout_vias
    else:
        z.proto.copper_settings.net.name = net
    return z


def kipy_copper_zone(
    zone_id: str,
    net: str,
    outline: PolygonWithHoles,
    layers: Sequence[BL.ValueType],
    zone_type: ZoneType.ValueType = ZoneType.ZT_COPPER,
    *,
    min_width: float = 0.0,
) -> kbt.Zone:
    """A copper zone of `net` drawn as `outline` on `layers`, unfilled."""
    z = kipy_zone(outline, list(layers), rule_area=False, zone_id=zone_id, net=net)
    z.proto.type = zone_type
    z.proto.copper_settings.min_thickness.value_nm = round(min_width * NM_PER_MM)
    return z


def netclass(name: str, clearance: float | None = None) -> NetClass:
    # NetClass() without a proto shares one default proto between instances.
    nc = NetClass(project_settings_pb2.NetClass(name=name))
    nc.clearance = None if clearance is None else nm(clearance)
    return nc


def _text_box(x: float, y: float) -> base_types_pb2.GraphicShape:
    shape = base_types_pb2.GraphicShape()
    polygon = kipy_rect(
        x - TEXT_HALF_WIDTH,
        y - TEXT_HALF_HEIGHT,
        x + TEXT_HALF_WIDTH,
        y + TEXT_HALF_HEIGHT,
    )
    shape.polygon.polygons.add().CopyFrom(polygon.proto)
    shape.attributes.fill.fill_type = base_types_pb2.GraphicFillType.GFT_FILLED
    return shape


@dataclass
class FakeClient:
    """Renders every text as a filled box of 2 x 1 mm around its position."""

    error: Exception | None = None
    renders: int = 0

    def send(self, command: Message, response_type: type[Message]) -> Message:
        assert isinstance(command, base_commands_pb2.GetTextAsShapes)
        self.renders += 1
        if self.error is not None:
            raise self.error
        response = base_commands_pb2.GetTextAsShapesResponse()
        for entry in command.text:
            if entry.HasField("text"):
                position = Text(entry.text).position
            else:
                position = TextBox(entry.textbox).top_left
            reply = response.text_with_shapes.add()
            reply.shapes.shapes.append(
                _text_box(position.x / NM_PER_MM, position.y / NM_PER_MM)
            )
        return response


@dataclass
class FakeProject:
    netclasses: list[NetClass]
    path: str = ""
    """The project's directory, empty when the board has no project."""
    name: str = ""
    """The project's name, its file's without `.kicad_pro`."""

    def get_net_classes(self) -> list[NetClass]:
        return self.netclasses


@dataclass
class FakeBoard:
    """Answers the reads the snapshot makes; `reads` counts them by method."""

    selection: list[Wrapper] = field(default_factory=list[Wrapper])
    pads: list[kbt.Pad] = field(default_factory=list[kbt.Pad])
    vias: list[kbt.Via] = field(default_factory=list[kbt.Via])
    tracks: list[kbt.Track | kbt.ArcTrack] = field(
        default_factory=list[kbt.Track | kbt.ArcTrack]
    )
    shapes: list[kbt.BoardShape] = field(default_factory=list[kbt.BoardShape])
    texts: list[kbt.BoardText | kbt.BoardTextBox] = field(
        default_factory=list[kbt.BoardText | kbt.BoardTextBox]
    )
    footprints: list[kbt.FootprintInstance] = field(
        default_factory=list[kbt.FootprintInstance]
    )
    zones: list[kbt.Zone] = field(default_factory=list[kbt.Zone])
    groups: list[kbt.Group] = field(default_factory=list[kbt.Group])
    """Groups that `get_items_by_id` finds, as members of other groups, and
    that `get_groups` gives unless `top_groups` is set."""
    top_groups: list[kbt.Group] | None = None
    """What `get_groups` gives instead of `groups`, as KiCad might leave out
    groups inside groups."""
    patterns: list[kbt.Group] = field(default_factory=list[kbt.Group])
    """Tuning patterns: `get_items_by_id` finds them and `get_as_string`'s
    board text lists them, but `get_groups` leaves them out, as KiCad does."""
    netclasses: dict[str, NetClass] = field(default_factory=dict[str, NetClass])
    default_clearance: float | None = 0.2
    drops_pads: bool = False
    enabled_layers: list[BL.ValueType] = field(
        default_factory=lambda: [BL.BL_F_Cu, BL.BL_B_Cu]
    )
    client: FakeClient = field(default_factory=FakeClient)
    grid_origin: tuple[float, float] = (0.0, 0.0)
    """mm."""
    project_path: str = ""
    """As `FakeProject.path`."""
    project_name: str = ""
    reads: Counter[str] = field(default_factory=Counter[str])

    def get_origin(self, origin_type: BoardOriginType.ValueType) -> Vector2:
        self.reads["get_origin"] += 1
        assert origin_type == BoardOriginType.BOT_GRID
        x, y = self.grid_origin
        return Vector2.from_xy(round(x * NM_PER_MM), round(y * NM_PER_MM))

    def get_selection(self) -> list[Wrapper]:
        self.reads["get_selection"] += 1
        return self.selection

    def get_items_by_id(self, ids: Sequence[KIID]) -> list[Wrapper]:
        self.reads["get_items_by_id"] += 1
        wanted = {i.value for i in ids}
        items = [*self.tracks, *self.vias, *self.groups, *self.patterns]
        return [i for i in items if i.id.value in wanted]

    def get_as_string(self) -> str:
        self.reads["get_as_string"] += 1
        generated = "".join(
            f'\t(generated\n\t\t(uuid "{p.id.value}")\n'
            f"\t\t(type tuning_pattern)\n\t\t(name {p.proto.name!r})\n\t)\n"
            for p in self.patterns
        )
        return f'(kicad_pcb\n\t(generator "pcbnew")\n{generated})\n'

    def get_groups(self) -> list[kbt.Group]:
        self.reads["get_groups"] += 1
        return self.groups if self.top_groups is None else self.top_groups

    def selected_ids(self) -> list[str]:
        return [cast(kbt.BoardItem, i).id.value for i in self.selection]

    def get_pads(self) -> list[kbt.Pad]:
        self.reads["get_pads"] += 1
        return self.pads

    def get_pad_shapes_as_polygons(
        self, pads: kbt.Pad | Sequence[kbt.Pad], layer: int
    ) -> PolygonWithHoles | list[PolygonWithHoles] | None:
        """A square around each pad on the layers its padstack lists."""
        self.reads["get_pad_shapes_as_polygons"] += 1

        def shape(p: kbt.Pad) -> PolygonWithHoles | None:
            if layer not in p.padstack.layers:
                return None
            x, y = p.position.x / NM_PER_MM, p.position.y / NM_PER_MM
            h = PAD_HALF_SIZE
            return kipy_rect(x - h, y - h, x + h, y + h)

        if isinstance(pads, kbt.Pad):
            return shape(pads)
        # KiCad silently leaves out pads it has no shape for.
        shapes = [shape(p) for p in pads[: -1 if self.drops_pads else None]]
        return [s for s in shapes if s is not None]

    def get_enabled_layers(self) -> list[BL.ValueType]:
        self.reads["get_enabled_layers"] += 1
        return self.enabled_layers

    def get_vias(self) -> list[kbt.Via]:
        self.reads["get_vias"] += 1
        return self.vias

    def get_tracks(self) -> list[kbt.Track | kbt.ArcTrack]:
        self.reads["get_tracks"] += 1
        return self.tracks

    def get_shapes(self) -> list[kbt.BoardShape]:
        self.reads["get_shapes"] += 1
        return self.shapes

    def get_text(self) -> list[kbt.BoardText | kbt.BoardTextBox]:
        self.reads["get_text"] += 1
        return self.texts

    def get_footprints(self) -> list[kbt.FootprintInstance]:
        self.reads["get_footprints"] += 1
        return self.footprints

    def get_zones(self) -> list[kbt.Zone]:
        self.reads["get_zones"] += 1
        return self.zones

    def get_nets(self) -> list[kbt.Net]:
        """The nets of the board's tracks, vias, pads and copper zones."""
        self.reads["get_nets"] += 1
        names = {
            *(t.net.name for t in self.tracks),
            *(v.net.name for v in self.vias),
            *(p.net.name for p in self.pads),
            *(z.net.name for z in self.zones if z.net is not None),
        }
        return [kbt.Net(name=n) for n in sorted(names)]

    def get_netclass_for_nets(self, nets: Sequence[kbt.Net]) -> dict[str, NetClass]:
        self.reads["get_netclass_for_nets"] += 1
        default = self.default_netclass()
        return {n.name: self.netclasses.get(n.name, default) for n in nets}

    def default_netclass(self) -> NetClass:
        return netclass("Default", self.default_clearance)

    def get_project(self) -> FakeProject:
        self.reads["get_project"] += 1
        return FakeProject(
            [self.default_netclass(), *self.netclasses.values()],
            self.project_path,
            self.project_name,
        )


def layer_snapshot(
    board: FakeBoard, layer: str, replaced: Collection[str] = frozenset()
) -> LayerSnapshot:
    """`layer` of a board read for that layer alone, as a run on it reads it."""
    return BoardSnapshot.read(cast(kipy.board.Board, board), [layer]).slice(
        layer, replaced
    )


COMMIT_CALLS = ["begin_commit", "delete", "create", "push_commit", "select"]
"""`CommitBoard.calls` after a committed run."""


@dataclass
class CommitClient(FakeClient):
    """Answers the item commands commit.py sends directly, besides text shapes."""

    board: CommitBoard | None = None

    def send(self, command: Message, response_type: type[Message]) -> Message:
        assert self.board is not None
        if isinstance(command, CreateItems):
            return self.board.answer_create(command)
        if isinstance(command, DeleteItems):
            return self.board.answer_delete(command)
        if isinstance(command, AddToSelection):
            return self.board.answer_select(command)
        return super().send(command, response_type)


@dataclass
class CommitBoard(FakeBoard):
    """A FakeBoard that also records commits, deletions and creations.

    `calls` lists the commit lifecycle in order: begin_commit, delete,
    create, push_commit, select, drop_commit, and refill_zones, which
    kibends must never call. `short_create` answers CreateItems with one item fewer than
    sent, all reported as created. `refuse_create_nets` refuses only the
    tracks on those nets, `refuse_create_vias` only the vias.
    `unanswered_delete` leaves those ids out of the DeleteItems answer.

    Like KiCad, the CreateItems answer gives each created item sent without
    an id a new one, listed in `assigned_ids`; an item sent with an id keeps
    it, so a test sees whether kibends cleared it. The push deselects the
    deleted items. AddToSelection, recorded in `calls` as select, raises
    `select_error` when it is set, and answers with an item kipy cannot
    unwrap, so reading the answer would fail.
    """

    calls: list[str] = field(default_factory=list[str])
    deleted: list[str] = field(default_factory=list[str])
    created: list[Wrapper] = field(default_factory=list[Wrapper])
    labels: list[str] = field(default_factory=list[str])
    refuse_create: bool = False
    refuse_create_nets: frozenset[str] = frozenset()
    short_create: bool = False
    refuse_create_vias: bool = False
    refuse_delete: frozenset[str] = frozenset()
    unanswered_delete: frozenset[str] = frozenset()
    push_error: BaseException | None = None
    assigned_ids: list[str] = field(default_factory=list[str])
    select_error: BaseException | None = None
    document: DocumentSpecifier = field(default_factory=DocumentSpecifier)

    def __post_init__(self) -> None:
        self.client = CommitClient(board=self)

    def begin_commit(self) -> Commit:
        self.calls.append("begin_commit")
        return Commit(KIID(value="commit"))

    def push_commit(self, commit: Commit, message: str = "") -> None:
        self.calls.append("push_commit")
        self.labels.append(message)
        if self.push_error is not None:
            raise self.push_error
        self.selection = [
            i
            for i in self.selection
            if cast(kbt.BoardItem, i).id.value not in self.deleted
        ]

    def drop_commit(self, commit: Commit) -> None:
        self.calls.append("drop_commit")

    def refill_zones(self, block: bool = True) -> None:
        self.calls.append("refill_zones")

    def answer_create(self, command: CreateItems) -> CreateItemsResponse:
        self.calls.append("create")
        response = CreateItemsResponse()
        answered = command.items[:-1] if self.short_create else command.items
        for packed in answered:
            item = kbt.unwrap(packed)
            refused = (
                self.refuse_create
                or (
                    isinstance(item, kbt.Track)
                    and item.net.name in self.refuse_create_nets
                )
                or (isinstance(item, kbt.Via) and self.refuse_create_vias)
            )
            result = response.created_items.add()
            result.status.code = (
                ItemStatusCode.ISC_INVALID_DATA if refused else ItemStatusCode.ISC_OK
            )
            if not refused:
                kiid = cast(kbt.BoardItem, item).id
                if not kiid.value:
                    kiid.value = f"new-{len(self.assigned_ids)}"
                    self.assigned_ids.append(kiid.value)
                self.created.append(item)
            result.item.CopyFrom(pack_any(item.proto))
        return response

    def answer_select(self, command: AddToSelection) -> SelectionResponse:
        self.calls.append("select")
        if self.select_error is not None:
            raise self.select_error
        created = {cast(kbt.BoardItem, i).id.value: i for i in self.created}
        self.selection += [created[i.value] for i in command.items]
        response = SelectionResponse()
        response.items.append(pack_any(KIID(value="not a board item")))
        return response

    def answer_delete(self, command: DeleteItems) -> DeleteItemsResponse:
        self.calls.append("delete")
        response = DeleteItemsResponse()
        for kiid in command.item_ids:
            if kiid.value in self.unanswered_delete:
                continue
            refused = kiid.value in self.refuse_delete
            response.deleted_items.add(
                id=kiid,
                status=ItemDeletionStatus.IDS_IMMUTABLE
                if refused
                else ItemDeletionStatus.IDS_OK,
            )
            if not refused:
                self.deleted.append(kiid.value)
        return response


class FakeKiCad:
    def __init__(self, board: FakeBoard) -> None:
        self.board = board

    def get_board(self) -> FakeBoard:
        return self.board


def via_ids(field: ObstacleField | None) -> set[str]:
    """The vias whose rings `field` holds."""
    assert isinstance(field, Obstacles)
    return {
        o.description.removeprefix("via ")
        for o in field.items
        if o.description.startswith("via ")
    }


RETURN_SIGNAL_VIA = (5.0, 0.0)
"""Where `return_via_board`'s SIG via sits: on the straight line between its
route's ends."""
RETURN_VIAS_APART = 0.8
"""The centre distance `return_via_board`'s two vias keep: their 0.6 mm
rings and the 0.2 mm clearance, mm."""


def return_via_board(ground: tuple[float, float], *, stitching: bool) -> CommitBoard:
    """A SIG route from (0, 0) to (10, 0), kinked on F.Cu and on B.Cu,
    through the selected SIG via at `RETURN_SIGNAL_VIA`, and a GND via at
    `ground`: unselected, or a selected stitching via in a GND plane on the
    inner layers."""
    route = [
        kipy_track("a", "SIG", (0, 0), (2.5, -1.0)),
        kipy_track("b", "SIG", (2.5, -1.0), RETURN_SIGNAL_VIA),
        kipy_track("c", "SIG", RETURN_SIGNAL_VIA, (7.5, 1.0), layer=BL.BL_B_Cu),
        kipy_track("d", "SIG", (7.5, 1.0), (10, 0), layer=BL.BL_B_Cu),
    ]
    signal = kipy_via("v", "SIG", *RETURN_SIGNAL_VIA)
    return_via = kipy_via("g", "GND", *ground)
    plane = kipy_copper_zone(
        "plane", "GND", kipy_rect(-2, -6, 12, 6), [BL.BL_In1_Cu, BL.BL_In2_Cu]
    )
    return CommitBoard(
        selection=[*route, signal, *([return_via] if stitching else [])],
        tracks=list(route),
        vias=[signal, return_via],
        zones=[plane] if stitching else [],
        enabled_layers=[BL.BL_F_Cu, BL.BL_In1_Cu, BL.BL_In2_Cu, BL.BL_B_Cu],
    )


NEUTRAL_TOLERANCE = 1e-12
"""How near `neutral_distance` brackets the distance, mm."""


def neutral_distance(params: EnergyParams, apart: float) -> float:
    """The centre distance at which a ground via and a regular one that
    keep `apart` stop pushing or pulling each other: where the slopes of
    their obstacle term and attraction cancel."""
    obstacle = obstacle_integrand(params)
    attraction = attraction_integrand(params)

    def slope(r: float) -> float:
        push = obstacle(np.array([r - apart]))[1]
        pull = attraction(np.array([r]), np.array([apart]))[1]
        return float(push[0] + pull[0])

    low, high = apart, apart + params.falloff_range
    rising = slope(high) > slope(low)
    while high - low > NEUTRAL_TOLERANCE:
        middle = (low + high) / 2
        if (slope(middle) > 0) == rising:
            high = middle
        else:
            low = middle
    return (low + high) / 2
