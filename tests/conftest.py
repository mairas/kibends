"""Shared fixtures and kipy fakes.

FakeBoard and the kipy builders are adapted from kistitch's
tests/test_snapshot.py (MIT, Copyright (c) 2025 Hat Labs Oy).
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import kipy.board_types as kbt
import pytest
from google.protobuf.message import Message
from kipy.common_types import Commit, Text, TextBox
from kipy.geometry import PolygonWithHoles, PolyLineNode, Vector2
from kipy.project_types import NetClass
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.board.board_types_pb2 import PadType, ZoneType
from kipy.proto.common.commands import base_commands_pb2
from kipy.proto.common.commands.editor_commands_pb2 import (
    CreateItems,
    CreateItemsResponse,
    DeleteItems,
    DeleteItemsResponse,
    ItemDeletionStatus,
    ItemStatusCode,
)
from kipy.proto.common.types import base_types_pb2, project_settings_pb2
from kipy.proto.common.types.base_types_pb2 import KIID, DocumentSpecifier
from kipy.wrapper import Wrapper
from loguru import logger

from kibends.chain import NM_PER_MM

PAD_HALF_SIZE = 0.25
TEXT_HALF_WIDTH = 1.0
TEXT_HALF_HEIGHT = 0.5


@pytest.fixture
def log() -> Iterator[list[str]]:
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{level} {message}")
    yield lines
    logger.remove(sink)


def nm(v: float) -> int:
    return round(v * NM_PER_MM)


def vec(x: float, y: float) -> Vector2:
    return Vector2.from_xy(nm(x), nm(y))


def kipy_rect(x0: float, y0: float, x1: float, y1: float) -> PolygonWithHoles:
    pwh = PolygonWithHoles()
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        pwh.outline.append(PolyLineNode.from_xy(nm(x), nm(y)))
    pwh.outline.closed = True
    return pwh


def kipy_track(
    track_id: str,
    net: str,
    start: tuple[float, float],
    end: tuple[float, float],
    width: float = 0.25,
    layer: int = BL.BL_F_Cu,
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


def kipy_arc(track_id: str, net: str = "SIG") -> kbt.ArcTrack:
    a = kbt.ArcTrack()
    a.proto.id.value = track_id
    a.net = kbt.Net(name=net)
    a.start = vec(0, 0)
    a.mid = vec(1, 1)
    a.end = vec(2, 0)
    a.width = nm(0.25)
    a.layer = BL.BL_F_Cu
    return a


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
    via_id: str, net: str, x: float, y: float, diameter: float = 0.6, drill: float = 0.3
) -> kbt.Via:
    v = kbt.Via()
    v.proto.id.value = via_id
    v.net = kbt.Net(name=net)
    v.position = vec(x, y)
    v.padstack.proto.copper_layers.add()
    v.diameter = nm(diameter)
    v.drill_diameter = nm(drill)
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


def kipy_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    layer: int = BL.BL_F_Cu,
    width: float = 0.1,
    net: str = "",
    shape_id: str = "shape",
) -> kbt.BoardSegment:
    s = kbt.BoardSegment()
    s.proto.id.value = shape_id
    s.start = vec(*start)
    s.end = vec(*end)
    s.layer = layer
    s.attributes.stroke.width = nm(width)
    s.net = kbt.Net(name=net)
    return s


def kipy_text(value: str, x: float, y: float, layer: int = BL.BL_F_Cu) -> kbt.BoardText:
    t = kbt.BoardText()
    t.proto.id.value = f"text-{value}"
    t.value = value
    t.position = vec(x, y)
    t.layer = layer
    return t


def kipy_field(
    value: str, x: float, y: float, layer: int = BL.BL_F_Cu, visible: bool = True
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
    zone_id: str = "zone",
) -> kbt.Zone:
    z = kbt.Zone()
    z.proto.id.value = zone_id
    z.proto.type = ZoneType.ZT_RULE_AREA if rule_area else ZoneType.ZT_COPPER
    z.proto.layers.extend(layers)
    z.proto.outline.polygons.add().CopyFrom(outline.proto)
    if rule_area:
        z.proto.rule_area_settings.keepout_tracks = keepout_tracks
    else:
        z.proto.copper_settings.net.name = "GND"
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

    def send(self, command: Message, response_type: type[Message]) -> Message:
        assert isinstance(command, base_commands_pb2.GetTextAsShapes)
        if self.error is not None:
            raise self.error
        response = base_commands_pb2.GetTextAsShapesResponse()
        for entry in command.text:
            text = Text(entry.text) if entry.HasField("text") else None
            box = TextBox(entry.textbox) if text is None else None
            position = text.position if text is not None else box.top_left  # type: ignore[union-attr]
            reply = response.text_with_shapes.add()
            reply.shapes.shapes.append(
                _text_box(position.x / NM_PER_MM, position.y / NM_PER_MM)
            )
        return response


@dataclass
class FakeProject:
    netclasses: list[NetClass]
    path: str = ""

    def get_net_classes(self) -> list[NetClass]:
        return self.netclasses


@dataclass
class FakeBoard:
    """Answers the reads the snapshot makes."""

    selection: list[Wrapper] = field(default_factory=list)
    pads: list[kbt.Pad] = field(default_factory=list)
    vias: list[kbt.Via] = field(default_factory=list)
    tracks: list[kbt.Track | kbt.ArcTrack] = field(default_factory=list)
    shapes: list[kbt.BoardShape] = field(default_factory=list)
    texts: list[kbt.BoardText | kbt.BoardTextBox] = field(default_factory=list)
    footprints: list[kbt.FootprintInstance] = field(default_factory=list)
    zones: list[kbt.Zone] = field(default_factory=list)
    netclasses: dict[str, NetClass] = field(default_factory=dict)
    default_clearance: float | None = 0.2
    name: str = "board.kicad_pcb"
    project_dir: str = ""
    drops_pads: bool = False
    client: FakeClient = field(default_factory=FakeClient)

    def get_selection(self) -> list[Wrapper]:
        return self.selection

    def get_pads(self) -> list[kbt.Pad]:
        return self.pads

    def get_pad_shapes_as_polygons(
        self, pads: kbt.Pad | Sequence[kbt.Pad], layer: int
    ) -> PolygonWithHoles | list[PolygonWithHoles]:
        def shape(p: kbt.Pad) -> PolygonWithHoles:
            x, y = p.position.x / NM_PER_MM, p.position.y / NM_PER_MM
            h = PAD_HALF_SIZE
            return kipy_rect(x - h, y - h, x + h, y + h)

        if isinstance(pads, kbt.Pad):
            return shape(pads)
        # KiCad silently leaves out pads it has no shape for.
        return [shape(p) for p in pads[: -1 if self.drops_pads else None]]

    def get_vias(self) -> list[kbt.Via]:
        return self.vias

    def get_tracks(self) -> list[kbt.Track | kbt.ArcTrack]:
        return self.tracks

    def get_shapes(self) -> list[kbt.BoardShape]:
        return self.shapes

    def get_text(self) -> list[kbt.BoardText | kbt.BoardTextBox]:
        return self.texts

    def get_footprints(self) -> list[kbt.FootprintInstance]:
        return self.footprints

    def get_zones(self) -> list[kbt.Zone]:
        return self.zones

    def get_netclass_for_nets(self, nets: Sequence[kbt.Net]) -> dict[str, NetClass]:
        default = self.default_netclass()
        return {n.name: self.netclasses.get(n.name, default) for n in nets}

    def default_netclass(self) -> NetClass:
        return netclass("Default", self.default_clearance)

    def get_project(self) -> FakeProject:
        return FakeProject(
            [self.default_netclass(), *self.netclasses.values()], self.project_dir
        )

    def get_as_string(self) -> str:
        """The board file text, holding only the straight tracks."""
        segments = "".join(
            f"\t(segment\n\t\t(start {t.start.x / NM_PER_MM} {t.start.y / NM_PER_MM})\n"
            f"\t\t(end {t.end.x / NM_PER_MM} {t.end.y / NM_PER_MM})\n"
            f"\t\t(width {t.width / NM_PER_MM})\n"
            f'\t\t(layer "F.Cu")\n\t\t(net "{t.net.name}")\n'
            f'\t\t(uuid "{t.id.value}")\n\t)\n'
            for t in self.tracks
            if type(t) is kbt.Track
        )
        return f"(kicad_pcb\n\t(version 20260206)\n{segments}\t(embedded_fonts no)\n)\n"


@dataclass
class CommitClient(FakeClient):
    """Answers the item commands commit.py sends directly, besides text shapes."""

    board: "CommitBoard | None" = None

    def send(self, command: Message, response_type: type[Message]) -> Message:
        assert self.board is not None
        if isinstance(command, CreateItems):
            return self.board.answer_create(command)
        if isinstance(command, DeleteItems):
            return self.board.answer_delete(command)
        return super().send(command, response_type)


@dataclass
class CommitBoard(FakeBoard):
    """A FakeBoard that also records commits, deletions and creations.

    `calls` lists the commit lifecycle in order: begin_commit, delete,
    create, push_commit, drop_commit, and refill_zones, which kibends must
    never call.
    """

    calls: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    created: list[Wrapper] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    refuse_create: bool = False
    refuse_delete: frozenset[str] = frozenset()
    push_error: Exception | None = None
    document: DocumentSpecifier = field(default_factory=DocumentSpecifier)

    def __post_init__(self) -> None:
        self.client = CommitClient(board=self)

    def get_items_by_id(self, ids: Sequence[KIID]) -> list[Wrapper]:
        wanted = {i.value for i in ids}
        return [t for t in self.tracks if t.id.value in wanted]

    def begin_commit(self) -> Commit:
        self.calls.append("begin_commit")
        return Commit(KIID(value="commit"))

    def push_commit(self, commit: Commit, message: str = "") -> None:
        self.calls.append("push_commit")
        self.labels.append(message)
        if self.push_error is not None:
            raise self.push_error

    def drop_commit(self, commit: Commit) -> None:
        self.calls.append("drop_commit")

    def refill_zones(self, block: bool = True) -> None:
        self.calls.append("refill_zones")

    def answer_create(self, command: CreateItems) -> CreateItemsResponse:
        self.calls.append("create")
        response = CreateItemsResponse()
        for packed in command.items:
            result = response.created_items.add()
            result.item.CopyFrom(packed)
            result.status.code = (
                ItemStatusCode.ISC_INVALID_DATA
                if self.refuse_create
                else ItemStatusCode.ISC_OK
            )
            if not self.refuse_create:
                self.created.append(kbt.unwrap(packed))
        return response

    def answer_delete(self, command: DeleteItems) -> DeleteItemsResponse:
        self.calls.append("delete")
        response = DeleteItemsResponse()
        for kiid in command.item_ids:
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

    def get_kicad_binary_path(self, binary_name: str) -> str:
        return ""
