"""Read the selection and the copper layers from a kipy Board.

Everything returned is free of kipy: segments use integer nanometres as
`kibends.chain` expects, the rest is mm floats and shapely geometry. Every
coordinate is in KiCad's board frame, with y pointing down.

The converters `_ring`, `_polygon`, `arc_points`, `circle_points`,
`_pad_shapes`, `_clearance_override`, the Edge.Cuts outline helpers and the
canonical layer naming are copied and trimmed from kistitch's snapshot.py
(MIT, Copyright (c) 2025 Hat Labs Oy).
"""

import enum
import math
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import kipy.board
import kipy.board_types as kbt
import shapely
from google.protobuf.message import Message
from kipy import common_types
from kipy.errors import ApiError
from kipy.geometry import PolygonWithHoles, PolyLine, Vector2
from kipy.proto.board.board_types_pb2 import BoardLayer, PadType, ZoneType
from kipy.proto.common import ApiStatusCode
from kipy.proto.common.commands import base_commands_pb2
from kipy.util.board_layer import canonical_name, layer_from_canonical_name
from loguru import logger
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

from kibends.chain import NM_PER_MM, Point2, SameNetCopper, SameNetTrack, Segment

EDGE_CUTS = "Edge.Cuts"
DEFAULT_NETCLASS = "Default"
# KiCad's built-in Default netclass clearance, for a project that reports none.
FALLBACK_CLEARANCE = 0.2

# Edge.Cuts arc endpoints don't meet exactly in floating point; snapping to
# 1 µm lets polygonize close the outline.
EDGE_DECIMALS = 3
ARC_SEGMENT_ANGLE = math.radians(5)
MIN_ARC_SEGMENTS = 4
COLLINEAR_EPSILON = 1e-12


class SelectionError(ValueError):
    """The selection holds nothing kibends can smooth."""


class PadKind(enum.Enum):
    SMD = "smd"
    PTH = "pth"
    NPTH = "npth"
    OTHER = "other"


@dataclass(frozen=True, eq=False)
class Pad:
    """A pad of any layer; `shape` is its copper on the snapshot's layer."""

    id: str
    name: str
    net: str
    kind: PadKind
    position: Point2
    drill: float
    shape: BaseGeometry | None
    clearance_override: float | None = None


@dataclass(frozen=True, eq=False)
class Track:
    id: str
    net: str
    width: float
    centreline: LineString


@dataclass(frozen=True, eq=False)
class Via:
    """A via, taken to have copper on every layer."""

    id: str
    net: str
    position: Point2
    diameter: float
    drill: float


@dataclass(frozen=True, eq=False)
class Graphic:
    """Copper drawn as a shape or text; `net` is empty for no net."""

    description: str
    net: str
    geometry: BaseGeometry


@dataclass(frozen=True, eq=False)
class LayerSnapshot:
    """The copper of one layer, without the tracks being replaced.

    Pads and vias are listed whatever their layers, since their holes cross
    every layer. Copper zones are not read: they reflow after an edit.
    """

    layer: str
    pads: tuple[Pad, ...]
    tracks: tuple[Track, ...]
    vias: tuple[Via, ...]
    graphics: tuple[Graphic, ...]
    keepouts: tuple[BaseGeometry, ...]
    """Rule areas on the layer that keep out tracks."""
    edge_paths: tuple[tuple[Point2, ...], ...]
    clearances: Mapping[str, float]
    """Netclass clearance per net name, in mm."""
    default_clearance: float

    def clearance(self, net: str) -> float:
        """The net's netclass clearance; the Default one for no net."""
        return self.clearances.get(net, self.default_clearance)

    def same_net_copper(self, net: str) -> list[SameNetCopper]:
        """Pad, via and copper shape outlines and the tracks of `net` on the layer.

        This is what `extract_chain` pins the chain to.
        """
        return [
            *(p.shape for p in self.pads if p.net == net and p.shape is not None),
            *(
                Point(v.position).buffer(v.diameter / 2)
                for v in self.vias
                if v.net == net
            ),
            *(SameNetTrack(t.centreline, t.width) for t in self.tracks if t.net == net),
            *(g.geometry for g in self.graphics if g.net == net),
        ]


@dataclass(frozen=True, eq=False)
class BoardSnapshot:
    """One read of the board, holding the copper of the layers asked for.

    Board-wide reads happen once; pad shapes and copper text, which KiCad
    hands out per layer, are read once per layer. Slicing makes no requests.
    """

    layers: frozenset[str]
    """The copper layers read, the ones `slice` accepts."""
    pads: Mapping[str, tuple[Pad, ...]]
    """Every pad, with its copper on each layer read."""
    tracks: Mapping[str, tuple[Track, ...]]
    vias: tuple[Via, ...]
    graphics: Mapping[str, tuple[Graphic, ...]]
    keepouts: Mapping[str, tuple[BaseGeometry, ...]]
    edge_paths: tuple[tuple[Point2, ...], ...]
    clearances: Mapping[str, float]
    """Netclass clearance of every net on the board, in mm."""
    default_clearance: float

    @classmethod
    def read(cls, board: kipy.board.Board, layers: Iterable[str]) -> BoardSnapshot:
        """Read the board once for the copper layers `layers`."""
        wanted = sorted(set(layers))
        footprints = board.get_footprints()
        all_tracks = board.get_tracks()
        kipy_pads = board.get_pads()
        vias = tuple(_read_via(v) for v in board.get_vias())
        shapes = [
            *board.get_shapes(),
            *(s for f in footprints for s in f.definition.shapes),
        ]
        texts = _texts(board, footprints)
        zones = board.get_zones()
        default_clearance = _default_clearance(board)
        shapeless = _read_pads(kipy_pads, footprints)
        pads_by_layer: dict[str, tuple[Pad, ...]] = {}
        for layer in wanted:
            on_layer = _pad_shapes(board, kipy_pads, layer_from_canonical_name(layer))
            pads_by_layer[layer] = tuple(
                replace(p, shape=on_layer.get(p.id)) for p in shapeless
            )
        graphics = {
            layer: (
                *_read_copper_shapes(shapes, layer),
                *_read_copper_text(board, texts, layer),
            )
            for layer in wanted
        }
        nets = {
            *(p.net for p in shapeless),
            *(t.net.name for t in all_tracks),
            *(v.net for v in vias),
            *(g.net for gs in graphics.values() for g in gs),
        } - {""}
        return cls(
            layers=frozenset(wanted),
            pads=pads_by_layer,
            tracks={
                layer: tuple(
                    _read_track(t)
                    for t in all_tracks
                    if canonical_name(t.layer) == layer
                )
                for layer in wanted
            },
            vias=vias,
            graphics=graphics,
            keepouts={layer: _read_track_keepouts(zones, layer) for layer in wanted},
            edge_paths=tuple(tuple(path) for path in read_edge_paths(shapes)),
            clearances=_read_clearances(board, sorted(nets), default_clearance),
            default_clearance=default_clearance,
        )

    def slice(self, layer: str, replaced_ids: Collection[str]) -> LayerSnapshot:
        """`layer` and everything crossing it, less the tracks being replaced."""
        if layer not in self.layers:
            raise ValueError(f"{layer} was not read from the board")
        snap = LayerSnapshot(
            layer=layer,
            pads=self.pads[layer],
            tracks=tuple(t for t in self.tracks[layer] if t.id not in replaced_ids),
            vias=self.vias,
            graphics=self.graphics[layer],
            keepouts=self.keepouts[layer],
            edge_paths=self.edge_paths,
            clearances=self.clearances,
            default_clearance=self.default_clearance,
        )
        logger.info(
            f"Snapshot of {layer}: {len(snap.pads)} pads, {len(snap.tracks)} tracks, "
            f"{len(snap.vias)} vias, {len(snap.graphics)} copper graphics, "
            f"{len(snap.keepouts)} track keepouts"
        )
        return snap


def read_selection(board: kipy.board.Board) -> list[Segment]:
    """The selected straight tracks; arcs are refused, anything else ignored."""
    items = board.get_selection()
    arcs = [i for i in items if isinstance(i, kbt.ArcTrack)]
    if arcs:
        ids = ", ".join(a.id.value for a in arcs)
        raise SelectionError(
            f"selection contains arc tracks, which kibends does not modify: {ids}"
        )
    tracks = [i for i in items if isinstance(i, kbt.Track)]
    others = Counter(type(i).__name__ for i in items if not isinstance(i, kbt.Track))
    if others:
        listing = ", ".join(f"{n} {name}" for name, n in sorted(others.items()))
        logger.warning(f"Ignoring selected items that are not tracks: {listing}")
    if not tracks:
        raise SelectionError("selection has no tracks")
    return [segment_from_track(t) for t in tracks]


def segment_from_track(t: kbt.Track) -> Segment:
    return Segment(
        id=t.id.value,
        start=(t.start.x, t.start.y),
        end=(t.end.x, t.end.y),
        width=t.width,
        layer=canonical_name(t.layer),
        net=t.net.name,
        locked=t.locked,
    )


# ---- geometry -----------------------------------------------------------------


def arc_points(start: Point2, mid: Point2, end: Point2) -> list[Point2]:
    """Tessellate the circular arc through start, mid and end."""
    (x1, y1), (x2, y2), (x3, y3) = start, mid, end
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < COLLINEAR_EPSILON:
        return [start, end]
    s1, s2, s3 = x1**2 + y1**2, x2**2 + y2**2, x3**2 + y3**2
    ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    a1, a2, a3 = (math.atan2(p[1] - uy, p[0] - ux) for p in (start, mid, end))
    sweep = (a3 - a1) % (2 * math.pi)
    if (a2 - a1) % (2 * math.pi) > sweep:
        sweep -= 2 * math.pi
    n = max(MIN_ARC_SEGMENTS, math.ceil(abs(sweep) / ARC_SEGMENT_ANGLE))
    pts = [
        (ux + r * math.cos(a1 + sweep * i / n), uy + r * math.sin(a1 + sweep * i / n))
        for i in range(1, n)
    ]
    # Keep the exact endpoints so neighbouring segments still meet.
    return [start, *pts, end]


def circle_points(centre: Point2, radius: float) -> list[Point2]:
    n = max(MIN_ARC_SEGMENTS, math.ceil(2 * math.pi / ARC_SEGMENT_ANGLE))
    cx, cy = centre
    pts = [
        (
            cx + radius * math.cos(2 * math.pi * i / n),
            cy + radius * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]
    return [*pts, pts[0]]


def valid_polygon(
    shell: Sequence[Point2], holes: Sequence[Sequence[Point2]] = ()
) -> BaseGeometry:
    poly = Polygon(shell, list(holes))
    return poly if poly.is_valid else poly.buffer(0)


def _snap(p: Point2) -> Point2:
    return (round(p[0], EDGE_DECIMALS), round(p[1], EDGE_DECIMALS))


def _edge_lines(paths: Iterable[Sequence[Point2]]) -> list[LineString]:
    return [LineString([_snap(p) for p in path]) for path in paths if len(path) >= 2]


def edge_paths_closed(paths: Iterable[Sequence[Point2]]) -> bool:
    """Whether the Edge.Cuts paths form closed loops and nothing else.

    An open outline around a closed cutout still polygonises to the cutout,
    so the dangling and cut edges are what tell an open outline apart.
    """
    lines = _edge_lines(paths)
    if not lines:
        return False
    polygons, cuts, dangles, invalid = shapely.polygonize_full(
        shapely.get_parts(unary_union(lines))
    )
    return not shapely.is_empty(polygons) and all(
        shapely.is_empty(g) for g in (cuts, dangles, invalid)
    )


def outline_from_edge_paths(paths: Iterable[Sequence[Point2]]) -> BaseGeometry:
    """Board area polygonised from Edge.Cuts paths.

    Closed Edge.Cuts loops nest even-odd: a loop inside the outline is a
    cutout, a loop inside a cutout is board again.
    """
    lines = _edge_lines(paths)
    if not lines:
        return Polygon()
    outline: BaseGeometry = Polygon()
    for face in polygonize(unary_union(lines)):
        outline = outline.symmetric_difference(Polygon(face.exterior))
    return outline


# ---- kipy conversion ------------------------------------------------------------


def _mm(v: Vector2) -> Point2:
    return (v.x / NM_PER_MM, v.y / NM_PER_MM)


def _ring(polyline: PolyLine) -> list[Point2]:
    pts: list[Point2] = []
    for node in polyline.nodes:
        if node.has_point:
            pts.append(_mm(node.point))
        else:
            a = node.arc
            pts.extend(arc_points(_mm(a.start), _mm(a.mid), _mm(a.end)))
    return pts


def _polygon(pwh: PolygonWithHoles) -> BaseGeometry:
    return valid_polygon(_ring(pwh.outline), [_ring(h) for h in pwh.holes])


def _pad_kind(pad_type: PadType.ValueType) -> PadKind:
    return {
        PadType.PT_SMD: PadKind.SMD,
        PadType.PT_PTH: PadKind.PTH,
        PadType.PT_NPTH: PadKind.NPTH,
    }.get(pad_type, PadKind.OTHER)


def _pad_shapes(
    board: kipy.board.Board, pads: Sequence[kbt.Pad], layer_id: BoardLayer.ValueType
) -> dict[str, BaseGeometry]:
    """Pad copper on one layer, by pad id.

    `get_pad_shapes_as_polygons` silently drops pads with no shape on the
    layer, so the result can't be zipped with the request. Ask only for pads
    whose padstack lists the layer and fall back to per-pad queries if the
    lengths still differ.
    """
    on = [p for p in pads if layer_id in set(p.padstack.layers)]
    if not on:
        return {}
    result: list[PolygonWithHoles | None] = list(
        board.get_pad_shapes_as_polygons(on, layer_id)
    )
    if len(result) != len(on):
        logger.debug("Pad shape query dropped pads; querying one by one")
        result = [board.get_pad_shapes_as_polygons(p, layer_id) for p in on]
    return {
        pad.id.value: _polygon(pwh)
        for pad, pwh in zip(on, result, strict=True)
        if pwh is not None
    }


def _clearance_override(proto_owner: Message, field_name: str) -> float | None:
    if not proto_owner.HasField(field_name):
        return None
    value_nm: int = getattr(proto_owner, field_name).value_nm
    return value_nm / NM_PER_MM


def _read_pads(
    pads: Sequence[kbt.Pad], footprints: Sequence[kbt.FootprintInstance]
) -> tuple[Pad, ...]:
    """Every pad with no `shape` yet; the caller sets each layer's copper."""
    refs = {f.id.value: f.reference_field.text.value for f in footprints}
    fp_overrides = {
        f.id.value: _clearance_override(f.proto.overrides, "copper_clearance")
        for f in footprints
    }
    out: list[Pad] = []
    for p in pads:
        drill = p.padstack.drill.diameter
        parent = p.parent.value if p.parent is not None else None
        override = _clearance_override(p.proto, "copper_clearance_override")
        if override is None and parent is not None:
            override = fp_overrides.get(parent)
        ref = refs.get(parent) if parent is not None else None
        out.append(
            Pad(
                id=p.id.value,
                name=f"{ref}:{p.number}" if ref else p.id.value,
                net=p.net.name,
                kind=_pad_kind(p.pad_type),
                position=_mm(p.position),
                # Oval drills: the larger axis is the conservative clearance.
                drill=max(drill.x, drill.y) / NM_PER_MM,
                shape=None,
                clearance_override=override,
            )
        )
    return tuple(out)


def _read_track(t: kbt.Track | kbt.ArcTrack) -> Track:
    if isinstance(t, kbt.ArcTrack):
        line = LineString(arc_points(_mm(t.start), _mm(t.mid), _mm(t.end)))
    else:
        line = LineString([_mm(t.start), _mm(t.end)])
    return Track(
        id=t.id.value, net=t.net.name, width=t.width / NM_PER_MM, centreline=line
    )


def _read_via(v: kbt.Via) -> Via:
    return Via(
        id=v.id.value,
        net=v.net.name,
        position=_mm(v.position),
        diameter=v.diameter / NM_PER_MM,
        drill=v.drill_diameter / NM_PER_MM,
    )


def _rectangle_ring(shape: common_types.Rectangle) -> list[Point2]:
    """The closed outline of a rectangle, with arcs at rounded corners."""
    (ax, ay), (bx, by) = _mm(shape.top_left), _mm(shape.bottom_right)
    x0, x1, y0, y1 = min(ax, bx), max(ax, bx), min(ay, by), max(ay, by)
    # kipy 0.8.0 has no accessor for the corner radius KiCad 10 added, and
    # copper text renders as a plain Rectangle, so the proto is the only source.
    proto = shape._graphic_proto  # pyright: ignore[reportPrivateUsage]
    r = proto.rectangle.corner_radius.value_nm / NM_PER_MM
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    if r <= 0:
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    d = r * (1 - math.sqrt(0.5))
    corners = [
        ((x1 - r, y0), (x1 - d, y0 + d), (x1, y0 + r)),
        ((x1, y1 - r), (x1 - d, y1 - d), (x1 - r, y1)),
        ((x0 + r, y1), (x0 + d, y1 - d), (x0, y1 - r)),
        ((x0, y0 + r), (x0 + d, y0 + d), (x0 + r, y0)),
    ]
    ring = [p for start, mid, end in corners for p in arc_points(start, mid, end)]
    return [*ring, ring[0]]


def _shape_geometry(
    shape: common_types.GraphicShape | kbt.BoardShape, min_stroke: float = 0.0
) -> BaseGeometry | None:
    """The area a drawn shape covers: its stroke, and its inside if filled.

    Returns None for a shape kind this does not convert.
    """
    half_stroke = max(shape.attributes.stroke.width / NM_PER_MM, min_stroke) / 2
    filled = shape.attributes.fill.filled
    area: BaseGeometry
    match shape:
        case common_types.Segment():
            return LineString([_mm(shape.start), _mm(shape.end)]).buffer(half_stroke)
        case common_types.Arc():
            path = arc_points(_mm(shape.start), _mm(shape.mid), _mm(shape.end))
            return LineString(path).buffer(half_stroke)
        case common_types.Circle():
            area = Point(_mm(shape.center)).buffer(shape.radius() / NM_PER_MM)
        case common_types.Rectangle():
            area = Polygon(_rectangle_ring(shape))
        case common_types.Polygon():
            area = unary_union([_polygon(p) for p in shape.polygons])
        case _:
            return None
    outline = area.buffer(half_stroke) if filled else area.boundary.buffer(half_stroke)
    return outline if not outline.is_empty else None


def _read_copper_shapes(shapes: Iterable[kbt.BoardShape], layer: str) -> list[Graphic]:
    graphics: list[Graphic] = []
    for s in shapes:
        if canonical_name(s.layer) != layer:
            continue
        geometry = _shape_geometry(s)
        if geometry is None:
            logger.warning(
                f"Ignoring copper {type(s).__name__} {s.id.value} on {layer}; "
                "only the DRC check covers it"
            )
            continue
        graphics.append(Graphic(f"copper shape {s.id.value}", s.net.name, geometry))
    return graphics


def _texts(
    board: kipy.board.Board, footprints: Sequence[kbt.FootprintInstance]
) -> list[kbt.BoardText | kbt.BoardTextBox]:
    texts: list[kbt.BoardText | kbt.BoardTextBox] = list(board.get_text())
    for f in footprints:
        for t in f.texts_and_fields:
            if isinstance(t, kbt.Field):
                if t.visible:
                    texts.append(t.text)
            else:
                texts.append(t)
    return texts


def _read_copper_text(
    board: kipy.board.Board,
    all_texts: Sequence[kbt.BoardText | kbt.BoardTextBox],
    layer: str,
) -> list[Graphic]:
    """Copper text on `layer` as polygons, rendered by KiCad's GetTextAsShapes."""
    texts = [t for t in all_texts if canonical_name(t.layer) == layer]
    if not texts:
        return []
    command = base_commands_pb2.GetTextAsShapes()
    for t in texts:
        entry = command.text.add()
        if isinstance(t, kbt.BoardText):
            entry.text.CopyFrom(t.as_text().proto)
        else:
            entry.textbox.CopyFrom(t.as_textbox().proto)
    try:
        reply = board.client.send(command, base_commands_pb2.GetTextAsShapesResponse)
    except ApiError as e:
        # A busy KiCad refuses every later request too, so going on without
        # the text would only postpone the failure.
        if e.code == ApiStatusCode.AS_BUSY:
            raise
        logger.warning(
            f"KiCad could not render the copper text on {layer} ({e}); "
            "only the DRC check covers it"
        )
        return []
    graphics: list[Graphic] = []
    for t, rendered in zip(texts, reply.text_with_shapes, strict=True):
        stroke = t.attributes.stroke_width / NM_PER_MM
        parts = [
            _shape_geometry(s, stroke)
            for s in common_types.CompoundShape(rendered.shapes).shapes
        ]
        geometry = unary_union([p for p in parts if p is not None])
        if not geometry.is_empty:
            graphics.append(Graphic(f"copper text '{t.value}'", "", geometry))
    return graphics


def _read_track_keepouts(
    zones: Iterable[kbt.Zone], layer: str
) -> tuple[BaseGeometry, ...]:
    return tuple(
        _polygon(PolygonWithHoles(p))
        for z in zones
        if z.type == ZoneType.ZT_RULE_AREA
        and z.proto.rule_area_settings.keepout_tracks
        and layer in {canonical_name(L) for L in z.layers}
        for p in z.proto.outline.polygons
    )


def _default_clearance(board: kipy.board.Board) -> float:
    for nc in board.get_project().get_net_classes():
        if nc.name == DEFAULT_NETCLASS and nc.clearance is not None:
            return nc.clearance / NM_PER_MM
    logger.warning(
        f"Project reports no {DEFAULT_NETCLASS} netclass clearance; "
        f"assuming {FALLBACK_CLEARANCE} mm"
    )
    return FALLBACK_CLEARANCE


def _read_clearances(
    board: kipy.board.Board, nets: Sequence[str], default: float
) -> dict[str, float]:
    if not nets:
        return {}
    classes = board.get_netclass_for_nets([kbt.Net(name=n) for n in nets])
    return {
        name: default if nc.clearance is None else nc.clearance / NM_PER_MM
        for name, nc in classes.items()
    }


def read_edge_paths(shapes: Iterable[kbt.BoardShape]) -> list[list[Point2]]:
    """The Edge.Cuts paths among `shapes`."""
    return [
        path
        for s in shapes
        if canonical_name(s.layer) == EDGE_CUTS
        for path in _edge_paths(s)
    ]


def _edge_paths(shape: kbt.BoardShape) -> list[list[Point2]]:
    match shape:
        case kbt.BoardSegment():
            return [[_mm(shape.start), _mm(shape.end)]]
        case kbt.BoardArc():
            return [arc_points(_mm(shape.start), _mm(shape.mid), _mm(shape.end))]
        case kbt.BoardCircle():
            return [circle_points(_mm(shape.center), shape.radius() / NM_PER_MM)]
        case kbt.BoardRectangle():
            return [_rectangle_ring(shape)]
        case kbt.BoardPolygon():
            return [
                [*ring, ring[0]]
                for pwh in shape.polygons
                for ring in (_ring(pwh.outline), *(_ring(h) for h in pwh.holes))
            ]
        case _:
            logger.warning(
                f"Ignoring unsupported Edge.Cuts shape {type(shape).__name__}"
            )
            return []
