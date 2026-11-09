"""Read the selection and the copper layers from a kipy Board.

Everything returned is free of kipy: segments use integer nanometres as
`kibends.chain` expects, the rest is mm floats and shapely geometry. Every
coordinate is in KiCad's board frame, with y pointing down.

A run reads the selection and the board once: the copper layers of the
selection, or every enabled copper layer when a selected via may move.
`BoardSnapshot.slice` gives one `LayerSnapshot` per layer without further
requests.

For each candidate via that may move, the snapshot records the zones of
its net that may connect to it (`ViaZone`, `_via_zones`). kibends never
reads a zone's fill, which holds where other copper stood at the last
refill and turns stale once the run moves copper: a zone's region is built
from outlines and priorities alone (`_zone_region`). Where such a via may
go is `vias`' to decide (`ZoneUnion`, `zone_unions`). Thermal reliefs,
island removal, hatched fills and minimum-width necks are not modelled.

The same read records the copper layers each net's zones cover
(`zone_layers`, rule areas left out) and the nets with zones, vias or pads
(`copper_nets`). From these `BoardSnapshot.ground_nets` resolves the
ground nets as kistitch does, with no further request: the nets given,
else those with zones on at least two inner layers, else GND if it has
copper, else none.

The converters `_ring`, `_polygon`, `arc_points`, `circle_points`,
`_pad_shapes`, `_clearance_override`, the Edge.Cuts outline helpers and the
canonical layer naming are copied and trimmed from kistitch's snapshot.py
(MIT, Copyright (c) 2025 Hat Labs Oy).
"""

import enum
import math
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import NamedTuple

import kipy.board
import kipy.board_types as kbt
import shapely
from google.protobuf.message import Message
from kipy import common_types
from kipy.errors import ApiError
from kipy.geometry import PolygonWithHoles, PolyLine, Vector2
from kipy.proto.board.board_commands_pb2 import BoardOriginType
from kipy.proto.board.board_types_pb2 import (
    BoardLayer,
    PadType,
    RuleAreaSettings,
    ViaType,
    ZoneType,
)
from kipy.proto.common import ApiStatusCode
from kipy.proto.common.commands import base_commands_pb2
from kipy.util.board_layer import (
    canonical_name,
    is_copper_layer,
    layer_from_canonical_name,
)
from loguru import logger
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

from kibends.chain import (
    NM_PER_MM,
    Contact,
    Point2,
    SameNetCopper,
    SameNetTrack,
    Segment,
)
from kibends.coords import UserFrame

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


OUTER_COPPER = frozenset({"F.Cu", "B.Cu"})
MIN_GROUND_INNER_LAYERS = 2
FALLBACK_GROUND = "GND"


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


class ViaKind(enum.Enum):
    THROUGH = "through"
    BLIND_BURIED = "blind/buried"
    BLIND = "blind"
    BURIED = "buried"
    MICRO = "micro"
    UNKNOWN = "unknown"


@dataclass(frozen=True, eq=False)
class Via:
    """A via, with copper and a hole on the layers of its span."""

    id: str
    net: str
    position: Point2
    diameter: float
    drill: float
    kind: ViaKind
    span: tuple[str, str]
    """The copper layers its drill starts and ends on."""
    locked: bool = False

    def reaches(self, layer: str) -> bool:
        """Whether `layer` lies within the via's span."""
        start, end = (layer_from_canonical_name(s) for s in self.span)
        return min(start, end) <= layer_from_canonical_name(layer) <= max(start, end)


class Disc(NamedTuple):
    centre: Point2
    radius: float
    """mm."""


class ViaContact(NamedTuple):
    """Same-net copper that touches a via; `track_id` is set for tracks."""

    description: str
    track_id: str | None = None


KICAD_MAX_ERROR = 0.005
"""How far, in mm, KiCad lets a polygon stray from the arc it approximates:
the board's "maximum allowed deviation", at its default. The API does not
expose the board's setting."""

CONTACT_MARGIN = KICAD_MAX_ERROR + 1 / NM_PER_MM
"""How far, in mm, a vertex that keeps a pad or via (`Contact`) stays inside
its outline: the pad shape KiCad hands out approximates its rounded edges
within KICAD_MAX_ERROR, and the commit rounds the vertex to the nanometre."""


@dataclass(frozen=True)
class ViaZone:
    """A copper zone of a via's net that may connect to the via: on a layer
    the via spans, its region reaches the via's ring."""

    name: str
    regions: Mapping[str, BaseGeometry]
    """By layer, for the layers on which it reaches the via's ring: where the
    next refill may fill it, before it keeps clear of other nets' copper
    (`_zone_region`)."""
    min_width: float
    """The zone's minimum fill width, mm."""
    clearance: float
    """The zone's own clearance to copper of other nets, mm."""

    @property
    def layers(self) -> tuple[str, ...]:
        return tuple(self.regions)

    @property
    def description(self) -> str:
        return f"zone {self.name} on {' and '.join(self.layers)}"

    @property
    def margin(self) -> float:
        """How far inside the region a via's centre stays for the next refill
        to connect it: half the zone's minimum width, so that the fill there
        is as wide as the refill keeps, plus KiCad's arc error, which the
        refill's rounded edges may take back.

        Not its whole ring: where the zone's clearance to another net's
        copper reaches the ring, the fill cuts through it, and KiCad
        connects a via whose ring the fill only partly covers."""
        return self.min_width / 2 + KICAD_MAX_ERROR


class ZoneUnion(NamedTuple):
    """The zones of a via's net on one layer, taken together: the refill
    connects the via through any of them."""

    names: tuple[str, ...]
    centres: BaseGeometry
    """Where the via's centre may lie for the next refill to connect it:
    inside the region of one of them by that zone's margin
    (`ViaZone.margin`), or inside the union of their regions by the largest
    of their margins, as on the seam of two zones that meet edge to edge,
    where each zone's fill reaches the seam."""


def zone_unions(zones: Iterable[ViaZone]) -> dict[str, ZoneUnion]:
    """By layer, `zones` there taken together."""
    on: dict[str, list[ViaZone]] = defaultdict(list)
    for zone in zones:
        for layer in zone.layers:
            on[layer].append(zone)
    return {
        layer: ZoneUnion(
            tuple(z.name for z in found),
            _centres(tuple((z.regions[layer], z.margin) for z in found)),
        )
        for layer, found in on.items()
    }


ZONE_SHAPES_CACHED = 256
"""How many shrunk regions and unions `zone_unions` keeps. The vias of one
plane share its region objects (`_via_zones`), so a run shrinks each region
once however many vias it reaches."""


@lru_cache(maxsize=ZONE_SHAPES_CACHED)
def _centres(zones: tuple[tuple[BaseGeometry, float], ...]) -> BaseGeometry:
    """`ZoneUnion.centres` of zones given as (region, margin) pairs."""
    shrunk = [_shrunk(region, margin) for region, margin in zones]
    if len(zones) == 1:
        return shrunk[0]
    whole = unary_union([region for region, _ in zones])
    return unary_union([*shrunk, whole.buffer(-max(m for _, m in zones))])


@lru_cache(maxsize=ZONE_SHAPES_CACHED)
def _shrunk(region: BaseGeometry, margin: float) -> BaseGeometry:
    return region.buffer(-margin)


class GroundSource(enum.Enum):
    """Where a run's ground nets come from (`BoardSnapshot.ground_nets`)."""

    GIVEN = "given"
    INNER_ZONES = "inner zones"
    """Nets with copper zones on at least two inner layers."""
    FALLBACK = "fallback"
    """No net has those; the net named GND, which has copper."""
    NONE = "none"


class GroundNets(NamedTuple):
    nets: frozenset[str]
    source: GroundSource


@dataclass(frozen=True)
class Selection:
    segments: list[Segment]
    """The selected straight tracks."""
    vias: tuple[Via, ...]


@dataclass(frozen=True, eq=False)
class Graphic:
    """Copper drawn as a shape or text; `net` is empty for no net."""

    description: str
    net: str
    geometry: BaseGeometry


@dataclass(frozen=True, eq=False)
class LayerSnapshot:
    """The copper of one layer, without the tracks being replaced.

    Pads are listed whatever their layers, since their holes cross every
    layer; vias only on the layers they span. Copper zones are not read:
    they reflow after an edit.
    """

    layer: str
    pads: tuple[Pad, ...]
    tracks: tuple[Track, ...]
    vias: tuple[Via, ...]
    graphics: tuple[Graphic, ...]
    track_keepouts: tuple[BaseGeometry, ...]
    """Rule areas on the layer that keep out tracks."""
    via_keepouts: tuple[BaseGeometry, ...]
    """Rule areas on the layer that keep out vias."""
    edge_paths: tuple[tuple[Point2, ...], ...]
    clearances: Mapping[str, float]
    """Netclass clearance per net name, in mm."""
    default_clearance: float
    diff_pair_gaps: Mapping[str, float]
    """Netclass differential pair gap per net name, where set, in mm."""

    def clearance(self, net: str) -> float:
        """The net's netclass clearance; the Default one for no net."""
        return self.clearances.get(net, self.default_clearance)

    def diff_pair_gap(self, net: str) -> float | None:
        """The net's netclass differential pair gap, if its netclass sets one."""
        return self.diff_pair_gaps.get(net)

    def same_net_copper(self, net: str) -> list[SameNetCopper]:
        """Pad and via outlines, as contacts, copper shape outlines and the
        tracks of `net` on the layer.

        This is what `extract_chain` pins the chain to, or, for the pads and
        vias, keeps one vertex of the chain inside, unless a vertex lies on
        the pad's position or the via's centre, which pins it there.
        """
        return [
            *(
                Contact(p.shape, CONTACT_MARGIN, p.position)
                for p in self.pads
                if p.net == net and p.shape is not None
            ),
            *(
                Contact(
                    Point(v.position).buffer(v.diameter / 2),
                    CONTACT_MARGIN,
                    v.position,
                )
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
    track_keepouts: Mapping[str, tuple[BaseGeometry, ...]]
    via_keepouts: Mapping[str, tuple[BaseGeometry, ...]]
    edge_paths: tuple[tuple[Point2, ...], ...]
    clearances: Mapping[str, float]
    """Netclass clearance of every net on the board, in mm."""
    default_clearance: float
    diff_pair_gaps: Mapping[str, float]
    """Netclass differential pair gap of the nets on the board whose
    netclass sets one, in mm."""
    via_contacts: Mapping[str, tuple[ViaContact, ...]]
    """By via id, for the vias asked for: the same-net copper touching each,
    or the copper it carries, on any copper layer, the via itself and
    copper zones left out."""
    via_zones: Mapping[str, tuple[ViaZone, ...]]
    """By via id, for the vias asked for: the copper zones of its net that
    may connect to it, with the regions it stays inside; empty when none
    does."""
    zone_layers: Mapping[str, frozenset[str]]
    """By net, the copper layers its copper zones lie on, rule areas left
    out."""
    copper_nets: frozenset[str]
    """The nets with a copper zone, a via or a pad on the board."""

    @classmethod
    def read(
        cls,
        board: kipy.board.Board,
        layers: Iterable[str],
        selected_vias: Sequence[Via] = (),
        *,
        carried: Mapping[str, Sequence[Disc]] | None = None,
        every_copper_layer: bool = False,
    ) -> BoardSnapshot:
        """Read the board once for the copper layers `layers`, or for every
        copper layer the board enables, and the copper touching
        `selected_vias` or the copper `carried` gives, by via id, as moving
        with each."""
        enabled = set(board.get_enabled_layers()) if every_copper_layer else None
        wanted = sorted(
            {*layers, *(canonical_name(L) for L in enabled or () if is_copper_layer(L))}
        )
        footprints = board.get_footprints()
        all_tracks = board.get_tracks()
        kipy_pads = board.get_pads()
        vias = tuple(via_from_kipy(v) for v in board.get_vias())
        shapes = [
            *board.get_shapes(),
            *(s for f in footprints for s in f.definition.shapes),
        ]
        texts = _texts(board, footprints)
        zones = board.get_zones()
        zone_layers = _zone_layers(zones)
        default_clearance = _default_clearance(board)
        shapeless = _read_pads(kipy_pads, footprints)
        pads_by_layer: dict[str, tuple[Pad, ...]] = {}
        pad_shapes: dict[str, dict[str, BaseGeometry]] = {}
        for layer in wanted:
            on_layer = _pad_shapes(board, kipy_pads, layer_from_canonical_name(layer))
            pad_shapes[layer] = on_layer
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
        clearances, diff_pair_gaps = _read_netclasses(
            board, sorted(nets), default_clearance
        )
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
            track_keepouts={
                layer: _read_keepouts(zones, layer, lambda r: r.keepout_tracks)
                for layer in wanted
            },
            via_keepouts={
                layer: _read_keepouts(zones, layer, lambda r: r.keepout_vias)
                for layer in wanted
            },
            edge_paths=tuple(tuple(path) for path in read_edge_paths(shapes)),
            clearances=clearances,
            default_clearance=default_clearance,
            diff_pair_gaps=diff_pair_gaps,
            via_contacts=_via_contacts(
                board,
                {
                    v: (
                        Disc(v.position, v.diameter / 2),
                        *(carried or {}).get(v.id, ()),
                    )
                    for v in selected_vias
                },
                _BoardCopper(
                    all_tracks, kipy_pads, shapeless, pad_shapes, vias, shapes
                ),
                enabled,
            ),
            via_zones=_via_zones(zones, selected_vias),
            zone_layers=zone_layers,
            copper_nets=frozenset(
                {*zone_layers, *(v.net for v in vias), *(p.net for p in shapeless)}
            )
            - {""},
        )

    def ground_nets(self, given: Collection[str] = ()) -> GroundNets:
        """The nets `given`, else those with copper zones on at least two
        inner layers, a power plane's net included, else GND if it has
        copper, else none; from what the read holds, without a request.
        As kistitch's `returnvia` resolves them."""
        if given:
            return GroundNets(frozenset(given), GroundSource.GIVEN)
        inner = frozenset(
            net
            for net, layers in self.zone_layers.items()
            if len(layers - OUTER_COPPER) >= MIN_GROUND_INNER_LAYERS
        )
        if inner:
            return GroundNets(inner, GroundSource.INNER_ZONES)
        if FALLBACK_GROUND in self.copper_nets:
            return GroundNets(frozenset({FALLBACK_GROUND}), GroundSource.FALLBACK)
        return GroundNets(frozenset(), GroundSource.NONE)

    def slice(self, layer: str, replaced_ids: Collection[str]) -> LayerSnapshot:
        """`layer` and everything crossing it, less the tracks being replaced."""
        if layer not in self.layers:
            raise ValueError(f"{layer} was not read from the board")
        snap = LayerSnapshot(
            layer=layer,
            pads=self.pads[layer],
            tracks=tuple(t for t in self.tracks[layer] if t.id not in replaced_ids),
            vias=tuple(v for v in self.vias if v.reaches(layer)),
            graphics=self.graphics[layer],
            track_keepouts=self.track_keepouts[layer],
            via_keepouts=self.via_keepouts[layer],
            edge_paths=self.edge_paths,
            clearances=self.clearances,
            default_clearance=self.default_clearance,
            diff_pair_gaps=self.diff_pair_gaps,
        )
        logger.debug(
            f"Snapshot of {layer}: {len(snap.pads)} pads, {len(snap.tracks)} tracks, "
            f"{len(snap.vias)} vias, {len(snap.graphics)} copper graphics, "
            f"{len(snap.track_keepouts)} track keepouts, "
            f"{len(snap.via_keepouts)} via keepouts"
        )
        return snap


def read_selection(board: kipy.board.Board) -> Selection:
    """The selected straight tracks and vias; arcs are refused, anything else
    ignored."""
    items = board.get_selection()
    arcs = [i for i in items if isinstance(i, kbt.ArcTrack)]
    if arcs:
        ids = ", ".join(a.id.value for a in arcs)
        raise SelectionError(
            f"selection contains arc tracks, which kibends does not modify: {ids}"
        )
    tracks = [i for i in items if isinstance(i, kbt.Track)]
    vias = [i for i in items if isinstance(i, kbt.Via)]
    if not tracks:
        raise SelectionError("selection has no tracks")
    return Selection(
        [segment_from_track(t) for t in tracks], tuple(map(via_from_kipy, vias))
    )


def read_user_frame(board: kipy.board.Board) -> UserFrame:
    """The frame kibends prints coordinates in, at the board's grid origin."""
    return UserFrame(_mm(board.get_origin(BoardOriginType.BOT_GRID)))


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


_VIA_KINDS = {
    ViaType.VT_THROUGH: ViaKind.THROUGH,
    ViaType.VT_BLIND_BURIED: ViaKind.BLIND_BURIED,
    ViaType.VT_BLIND: ViaKind.BLIND,
    ViaType.VT_BURIED: ViaKind.BURIED,
    ViaType.VT_MICRO: ViaKind.MICRO,
}


def via_from_kipy(v: kbt.Via) -> Via:
    drill = v.padstack.drill
    span = (drill.start_layer, drill.end_layer)
    if BoardLayer.BL_UNKNOWN in span:
        # As kipy's Via.type setter does; an unknown layer would otherwise
        # take the via off every layer.
        span = (BoardLayer.BL_F_Cu, BoardLayer.BL_B_Cu)
    return Via(
        id=v.id.value,
        net=v.net.name,
        position=_mm(v.position),
        diameter=v.diameter / NM_PER_MM,
        drill=v.drill_diameter / NM_PER_MM,
        kind=_VIA_KINDS.get(v.type, ViaKind.UNKNOWN),
        span=(canonical_name(span[0]), canonical_name(span[1])),
        locked=v.locked,
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
                "only KiCad's DRC covers it"
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
            "only KiCad's DRC covers it"
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


def _read_keepouts(
    zones: Iterable[kbt.Zone],
    layer: str,
    keeps_out: Callable[[RuleAreaSettings], bool],
) -> tuple[BaseGeometry, ...]:
    """The outlines of the rule areas on `layer` whose settings `keeps_out`."""
    return tuple(
        _polygon(PolygonWithHoles(p))
        for z in zones
        if z.type == ZoneType.ZT_RULE_AREA
        and keeps_out(z.proto.rule_area_settings)
        and layer in {canonical_name(L) for L in z.layers}
        for p in z.proto.outline.polygons
    )


@dataclass(frozen=True)
class _BoardCopper:
    """What one board read already holds, for finding the copper at a via."""

    tracks: Sequence[kbt.Track | kbt.ArcTrack]
    kipy_pads: Sequence[kbt.Pad]
    pads: Sequence[Pad]
    pad_shapes: Mapping[str, Mapping[str, BaseGeometry]]
    """Pad copper by layer read, then by pad id."""
    vias: Sequence[Via]
    shapes: Sequence[kbt.BoardShape]


class _Copper(NamedTuple):
    net: str
    geometry: BaseGeometry
    reach: float
    """How far the copper extends beyond `geometry`: a track's half width."""
    contact: ViaContact


def _via_contacts(
    board: kipy.board.Board,
    selected: Mapping[Via, Sequence[Disc]],
    copper: _BoardCopper,
    enabled: set[BoardLayer.ValueType] | None,
) -> dict[str, tuple[ViaContact, ...]]:
    """The same-net copper touching any of the discs of each selected via on
    any copper layer.

    Pad copper on layers the snapshot did not read costs one request per
    such layer, asked only for pads of the vias' nets, and one for the
    board's `enabled` layers unless the caller has them.
    """
    if not selected:
        return {}
    nets = {v.net for v in selected}
    pad_shapes = dict(copper.pad_shapes)
    of_nets = [p for p in copper.kipy_pads if p.net.name in nets]
    unread = {L for p in of_nets for L in p.padstack.layers if is_copper_layer(L)} - {
        layer_from_canonical_name(layer) for layer in pad_shapes
    }
    if unread:
        # A through-hole pad may list every copper layer KiCad knows of.
        unread &= set(board.get_enabled_layers()) if enabled is None else enabled
    for layer_id in sorted(unread):
        pad_shapes[canonical_name(layer_id)] = _pad_shapes(board, of_nets, layer_id)
    names = {p.id: p.name for p in copper.pads}
    found = [
        _Copper(
            t.net.name,
            track.centreline,
            track.width / 2,
            ViaContact(f"track {track.id} on {canonical_name(t.layer)}", track.id),
        )
        for t in copper.tracks
        if t.net.name in nets
        for track in (_read_track(t),)
    ]
    found += [
        _Copper(p.net.name, shape, 0.0, ViaContact(f"pad {names[p.id.value]}"))
        for p in of_nets
        for on_layer in pad_shapes.values()
        if (shape := on_layer.get(p.id.value)) is not None
    ]
    found += [
        _Copper(
            s.net.name,
            geometry,
            0.0,
            ViaContact(f"copper shape {s.id.value} on {canonical_name(s.layer)}"),
        )
        for s in copper.shapes
        if s.net.name in nets
        and is_copper_layer(s.layer)
        and (geometry := _shape_geometry(s)) is not None
    ]

    def touching(via: Via, discs: Sequence[Disc]) -> set[ViaContact]:
        return {
            c.contact
            for c in found
            if c.net == via.net
            and any(
                c.geometry.distance(Point(d.centre)) <= d.radius + c.reach
                for d in discs
            )
        } | {
            ViaContact(f"via {other.id}")
            for other in copper.vias
            if other.net == via.net
            and other.id != via.id
            and any(
                math.dist(other.position, d.centre) <= d.radius + other.diameter / 2
                for d in discs
            )
        }

    return {
        via.id: tuple(sorted(touching(via, discs))) for via, discs in selected.items()
    }


def _via_zones(
    zones: Sequence[kbt.Zone], vias: Sequence[Via]
) -> dict[str, tuple[ViaZone, ...]]:
    """By via id, the copper zones of each via's net that may connect to
    it, each with its region on the layers where it may: those the via
    spans on which the region reaches its ring (`_zone_region`). Whether
    its centre lies deep enough inside to move is `vias.movable_vias`'s to
    judge.

    Fills are never read: a fill holds where other copper stood at the last
    refill. Teardrops are not copper zones: KiCad reapplies them on the next
    refill. Each region is built once, when a via its outline reaches first
    asks for it.
    """
    outlines: dict[int, BaseGeometry] = {}
    regions: dict[tuple[int, str], BaseGeometry] = {}

    def outline(k: int) -> BaseGeometry:
        if k not in outlines:
            outlines[k] = _outline(zones[k])
        return outlines[k]

    def region(k: int, layer: str) -> BaseGeometry:
        if (k, layer) not in regions:
            regions[k, layer] = _zone_region(zones, zones[k], layer)
        return regions[k, layer]

    found: dict[str, tuple[ViaZone, ...]] = {}
    for via in vias:
        centre = Point(via.position)
        ring = via.diameter / 2
        connected: list[ViaZone] = []
        for k, z in enumerate(zones):
            if (
                z.type != ZoneType.ZT_COPPER
                or z.net is None
                or z.net.name != via.net
                or outline(k).distance(centre) > ring
            ):
                continue
            reached = {
                layer: region(k, layer)
                for layer in (canonical_name(L) for L in sorted(z.layers))
                if via.reaches(layer) and region(k, layer).distance(centre) <= ring
            }
            if reached:
                connected.append(
                    ViaZone(
                        z.name or z.id.value,
                        reached,
                        (z.min_thickness or 0) / NM_PER_MM,
                        (z.clearance or 0) / NM_PER_MM,
                    )
                )
        found[via.id] = tuple(connected)
    return found


def _zone_layers(zones: Iterable[kbt.Zone]) -> dict[str, frozenset[str]]:
    """By net, the copper layers its copper zones lie on."""
    found: dict[str, set[str]] = defaultdict(set)
    for z in zones:
        if z.type == ZoneType.ZT_COPPER and z.net is not None and z.net.name:
            found[z.net.name].update(
                canonical_name(L) for L in z.layers if is_copper_layer(L)
            )
    return {net: frozenset(layers) for net, layers in found.items()}


def _zone_region(zones: Sequence[kbt.Zone], zone: kbt.Zone, layer: str) -> BaseGeometry:
    """Where the next refill may fill `zone` on `layer`, before it keeps
    clear of other nets' copper: its outline, with its own cut-outs, less
    the outline of each copper zone of another net that may fill first,
    having the higher or the same priority (KiCad breaks a tie by uuid),
    grown by the larger of the two zones' clearances, and the outline of
    each rule area keeping out copper pour.

    Outlines alone, which the run does not move."""
    net = zone.net.name if zone.net else ""
    own_clearance = (zone.clearance or 0) / NM_PER_MM
    areas: list[BaseGeometry] = []
    for z in zones:
        if layer not in {canonical_name(L) for L in z.layers}:
            continue
        outline = [_polygon(PolygonWithHoles(p)) for p in z.proto.outline.polygons]
        if z.type == ZoneType.ZT_RULE_AREA:
            if z.proto.rule_area_settings.keepout_copper:
                areas.extend(outline)
        elif (
            z.type == ZoneType.ZT_COPPER
            and (z.net.name if z.net else "") != net
            and z.priority >= zone.priority
        ):
            gap = max(own_clearance, (z.clearance or 0) / NM_PER_MM)
            areas.extend(o.buffer(gap) for o in outline)
    return _outline(zone).difference(unary_union(areas))


def _outline(zone: kbt.Zone) -> BaseGeometry:
    """`zone`'s outline, with its own cut-outs."""
    return unary_union(
        [_polygon(PolygonWithHoles(p)) for p in zone.proto.outline.polygons]
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


def _read_netclasses(
    board: kipy.board.Board, nets: Sequence[str], default: float
) -> tuple[dict[str, float], dict[str, float]]:
    """Clearance and, where set, differential pair gap of each net, mm."""
    if not nets:
        return {}, {}
    classes = board.get_netclass_for_nets([kbt.Net(name=n) for n in nets])
    clearances = {
        name: default if nc.clearance is None else nc.clearance / NM_PER_MM
        for name, nc in classes.items()
    }
    gaps = {
        name: nc.diff_pair_gap / NM_PER_MM
        for name, nc in classes.items()
        if nc.diff_pair_gap is not None
    }
    return clearances, gaps


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
