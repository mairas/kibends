"""Obstacles on the chain's layer, or on every layer a moving via spans,
read from the board's snapshot; the fields, the clearance rules and the
builders are Rust's.

Each obstacle is raw geometry (a copper polygon, a track centreline, a via or
hole centre) plus an inflation: the distance from that geometry at which the
chain's centreline, or the via's centre, reaches the required clearance.
This module reads each layer's snapshot into plain records once
(`core_layer`, with `plain_shapes`), and the Rust builders pick the items
of each field and inflate them by the clearance rules (`kibends._core`;
their design is in `kibends_core::field::build` and `kibends_core::rules`).
Each built field comes back with its items named from the snapshot's
records (`Obstacle`). Python keeps every shapely construction: buffers,
unions, the board outline, the zones' regions (`containment`).

A chain's static field is one per (layer, net, width), from other nets'
copper, holes, keepouts and the board edge (`build_obstacles`); no copper
of the chain's net is an obstacle. A moving via's field spans every layer
it spans (`split_via_obstacles`). What only limits a via without
repelling it is held apart, as the board edge is: the outside of where its
zones connect it (`containment`) and the fabrication rules within its net
(`own_net_stops`).

With ground nets, the vias standing still that a moving via pairs with,
ground with regular or of two ground nets (`vias.pairs_with`), leave its
field and are built apart, each as the field would count it
(`split_via_obstacles`): `vias.ViaInput.copper` takes them back in, so
the distances it gives are the unsplit field's, and the optimizer's
pairwise term (`via_pairs`) gives their energy.

The other selected chains are obstacles of their own kind (`ChainField`),
rebuilt at their current shapes, and a `CompositeField` joins fields; both
are Rust's too, as are the energy's searches over them
(`kibends_core::seen`, `kibends_core::neighbours`). Every field the
optimizer takes is a Rust field (`core_field`).
"""

import enum
import math
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from weakref import WeakKeyDictionary

import numpy as np
import shapely
from loguru import logger
from numpy.typing import NDArray
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from kibends import _core
from kibends._core import testing
from kibends.chain import Point2
from kibends.energy import (
    BoolArray,
    DistanceField,
    FloatArray,
    IntArray,
    ObstacleField,
)
from kibends.snapshot import (
    LayerSnapshot,
    PadKind,
    Via,
    ViaZone,
    edge_paths_closed,
    layer_ordinal,
    outline_from_edge_paths,
    zone_unions,
)

ObjectArray = NDArray[np.object_]

# The API cannot report the board's copper-to-edge and copper-to-hole rules.
# 0.3 mm to the edge meets common fabs' minimum for routed outlines, and
# 0.25 mm to holes and between holes are the hole clearance and hole-to-hole
# minimum KiCad's own project templates set. KiCad's DRC, which the user runs,
# enforces the board's real rules.
EDGE_CLEARANCE = 0.3
HOLE_CLEARANCE = 0.25
HOLE_TO_HOLE = 0.25

# Closer than this to the nearest feature, the direction to it is noise; the
# boundary's normal is used instead. Far below KiCad's 1 nm resolution.
DEGENERATE_DISTANCE = 1e-9
# Copper this much farther from a chain than both half widths still touches
# it: 1 nm, KiCad's coordinate resolution.
TOUCH_TOLERANCE = 1e-6
# Other chains' edges are split to at most this length, so that a point near an
# edge is also near its midpoint, where the k-d tree finds it. Chains resampled
# at the final spacing, `OptimizerSettings.spacing` of at most 0.1 mm, are
# never split; at a coarse level, edges longer than this are.
MAX_EDGE_LENGTH = 0.5

POLYGON_TYPE = 3
"""shapely's type id of a Polygon."""


def plain_shapes(geometries: Iterable[BaseGeometry]) -> _core.Shapes:
    """shapely geometries as the plain coordinates the Rust field takes:
    each geometry's parts, a polygon's rings exterior first, a line's or a
    point's coordinates as one ring."""
    shapes: ObjectArray = np.empty(0, dtype=object)
    listed = list(geometries)
    if listed:
        shapes = np.empty(len(listed), dtype=object)
        shapes[:] = listed
    parts, part_of = shapely.get_parts(shapes, return_index=True)
    # Untyped **kwargs in types-shapely, as in Obstacles.__init__.
    types: NDArray[np.int32] = shapely.get_type_id(parts)  # pyright: ignore[reportUnknownMemberType]
    polygon = types == POLYGON_TYPE
    polygons = np.flatnonzero(polygon)
    rings, ring_of = shapely.get_rings(parts[polygons], return_index=True)
    ring_part = np.concatenate([polygons[ring_of], np.flatnonzero(~polygon)])
    order = np.argsort(ring_part, kind="stable")
    every_ring = np.concatenate([rings, parts[~polygon]])[order]
    coords, coord_ring = shapely.get_coordinates(every_ring, return_index=True)
    return _core.Shapes(
        np.asarray(coords, dtype=float).reshape(-1, 2),
        _ends(coord_ring, len(every_ring)),
        _ends(ring_part[order], len(parts)),
        polygon,
        _ends(part_of, len(shapes)),
        shapely.get_type_id(shapes).astype(np.int32),  # pyright: ignore[reportUnknownMemberType]
    )


def _ends(owner: IntArray, count: int) -> IntArray:
    """Where each of `count` runs ends, of items each of the run `owner`
    gives, ascending."""
    return np.cumsum(np.bincount(owner, minlength=count)).astype(np.intp)


@dataclass(frozen=True)
class Obstacle:
    description: str
    geometry: BaseGeometry
    """Polygonal copper, or a centreline or centre point to be inflated."""
    inflation: float
    """Distance from `geometry` at which the chain's centreline is at clearance."""
    net: str | None = None
    """The net of copper, None for what is not copper."""
    extent: float = 0.0
    """How far copper reaches beyond `geometry`, mm: half a track's width or
    a via's diameter."""
    layer: str | None = None
    """The layer of copper, None for what is not copper of one layer."""


class Obstacles:
    """Signed distance to the nearest inflated obstacle, and edge checks:
    `items`, and the outside of `board_outline` inflated by
    `edge_inflation`, as a Rust field (`_core.Obstacles`).

    Positive in free space, zero exactly at the required clearance, negative
    inside an inflated obstacle or outside the board. Where obstacles overlap,
    the deepest one counts. `smooth_distance` is the smooth minimum over the
    obstacles: the items of one geometry, every connected run of
    centrelines of one inflation that meet end to end, and the board edge.
    """

    def __init__(
        self,
        items: Sequence[Obstacle],
        board_outline: BaseGeometry | None = None,
        edge_inflation: float = 0.0,
    ) -> None:
        self.items = tuple(items)
        self.core = _core.Obstacles(
            plain_shapes(o.geometry for o in self.items),
            np.array([o.inflation for o in self.items], dtype=float),
            [o.net for o in self.items],
            np.array([o.extent for o in self.items], dtype=float),
            [o.layer for o in self.items],
            None if board_outline is None else plain_shapes([board_outline]),
            edge_inflation,
        )

    @classmethod
    def of(cls, core: _core.Obstacles, items: Sequence[Obstacle]) -> Obstacles:
        """The field `core` answers for, of `items`."""
        field = cls.__new__(cls)
        field.items = tuple(items)
        field.core = core
        return field

    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """Distance (n,) and its unit gradient (n, 2) at (n, 2) points in mm.

        With no obstacles and no board outline the distance is +inf and the
        gradient zero. Points farther than `reach` from every inflated
        obstacle and from the inflated board edge get +inf and a zero
        gradient; points outside the board are always reported.
        """
        return self.core.signed_distance(_points(points), reach)

    def smooth_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """`signed_distance` as the energy takes it: the smooth minimum over
        the obstacles, from the field's edges (`static_edges`)."""
        return self.core.smooth_distance(_points(points), reach)

    @property
    def static_edges(self) -> StaticEdges:
        """The field as edges, its items in order, then the board edge."""
        return self.core.static_edges

    def segments_clear(
        self, starts: FloatArray, ends: FloatArray, tolerance: float = 0.0
    ) -> BoolArray:
        """Whether each segment from starts[i] to ends[i] stays out of every
        inflated obstacle and inside the inflated board edge.

        Touching an inflated boundary, which is exactly at clearance, is clear,
        and so is entering it by at most `tolerance` mm. The tolerance does not
        extend the board outline itself, which only matters with an edge
        clearance below it.
        """
        return self.core.segments_clear(_points(starts), _points(ends), tolerance)


def _points(points: FloatArray) -> FloatArray:
    """`points` as an (n, 2) array of floats."""
    return np.asarray(points, dtype=float).reshape(-1, 2)


class CompositeField:
    """The union of several fields: the nearest of them answers each point,
    and the smooth distance is over all their obstacles, as a Rust field
    (`_core.CompositeField`).

    A point beyond `reach` of every field is +inf with a zero gradient. A
    segment is clear only if it is clear in every field.
    """

    def __init__(self, fields: Sequence[ObstacleField]) -> None:
        self.fields = tuple(fields)
        self.core = _core.CompositeField([core_field(f) for f in self.fields])

    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """Distance (n,) and its unit gradient (n, 2) at (n, 2) points in mm."""
        return self.core.signed_distance(_points(points), reach)

    def smooth_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """The smooth minimum of the fields' smooth distances
        (`energy.combine_fields`). It is the one over all their obstacles
        but for those SMOOTH_REACH leaves out within each field: up to
        τ·e⁻¹⁰ per obstacle apart from a single search over every field's
        edges."""
        return self.core.smooth_distance(_points(points), reach)

    def segments_clear(
        self, starts: FloatArray, ends: FloatArray, tolerance: float = 0.0
    ) -> BoolArray:
        """Whether each segment from starts[i] to ends[i] is clear in every
        field, each allowing `tolerance` inside its clearance."""
        return self.core.segments_clear(_points(starts), _points(ends), tolerance)


CORE_FIELDS = (
    _core.Obstacles,
    _core.ChainField,
    _core.CompositeField,
    testing.FreeSpace,
    testing.Disc,
    testing.Box,
    testing.Walls,
    testing.MisleadingWalls,
    testing.Union,
    testing.ReachIgnored,
    testing.NanField,
)
"""The kinds of `_core.Field`."""


def core_field(field: DistanceField) -> _core.Field:
    """The Rust field that answers for `field`; raises TypeError for a
    field of another kind."""
    if isinstance(field, Obstacles | ChainField | CompositeField):
        return field.core
    if isinstance(field, CORE_FIELDS):
        return field
    raise TypeError(f"{type(field).__name__} is not a field of kibends._core")


LINE = _core.LINE
"""A `StaticEdges` item measured to its own geometry: a centreline or a
centre."""
AREA = _core.AREA
"""A `StaticEdges` item measured to the boundary of its area, negative
inside it: copper, a keepout."""
OUTSIDE = _core.OUTSIDE
"""A `StaticEdges` item measured to the boundary of its area, negative
outside it: the board edge, the region a via's zones connect it in."""

INDEXED_EDGES = _core.INDEXED_EDGES
"""A `StaticEdges` item with more edges than this, such as copper text, a
logo or a board outline, finds its edges near a point with a tree over
them rather than measuring every one."""

StaticEdges = _core.StaticEdges


@dataclass(frozen=True, eq=False)
class ChainCentreline:
    """A selected chain's identity and polyline.

    It is a chain's own input to the optimizer, and, at the chain's current
    shape, an obstacle to the other selected chains on its layer.
    """

    description: str
    net: str
    width: float
    clearance: float
    """Its net's netclass clearance in mm."""
    points: FloatArray
    """Its polyline, (n, 2) in mm."""


class ChainField:
    """Signed distance to other chains' inflated centrelines, and edge checks,
    as a Rust field (`_core.ChainField`; its design is in
    `kibends_core::field::chain`).

    The same answers as `Obstacles` with each centreline as an obstacle,
    except that zero-length edges and zero-length query segments count like
    any other. Its smooth distance takes the edges of one obstacle, a chain
    or a via, as one: the nearest of them, then the smooth minimum over the
    obstacles.
    """

    def __init__(
        self,
        starts: FloatArray,
        ends: FloatArray,
        inflations: FloatArray,
        obstacles: IntArray,
    ) -> None:
        """Edges from starts[i] to ends[i], (m, 2) in mm, each inflated by
        inflations[i] and part of the obstacle obstacles[i]."""
        self.core = _core.ChainField(
            _points(starts),
            _points(ends),
            np.asarray(inflations, dtype=float),
            np.asarray(obstacles, dtype=np.intp),
        )

    @classmethod
    def of(cls, core: _core.ChainField) -> ChainField:
        """The field `core` answers for."""
        field = cls.__new__(cls)
        field.core = core
        return field

    @property
    def static_edges(self) -> StaticEdges:
        """The field as `StaticEdges`, each edge an item of its own, for a
        field that stands still. Its distances differ from
        `smooth_distance`'s by round-off, since a static search measures by
        GEOS's point-to-segment formula."""
        return self.core.static_edges

    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """Distance (n,) and its unit gradient (n, 2) at (n, 2) points in mm.

        With no edges the distance is +inf and the gradient zero everywhere.
        Points farther than `reach` from every inflated edge get +inf and a
        zero gradient.
        """
        return self.core.signed_distance(_points(points), reach)

    def smooth_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        """`signed_distance` as the energy takes it: the smooth minimum over
        the obstacles of each one's nearest edge."""
        return self.core.smooth_distance(_points(points), reach)

    def segments_clear(
        self, starts: FloatArray, ends: FloatArray, tolerance: float = 0.0
    ) -> BoolArray:
        """Whether each segment from starts[i] to ends[i] stays out of every
        inflated edge.

        Touching an inflated boundary, which is exactly at clearance, is clear,
        and so is entering it by at most `tolerance` mm.
        """
        return self.core.segments_clear(_points(starts), _points(ends), tolerance)


def chain_inflation(
    ignored: frozenset[str], width: float, clearance: float, other: ChainCentreline
) -> float | None:
    """The distance a track of `width` and netclass `clearance` (mm) keeps
    its centreline from `other`'s, mm: both half widths and the larger of
    both clearances, as from an unselected track; None for a chain of a net
    in `ignored`, the track's own and the other net of its diff pair, which
    may come close, overlap or cross."""
    if other.net in ignored:
        return None
    return track_inflation(width, clearance, other.width, other.clearance)


track_inflation = _core.track_inflation
"""The distance a track of `width` and `clearance` keeps its centreline
from that of a track of another net of `other_width` and
`other_clearance`, mm: both half widths and the larger of both
clearances."""


def chains_field(
    inflated: Iterable[
        tuple[ChainCentreline | Sequence[ChainCentreline], float | None]
    ],
) -> ChainField:
    """The centrelines as a field, each inflated by its own amount and an
    obstacle of its own, or several as the pieces of one; those with None
    are left out. Each is split into edges of at most MAX_EDGE_LENGTH."""
    polylines: list[FloatArray] = []
    inflations: list[float] = []
    obstacles: list[int] = []
    count = 0
    for o, inflation in inflated:
        if inflation is None:
            continue
        for piece in [o] if isinstance(o, ChainCentreline) else o:
            polylines.append(_points(piece.points))
            inflations.append(inflation)
            obstacles.append(count)
        count += 1
    return ChainField.of(
        _core.chains_field(
            polylines,
            np.array(inflations, dtype=float),
            np.array(obstacles, dtype=np.intp),
        )
    )


def zones_on(zones: Iterable[ViaZone], layers: Iterable[str]) -> tuple[ViaZone, ...]:
    """The zones among `zones` a via stays inside on any of `layers`."""
    wanted = set(layers)
    return tuple(z for z in zones if wanted.intersection(z.layers))


_LAYERS: WeakKeyDictionary[LayerSnapshot, _core.BoardLayer] = WeakKeyDictionary()
_VIAS: WeakKeyDictionary[Via, _core.Via] = WeakKeyDictionary()


def core_layer(snapshot: LayerSnapshot) -> _core.BoardLayer:
    """The snapshot's layer as the Rust builders take it, read once per
    snapshot."""
    layer = _LAYERS.get(snapshot)
    if layer is None:
        layer = _LAYERS[snapshot] = _read_layer(snapshot)
    return layer


def _read_layer(snapshot: LayerSnapshot) -> _core.BoardLayer:
    pads = snapshot.pads
    shaped = [k for k, p in enumerate(pads) if p.shape is not None]
    shape_of = np.full(len(pads), -1, dtype=np.intp)
    shape_of[shaped] = np.arange(len(shaped), dtype=np.intp)
    outline = _board_outline(snapshot.edge_paths)
    return _core.BoardLayer(
        snapshot.layer,
        layer_ordinal(snapshot.layer),
        [p.net for p in pads],
        np.array([p.position for p in pads], dtype=float).reshape(-1, 2),
        np.array([p.drill for p in pads], dtype=float),
        np.array([p.kind is not PadKind.NPTH for p in pads], dtype=np.bool_),
        np.array(
            [
                math.nan if p.clearance_override is None else p.clearance_override
                for p in pads
            ],
            dtype=float,
        ),
        shape_of,
        plain_shapes(s for p in pads if (s := p.shape) is not None),
        [t.net for t in snapshot.tracks],
        np.array([t.width for t in snapshot.tracks], dtype=float),
        plain_shapes(t.centreline for t in snapshot.tracks),
        [core_via(v) for v in snapshot.vias],
        [g.net for g in snapshot.graphics],
        plain_shapes(g.geometry for g in snapshot.graphics),
        plain_shapes(snapshot.track_keepouts),
        plain_shapes(snapshot.via_keepouts),
        None if outline is None else plain_shapes([outline]),
        dict(snapshot.clearances),
        snapshot.default_clearance,
    )


def core_via(via: Via) -> _core.Via:
    """`via` as the Rust rules take it, read once per via."""
    found = _VIAS.get(via)
    if found is None:
        low, high = via.layer_range
        x, y = via.position
        found = _VIAS[via] = _core.Via(
            via.id, via.net, x, y, via.diameter, via.drill, low, high
        )
    return found


def core_zones(zones: Iterable[ViaZone]) -> list[_core.Zone]:
    """`zones` as the Rust rules take them."""
    return [
        _core.Zone(z.clearance, z.margin, [layer_ordinal(layer) for layer in z.layers])
        for z in zones
    ]


def _board_outline(edge_paths: Sequence[Sequence[Point2]]) -> BaseGeometry | None:
    """The board area, or None if Edge.Cuts is not closed."""
    if edge_paths_closed(edge_paths):
        return outline_from_edge_paths(edge_paths)
    return None


class _Source(enum.IntEnum):
    """What a built field's item is made of: Rust's
    `kibends_core::field::build::Source`."""

    PAD = 0
    TRACK = 1
    VIA_RING = 2
    GRAPHIC = 3
    TRACK_KEEPOUT = 4
    VIA_KEEPOUT = 5
    PAD_HOLE = 6
    VIA_HOLE = 7
    PAD_STOP = 8


Built = tuple[
    _core.Obstacles, tuple[IntArray, IntArray, IntArray, IntArray, FloatArray]
]
"""A field the Rust builders built, and its items' sources: each one's
kind, the layer of its record and its index there, the layer of its
copper, -1 for none, and its inflation."""


def _named(built: Built, layers: Sequence[LayerSnapshot]) -> Obstacles:
    """The field of `built` with its items named from the records of
    `layers`."""
    core, (sources, owners, indices, coppers, inflations) = built
    items: list[Obstacle] = []
    kinds: list[int] = sources.tolist()
    records: list[int] = owners.tolist()
    at: list[int] = indices.tolist()
    copper_layers: list[int] = coppers.tolist()
    inflated: list[float] = inflations.tolist()
    for source, owner, index, copper, inflation in zip(
        kinds, records, at, copper_layers, inflated, strict=True
    ):
        s = layers[owner]
        on = layers[copper].layer if copper >= 0 else None
        match _Source(source):
            case _Source.PAD:
                pad = s.pads[index]
                assert pad.shape is not None
                item = Obstacle(
                    f"pad {pad.name}", pad.shape, inflation, pad.net, 0.0, on
                )
            case _Source.TRACK:
                t = s.tracks[index]
                item = Obstacle(
                    f"track {t.id}", t.centreline, inflation, t.net, t.width / 2, on
                )
            case _Source.VIA_RING:
                v = s.vias[index]
                item = Obstacle(
                    f"via {v.id}",
                    Point(v.position),
                    inflation,
                    v.net,
                    v.diameter / 2,
                    on,
                )
            case _Source.GRAPHIC:
                g = s.graphics[index]
                item = Obstacle(g.description, g.geometry, inflation, g.net, 0.0, on)
            case _Source.TRACK_KEEPOUT:
                item = Obstacle("track keepout", s.track_keepouts[index], inflation)
            case _Source.VIA_KEEPOUT:
                item = Obstacle("via keepout", s.via_keepouts[index], inflation)
            case _Source.PAD_HOLE:
                pad = s.pads[index]
                item = Obstacle(
                    f"hole of pad {pad.name}", Point(pad.position), inflation
                )
            case _Source.VIA_HOLE:
                v = s.vias[index]
                item = Obstacle(f"hole of via {v.id}", Point(v.position), inflation)
            case _Source.PAD_STOP:
                pad = s.pads[index]
                assert pad.shape is not None
                item = Obstacle(f"pad {pad.name}", pad.shape, inflation)
        items.append(item)
    return Obstacles.of(core, items)


def _warn_unclosed(layer: _core.BoardLayer) -> None:
    if not layer.has_outline:
        logger.warning(
            "Edge.Cuts has no closed outline; the board edge is not kept clear "
            "while smoothing, only KiCad's DRC covers it"
        )


def build_obstacles(
    snapshot: LayerSnapshot,
    net: str,
    width: float,
    *,
    edge_clearance: float = EDGE_CLEARANCE,
    hole_clearance: float = HOLE_CLEARANCE,
) -> Obstacles:
    """Obstacles for a track of `net` and `width` mm on the snapshot's layer
    (`kibends_core::field::build::chain_obstacles`).

    Copper clearance is the larger of both items' netclass clearances, with a
    pad's own override replacing its netclass. Copper of `net` is never an
    obstacle; its holes are not either, except unplated ones.
    """
    layer = core_layer(snapshot)
    _warn_unclosed(layer)
    built = _core.build_obstacles(layer, net, width, edge_clearance, hole_clearance)
    return _named(built, [snapshot])


def build_pair_obstacles(
    snapshot: LayerSnapshot,
    a: ChainCentreline,
    b: ChainCentreline,
    gap: float,
    *,
    edge_clearance: float = EDGE_CLEARANCE,
    hole_clearance: float = HOLE_CLEARANCE,
) -> Obstacles:
    """Obstacles for the centreline of a coupled section of a differential
    pair, its tracks `a` and `b` `gap` mm apart, on the snapshot's layer
    (`kibends_core::field::build::pair_obstacles`).

    The centreline keeps clear as one track as wide as the pair would.
    Copper of the pair's own nets is an obstacle like any other's, but for
    copper that overlaps the section's track of its own net.
    """
    layer = core_layer(snapshot)
    _warn_unclosed(layer)
    built = _core.build_pair_obstacles(
        layer,
        a.net,
        _points(a.points),
        b.net,
        _points(b.points),
        a.width,
        gap,
        edge_clearance,
        hole_clearance,
    )
    return _named(built, [snapshot])


def split_via_obstacles(
    layers: Sequence[LayerSnapshot],
    via: Via,
    split: Collection[str],
    *,
    moving: Collection[str] = frozenset(),
    edge_clearance: float = EDGE_CLEARANCE,
    hole_clearance: float = HOLE_CLEARANCE,
    hole_to_hole: float = HOLE_TO_HOLE,
    zones: Sequence[ViaZone] = (),
) -> tuple[ObstacleField, Obstacles | None]:
    """Obstacles for `via` at any position, over every layer of `layers` it
    spans, and the vias of other nets whose ids are in `split` apart, as
    obstacles to `via`, each as the field would count it, or None if it
    picks none (`kibends_core::field::build::via_obstacles`).

    Nothing of its own net is among them, nor is the outside of the regions
    of `zones`: those only limit where the via may go (`containment`,
    `own_net_stops`). The via itself and the vias in `moving`, which the
    caller adds at their current positions, are left out.
    """
    field, apart = _core.split_via_obstacles(
        [core_layer(s) for s in layers],
        core_via(via),
        list(split),
        list(moving),
        edge_clearance,
        hole_clearance,
        hole_to_hole,
        core_zones(zones),
    )
    return _named(field, layers), None if apart is None else _named(apart, layers)


def own_net_stops(
    layers: Sequence[LayerSnapshot],
    via: Via,
    *,
    moving: Collection[str] = frozenset(),
    hole_clearance: float = HOLE_CLEARANCE,
    hole_to_hole: float = HOLE_TO_HOLE,
    allow_via_in_pad: bool = False,
) -> ObstacleField | None:
    """What of `via`'s net on the layers of `layers` it spans limits it, as
    obstacles that stop it without repelling it, or None if nothing does
    (`kibends_core::field::build::own_net_stops`): the holes of its net's
    vias, but those in `moving`, and of its net's plated pads keep
    hole-to-hole from its hole, and its ring may touch its net's pads but
    not overlap them, unless `allow_via_in_pad`."""
    items, vias = _core.own_net_stops(
        [core_layer(s) for s in layers],
        core_via(via),
        list(moving),
        hole_clearance,
        hole_to_hole,
        allow_via_in_pad,
    )
    parts: list[ObstacleField] = []
    if items is not None:
        parts.append(_named(items, layers))
    if vias is not None:
        parts.append(ChainField.of(vias))
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else CompositeField(parts)


def containment(zones: Iterable[ViaZone]) -> ObstacleField | None:
    """The outside of where `zones` connect a via, as one field for its
    centre, or None without zones: on each layer where they reach the via's
    ring, the outside of where its centre may lie for the next refill to
    connect it again through one of them (`ZoneUnion.centres`), held as the
    board edge is.

    The regions are built from outlines alone; kibends never reads a fill.
    The clearance the refill keeps from other nets' copper in a zone is the
    via's to keep instead, at the copper's current place
    (`kibends_core::rules::zone_inflation`).
    """
    # A region the same on several layers is one obstacle, not one per layer.
    regions = {u.centres.wkb: u.centres for u in zone_unions(zones).values()}
    parts = [Obstacles((), region) for region in regions.values()]
    return CompositeField(parts) if parts else None


def moving_via_obstacles(
    layer: str,
    net: str,
    width: float,
    clearance: float,
    vias: Iterable[Via],
    clearance_of: Callable[[str], float],
    *,
    hole_clearance: float = HOLE_CLEARANCE,
    zones: Mapping[str, Sequence[ViaZone]] = MappingProxyType({}),
) -> ChainField | None:
    """`vias`, at their positions, as obstacles to a track of `net`, `width`
    and netclass `clearance` (mm) on `layer`, or None if none is one
    (`kibends_core::field::build::moving_via_obstacles`).

    Each counts as `build_obstacles` counts a via on the board, with
    `clearance_of` giving a via's netclass clearance. A via of `net` is no
    obstacle, nor is one not spanning `layer`. A via of another net in one
    of the zones `zones` gives by via id, on `layer`, keeps the zone's
    clearance and margin from the track.
    """
    reaching = [v for v in vias if v.reaches(layer)]
    found = _core.moving_via_obstacles(
        layer_ordinal(layer),
        net,
        width,
        clearance,
        [core_via(v) for v in reaching],
        [clearance_of(v.net) for v in reaching],
        hole_clearance,
        [core_zones(zones.get(v.id, ())) for v in reaching],
    )
    return None if found is None else ChainField.of(found)


def via_via_inflation(
    via: Via,
    other: Via,
    clearance_of: Callable[[str], float],
    *,
    hole_clearance: float = HOLE_CLEARANCE,
    hole_to_hole: float = HOLE_TO_HOLE,
    zones: Sequence[ViaZone] = (),
    other_zones: Sequence[ViaZone] = (),
) -> float | None:
    """The distance `via`'s centre keeps from `other`'s, mm, as
    `split_via_obstacles` counts a via on the board; None if no rule
    applies, as for two vias whose spans do not overlap
    (`kibends_core::rules::via_via_inflation`). `zones` are the zones `via`
    stays inside and `other_zones` those `other` stays inside: given both,
    the distance is the same both ways."""
    if not via.shares_layer(other):
        return None
    return _core.via_via_inflation(
        core_via(via),
        clearance_of(via.net),
        core_via(other),
        clearance_of(other.net),
        hole_clearance,
        hole_to_hole,
        core_zones(zones),
        core_zones(other_zones),
    )


def via_chain_inflation(
    via: Via,
    chain: ChainCentreline,
    clearance_of: Callable[[str], float],
    *,
    hole_clearance: float = HOLE_CLEARANCE,
    zones: Sequence[ViaZone] = (),
) -> float | None:
    """The distance `via`'s centre keeps from `chain`'s centreline, mm, with
    `zones` the zones the via stays inside on the chain's layer; None for a
    chain of its net, which is no obstacle to it
    (`kibends_core::rules::via_chain_inflation`)."""
    if chain.net == via.net:
        return None
    return _core.via_chain_inflation(
        core_via(via),
        clearance_of(via.net),
        chain.net,
        chain.width,
        chain.clearance,
        hole_clearance,
        core_zones(zones),
    )
