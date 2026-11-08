import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from functools import partial
from typing import cast

import kipy.board
import kipy.board_types as kbt
import pytest
from conftest import (
    PAD_HALF_SIZE,
    TEXT_HALF_HEIGHT,
    TEXT_HALF_WIDTH,
    FakeBoard,
    FakeClient,
    kipy_arc,
    kipy_bezier,
    kipy_board_arc,
    kipy_circle,
    kipy_field,
    kipy_filled_zone,
    kipy_footprint,
    kipy_pad,
    kipy_rect,
    kipy_rectangle,
    kipy_segment,
    kipy_text,
    kipy_track,
    kipy_via,
    kipy_zone,
    layer_snapshot,
    netclass,
)
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.board.board_types_pb2 import PadType, ViaType, ZoneType
from kipy.proto.common import ApiStatusCode
from kipy.util.board_layer import iter_copper_layers
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from kibends.chain import NM_PER_MM, SameNetTrack, Segment, extract_chain
from kibends.coords import BOARD_FRAME, UserFrame
from kibends.snapshot import (
    BoardSnapshot,
    Disc,
    LayerSnapshot,
    PadKind,
    SelectionError,
    Via,
    ViaContact,
    ViaKind,
    arc_points,
    edge_paths_closed,
    outline_from_edge_paths,
    read_selection,
    read_user_frame,
)

LAYER = "F.Cu"


def board(fake: FakeBoard) -> kipy.board.Board:
    return cast(kipy.board.Board, fake)


def snapshot(fake: FakeBoard, replaced: frozenset[str] = frozenset()) -> LayerSnapshot:
    return layer_snapshot(fake, LAYER, replaced)


class TestReadSelection:
    def test_tracks_become_segments_in_nanometres(self) -> None:
        fake = FakeBoard(
            selection=[
                kipy_track("a", "SIG", (1, 2), (3, 2.5), width=0.3),
                kipy_track("b", "SIG", (3, 2.5), (4, 4), locked=True),
            ]
        )

        segments = read_selection(board(fake)).segments

        assert segments == [
            Segment(
                "a",
                (1_000_000, 2_000_000),
                (3_000_000, 2_500_000),
                300_000,
                "F.Cu",
                "SIG",
            ),
            Segment(
                "b",
                (3_000_000, 2_500_000),
                (4_000_000, 4_000_000),
                250_000,
                "F.Cu",
                "SIG",
                locked=True,
            ),
        ]

    def test_layer_is_canonical(self) -> None:
        fake = FakeBoard(
            selection=[kipy_track("a", "SIG", (0, 0), (1, 0), layer=BL.BL_In1_Cu)]
        )

        assert read_selection(board(fake)).segments[0].layer == "In1.Cu"

    def test_other_selected_items_are_ignored_with_a_warning(
        self, log: list[str]
    ) -> None:
        fake = FakeBoard(
            selection=[
                kipy_footprint("fp"),
                kipy_track("a", "SIG", (0, 0), (1, 0)),
                kipy_footprint("fp2"),
            ]
        )

        selection = read_selection(board(fake))

        assert [s.id for s in selection.segments] == ["a"]
        assert selection.vias == ()
        warnings = [line for line in log if line.startswith("WARNING")]
        assert len(warnings) == 1
        assert "2 FootprintInstance" in warnings[0]

    def test_selected_vias_are_kept_with_type_span_and_lock(
        self, log: list[str]
    ) -> None:
        fake = FakeBoard(
            selection=[
                kipy_via("v", "SIG", 1, 2, diameter=0.5, drill=0.25),
                kipy_track("a", "SIG", (0, 0), (1, 0)),
                kipy_via(
                    "w",
                    "SIG",
                    3,
                    0,
                    locked=True,
                    via_type=ViaType.VT_BLIND,
                    span=(BL.BL_F_Cu, BL.BL_In2_Cu),
                ),
            ]
        )

        selection = read_selection(board(fake))

        assert plain(selection.vias) == plain(
            (
                Via(
                    "v", "SIG", (1.0, 2.0), 0.5, 0.25, ViaKind.THROUGH, ("F.Cu", "B.Cu")
                ),
                Via(
                    "w",
                    "SIG",
                    (3.0, 0.0),
                    0.6,
                    0.3,
                    ViaKind.BLIND,
                    ("F.Cu", "In2.Cu"),
                    locked=True,
                ),
            )
        )
        assert not [line for line in log if line.startswith("WARNING")]

    def test_arc_track_is_an_error_naming_it(self) -> None:
        fake = FakeBoard(
            selection=[kipy_track("a", "SIG", (0, 0), (1, 0)), kipy_arc("arc1")]
        )

        with pytest.raises(SelectionError, match=r"arc.*arc1"):
            read_selection(board(fake))

    def test_no_tracks_is_an_error(self) -> None:
        fake = FakeBoard(selection=[kipy_via("v", "SIG", 0, 0)])

        with pytest.raises(SelectionError, match="no tracks"):
            read_selection(board(fake))

    def test_empty_selection_is_an_error(self) -> None:
        with pytest.raises(SelectionError, match="no tracks"):
            read_selection(board(FakeBoard()))


def test_user_frame_is_at_the_grid_origin() -> None:
    frame = read_user_frame(board(FakeBoard(grid_origin=(50.0, 40.25))))

    assert frame == UserFrame((50.0, 40.25))


class TestCopperOnLayer:
    def test_replaced_tracks_and_other_layers_are_left_out(self) -> None:
        fake = FakeBoard(
            tracks=[
                kipy_track("sel", "SIG", (0, 0), (1, 0)),
                kipy_track("other", "GND", (0, 1), (1, 1), width=0.4),
                kipy_track("back", "GND", (0, 2), (1, 2), layer=BL.BL_B_Cu),
            ]
        )

        snap = snapshot(fake, frozenset({"sel"}))

        assert [t.id for t in snap.tracks] == ["other"]
        assert snap.tracks[0].width == pytest.approx(0.4)
        assert snap.tracks[0].net == "GND"
        assert snap.tracks[0].centreline.equals(LineString([(0, 1), (1, 1)]))

    def test_arc_tracks_of_other_nets_are_read_as_polylines(self) -> None:
        snap = snapshot(FakeBoard(tracks=[kipy_arc("arc", net="GND")]))

        (track,) = snap.tracks
        assert track.centreline.coords[0] == pytest.approx((0, 0))
        assert track.centreline.coords[-1] == pytest.approx((2, 0))
        assert track.centreline.distance(Point(1, 1)) == pytest.approx(0, abs=1e-9)

    def test_pad_shape_on_the_layer_and_its_hole(self) -> None:
        pads = [
            kipy_pad(
                "tht",
                "GND",
                2,
                3,
                layers=(BL.BL_F_Cu, BL.BL_B_Cu),
                pad_type=PadType.PT_PTH,
                drill=0.8,
            ),
            kipy_pad("back", "GND", 5, 5, layers=(BL.BL_B_Cu,)),
        ]

        tht, back = snapshot(FakeBoard(pads=pads)).pads

        assert tht.kind is PadKind.PTH
        assert tht.drill == pytest.approx(0.8)
        assert tht.position == pytest.approx((2, 3))
        assert tht.shape is not None
        assert tht.shape.area == pytest.approx((2 * PAD_HALF_SIZE) ** 2)
        assert back.shape is None
        assert back.drill == 0

    def test_pad_shapes_are_read_for_the_snapshot_layer(self) -> None:
        pads = [
            kipy_pad("front", "GND", 2, 3),
            kipy_pad("back", "GND", 5, 5, layers=(BL.BL_B_Cu,)),
        ]

        front, back = layer_snapshot(FakeBoard(pads=pads), "B.Cu").pads

        assert front.shape is None
        assert back.shape is not None
        assert back.shape.centroid.coords[0] == pytest.approx((5, 5))

    def test_pad_shape_fallback_fills_every_pad_when_the_batch_drops_one(
        self,
    ) -> None:
        pads = [kipy_pad(f"p{i}", "GND", i, 0) for i in range(3)]

        snap = snapshot(FakeBoard(pads=pads, drops_pads=True))

        assert all(p.shape is not None for p in snap.pads)
        assert snap.pads[2].shape is not None
        assert snap.pads[2].shape.centroid.x == pytest.approx(2)

    def test_pad_names_its_footprint_and_number(self) -> None:
        pads = [kipy_pad("p", "GND", 0, 0, parent="fp", number="3")]
        fake = FakeBoard(pads=pads, footprints=[kipy_footprint("fp", "J7")])

        assert snapshot(fake).pads[0].name == "J7:3"

    def test_pad_inherits_footprint_clearance_override_unless_it_has_its_own(
        self,
    ) -> None:
        pads = [
            kipy_pad("own", "GND", 0, 0, override=0.2, parent="fp"),
            kipy_pad("inherits", "GND", 1, 0, parent="fp"),
            kipy_pad("free", "GND", 2, 0),
        ]
        fake = FakeBoard(pads=pads, footprints=[kipy_footprint("fp", override=1.6)])

        snap = snapshot(fake)

        assert [p.clearance_override for p in snap.pads] == [0.2, 1.6, None]

    def test_vias_are_read_with_their_sizes(self) -> None:
        (via,) = snapshot(FakeBoard(vias=[kipy_via("v", "GND", 1, 2, 0.8, 0.4)])).vias

        assert via.position == pytest.approx((1, 2))
        assert via.diameter == pytest.approx(0.8)
        assert via.drill == pytest.approx(0.4)


class TestClearances:
    def test_netclass_clearance_is_used_when_set(self) -> None:
        fake = FakeBoard(
            tracks=[kipy_track("t", "HV", (0, 0), (1, 0))],
            netclasses={"HV": netclass("HighVoltage", 0.5)},
        )

        assert snapshot(fake).clearance("HV") == pytest.approx(0.5)

    def test_unset_netclass_clearance_falls_back_to_default(self) -> None:
        fake = FakeBoard(
            tracks=[kipy_track("t", "SIG", (0, 0), (1, 0))],
            netclasses={"SIG": netclass("Bare")},
            default_clearance=0.15,
        )

        assert snapshot(fake).clearance("SIG") == pytest.approx(0.15)

    def test_no_net_and_unknown_nets_get_the_default(self) -> None:
        snap = snapshot(FakeBoard(default_clearance=0.15))

        assert snap.clearance("") == pytest.approx(0.15)
        assert snap.clearance("NOT_ON_LAYER") == pytest.approx(0.15)

    def test_replaced_net_is_looked_up_even_without_other_copper(self) -> None:
        fake = FakeBoard(
            tracks=[kipy_track("sel", "HV", (0, 0), (1, 0))],
            netclasses={"HV": netclass("HighVoltage", 0.5)},
        )

        assert snapshot(fake, frozenset({"sel"})).clearance("HV") == pytest.approx(0.5)


class TestDiffPairGaps:
    def test_netclass_diff_pair_gap_is_read_when_set(self) -> None:
        fake = FakeBoard(
            tracks=[kipy_track("t", "USB_D+", (0, 0), (1, 0))],
            netclasses={"USB_D+": netclass("USB", diff_pair_gap=0.15)},
        )

        assert snapshot(fake).diff_pair_gap("USB_D+") == pytest.approx(0.15)

    def test_unset_netclass_diff_pair_gap_is_unknown(self) -> None:
        fake = FakeBoard(
            tracks=[kipy_track("t", "USB_D+", (0, 0), (1, 0))],
            netclasses={"USB_D+": netclass("Bare")},
        )

        assert snapshot(fake).diff_pair_gap("USB_D+") is None


class TestSameNetCopper:
    def test_same_net_pads_vias_tracks_and_copper_shapes(self) -> None:
        fake = FakeBoard(
            pads=[kipy_pad("p", "SIG", 0, 0), kipy_pad("q", "GND", 5, 0)],
            vias=[kipy_via("v", "SIG", 2, 0, 0.6), kipy_via("w", "GND", 3, 0)],
            tracks=[
                kipy_track("sel", "SIG", (0, 0), (2, 0)),
                kipy_track("stub", "SIG", (2, 0), (2, 3), width=0.3),
                kipy_track("gnd", "GND", (0, 5), (5, 5)),
            ],
            shapes=[
                kipy_segment((0, 7), (4, 7), width=0.2, net="SIG", shape_id="s"),
                kipy_segment((0, 8), (4, 8), width=0.2, net="GND", shape_id="g"),
            ],
        )

        copper = snapshot(fake, frozenset({"sel"})).same_net_copper("SIG")

        pad, via, stub, shape = copper
        assert isinstance(pad, Polygon) and pad.contains(Point(0, 0))
        assert isinstance(via, Polygon)
        assert via.area == pytest.approx(math.pi * 0.3**2, rel=1e-2)
        assert isinstance(stub, SameNetTrack)
        assert stub.centreline.equals(LineString([(2, 0), (2, 3)]))
        assert stub.width == pytest.approx(0.3)
        assert isinstance(shape, Polygon) and shape.contains(Point(2, 7))

    def test_same_net_copper_shape_over_the_route_pins_the_chain(self) -> None:
        fake = FakeBoard(
            shapes=[kipy_segment((5, -1), (5, 1), width=1, net="SIG")],
        )
        chain_segment = Segment("a", (0, 0), (10 * NM_PER_MM, 0), 250_000, LAYER, "SIG")

        chain = extract_chain(
            [chain_segment], snapshot(fake).same_net_copper("SIG"), frame=BOARD_FRAME
        )

        assert chain.points == ((0, 0), (4.5, 0), (5.5, 0), (10, 0))
        assert chain.pinned == (True, True, True, True)

    def test_same_net_copper_zone_is_never_included(self) -> None:
        zone = kipy_zone(kipy_rect(-5, -5, 5, 5), [BL.BL_F_Cu], rule_area=False)

        assert snapshot(FakeBoard(zones=[zone])).same_net_copper("GND") == []


class TestGraphicsAndText:
    def test_copper_shapes_on_the_layer_including_footprint_ones(self) -> None:
        fake = FakeBoard(
            shapes=[
                kipy_segment((0, 0), (4, 0), width=0.2, net="GND", shape_id="g"),
                kipy_segment((0, 1), (4, 1), layer=BL.BL_B_Cu, shape_id="back"),
                kipy_segment((0, 2), (4, 2), layer=BL.BL_F_SilkS, shape_id="silk"),
            ],
            footprints=[
                kipy_footprint(
                    "fp", items=[kipy_segment((0, 3), (4, 3), shape_id="fpcu")]
                )
            ],
        )

        graphics = snapshot(fake).graphics

        assert [(g.description, g.net) for g in graphics] == [
            ("copper shape g", "GND"),
            ("copper shape fpcu", ""),
        ]
        assert graphics[0].geometry.area == pytest.approx(
            4 * 0.2 + math.pi * 0.01, rel=1e-2
        )

    @pytest.mark.parametrize(
        ("shape", "centre", "on_outline"),
        [
            (partial(kipy_circle, (5, 5), 1), (5, 5), (6, 5)),
            (partial(kipy_rectangle, (2, 2), (6, 4)), (4, 3), (6, 3)),
        ],
        ids=["circle", "rectangle"],
    )
    @pytest.mark.parametrize("filled", [True, False], ids=["filled", "unfilled"])
    def test_closed_copper_shape_covers_its_inside_only_when_filled(
        self,
        shape: Callable[[bool], kbt.BoardShape],
        centre: tuple[float, float],
        on_outline: tuple[float, float],
        filled: bool,
    ) -> None:
        (graphic,) = snapshot(FakeBoard(shapes=[shape(filled)])).graphics

        assert graphic.geometry.contains(Point(on_outline))
        assert graphic.geometry.contains(Point(centre)) == filled

    @pytest.mark.parametrize("filled", [True, False], ids=["filled", "unfilled"])
    def test_copper_rounded_rectangle_follows_its_rounded_corners(
        self, filled: bool
    ) -> None:
        d = 1 - math.sqrt(0.5)
        rect = kipy_rectangle((2, 2), (6, 4), filled, corner_radius=1)

        (graphic,) = snapshot(FakeBoard(shapes=[rect])).graphics

        assert graphic.geometry.contains(Point(2 + d, 2 + d))
        assert not graphic.geometry.contains(Point(2.1, 2.1))

    def test_unconvertible_copper_shape_is_skipped_with_a_warning(
        self, log: list[str]
    ) -> None:
        fake = FakeBoard(shapes=[kipy_bezier("bz")])

        assert snapshot(fake).graphics == ()
        assert any(
            line.startswith("WARNING") and "Ignoring copper BoardBezier bz" in line
            for line in log
        )

    def test_board_text_and_visible_footprint_fields_become_polygons(self) -> None:
        fake = FakeBoard(
            texts=[
                kipy_text("TOP", 10, 10),
                kipy_text("SILK", 20, 20, layer=BL.BL_F_SilkS),
            ],
            footprints=[
                kipy_footprint(
                    "fp",
                    items=[
                        kipy_field("R1", 5, 5),
                        kipy_field("hidden", 6, 6, visible=False),
                    ],
                )
            ],
        )

        graphics = snapshot(fake).graphics

        assert [g.description for g in graphics] == [
            "copper text 'TOP'",
            "copper text 'R1'",
        ]
        assert all(g.net == "" for g in graphics)
        assert graphics[0].geometry.area == pytest.approx(
            4 * TEXT_HALF_WIDTH * TEXT_HALF_HEIGHT
        )
        assert graphics[1].geometry.contains(Point(5, 5))

    def test_text_the_api_cannot_render_is_skipped_with_a_warning(
        self, log: list[str]
    ) -> None:
        fake = FakeBoard(
            texts=[kipy_text("TOP", 10, 10)],
            client=FakeClient(error=ApiError("unsupported")),
        )

        assert snapshot(fake).graphics == ()
        assert any(line.startswith("WARNING") and "text" in line for line in log)

    @pytest.mark.parametrize(
        "error",
        [
            ApiError("busy", code=ApiStatusCode.AS_BUSY),
            KiCadConnectionError("timed out"),
        ],
        ids=["busy", "unreachable"],
    )
    def test_busy_or_unreachable_kicad_ends_the_read(self, error: Exception) -> None:
        fake = FakeBoard(texts=[kipy_text("TOP", 10, 10)], client=FakeClient(error))

        with pytest.raises(type(error)) as raised:
            snapshot(fake)

        assert raised.value is error


class TestKeepoutsAndEdge:
    def test_track_keepouts_on_the_layer(self) -> None:
        zones = [
            kipy_zone(kipy_rect(0, 0, 2, 2), [BL.BL_F_Cu, BL.BL_B_Cu], zone_id="k"),
            kipy_zone(kipy_rect(0, 0, 3, 3), [BL.BL_F_Cu], keepout_tracks=False),
            kipy_zone(kipy_rect(0, 0, 4, 4), [BL.BL_B_Cu]),
            kipy_zone(kipy_rect(0, 0, 5, 5), [BL.BL_F_Cu], rule_area=False),
        ]

        (keepout,) = snapshot(FakeBoard(zones=zones)).keepouts

        assert keepout.area == pytest.approx(4)

    def test_edge_paths_from_board_and_footprint_edge_cuts(self) -> None:
        def edge(a: tuple[float, float], b: tuple[float, float]) -> kbt.BoardSegment:
            return kipy_segment(a, b, layer=BL.BL_Edge_Cuts)

        fake = FakeBoard(
            shapes=[
                edge((0, 0), (10, 0)),
                edge((10, 0), (10, 10)),
                edge((10, 10), (0, 10)),
            ],
            footprints=[kipy_footprint("fp", items=[edge((0, 10), (0, 0))])],
        )

        snap = snapshot(fake)

        assert edge_paths_closed(snap.edge_paths)
        assert outline_from_edge_paths(snap.edge_paths).area == pytest.approx(100)

    def test_edge_rectangle_is_a_closed_outline(self) -> None:
        fake = FakeBoard(
            shapes=[kipy_rectangle((0, 0), (10, 6), layer=BL.BL_Edge_Cuts)]
        )

        snap = snapshot(fake)

        assert edge_paths_closed(snap.edge_paths)
        assert outline_from_edge_paths(snap.edge_paths).area == pytest.approx(60)

    def test_edge_rectangle_with_corner_radius_has_arc_corners(self) -> None:
        rect = kipy_rectangle((0, 0), (10, 6), layer=BL.BL_Edge_Cuts, corner_radius=1)

        snap = snapshot(FakeBoard(shapes=[rect]))

        assert edge_paths_closed(snap.edge_paths)
        # Chords of 5° arcs lose about 1e-3 mm² per corner.
        assert outline_from_edge_paths(snap.edge_paths).area == pytest.approx(
            60 - (4 - math.pi), abs=1e-2
        )

    def test_edge_corner_radius_over_half_the_short_side_gives_a_stadium(
        self,
    ) -> None:
        rect = kipy_rectangle((0, 0), (4, 2), layer=BL.BL_Edge_Cuts, corner_radius=1.5)

        (path,) = snapshot(FakeBoard(shapes=[rect])).edge_paths

        # The radius clamps to 1 mm, half the 2 mm side.
        assert Polygon(path).is_valid
        assert Polygon(path).area == pytest.approx(8 - (4 - math.pi), abs=1e-2)

    def test_edge_rounded_rectangle_of_segments_and_arcs_closes(self) -> None:
        # A 10 x 6 mm board with 1 mm corner radii.
        d = 1 - math.sqrt(0.5)

        def edge(a: tuple[float, float], b: tuple[float, float]) -> kbt.BoardSegment:
            return kipy_segment(a, b, layer=BL.BL_Edge_Cuts)

        def corner(
            a: tuple[float, float], mid: tuple[float, float], b: tuple[float, float]
        ) -> kbt.BoardArc:
            return kipy_board_arc(a, mid, b, layer=BL.BL_Edge_Cuts)

        fake = FakeBoard(
            shapes=[
                edge((1, 0), (9, 0)),
                corner((9, 0), (10 - d, d), (10, 1)),
                edge((10, 1), (10, 5)),
                corner((10, 5), (10 - d, 6 - d), (9, 6)),
                edge((9, 6), (1, 6)),
                corner((1, 6), (d, 6 - d), (0, 5)),
                edge((0, 5), (0, 1)),
                corner((0, 1), (d, d), (1, 0)),
            ]
        )

        snap = snapshot(fake)

        assert edge_paths_closed(snap.edge_paths)
        # Chords of 5° arcs lose about 1e-3 mm² per corner.
        assert outline_from_edge_paths(snap.edge_paths).area == pytest.approx(
            60 - (4 - math.pi), abs=1e-2
        )


def plain(value: object) -> object:
    """`value` with snapshot dataclasses and geometry made comparable with ==."""
    if isinstance(value, BaseGeometry):
        return value.wkb
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(plain(getattr(value, f.name)) for f in fields(value))
    if isinstance(value, Mapping):
        items = cast(Mapping[object, object], value).items()
        return {k: plain(v) for k, v in items}
    if isinstance(value, tuple | list):
        return tuple(plain(v) for v in cast(Sequence[object], value))
    return value


def two_layer_board() -> FakeBoard:
    """Copper of every kind on F.Cu and B.Cu, with a replaced track on each."""
    return FakeBoard(
        tracks=[
            kipy_track("f-sel", "SIG", (0, 0), (1, 0)),
            kipy_track("f-gnd", "GND", (0, 1), (1, 1), width=0.4),
            kipy_track("b-sel", "B_ONLY", (0, 2), (1, 2), layer=BL.BL_B_Cu),
            kipy_track("b-gnd", "GND", (0, 3), (1, 3), layer=BL.BL_B_Cu),
        ],
        pads=[
            kipy_pad("top", "SIG", 2, 0),
            kipy_pad("bottom", "GND", 2, 3, layers=[BL.BL_B_Cu]),
            kipy_pad(
                "thru",
                "PWR",
                4,
                4,
                layers=[BL.BL_F_Cu, BL.BL_B_Cu],
                pad_type=PadType.PT_PTH,
                drill=0.8,
            ),
        ],
        vias=[kipy_via("v", "GND", 5, 5)],
        shapes=[
            kipy_segment((0, 6), (2, 6), net="PWR", shape_id="f-shape"),
            kipy_segment((0, 7), (2, 7), layer=BL.BL_B_Cu, shape_id="b-shape"),
            kipy_segment((0, 0), (9, 0), layer=BL.BL_Edge_Cuts, shape_id="edge"),
        ],
        texts=[
            kipy_text("TOP", 10, 10),
            kipy_text("BOTTOM", 10, 12, layer=BL.BL_B_Cu),
        ],
        zones=[
            kipy_zone(kipy_rect(0, 0, 2, 2), [BL.BL_F_Cu], zone_id="f-k"),
            kipy_zone(kipy_rect(0, 0, 3, 3), [BL.BL_B_Cu], zone_id="b-k"),
        ],
        netclasses={"PWR": netclass("Power", 0.3), "B_ONLY": netclass("B", 0.25)},
    )


class TestBoardSnapshot:
    LAYERS = ("F.Cu", "B.Cu")
    REPLACED = frozenset({"f-sel", "b-sel"})

    def test_slices_equal_single_layer_reads(self) -> None:
        fake = two_layer_board()

        snap = BoardSnapshot.read(board(fake), self.LAYERS)

        for layer in self.LAYERS:
            alone = layer_snapshot(fake, layer, self.REPLACED)
            assert plain(snap.slice(layer, self.REPLACED)) == plain(alone)

    def test_two_layers_read_each_board_wide_kind_once(self) -> None:
        fake = two_layer_board()

        snap = BoardSnapshot.read(board(fake), self.LAYERS)
        for layer in self.LAYERS:
            snap.slice(layer, self.REPLACED)

        assert fake.reads == {
            "get_footprints": 1,
            "get_tracks": 1,
            "get_pads": 1,
            "get_vias": 1,
            "get_shapes": 1,
            "get_text": 1,
            "get_zones": 1,
            "get_project": 1,
            "get_netclass_for_nets": 1,
            "get_pad_shapes_as_polygons": 2,
        }
        assert fake.client.renders == 2

    def test_layer_without_copper_gives_an_empty_slice(self) -> None:
        fake = FakeBoard(
            tracks=[kipy_track("t", "SIG", (0, 0), (1, 0))],
            pads=[kipy_pad("p", "SIG", 2, 0)],
            texts=[kipy_text("TOP", 10, 10)],
        )

        inner = BoardSnapshot.read(board(fake), ["F.Cu", "In1.Cu"]).slice(
            "In1.Cu", frozenset()
        )

        assert (inner.tracks, inner.graphics, inner.keepouts) == ((), (), ())
        assert [(p.id, p.shape) for p in inner.pads] == [("p", None)]
        assert inner.clearance("SIG") == pytest.approx(0.2)

    def test_a_via_is_on_the_layers_its_drill_spans(self) -> None:
        fake = FakeBoard(
            vias=[
                kipy_via("through", "SIG", 0, 0),
                kipy_via(
                    "blind",
                    "SIG",
                    1,
                    0,
                    via_type=ViaType.VT_BLIND,
                    span=(BL.BL_F_Cu, BL.BL_In1_Cu),
                ),
                kipy_via(
                    "buried",
                    "SIG",
                    2,
                    0,
                    via_type=ViaType.VT_BURIED,
                    span=(BL.BL_In2_Cu, BL.BL_In1_Cu),
                ),
            ]
        )
        layers = ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]

        snap = BoardSnapshot.read(board(fake), layers)

        assert {
            layer: [v.id for v in snap.slice(layer, ()).vias] for layer in layers
        } == {
            "F.Cu": ["through", "blind"],
            "In1.Cu": ["through", "blind", "buried"],
            "In2.Cu": ["through", "buried"],
            "B.Cu": ["through"],
        }

    @pytest.mark.parametrize(
        "span",
        [
            (BL.BL_UNKNOWN, BL.BL_UNKNOWN),
            (BL.BL_F_Cu, BL.BL_UNKNOWN),
            (BL.BL_UNKNOWN, BL.BL_In1_Cu),
        ],
    )
    def test_a_via_with_an_unknown_drill_layer_spans_the_board(
        self, span: tuple[BL.ValueType, BL.ValueType]
    ) -> None:
        fake = FakeBoard(vias=[kipy_via("v", "SIG", 0, 0, span=span)])
        layers = ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]

        snap = BoardSnapshot.read(board(fake), layers)

        assert snap.vias[0].span == ("F.Cu", "B.Cu")
        assert all(snap.slice(layer, ()).vias for layer in layers)

    def test_slicing_a_layer_not_read_is_an_error(self) -> None:
        snap = BoardSnapshot.read(board(FakeBoard()), ["F.Cu"])

        with pytest.raises(ValueError, match=r"B\.Cu"):
            snap.slice("B.Cu", frozenset())


class TestViaContacts:
    VIA = (5.0, 5.0)

    def contact_board(self) -> FakeBoard:
        x, y = self.VIA
        return FakeBoard(
            selection=[
                kipy_via("v", "SIG", x, y),
                kipy_track("front", "SIG", (0, y), self.VIA),
            ],
            vias=[
                kipy_via("v", "SIG", x, y),
                kipy_via("w", "SIG", x + 0.5, y),
                kipy_via("far", "SIG", x + 2, y),
                kipy_via("other", "GND", x, y + 0.5),
            ],
            tracks=[
                kipy_track("front", "SIG", (0, y), self.VIA),
                kipy_track("inner", "SIG", self.VIA, (x, 9), layer=BL.BL_In1_Cu),
                kipy_track("far", "SIG", (0, 0), (1, 0)),
                kipy_track("gnd", "GND", (x, 0), (x, 10)),
            ],
            pads=[
                kipy_pad("back", "SIG", x, y + 0.4, layers=(BL.BL_B_Cu,)),
                kipy_pad("gnd", "GND", x, y),
            ],
            shapes=[kipy_segment(self.VIA, (7, y), BL.BL_B_Cu, net="SIG")],
            zones=[
                kipy_filled_zone("fill", "SIG", {BL.BL_In2_Cu: kipy_rect(4, 4, 6, 6)}),
                kipy_filled_zone("away", "SIG", {BL.BL_In3_Cu: kipy_rect(0, 0, 1, 1)}),
                kipy_filled_zone("gnd", "GND", {BL.BL_F_Cu: kipy_rect(4, 4, 6, 6)}),
                kipy_filled_zone(
                    "drop",
                    "SIG",
                    {BL.BL_F_Cu: kipy_rect(4, 4, 6, 6)},
                    ZoneType.ZT_TEARDROP,
                ),
            ],
        )

    def test_same_net_copper_touching_on_any_layer(self) -> None:
        fake = self.contact_board()
        (via,) = read_selection(board(fake)).vias

        snap = BoardSnapshot.read(board(fake), ["F.Cu"], [via])

        assert snap.via_contacts == {
            "v": (
                ViaContact("copper shape shape on B.Cu"),
                ViaContact("pad back"),
                ViaContact("track front on F.Cu", "front"),
                ViaContact("track inner on In1.Cu", "inner"),
                ViaContact("via w"),
                ViaContact("zone fill on In2.Cu"),
            )
        }

    def test_copper_touching_what_a_via_carries_counts(self) -> None:
        fake = self.contact_board()
        (via,) = read_selection(board(fake)).vias
        # Clear of the via, 0.1 mm from the far track and the far fill.
        end = Disc((1.1, 0.0), 0.2)

        snap = BoardSnapshot.read(board(fake), ["F.Cu"], [via], carried={"v": [end]})

        assert {
            ViaContact("track far on F.Cu", "far"),
            ViaContact("zone away on In3.Cu"),
        } <= set(snap.via_contacts["v"])

    def test_pads_on_layers_not_read_cost_one_read_per_layer(self) -> None:
        fake = self.contact_board()
        (via,) = read_selection(board(fake)).vias

        BoardSnapshot.read(board(fake), ["F.Cu"], [via])

        assert fake.reads["get_pad_shapes_as_polygons"] == 2

    @pytest.mark.parametrize(
        ("layers", "shape_reads", "layer_reads"),
        [(["F.Cu"], 2, 1), (["F.Cu", "B.Cu"], 2, 1)],
    )
    def test_pad_layers_the_board_lacks_are_not_read(
        self, layers: list[str], shape_reads: int, layer_reads: int
    ) -> None:
        via = kipy_via("v", "SIG", *self.VIA)
        pth = kipy_pad("pth", "SIG", 0, 0, layers=list(iter_copper_layers()))
        fake = FakeBoard(
            selection=[via, kipy_track("t", "SIG", (0, 0), self.VIA)],
            vias=[via],
            pads=[pth],
        )
        (selected,) = read_selection(board(fake)).vias

        BoardSnapshot.read(board(fake), layers, [selected])

        assert fake.reads["get_pad_shapes_as_polygons"] == shape_reads
        assert fake.reads["get_enabled_layers"] == layer_reads

    def test_no_selected_via_reads_nothing_more(self) -> None:
        fake = self.contact_board()

        snap = BoardSnapshot.read(board(fake), ["F.Cu"])

        assert snap.via_contacts == {}
        assert fake.reads["get_pad_shapes_as_polygons"] == 1


class TestGeometryHelpers:
    def test_arc_points_keep_exact_endpoints_and_pass_through_mid(self) -> None:
        start, mid, end = (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)

        pts = arc_points(start, mid, end)

        assert pts[0] == start and pts[-1] == end
        assert all(math.hypot(x, y) == pytest.approx(1.0) for x, y in pts)
        assert all(y >= -1e-12 for _, y in pts)

    def test_open_outline_is_not_closed(self) -> None:
        square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

        assert not edge_paths_closed([square])
        assert edge_paths_closed([[*square, (0.0, 0.0)]])

    def test_internal_cutout_is_a_hole_in_the_outline(self) -> None:
        def square(x0: float, x1: float) -> list[tuple[float, float]]:
            return [(x0, x0), (x1, x0), (x1, x1), (x0, x1), (x0, x0)]

        outline = outline_from_edge_paths([square(0, 10), square(4, 6)])

        assert outline.area == pytest.approx(96)
        assert not outline.contains(Point(5, 5))
