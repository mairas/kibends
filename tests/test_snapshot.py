import math
from collections.abc import Callable
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
    kipy_footprint,
    kipy_pad,
    kipy_rect,
    kipy_rectangle,
    kipy_segment,
    kipy_text,
    kipy_track,
    kipy_via,
    kipy_zone,
    netclass,
)
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.board.board_types_pb2 import PadType
from kipy.proto.common import ApiStatusCode
from shapely.geometry import LineString, Point, Polygon

from kibends.chain import NM_PER_MM, SameNetTrack, Segment, extract_chain
from kibends.snapshot import (
    LayerSnapshot,
    PadKind,
    SelectionError,
    arc_points,
    edge_paths_closed,
    outline_from_edge_paths,
    read_selection,
)

LAYER = "F.Cu"


def board(fake: FakeBoard) -> kipy.board.Board:
    return cast(kipy.board.Board, fake)


def snapshot(fake: FakeBoard, replaced: frozenset[str] = frozenset()) -> LayerSnapshot:
    return LayerSnapshot.from_board(board(fake), LAYER, replaced)


class TestReadSelection:
    def test_tracks_become_segments_in_nanometres(self) -> None:
        fake = FakeBoard(
            selection=[
                kipy_track("a", "SIG", (1, 2), (3, 2.5), width=0.3),
                kipy_track("b", "SIG", (3, 2.5), (4, 4), locked=True),
            ]
        )

        segments = read_selection(board(fake))

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

        assert read_selection(board(fake))[0].layer == "In1.Cu"

    def test_other_selected_items_are_ignored_with_a_warning(
        self, log: list[str]
    ) -> None:
        fake = FakeBoard(
            selection=[
                kipy_via("v", "SIG", 0, 0),
                kipy_track("a", "SIG", (0, 0), (1, 0)),
                kipy_footprint("fp"),
            ]
        )

        segments = read_selection(board(fake))

        assert [s.id for s in segments] == ["a"]
        warnings = [line for line in log if line.startswith("WARNING")]
        assert len(warnings) == 1
        assert "Via" in warnings[0] and "FootprintInstance" in warnings[0]

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

        chain = extract_chain([chain_segment], snapshot(fake).same_net_copper("SIG"))

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
            (lambda filled: kipy_circle((5, 5), 1, filled), (5, 5), (6, 5)),
            (lambda filled: kipy_rectangle((2, 2), (6, 4), filled), (4, 3), (6, 3)),
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
