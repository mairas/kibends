from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
from typing import cast

import kipy.board
import kipy.board_types as kbt
import numpy as np
import pytest
from conftest import (
    PAD_HALF_SIZE,
    FakeBoard,
    kipy_pad,
    kipy_segment,
    kipy_track,
    kipy_via,
    netclass,
)
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.board.board_types_pb2 import PadType, ViaType
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from kibends.energy import FloatArray, ObstacleField
from kibends.obstacles import (
    EDGE_CLEARANCE,
    HOLE_CLEARANCE,
    HOLE_TO_HOLE,
    ChainCentreline,
    build_obstacles,
    build_via_obstacles,
    chain_obstacles_for_via,
    moving_via_obstacles,
    via_obstacles_for_via,
)
from kibends.snapshot import BoardSnapshot, LayerSnapshot, Via, ViaZone

NET = "SIG"
VIA_ID = "moving"
DIAMETER = 0.6
DRILL = 0.3
RADIUS = DIAMETER / 2
HOLE_RADIUS = DRILL / 2
DEFAULT_CLEARANCE = 0.2
# A via's copper keeps the copper clearance with its ring, and the hole
# clearance with its hole; with the defaults the ring is the farther reach.
COPPER = max(DEFAULT_CLEARANCE + RADIUS, HOLE_CLEARANCE + HOLE_RADIUS)
NM_TOLERANCE = 1e-6
BOTH = ("F.Cu", "B.Cu")
BOTH_LAYERS = (BL.BL_F_Cu, BL.BL_B_Cu)


def points(*xy: tuple[float, float]) -> FloatArray:
    return np.array(xy, dtype=float)


def moving_via(x: float = 0.0, y: float = 0.0) -> kbt.Via:
    return kipy_via(VIA_ID, NET, x, y, diameter=DIAMETER, drill=DRILL)


def slices(
    fake: FakeBoard,
    layers: Sequence[str] = BOTH,
    replaced: Collection[str] = frozenset(),
) -> list[LayerSnapshot]:
    board = BoardSnapshot.read(cast(kipy.board.Board, fake), layers)
    return [board.slice(layer, replaced) for layer in layers]


def via_of(layers: Sequence[LayerSnapshot], via_id: str = VIA_ID) -> Via:
    return next(v for v in layers[0].vias if v.id == via_id)


def field_of(
    fake: FakeBoard,
    layers: Sequence[str] = BOTH,
    *,
    hole_clearance: float = HOLE_CLEARANCE,
    hole_to_hole: float = HOLE_TO_HOLE,
    allow_via_in_pad: bool = False,
) -> ObstacleField:
    snaps = slices(fake, layers)
    return build_via_obstacles(
        snaps,
        via_of(snaps),
        hole_clearance=hole_clearance,
        hole_to_hole=hole_to_hole,
        allow_via_in_pad=allow_via_in_pad,
    )


def edge_segments(corners: list[tuple[float, float]]) -> list[kbt.BoardShape]:
    path = [*corners, corners[0]]
    return [kipy_segment(a, b, layer=BL.BL_Edge_Cuts) for a, b in pairwise(path)]


PAD_ON_TOP = kipy_pad("top", "GND", 5, 0, layers=(BL.BL_F_Cu,))
TRACK_BELOW = kipy_track("below", "GND", (0, 3), (10, 3), layer=BL.BL_B_Cu)
TRACK_HALF = 0.25 / 2


class TestViaField:
    def test_the_nearer_of_a_pad_on_top_and_a_track_below_limits_it(self) -> None:
        fake = FakeBoard(pads=[PAD_ON_TOP], tracks=[TRACK_BELOW], vias=[moving_via()])

        distance, gradient = field_of(fake).signed_distance(points((2, 0), (5, 1.8)))

        to_pad = (5 - PAD_HALF_SIZE) - 2 - COPPER
        to_track = 3 - 1.8 - TRACK_HALF - COPPER
        assert to_track < (1.8 - PAD_HALF_SIZE) - COPPER < 3 - TRACK_HALF - COPPER
        assert distance == pytest.approx([to_pad, to_track], abs=NM_TOLERANCE)
        assert gradient == pytest.approx(np.array([(-1, 0), (0, -1)]))

    def test_a_layer_with_no_copper_nearby_contributes_nothing(self) -> None:
        fake = FakeBoard(pads=[PAD_ON_TOP], vias=[moving_via()])
        probes = points((2, 0), (4, 1), (8, -3))

        both, _ = field_of(fake).signed_distance(probes)
        top_only, _ = field_of(fake, ("F.Cu",)).signed_distance(probes)

        assert both == pytest.approx(top_only)

    def test_the_hole_clearance_reaches_farther_when_it_is_larger(self) -> None:
        fake = FakeBoard(pads=[PAD_ON_TOP], vias=[moving_via()])

        distance, _ = field_of(fake, hole_clearance=0.6).signed_distance(points((2, 0)))

        expected = (5 - PAD_HALF_SIZE) - 2 - (HOLE_RADIUS + 0.6)
        assert distance[0] == pytest.approx(expected, abs=NM_TOLERANCE)

    def test_larger_netclass_clearance_of_either_net_wins(self) -> None:
        fake = FakeBoard(
            pads=[PAD_ON_TOP],
            vias=[moving_via()],
            netclasses={"GND": netclass("Power", 0.5)},
        )

        distance, _ = field_of(fake).signed_distance(points((2, 0)))

        expected = (5 - PAD_HALF_SIZE) - 2 - (0.5 + RADIUS)
        assert distance[0] == pytest.approx(expected, abs=NM_TOLERANCE)

    def test_with_via_in_pad_its_own_nets_pads_are_no_obstacle(self) -> None:
        fake = FakeBoard(
            pads=[kipy_pad("mine", NET, 1, 0, layers=(BL.BL_F_Cu,))],
            tracks=[kipy_track("stub", NET, (0, 1), (5, 1), layer=BL.BL_B_Cu)],
            vias=[moving_via()],
        )

        field = field_of(fake, allow_via_in_pad=True)
        distance, _ = field.signed_distance(points((1, 0), (3, 1)))

        assert distance[0] == pytest.approx(1 - 0.125 - RADIUS, abs=NM_TOLERANCE)
        assert distance[1] == pytest.approx(-0.125 - RADIUS, abs=NM_TOLERANCE)

    def test_with_via_in_pad_a_pads_hole_still_counts(self) -> None:
        pth = kipy_pad(
            "pth", NET, 0, 0, layers=BOTH_LAYERS, pad_type=PadType.PT_PTH, drill=1.0
        )
        fake = FakeBoard(pads=[pth], vias=[moving_via(5, 0)])

        distance, _ = field_of(fake, allow_via_in_pad=True).signed_distance(
            points((2, 0))
        )

        expected = 2 - (0.5 + HOLE_TO_HOLE + HOLE_RADIUS)
        assert distance[0] == pytest.approx(expected, abs=NM_TOLERANCE)

    def test_same_net_copper_may_touch_the_ring(self) -> None:
        fake = FakeBoard(
            pads=[kipy_pad("mine", NET, 1, 0, layers=(BL.BL_F_Cu,))],
            tracks=[kipy_track("stub", NET, (0, 1), (5, 1), layer=BL.BL_B_Cu)],
            vias=[moving_via()],
        )

        distance, _ = field_of(fake).signed_distance(points((0, 0), (3, 1)))

        pad_edge = 1 - PAD_HALF_SIZE
        assert distance[0] == pytest.approx(pad_edge - RADIUS, abs=NM_TOLERANCE)
        half_width = 0.125
        assert distance[1] == pytest.approx(-half_width - RADIUS, abs=NM_TOLERANCE)

    def test_the_via_is_no_obstacle_to_itself(self) -> None:
        distance, _ = field_of(FakeBoard(vias=[moving_via()])).signed_distance(
            points((0, 0))
        )

        assert np.isinf(distance[0])

    def test_other_moving_vias_are_left_to_the_dynamic_fields(self) -> None:
        fake = FakeBoard(vias=[moving_via(), kipy_via("also", "GND", 2, 0)])
        snaps = slices(fake)

        field = build_via_obstacles(snaps, via_of(snaps), moving={"also"})
        distance, _ = field.signed_distance(points((1, 0)))

        assert np.isinf(distance[0])

    def test_board_edge_keeps_its_clearance_from_the_ring(self) -> None:
        fake = FakeBoard(
            shapes=edge_segments([(0, 0), (20, 0), (20, 10), (0, 10)]),
            vias=[moving_via(5, 5)],
        )

        distance, gradient = field_of(fake).signed_distance(points((1, 5)))

        assert distance[0] == pytest.approx(
            1 - EDGE_CLEARANCE - RADIUS, abs=NM_TOLERANCE
        )
        assert gradient[0] == pytest.approx((1, 0))

    def test_segments_clear_checks_every_layer(self) -> None:
        fake = FakeBoard(tracks=[TRACK_BELOW], vias=[moving_via()])

        clear = field_of(fake).segments_clear(
            points((1, 0), (1, 0)), points((1, 2), (1, 2.5))
        )

        assert list(clear) == [True, False]


FILL = box(-2, -1, 2, 1)
"""A zone fill 4 mm by 2 mm around the via at (0, 0)."""


def zone(
    fills: Mapping[str, BaseGeometry] | None = None, min_width: float = 0.0
) -> ViaZone:
    """A zone of the via's net filled with `FILL` on In1.Cu, or with
    `fills` by layer."""
    return ViaZone("plane", fills or {"In1.Cu": FILL}, min_width)


MARGIN = zone().margin
"""How far inside the fill of a zone without a minimum width a via's centre
stays."""


class TestZoneContainment:
    def test_keeps_the_centre_inside_the_fill_by_the_margin(self) -> None:
        snaps = slices(FakeBoard(vias=[moving_via()]))

        field = build_via_obstacles(snaps, via_of(snaps), zones=[zone()])
        distance, gradient = field.signed_distance(points((0, 0.5), (0, 1.2)))

        assert distance == pytest.approx([0.5 - MARGIN, -0.2 - MARGIN])
        assert gradient == pytest.approx(np.array([(0, -1), (0, -1)]))

    def test_a_wider_minimum_width_keeps_it_deeper(self) -> None:
        snaps = slices(FakeBoard(vias=[moving_via()]))

        field = build_via_obstacles(snaps, via_of(snaps), zones=[zone(min_width=1.0)])
        distance, _ = field.signed_distance(points((0, 0.5)))

        assert distance[0] == pytest.approx(0.5 - zone(min_width=1.0).margin)

    def test_keeps_the_centre_out_of_a_hole_in_the_fill(self) -> None:
        snaps = slices(FakeBoard(vias=[moving_via()]))
        holed = FILL.difference(box(1, -0.5, 1.5, 0.5))

        field = build_via_obstacles(
            snaps, via_of(snaps), zones=[zone({"In1.Cu": holed})]
        )
        distance, gradient = field.signed_distance(points((0.5, 0), (1.25, 0)))

        assert distance == pytest.approx([0.5 - MARGIN, -0.25 - MARGIN])
        assert gradient == pytest.approx(np.array([(-1, 0), (-1, 0)]))

    def test_keeps_the_centre_inside_the_fill_on_each_layer(self) -> None:
        snaps = slices(FakeBoard(vias=[moving_via()]))
        narrower = box(-2, -1, 2, 0.6)

        field = build_via_obstacles(
            snaps,
            via_of(snaps),
            zones=[zone({"In1.Cu": FILL, "In2.Cu": narrower})],
        )
        distance, _ = field.signed_distance(points((0, 0.5), (1.5, 0)))

        assert distance == pytest.approx([0.1 - MARGIN, 0.5 - MARGIN])

    def test_keeps_the_centre_inside_the_fill_of_each_zone(self) -> None:
        snaps = slices(FakeBoard(vias=[moving_via()]))
        across = ViaZone("strip", {"B.Cu": box(-0.5, -3, 0.5, 3)}, 0.0)

        field = build_via_obstacles(snaps, via_of(snaps), zones=[zone(), across])
        distance, _ = field.signed_distance(points((0, 0.5), (0.4, 0)))

        assert distance == pytest.approx([0.5 - MARGIN, 0.1 - MARGIN])

    def test_a_move_out_of_the_fill_is_not_clear(self) -> None:
        snaps = slices(FakeBoard(vias=[moving_via()]))

        field = build_via_obstacles(snaps, via_of(snaps), zones=[zone()])
        clear = field.segments_clear(points((0, 0), (0, 0)), points((0, 0.9), (0, 1.2)))

        assert list(clear) == [True, False]


class TestViaDrill:
    def test_another_net_hole_keeps_the_hole_clearance_from_the_ring(self) -> None:
        mounting_hole = kipy_pad(
            "mh", "", 0, 0, layers=(), pad_type=PadType.PT_NPTH, drill=1.0
        )
        fake = FakeBoard(pads=[mounting_hole], vias=[moving_via(5, 0)])

        distance, _ = field_of(fake).signed_distance(points((2, 0)))

        expected = 2 - max(
            0.5 + HOLE_CLEARANCE + RADIUS, 0.5 + HOLE_TO_HOLE + HOLE_RADIUS
        )
        assert distance[0] == pytest.approx(expected, abs=NM_TOLERANCE)

    def test_hole_to_hole_limits_it_with_no_copper_on_any_layer(self) -> None:
        mounting_hole = kipy_pad(
            "mh", "", 0, 0, layers=(), pad_type=PadType.PT_NPTH, drill=1.0
        )
        fake = FakeBoard(pads=[mounting_hole], vias=[moving_via(5, 0)])

        distance, _ = field_of(fake, hole_to_hole=1.0).signed_distance(points((2, 0)))

        expected = 2 - (0.5 + 1.0 + HOLE_RADIUS)
        assert distance[0] == pytest.approx(expected, abs=NM_TOLERANCE)

    def test_a_same_net_pad_hole_counts_only_hole_to_hole(self) -> None:
        tht = kipy_pad(
            "tht",
            NET,
            0,
            0,
            layers=(BL.BL_F_Cu, BL.BL_B_Cu),
            pad_type=PadType.PT_PTH,
            drill=1.0,
        )
        fake = FakeBoard(pads=[tht], vias=[moving_via(5, 0)])

        distance, _ = field_of(fake).signed_distance(points((2, 0)))

        expected = 2 - (0.5 + HOLE_TO_HOLE + HOLE_RADIUS)
        assert distance[0] == pytest.approx(expected, abs=NM_TOLERANCE)

    def test_a_same_net_via_hole_counts_only_hole_to_hole(self) -> None:
        # A ring this narrow touches the via's ring nearer than hole-to-hole.
        stitch = kipy_via("stitch", NET, 0, 0, diameter=0.45, drill=0.4)
        fake = FakeBoard(vias=[stitch, moving_via(5, 0)])

        distance, _ = field_of(fake).signed_distance(points((2, 0)))

        assert 0.225 + RADIUS < 0.2 + HOLE_TO_HOLE + HOLE_RADIUS
        expected = 2 - (0.2 + HOLE_TO_HOLE + HOLE_RADIUS)
        assert distance[0] == pytest.approx(expected, abs=NM_TOLERANCE)

    def test_an_unplated_hole_of_its_own_net_keeps_the_hole_clearance(
        self,
    ) -> None:
        npth = kipy_pad(
            "npth", NET, 0, 0, layers=(), pad_type=PadType.PT_NPTH, drill=1.0
        )
        fake = FakeBoard(pads=[npth], vias=[moving_via(5, 0)])

        distance, _ = field_of(fake).signed_distance(points((2, 0)))

        ring = 0.5 + HOLE_CLEARANCE + RADIUS
        assert ring > 0.5 + HOLE_TO_HOLE + HOLE_RADIUS
        assert distance[0] == pytest.approx(2 - ring, abs=NM_TOLERANCE)

    def test_holes_come_from_every_layer_read(self) -> None:
        # Of the via's own net, so only its hole can make the distance finite.
        blind = kipy_via(
            "blind",
            NET,
            0,
            0,
            via_type=ViaType.VT_BLIND,
            span=(BL.BL_In1_Cu, BL.BL_B_Cu),
        )
        fake = FakeBoard(vias=[blind, moving_via(5, 0)])

        top, _ = field_of(fake, ("F.Cu",)).signed_distance(points((2, 0)))
        both, _ = field_of(fake).signed_distance(points((2, 0)))

        assert np.isinf(top[0])
        assert np.isfinite(both[0])


CHAIN_WIDTH = 0.25
GND_VIA = kipy_via("gnd", "GND", 3, 0, diameter=0.8, drill=0.4)


def board_via(fake: FakeBoard, via_id: str) -> Via:
    return via_of(slices(fake, ("F.Cu",)), via_id)


class TestViaAsObstacleToChains:
    def test_matches_the_same_via_fixed_on_the_board(self) -> None:
        fake = FakeBoard(vias=[GND_VIA], netclasses={"GND": netclass("Power", 0.3)})
        snap = slices(fake, ("F.Cu",))[0]
        probes = np.random.default_rng(1).uniform(-2, 8, (200, 2))

        fixed = build_obstacles(snap, NET, CHAIN_WIDTH)
        moving = moving_via_obstacles(
            "F.Cu", NET, CHAIN_WIDTH, snap.clearance(NET), snap.vias, snap.clearance
        )

        assert moving is not None
        expected, expected_gradient = fixed.signed_distance(probes)
        distance, gradient = moving.signed_distance(probes)
        assert distance == pytest.approx(expected, abs=NM_TOLERANCE)
        assert gradient == pytest.approx(expected_gradient, abs=NM_TOLERANCE)

    def test_a_disc_of_the_via_diameter_with_copper_clearance(self) -> None:
        via = board_via(FakeBoard(vias=[GND_VIA]), "gnd")

        field = moving_via_obstacles(
            "F.Cu", NET, CHAIN_WIDTH, DEFAULT_CLEARANCE, [via], lambda _: 0.3
        )

        assert field is not None
        distance, _ = field.signed_distance(points((5, 0)))
        expected = 2 - (0.4 + 0.3 + CHAIN_WIDTH / 2)
        assert distance[0] == pytest.approx(expected, abs=NM_TOLERANCE)

    def test_follows_the_via_to_its_current_position(self) -> None:
        via = replace(board_via(FakeBoard(vias=[GND_VIA]), "gnd"), position=(0.0, 0.0))

        field = moving_via_obstacles(
            "F.Cu", NET, CHAIN_WIDTH, DEFAULT_CLEARANCE, [via], lambda _: 0.2
        )

        assert field is not None
        distance, _ = field.signed_distance(points((2, 0)))
        expected = 2 - (0.4 + 0.2 + CHAIN_WIDTH / 2)
        assert distance[0] == pytest.approx(expected, abs=NM_TOLERANCE)

    def test_a_same_net_via_may_touch_but_not_overlap(self) -> None:
        via = board_via(FakeBoard(vias=[moving_via()]), VIA_ID)

        field = moving_via_obstacles(
            "F.Cu", NET, CHAIN_WIDTH, DEFAULT_CLEARANCE, [via], lambda _: 0.2
        )

        assert field is not None
        distance, _ = field.signed_distance(points((2, 0)))
        assert distance[0] == pytest.approx(2 - (RADIUS + CHAIN_WIDTH / 2))

    def test_nothing_on_a_layer_the_via_does_not_span(self) -> None:
        blind = kipy_via(
            "blind",
            "GND",
            0,
            0,
            via_type=ViaType.VT_BLIND,
            span=(BL.BL_F_Cu, BL.BL_In1_Cu),
        )
        via = board_via(FakeBoard(vias=[blind]), "blind")

        field = moving_via_obstacles(
            "B.Cu", NET, CHAIN_WIDTH, DEFAULT_CLEARANCE, [via], lambda _: 0.2
        )

        assert field is None


PROBES = np.random.default_rng(2).uniform(-3, 5, (200, 2))
GND_TRACK = kipy_track("gnd", "GND", (-5, 2), (5, 2), width=0.3)
POWER = {"GND": netclass("Power", 0.3)}


def gnd_chain() -> ChainCentreline:
    return ChainCentreline("chain GND", "GND", 0.3, 0.3, points((-5, 2), (5, 2)))


class TestChainsAsObstaclesToAVia:
    def test_the_ring_keeps_the_copper_clearance(self) -> None:
        via = board_via(FakeBoard(vias=[moving_via()]), VIA_ID)

        field = chain_obstacles_for_via(via, [gnd_chain()], lambda _: DEFAULT_CLEARANCE)
        distance, _ = field.signed_distance(points((0, 0)))

        assert distance[0] == pytest.approx(2 - (0.15 + 0.3 + RADIUS))

    def test_the_hole_keeps_the_hole_clearance_where_it_reaches_farther(
        self,
    ) -> None:
        via = board_via(FakeBoard(vias=[moving_via()]), VIA_ID)

        field = chain_obstacles_for_via(
            via, [gnd_chain()], lambda _: DEFAULT_CLEARANCE, hole_clearance=0.6
        )
        distance, _ = field.signed_distance(points((0, 0)))

        assert 0.6 + HOLE_RADIUS > 0.3 + RADIUS
        assert distance[0] == pytest.approx(2 - (0.15 + 0.6 + HOLE_RADIUS))

    def test_matches_the_same_track_fixed_on_the_board(self) -> None:
        fake = FakeBoard(tracks=[GND_TRACK], vias=[moving_via()], netclasses=POWER)
        snaps = slices(fake, ("F.Cu",))
        via = via_of(snaps)

        fixed = build_via_obstacles(snaps, via, hole_clearance=0.6)
        moving = chain_obstacles_for_via(
            via, [gnd_chain()], lambda _: DEFAULT_CLEARANCE, hole_clearance=0.6
        )

        expected, expected_gradient = fixed.signed_distance(PROBES)
        distance, gradient = moving.signed_distance(PROBES)
        assert distance == pytest.approx(expected, abs=NM_TOLERANCE)
        assert gradient == pytest.approx(expected_gradient, abs=NM_TOLERANCE)

    def test_a_same_net_chain_may_touch_the_ring_but_not_overlap_it(self) -> None:
        via = board_via(FakeBoard(vias=[moving_via()]), VIA_ID)
        mine = ChainCentreline("chain SIG", NET, 0.25, 0.2, points((-5, 1), (5, 1)))

        field = chain_obstacles_for_via(via, [mine], lambda _: DEFAULT_CLEARANCE)
        distance, _ = field.signed_distance(points((0, 0)))

        assert distance[0] == pytest.approx(1 - (0.25 / 2 + RADIUS))


class TestMovingViasAsObstaclesToAVia:
    def test_another_net_via_keeps_the_larger_of_every_rule(self) -> None:
        fake = FakeBoard(vias=[moving_via(), GND_VIA])
        via, other = board_via(fake, VIA_ID), board_via(fake, "gnd")

        field = via_obstacles_for_via(via, [other], lambda _: DEFAULT_CLEARANCE)

        assert field is not None
        distance, _ = field.signed_distance(points((0, 0)))
        ring = 0.4 + DEFAULT_CLEARANCE + RADIUS
        assert ring > 0.4 + HOLE_CLEARANCE + HOLE_RADIUS
        assert distance[0] == pytest.approx(3 - ring)

    def test_hole_to_hole_wins_when_it_reaches_farther(self) -> None:
        fake = FakeBoard(vias=[moving_via(), GND_VIA])
        via, other = board_via(fake, VIA_ID), board_via(fake, "gnd")

        field = via_obstacles_for_via(
            via, [other], lambda _: DEFAULT_CLEARANCE, hole_to_hole=1.0
        )

        assert field is not None
        distance, _ = field.signed_distance(points((0, 0)))
        assert distance[0] == pytest.approx(3 - (0.2 + 1.0 + HOLE_RADIUS))

    def test_a_same_net_via_counts_only_hole_to_hole(self) -> None:
        stitch = kipy_via("stitch", NET, 3, 0, diameter=0.8, drill=0.4)
        fake = FakeBoard(vias=[moving_via(), stitch])
        via, other = board_via(fake, VIA_ID), board_via(fake, "stitch")

        field = via_obstacles_for_via(via, [other], lambda _: DEFAULT_CLEARANCE)

        assert field is not None
        distance, _ = field.signed_distance(points((0, 0)))
        assert distance[0] == pytest.approx(3 - (0.2 + HOLE_TO_HOLE + HOLE_RADIUS))

    def test_matches_the_same_via_fixed_on_the_board(self) -> None:
        fake = FakeBoard(vias=[moving_via(), GND_VIA], netclasses=POWER)
        snaps = slices(fake)
        via, other = via_of(snaps), via_of(snaps, "gnd")

        fixed = build_via_obstacles(snaps, via, hole_clearance=0.6)
        moving = via_obstacles_for_via(
            via, [other], snaps[0].clearance, hole_clearance=0.6
        )

        assert moving is not None
        expected, expected_gradient = fixed.signed_distance(PROBES)
        distance, gradient = moving.signed_distance(PROBES)
        assert distance == pytest.approx(expected, abs=NM_TOLERANCE)
        assert gradient == pytest.approx(expected_gradient, abs=NM_TOLERANCE)

    def test_the_via_is_no_obstacle_to_itself(self) -> None:
        via = board_via(FakeBoard(vias=[moving_via()]), VIA_ID)

        assert via_obstacles_for_via(via, [via], lambda _: DEFAULT_CLEARANCE) is None
