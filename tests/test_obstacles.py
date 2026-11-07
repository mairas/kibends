import math
from itertools import pairwise
from typing import cast

import kipy.board
import kipy.board_types as kbt
import numpy as np
import pytest
from conftest import (
    PAD_HALF_SIZE,
    FakeBoard,
    kipy_footprint,
    kipy_pad,
    kipy_rect,
    kipy_segment,
    kipy_text,
    kipy_track,
    kipy_via,
    kipy_zone,
    netclass,
)
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.board.board_types_pb2 import PadType
from numpy.typing import NDArray
from shapely.geometry import LineString, Point, Polygon, box

from kibends.energy import DistanceField, EnergyParams, clearance_energy, total_energy
from kibends.obstacles import (
    EDGE_CLEARANCE,
    HOLE_CLEARANCE,
    Obstacle,
    Obstacles,
    build_obstacles,
)
from kibends.snapshot import LayerSnapshot

LAYER = "F.Cu"
NET = "SIG"
WIDTH = 0.25
HALF = WIDTH / 2
DEFAULT_CLEARANCE = 0.2
NM_TOLERANCE = 1e-6
FD_STEP = 1e-6

FloatArray = NDArray[np.float64]


def obstacles_of(
    fake: FakeBoard,
    selected: frozenset[str] = frozenset(),
    **clearances: float,
) -> Obstacles:
    snap = LayerSnapshot.from_board(cast(kipy.board.Board, fake), LAYER, selected)
    return build_obstacles(snap, NET, WIDTH, **clearances)


def by_description(obstacles: Obstacles) -> dict[str, float]:
    return {o.description: o.inflation for o in obstacles.items}


def edge_segments(
    corners: list[tuple[float, float]], closed: bool = True
) -> list[kbt.BoardShape]:
    path = [*corners, corners[0]] if closed else corners
    return [kipy_segment(a, b, layer=BL.BL_Edge_Cuts) for a, b in pairwise(path)]


def points(*xy: tuple[float, float]) -> FloatArray:
    return np.array(xy, dtype=float)


class TestObstacleSet:
    def test_only_the_other_net_pad_inflated_by_the_larger_clearance(self) -> None:
        fake = FakeBoard(
            pads=[
                kipy_pad("same", NET, 0, 0, parent="fp", number="1"),
                kipy_pad("other", "HV", 3, 0, parent="fp", number="2"),
            ],
            footprints=[kipy_footprint("fp", "U1")],
            netclasses={"HV": netclass("HighVoltage", 0.5)},
        )

        assert by_description(obstacles_of(fake)) == {
            "pad U1:2": pytest.approx(0.5 + HALF)
        }

    def test_own_larger_clearance_wins_over_the_pads(self) -> None:
        fake = FakeBoard(
            pads=[kipy_pad("other", "GND", 3, 0)],
            tracks=[kipy_track("sel", NET, (0, 0), (1, 0))],
            netclasses={NET: netclass("Wide", 0.4)},
        )

        assert by_description(obstacles_of(fake, frozenset({"sel"}))) == {
            "pad other": pytest.approx(0.4 + HALF)
        }

    def test_same_net_tht_pad_at_the_endpoint_is_no_obstacle(self) -> None:
        fake = FakeBoard(
            pads=[
                kipy_pad(
                    "tht",
                    NET,
                    0,
                    0,
                    layers=(BL.BL_F_Cu, BL.BL_B_Cu),
                    pad_type=PadType.PT_PTH,
                    drill=1.0,
                )
            ],
            tracks=[kipy_track("sel", NET, (0, 0), (5, 0))],
        )

        assert obstacles_of(fake, frozenset({"sel"})).items == ()

    def test_other_net_tht_pad_and_its_hole(self) -> None:
        fake = FakeBoard(
            pads=[
                kipy_pad(
                    "tht",
                    "GND",
                    0,
                    0,
                    layers=(BL.BL_F_Cu, BL.BL_B_Cu),
                    pad_type=PadType.PT_PTH,
                    drill=1.0,
                )
            ]
        )

        assert by_description(obstacles_of(fake, hole_clearance=0.3)) == {
            "pad tht": pytest.approx(DEFAULT_CLEARANCE + HALF),
            "hole of pad tht": pytest.approx(0.5 + 0.3 + HALF),
        }

    def test_unset_netclass_clearance_uses_the_default(self) -> None:
        fake = FakeBoard(
            pads=[kipy_pad("other", "GND", 3, 0)],
            netclasses={NET: netclass("Bare"), "GND": netclass("AlsoBare")},
            default_clearance=0.15,
        )

        assert by_description(obstacles_of(fake)) == {
            "pad other": pytest.approx(0.15 + HALF)
        }

    def test_pad_clearance_override_larger_than_both_netclasses(self) -> None:
        fake = FakeBoard(
            pads=[kipy_pad("other", "GND", 3, 0, override=0.7)],
            netclasses={"GND": netclass("Power", 0.3)},
        )

        assert by_description(obstacles_of(fake)) == {
            "pad other": pytest.approx(0.7 + HALF)
        }

    def test_npth_mounting_hole_without_net_is_an_obstacle_on_every_layer(
        self,
    ) -> None:
        fake = FakeBoard(
            pads=[
                kipy_pad("mh", "", 5, 5, layers=(), pad_type=PadType.PT_NPTH, drill=3.2)
            ]
        )

        obstacles = obstacles_of(fake)

        assert by_description(obstacles) == {
            "hole of pad mh": pytest.approx(1.6 + HOLE_CLEARANCE + HALF)
        }
        (hole,) = obstacles.items
        assert hole.geometry.equals(Point(5, 5))

    def test_other_net_tracks_and_vias_but_not_same_net_ones(self) -> None:
        fake = FakeBoard(
            tracks=[
                kipy_track("sel", NET, (0, 0), (5, 0)),
                kipy_track("stub", NET, (5, 0), (5, 3)),
                kipy_track("gnd", "GND", (0, 2), (5, 2), width=0.4),
            ],
            vias=[
                kipy_via("v1", NET, 5, 3),
                kipy_via("v2", "GND", 8, 8, diameter=0.8, drill=0.4),
            ],
        )

        assert by_description(obstacles_of(fake, frozenset({"sel"}))) == {
            "track gnd": pytest.approx(0.2 + DEFAULT_CLEARANCE + HALF),
            "via v2": pytest.approx(0.4 + DEFAULT_CLEARANCE + HALF),
            "hole of via v2": pytest.approx(0.2 + HOLE_CLEARANCE + HALF),
        }

    def test_copper_text_of_a_footprint_is_an_obstacle(self) -> None:
        fake = FakeBoard(texts=[kipy_text("LOGO", 10, 10)])

        assert by_description(obstacles_of(fake)) == {
            "copper text 'LOGO'": pytest.approx(DEFAULT_CLEARANCE + HALF)
        }

    def test_copper_shapes_of_other_nets_and_no_net_only(self) -> None:
        fake = FakeBoard(
            shapes=[
                kipy_segment((0, 0), (1, 0), net=NET, shape_id="mine"),
                kipy_segment((0, 1), (1, 1), net="GND", shape_id="gnd"),
                kipy_segment((0, 2), (1, 2), shape_id="loose"),
            ]
        )

        assert set(by_description(obstacles_of(fake))) == {
            "copper shape gnd",
            "copper shape loose",
        }

    def test_track_keepout_is_inflated_by_half_the_width(self) -> None:
        fake = FakeBoard(zones=[kipy_zone(kipy_rect(0, 0, 2, 2), [BL.BL_F_Cu])])

        assert by_description(obstacles_of(fake)) == {
            "track keepout": pytest.approx(HALF)
        }

    def test_copper_zones_are_never_obstacles(self) -> None:
        zone = kipy_zone(kipy_rect(0, 0, 2, 2), [BL.BL_F_Cu], rule_area=False)

        assert obstacles_of(FakeBoard(zones=[zone])).items == ()


class TestBoardEdge:
    def test_point_inside_the_outline_near_the_edge_is_free(self) -> None:
        fake = FakeBoard(shapes=edge_segments([(0, 0), (20, 0), (20, 10), (0, 10)]))

        distance, gradient = obstacles_of(fake).signed_distance(points((1, 5)))

        expected = 1 - EDGE_CLEARANCE - HALF
        assert expected > 0
        assert distance[0] == pytest.approx(expected, abs=NM_TOLERANCE)
        assert gradient[0] == pytest.approx((1, 0))

    def test_edge_clearance_comes_from_the_caller(self) -> None:
        fake = FakeBoard(shapes=edge_segments([(0, 0), (20, 0), (20, 10), (0, 10)]))

        distance, _ = obstacles_of(fake, edge_clearance=0.5).signed_distance(
            points((1, 5))
        )

        assert distance[0] == pytest.approx(1 - 0.5 - HALF, abs=NM_TOLERANCE)

    def test_open_outline_warns_and_adds_no_edge_term(self, log: list[str]) -> None:
        fake = FakeBoard(
            shapes=edge_segments([(0, 0), (20, 0), (20, 10), (0, 10)], closed=False)
        )

        distance, _ = obstacles_of(fake).signed_distance(points((0.1, 5), (-5, 5)))

        assert np.all(np.isinf(distance))
        assert any(line.startswith("WARNING") and "Edge.Cuts" in line for line in log)


SQUARE = Obstacles([Obstacle("square", box(0, 0, 2, 2), 0.5)])


class TestSignedDistance:
    @pytest.mark.parametrize(
        ("point", "distance", "gradient"),
        [
            ((4, 1), 1.5, (1, 0)),
            ((2.5, 1), 0.0, (1, 0)),
            ((2, 1), -0.5, (1, 0)),
            ((0.5, 1), -1.0, (-1, 0)),
            ((1, 1.8), -0.7, (0, 1)),
            ((3, 3), math.sqrt(2) - 0.5, (math.sqrt(0.5), math.sqrt(0.5))),
            ((0, 0), -0.5, (-math.sqrt(0.5), -math.sqrt(0.5))),
        ],
        ids=[
            "outside",
            "on-inflated-boundary",
            "on-raw-boundary",
            "inside-near-left",
            "inside-near-top",
            "outside-corner",
            "on-raw-corner",
        ],
    )
    def test_square(
        self,
        point: tuple[float, float],
        distance: float,
        gradient: tuple[float, float],
    ) -> None:
        d, g = SQUARE.signed_distance(points(point))

        assert d[0] == pytest.approx(distance, abs=NM_TOLERANCE)
        assert g[0] == pytest.approx(gradient, abs=1e-6)

    def test_vectorised_over_many_points(self) -> None:
        d, g = SQUARE.signed_distance(points((4, 1), (0.5, 1), (1, 5)))

        assert d == pytest.approx([1.5, -1.0, 2.5], abs=NM_TOLERANCE)
        assert g.shape == (3, 2)

    def test_round_obstacle_centre_has_a_unit_gradient(self) -> None:
        via = Obstacles([Obstacle("via", Point(1, 1), 0.6)])

        d, g = via.signed_distance(points((1, 1), (2, 1)))

        assert d == pytest.approx([-0.6, 0.4], abs=NM_TOLERANCE)
        assert np.linalg.norm(g, axis=1) == pytest.approx([1, 1])
        assert g[1] == pytest.approx((1, 0))

    def test_point_on_a_track_centreline_gets_its_normal(self) -> None:
        track = Obstacles([Obstacle("track", LineString([(0, 0), (4, 0)]), 0.3)])

        d, g = track.signed_distance(points((2, 0)))

        assert d[0] == pytest.approx(-0.3, abs=NM_TOLERANCE)
        assert abs(g[0][1]) == pytest.approx(1)

    def test_a_farther_obstacle_with_more_clearance_can_be_nearer(self) -> None:
        field = Obstacles(
            [Obstacle("small", Point(0, 0), 0.1), Obstacle("wide", Point(3, 0), 2.5)]
        )

        d, g = field.signed_distance(points((1, 0)))

        assert d[0] == pytest.approx(-0.5, abs=NM_TOLERANCE)
        assert g[0] == pytest.approx((-1, 0))

    def test_overlapping_obstacles_take_the_deepest(self) -> None:
        field = Obstacles(
            [
                Obstacle("a", box(0, 0, 2, 2), 0.0),
                Obstacle("b", box(1, 0, 3, 2), 0.0),
            ]
        )

        d, _ = field.signed_distance(points((1.2, 1)))

        assert d[0] == pytest.approx(-0.8, abs=NM_TOLERANCE)

    def test_no_obstacles_and_no_edge_is_free_everywhere(self) -> None:
        d, g = Obstacles([]).signed_distance(points((0, 0), (5, 5)))

        assert np.all(np.isposinf(d))
        assert np.all(g == 0)

    def test_board_outline_inside_free_outside_and_cutouts_blocked(self) -> None:
        outline = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)], [[(4, 4), (6, 4), (6, 6), (4, 6)]]
        )
        field = Obstacles([], outline, edge_inflation=0.5)

        d, g = field.signed_distance(points((1, 5), (-1, 5), (0, 5), (5, 5), (3, 5)))

        assert d == pytest.approx([0.5, -1.5, -0.5, -1.5, 0.5], abs=NM_TOLERANCE)
        assert g[:3] == pytest.approx(np.array([(1, 0), (1, 0), (1, 0)]))
        assert g[4] == pytest.approx((-1, 0))

    def test_gradient_matches_finite_differences(self) -> None:
        field = Obstacles(
            [
                Obstacle("pad", box(0, 0, 2, 1), 0.3),
                Obstacle("via", Point(4, 3), 0.5),
                Obstacle("track", LineString([(-2, 4), (1, 6)]), 0.25),
            ],
            box(-5, -5, 10, 10),
            edge_inflation=0.4,
        )
        rng = np.random.default_rng(1)
        pts = rng.uniform(-6, 11, size=(200, 2))

        _, analytic = field.signed_distance(pts)
        step = np.eye(2) * FD_STEP
        numeric = np.stack(
            [
                (field.signed_distance(pts + s)[0] - field.signed_distance(pts - s)[0])
                / (2 * FD_STEP)
                for s in step
            ],
            axis=1,
        )

        # Medial axes, where the nearest feature switches, have no gradient.
        agree = np.linalg.norm(analytic - numeric, axis=1) < 1e-4
        assert agree.mean() > 0.95


class TestSegmentsClear:
    def test_segments_against_an_inflated_square(self) -> None:
        starts = points((3, -1), (-1, 1), (2.5, -1), (4, 4), (0, 3), (5, 5))
        ends = points((3, 3), (3, 1), (2.5, 3), (4, 4), (2, 2.4), (5, 5))

        clear = SQUARE.segments_clear(starts, ends)

        assert clear.tolist() == [True, False, True, True, False, True]

    def test_clear_endpoints_with_the_middle_inside(self) -> None:
        starts, ends = points((-1, 1)), points((3, 1))
        d, _ = SQUARE.signed_distance(np.vstack([starts, ends]))

        assert np.all(d > 0)
        assert SQUARE.segments_clear(starts, ends).tolist() == [False]

    def test_segment_crossing_a_track_centreline(self) -> None:
        track = Obstacles([Obstacle("track", LineString([(0, 0), (4, 0)]), 0.3)])

        clear = track.segments_clear(points((2, -1), (2, 1)), points((2, 1), (2, 2)))

        assert clear.tolist() == [False, True]

    def test_segments_against_the_board_edge(self) -> None:
        outline = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)], [[(4, 4), (6, 4), (6, 6), (4, 6)]]
        )
        field = Obstacles([], outline, edge_inflation=0.5)
        starts = points((1, 1), (1, 1), (1, 5), (-3, -3), (1, 1))
        ends = points((9, 1), (11, 1), (9, 5), (-2, -2), (9, 0.4))

        clear = field.segments_clear(starts, ends)

        assert clear.tolist() == [True, False, False, False, False]

    def test_no_obstacles_everything_is_clear(self) -> None:
        clear = Obstacles([]).segments_clear(points((0, 0)), points((1, 1)))

        assert clear.tolist() == [True]

    def test_against_a_board_snapshot(self) -> None:
        fake = FakeBoard(pads=[kipy_pad("p", "GND", 5, 0)])
        field = obstacles_of(fake)
        reach = PAD_HALF_SIZE + DEFAULT_CLEARANCE + HALF

        clear = field.segments_clear(
            points((0, 0), (0, reach + 0.01)), points((10, 0), (10, reach + 0.01))
        )

        assert clear.tolist() == [False, True]


class TestAsDistanceField:
    def test_energy_terms_accept_a_board_snapshot_field(self) -> None:
        fake = FakeBoard(
            pads=[kipy_pad("p", "GND", 5, 0.5)],
            shapes=edge_segments([(-5, -5), (15, -5), (15, 5), (-5, 5)]),
        )
        field: DistanceField = obstacles_of(fake)
        polyline = points((0, 0), (5, 0), (10, 0))

        value, gradient = clearance_energy(
            polyline, np.zeros(3, dtype=bool), field, standoff=0.2, points_per_edge=3
        )

        assert value > 0
        assert np.all(np.isfinite(gradient))

    def test_energy_is_finite_with_no_obstacles_at_all(self) -> None:
        params = EnergyParams(
            stiffness=1.0,
            repulsion=1.0,
            falloff_range=1.0,
            clearance_weight=10.0,
            standoff=0.2,
            points_per_edge=3,
        )
        polyline = points((0, 0), (5, 1), (10, 0))

        value, gradient = total_energy(
            polyline, np.zeros(3, dtype=bool), Obstacles([]), params
        )

        assert np.isfinite(value)
        assert np.all(np.isfinite(gradient))
