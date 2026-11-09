import math
from itertools import pairwise

import kipy.board_types as kbt
import numpy as np
import pytest
import shapely
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
    layer_snapshot,
    netclass,
)
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.board.board_types_pb2 import PadType
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box

from kibends.energy import (
    DistanceField,
    EnergyParams,
    FloatArray,
    ObstacleField,
    clearance_energy,
    total_energy,
)
from kibends.obstacles import (
    EDGE_CLEARANCE,
    HOLE_CLEARANCE,
    ChainCentreline,
    CompositeField,
    Obstacle,
    Obstacles,
    build_obstacles,
    build_pair_obstacles,
    chain_obstacles,
)

LAYER = "F.Cu"
NET = "SIG"
WIDTH = 0.25
HALF = WIDTH / 2
DEFAULT_CLEARANCE = 0.2
NM_TOLERANCE = 1e-6
FD_STEP = 1e-6


def obstacles_of(
    fake: FakeBoard,
    replaced: frozenset[str] = frozenset(),
    **clearances: float,
) -> Obstacles:
    snap = layer_snapshot(fake, LAYER, replaced)
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

    def test_an_unplated_hole_of_its_own_net_is_an_obstacle(self) -> None:
        fake = FakeBoard(
            pads=[
                kipy_pad("np", NET, 5, 5, layers=(), pad_type=PadType.PT_NPTH, drill=1)
            ]
        )

        assert by_description(obstacles_of(fake)) == {
            "hole of pad np": pytest.approx(0.5 + HOLE_CLEARANCE + HALF)
        }

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

    def test_a_via_keepout_is_no_obstacle_to_a_track(self) -> None:
        keepout = kipy_zone(
            kipy_rect(0, 0, 2, 2), [BL.BL_F_Cu], keepout_tracks=False, keepout_vias=True
        )

        assert obstacles_of(FakeBoard(zones=[keepout])).items == ()

    def test_copper_zones_are_never_obstacles(self) -> None:
        zone = kipy_zone(kipy_rect(0, 0, 2, 2), [BL.BL_F_Cu], rule_area=False)

        assert obstacles_of(FakeBoard(zones=[zone])).items == ()


PAIR_GAP = 0.15
PAIR_HALF_EXTENT = WIDTH + PAIR_GAP / 2
"""From a pair's centreline to the outer edge of either track."""


def pair_track(net: str, y: float) -> ChainCentreline:
    """A straight section track from x = 0 to 5."""
    return ChainCentreline(net, net, WIDTH, DEFAULT_CLEARANCE, points((0, y), (5, y)))


PAIR_TRACKS: list[kbt.Track | kbt.ArcTrack] = [
    kipy_track("P", "P", (0, 0), (5, 0)),
    kipy_track("N", "N", (0, WIDTH + PAIR_GAP), (5, WIDTH + PAIR_GAP)),
    kipy_track("P fan-out", "P", (5, 0), (6, -1)),
]


def pair_obstacles_of(fake: FakeBoard, **clearances: float) -> Obstacles:
    snap = layer_snapshot(fake, LAYER, frozenset({"P", "N", "P fan-out"}))
    pitch = WIDTH + PAIR_GAP
    return build_pair_obstacles(
        snap, pair_track("P", 0.0), pair_track("N", pitch), PAIR_GAP, **clearances
    )


class TestPairObstacleSet:
    def test_other_net_copper_keeps_the_larger_clearance_from_the_outer_edge(
        self,
    ) -> None:
        fake = FakeBoard(
            pads=[kipy_pad("hv", "HV", 2, 3), kipy_pad("gnd", "GND", 4, 3)],
            tracks=PAIR_TRACKS,
            netclasses={"HV": netclass("HighVoltage", 0.5), "N": netclass("DP", 0.3)},
        )

        assert by_description(pair_obstacles_of(fake)) == {
            "pad hv": pytest.approx(0.5 + PAIR_HALF_EXTENT),
            "pad gnd": pytest.approx(0.3 + PAIR_HALF_EXTENT),
        }

    def test_own_nets_copper_is_an_obstacle_unless_the_pair_connects_to_it(
        self,
    ) -> None:
        fake = FakeBoard(
            pads=[
                kipy_pad("p end", "P", 0, 0, pad_type=PadType.PT_PTH, drill=0.3),
                kipy_pad("n end", "N", 5, WIDTH + PAIR_GAP),
                kipy_pad("p loose", "P", 2, 2, pad_type=PadType.PT_PTH, drill=0.3),
            ],
            tracks=[
                *PAIR_TRACKS,
                kipy_track("n stub", "N", (0, WIDTH + PAIR_GAP), (0, 2)),
                kipy_track("n loose", "N", (3, 2), (4, 2)),
            ],
            vias=[kipy_via("p via", "P", 5, 0), kipy_via("n via", "N", 1, -2)],
        )

        assert by_description(pair_obstacles_of(fake, hole_clearance=0.3)) == {
            "pad p loose": pytest.approx(DEFAULT_CLEARANCE + PAIR_HALF_EXTENT),
            "hole of pad p loose": pytest.approx(0.15 + 0.3 + PAIR_HALF_EXTENT),
            "track n loose": pytest.approx(HALF + DEFAULT_CLEARANCE + PAIR_HALF_EXTENT),
            "via n via": pytest.approx(0.3 + DEFAULT_CLEARANCE + PAIR_HALF_EXTENT),
            "hole of via n via": pytest.approx(0.15 + 0.3 + PAIR_HALF_EXTENT),
        }

    def test_copper_a_fan_out_connects_to_stays_an_obstacle(self) -> None:
        fake = FakeBoard(pads=[kipy_pad("p fan-out", "P", 6, -1)], tracks=PAIR_TRACKS)

        obstacles = pair_obstacles_of(fake)

        assert by_description(obstacles) == {
            "pad p fan-out": pytest.approx(DEFAULT_CLEARANCE + PAIR_HALF_EXTENT)
        }

    def test_board_edge_and_keepouts_keep_from_the_outer_edge(self) -> None:
        fake = FakeBoard(
            zones=[kipy_zone(kipy_rect(0, 3, 2, 4), [BL.BL_F_Cu])],
            shapes=edge_segments([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
        )

        obstacles = pair_obstacles_of(fake, edge_clearance=0.4)

        assert by_description(obstacles) == {
            "track keepout": pytest.approx(PAIR_HALF_EXTENT)
        }
        inside = np.array([[10.0 - 0.4 - PAIR_HALF_EXTENT, 0.0]])
        assert obstacles.signed_distance(inside)[0][0] == pytest.approx(0.0, abs=1e-9)


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

    REACH = 1.0
    MIXED = Obstacles(
        [
            Obstacle("square", box(0, 0, 2, 2), 0.5),
            Obstacle("via", Point(6, 1), 0.3),
            Obstacle("track", LineString([(0, 5), (6, 5)]), 0.2),
        ],
        Polygon([(-3, -3), (9, -3), (9, 8), (-3, 8)]),
        edge_inflation=0.3,
    )

    def test_within_reach_matches_the_exact_distance(self) -> None:
        rng = np.random.default_rng(1)
        probe = rng.uniform((-4, -4), (10, 9), size=(400, 2))
        exact_d, exact_g = self.MIXED.signed_distance(probe)
        near = exact_d <= self.REACH

        d, g = self.MIXED.signed_distance(probe, reach=self.REACH)

        assert near.sum() > 50
        assert d[near] == pytest.approx(exact_d[near], abs=NM_TOLERANCE)
        assert g[near] == pytest.approx(exact_g[near], abs=NM_TOLERANCE)

    def test_beyond_reach_may_be_infinite_with_no_gradient(self) -> None:
        far = points((4, 2.8), (3, 6.6))
        exact_d, _ = self.MIXED.signed_distance(far)

        d, g = self.MIXED.signed_distance(far, reach=self.REACH)

        assert np.all(exact_d > self.REACH)
        assert np.all(np.isposinf(d) | (d == pytest.approx(exact_d)))
        assert np.all(g[np.isposinf(d)] == 0)

    def test_outside_the_board_is_reported_whatever_the_reach(self) -> None:
        d, _ = self.MIXED.signed_distance(points((20, 20)), reach=self.REACH)

        assert d[0] < 0

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


OTHER_NET = "OTHER"
OTHER_WIDTH = 0.3
OTHER_CLEARANCE = 0.3
PAIR_HALF_WIDTHS = HALF + OTHER_WIDTH / 2
OTHER_NET_INFLATION = OTHER_CLEARANCE + PAIR_HALF_WIDTHS


def other_chain(net: str, *xy: tuple[float, float]) -> ChainCentreline:
    return ChainCentreline(
        f"chain {net}", net, OTHER_WIDTH, OTHER_CLEARANCE, points(*xy)
    )


def neighbours(*others: ChainCentreline) -> ObstacleField:
    return chain_obstacles(NET, WIDTH, DEFAULT_CLEARANCE, others)


BELOW = other_chain(OTHER_NET, (0, 0), (10, 0))
ABOVE = other_chain("THIRD", (0, 2), (10, 2))


class TestChainObstacles:
    def test_a_point_between_two_chains_sees_the_nearer_one(self) -> None:
        d, g = neighbours(BELOW, ABOVE).signed_distance(points((5, 0.8), (5, 1.5)))

        assert d == pytest.approx(
            [0.8 - OTHER_NET_INFLATION, 0.5 - OTHER_NET_INFLATION], abs=NM_TOLERANCE
        )
        assert g[0] == pytest.approx((0, 1), abs=1e-9)
        assert g[1] == pytest.approx((0, -1), abs=1e-9)

    def test_chains_of_different_vertex_counts(self) -> None:
        bent = other_chain("THIRD", (0, 2), (5, 2), (10, 2))

        d, _ = neighbours(BELOW, bent).signed_distance(points((5, 0.8)))

        assert d[0] == pytest.approx(0.8 - OTHER_NET_INFLATION, abs=NM_TOLERANCE)

    def test_a_long_chain_has_no_gap_between_its_pieces(self) -> None:
        # At 1 mm spacing a missing edge leaves its midpoint 0.5 mm from the
        # nearest vertex, beyond the same-net band of both half widths.
        xs = np.arange(41.0)
        long_chain = ChainCentreline(
            "long",
            NET,
            OTHER_WIDTH,
            OTHER_CLEARANCE,
            np.column_stack([xs, np.zeros_like(xs)]),
        )
        midpoints = xs[:-1] + 0.5
        height = PAIR_HALF_WIDTHS / 2
        field = neighbours(long_chain)

        d, _ = field.signed_distance(
            np.column_stack([midpoints, np.full_like(midpoints, height)])
        )
        clear = field.segments_clear(
            np.column_stack([midpoints, np.ones_like(midpoints)]),
            np.column_stack([midpoints, -np.ones_like(midpoints)]),
        )

        assert d == pytest.approx(
            np.full(len(midpoints), height - PAIR_HALF_WIDTHS), abs=NM_TOLERANCE
        )
        assert not clear.any()

    def test_the_larger_clearance_of_the_pair_wins(self) -> None:
        field = chain_obstacles(NET, WIDTH, 0.5, [BELOW])

        d, _ = field.signed_distance(points((5, 1)))

        assert d[0] == pytest.approx(1 - 0.5 - PAIR_HALF_WIDTHS, abs=NM_TOLERANCE)

    def test_a_same_net_chain_is_inflated_by_both_half_widths_only(self) -> None:
        between = PAIR_HALF_WIDTHS + OTHER_CLEARANCE / 2
        probe = points((5, between))
        same = neighbours(other_chain(NET, (0, 0), (10, 0)))

        d_same, _ = same.signed_distance(probe)
        d_other, _ = neighbours(BELOW).signed_distance(probe)

        assert d_same[0] == pytest.approx(between - PAIR_HALF_WIDTHS, abs=NM_TOLERANCE)
        assert d_same[0] > 0
        assert d_other[0] < 0

    def test_gradient_points_away_from_the_nearest_chain(self) -> None:
        diagonal = neighbours(other_chain(OTHER_NET, (0, 0), (4, 4)))

        _, g = diagonal.signed_distance(points((1, 3)))

        assert g[0] == pytest.approx((-math.sqrt(0.5), math.sqrt(0.5)), abs=1e-9)

    def test_segments_crossing_the_band_are_not_clear(self) -> None:
        inside = OTHER_NET_INFLATION / 2
        outside = OTHER_NET_INFLATION * 2

        clear = neighbours(BELOW).segments_clear(
            points((5, 1), (1, outside), (1, 1)),
            points((5, -1), (9, outside), (3, inside)),
        )

        assert clear.tolist() == [False, True, False]

    def test_segments_touching_the_band_are_clear(self) -> None:
        """A pair's fan-outs meeting its coupled tracks at a clearance equal
        to its gap: each segment ends exactly at clearance from the other."""
        width, gap = 0.2, 0.15
        half_pitch = (width + gap) / 2
        other = ChainCentreline(
            "chain P",
            "P",
            width,
            gap,
            points((-1, 1 + half_pitch), (0, half_pitch), (5, half_pitch)),
        )
        steps = np.arange(1, 200, dtype=float) * 1e-3
        starts: FloatArray = np.column_stack([-steps, -half_pitch - steps])
        ends = np.tile((0.0, -half_pitch), (len(steps), 1))

        clear = chain_obstacles("N", width, gap, [other]).segments_clear(starts, ends)

        assert clear.all()

    def test_no_other_chains_is_free_everywhere(self) -> None:
        empty = neighbours()

        d, g = empty.signed_distance(points((0, 0), (5, 5)))

        assert np.all(np.isinf(d))
        assert np.all(g == 0)
        assert empty.segments_clear(points((0, 0)), points((5, 5))).tolist() == [True]

    def test_a_zero_length_edge_blocks_segments_around_it(self) -> None:
        dot = neighbours(other_chain(OTHER_NET, (5, 0), (5, 0)))

        clear = dot.segments_clear(points((4, 0.1), (4, 5)), points((6, 0.1), (6, 5)))

        assert clear.tolist() == [False, True]

    def test_a_zero_length_segment_inside_the_band_is_not_clear(self) -> None:
        assert neighbours(BELOW).segments_clear(
            points((5, 0)), points((5, 0))
        ).tolist() == [False]

    def test_a_point_on_a_centreline_gets_its_normal(self) -> None:
        d, g = neighbours(BELOW).signed_distance(points((5, 0)))

        assert d[0] == pytest.approx(-OTHER_NET_INFLATION, abs=NM_TOLERANCE)
        assert g[0] == pytest.approx((0, 1), abs=1e-9)


def random_chains(rng: np.random.Generator) -> list[ChainCentreline]:
    """Random walks of ragged length on both nets, with edges from much
    shorter to much longer than the resampling spacing and one repeated
    vertex."""
    chains: list[ChainCentreline] = []
    for i, net in enumerate([NET, OTHER_NET, "THIRD", NET]):
        count = int(rng.integers(2, 30))
        heading = rng.uniform(0, 2 * math.pi) + np.cumsum(rng.normal(0, 0.6, count - 1))
        steps = rng.uniform(0.05, 1.5) * rng.uniform(0.2, 1.0, (count - 1, 1))
        offsets = steps * np.column_stack([np.cos(heading), np.sin(heading)])
        vertices = rng.uniform(0, 3, 2) + np.vstack([(0, 0), np.cumsum(offsets, 0)])
        if i == 0:
            vertices = np.insert(vertices, 1, vertices[1], axis=0)
        chains.append(
            ChainCentreline(
                f"chain {i}",
                net,
                float(rng.uniform(0.1, 0.5)),
                float(rng.uniform(0.1, 0.4)),
                vertices,
            )
        )
    return chains


def shapely_chain_field(
    net: str, width: float, clearance: float, others: list[ChainCentreline]
) -> Obstacles:
    """The same field as whole-chain centrelines in a shapely `Obstacles`."""
    return Obstacles(
        [
            Obstacle(
                o.description,
                LineString(o.points),
                width / 2
                + o.width / 2
                + (max(clearance, o.clearance) if o.net != net else 0.0),
            )
            for o in others
        ]
    )


def probes_around(
    chains: list[ChainCentreline], rng: np.random.Generator
) -> FloatArray:
    """Points at random, and near the vertices and edge midpoints of chains."""
    vertices = np.vstack([c.points for c in chains])
    midpoints = np.vstack([(c.points[1:] + c.points[:-1]) / 2 for c in chains])
    anchors = np.vstack([vertices, midpoints])
    angle = rng.uniform(0, 2 * math.pi, len(anchors))
    radius = rng.uniform(0, 1.5, (len(anchors), 1))
    near = anchors + radius * np.column_stack([np.cos(angle), np.sin(angle)])
    low, high = vertices.min(0) - 2, vertices.max(0) + 2
    return np.vstack([near, rng.uniform(low, high, (200, 2))])


# Closer than this to a centreline, the gradient is the direction of a
# micrometre offset and only agrees to within rounding of the nearest point.
GRADIENT_COMPARISON_MIN_DISTANCE = 1e-3


class TestChainFieldMatchesShapely:
    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("reach", [math.inf, 1.0, 0.2])
    def test_signed_distance(self, seed: int, reach: float) -> None:
        rng = np.random.default_rng(seed)
        chains = random_chains(rng)
        probes = probes_around(chains, rng)
        reference = shapely_chain_field(NET, WIDTH, DEFAULT_CLEARANCE, chains)

        d, g = chain_obstacles(NET, WIDTH, DEFAULT_CLEARANCE, chains).signed_distance(
            probes, reach
        )
        d_ref, g_ref = reference.signed_distance(probes, reach)

        np.testing.assert_array_equal(np.isinf(d), np.isinf(d_ref))
        finite = np.isfinite(d_ref)
        np.testing.assert_allclose(d[finite], d_ref[finite], rtol=0, atol=1e-9)
        centrelines = MultiLineString([c.points for c in chains])
        away = finite & (
            centrelines.distance(shapely.points(probes))
            > GRADIENT_COMPARISON_MIN_DISTANCE
        )
        np.testing.assert_allclose(g[away], g_ref[away], rtol=0, atol=1e-9)
        np.testing.assert_array_equal(g[~finite], 0.0)

    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("tolerance", [0.0, 0.05])
    def test_segments_clear(self, seed: int, tolerance: float) -> None:
        rng = np.random.default_rng(seed)
        chains = random_chains(rng)
        starts = probes_around(chains, rng)
        angle = rng.uniform(0, 2 * math.pi, len(starts))
        length = rng.uniform(0, 3, (len(starts), 1))
        ends = starts + length * np.column_stack([np.cos(angle), np.sin(angle)])
        reference = shapely_chain_field(NET, WIDTH, DEFAULT_CLEARANCE, chains)

        clear = chain_obstacles(NET, WIDTH, DEFAULT_CLEARANCE, chains).segments_clear(
            starts, ends, tolerance
        )

        np.testing.assert_array_equal(
            clear, reference.segments_clear(starts, ends, tolerance)
        )
        assert 0 < clear.sum() < len(clear)


SPREAD = points((4, 1), (0.5, 1), (1, 5), (2.5, 1), (0, 0), (-3, 2), (12, 0.1))


class TestCompositeField:
    def test_satisfies_the_optimizer_protocol(self) -> None:
        field: ObstacleField = CompositeField([SQUARE])

        assert field.segments_clear(points((4, 4)), points((4, 4))).tolist() == [True]

    @pytest.mark.parametrize("reach", [math.inf, 1.0])
    def test_composite_of_one_field_is_that_field(self, reach: float) -> None:
        d, g = CompositeField([SQUARE]).signed_distance(SPREAD, reach)
        d_square, g_square = SQUARE.signed_distance(SPREAD, reach)

        np.testing.assert_array_equal(d, d_square)
        np.testing.assert_array_equal(g, g_square)

    @pytest.mark.parametrize("reach", [math.inf, 1.0])
    def test_an_empty_dynamic_field_adds_nothing(self, reach: float) -> None:
        d, g = CompositeField([SQUARE, neighbours()]).signed_distance(SPREAD, reach)
        d_square, g_square = SQUARE.signed_distance(SPREAD, reach)

        np.testing.assert_array_equal(d, d_square)
        np.testing.assert_array_equal(g, g_square)

    def test_takes_the_smallest_distance_with_its_gradient(self) -> None:
        composite = CompositeField(
            [SQUARE, neighbours(other_chain(OTHER_NET, (8, -5), (8, 5)))]
        )

        d, g = composite.signed_distance(points((4, 1), (7, 1)))

        assert d == pytest.approx([1.5, 1 - OTHER_NET_INFLATION], abs=NM_TOLERANCE)
        assert g[0] == pytest.approx((1, 0), abs=1e-9)
        assert g[1] == pytest.approx((-1, 0), abs=1e-9)

    def test_beyond_reach_of_every_field_is_infinite(self) -> None:
        composite = CompositeField([SQUARE, neighbours(BELOW)])

        d, g = composite.signed_distance(points((5, 30)), reach=1.0)

        assert np.isinf(d[0])
        assert g[0] == pytest.approx((0, 0))

    def test_no_fields_is_free_everywhere(self) -> None:
        d, g = CompositeField([]).signed_distance(SPREAD)

        assert np.all(np.isinf(d))
        assert np.all(g == 0)

    def test_a_segment_is_clear_only_if_clear_in_every_field(self) -> None:
        composite = CompositeField([SQUARE, neighbours(BELOW)])

        clear = composite.segments_clear(
            points((-1, 1), (5, 1), (4, 4)), points((3, 1), (5, -1), (6, 4))
        )

        assert clear.tolist() == [False, False, True]
