import math

import numpy as np
import pytest
from distance_fields import FreeSpace
from shapely.geometry import LineString

from kibends.chain import Chain, Point2
from kibends.energy import BoolArray, FloatArray
from kibends.pairs import (
    Pair,
    PairError,
    Piece,
    bend_radius_limit,
    centreline,
    check_bend_radius,
    complementary_net,
    find_pair,
    gap_range,
    nets_pair,
    offset_pair,
)
from kibends.simplify import DEFAULT_CHORD_TOLERANCE, simplify

WIDTH = 0.2
GAP = 0.15
HALF_PITCH = (GAP + WIDTH) / 2
ROUND_TRIP_TOLERANCE = 1e-9
# Hand-routed coordinates are rounded to KiCad's 1 nm grid.
NM_ROUNDING_TOLERANCE = 1e-5
CHORD = DEFAULT_CHORD_TOLERANCE
TAN_HALF_45 = math.tan(math.pi / 8)


def chain(net: str, *points: Point2, width: float = WIDTH) -> Chain:
    return Chain(
        points=points,
        pinned=(True,) + (False,) * (len(points) - 2) + (True,),
        width=width,
        layer="F.Cu",
        net=net,
        segment_ids=(),
    )


def array(*points: Point2) -> FloatArray:
    return np.array(points, dtype=float)


def fanned_out_pair() -> tuple[Chain, Chain]:
    """Coupled from x = 1 to x = 9, fanned out at 45° to pads at both ends."""
    fan = 1.0 + HALF_PITCH
    a = chain(
        "USB_D+",
        (0.0, -fan),
        (1.0, -HALF_PITCH),
        (9.0, -HALF_PITCH),
        (10.0, -fan),
    )
    b = chain(
        "USB_D-",
        (0.0, fan),
        (1.0, HALF_PITCH),
        (9.0, HALF_PITCH),
        (10.0, fan),
    )
    return a, b


def corner_pair() -> tuple[FloatArray, FloatArray]:
    """Two 45° corners, drawn as offsets of the centreline
    (0, 0) → (5, 0) → (8, 3) → (13, 3)."""
    shift = TAN_HALF_45 * HALF_PITCH
    a = array(
        (0.0, HALF_PITCH),
        (5.0 - shift, HALF_PITCH),
        (8.0 - shift, 3.0 + HALF_PITCH),
        (13.0, 3.0 + HALF_PITCH),
    )
    b = array(
        (0.0, -HALF_PITCH),
        (5.0 + shift, -HALF_PITCH),
        (8.0 + shift, 3.0 - HALF_PITCH),
        (13.0, 3.0 - HALF_PITCH),
    )
    return a, b


def free_ends(n: int) -> BoolArray:
    fixed = np.zeros(n, dtype=bool)
    fixed[[0, -1]] = True
    return fixed


def arc(radius: float, points: int = 181) -> FloatArray:
    angles = np.linspace(0.0, np.pi / 2, points)
    return radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def hausdorff(a: FloatArray, b: FloatArray) -> float:
    return LineString(a).hausdorff_distance(LineString(b))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("USB_D+", "USB_D-"),
        ("LVDS_P", "LVDS_N"),
        ("CLK_P", "CLK_N"),
        ("/bus/TX2P", "/bus/TX2N"),
        ("DP_P_0", "DP_N_0"),
    ],
)
def test_kicad_suffixes_pair(first: str, second: str) -> None:
    assert nets_pair(first, second)
    assert nets_pair(second, first)


@pytest.mark.parametrize(
    ("first", "second"),
    [("A", "B"), ("SIG+", "OTHER-"), ("USB_D+", "USB_D+"), ("clk_p", "clk_n")],
)
def test_other_names_do_not_pair(first: str, second: str) -> None:
    assert not nets_pair(first, second)


def test_name_without_pair_suffix_has_no_complement() -> None:
    assert complementary_net("GND") is None


def only_section(pair: Pair | None) -> tuple[FloatArray, FloatArray]:
    assert pair is not None
    assert len(pair.sections) == 1
    return pair.coupled(pair.sections[0])


def from_array(net: str, points: FloatArray) -> Chain:
    return chain(net, *((x, y) for x, y in points.tolist()))


def test_fanned_out_pair_splits_into_coupled_middle_and_four_ends() -> None:
    a, b = fanned_out_pair()

    pair = find_pair(a, b, CHORD)

    coupled_a, coupled_b = only_section(pair)
    assert pair is not None
    np.testing.assert_allclose(coupled_a, [(1.0, -HALF_PITCH), (9.0, -HALF_PITCH)])
    np.testing.assert_allclose(coupled_b, [(1.0, HALF_PITCH), (9.0, HALF_PITCH)])
    fan = 1.0 + HALF_PITCH
    ends_a, ends_b = pair.uncoupled()
    assert ends_a == [Piece(0, 1, None, 0), Piece(2, 3, 0, None)]
    assert ends_b == [Piece(0, 1, None, 0), Piece(2, 3, 0, None)]
    np.testing.assert_allclose(pair.a.between(0, 1), [(0.0, -fan), (1.0, -HALF_PITCH)])
    np.testing.assert_allclose(pair.b.between(2, 3), [(9.0, HALF_PITCH), (10.0, fan)])
    assert pair.gap == pytest.approx(GAP)


def test_coupled_end_on_a_straight_track_is_opposite_the_fan_out() -> None:
    _, b = fanned_out_pair()
    a = chain("USB_D+", (0.0, -HALF_PITCH), (10.0, -HALF_PITCH))

    pair = find_pair(a, b, CHORD)

    coupled_a, _ = only_section(pair)
    assert pair is not None
    np.testing.assert_allclose(coupled_a, [(1.0, -HALF_PITCH), (9.0, -HALF_PITCH)])
    head = pair.uncoupled()[0][0]
    np.testing.assert_allclose(
        pair.a.between(head.start, head.end), [(0.0, -HALF_PITCH), (1.0, -HALF_PITCH)]
    )
    assert pair.a.pinned.tolist() == [True, False, False, True]


def test_chains_running_opposite_ways_still_pair() -> None:
    a, b = fanned_out_pair()
    reversed_b = chain("USB_D-", *reversed(b.points))

    coupled_a, coupled_b = only_section(find_pair(a, reversed_b, CHORD))

    np.testing.assert_allclose(coupled_b, [(9.0, HALF_PITCH), (1.0, HALF_PITCH)])
    middle = centreline(coupled_a, coupled_b, WIDTH).points
    np.testing.assert_allclose(middle[[0, -1]], [(1.0, 0.0), (9.0, 0.0)])
    np.testing.assert_allclose(middle[:, 1], 0.0, atol=ROUND_TRIP_TOLERANCE)


def split_pair() -> tuple[Chain, Chain]:
    """Coupled for 5 mm, apart around a 1 mm jog, then coupled for 3 mm."""
    away = HALF_PITCH + 1.0
    a = chain(
        "LVDS_P",
        (0.0, -HALF_PITCH),
        (5.0, -HALF_PITCH),
        (6.0, -away),
        (7.0, -away),
        (8.0, -HALF_PITCH),
        (11.0, -HALF_PITCH),
    )
    b = chain("LVDS_N", (0.0, HALF_PITCH), (11.0, HALF_PITCH))
    return a, b


def test_pair_split_apart_gives_two_sections() -> None:
    a, b = split_pair()

    pair = find_pair(a, b, CHORD)

    assert pair is not None
    assert len(pair.sections) == 2
    first, second = (pair.coupled(s) for s in pair.sections)
    np.testing.assert_allclose(first[0], [(0.0, -HALF_PITCH), (5.0, -HALF_PITCH)])
    np.testing.assert_allclose(first[1], [(0.0, HALF_PITCH), (5.0, HALF_PITCH)])
    np.testing.assert_allclose(second[0], [(8.0, -HALF_PITCH), (11.0, -HALF_PITCH)])
    np.testing.assert_allclose(second[1], [(8.0, HALF_PITCH), (11.0, HALF_PITCH)])
    pieces_a, pieces_b = pair.uncoupled()
    assert pieces_a == [Piece(0, 0, None, 0), Piece(1, 4, 0, 1), Piece(5, 5, 1, None)]
    assert pieces_b == [Piece(0, 0, None, 0), Piece(1, 2, 0, 1), Piece(3, 3, 1, None)]
    np.testing.assert_allclose(
        pair.b.between(1, 2), [(5.0, HALF_PITCH), (8.0, HALF_PITCH)]
    )


def test_split_pair_running_opposite_ways_tags_b_pieces_by_section() -> None:
    a, b = split_pair()
    reversed_b = chain("LVDS_N", *reversed(b.points))

    pair = find_pair(a, reversed_b, CHORD)

    assert pair is not None
    np.testing.assert_allclose(
        pair.b.points,
        [(11.0, HALF_PITCH), (8.0, HALF_PITCH), (5.0, HALF_PITCH), (0.0, HALF_PITCH)],
    )
    assert pair.sections[0][2:5] == (2, 3, True)
    assert pair.sections[1][2:5] == (0, 1, True)
    _, pieces_b = pair.uncoupled()
    assert pieces_b == [Piece(0, 0, None, 1), Piece(1, 2, 1, 0), Piece(3, 3, 0, None)]
    assert pair.b.pinned.tolist() == [True, False, False, True]


def test_sections_either_side_of_a_short_bump_stay_disjoint() -> None:
    """Both tracks bulge apart for 0.2 mm, within a pitch of both sections."""
    bulge = HALF_PITCH + 0.1
    a = chain(
        "LVDS_P",
        (0.0, -HALF_PITCH),
        (5.0, -HALF_PITCH),
        (5.1, -bulge),
        (5.2, -HALF_PITCH),
        (11.0, -HALF_PITCH),
    )
    b = chain(
        "LVDS_N",
        (0.0, HALF_PITCH),
        (5.0, HALF_PITCH),
        (5.1, bulge),
        (5.2, HALF_PITCH),
        (11.0, HALF_PITCH),
    )

    pair = find_pair(a, b, CHORD)

    assert pair is not None
    first, second = pair.sections
    assert first.a_end <= second.a_start
    assert first.b_end <= second.b_start


def test_short_coincidental_parallel_stretch_is_no_pair() -> None:
    a = chain(
        "LVDS_P", (0.0, -3.0), (2.0, -HALF_PITCH), (2.5, -HALF_PITCH), (4.5, -3.0)
    )
    b = chain("LVDS_N", (0.0, 3.0), (2.0, HALF_PITCH), (2.5, HALF_PITCH), (4.5, 3.0))

    assert find_pair(a, b, CHORD) is None


def test_stretch_at_another_spacing_does_not_set_the_pitch() -> None:
    """6 mm at the pair's pitch, then 2 mm with the gap widened to 0.5 mm."""
    jog = HALF_PITCH + 0.35
    a = chain(
        "LVDS_P",
        (0.0, -HALF_PITCH),
        (6.0, -HALF_PITCH),
        (6.35, -jog),
        (8.35, -jog),
    )
    b = chain("LVDS_N", (0.0, HALF_PITCH), (8.35, HALF_PITCH))

    pair = find_pair(a, b, CHORD)

    coupled_a, coupled_b = only_section(pair)
    assert pair is not None
    np.testing.assert_allclose(coupled_a, [(0.0, -HALF_PITCH), (6.0, -HALF_PITCH)])
    np.testing.assert_allclose(coupled_b, [(0.0, HALF_PITCH), (6.0, HALF_PITCH)])
    assert pair.gap == pytest.approx(GAP)


def test_u_turn_is_one_coupled_section() -> None:
    angles = np.linspace(-np.pi / 2, np.pi / 2, 91)
    turn = np.stack([4.0 + np.cos(angles), np.sin(angles)], axis=1)
    path = np.concatenate([array((0.0, -1.0)), turn, array((0.0, 1.0))])
    right, left = offset_pair(path, GAP, WIDTH)

    pair = find_pair(from_array("CLK_P", right), from_array("CLK_N", left), CHORD)

    coupled_a, coupled_b = only_section(pair)
    assert pair is not None
    assert len(coupled_a) == len(right)
    assert len(coupled_b) == len(left)
    assert pair.gap == pytest.approx(GAP, abs=1e-6)


LEG_45 = 5.0 / math.sqrt(2)
SHARP_TURNS = {
    "90°": array((0.0, 0.0), (5.0, 0.0), (5.0, 5.0)),
    "135°": array(
        (0.0, 0.0), (5.0, 0.0), (5.0 - 5.0 / math.sqrt(2), 5.0 / math.sqrt(2))
    ),
    "1 mm hairpin": array((0.0, 0.0), (5.0, 0.0), (5.0, 1.0), (0.0, 1.0)),
}


@pytest.mark.parametrize("turn", SHARP_TURNS)
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("reverse_b", [False, True])
def test_hand_routed_sharp_turn_is_one_section_end_to_end(
    turn: str, swap: bool, reverse_b: bool
) -> None:
    right, left = offset_pair(SHARP_TURNS[turn], GAP, WIDTH)
    a, b = (left, right) if swap else (right, left)
    b = b[::-1] if reverse_b else b

    pair = find_pair(from_array("CLK_P", a), from_array("CLK_N", b), CHORD)

    coupled_a, coupled_b = only_section(pair)
    np.testing.assert_allclose(coupled_a, a)
    np.testing.assert_allclose(coupled_b, b)


def test_nm_rounded_pair_gets_no_sliver_vertices_without_chord_tolerance() -> None:
    turn = np.radians(30.0)
    rotation = np.array([[np.cos(turn), np.sin(turn)], [-np.sin(turn), np.cos(turn)]])
    a, b = (np.round(track @ rotation, 6) for track in corner_pair())

    pair = find_pair(from_array("CLK_P", a), from_array("CLK_N", b), 0.0)

    assert pair is not None
    assert len(pair.a.points) == len(a)
    assert len(pair.b.points) == len(b)


def test_densely_drawn_pair_is_coupled_from_end_to_end() -> None:
    x = np.linspace(0.0, 10.0, 201)
    a = np.stack([x, np.full_like(x, -HALF_PITCH)], axis=1)
    b = np.stack([x, np.full_like(x, HALF_PITCH)], axis=1)

    pair = find_pair(from_array("CLK_P", a), from_array("CLK_N", b), CHORD)

    assert pair is not None
    assert pair.sections[0][:4] == (0, 200, 0, 200)


def test_coupled_end_is_found_with_a_wide_chord_tolerance() -> None:
    """A wide band lets the straight track's run overshoot the fan-out by
    more than a pitch; the other track's run end still places the end."""
    _, b = fanned_out_pair()
    a = chain("USB_D+", (0.0, -HALF_PITCH), (10.0, -HALF_PITCH))

    coupled_a, _ = only_section(find_pair(a, b, 0.2))

    np.testing.assert_allclose(coupled_a, [(1.0, -HALF_PITCH), (9.0, -HALF_PITCH)])


def test_pair_coupled_only_on_the_outside_of_a_tight_turn_is_no_pair() -> None:
    """Around a 180° turn of radius 0.5 the outer track runs 2.1 mm coupled,
    the inner only 1.0 mm, short of the 1.75 mm minimum."""
    angles = np.linspace(-np.pi / 2, np.pi / 2, 61)
    turn = 0.5 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    inner, outer = offset_pair(turn, GAP, WIDTH)
    inner = np.concatenate(
        [array((-1.0, inner[0, 1])), inner, array((-1.0, inner[-1, 1]))]
    )
    outer = np.concatenate(
        [
            outer[:1] + array((-1.0, -1.0)),
            outer,
            outer[-1:] + array((-1.0, 1.0)),
        ]
    )

    assert (
        find_pair(from_array("CLK_P", inner), from_array("CLK_N", outer), CHORD) is None
    )


def test_gap_widened_by_a_third_ends_the_section() -> None:
    """6 mm at a 0.15 mm gap, then 3 mm at 0.2 mm."""
    wider = HALF_PITCH + 0.05
    a = chain(
        "LVDS_P",
        (0.0, -HALF_PITCH),
        (6.0, -HALF_PITCH),
        (6.05, -wider),
        (9.0, -wider),
    )
    b = chain("LVDS_N", (0.0, HALF_PITCH), (9.0, HALF_PITCH))

    coupled_a, _ = only_section(find_pair(a, b, CHORD))

    np.testing.assert_allclose(coupled_a, [(0.0, -HALF_PITCH), (6.0, -HALF_PITCH)])


def test_each_section_keeps_its_own_gap() -> None:
    a, b = split_pair()
    nudged = [(x, y - 0.02) if x >= 8.0 else (x, y) for x, y in a.points]

    pair = find_pair(chain("LVDS_P", *nudged), b, CHORD)

    assert pair is not None
    first, second = pair.sections
    assert first.gap == pytest.approx(GAP)
    assert second.gap == pytest.approx(GAP + 0.02)
    assert not first.b_reversed


def test_track_points_cannot_be_changed_through_between() -> None:
    a, b = fanned_out_pair()
    pair = find_pair(a, b, CHORD)
    assert pair is not None

    pair.a.between(0, 1)[0] = (5.0, 5.0)

    np.testing.assert_allclose(pair.a.points[0], a.points[0])


def test_tracks_never_within_tolerance_are_no_pair() -> None:
    a = chain("LVDS_P", (0.0, 0.0), (10.0, 0.0))
    b = chain("LVDS_N", (0.0, 5.0), (10.0, 5.0))

    assert find_pair(a, b, CHORD) is None


def test_diverging_tracks_are_no_pair() -> None:
    a = chain("LVDS_P", (0.0, 0.0), (10.0, -2.0))
    b = chain("LVDS_N", (0.0, 2 * HALF_PITCH), (10.0, 2.0))

    assert find_pair(a, b, CHORD) is None


def test_different_widths_are_refused_naming_the_nets() -> None:
    a, b = fanned_out_pair()
    wider = chain("USB_D-", *b.points, width=0.3)

    with pytest.raises(PairError, match=r"USB_D\+.*USB_D-"):
        find_pair(a, wider, CHORD)


def test_unpaired_nets_are_refused() -> None:
    a = chain("A", (0.0, 0.0), (10.0, 0.0))
    b = chain("B", (0.0, 0.35), (10.0, 0.35))

    with pytest.raises(PairError, match=r"A.*B"):
        find_pair(a, b, CHORD)


@pytest.mark.parametrize("inner_first", [True, False])
def test_hand_routed_pair_with_45_degree_corners(inner_first: bool) -> None:
    a, b = corner_pair() if inner_first else corner_pair()[::-1]
    a_nm, b_nm = np.round(a, 6), np.round(b, 6)

    pair = find_pair(from_array("CLK_P", a_nm), from_array("CLK_N", b_nm), CHORD)

    coupled_a, coupled_b = only_section(pair)
    assert pair is not None
    np.testing.assert_allclose(coupled_a, a_nm)
    np.testing.assert_allclose(coupled_b, b_nm)
    assert pair.gap == pytest.approx(GAP, abs=NM_ROUNDING_TOLERANCE)


def curved_path() -> FloatArray:
    """5 mm straight, a 90° arc of radius 2, 5 mm straight."""
    return np.concatenate([array((2.0, -5.0)), arc(2.0), array((-5.0, 2.0))])


def test_offset_pair_around_a_dense_arc_is_coupled_throughout() -> None:
    right, left = offset_pair(curved_path(), GAP, WIDTH)

    pair = find_pair(from_array("CLK_P", right), from_array("CLK_N", left), CHORD)

    coupled_a, coupled_b = only_section(pair)
    assert pair is not None
    assert len(coupled_a) == len(right)
    assert len(coupled_b) == len(left)
    assert pair.gap == pytest.approx(GAP, abs=1e-6)


def test_curved_pair_simplified_track_by_track_is_coupled_throughout() -> None:
    right, left = offset_pair(curved_path(), GAP, WIDTH)
    simplified = [
        simplify(track, free_ends(len(track)), FreeSpace()) for track in (right, left)
    ]

    pair = find_pair(
        from_array("CLK_P", simplified[0]), from_array("CLK_N", simplified[1]), CHORD
    )

    assert pair is not None
    np.testing.assert_array_equal(pair.a.points, simplified[0])
    np.testing.assert_array_equal(pair.b.points, simplified[1])
    (section,) = pair.sections
    assert section[:4] == (0, len(simplified[0]) - 1, 0, len(simplified[1]) - 1)
    assert pair.gap == pytest.approx(GAP, abs=CHORD)


def test_centreline_of_a_pair_simplified_track_by_track_stays_smooth() -> None:
    path = curved_path()
    right, left = offset_pair(path, GAP, WIDTH)
    a, b = (simplify(t, free_ends(len(t)), FreeSpace()) for t in (right, left))

    for first, second in ((a, b), (b, a)):
        middle = centreline(first, second, WIDTH).points

        assert not check_bend_radius(middle, GAP, WIDTH).too_tight
        assert hausdorff(middle, path) <= CHORD


def test_straight_pair_round_trips_through_centreline_and_offset() -> None:
    a = array((0.0, -HALF_PITCH), (10.0, -HALF_PITCH))
    b = array((0.0, HALF_PITCH), (10.0, HALF_PITCH))
    middle = centreline(a, b, WIDTH)
    right, left = offset_pair(middle.points, GAP, WIDTH)

    assert not middle.a_on_right
    assert hausdorff(right, b) < ROUND_TRIP_TOLERANCE
    assert hausdorff(left, a) < ROUND_TRIP_TOLERANCE


@pytest.mark.parametrize("inner_first", [True, False])
def test_pair_with_45_degree_corners_round_trips(inner_first: bool) -> None:
    inner, outer = corner_pair()
    a, b = (inner, outer) if inner_first else (outer, inner)

    middle = centreline(a, b, WIDTH)
    right, left = offset_pair(middle.points, GAP, WIDTH)

    track_a, track_b = (right, left) if middle.a_on_right else (left, right)
    assert hausdorff(track_a, a) < ROUND_TRIP_TOLERANCE
    assert hausdorff(track_b, b) < ROUND_TRIP_TOLERANCE


def test_arc_offsets_to_tracks_at_radius_plus_and_minus_half_pitch() -> None:
    radius = 2.0

    right, left = offset_pair(arc(radius), GAP, WIDTH)

    np.testing.assert_allclose(np.hypot(*right.T), radius - HALF_PITCH, atol=1e-5)
    np.testing.assert_allclose(np.hypot(*left.T), radius + HALF_PITCH, atol=1e-5)


def test_bend_at_the_radius_limit_passes_and_offsets_without_folding() -> None:
    at_limit = arc(bend_radius_limit(GAP, WIDTH))

    right, left = offset_pair(at_limit, GAP, WIDTH)

    assert not check_bend_radius(at_limit, GAP, WIDTH).too_tight
    assert LineString(right).is_simple
    assert LineString(left).is_simple


def test_radius_limit_keeps_inner_track_edge_off_the_centre() -> None:
    assert bend_radius_limit(GAP, WIDTH) == pytest.approx(HALF_PITCH + WIDTH / 2)


def test_bend_tighter_than_the_limit_is_reported() -> None:
    limit = bend_radius_limit(GAP, WIDTH)

    check = check_bend_radius(arc(0.9 * limit), GAP, WIDTH)

    assert check.too_tight
    assert check.radius == pytest.approx(0.9 * limit, rel=1e-4)


def test_bend_above_the_limit_passes() -> None:
    limit = bend_radius_limit(GAP, WIDTH)

    check = check_bend_radius(arc(1.1 * limit), GAP, WIDTH)

    assert not check.too_tight


def test_straight_centreline_has_no_bend() -> None:
    check = check_bend_radius(array((0.0, 0.0), (5.0, 0.0)), GAP, WIDTH)

    assert check.radius == math.inf
    assert not check.too_tight


def test_gap_range_reports_narrowing() -> None:
    a = array((0.0, 0.0), (10.0, 0.0))
    b = array((0.0, 2 * HALF_PITCH), (5.0, 2 * HALF_PITCH), (10.0, 0.3))

    smallest, largest = gap_range(a, b, WIDTH)

    assert smallest == pytest.approx(0.3 - WIDTH)
    assert largest == pytest.approx(GAP)


@pytest.mark.parametrize("turn", [math.pi / 4, math.pi / 2])
def test_gap_range_does_not_count_the_outer_corner_of_a_mitred_turn(
    turn: float,
) -> None:
    centreline = array((0.0, 0.0), (5.0, 0.0), (5.0 + math.cos(turn), math.sin(turn)))
    right, left = offset_pair(centreline, GAP, WIDTH)

    assert gap_range(right, left, WIDTH) == pytest.approx((GAP, GAP))
