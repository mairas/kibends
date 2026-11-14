import math

import numpy as np
import pytest
from conftest import (
    CENTRAL_ATOL,
    CENTRAL_RTOL,
    CENTRAL_STEP,
    NM_ROUNDING,
    POSITION_ROUND_OFF,
)
from distance_fields import FreeSpace
from shapely.geometry import LineString

from kibends import pairs as pairs_module
from kibends.bursts import DEFAULT_CHORD_TOLERANCE
from kibends.chain import Chain, Point2
from kibends.energy import BoolArray, FloatArray
from kibends.pairs import (
    Pair,
    PairError,
    Piece,
    _coupled_spans,  # pyright: ignore[reportPrivateUsage]
    _ends,  # pyright: ignore[reportPrivateUsage]
    _SectionEnds,  # pyright: ignore[reportPrivateUsage]
    bend_radius_limit,
    centreline,
    check_bend_radius,
    complementary_net,
    find_pair,
    nets_pair,
    offset_pair,
    offset_pair_pullback,
)
from kibends.simplify import simplify

WIDTH = 0.2
GAP = 0.15
HALF_PITCH = (GAP + WIDTH) / 2
SAMPLED_GAP = 1e-6
"""mm the gap measured between offsets of a densely drawn curve may differ
from the offset's gap."""
MITRED_RADIUS = 1e-5
"""mm a mitred offset vertex of `arc` may lie off the offset radius."""
SAMPLED_RADIUS = 1e-4
"""Relative error of a bend radius measured on `arc`'s vertices."""
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
    np.testing.assert_allclose(middle[:, 1], 0.0, atol=POSITION_ROUND_OFF)


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


@pytest.mark.parametrize("chord", [0.0, 0.01, 0.025])
def test_pair_split_apart_gives_two_sections(chord: float) -> None:
    a, b = split_pair()

    pair = find_pair(a, b, chord)

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


@pytest.mark.parametrize("chord", [0.0, 0.01, 0.025])
def test_split_pair_running_opposite_ways_tags_b_pieces_by_section(
    chord: float,
) -> None:
    a, b = split_pair()
    reversed_b = chain("LVDS_N", *reversed(b.points))

    pair = find_pair(a, reversed_b, chord)

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


def test_a_parallel_stretch_shorter_than_two_pitches_is_coupled() -> None:
    a = chain(
        "LVDS_P", (0.0, -3.0), (2.0, -HALF_PITCH), (2.5, -HALF_PITCH), (4.5, -3.0)
    )
    b = chain("LVDS_N", (0.0, 3.0), (2.0, HALF_PITCH), (2.5, HALF_PITCH), (4.5, 3.0))

    coupled_a, coupled_b = only_section(find_pair(a, b, CHORD))

    np.testing.assert_allclose(coupled_a, [(2.0, -HALF_PITCH), (2.5, -HALF_PITCH)])
    np.testing.assert_allclose(coupled_b, [(2.0, HALF_PITCH), (2.5, HALF_PITCH)])


@pytest.mark.parametrize("split_a", [False, True])
def test_a_finely_split_arc_pairs_as_one_section(split_a: bool) -> None:
    """A 90° turn of radius 1, one track split every 5°, the other every 3°,
    both into segments far shorter than a pitch."""
    coarse, fine = offset_pair(arc(1.0, 19), GAP, WIDTH)[0], arc(1.0, 31)
    fine = offset_pair(fine, GAP, WIDTH)[1]
    a, b = (fine, coarse) if split_a else (coarse, fine)

    coupled_a, coupled_b = only_section(
        find_pair(from_array("CLK_P", a), from_array("CLK_N", b), CHORD)
    )

    assert len(coupled_a) == len(a)
    assert len(coupled_b) == len(b)


@pytest.mark.parametrize("chord", [0.0, 0.01, 0.025])
def test_stretch_at_another_spacing_does_not_set_the_pitch(chord: float) -> None:
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

    pair = find_pair(a, b, chord)

    coupled_a, coupled_b = only_section(pair)
    assert pair is not None
    np.testing.assert_allclose(coupled_a, [(0.0, -HALF_PITCH), (6.0, -HALF_PITCH)])
    np.testing.assert_allclose(coupled_b, [(0.0, HALF_PITCH), (6.0, HALF_PITCH)])
    assert pair.gap == pytest.approx(GAP)


def test_section_ends_of_a_run_shorter_than_a_pitch_do_not_cross() -> None:
    """The run reaches 5.0..5.05 mm; in-band vertices lie inward of each run
    end, past the other, but within a pitch of it."""
    step = 0.02
    a = array((0.0, -HALF_PITCH), (10.0, -HALF_PITCH))
    b = array((0.0, HALF_PITCH), (10.0, HALF_PITCH))
    candidates = array((4.8, 4.8), (5.25, 5.25))

    ends = _ends(a, b, (5.0, 5.05), (5.0, 5.05), candidates, 2 * HALF_PITCH, step)

    assert ends.a == pytest.approx((5.0, 5.05))
    assert ends.b == pytest.approx((5.0, 5.05))


def test_a_span_running_backwards_along_a_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no in-band vertex near, `_ends` leaves a run's ends where the
    run and b's run, projected onto a, end; when the two do not overlap the
    ends cross by more than a sample step. Such a span is no section."""
    a = array((0.0, -HALF_PITCH), (10.0, -HALF_PITCH))
    b = array((0.0, HALF_PITCH), (10.0, HALF_PITCH))
    no_candidates = np.zeros((0, 2))
    crossed = _ends(a, b, (2.0, 3.0), (5.0, 6.0), no_candidates, 2 * HALF_PITCH, 0.05)
    assert crossed.a[0] - crossed.a[1] > 1.0

    def crossing_ends(*_: object) -> _SectionEnds:
        return crossed

    monkeypatch.setattr(pairs_module, "_ends", crossing_ends)

    assert _coupled_spans(a, b, WIDTH, CHORD) == []


def test_pair_widening_briefly_at_its_pads_is_coupled_from_the_jog() -> None:
    """A hand-routed pair (gap 0.25 mm) that widens by 17 µm for 0.25 mm
    after its pads, then runs on at 45° to a fan-out. The pad stubs shadow
    the widened stretch, so its sampled run is shorter than a sample step:
    the section starts at the jog back to the pair's gap."""
    width = 0.1326
    pinned = (True, True, False, False, False, False, True)
    a = Chain(
        points=(
            (130.891301, 82.713801),
            (130.9, 82.7225),
            (130.9, 82.4725),
            (130.8913, 82.4638),
            (130.8913, 82.28924),
            (131.940375, 81.240165),
            (132.200165, 81.240165),
        ),
        pinned=pinned,
        width=width,
        layer="F.Cu",
        net="USB_D+",
        segment_ids=(),
    )
    b = Chain(
        points=(
            (130.508699, 82.713801),
            (130.5, 82.7225),
            (130.5, 82.4725),
            (130.5087, 82.4638),
            (130.5087, 82.13076),
            (131.669835, 80.969625),
            (131.669835, 80.709835),
        ),
        pinned=pinned,
        width=width,
        layer="F.Cu",
        net="USB_D-",
        segment_ids=(),
    )

    pair = find_pair(a, b, CHORD)

    assert pair is not None
    (section,) = pair.sections
    assert not section.b_reversed
    coupled_a, coupled_b = pair.coupled(section)
    np.testing.assert_allclose(coupled_a, a.points[2:6])
    np.testing.assert_allclose(coupled_b, b.points[3:6])
    assert pair.gap == pytest.approx(0.25, abs=NM_ROUNDING)


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
    assert pair.gap == pytest.approx(GAP, abs=SAMPLED_GAP)


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


def test_a_pair_around_a_tight_turn_is_coupled_around_it() -> None:
    """Around a 180° turn of radius 0.5 the inner track runs 1.0 mm coupled,
    under three pitches, then both fan out."""
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

    pair = find_pair(from_array("CLK_P", inner), from_array("CLK_N", outer), CHORD)

    assert pair is not None
    (section,) = pair.sections
    assert section[:4] == (1, len(inner) - 2, 1, len(outer) - 2)


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
    assert pair.gap == pytest.approx(GAP, abs=NM_ROUNDING)


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
    assert pair.gap == pytest.approx(GAP, abs=SAMPLED_GAP)


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
        # Offset from one track, which strays up to CHORD, by half a pitch
        # measured to the other, which strays up to CHORD too.
        assert hausdorff(middle, path) <= 2 * CHORD


def test_straight_pair_round_trips_through_centreline_and_offset() -> None:
    a = array((0.0, -HALF_PITCH), (10.0, -HALF_PITCH))
    b = array((0.0, HALF_PITCH), (10.0, HALF_PITCH))
    middle = centreline(a, b, WIDTH)
    right, left = offset_pair(middle.points, GAP, WIDTH)

    assert not middle.a_on_right
    assert hausdorff(right, b) < POSITION_ROUND_OFF
    assert hausdorff(left, a) < POSITION_ROUND_OFF


@pytest.mark.parametrize("inner_first", [True, False])
def test_pair_with_45_degree_corners_round_trips(inner_first: bool) -> None:
    inner, outer = corner_pair()
    a, b = (inner, outer) if inner_first else (outer, inner)

    middle = centreline(a, b, WIDTH)
    right, left = offset_pair(middle.points, GAP, WIDTH)

    track_a, track_b = (right, left) if middle.a_on_right else (left, right)
    assert hausdorff(track_a, a) < POSITION_ROUND_OFF
    assert hausdorff(track_b, b) < POSITION_ROUND_OFF


def test_arc_offsets_to_tracks_at_radius_plus_and_minus_half_pitch() -> None:
    radius = 2.0

    right, left = offset_pair(arc(radius), GAP, WIDTH)

    np.testing.assert_allclose(
        np.hypot(*right.T), radius - HALF_PITCH, atol=MITRED_RADIUS
    )
    np.testing.assert_allclose(
        np.hypot(*left.T), radius + HALF_PITCH, atol=MITRED_RADIUS
    )


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
    assert check.radius == pytest.approx(0.9 * limit, rel=SAMPLED_RADIUS)


def test_bend_above_the_limit_passes() -> None:
    limit = bend_radius_limit(GAP, WIDTH)

    check = check_bend_radius(arc(1.1 * limit), GAP, WIDTH)

    assert not check.too_tight


def test_straight_centreline_has_no_bend() -> None:
    check = check_bend_radius(array((0.0, 0.0), (5.0, 0.0)), GAP, WIDTH)

    assert check.radius == math.inf
    assert not check.too_tight


def turned(turn: float) -> FloatArray:
    """A centreline of four unit edges turning by `turn` at its middle vertex."""
    after = np.array([math.cos(turn), math.sin(turn)])
    middle = np.array([2.0, 0.0])
    return np.array(
        [(0.0, 0.0), (1.0, 0.0), middle, middle + after, middle + 2 * after]
    )


FOLDED = array((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (1.5, 0.0), (1.0, 0.0))
"""A centreline folded straight back on itself at its middle vertex, as
an L-BFGS trial step can leave it."""


def test_a_fold_offsets_to_finite_tracks() -> None:
    right, left = offset_pair(FOLDED, GAP, WIDTH)

    assert np.isfinite(right).all()
    assert np.isfinite(left).all()


def test_the_offset_pullback_at_a_fold_is_finite() -> None:
    rng = np.random.default_rng(0)
    at_right, at_left = rng.normal(size=FOLDED.shape), rng.normal(size=FOLDED.shape)

    gradient = offset_pair_pullback(FOLDED, GAP, WIDTH, at_right, at_left)

    assert np.isfinite(gradient).all()


@pytest.mark.parametrize("turn", [0.5 * math.pi, 0.99 * math.pi])
def test_the_offset_pullback_matches_finite_differences(turn: float) -> None:
    # 0.99π lies past the turn beyond which the miter stops growing.
    points = turned(turn)
    rng = np.random.default_rng(1)
    at_right, at_left = rng.normal(size=points.shape), rng.normal(size=points.shape)

    def value(centreline: FloatArray) -> float:
        right, left = offset_pair(centreline, GAP, WIDTH)
        return float(np.sum(at_right * right) + np.sum(at_left * left))

    analytic = offset_pair_pullback(points, GAP, WIDTH, at_right, at_left)

    numeric = np.zeros_like(points)
    for k in range(len(points)):
        for axis in (0, 1):
            up, down = points.copy(), points.copy()
            up[k, axis] += CENTRAL_STEP
            down[k, axis] -= CENTRAL_STEP
            numeric[k, axis] = (value(up) - value(down)) / (2 * CENTRAL_STEP)
    np.testing.assert_allclose(analytic, numeric, rtol=CENTRAL_RTOL, atol=CENTRAL_ATOL)


def test_a_fold_is_too_tight() -> None:
    check = check_bend_radius(FOLDED, GAP, WIDTH)

    assert check.radius == 0.0
    assert check.too_tight


def test_a_repeated_vertex_is_too_tight() -> None:
    # As two vertices of an outer track on either side of the point square
    # across from the inner track's corner offset to the centreline's corner.
    repeated = array((0.0, 0.0), (5.0, 0.0), (5.0, 0.0), (8.0, 3.0))

    check = check_bend_radius(repeated, GAP, WIDTH)

    assert check.radius == 0.0
    assert check.too_tight


FOLD_DIRECTIONS = 360
"""Directions, one degree apart, to fold a centreline along: off the axes,
rounding leaves 1 + cos of a fold a hair above 0."""


def test_a_fold_off_the_axes_is_too_tight() -> None:
    for angle in np.radians(np.arange(FOLD_DIRECTIONS)):
        turn = array(
            (math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))
        )
        assert check_bend_radius(FOLDED @ turn, GAP, WIDTH).too_tight, angle


def pinned(net: str, points: FloatArray, *at: int) -> Chain:
    """`points` as a chain of `net`, pinned at its ends and at vertices `at`."""
    pins = [k in (0, len(points) - 1, *at) for k in range(len(points))]
    return Chain(
        tuple((x, y) for x, y in points.tolist()),
        tuple(pins),
        WIDTH,
        "F.Cu",
        net,
        (),
    )


def section_ends(pair: Pair) -> list[tuple[FloatArray, FloatArray]]:
    """Each section's ends on a and on b, in order along a."""
    ends: list[tuple[FloatArray, FloatArray]] = []
    for s in pair.sections:
        b = [s.b_end, s.b_start] if s.b_reversed else [s.b_start, s.b_end]
        ends.append((pair.a.points[[s.a_start, s.a_end]], pair.b.points[b]))
    return ends


def straight(y: float, *xs: float) -> FloatArray:
    return array(*((x, y) for x in xs))


@pytest.mark.parametrize("b_reversed", [False, True])
def test_pins_between_a_stretch_s_ends_stay_inside_its_section(
    b_reversed: bool,
) -> None:
    a = pinned("USB_D+", straight(-HALF_PITCH, 0.0, 4.0, 10.0), 1)
    b_points = straight(HALF_PITCH, 0.0, 7.0, 10.0)
    b = pinned("USB_D-", b_points[::-1] if b_reversed else b_points, 1)

    pair = find_pair(a, b, CHORD)

    assert pair is not None
    assert [s.b_reversed for s in pair.sections] == [b_reversed]
    ((ends_a, ends_b),) = section_ends(pair)
    np.testing.assert_allclose(ends_a[:, 0], [0.0, 10.0])
    np.testing.assert_allclose(ends_b[:, 0], [0.0, 10.0])
    assert pinned_inside(pair) == 2


def pinned_inside(pair: Pair) -> int:
    """How many pins lie between the ends of a section, on either track."""
    return sum(
        int(pair.a.pinned[s.a_start + 1 : s.a_end].sum())
        + int(pair.b.pinned[s.b_start + 1 : s.b_end].sum())
        for s in pair.sections
    )
