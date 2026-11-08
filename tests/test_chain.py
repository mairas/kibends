import pytest
from shapely.geometry import LineString, Point, Polygon, box

from kibends.chain import (
    NM_PER_MM,
    Chain,
    ChainError,
    SameNetTrack,
    Segment,
    extract_chain,
    extract_chains,
)

WIDTH_NM = 250_000
LAYER = "F.Cu"
NET = "SIG"
PIN_TOLERANCE_MM = 1e-6
WIDTH_MM = WIDTH_NM / NM_PER_MM


def nm(x_mm: float, y_mm: float) -> tuple[int, int]:
    return round(x_mm * NM_PER_MM), round(y_mm * NM_PER_MM)


def seg(
    id: str,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    width: int = WIDTH_NM,
    layer: str = LAYER,
    net: str = NET,
    locked: bool = False,
) -> Segment:
    return Segment(
        id=id,
        start=nm(*start),
        end=nm(*end),
        width=width,
        layer=layer,
        net=net,
        locked=locked,
    )


def l_z_segments() -> list[Segment]:
    """(0,0) → (10,0) → (10,5) → (20,5), given out of order and mixed in direction."""
    return [
        seg("c", (20, 5), (10, 5)),
        seg("a", (0, 0), (10, 0)),
        seg("b", (10, 5), (10, 0)),
    ]


def assert_points(chain: Chain, expected: list[tuple[float, float]]) -> None:
    assert len(chain.points) == len(expected)
    for got, want in zip(chain.points, expected, strict=True):
        assert got == pytest.approx(want, abs=PIN_TOLERANCE_MM)


def pinned_points(chain: Chain) -> list[tuple[float, float]]:
    return [p for p, pin in zip(chain.points, chain.pinned, strict=True) if pin]


class TestOrdering:
    def test_out_of_order_mixed_direction_segments_form_one_polyline(self) -> None:
        chain = extract_chain(l_z_segments())

        assert_points(chain, [(0, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, False, False, True)
        assert chain.width == pytest.approx(0.25)
        assert chain.layer == LAYER
        assert chain.net == NET
        assert sorted(chain.segment_ids) == ["a", "b", "c"]

    def test_starts_at_lexicographically_smaller_endpoint(self) -> None:
        reversed_input = [
            seg("a", (20, 5), (10, 5)),
            seg("b", (10, 5), (10, 0)),
            seg("c", (10, 0), (0, 0)),
        ]

        chain = extract_chain(reversed_input)

        assert chain.points[0] == pytest.approx((0, 0))
        assert chain.points[-1] == pytest.approx((20, 5))

    def test_single_segment(self) -> None:
        chain = extract_chain([seg("a", (3, 4), (1, 2))])

        assert_points(chain, [(1, 2), (3, 4)])
        assert chain.pinned == (True, True)

    def test_chain_is_frozen(self) -> None:
        chain = extract_chain(l_z_segments())

        # The write pyright rejects statically is the one under test.
        with pytest.raises(AttributeError):
            chain.layer = "B.Cu"  # pyright: ignore[reportAttributeAccessIssue]


class TestDedupe:
    def test_duplicate_and_zero_length_segments_are_dropped(self) -> None:
        segments = [
            *l_z_segments(),
            seg("dup", (10, 0), (0, 0)),
            seg("dot", (10, 5), (10, 5)),
        ]

        chain = extract_chain(segments)

        assert_points(chain, [(0, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, False, False, True)

    def test_dropped_duplicates_are_still_replaced(self) -> None:
        segments = [
            *l_z_segments(),
            seg("dup", (10, 0), (0, 0)),
            seg("dot", (10, 5), (10, 5)),
        ]

        chain = extract_chain(segments)

        assert sorted(chain.segment_ids) == ["a", "b", "c", "dot", "dup"]

    def test_zero_length_segment_away_from_chain_is_a_separate_piece(self) -> None:
        segments = [*l_z_segments(), seg("stray", (50, 50), (50, 50))]

        with pytest.raises(ChainError, match=r"separate pieces.*\(50\.000, 50\.000\)"):
            extract_chain(segments)

    def test_only_zero_length_segments_is_an_error(self) -> None:
        with pytest.raises(ChainError, match="no tracks"):
            extract_chain([seg("dot", (1, 1), (1, 1))])


class TestPinning:
    def test_unselected_track_ending_mid_segment_inserts_pinned_vertex(self) -> None:
        tee = SameNetTrack(LineString([(5, -3), (5, 0)]), WIDTH_MM)

        chain = extract_chain(l_z_segments(), [tee])

        assert_points(chain, [(0, 0), (5, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, False, True)

    def test_track_ending_at_existing_vertex_pins_it_without_insertion(self) -> None:
        tee = SameNetTrack(LineString([(10, -3), (10, 0)]), WIDTH_MM)

        chain = extract_chain(l_z_segments(), [tee])

        assert_points(chain, [(0, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, True)

    def test_track_crossing_the_chain_pins_the_crossing(self) -> None:
        crossing = SameNetTrack(LineString([(10 - 2, 2.5), (10 + 2, 2.5)]), WIDTH_MM)

        chain = extract_chain(l_z_segments(), [crossing])

        assert_points(chain, [(0, 0), (10, 0), (10, 2.5), (10, 5), (20, 5)])
        assert chain.pinned == (True, False, True, False, True)

    def test_track_overlapping_the_chain_pins_both_overlap_ends(self) -> None:
        overlap = SameNetTrack(LineString([(2, 0), (6, 0)]), WIDTH_MM)

        chain = extract_chain(l_z_segments(), [overlap])

        assert_points(chain, [(0, 0), (2, 0), (6, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, True, False, False, True)

    def test_track_whose_copper_reaches_the_chain_copper_pins_nearest_point(
        self,
    ) -> None:
        branch = SameNetTrack(LineString([(2.5, 3), (2.5, 0.2)]), WIDTH_MM)

        chain = extract_chain(l_z_segments(), [branch])

        assert_points(chain, [(0, 0), (2.5, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, False, True)

    def test_wide_track_reaches_further(self) -> None:
        branch = SameNetTrack(LineString([(2.5, 3), (2.5, 0.6)]), 1.0)

        chain = extract_chain(l_z_segments(), [branch])

        assert pinned_points(chain)[1] == pytest.approx((2.5, 0), abs=PIN_TOLERANCE_MM)

    def test_track_whose_copper_stops_short_of_the_chain_copper_does_not_pin(
        self,
    ) -> None:
        branch = SameNetTrack(LineString([(2.5, 3), (2.5, 0.3)]), WIDTH_MM)

        chain = extract_chain(l_z_segments(), [branch])

        assert chain.pinned == (True, False, False, True)

    def test_pad_outline_crossed_pins_entry_and_exit(self) -> None:
        pad = box(4, -1, 6, 1)

        chain = extract_chain(l_z_segments(), [pad])

        assert_points(chain, [(0, 0), (4, 0), (6, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, True, False, False, True)

    def test_round_via_around_a_corner_pins_both_crossings(self) -> None:
        via = Point(10, 0).buffer(0.3)

        chain = extract_chain(l_z_segments(), [via])

        assert_points(chain, [(0, 0), (9.7, 0), (10, 0), (10, 0.3), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, True, False, True)

    def test_chain_ending_inside_pad_pins_endpoint_and_exit(self) -> None:
        pad = box(-1, -1, 1, 1)

        chain = extract_chain(l_z_segments(), [pad])

        assert_points(chain, [(0, 0), (1, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, False, True)

    def test_pad_grazing_the_chain_copper_pins_nearest_point(self) -> None:
        half_width = WIDTH_NM / NM_PER_MM / 2
        pad = box(4, half_width / 2, 6, 2)

        chain = extract_chain(l_z_segments(), [pad])

        assert len(pinned_points(chain)) == 3
        assert pinned_points(chain)[1][1] == pytest.approx(0)
        assert 4 <= pinned_points(chain)[1][0] <= 6

    def test_copper_beyond_the_chain_copper_does_not_pin(self) -> None:
        pad = box(4, 1, 6, 2)

        chain = extract_chain(l_z_segments(), [pad])

        assert chain.pinned == (True, False, False, True)

    def test_same_net_fill_containing_the_chain_pins_only_endpoints(self) -> None:
        fill = Polygon([(-5, -5), (30, -5), (30, 10), (-5, 10)])

        chain = extract_chain(l_z_segments(), [fill])

        assert chain.pinned == (True, False, False, True)

    def test_no_surrounding_copper_pins_only_endpoints(self) -> None:
        chain = extract_chain(l_z_segments(), [])

        assert chain.pinned == (True, False, False, True)


class TestLocked:
    def test_locked_end_segment_is_dropped_with_warning(self, log: list[str]) -> None:
        segments = [*l_z_segments(), seg("lock", (20, 5), (25, 5), locked=True)]

        chain = extract_chain(segments)

        assert_points(chain, [(0, 0), (10, 0), (10, 5), (20, 5)])
        assert "lock" not in chain.segment_ids
        assert any(line.startswith("WARNING") and "lock" in line for line in log)

    def test_locked_segment_continuing_the_chain_pins_only_the_ends(self) -> None:
        segments = [*l_z_segments(), seg("lock", (20, 5), (25, 5), locked=True)]

        chain = extract_chain(segments)

        assert chain.pinned == (True, False, False, True)

    def test_locked_branch_whose_copper_reaches_the_chain_pins_it(self) -> None:
        segments = [*l_z_segments(), seg("lock", (2.5, 3), (2.5, 0.2), locked=True)]

        chain = extract_chain(segments)

        assert pinned_points(chain)[1] == pytest.approx((2.5, 0), abs=PIN_TOLERANCE_MM)

    def test_locked_segment_splitting_the_selection_gives_two_pinned_chains(
        self,
    ) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("lock", (10, 0), (10, 5), locked=True),
            seg("c", (10, 5), (20, 5)),
        ]

        first, second = extract_chains(segments)

        assert_points(first, [(0, 0), (10, 0)])
        assert_points(second, [(10, 5), (20, 5)])
        assert first.pinned == second.pinned == (True, True)
        assert (first.segment_ids, second.segment_ids) == (("a",), ("c",))

    def test_locked_segment_splitting_the_selection_pins_where_it_touches(
        self,
    ) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("lock", (10, 0), (20, 0), locked=True),
            seg("c", (15, 0), (15, 10)),
        ]

        first, second = extract_chains(segments)

        assert pinned_points(first) == [(0, 0), (10, 0)]
        assert pinned_points(second) == [(15, 0), (15, 10)]

    def test_extract_chain_still_refuses_a_locked_splitter(self) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("lock", (10, 0), (10, 5), locked=True),
            seg("c", (10, 5), (20, 5)),
        ]

        with pytest.raises(ChainError, match=r"locked.*lock.*split"):
            extract_chain(segments)

    def test_locked_branch_mid_chain_is_dropped_and_its_junction_pinned(self) -> None:
        segments = [*l_z_segments(), seg("lock", (10, 0), (15, -5), locked=True)]

        chain = extract_chain(segments)

        assert_points(chain, [(0, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, True)

    def test_all_segments_locked_is_an_error(self) -> None:
        segments = [seg("a", (0, 0), (10, 0), locked=True)]

        with pytest.raises(ChainError, match="no tracks"):
            extract_chain(segments)


class TestTopologyErrors:
    def test_empty_selection_is_an_error(self) -> None:
        with pytest.raises(ChainError, match="no tracks"):
            extract_chain([])

    def test_loop_is_an_error_saying_the_chain_is_closed(self) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("b", (10, 0), (10, 10)),
            seg("c", (10, 10), (0, 0)),
        ]

        with pytest.raises(ChainError, match="closed"):
            extract_chain(segments)

    def test_branch_is_an_error_naming_the_branch_point(self) -> None:
        segments = [*l_z_segments(), seg("d", (10, 0), (15, -5))]

        with pytest.raises(ChainError, match=r"branch.*\(10\.000, 0\.000\)"):
            extract_chain(segments)

    def test_selected_track_ending_mid_segment_is_a_branch(self) -> None:
        segments = [*l_z_segments(), seg("tee", (5, -3), (5, 0))]

        with pytest.raises(ChainError, match=r"branch.*\(5\.000, 0\.000\).*tee"):
            extract_chain(segments)

    def test_overlapping_collinear_segments_are_a_branch(self) -> None:
        segments = [seg("a", (0, 0), (10, 0)), seg("b", (0, 0), (5, 0))]

        with pytest.raises(ChainError, match=r"branch.*\(5\.000, 0\.000\)"):
            extract_chain(segments)


class TestMixedAttributes:
    def test_mixed_widths_are_an_error_naming_the_widths(self) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("b", (10, 0), (20, 0), width=400_000),
        ]

        with pytest.raises(ChainError, match=r"widths.*0\.25.*a.*0\.4.*b"):
            extract_chain(segments)

    def test_mixed_nets_are_an_error_naming_the_nets(self) -> None:
        segments = [seg("a", (0, 0), (10, 0)), seg("b", (10, 0), (20, 0), net="GND")]

        with pytest.raises(ChainError, match=r"nets.*GND.*b.*SIG.*a"):
            extract_chain(segments)

    def test_two_layers_joined_at_a_via_are_not_one_chain(self) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("b", (10, 0), (20, 0), layer="B.Cu"),
        ]

        with pytest.raises(
            ChainError,
            match=r"layers.*B\.Cu.*F\.Cu.*; tracks on several layers do "
            r"not form one chain$",
        ):
            extract_chain(segments)

    def test_locked_segment_on_another_layer_does_not_count(self) -> None:
        segments = [
            *l_z_segments(),
            seg("lock", (20, 5), (25, 5), layer="B.Cu", locked=True),
        ]

        chain = extract_chain(segments)

        assert chain.layer == LAYER


class TestSeveralChains:
    def test_disjoint_chains_of_different_nets_give_one_chain_each(self) -> None:
        segments = [
            seg("g1", (0, 10), (10, 10), net="GND"),
            *l_z_segments(),
            seg("g2", (10, 10), (10, 20), net="GND"),
        ]
        gnd_pad = box(4, 9, 6, 11)

        def copper(_layer: str, net: str) -> list[Polygon]:
            return [gnd_pad] if net == "GND" else []

        gnd, sig = extract_chains(segments, copper)

        assert (gnd.net, sig.net) == ("GND", NET)
        assert_points(gnd, [(0, 10), (4, 10), (6, 10), (10, 10), (10, 20)])
        assert gnd.pinned == (True, True, True, False, True)
        assert sorted(gnd.segment_ids) == ["g1", "g2"]
        assert_points(sig, [(0, 0), (10, 0), (10, 5), (20, 5)])
        assert sig.pinned == (True, False, False, True)

    def test_one_chain_is_what_extract_chain_gives(self) -> None:
        segments = [*l_z_segments(), seg("dup", (10, 0), (0, 0))]
        tee = SameNetTrack(LineString([(5, -3), (5, 0)]), WIDTH_MM)

        assert extract_chains(segments, lambda _layer, _net: [tee]) == [
            extract_chain(segments, [tee])
        ]

    def test_selected_chains_of_one_net_pin_each_other_where_they_touch(
        self,
    ) -> None:
        # B ends 0.05 mm off A's centreline: inside A's copper, yet no branch.
        segments = [
            seg("a1", (0, 0), (4, 2)),
            seg("a2", (4, 2), (6, 2)),
            seg("a3", (6, 2), (10, 0)),
            seg("b", (5, 2.05), (5, 6)),
        ]

        a, b = extract_chains(segments)

        assert_points(a, [(0, 0), (4, 2), (5, 2), (6, 2), (10, 0)])
        assert a.pinned == (True, False, True, False, True)
        assert_points(b, [(5, 2.05), (5, 6)])
        assert b.pinned == (True, True)

    def test_disjoint_pieces_of_one_net_give_two_chains(self) -> None:
        segments = [seg("b", (20, 0), (30, 0)), seg("a", (10, 0), (0, 0))]

        first, second = extract_chains(segments)

        assert_points(first, [(0, 0), (10, 0)])
        assert_points(second, [(20, 0), (30, 0)])
        assert (first.segment_ids, second.segment_ids) == (("a",), ("b",))

    def test_extract_chain_still_refuses_several_pieces(self) -> None:
        segments = [seg("a", (0, 0), (10, 0)), seg("b", (20, 0), (30, 0))]

        with pytest.raises(ChainError, match=r"2 separate pieces.*\(20\.000, 0\.000\)"):
            extract_chain(segments)

    def test_neck_down_gives_two_chains_pinned_at_the_junction(self) -> None:
        segments = [
            seg("wide", (10, 0), (20, 0), width=500_000),
            seg("thin", (0, 0), (10, 0)),
        ]

        thin, wide = extract_chains(segments)

        assert (thin.width, wide.width) == (pytest.approx(0.25), pytest.approx(0.5))
        assert_points(thin, [(0, 0), (10, 0)])
        assert_points(wide, [(10, 0), (20, 0)])

    def test_same_net_tee_with_a_narrower_stub_names_the_branch_point(self) -> None:
        segments = [*l_z_segments(), seg("stub", (10, 0), (15, -5), width=150_000)]

        with pytest.raises(ChainError, match=r"branch.*\(10\.000, 0\.000\).*stub"):
            extract_chains(segments)

    def test_stub_of_another_width_ending_mid_segment_is_a_branch(self) -> None:
        segments = [*l_z_segments(), seg("tee", (5, -3), (5, 0), width=500_000)]

        with pytest.raises(ChainError, match=r"branch.*\(5\.000, 0\.000\).*tee"):
            extract_chains(segments)

    def test_same_width_tee_is_a_branch(self) -> None:
        segments = [*l_z_segments(), seg("d", (10, 0), (15, -5))]

        with pytest.raises(ChainError, match=r"branch.*\(10\.000, 0\.000\)"):
            extract_chains(segments)

    def test_tracks_of_two_widths_on_the_same_span_are_refused(self) -> None:
        segments = [seg("a", (0, 0), (10, 0)), seg("b", (10, 0), (0, 0), width=500_000)]

        with pytest.raises(ChainError, match=r"a and b overlap.*different widths"):
            extract_chains(segments)

    def test_tee_across_nets_is_two_chains(self) -> None:
        segments = [*l_z_segments(), seg("d", (10, 0), (15, -5), net="GND")]

        gnd, sig = extract_chains(segments)

        assert gnd.segment_ids == ("d",)
        assert sorted(sig.segment_ids) == ["a", "b", "c"]

    def test_net_with_only_a_zero_length_track_is_refused(self) -> None:
        segments = [seg("a", (0, 0), (10, 0)), seg("z", (50, 50), (50, 50), net="G")]

        with pytest.raises(ChainError, match=r"z.*\(50\.000, 50\.000\)"):
            extract_chains(segments)

    def test_track_ending_at_the_junction_tolerance_is_a_branch(self) -> None:
        segments = [
            seg("a", (0, 0), (1, 0)),
            seg("tee", (0.5, -3), (0.5, -10 / NM_PER_MM)),
        ]

        with pytest.raises(ChainError, match=r"branch.*tee ends in the middle of a"):
            extract_chains(segments)

    def test_isolated_zero_length_track_is_refused(self) -> None:
        segments = [*l_z_segments(), seg("stray", (50, 50), (50, 50))]

        with pytest.raises(ChainError, match=r"stray.*\(50\.000, 50\.000\)"):
            extract_chains(segments)

    def test_two_layers_joined_at_a_via_give_a_chain_on_each(self) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("b", (10, 0), (20, 0), layer="B.Cu"),
        ]
        via = Point(10, 0).buffer(0.3)
        asked: list[tuple[str, str]] = []

        def copper(layer: str, net: str) -> list[Polygon]:
            asked.append((layer, net))
            return [via]

        back, front = extract_chains(segments, copper)

        assert (back.layer, back.segment_ids) == ("B.Cu", ("b",))
        assert (front.layer, front.segment_ids) == (LAYER, ("a",))
        assert_points(back, [(10, 0), (10.3, 0), (20, 0)])
        assert_points(front, [(0, 0), (9.7, 0), (10, 0)])
        assert back.pinned == front.pinned == (True, True, True)
        assert sorted(asked) == [("B.Cu", NET), (LAYER, NET)]

    def test_same_net_tracks_crossing_on_two_layers_do_not_branch(self) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("b", (5, -5), (5, 5), layer="B.Cu"),
        ]

        back, front = extract_chains(segments)

        assert_points(back, [(5, -5), (5, 5)])
        assert_points(front, [(0, 0), (10, 0)])

    def test_locked_segment_pins_only_its_own_layer(self) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("b", (0, 1), (10, 1), layer="B.Cu"),
            seg("lock", (5, 1), (5, -5), locked=True),
        ]

        back, front = extract_chains(segments)

        assert back.pinned == (True, True)
        assert_points(front, [(0, 0), (5, 0), (10, 0)])
        assert front.pinned == (True, True, True)
