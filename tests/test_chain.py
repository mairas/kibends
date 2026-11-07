from collections.abc import Iterator

import pytest
from loguru import logger
from shapely.geometry import LineString, Point, Polygon, box

from kibends.chain import NM_PER_MM, Chain, ChainError, Segment, extract_chain

WIDTH_NM = 250_000
LAYER = "F.Cu"
NET = "SIG"
PIN_TOLERANCE_MM = 1e-6


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


@pytest.fixture
def log() -> Iterator[list[str]]:
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{level} {message}")
    yield lines
    logger.remove(sink)


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

        with pytest.raises(AttributeError):
            chain.layer = "B.Cu"  # type: ignore[misc]


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
        tee = LineString([(5, -3), (5, 0)])

        chain = extract_chain(l_z_segments(), [tee])

        assert_points(chain, [(0, 0), (5, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, False, True)

    def test_track_ending_a_hair_short_of_the_chain_still_pins(self) -> None:
        tee = LineString([(5, -3), (5, -1e-9)])

        chain = extract_chain(l_z_segments(), [tee])

        assert pinned_points(chain)[1] == pytest.approx((5, 0), abs=PIN_TOLERANCE_MM)

    def test_track_ending_at_existing_vertex_pins_it_without_insertion(self) -> None:
        tee = LineString([(10, -3), (10, 0)])

        chain = extract_chain(l_z_segments(), [tee])

        assert_points(chain, [(0, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, True)

    def test_track_crossing_the_chain_pins_the_crossing(self) -> None:
        crossing = LineString([(10 - 2, 2.5), (10 + 2, 2.5)])

        chain = extract_chain(l_z_segments(), [crossing])

        assert_points(chain, [(0, 0), (10, 0), (10, 2.5), (10, 5), (20, 5)])
        assert chain.pinned == (True, False, True, False, True)

    def test_track_overlapping_the_chain_pins_both_overlap_ends(self) -> None:
        overlap = LineString([(2, 0), (6, 0)])

        chain = extract_chain(l_z_segments(), [overlap])

        assert_points(chain, [(0, 0), (2, 0), (6, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, True, False, False, True)

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

    def test_locked_segment_splitting_the_chain_is_an_error(self) -> None:
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

    def test_disconnected_pieces_are_an_error_naming_their_ends(self) -> None:
        segments = [seg("a", (0, 0), (10, 0)), seg("b", (20, 0), (30, 0))]

        with pytest.raises(ChainError, match=r"2 separate pieces.*\(20\.000, 0\.000\)"):
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

    def test_two_layers_joined_at_a_via_suggest_one_run_per_layer(self) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("b", (10, 0), (20, 0), layer="B.Cu"),
        ]

        with pytest.raises(ChainError, match=r"layers.*B\.Cu.*F\.Cu.*once per layer"):
            extract_chain(segments)

    def test_locked_segment_on_another_layer_does_not_count(self) -> None:
        segments = [
            *l_z_segments(),
            seg("lock", (20, 5), (25, 5), layer="B.Cu", locked=True),
        ]

        chain = extract_chain(segments)

        assert chain.layer == LAYER
