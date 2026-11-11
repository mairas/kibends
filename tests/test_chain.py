import math
from collections.abc import Callable
from dataclasses import replace
from itertools import pairwise

import pytest
from conftest import one_chain
from shapely.geometry import LineString, Point, Polygon, box

from kibends.chain import (
    NM_PER_MM,
    Chain,
    ChainError,
    Contact,
    SameNetTrack,
    Segment,
    extract_chains,
    shared_track_groups,
)
from kibends.coords import BOARD_FRAME, UserFrame
from kibends.snapshot import Arc

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


@pytest.mark.parametrize(
    ("segment_ids", "groups"),
    [
        ([{"a"}, {"a", "b"}, {"b"}, {"c"}], [0, 0, 0, 3]),
        ([{"x"}, {"y"}, {"z"}, {"y", "x"}, {"z", "y"}], [0, 0, 0, 0, 0]),
        ([{"x"}, {"y"}], [0, 1]),
    ],
)
def test_shared_track_groups_join_chains_through_others(
    segment_ids: list[set[str]], groups: list[int]
) -> None:
    assert shared_track_groups(segment_ids) == groups


class TestOrdering:
    def test_out_of_order_mixed_direction_segments_form_one_polyline(self) -> None:
        chain = one_chain(l_z_segments(), frame=BOARD_FRAME)

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

        chain = one_chain(reversed_input, frame=BOARD_FRAME)

        assert chain.points[0] == pytest.approx((0, 0))
        assert chain.points[-1] == pytest.approx((20, 5))

    def test_single_segment(self) -> None:
        chain = one_chain([seg("a", (3, 4), (1, 2))], frame=BOARD_FRAME)

        assert_points(chain, [(1, 2), (3, 4)])
        assert chain.pinned == (True, True)

    def test_chain_is_frozen(self) -> None:
        chain = one_chain(l_z_segments(), frame=BOARD_FRAME)

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

        chain = one_chain(segments, frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, False, False, True)

    def test_dropped_duplicates_are_still_replaced(self) -> None:
        segments = [
            *l_z_segments(),
            seg("dup", (10, 0), (0, 0)),
            seg("dot", (10, 5), (10, 5)),
        ]

        chain = one_chain(segments, frame=BOARD_FRAME)

        assert sorted(chain.segment_ids) == ["a", "b", "c", "dot", "dup"]

    def test_a_nanometre_track_at_a_corner_adds_no_vertex(self) -> None:
        corner = nm(10, 0)
        off = (corner[0] + 1, corner[1] + 1)
        a, b, c = sorted(l_z_segments(), key=lambda s: s.id)
        segments = [
            a,
            Segment("short", corner, off, WIDTH_NM, LAYER, NET),
            replace(b, end=off),
            c,
        ]

        chain = one_chain(segments, frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, False, False, True)
        assert sorted(chain.segment_ids) == ["a", "b", "c", "short"]

    def test_zero_length_segment_away_from_chain_is_left_out(self) -> None:
        segments = [*l_z_segments(), seg("stray", (50, 50), (50, 50))]

        chain = one_chain(segments, frame=BOARD_FRAME)

        assert sorted(chain.segment_ids) == ["a", "b", "c"]

    def test_only_zero_length_segments_is_an_error(self) -> None:
        with pytest.raises(ChainError, match="no tracks"):
            one_chain([seg("dot", (1, 1), (1, 1))], frame=BOARD_FRAME)


class TestPinning:
    def test_unselected_track_ending_mid_segment_inserts_pinned_vertex(self) -> None:
        tee = SameNetTrack(LineString([(5, -3), (5, 0)]), WIDTH_MM)

        chain = one_chain(l_z_segments(), [tee], frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (5, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, False, True)

    def test_track_ending_at_existing_vertex_pins_it_without_insertion(self) -> None:
        tee = SameNetTrack(LineString([(10, -3), (10, 0)]), WIDTH_MM)

        chain = one_chain(l_z_segments(), [tee], frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, True)

    def test_track_crossing_the_chain_pins_the_crossing(self) -> None:
        crossing = SameNetTrack(LineString([(10 - 2, 2.5), (10 + 2, 2.5)]), WIDTH_MM)

        chain = one_chain(l_z_segments(), [crossing], frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (10, 0), (10, 2.5), (10, 5), (20, 5)])
        assert chain.pinned == (True, False, True, False, True)

    def test_track_overlapping_the_chain_pins_both_overlap_ends(self) -> None:
        overlap = SameNetTrack(LineString([(2, 0), (6, 0)]), WIDTH_MM)

        chain = one_chain(l_z_segments(), [overlap], frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (2, 0), (6, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, True, False, False, True)

    def test_track_whose_copper_reaches_the_chain_copper_pins_nearest_point(
        self,
    ) -> None:
        branch = SameNetTrack(LineString([(2.5, 3), (2.5, 0.2)]), WIDTH_MM)

        chain = one_chain(l_z_segments(), [branch], frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (2.5, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, False, True)

    def test_wide_track_reaches_further(self) -> None:
        branch = SameNetTrack(LineString([(2.5, 3), (2.5, 0.6)]), 1.0)

        chain = one_chain(l_z_segments(), [branch], frame=BOARD_FRAME)

        assert pinned_points(chain)[1] == pytest.approx((2.5, 0), abs=PIN_TOLERANCE_MM)

    def test_track_whose_copper_stops_short_of_the_chain_copper_does_not_pin(
        self,
    ) -> None:
        branch = SameNetTrack(LineString([(2.5, 3), (2.5, 0.3)]), WIDTH_MM)

        chain = one_chain(l_z_segments(), [branch], frame=BOARD_FRAME)

        assert chain.pinned == (True, False, False, True)

    def test_pad_outline_crossed_pins_entry_and_exit(self) -> None:
        pad = box(4, -1, 6, 1)

        chain = one_chain(l_z_segments(), [pad], frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (4, 0), (6, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, True, False, False, True)

    def test_round_via_around_a_corner_pins_both_crossings(self) -> None:
        via = Point(10, 0).buffer(0.3)

        chain = one_chain(l_z_segments(), [via], frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (9.7, 0), (10, 0), (10, 0.3), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, True, False, True)

    def test_chain_ending_inside_pad_pins_endpoint_and_exit(self) -> None:
        pad = box(-1, -1, 1, 1)

        chain = one_chain(l_z_segments(), [pad], frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (1, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, False, True)

    def test_pad_grazing_the_chain_copper_pins_nearest_point(self) -> None:
        half_width = WIDTH_NM / NM_PER_MM / 2
        pad = box(4, half_width / 2, 6, 2)

        chain = one_chain(l_z_segments(), [pad], frame=BOARD_FRAME)

        assert len(pinned_points(chain)) == 3
        assert pinned_points(chain)[1][1] == pytest.approx(0)
        assert 4 <= pinned_points(chain)[1][0] <= 6

    def test_copper_beyond_the_chain_copper_does_not_pin(self) -> None:
        pad = box(4, 1, 6, 2)

        chain = one_chain(l_z_segments(), [pad], frame=BOARD_FRAME)

        assert chain.pinned == (True, False, False, True)

    def test_same_net_fill_containing_the_chain_pins_only_endpoints(self) -> None:
        fill = Polygon([(-5, -5), (30, -5), (30, 10), (-5, 10)])

        chain = one_chain(l_z_segments(), [fill], frame=BOARD_FRAME)

        assert chain.pinned == (True, False, False, True)

    def test_no_surrounding_copper_pins_only_endpoints(self) -> None:
        chain = one_chain(l_z_segments(), [], frame=BOARD_FRAME)

        assert chain.pinned == (True, False, False, True)


CONTACT_MARGIN_MM = 0.01


def contact(shape: Polygon, anchor: tuple[float, float] | None = None) -> Contact:
    """`shape` as a contact, anchored at `anchor` or else at its centre, as
    pads and vias usually are."""
    if anchor is None:
        x0, y0, x1, y1 = shape.bounds
        anchor = ((x0 + x1) / 2, (y0 + y1) / 2)
    return Contact(shape, CONTACT_MARGIN_MM, anchor)


def slide_points(chain: Chain) -> list[tuple[float, float]]:
    return [chain.points[i] for i, _ in chain.slides]


def assert_slides_within(chain: Chain, index: int, shape: Polygon) -> None:
    (slide,) = [region for i, region in chain.slides if i == index]
    assert slide.equals(shape.buffer(-CONTACT_MARGIN_MM))


class TestContacts:
    """A pad or a via that stays is a region: the chain keeps one vertex
    inside it, free to slide there, and the rest is free, unless a vertex
    lies on its anchor, which pins it there."""

    def test_an_end_inside_a_pad_slides_and_the_pad_pins_nothing_else(
        self,
    ) -> None:
        # The resistor pad: one track lies inside it, and the corner where
        # the second leaves it lies on its edge.
        pad = box(-0.6, -0.4, 0.5, 0.4)
        segments = [seg("in", (0, 0), (0.5, 0)), seg("out", (0.5, 0), (3, 3))]

        chain = one_chain(segments, [contact(pad)], frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (0.5, 0), (3, 3)])
        assert chain.pinned == (False, False, True)
        assert [i for i, _ in chain.slides] == [0]
        assert_slides_within(chain, 0, pad)

    def test_a_chain_passing_over_a_pad_keeps_one_sliding_vertex_inside(
        self,
    ) -> None:
        pad = box(4, -1, 6, 1)

        chain = one_chain(l_z_segments(), [contact(pad)], frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (5, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, False, False, False, True)
        assert [i for i, _ in chain.slides] == [1]
        assert_slides_within(chain, 1, pad)

    def test_a_pad_touched_at_two_points_keeps_one(self) -> None:
        # The chain enters the pad twice: 2.5 mm inside on the first pass,
        # 2 mm on the second; the longer one keeps the sliding vertex.
        pad = box(4.5, -1, 9, 1)
        segments = [
            seg("a", (0, 0), (6, 0)),
            seg("b", (6, 0), (6, 3)),
            seg("c", (6, 3), (8, 3)),
            seg("d", (8, 3), (8, 0)),
            seg("e", (8, 0), (14, 0)),
        ]

        chain = one_chain(segments, [contact(pad)], frame=BOARD_FRAME)

        assert sum(chain.pinned) == 2
        assert slide_points(chain) == [
            pytest.approx((4.5 + 1.25, 0), abs=PIN_TOLERANCE_MM)
        ]
        assert_slides_within(chain, chain.slides[0][0], pad)

    def test_an_end_inside_a_via_slides_within_its_ring(self) -> None:
        via = Point(0.1, 0).buffer(0.3)

        chain = one_chain(l_z_segments(), [contact(via)], frame=BOARD_FRAME)

        assert chain.pinned == (False, False, False, True)
        assert_slides_within(chain, 0, via)

    def test_an_end_on_a_contact_s_anchor_is_pinned_there(self) -> None:
        # KiCad drags only the track ends at a via's centre along with it.
        via = Point(0, 0).buffer(0.3)

        chain = one_chain(l_z_segments(), [contact(via)], frame=BOARD_FRAME)

        assert chain.points[0] == (0, 0)
        assert chain.pinned == (True, False, False, True)
        assert chain.slides == ()

    def test_a_vertex_on_a_contact_s_anchor_is_pinned_there(self) -> None:
        pad = box(9, -1, 11, 1)

        chain = one_chain(l_z_segments(), [contact(pad)], frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, True)
        assert chain.slides == ()

    def test_an_anchored_vertex_holds_every_contact_reaching_it(self) -> None:
        # A via on the corner, in a pad off the pad's anchor: the corner
        # stays on the via's centre and holds the pad too.
        pad = box(8, -1, 11, 1)
        via = Point(10, 0).buffer(0.3)

        chain = one_chain(
            l_z_segments(), [contact(pad), contact(via)], frame=BOARD_FRAME
        )

        assert_points(chain, [(0, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, True)
        assert chain.slides == ()

    def test_an_end_that_other_copper_also_reaches_stays_pinned(self) -> None:
        pad = box(-1, -1, 1, 1)
        stub = SameNetTrack(LineString([(0, -3), (0, 0)]), WIDTH_MM)

        chain = one_chain(
            l_z_segments(), [contact(pad, (0.5, 0)), stub], frame=BOARD_FRAME
        )

        assert chain.pinned[0]
        assert chain.slides == ()

    def test_an_end_in_two_overlapping_contacts_slides_within_both(self) -> None:
        pad = box(-1, -1, 1, 1)
        via = Point(0.2, 0).buffer(0.3)

        chain = one_chain(
            l_z_segments(),
            [contact(pad, (0.5, 0)), contact(via)],
            frame=BOARD_FRAME,
        )

        assert chain.pinned == (False, False, False, True)
        ((index, region),) = chain.slides
        assert index == 0
        both = pad.buffer(-CONTACT_MARGIN_MM).intersection(
            via.buffer(-CONTACT_MARGIN_MM)
        )
        assert region.equals(both)

    def test_a_contact_grazing_the_chain_copper_pins_as_other_copper(self) -> None:
        half_width = WIDTH_NM / NM_PER_MM / 2
        pad = box(4, half_width / 2, 6, 2)

        chain = one_chain(l_z_segments(), [contact(pad)], frame=BOARD_FRAME)

        assert len(pinned_points(chain)) == 3
        assert chain.slides == ()

    def test_an_end_reaching_another_selected_track_s_body_stays_pinned(
        self,
    ) -> None:
        # X ends inside the pad, its copper on Y's; the pad keeps X's end
        # from joining Y, so Y keeps X only where X's end is.
        pad = box(-0.5, -0.5, 0.5, 0.5)
        width = 200_000
        segments = [
            seg("x1", (-3, 2), (-3, 0), width=width),
            seg("x2", (-3, 0), (0.45, 0), width=width),
            seg("y1", (0.62, -3), (0.62, 3), width=width),
            seg("y2", (0.62, 3), (3, 3), width=width),
        ]

        x, y = extract_chains(
            segments, lambda _l, _n: [contact(pad)], frame=BOARD_FRAME
        )

        assert x.points[-1] == pytest.approx((0.45, 0))
        assert x.pinned[-1]
        assert all(i != len(x.points) - 1 for i, _ in x.slides)
        assert (0.62, 0.0) in [pytest.approx(p) for p in pinned_points(y)]

    def test_a_vertex_in_two_contacts_slides_within_both(self) -> None:
        # The chain passes over a via inside a pad: both keep the one
        # vertex at their middle.
        pad = box(4, -1, 6, 1)
        via = Point(5, 0).buffer(0.3)

        chain = one_chain(
            l_z_segments(), [contact(pad), contact(via)], frame=BOARD_FRAME
        )

        ((index, region),) = chain.slides
        assert chain.points[index] == pytest.approx((5, 0))
        both = pad.buffer(-CONTACT_MARGIN_MM).intersection(
            via.buffer(-CONTACT_MARGIN_MM)
        )
        assert region.equals(both)

    def test_a_contact_the_anchored_vertex_misses_keeps_its_own_slide(
        self,
    ) -> None:
        via = Point(10, 0).buffer(0.3)
        pad = box(14, 4, 16, 6)

        chain = one_chain(
            l_z_segments(), [contact(via), contact(pad)], frame=BOARD_FRAME
        )

        assert pinned_points(chain) == [(0, 0), (10, 0), (20, 5)]
        assert slide_points(chain) == [pytest.approx((15, 5))]
        assert_slides_within(chain, chain.slides[0][0], pad)

    def test_an_end_in_a_pad_anchored_at_the_next_vertex_slides(self) -> None:
        # The corner on the anchor stays there and keeps the pad; the end
        # beside it is held by nothing else, so it slides within the pad.
        pad = box(-0.6, -0.4, 0.6, 0.4)
        segments = [seg("in", (0, 0), (0.5, 0)), seg("out", (0.5, 0), (3, 3))]

        chain = one_chain(segments, [contact(pad, (0.5, 0))], frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (0.5, 0), (3, 3)])
        assert chain.pinned == (False, True, True)
        assert_slides_within(chain, 0, pad)

    def test_a_pin_where_a_vertex_would_slide_pins_it(self) -> None:
        pad = box(4, -1, 6, 1)
        stub = SameNetTrack(LineString([(5, -3), (5, 0)]), WIDTH_MM)

        chain = one_chain(l_z_segments(), [contact(pad), stub], frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (5, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, False, True)
        assert chain.slides == ()

    def test_a_contact_on_a_junction_pins_every_branch(self) -> None:
        pad = Point(10, 0).buffer(0.3)

        chains = extract_chains(
            tee_segments(), lambda _l, _n: [contact(pad)], frame=BOARD_FRAME
        )

        for chain in chains:
            assert (10.0, 0.0) in pinned_points(chain)
            assert all(chain.points[i] != (10.0, 0.0) for i, _ in chain.slides)

    def test_pinning_the_slides_pins_their_vertices(self) -> None:
        pad = box(4, -1, 6, 1)
        chain = one_chain(l_z_segments(), [contact(pad)], frame=BOARD_FRAME)

        pinned = chain.with_slides_pinned()

        assert pinned.pinned == (True, True, False, False, True)
        assert pinned.slides == ()


class TestLocked:
    def test_locked_end_segment_is_dropped_with_warning(self, log: list[str]) -> None:
        segments = [*l_z_segments(), seg("lock", (20, 5), (25, 5), locked=True)]

        chain = one_chain(segments, frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (10, 0), (10, 5), (20, 5)])
        assert "lock" not in chain.segment_ids
        assert any(line.startswith("WARNING") and "lock" in line for line in log)

    def test_locked_segment_continuing_the_chain_pins_only_the_ends(self) -> None:
        segments = [*l_z_segments(), seg("lock", (20, 5), (25, 5), locked=True)]

        chain = one_chain(segments, frame=BOARD_FRAME)

        assert chain.pinned == (True, False, False, True)

    def test_locked_branch_whose_copper_reaches_the_chain_pins_it(self) -> None:
        segments = [*l_z_segments(), seg("lock", (2.5, 3), (2.5, 0.2), locked=True)]

        chain = one_chain(segments, frame=BOARD_FRAME)

        assert pinned_points(chain)[1] == pytest.approx((2.5, 0), abs=PIN_TOLERANCE_MM)

    def test_locked_segment_splitting_the_selection_gives_two_pinned_chains(
        self,
    ) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("lock", (10, 0), (10, 5), locked=True),
            seg("c", (10, 5), (20, 5)),
        ]

        first, second = extract_chains(segments, frame=BOARD_FRAME)

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

        first, second = extract_chains(segments, frame=BOARD_FRAME)

        assert pinned_points(first) == [(0, 0), (10, 0)]
        assert pinned_points(second) == [(15, 0), (15, 10)]

    def test_locked_branch_mid_chain_is_dropped_and_its_junction_pinned(self) -> None:
        segments = [*l_z_segments(), seg("lock", (10, 0), (15, -5), locked=True)]

        chain = one_chain(segments, frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (10, 0), (10, 5), (20, 5)])
        assert chain.pinned == (True, True, False, True)

    def test_all_segments_locked_is_an_error(self) -> None:
        segments = [seg("a", (0, 0), (10, 0), locked=True)]

        with pytest.raises(ChainError, match="no tracks"):
            one_chain(segments, frame=BOARD_FRAME)


class TestTopologyErrors:
    def test_empty_selection_is_an_error(self) -> None:
        with pytest.raises(ChainError, match="no tracks"):
            one_chain([], frame=BOARD_FRAME)

    def test_track_ending_inside_a_collinear_one_merges_with_it(self) -> None:
        segments = [seg("a", (0, 0), (10, 0)), seg("b", (0, 0), (5, 0))]

        chain = one_chain(segments, frame=BOARD_FRAME)

        assert_points(chain, [(0, 0), (5, 0), (10, 0)])
        assert chain.pinned == (True, False, True)
        assert chain.segment_ids == ("a", "b")

    @pytest.mark.parametrize(
        ("a_width", "b_width"), [(250_000, 500_000), (500_000, 250_000)]
    )
    def test_collinear_tracks_of_two_widths_on_one_span_stay_two_pinned_chains(
        self, a_width: int, b_width: int
    ) -> None:
        segments = [
            seg("a", (0, 0), (10, 0), width=a_width),
            seg("b", (0, 0), (5, 0), width=b_width),
        ]

        a, b = sorted(
            extract_chains(segments, frame=BOARD_FRAME), key=lambda c: c.segment_ids
        )

        assert (a.width, b.width) == (a_width / NM_PER_MM, b_width / NM_PER_MM)
        assert_points(a, [(0, 0), (5, 0), (10, 0)])
        assert a.pinned == (True, True, True)
        assert_points(b, [(0, 0), (5, 0)])
        assert b.pinned == (True, True)


class TestMixedAttributes:
    def test_locked_segment_on_another_layer_does_not_count(self) -> None:
        segments = [
            *l_z_segments(),
            seg("lock", (20, 5), (25, 5), layer="B.Cu", locked=True),
        ]

        chain = one_chain(segments, frame=BOARD_FRAME)

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

        gnd, sig = extract_chains(segments, copper, frame=BOARD_FRAME)

        assert (gnd.net, sig.net) == ("GND", NET)
        assert_points(gnd, [(0, 10), (4, 10), (6, 10), (10, 10), (10, 20)])
        assert gnd.pinned == (True, True, True, False, True)
        assert sorted(gnd.segment_ids) == ["g1", "g2"]
        assert_points(sig, [(0, 0), (10, 0), (10, 5), (20, 5)])
        assert sig.pinned == (True, False, False, True)

    def test_selected_chains_of_one_net_crossing_away_from_their_ends_do_not_pin(
        self,
    ) -> None:
        # Copper of one net does not interact, so a crossing ties nothing.
        segments = [
            seg("a1", (0, 0), (4, 2)),
            seg("a2", (4, 2), (6, 2)),
            seg("a3", (6, 2), (10, 0)),
            seg("b", (5, 1), (5, 6)),
        ]

        a, b = extract_chains(segments, frame=BOARD_FRAME)

        assert_points(a, [(0, 0), (4, 2), (6, 2), (10, 0)])
        assert a.pinned == (True, False, False, True)
        assert_points(b, [(5, 1), (5, 6)])
        assert b.pinned == (True, True)

    def test_selected_chains_of_one_net_overlapping_along_a_stretch_do_not_pin(
        self,
    ) -> None:
        segments = [
            seg("a1", (0, 0), (4, 1)),
            seg("a2", (4, 1), (6, 1)),
            seg("a3", (6, 1), (10, 0)),
            seg("b1", (0, 2), (4, 1.1)),
            seg("b2", (4, 1.1), (6, 1.1)),
            seg("b3", (6, 1.1), (10, 2)),
        ]

        a, b = extract_chains(segments, frame=BOARD_FRAME)

        assert a.pinned == (True, False, False, True)
        assert b.pinned == (True, False, False, True)

    def test_an_end_in_another_selected_chain_s_copper_still_pins_it(self) -> None:
        # The end crosses past the centreline too far to join, so it stays,
        # and the chain it reaches is pinned where it does.
        segments = [seg("a", (0, 0), (10, 0)), seg("s", (5, 5), (5, -0.15))]

        a, s = extract_chains(segments, frame=BOARD_FRAME)

        assert (5, 0) in [tuple(round(c, 6) for c in p) for p in pinned_points(a)]
        assert s.pinned[0] and s.pinned[-1]

    def test_disjoint_pieces_of_one_net_give_two_chains(self) -> None:
        segments = [seg("b", (20, 0), (30, 0)), seg("a", (10, 0), (0, 0))]

        first, second = extract_chains(segments, frame=BOARD_FRAME)

        assert_points(first, [(0, 0), (10, 0)])
        assert_points(second, [(20, 0), (30, 0)])
        assert (first.segment_ids, second.segment_ids) == (("a",), ("b",))

    def test_a_width_change_gives_a_chain_per_width(self) -> None:
        segments = [
            seg("wide", (10, 0), (20, 0), width=500_000),
            seg("thin", (0, 0), (10, 0)),
        ]

        thin, wide = extract_chains(segments, frame=BOARD_FRAME)

        assert (thin.width, wide.width) == (pytest.approx(0.25), pytest.approx(0.5))
        assert_points(thin, [(0, 0), (10, 0)])
        assert_points(wide, [(10, 0), (20, 0)])
        # Collinear but leaving the point opposite ways: loose, not overlapping.
        assert thin.pinned == (True, False)
        assert wide.pinned == (False, True)

    def test_stub_of_another_width_ending_mid_segment_is_a_branch(self) -> None:
        segments = [*l_z_segments(), seg("tee", (5, -3), (5, 0), width=500_000)]

        left, tee, right = extract_chains(segments, frame=BOARD_FRAME)

        assert tee.width == pytest.approx(0.5)
        assert left.points[-1] == tee.points[-1] == right.points[0] == (5.0, 0.0)
        assert (left.pinned[-1], tee.pinned[-1], right.pinned[0]) == (
            False,
            False,
            False,
        )

    def test_tracks_of_two_widths_on_the_same_span_give_a_chain_each(self) -> None:
        segments = [seg("a", (0, 0), (10, 0)), seg("b", (10, 0), (0, 0), width=500_000)]

        a, b = extract_chains(segments, frame=BOARD_FRAME)

        assert (a.segment_ids, b.segment_ids) == (("a",), ("b",))
        assert_points(a, [(0, 0), (10, 0)])
        assert_points(b, [(0, 0), (10, 0)])
        assert a.pinned == b.pinned == (True, True)

    def test_tee_across_nets_is_two_chains(self) -> None:
        segments = [*l_z_segments(), seg("d", (10, 0), (15, -5), net="GND")]

        gnd, sig = extract_chains(segments, frame=BOARD_FRAME)

        assert gnd.segment_ids == ("d",)
        assert sorted(sig.segment_ids) == ["a", "b", "c"]

    def test_net_with_only_a_zero_length_track_leaves_it_out(self) -> None:
        segments = [seg("a", (0, 0), (10, 0)), seg("z", (50, 50), (50, 50), net="G")]

        (chain,) = extract_chains(segments, frame=BOARD_FRAME)

        assert chain.segment_ids == ("a",)

    def test_isolated_zero_length_track_is_left_out(self) -> None:
        segments = [*l_z_segments(), seg("stray", (50, 50), (50, 50))]

        (chain,) = extract_chains(segments, frame=BOARD_FRAME)

        assert sorted(chain.segment_ids) == ["a", "b", "c"]

    def test_a_dot_of_another_width_on_a_chain_end_is_left_out(self) -> None:
        segments = [*l_z_segments(), seg("dot", (0, 0), (0, 0), width=2 * WIDTH_NM)]

        (chain,) = extract_chains(segments, frame=BOARD_FRAME)

        assert sorted(chain.segment_ids) == ["a", "b", "c"]

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

        back, front = extract_chains(segments, copper, frame=BOARD_FRAME)

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

        back, front = extract_chains(segments, frame=BOARD_FRAME)

        assert_points(back, [(5, -5), (5, 5)])
        assert_points(front, [(0, 0), (10, 0)])

    def test_locked_segment_pins_only_its_own_layer(self) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("b", (0, 1), (10, 1), layer="B.Cu"),
            seg("lock", (5, 1), (5, -5), locked=True),
        ]

        back, front = extract_chains(segments, frame=BOARD_FRAME)

        assert back.pinned == (True, True)
        assert_points(front, [(0, 0), (5, 0), (10, 0)])
        assert front.pinned == (True, True, True)


# A grid origin at (10, 20) mm puts board point (x, y) at (x - 10, y - 20).
GRID_FRAME = UserFrame((10.0, 20.0))


def square_loop() -> list[Segment]:
    """A closed square, 10 mm a side, from (0, 0) to (10, 10)."""
    return [
        seg("a", (0, 0), (10, 0)),
        seg("b", (10, 0), (10, 10)),
        seg("c", (10, 10), (0, 10)),
        seg("d", (0, 10), (0, 0)),
    ]


def test_left_out_tracks_are_logged_at_debug_in_the_frame(log: list[str]) -> None:
    segments = [seg("a", (0, 0), (10, 0)), seg("z", (50, 50), (50, 50), net="G")]

    (chain,) = extract_chains(segments, frame=GRID_FRAME)

    assert chain.segment_ids == ("a",)
    (line,) = log
    assert line.startswith("DEBUG")
    assert "z" in line and "(40.000, 30.000) mm" in line


class TestLoops:
    """A closed loop is cut where copper of its net touches it, and at its
    first vertex where none does."""

    def test_a_pad_on_a_corner_cuts_the_loop_there(self) -> None:
        pad = contact(box(9.5, -0.5, 10.5, 0.5))

        chain = one_chain(square_loop(), [pad], frame=BOARD_FRAME)

        assert chain.points[0] == chain.points[-1] == (10, 0)
        assert chain.pinned == (True, False, False, False, True)
        assert sorted(chain.points[1:-1]) == [(0, 0), (0, 10), (10, 10)]
        assert sorted(chain.segment_ids) == ["a", "b", "c", "d"]
        assert chain.slides == ()

    def test_a_loop_touched_at_two_points_is_cut_at_one_and_pinned_at_the_other(
        self,
    ) -> None:
        tees = [
            SameNetTrack(LineString([(5, -3), (5, 0)]), WIDTH_MM),
            SameNetTrack(LineString([(5, 13), (5, 10)]), WIDTH_MM),
        ]

        chain = one_chain(square_loop(), tees, frame=BOARD_FRAME)

        assert chain.points[0] == chain.points[-1]
        assert chain.pinned[0] and chain.pinned[-1]
        assert sorted(set(pinned_points(chain))) == [(5, 0), (5, 10)]
        assert len(pinned_points(chain)) == 3
        assert len(chain.points) == 7

    def test_a_pad_the_loop_passes_pins_the_vertex_it_keeps_there(self) -> None:
        pad = contact(box(4, -1, 6, 1))

        chain = one_chain(square_loop(), [pad], frame=BOARD_FRAME)

        assert chain.points[0] == chain.points[-1]
        assert chain.points[0] == pytest.approx((5, 0), abs=PIN_TOLERANCE_MM)
        assert pinned_points(chain) == [chain.points[0]] * 2
        assert chain.slides == ()

    def test_a_pad_off_its_anchor_on_the_first_corner_keeps_a_sliding_vertex(
        self,
    ) -> None:
        # The walk starts at (0, 0), inside the pad, and the loop is cut at
        # the track instead.
        pad = box(-0.5, -0.5, 0.5, 0.5)
        tee = SameNetTrack(LineString([(5, 13), (5, 10)]), WIDTH_MM)

        chain = one_chain(
            square_loop(), [contact(pad, anchor=(0.2, 0.2)), tee], frame=BOARD_FRAME
        )

        assert chain.points[0] == chain.points[-1] == (5, 10)
        assert pinned_points(chain) == [(5, 10), (5, 10)]
        (slide,) = chain.slides
        index, region = slide
        assert region.equals(pad.buffer(-CONTACT_MARGIN_MM))
        assert Point(chain.points[index]).within(region)
        assert (0, 0) in chain.points

    def test_a_loop_off_a_junction_is_cut_there_and_ends_loose(self) -> None:
        segments = [*square_loop(), seg("stub", (10, 0), (15, -5))]

        loop, stub = extract_chains(segments, frame=BOARD_FRAME)

        assert loop.points[0] == loop.points[-1] == (10, 0)
        assert not any(loop.pinned)
        assert stub.pinned == (False, True)

    def test_a_cut_on_the_loop_opens_it_there_with_both_ends_pinned(self) -> None:
        def cuts(_layer: str, _net: str) -> list[tuple[float, float]]:
            return [(10.0, 10.0)]

        (chain,) = extract_chains(square_loop(), frame=BOARD_FRAME, cuts=cuts)

        assert chain.points[0] == chain.points[-1] == (10, 10)
        assert chain.pinned == (True, False, False, False, True)
        assert chain.slides == ()

    def test_a_loop_nothing_touches_is_cut_and_pinned_at_its_first_vertex(
        self,
    ) -> None:
        chain = one_chain(square_loop(), frame=BOARD_FRAME)

        assert chain.points[0] == chain.points[-1] == (0, 0)
        assert chain.pinned == (True, False, False, False, True)


def cut_at(
    point: tuple[float, float], layer: str = LAYER, net: str = NET
) -> Callable[[str, str], list[tuple[float, float]]]:
    """Cuts at `point` for `net` on `layer` alone."""

    def cuts(on: str, of: str) -> list[tuple[float, float]]:
        return [point] if (on, of) == (layer, net) else []

    return cuts


class TestCuts:
    """A cut, a selected via that may move, ends the chains through it."""

    def test_a_cut_at_a_vertex_ends_both_sides_there_pinned(self) -> None:
        a, bc = extract_chains(l_z_segments(), frame=BOARD_FRAME, cuts=cut_at((10, 0)))

        assert_points(a, [(0, 0), (10, 0)])
        assert_points(bc, [(10, 0), (10, 5), (20, 5)])
        assert a.pinned == (True, True)
        assert bc.pinned == (True, False, True)
        assert a.slides == bc.slides == ()

    def test_held_segments_are_a_held_chain_of_their_own(self) -> None:
        # "b" and "c" came from a selected group.
        segments = [replace(s, held=s.id != "a") for s in l_z_segments()]

        a, bc = extract_chains(segments, frame=BOARD_FRAME)

        assert_points(a, [(0, 0), (10, 0)])
        assert_points(bc, [(10, 0), (10, 5), (20, 5)])
        assert (a.held, a.pinned) == (False, (True, True))
        assert (bc.held, bc.pinned) == (True, (True, True, True))

    def test_a_cut_away_from_every_vertex_leaves_the_chain_whole(self) -> None:
        (chain,) = extract_chains(
            l_z_segments(), frame=BOARD_FRAME, cuts=cut_at((5, 0))
        )

        assert_points(chain, [(0, 0), (10, 0), (10, 5), (20, 5)])

    @pytest.mark.parametrize(
        ("layer", "net"), [("B.Cu", NET), (LAYER, "OTHER")], ids=["layer", "net"]
    )
    def test_a_cut_of_another_layer_or_net_leaves_the_chain_whole(
        self, layer: str, net: str
    ) -> None:
        (chain,) = extract_chains(
            l_z_segments(), frame=BOARD_FRAME, cuts=cut_at((10, 0), layer, net)
        )

        assert len(chain.points) == 4

    def test_a_loop_cut_at_two_points_is_two_pieces_pinned_at_both(self) -> None:
        def cuts(_layer: str, _net: str) -> list[tuple[float, float]]:
            return [(10.0, 0.0), (0.0, 10.0)]

        chains = extract_chains(square_loop(), frame=BOARD_FRAME, cuts=cuts)

        assert len(chains) == 2
        for chain in chains:
            assert {chain.points[0], chain.points[-1]} == {(10, 0), (0, 10)}
            assert chain.pinned[0] and chain.pinned[-1]
            assert not any(chain.pinned[1:-1])
        assert sorted(i for c in chains for i in c.segment_ids) == list("abcd")

    def test_a_cut_at_a_junction_leaves_its_branches_loose(self) -> None:
        a, bc, d = extract_chains(
            tee_segments(), frame=BOARD_FRAME, cuts=cut_at((10, 0))
        )

        assert (a.pinned[-1], bc.pinned[0], d.pinned[0]) == (False, False, False)


def tee_segments() -> list[Segment]:
    """The L-Z route with a branch "d" leaving its corner at (10, 0)."""
    return [*l_z_segments(), seg("d", (10, 0), (15, -5))]


class TestBranches:
    def test_a_tee_splits_into_chains_ending_loose_at_the_junction(self) -> None:
        a, bc, d = extract_chains(tee_segments(), frame=BOARD_FRAME)

        assert_points(a, [(0, 0), (10, 0)])
        assert_points(bc, [(10, 0), (10, 5), (20, 5)])
        assert_points(d, [(10, 0), (15, -5)])
        assert a.pinned == (True, False)
        assert bc.pinned == (False, False, True)
        assert d.pinned == (False, True)
        assert a.points[-1] == bc.points[0] == d.points[0]
        assert [a.segment_ids, sorted(bc.segment_ids), d.segment_ids] == [
            ("a",),
            ["b", "c"],
            ("d",),
        ]

    def test_the_split_does_not_depend_on_the_input_order(self) -> None:
        segments = [
            *tee_segments(),
            seg("e", (10, 0), (5, -5)),
            seg("f", (5, -5), (0, -5)),
        ]
        reordered = [
            Segment(s.id, s.end, s.start, s.width, s.layer, s.net)
            for s in reversed(segments)
        ]

        chains = extract_chains(segments, frame=BOARD_FRAME)
        again = extract_chains(reordered, frame=BOARD_FRAME)

        assert [(c.points, c.pinned) for c in chains] == [
            (c.points, c.pinned) for c in again
        ]
        assert [c.points[0] for c in chains] == [(0, -5), (0, 0), (10, 0), (10, 0)]

    def test_a_pad_on_the_junction_pins_every_branch(self) -> None:
        pad = Point(10, 0).buffer(0.3)

        chains = extract_chains(tee_segments(), lambda _l, _n: [pad], frame=BOARD_FRAME)

        for chain in chains:
            assert (10.0, 0.0) in pinned_points(chain)

    def test_a_locked_track_ending_at_the_junction_pins_it(self) -> None:
        segments = [*tee_segments(), seg("lock", (10, 0), (10, -5), locked=True)]

        chains = extract_chains(segments, frame=BOARD_FRAME)

        assert len(chains) == 3
        for chain in chains:
            assert (10.0, 0.0) in pinned_points(chain)

    def test_branches_of_different_widths_meet_loose(self) -> None:
        segments = [*l_z_segments(), seg("d", (10, 0), (15, -5), width=500_000)]

        a, bc, d = extract_chains(segments, frame=BOARD_FRAME)

        assert (a.width, bc.width, d.width) == (
            pytest.approx(0.25),
            pytest.approx(0.25),
            pytest.approx(0.5),
        )
        assert (a.pinned[-1], bc.pinned[0], d.pinned[0]) == (False, False, False)

    def test_copper_reaching_only_the_widest_branch_pins_every_branch(self) -> None:
        segments = [*l_z_segments(), seg("d", (10, 0), (15, -5), width=500_000)]
        # 0.2 mm from the junction: within the wide track's half width only.
        pad = Point(10, -0.5).buffer(0.3)

        a, bc, d = extract_chains(segments, lambda _l, _n: [pad], frame=BOARD_FRAME)

        assert (a.pinned[-1], bc.pinned[0], d.pinned[0]) == (True, True, True)

    def test_a_width_change_end_to_end_meets_loose(self) -> None:
        segments = [
            seg("w1", (0, 0), (8, 0), width=500_000),
            seg("w2", (8, 0), (12, 4), width=500_000),
            seg("thin", (12, 4), (20, 4)),
        ]

        thin, wide = sorted(
            extract_chains(segments, frame=BOARD_FRAME), key=lambda c: c.width
        )

        assert thin.points[0] == wide.points[-1] == (12.0, 4.0)
        assert thin.pinned == (False, True)
        assert wide.pinned == (True, False, False)

    @pytest.mark.parametrize(
        "copper",
        [
            SameNetTrack(LineString([(12, 4), (12, 8)]), WIDTH_MM),
            box(11.8, 3.8, 12.2, 4.2),
            contact(Point(12, 4).buffer(0.3), (12.0, 4.0)),
        ],
        ids=["unselected track", "pad", "staying via"],
    )
    def test_copper_at_a_width_change_pins_both_ends(
        self, copper: SameNetTrack | Polygon | Contact
    ) -> None:
        segments = [
            seg("wide", (0, 0), (12, 4), width=500_000),
            seg("thin", (12, 4), (20, 4)),
        ]

        thin, wide = sorted(
            extract_chains(segments, lambda _l, _n: [copper], frame=BOARD_FRAME),
            key=lambda c: c.width,
        )

        assert thin.points[0] == wide.points[-1] == (12.0, 4.0)
        assert thin.pinned[0] and wide.pinned[-1]

    def test_a_track_ending_mid_track_splits_it_at_a_junction(self) -> None:
        segments = [*l_z_segments(), seg("tee", (5, -3), (5, 0))]

        left, tee, right = extract_chains(segments, frame=BOARD_FRAME)

        assert_points(left, [(0, 0), (5, 0)])
        assert_points(tee, [(5, -3), (5, 0)])
        assert_points(right, [(5, 0), (10, 0), (10, 5), (20, 5)])
        assert left.pinned == (True, False)
        assert tee.pinned == (True, False)
        assert right.pinned == (False, False, False, True)
        assert left.points[-1] == tee.points[-1] == right.points[0]
        assert [left.segment_ids, tee.segment_ids, sorted(right.segment_ids)] == [
            ("a",),
            ("tee",),
            ["a", "b", "c"],
        ]

    @pytest.mark.parametrize(
        ("start", "tip", "at"),
        [
            ((5, -3), (5, -0.1), (5, 0)),
            ((5, -3), (5, -0.25), (5, 0)),
            ((3.3, 3), (3.3, 0.2), (3.3, 0)),
        ],
    )
    def test_an_end_in_the_tracks_copper_moves_onto_its_centreline(
        self,
        start: tuple[float, float],
        tip: tuple[float, float],
        at: tuple[float, float],
    ) -> None:
        segments = [*l_z_segments(), seg("tee", start, tip)]

        left, tee, right = extract_chains(segments, frame=BOARD_FRAME)

        (tee_end,) = [i for i in (0, -1) if tee.points[i] == at]
        assert left.points[-1] == right.points[0] == at
        assert (left.pinned[-1], tee.pinned[tee_end], right.pinned[0]) == (
            False,
            False,
            False,
        )

    def test_an_end_short_of_the_tracks_copper_stays_apart(self) -> None:
        segments = [*l_z_segments(), seg("tee", (5, -3), (5, -0.26))]

        chains = extract_chains(segments, frame=BOARD_FRAME)

        assert [sorted(c.segment_ids) for c in chains] == [["a", "b", "c"], ["tee"]]

    def test_an_off_centreline_end_snaps_to_the_nearest_point_on_a_diagonal(
        self,
    ) -> None:
        segments = [seg("a", (0, 0), (3, 1)), seg("tee", (1, 3), (1.5, 0.6))]

        left, tee, right = extract_chains(segments, frame=BOARD_FRAME)

        # (1.5, 0.6) projects onto (3, 1) at t = 0.51.
        assert tee.points[-1] == left.points[-1] == right.points[0] == (1.53, 0.51)

    def test_two_stubs_split_one_track_into_three(self) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("s", (3, -3), (3, 0)),
            seg("t", (7, -3), (7, 0)),
        ]

        chains = extract_chains(segments, frame=BOARD_FRAME)

        assert [(c.points, c.segment_ids) for c in chains] == [
            (((0, 0), (3, 0)), ("a",)),
            (((3, -3), (3, 0)), ("s",)),
            (((3, 0), (7, 0)), ("a",)),
            (((7, -3), (7, 0)), ("t",)),
            (((7, 0), (10, 0)), ("a",)),
        ]
        assert [c.pinned for c in chains] == [
            (True, False),
            (True, False),
            (False, False),
            (True, False),
            (False, True),
        ]

    @pytest.mark.parametrize("off", [0, 0.1])
    def test_stubs_from_both_sides_meet_at_one_junction(self, off: float) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("up", (5, -3), (5, -off)),
            seg("down", (5, 3), (5, off)),
        ]

        chains = extract_chains(segments, frame=BOARD_FRAME)

        assert len(chains) == 4
        assert sorted(c.segment_ids for c in chains) == [
            ("a",),
            ("a",),
            ("down",),
            ("up",),
        ]
        for c in chains:
            (end,) = [i for i in (0, -1) if c.points[i] == (5, 0)]
            assert not c.pinned[end]

    def test_the_tracks_ending_at_a_vee_on_a_track_all_join_it(self) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("l", (3, -3), (5, 0)),
            seg("r", (5, 0), (7, -3)),
        ]

        chains = extract_chains(segments, frame=BOARD_FRAME)

        assert sorted(c.segment_ids for c in chains) == [("a",), ("a",), ("l",), ("r",)]

    def test_a_stub_crossing_past_the_centreline_keeps_its_tip_and_pad(
        self,
    ) -> None:
        segments = [
            seg("a", (0, 0), (4, 0)),
            seg("b", (4, 0), (4, 4)),
            seg("t", (2, -3), (2, 0.2)),
        ]
        pad = Point(2, 0.55).buffer(0.25)

        route, stub = extract_chains(
            segments, lambda _layer, _net: [pad], frame=BOARD_FRAME
        )

        assert route.segment_ids == ("a", "b")
        # The route has no end where the stub crosses it, so it pins nothing.
        assert_points(stub, [(2, -3), (2, 0.2)])
        assert stub.pinned == (True, True)

    @pytest.mark.parametrize(
        ("half_length", "width"), [(0.2, WIDTH_NM), (0.1, 500_000)]
    )
    def test_a_short_track_across_another_keeps_both_ends(
        self, half_length: float, width: int
    ) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("t", (5, -half_length), (5, half_length), width=width),
        ]

        a, t = extract_chains(segments, frame=BOARD_FRAME)

        assert a.segment_ids == ("a",)
        assert_points(t, [(5, -half_length), (5, half_length)])
        assert t.pinned == (True, True)
        assert (5, 0) in pinned_points(a)

    def test_an_offset_neck_down_stays_pinned_by_overlap(self) -> None:
        segments = [
            seg("thin", (0, 0), (10, 0)),
            seg("wide", (9.9, 0.01), (20, 0.01), width=500_000),
        ]

        thin, wide = extract_chains(segments, frame=BOARD_FRAME)

        assert (thin.points[0], thin.points[-1]) == ((0, 0), (10, 0))
        assert (wide.points[0], wide.points[-1]) == ((9.9, 0.01), (20, 0.01))
        assert all(thin.pinned[1:]) and all(wide.pinned[:-1])

    @pytest.mark.parametrize(
        "route",
        [
            [((0, 0), (10, 0)), ((10, 0), (9.8, 0.2)), ((9.8, 0.2), (9.8, 5))],
            [((0, 0), (10, 0)), ((10, 0), (10, 0.2)), ((10, 0.2), (1, 0.2))],
        ],
        ids=["acute bend", "hairpin"],
    )
    def test_a_route_reaching_its_own_track_stays_one_chain(
        self, route: list[tuple[tuple[float, float], tuple[float, float]]]
    ) -> None:
        segments = [seg(f"s{i}", a, b) for i, (a, b) in enumerate(route)]

        (chain,) = extract_chains(segments, frame=BOARD_FRAME)

        assert_points(chain, [route[0][0], *(b for _, b in route)])
        assert chain.pinned == (True, False, False, True)

    def test_two_stubs_ending_close_inside_one_track_both_join_it(self) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("t", (5, -3), (5, -0.2)),
            seg("t2", (5.1, -0.2), (5.1, -3)),
        ]

        chains = extract_chains(segments, frame=BOARD_FRAME)

        assert [(c.points[0], c.points[-1], c.segment_ids) for c in chains] == [
            ((0, 0), (5, 0), ("a",)),
            ((5, -3), (5, 0), ("t",)),
            ((5, 0), (5.1, 0), ("a",)),
            ((5.1, -3), (5.1, 0), ("t2",)),
            ((5.1, 0), (10, 0), ("a",)),
        ]

    @pytest.mark.parametrize(("width", "joins"), [(500_000, True), (WIDTH_NM, False)])
    def test_the_stubs_own_width_counts_in_its_reach(
        self, width: int, joins: bool
    ) -> None:
        segments = [
            seg("a", (0, 0), (10, 0)),
            seg("t", (5, -3), (5, -0.3), width=width),
        ]

        chains = extract_chains(segments, frame=BOARD_FRAME)

        assert len(chains) == (3 if joins else 2)

    @pytest.mark.parametrize(
        ("stub", "width", "pad"),
        [
            (((2, -3), (5, -0.3)), 500_000, Point(5.4, -0.4).buffer(0.2)),
            (((5, 0.2), (0, 3)), WIDTH_NM, box(5.1, 0.15, 5.5, 0.4)),
        ],
    )
    def test_an_end_that_other_copper_reaches_keeps_its_tip(
        self,
        stub: tuple[tuple[float, float], tuple[float, float]],
        width: int,
        pad: Polygon,
    ) -> None:
        # The pad touches only the copper of the stub's tip.
        segments = [seg("a", (0, 0), (10, 0)), seg("t", *stub, width=width)]

        a, t = sorted(
            extract_chains(segments, lambda _layer, _net: [pad], frame=BOARD_FRAME),
            key=lambda c: c.segment_ids,
        )

        assert a.segment_ids == ("a",)
        (tip,) = [i for i in (0, -1) if t.points[i] == stub[1]]
        assert t.pinned[tip]
        assert LineString(t.points).buffer(t.width / 2).intersects(pad)

    @pytest.mark.parametrize("off", [0, 0.1])
    def test_a_route_ending_on_its_own_track_stays_one_chain(self, off: float) -> None:
        route: list[tuple[float, float]] = [(0, 0), (10, 0), (10, 5), (5, 5), (5, off)]
        segments = [seg(f"s{i}", a, b) for i, (a, b) in enumerate(pairwise(route))]

        (chain,) = extract_chains(segments, frame=BOARD_FRAME)

        assert_points(chain, route)
        assert chain.pinned == (True, False, False, False, True)

    @pytest.mark.parametrize("off", [0, 0.1])
    def test_a_track_running_on_past_anothers_end_is_pinned_by_overlap(
        self, off: float
    ) -> None:
        segments = [seg("a", (0, 0), (10, 0)), seg("s", (9.9, off), (20, 0.5))]

        a, s = extract_chains(segments, frame=BOARD_FRAME)

        # Each pins the other where its end reaches it.
        assert_points(a, [(0, 0), (9.9, 0), (10, 0)])
        assert a.pinned == (True, True, True)
        assert (s.points[0], s.points[-1]) == ((9.9, off), (20, 0.5))
        assert len(s.points) == 3
        assert all(s.pinned)

    @pytest.mark.parametrize(
        ("stub", "joins"),
        [
            (((5, 5), (5, -0.05)), True),
            (((10.14, 5), (5.06, -0.08)), True),
            (((5, 5), (5, -0.15)), False),
        ],
        ids=["overshoot", "diagonal overshoot", "past the half width"],
    )
    def test_a_stub_ending_past_the_centreline_joins_within_the_half_width(
        self, stub: tuple[tuple[float, float], tuple[float, float]], joins: bool
    ) -> None:
        segments = [seg("a", (0, 0), (10, 0)), seg("s", *stub)]

        chains = extract_chains(segments, frame=BOARD_FRAME)

        assert len(chains) == (3 if joins else 2)
        (s,) = [c for c in chains if c.segment_ids == ("s",)]
        assert (0 in (s.points[0][1], s.points[-1][1])) == joins


Span = tuple[tuple[float, float], tuple[float, float]]


class TestWidthsOverlapping:
    """Tracks of two widths whose copper overlaps along one line stay
    separate chains of their own widths, pinned where they overlap."""

    def test_a_thin_route_running_inside_a_wide_track(self) -> None:
        segments = [
            seg("w", (0, 0), (10, 0), width=500_000),
            seg("t", (3, 0.05), (6, 0.05)),
            seg("u", (6, 0.05), (6, 5)),
        ]

        chains = {c.segment_ids: c for c in extract_chains(segments, frame=BOARD_FRAME)}

        assert set(chains) == {("w",), ("t", "u")}
        w, tu = chains[("w",)], chains[("t", "u")]
        assert (w.width, tu.width) == (0.5, 0.25)
        assert (w.points[0], w.points[-1]) == ((0, 0), (10, 0))
        assert any(w.pinned[1:-1])
        assert (tu.points[0], tu.points[-1]) == ((3, 0.05), (6, 5))
        assert tu.pinned[0] and tu.pinned[-1]

    @pytest.mark.parametrize(
        ("thin", "wide"),
        [
            ([((0, 0), (10, 0))], [((9.775, 0), (20, 0))]),
            ([((9.775, 0), (20, 0))], [((0, 0), (10, 0))]),
            ([((0, 0), (10, 0))], [((9.775, 0), (10, 0)), ((10, 0), (20, 0))]),
            ([((0, 0), (10, 0))], [((4, 0), (6, 0))]),
        ],
        ids=["neck-down", "neck-up", "wide segment on the overlap", "wide patch"],
    )
    def test_a_neck_down_overlapping_along_its_line(
        self, thin: list[Span], wide: list[Span]
    ) -> None:
        segments = [
            *(seg(f"t{i}", a, b) for i, (a, b) in enumerate(thin)),
            *(seg(f"w{i}", a, b, width=500_000) for i, (a, b) in enumerate(wide)),
        ]
        overlap = LineString(thin[0]).intersection(LineString(wide[0]))
        start, end = overlap.coords[0], overlap.coords[-1]

        chains = extract_chains(segments, frame=BOARD_FRAME)

        for c in chains:
            assert {i[0] for i in c.segment_ids} == {"t" if c.width == 0.25 else "w"}
        for width in (0.25, 0.5):
            pins = {
                p
                for c in chains
                if c.width == width
                for p, pin in zip(c.points, c.pinned, strict=True)
                if pin
            }
            assert start in pins or end in pins


ARC_CENTRE = (5.0, 3.0)
ARC_RADIUS = 3.0
ARC_MID = (
    ARC_CENTRE[0] + ARC_RADIUS * math.sqrt(0.5),
    ARC_CENTRE[1] - ARC_RADIUS * math.sqrt(0.5),
)


def quarter_arc(*, locked: bool = False) -> list[Segment]:
    """The segments of a 90° arc from (5, 0) to (8, 3) through ARC_MID."""
    return Arc(
        "arc", nm(5, 0), nm(*ARC_MID), nm(8, 3), WIDTH_NM, LAYER, NET, locked
    ).segments()


class TestArcs:
    """A selected arc is a track split into several edges of one id."""

    def test_track_arc_track_is_one_chain(self) -> None:
        segments = [seg("a", (0, 0), (5, 0)), *quarter_arc(), seg("c", (8, 3), (8, 8))]

        (chain,) = extract_chains(segments, frame=BOARD_FRAME)

        assert sorted(chain.segment_ids) == ["a", "arc", "c"]
        assert (chain.points[0], chain.points[-1]) == ((0, 0), (8, 8))
        assert chain.pinned[0] and chain.pinned[-1] and not any(chain.pinned[1:-1])

    def test_locked_arc_splits_and_pins_where_it_touches(self, log: list[str]) -> None:
        segments = [
            seg("a", (0, 0), (5, 0)),
            *quarter_arc(locked=True),
            seg("c", (8, 3), (8, 8)),
        ]

        first, second = extract_chains(segments, frame=BOARD_FRAME)

        assert_points(first, [(0, 0), (5, 0)])
        assert_points(second, [(8, 3), (8, 8)])
        assert first.pinned == second.pinned == (True, True)
        (warning,) = [line for line in log if line.startswith("WARNING")]
        assert warning.count("arc") == 1

    def test_track_ending_mid_arc_joins_it_there(self) -> None:
        # 2° past the vertex at ARC_MID, so partway along a 5° piece.
        angle = math.radians(-45 + 2)
        on_arc = (
            ARC_CENTRE[0] + ARC_RADIUS * math.cos(angle),
            ARC_CENTRE[1] + ARC_RADIUS * math.sin(angle),
        )
        arc = quarter_arc()
        vertices = {
            (x / NM_PER_MM, y / NM_PER_MM) for s in arc for x, y in (s.start, s.end)
        }
        segments = [
            seg("a", (0, 0), (5, 0)),
            *arc,
            seg("c", (8, 3), (8, 8)),
            Segment("tee", nm(10, -2), nm(*on_arc), WIDTH_NM, LAYER, NET),
        ]

        chains = extract_chains(segments, frame=BOARD_FRAME)

        assert sorted(sorted(c.segment_ids) for c in chains) == [
            ["a", "arc"],
            ["arc", "c"],
            ["tee"],
        ]
        (tee,) = [c for c in chains if c.segment_ids == ("tee",)]
        joint = tee.points[0] if tee.points[-1] == (10, -2) else tee.points[-1]
        assert Point(joint).distance(Point(on_arc)) < WIDTH_MM / 2
        assert joint not in vertices
        assert all(joint in (c.points[0], c.points[-1]) for c in chains)
        groups = shared_track_groups([c.segment_ids for c in chains])
        arc_groups = {
            g for g, c in zip(groups, chains, strict=True) if "arc" in c.segment_ids
        }
        assert len(arc_groups) == 1 and len(set(groups)) == 2
