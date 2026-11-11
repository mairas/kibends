from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import kipy.board
import kipy.board_types as kbt
import pytest
from conftest import (
    BOARD_FRAME,
    FakeBoard,
    extract_chains,
    kipy_copper_zone,
    kipy_pad,
    kipy_rect,
    kipy_track,
    kipy_via,
    kipy_zone,
)
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.board.board_types_pb2 import ViaType, ZoneType
from kipy.wrapper import Wrapper

from kibends.chain import Chain
from kibends.coords import UserFrame
from kibends.snapshot import (
    KICAD_MAX_ERROR,
    BoardSnapshot,
    Via,
    read_selection,
)
from kibends.vias import (
    ChainEnd,
    MovableVia,
    PairKind,
    StaticVia,
    carried_copper,
    chain_cuts,
    may_move,
    movable_vias,
    pair_kind,
    pairs_with,
)

VIA = (4.0, 4.0)
CORNER = (4.0, 0.0)
"""Where `front` turns."""


def front() -> list[kbt.Track]:
    return [
        kipy_track("a", "SIG", (0, 0), (4, 0)),
        kipy_track("b", "SIG", (4, 0), VIA),
    ]


def back() -> list[kbt.Track]:
    return [
        kipy_track("e", "SIG", VIA, (8, 4), layer=BL.BL_B_Cu),
        kipy_track("f", "SIG", (8, 4), (8, 8), layer=BL.BL_B_Cu),
    ]


def inner() -> list[kbt.Track]:
    return [
        kipy_track("g", "SIG", VIA, (8, 4), layer=BL.BL_In1_Cu),
        kipy_track("h", "SIG", (8, 4), (8, 8), layer=BL.BL_In1_Cu),
    ]


def blind(via_id: str = "v", net: str = "SIG") -> kbt.Via:
    """A blind via at `VIA` from F.Cu to In1.Cu."""
    return kipy_via(
        via_id, net, *VIA, via_type=ViaType.VT_BLIND, span=(BL.BL_F_Cu, BL.BL_In1_Cu)
    )


def judge(
    selection: list[Wrapper],
    tracks: list[kbt.Track],
    vias: list[kbt.Via],
    *,
    pads: Sequence[kbt.Pad] = (),
    zones: Sequence[kbt.Zone] = (),
    frame: UserFrame = BOARD_FRAME,
) -> tuple[list[MovableVia], list[Chain]]:
    """The selection's chains and movable vias, read as a run reads them
    before it knows which vias move: cut at every selected via that may."""
    fake = cast(
        kipy.board.Board,
        FakeBoard(
            selection=selection,
            tracks=list[kbt.Track | kbt.ArcTrack](tracks),
            vias=vias,
            pads=list(pads),
            zones=list(zones),
        ),
    )
    chosen = read_selection(fake)
    replaced = {s.id for s in chosen.segments if not s.locked}
    layers = {s.layer for s in chosen.segments if not s.locked}
    cuts = chain_cuts([v for v in chosen.vias if may_move(v, chosen.nets)])
    bare = extract_chains(
        chosen.segments, lambda layer, net: (), frame=BOARD_FRAME, cuts=cuts
    )
    whole = BoardSnapshot.read(
        fake, layers, chosen.vias, carried=carried_copper(chosen.vias, bare)
    )
    snapshots = {layer: whole.slice(layer, replaced) for layer in layers}
    chains = extract_chains(
        chosen.segments,
        lambda layer, net: snapshots[layer].same_net_copper(net),
        frame=BOARD_FRAME,
        cuts=cuts,
    )
    movable = movable_vias(
        chosen.vias,
        whole.via_contacts,
        chains,
        frame=frame,
        zones=whole.via_zones,
        nets=chosen.nets,
    )
    return movable, chains


def pinned_points(chain: Chain) -> list[tuple[float, float]]:
    return [p for p, pin in zip(chain.points, chain.pinned, strict=True) if pin]


def warnings(log: list[str]) -> list[str]:
    """The log lines saying why a via stays fixed, which are debug lines."""
    return [line for line in log if "stays fixed" in line]


def fixed(log: list[str], v: kbt.Via) -> str:
    """The one warning that `v`, joining `front` and `back`, stays fixed."""
    movable, _ = judge([*front(), *back(), v], [*front(), *back()], [v])
    assert movable == []
    (warning,) = warnings(log)
    assert "Via v at (4.000, 4.000)" in warning and "stays fixed" in warning
    return warning


class TestMovableVias:
    def test_through_via_joining_two_selected_chain_ends_is_movable(
        self, log: list[str]
    ) -> None:
        v = kipy_via("v", "SIG", *VIA)

        movable, chains = judge([*front(), *back(), v], [*front(), *back()], [v])

        assert [c.layer for c in chains] == ["B.Cu", "F.Cu"]
        (via,) = movable
        assert via.via.id == "v"
        assert via.ends == (ChainEnd(chain=0, last=False), ChainEnd(chain=1, last=True))
        assert warnings(log) == []

    def test_track_split_at_a_junction_counts_for_its_part_at_the_via(
        self, log: list[str]
    ) -> None:
        split = [
            kipy_track("a", "SIG", VIA, (4, 10)),
            kipy_track("s", "SIG", (0, 7), (4, 7)),
        ]
        v = kipy_via("v", "SIG", *VIA)

        movable, _ = judge([*split, *back(), v], [*split, *back()], [v])

        assert [len(m.ends) for m in movable] == [2]
        assert warnings(log) == []

    @pytest.mark.parametrize(
        ("gap", "joins"),
        [
            # The via's radius 0.3 plus the track's half width 0.125, less a hair.
            (0.42, True),
            (0.43, False),
        ],
    )
    def test_chain_end_counts_while_its_copper_overlaps_the_via(
        self, log: list[str], gap: float, joins: bool
    ) -> None:
        short = [
            kipy_track("a", "SIG", (0, 0), (4, 0)),
            kipy_track("b", "SIG", (4, 0), (VIA[0], VIA[1] - gap)),
        ]
        v = kipy_via("v", "SIG", *VIA)
        # A plane of its net, so that the via is valid with one end too.
        zone = kipy_copper_zone(
            "plane", "SIG", kipy_rect(3, 3, 5, 5), [BL.BL_In1_Cu, BL.BL_In2_Cu]
        )

        movable, _ = judge([*short, *back(), v], [*short, *back()], [v], zones=[zone])

        assert [len(m.ends) for m in movable] == [2 if joins else 1]
        assert warnings(log) == []

    def test_locked_via_stays_fixed(self, log: list[str]) -> None:
        assert "it is locked" in fixed(log, kipy_via("v", "SIG", *VIA, locked=True))

    @pytest.mark.parametrize(
        ("net", "partner"), [("USB_D+", "USB_D-"), ("LVDS_N", "LVDS_P")]
    )
    def test_via_of_a_diff_pair_net_stays_fixed(
        self, log: list[str], net: str, partner: str
    ) -> None:
        # With the other net on the board but none of its tracks selected.
        tracks = [
            kipy_track("a", net, (0, 0), VIA),
            kipy_track("e", net, VIA, (8, 4), layer=BL.BL_B_Cu),
        ]
        far = kipy_track("far", partner, (20, 20), (24, 20))
        v = kipy_via("v", net, *VIA)

        movable, _ = judge([*tracks, v], [*tracks, far], [v])

        assert movable == []
        (warning,) = warnings(log)
        assert f"it is on {net}, a net of a diff pair" in warning

    @pytest.mark.parametrize("net", ["VIN", "RESET_N", "/PWR_EN", "VBAT+"])
    def test_via_named_like_a_diff_pair_without_its_partner_moves(
        self, log: list[str], net: str
    ) -> None:
        # KiCad pairs a net only with a complement that exists.
        tracks = [
            kipy_track("a", net, (0, 0), VIA),
            kipy_track("e", net, VIA, (8, 4), layer=BL.BL_B_Cu),
        ]
        v = kipy_via("v", net, *VIA)

        movable, _ = judge([*tracks, v], tracks, [v])

        assert [m.via.id for m in movable] == ["v"]
        assert warnings(log) == []

    def test_warning_gives_the_position_in_the_frame(self, log: list[str]) -> None:
        v = kipy_via("v", "SIG", *VIA, locked=True)

        judge(
            [*front(), *back(), v],
            [*front(), *back()],
            [v],
            frame=UserFrame((1.0, 5.0)),
        )

        (warning,) = warnings(log)
        assert "Via v at (3.000, -1.000) stays fixed" in warning

    def test_blind_via_joining_an_end_on_each_of_its_layers_moves(
        self, log: list[str]
    ) -> None:
        v = blind()

        movable, chains = judge([*front(), *inner(), v], [*front(), *inner()], [v])

        assert [c.layer for c in chains] == ["F.Cu", "In1.Cu"]
        (via,) = movable
        assert via.ends == (ChainEnd(chain=0, last=True), ChainEnd(chain=1, last=False))
        assert warnings(log) == []

    @pytest.mark.parametrize("selected", [False, True])
    def test_track_ending_at_a_blind_via_on_a_layer_it_misses_is_ignored(
        self, log: list[str], selected: bool
    ) -> None:
        v = blind()
        under = kipy_track("u", "SIG", VIA, (4, 9), layer=BL.BL_B_Cu)
        route = [*front(), *inner()]

        movable, chains = judge(
            [*route, *([under] if selected else []), v], [*route, under], [v]
        )

        (via,) = movable
        carried = {chains[end.chain].layer for end in via.ends}
        assert carried == {"F.Cu", "In1.Cu"}
        assert warnings(log) == []

    def test_pad_under_a_blind_via_on_a_layer_it_misses_leaves_it_free(
        self, log: list[str]
    ) -> None:
        v = blind()
        pad = kipy_pad("p", "SIG", *VIA, layers=(BL.BL_B_Cu,))
        route = [*front(), *inner()]

        movable, _ = judge([*route, v], route, [v], pads=[pad])

        assert [m.via.id for m in movable] == ["v"]
        assert warnings(log) == []

    def test_unselected_third_track_pins_it(self, log: list[str]) -> None:
        v = kipy_via("v", "SIG", *VIA)
        third = kipy_track("g", "SIG", VIA, (4, 9), layer=BL.BL_In1_Cu)

        movable, _ = judge([*front(), *back(), v], [*front(), *back(), third], [v])

        assert movable == []
        (warning,) = warnings(log)
        assert "track g on In1.Cu" in warning

    def test_selected_locked_track_pins_it_without_being_called_unselected(
        self, log: list[str]
    ) -> None:
        v = kipy_via("v", "SIG", *VIA)
        e, f = back()
        e.locked = True

        movable, _ = judge([*front(), e, f, v], [*front(), e, f], [v])

        assert movable == []
        (warning,) = [w for w in warnings(log) if "stays fixed" in w]
        assert "stays fixed: track e on B.Cu connects to it" in warning

    def test_same_net_pad_under_it_pins_it(self, log: list[str]) -> None:
        v = kipy_via("v", "SIG", *VIA)
        pad = kipy_pad("p", "SIG", *VIA, layers=(BL.BL_B_Cu,))

        movable, _ = judge([*front(), *back(), v], [*front(), *back()], [v], pads=[pad])

        assert movable == []
        (warning,) = warnings(log)
        assert "pad p" in warning

    def test_same_net_zone_leaves_it_movable_inside_the_zone(
        self, log: list[str]
    ) -> None:
        v = kipy_via("v", "SIG", *VIA)
        zone = kipy_copper_zone(
            "plane", "SIG", kipy_rect(3, 3, 5, 5), [BL.BL_In1_Cu, BL.BL_In2_Cu]
        )

        movable, _ = judge(
            [*front(), *back(), v], [*front(), *back()], [v], zones=[zone]
        )

        (via,) = movable
        (inside,) = via.zones
        assert inside.description == "zone plane on In1.Cu and In2.Cu"
        assert warnings(log) == []

    def test_zone_and_unselected_track_fix_it_naming_only_the_track(
        self, log: list[str]
    ) -> None:
        v = kipy_via("v", "SIG", *VIA)
        zone = kipy_copper_zone("plane", "SIG", kipy_rect(3, 3, 5, 5), [BL.BL_In2_Cu])
        stub = kipy_track("s", "SIG", VIA, (4, 6), layer=BL.BL_In1_Cu)

        movable, _ = judge(
            [*front(), *back(), v], [*front(), *back(), stub], [v], zones=[zone]
        )

        assert movable == []
        (warning,) = warnings(log)
        assert warning.rstrip().endswith(
            "stays fixed: track s on In1.Cu connects to it"
        )

    @pytest.mark.parametrize(("past", "moves"), [(0.001, True), (-0.001, False)])
    def test_moves_only_with_its_centre_deep_enough_in_the_region(
        self, log: list[str], past: float, moves: bool
    ) -> None:
        # The region ends inside the ring; the refill still connects the via
        # when its centre lies half the zone's minimum width and the arc
        # error inside. Short of that, it may be connected now, and moving
        # could lose that.
        v = kipy_via("v", "SIG", *VIA)
        min_width = 0.2
        edge = VIA[0] + min_width / 2 + KICAD_MAX_ERROR + past
        zone = kipy_copper_zone(
            "plane",
            "SIG",
            kipy_rect(2, 2, edge, 6),
            [BL.BL_In2_Cu],
            min_width=min_width,
        )

        movable, _ = judge(
            [*front(), *back(), v], [*front(), *back()], [v], zones=[zone]
        )

        assert len(movable) == int(moves)
        if moves:
            assert warnings(log) == []
        else:
            (warning,) = warnings(log)
            assert warning.rstrip().endswith(
                "stays fixed: it sits at the edge of zone plane on In2.Cu"
            )

    @pytest.mark.parametrize(("short", "moves"), [(0.1, False), (0.31, True)])
    def test_a_zone_reaching_only_its_ring_fixes_it(
        self, log: list[str], short: float, moves: bool
    ) -> None:
        # The zone ends `short` of the centre of the via, of radius 0.3:
        # reaching the ring, it may connect the via now.
        v = kipy_via("v", "SIG", *VIA)
        edge = VIA[0] - short
        zone = kipy_copper_zone(
            "plane", "SIG", kipy_rect(2, 2, edge, 6), [BL.BL_In2_Cu]
        )

        movable, _ = judge(
            [*front(), *back(), v], [*front(), *back()], [v], zones=[zone]
        )

        if moves:
            (via,) = movable
            assert via.zones == ()
            assert warnings(log) == []
        else:
            assert movable == []
            (warning,) = warnings(log)
            assert warning.rstrip().endswith(
                "stays fixed: it sits at the edge of zone plane on In2.Cu"
            )

    def test_a_zone_the_refill_keeps_off_it_leaves_it_free(
        self, log: list[str]
    ) -> None:
        # A rule area keeping out pour over the via leaves the zone no region
        # there.
        v = kipy_via("v", "SIG", *VIA)
        zone = kipy_copper_zone("plane", "SIG", kipy_rect(3, 3, 5, 5), [BL.BL_In2_Cu])
        area = kipy_zone(kipy_rect(2, 2, 6, 6), [BL.BL_In2_Cu], keepout_tracks=False)
        area.proto.rule_area_settings.keepout_copper = True

        movable, _ = judge(
            [*front(), *back(), v], [*front(), *back()], [v], zones=[zone, area]
        )

        (via,) = movable
        assert via.zones == ()
        assert warnings(log) == []

    def test_teardrop_leaves_it_movable(self, log: list[str]) -> None:
        v = kipy_via("v", "SIG", *VIA)
        teardrop = kipy_copper_zone(
            "drop",
            "SIG",
            kipy_rect(3.9, 3.0, 4.1, 3.8),
            [BL.BL_F_Cu],
            ZoneType.ZT_TEARDROP,
        )

        movable, _ = judge(
            [*front(), *back(), v], [*front(), *back()], [v], zones=[teardrop]
        )

        assert [m.via.id for m in movable] == ["v"]
        assert warnings(log) == []

    def test_other_net_and_distant_zones_leave_it_movable(self, log: list[str]) -> None:
        v = kipy_via("v", "SIG", *VIA)
        zones = [
            kipy_copper_zone("gnd", "GND", kipy_rect(3, 3, 5, 5), [BL.BL_In1_Cu]),
            kipy_copper_zone("far", "SIG", kipy_rect(10, 10, 12, 12), [BL.BL_In2_Cu]),
        ]

        movable, _ = judge(
            [*front(), *back(), v], [*front(), *back()], [v], zones=zones
        )

        assert [m.via.id for m in movable] == ["v"]
        assert movable[0].zones == ()
        assert warnings(log) == []

    def test_via_whose_tracks_are_not_selected_stays_fixed(
        self, log: list[str]
    ) -> None:
        v = kipy_via("v", "SIG", *VIA)
        elsewhere = kipy_track("x", "NET2", (0, 20), (5, 20))

        movable, _ = judge([elsewhere, v], [*front(), *back(), elsewhere], [v])

        assert movable == []
        (warning,) = warnings(log)
        assert "stays fixed: track b on F.Cu, track e on B.Cu connect to it" in warning

    def test_via_no_selected_track_ends_at_stays_fixed(self, log: list[str]) -> None:
        v = kipy_via("v", "SIG", *VIA)
        elsewhere = kipy_track("x", "NET2", (0, 20), (5, 20))

        movable, _ = judge([elsewhere, v], [elsewhere], [v])

        assert movable == []
        (warning,) = warnings(log)
        assert warning.rstrip().endswith(
            "stays fixed: no selected track ends at it and no zone of its net "
            "reaches it"
        )

    def test_via_ending_one_selected_chain_in_a_zone_of_its_net_moves(
        self, log: list[str]
    ) -> None:
        # A pad's fan-out to a plane: one track down to the via, the plane
        # on the inner layers.
        v = kipy_via("v", "SIG", *VIA)
        zone = kipy_copper_zone(
            "plane", "SIG", kipy_rect(3, 3, 5, 5), [BL.BL_In1_Cu, BL.BL_In2_Cu]
        )

        movable, _ = judge([*front(), v], front(), [v], zones=[zone])

        (via,) = movable
        assert via.ends == (ChainEnd(chain=0, last=True),)
        (inside,) = via.zones
        assert inside.description == "zone plane on In1.Cu and In2.Cu"
        assert warnings(log) == []

    def test_via_a_selected_chain_passes_moves_carrying_both_sides(
        self, log: list[str]
    ) -> None:
        v = kipy_via("v", "SIG", *CORNER)

        movable, chains = judge([*front(), v], front(), [v])

        (via,) = movable
        assert via.ends == (ChainEnd(chain=0, last=True), ChainEnd(chain=1, last=False))
        assert [c.points[-1] for c in chains[:1]] == [CORNER]
        assert chains[1].points[0] == CORNER
        assert all(c.pinned[0] and c.pinned[-1] for c in chains)
        assert warnings(log) == []

    def test_via_a_chain_passes_carries_a_track_of_another_layer_too(
        self, log: list[str]
    ) -> None:
        v = kipy_via("v", "SIG", *CORNER)
        down = kipy_track("g", "SIG", CORNER, (0, 4), layer=BL.BL_B_Cu)

        movable, chains = judge([*front(), down, v], [*front(), down], [v])

        (via,) = movable
        assert [(chains[e.chain].layer, e.last) for e in via.ends] == [
            ("B.Cu", True),
            ("F.Cu", True),
            ("F.Cu", False),
        ]
        assert warnings(log) == []

    @pytest.mark.parametrize("locked", [False, True], ids=["unselected", "locked"])
    def test_a_chain_through_a_via_that_may_not_move_stays_whole(
        self, log: list[str], locked: bool
    ) -> None:
        v = kipy_via("v", "SIG", *CORNER, locked=locked)

        movable, chains = judge([*front(), *([v] if locked else [])], front(), [v])

        assert movable == []
        (chain,) = chains
        assert CORNER in pinned_points(chain)
        assert len(warnings(log)) == int(locked)

    def test_an_unselected_track_ending_at_a_via_a_chain_passes_holds_it(
        self, log: list[str]
    ) -> None:
        v = kipy_via("v", "SIG", *CORNER)
        stub = kipy_track("s", "SIG", CORNER, (8, 0))

        movable, _ = judge([*front(), v], [*front(), stub], [v])

        assert movable == []
        (warning,) = warnings(log)
        assert warning.rstrip().endswith("stays fixed: track s on F.Cu connects to it")

    def test_via_a_selected_chain_passes_without_a_vertex_at_its_centre_stays(
        self, log: list[str]
    ) -> None:
        v = kipy_via("v", "SIG", 2, 0)

        movable, _ = judge([*front(), v], front(), [v])

        assert movable == []
        (warning,) = warnings(log)
        assert "track a on F.Cu" in warning

    @pytest.mark.parametrize("selected", [False, True])
    def test_unselected_via_holds_the_chain_as_before(
        self, log: list[str], selected: bool
    ) -> None:
        # The chain passes over it, so it keeps one vertex inside it.
        v = kipy_via("v", "SIG", 2, 0)

        movable, chains = judge([*front(), *([v] if selected else [])], front(), [v])

        assert movable == []
        (chain,) = chains
        assert pinned_points(chain) == [(0, 0), VIA]
        assert [chain.points[i] for i, _ in chain.slides] == [pytest.approx((2, 0))]
        assert len(warnings(log)) == int(selected)


def sig_zone(
    name: str,
    corners: tuple[float, float, float, float],
    layers: Sequence[BL.ValueType] = (BL.BL_In2_Cu,),
    min_width: float = 0.0,
) -> kbt.Zone:
    """A zone of the net of `front` and `back`, a rectangle by its corners."""
    return kipy_copper_zone(
        name, "SIG", kipy_rect(*corners), list(layers), min_width=min_width
    )


def judged(*zones: kbt.Zone) -> list[MovableVia]:
    """`judge` of a via at `VIA` joining `front` and `back`, with `zones`."""
    v = kipy_via("v", "SIG", *VIA)
    movable, _ = judge([*front(), *back(), v], [*front(), *back()], [v], zones=zones)
    return movable


def fixed_reason(log: list[str]) -> str:
    (warning,) = warnings(log)
    return warning.rstrip().split("stays fixed: ")[1]


class TestZonesOfItsNetOnOneLayer:
    """Zones of the via's net that overlap or meet on a layer count as one:
    the refill connects the via through any of them."""

    def test_one_deep_in_a_zone_whose_ring_another_crosses_moves(
        self, log: list[str]
    ) -> None:
        # The edge of `small` lies 0.1 mm short of the centre, inside the
        # ring; `big` holds the via deep inside.
        big = sig_zone("big", (2, 2, 6, 6))
        small = sig_zone("small", (2.5, 2.5, 3.9, 5.5))

        movable = judged(big, small)

        (via,) = movable
        assert [z.name for z in via.zones] == ["big", "small"]
        assert warnings(log) == []

    def test_one_on_the_seam_of_two_zones_meeting_edge_to_edge_moves(
        self, log: list[str]
    ) -> None:
        left = sig_zone("left", (2, 2, VIA[0], 6))
        right = sig_zone("right", (VIA[0], 2, 6, 6))

        movable = judged(left, right)

        (via,) = movable
        assert [z.name for z in via.zones] == ["left", "right"]
        assert warnings(log) == []

    def test_one_near_the_outer_edge_of_their_union_stays_fixed(
        self, log: list[str]
    ) -> None:
        # Both end inside the ring, the union too.
        left = sig_zone("left", (2, 2, 3.8, 6))
        over = sig_zone("over", (3, 2, 3.9, 6))

        assert judged(left, over) == []
        assert fixed_reason(log) == (
            "it sits at the edge of zones left and over on In2.Cu"
        )

    @pytest.mark.parametrize("swapped", [False, True])
    @pytest.mark.parametrize(("wide", "moves"), [(0.2, True), (0.3, False)])
    def test_on_a_seam_the_widest_minimum_width_among_them_sets_the_depth(
        self, log: list[str], wide: float, moves: bool, swapped: bool
    ) -> None:
        # On the seam the via lies in neither zone deep enough alone; their
        # union ends 0.13 mm past the centre: deep enough for a minimum
        # width of 0.2 mm, not for 0.3 mm, whichever zone has it.
        edge = VIA[1] + 0.13
        narrow = sig_zone("narrow", (2, 2, VIA[0], edge), min_width=0.1)
        wider = sig_zone("wider", (VIA[0], 2, 6, edge), min_width=wide)
        zones = (wider, narrow) if swapped else (narrow, wider)

        assert len(judged(*zones)) == int(moves)

    @pytest.mark.parametrize("swapped", [False, True])
    def test_one_deep_enough_in_a_zone_by_its_own_margin_moves(
        self, log: list[str], swapped: bool
    ) -> None:
        # `narrow` forms the union's edge 0.13 mm past the centre, deep
        # enough for its own minimum width of 0.1 mm; `wider` lies inside
        # it, its edge inside the ring, with a minimum width too wide for
        # that depth.
        edge = VIA[0] + 0.13
        narrow = sig_zone("narrow", (2, 2, edge, 6), min_width=0.1)
        wider = sig_zone("wider", (3, 3, VIA[0] + 0.05, 5), min_width=0.3)
        zones = (wider, narrow) if swapped else (narrow, wider)

        (via,) = judged(*zones)
        assert {z.name for z in via.zones} == {"narrow", "wider"}
        assert warnings(log) == []

    def test_logs_once_per_layer_the_zones_it_stays_inside(
        self, log: list[str]
    ) -> None:
        both = [BL.BL_In1_Cu, BL.BL_In2_Cu]
        left = sig_zone("left", (2, 2, VIA[0] + 0.5, 6), both)
        right = sig_zone("right", (VIA[0], 2, 6, 6), [BL.BL_In2_Cu])

        assert len(judged(left, right)) == 1
        assert [line.rstrip() for line in log if "stays inside" in line] == [
            "DEBUG Via v stays inside zone left on In1.Cu",
            "DEBUG Via v stays inside zones left and right on In2.Cu",
        ]

    def test_names_each_group_of_zones_at_whose_edge_it_sits(
        self, log: list[str]
    ) -> None:
        # Alone on In1.Cu, `small` sits at the via's edge there; on In2.Cu
        # `other` does too, and so does their union.
        small = sig_zone("small", (2.5, 2.5, 3.9, 5.5), [BL.BL_In1_Cu, BL.BL_In2_Cu])
        other = sig_zone("other", (3, 3, 3.95, 5), [BL.BL_In2_Cu])

        assert judged(small, other) == []
        assert fixed_reason(log) == (
            "it sits at the edge of zone small on In1.Cu; "
            "zones small and other on In2.Cu"
        )

    def test_one_between_two_disjoint_zones_reaching_its_ring_stays_fixed(
        self, log: list[str]
    ) -> None:
        left = sig_zone("left", (2, 2, VIA[0] - 0.2, 6))
        right = sig_zone("right", (VIA[0] + 0.2, 2, 6, 6))

        assert judged(left, right) == []
        assert fixed_reason(log) == (
            "it sits at the edge of zones left and right on In2.Cu"
        )

    @pytest.mark.parametrize(("split_at", "moves"), [(5.0, True), (4.1, False)])
    def test_one_in_a_zone_another_net_splits_is_judged_by_its_piece(
        self, log: list[str], split_at: float, moves: bool
    ) -> None:
        # A higher-priority GND strip across `plane` cuts its region in two;
        # the via lies inside the lower piece, deep or 0.1 mm from its edge,
        # short of the margin of a minimum width of 0.3 mm.
        plane = sig_zone("plane", (2, 2, 6, 6), min_width=0.3)
        strip = kipy_copper_zone(
            "strip", "GND", kipy_rect(1, split_at, 7, split_at + 0.2), [BL.BL_In2_Cu]
        )
        strip.priority = 1

        movable = judged(plane, strip)

        assert len(movable) == int(moves)
        if moves:
            (via,) = movable
            (zone,) = via.zones
            assert zone.regions["In2.Cu"].geom_type == "MultiPolygon"
        else:
            assert fixed_reason(log) == "it sits at the edge of zone plane on In2.Cu"

    def test_each_layer_is_judged_against_its_own_zones(self, log: list[str]) -> None:
        # `small` reaches the ring on In1.Cu and In2.Cu, `big` holds the via
        # only on In2.Cu: on In1.Cu, `small` sits at its edge alone.
        big = sig_zone("big", (2, 2, 6, 6))
        small = sig_zone("small", (2.5, 2.5, 3.9, 5.5), [BL.BL_In1_Cu, BL.BL_In2_Cu])

        assert judged(big, small) == []
        assert fixed_reason(log) == "it sits at the edge of zone small on In1.Cu"

    def test_layers_with_the_same_zones_at_the_edge_are_named_once(
        self, log: list[str]
    ) -> None:
        both = [BL.BL_In1_Cu, BL.BL_In2_Cu]
        small = sig_zone("small", (2.5, 2.5, 3.9, 5.5), both)

        assert judged(small) == []
        assert fixed_reason(log) == (
            "it sits at the edge of zone small on In1.Cu and In2.Cu"
        )


def plane(name: str = "plane") -> kbt.Zone:
    """A zone of the stitching net around `VIA` on both outer layers."""
    outline = kipy_rect(2, 2, 6, 6)
    return kipy_copper_zone(name, "GND", outline, [BL.BL_F_Cu, BL.BL_B_Cu])


def nearby() -> kbt.Track:
    """A selected track of another net beside `VIA`."""
    return kipy_track("x", "SIG", (0, 3), (8, 3))


def stitching() -> kbt.Via:
    return kipy_via("s", "GND", *VIA)


class TestStitchingVias:
    """Vias that no selected track ends at, which only zones of their net touch."""

    def test_one_a_zone_of_its_net_touches_is_movable(self, log: list[str]) -> None:
        s = stitching()
        movable, _ = judge([nearby(), s], [nearby()], [s], zones=[plane()])

        (via,) = movable
        assert via.via.id == "s"
        assert via.ends == ()
        (inside,) = via.zones
        assert inside.description == "zone plane on F.Cu and B.Cu"
        assert warnings(log) == []

    def test_one_no_zone_touches_stays_fixed(self, log: list[str]) -> None:
        s = stitching()
        movable, _ = judge([nearby(), s], [nearby()], [s])

        assert movable == []
        (warning,) = warnings(log)
        assert warning.rstrip().endswith(
            "stays fixed: no selected track ends at it and no zone of its net "
            "reaches it"
        )

    def test_locked_one_stays_fixed(self, log: list[str]) -> None:
        s = kipy_via("s", "GND", *VIA, locked=True)

        movable, _ = judge([nearby(), s], [nearby()], [s], zones=[plane()])

        assert movable == []
        (warning,) = warnings(log)
        assert warning.rstrip().endswith("stays fixed: it is locked")

    def test_buried_one_moves_inside_a_zone_on_one_of_its_layers(
        self, log: list[str]
    ) -> None:
        s = kipy_via(
            "s",
            "GND",
            *VIA,
            via_type=ViaType.VT_BLIND_BURIED,
            span=(BL.BL_In1_Cu, BL.BL_In2_Cu),
        )
        zone = kipy_copper_zone("plane", "GND", kipy_rect(2, 2, 6, 6), [BL.BL_In2_Cu])

        movable, _ = judge([nearby(), s], [nearby()], [s], zones=[zone])

        (via,) = movable
        assert via.ends == ()
        (inside,) = via.zones
        assert inside.description == "zone plane on In2.Cu"
        assert warnings(log) == []

    def test_one_an_unselected_track_reaches_stays_fixed(self, log: list[str]) -> None:
        s = stitching()
        stub = kipy_track("t", "GND", VIA, (4, 7))

        movable, _ = judge([nearby(), s], [nearby(), stub], [s], zones=[plane()])

        assert movable == []
        (warning,) = warnings(log)
        assert warning.rstrip().endswith("stays fixed: track t on F.Cu connects to it")


SHARED_END = (4.45, 4.0)
"""Within reach of a via at `VIA` and of one at `BESIDE` for a 0.4 mm track."""
BESIDE = (4.9, 4.0)


def front_to(end: tuple[float, float], width: float = 0.25) -> list[kbt.Track]:
    return [
        kipy_track("a", "SIG", (0, 0), (end[0], 0), width=width),
        kipy_track("b", "SIG", (end[0], 0), end, width=width),
    ]


class TestCopperAtCarriedEnds:
    """Copper touching a chain end the via would carry, but not the via."""

    def test_pad_by_a_wide_end_pins_it(self, log: list[str]) -> None:
        # The end's copper reaches y = 4.5, the via's only 4.3; the pad's
        # edge is at 4.4.
        wide = front_to(VIA, width=1.0)
        v = kipy_via("v", "SIG", *VIA)
        pad = kipy_pad("p", "SIG", 4.0, 4.65, layers=(BL.BL_F_Cu,))

        movable, _ = judge([*wide, *back(), v], [*wide, *back()], [v], pads=[pad])

        assert movable == []
        (warning,) = warnings(log)
        assert "pad p" in warning

    def test_narrow_track_at_an_off_centre_end_pins_it(self, log: list[str]) -> None:
        end = (4.0, 3.64)
        stub = kipy_track("s", "SIG", end, (2.5, 3.64), width=0.1)
        v = kipy_via("v", "SIG", *VIA)

        movable, _ = judge(
            [*front_to(end), *back(), v], [*front_to(end), *back(), stub], [v]
        )

        assert movable == []
        (warning,) = warnings(log)
        assert "track s on F.Cu" in warning

    def test_unselected_via_sharing_an_end_pins_it(self, log: list[str]) -> None:
        front = front_to(SHARED_END, width=0.4)
        v, w = kipy_via("v", "SIG", *VIA), kipy_via("w", "SIG", *BESIDE)

        movable, _ = judge([*front, *back(), v], [*front, *back()], [v, w])

        assert movable == []
        (warning,) = warnings(log)
        assert "via w" in warning

    def test_two_selected_vias_sharing_an_end_pin_each_other(
        self, log: list[str]
    ) -> None:
        front = front_to(SHARED_END, width=0.4)
        beside = [
            kipy_track("g", "SIG", BESIDE, (9, 8), layer=BL.BL_In1_Cu),
            kipy_track("h", "SIG", (9, 8), (12, 8), layer=BL.BL_In1_Cu),
        ]
        v, w = kipy_via("v", "SIG", *VIA), kipy_via("w", "SIG", *BESIDE)
        route = [*front, *back(), *beside]

        movable, _ = judge([*route, v, w], route, [v, w])

        assert movable == []
        held = warnings(log)
        assert len(held) == 2
        assert "Via v " in held[0] and "via w" in held[0]
        assert "Via w " in held[1] and "via v" in held[1]


GROUND = frozenset({"GND", "AGND"})


def via_on(net: str, span: tuple[str, str] = ("F.Cu", "B.Cu")) -> Via:
    return Via(net, net, (0.0, 0.0), 0.6, 0.3, span)


def test_a_via_cuts_only_its_own_net_on_the_layers_it_spans() -> None:
    blind_via = replace(via_on("SIG", ("F.Cu", "In1.Cu")), position=CORNER)
    other_net = replace(via_on("GND"), position=VIA)

    cuts = chain_cuts([blind_via, other_net])

    assert cuts("F.Cu", "SIG") == [CORNER]
    assert cuts("In1.Cu", "SIG") == [CORNER]
    assert cuts("B.Cu", "SIG") == []
    assert cuts("F.Cu", "GND") == [VIA]
    assert cuts("F.Cu", "OTHER") == []


@pytest.mark.parametrize(
    ("net", "other", "expected"),
    [
        ("SIG", "GND", PairKind.RETURN),
        ("GND", "SIG", PairKind.RETURN),
        ("GND", "AGND", PairKind.GROUNDS),
        ("GND", "GND", PairKind.SPREAD),
        ("SIG", "SIG", None),
        ("SIG", "PWR", None),
    ],
)
def test_the_pair_kind_follows_the_nets_and_their_classes(
    net: str, other: str, expected: PairKind | None
) -> None:
    kind = pair_kind(via_on(net), net in GROUND, via_on(other), other in GROUND)

    assert kind is expected


def test_vias_whose_spans_do_not_overlap_do_not_pair() -> None:
    top = via_on("SIG", ("F.Cu", "In1.Cu"))
    bottom = via_on("GND", ("In2.Cu", "B.Cu"))

    assert pair_kind(top, False, bottom, True) is None


@pytest.mark.parametrize(
    ("net", "other", "expected"),
    [
        ("SIG", "GND", True),
        ("GND", "SIG", True),
        ("GND", "AGND", True),
        ("GND", "GND", False),
        ("SIG", "SIG", False),
        ("SIG", "PWR", False),
    ],
)
def test_pairs_are_ground_and_regular_or_two_ground_nets(
    net: str, other: str, expected: bool
) -> None:
    standing = StaticVia(via_on(other), other in GROUND, 0.2)

    assert pairs_with(via_on(net), net in GROUND, standing) is expected
