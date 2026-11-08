from collections.abc import Sequence
from typing import cast

import kipy.board
import kipy.board_types as kbt
import pytest
from conftest import (
    FakeBoard,
    kipy_filled_zone,
    kipy_pad,
    kipy_rect,
    kipy_track,
    kipy_via,
)
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.board.board_types_pb2 import ViaType, ZoneType
from kipy.wrapper import Wrapper

from kibends.chain import Chain, extract_chains
from kibends.coords import BOARD_FRAME
from kibends.snapshot import BoardSnapshot, read_selection
from kibends.vias import ChainEnd, MovableVia, carried_copper, movable_vias

VIA = (4.0, 4.0)


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


def judge(
    selection: list[Wrapper],
    tracks: list[kbt.Track],
    vias: list[kbt.Via],
    *,
    pads: Sequence[kbt.Pad] = (),
    zones: Sequence[kbt.Zone] = (),
) -> tuple[list[MovableVia], list[Chain]]:
    """The selection's chains and movable vias, read as a run reads them."""
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
    bare = extract_chains(chosen.segments, lambda layer, net: (), frame=BOARD_FRAME)
    whole = BoardSnapshot.read(
        fake, layers, chosen.vias, carried=carried_copper(chosen.vias, bare)
    )
    snapshots = {layer: whole.slice(layer, replaced) for layer in layers}
    chains = extract_chains(
        chosen.segments,
        lambda layer, net: snapshots[layer].same_net_copper(net),
        frame=BOARD_FRAME,
    )
    return movable_vias(chosen.vias, whole.via_contacts, chains), chains


def warnings(log: list[str]) -> list[str]:
    return [line for line in log if line.startswith("WARNING")]


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

        movable, _ = judge([*short, *back(), v], [*short, *back()], [v])

        assert [len(m.ends) for m in movable] == ([2] if joins else [])
        assert len(warnings(log)) == int(not joins)

    def test_locked_via_stays_fixed(self, log: list[str]) -> None:
        assert "it is locked" in fixed(log, kipy_via("v", "SIG", *VIA, locked=True))

    def test_blind_via_stays_fixed(self, log: list[str]) -> None:
        blind = kipy_via(
            "v", "SIG", *VIA, via_type=ViaType.VT_BLIND, span=(BL.BL_F_Cu, BL.BL_B_Cu)
        )

        warning = fixed(log, blind)

        assert "blind via" in warning and "only through vias move" in warning

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
        assert "chain end it carries: track e on B.Cu" in (warning)

    def test_same_net_pad_under_it_pins_it(self, log: list[str]) -> None:
        v = kipy_via("v", "SIG", *VIA)
        pad = kipy_pad("p", "SIG", *VIA, layers=(BL.BL_B_Cu,))

        movable, _ = judge([*front(), *back(), v], [*front(), *back()], [v], pads=[pad])

        assert movable == []
        (warning,) = warnings(log)
        assert "pad p" in warning

    def test_same_net_zone_on_an_inner_layer_pins_it(self, log: list[str]) -> None:
        v = kipy_via("v", "SIG", *VIA)
        fills = {BL.BL_In2_Cu: kipy_rect(3, 3, 5, 5)}
        zone = kipy_filled_zone("plane", "SIG", fills)

        movable, _ = judge(
            [*front(), *back(), v], [*front(), *back()], [v], zones=[zone]
        )

        assert movable == []
        (warning,) = warnings(log)
        assert "zone plane on In2.Cu" in warning

    def test_teardrop_leaves_it_movable(self, log: list[str]) -> None:
        v = kipy_via("v", "SIG", *VIA)
        drop = {BL.BL_F_Cu: kipy_rect(3.9, 3.0, 4.1, 3.8)}
        teardrop = kipy_filled_zone("drop", "SIG", drop, ZoneType.ZT_TEARDROP)

        movable, _ = judge(
            [*front(), *back(), v], [*front(), *back()], [v], zones=[teardrop]
        )

        assert [m.via.id for m in movable] == ["v"]
        assert warnings(log) == []

    def test_other_net_and_distant_fills_leave_it_movable(self, log: list[str]) -> None:
        v = kipy_via("v", "SIG", *VIA)
        zones = [
            kipy_filled_zone("gnd", "GND", {BL.BL_In1_Cu: kipy_rect(3, 3, 5, 5)}),
            kipy_filled_zone("far", "SIG", {BL.BL_In2_Cu: kipy_rect(10, 10, 12, 12)}),
        ]

        movable, _ = judge(
            [*front(), *back(), v], [*front(), *back()], [v], zones=zones
        )

        assert [m.via.id for m in movable] == ["v"]
        assert warnings(log) == []

    def test_via_whose_tracks_are_not_selected_stays_fixed(
        self, log: list[str]
    ) -> None:
        v = kipy_via("v", "SIG", *VIA)
        elsewhere = kipy_track("x", "NET2", (0, 20), (5, 20))

        movable, _ = judge([elsewhere, v], [*front(), *back(), elsewhere], [v])

        assert movable == []
        (warning,) = warnings(log)
        assert "track b on F.Cu" in warning and "track e on B.Cu" in warning

    def test_via_ending_one_selected_chain_stays_fixed(self, log: list[str]) -> None:
        v = kipy_via("v", "SIG", *VIA)

        movable, _ = judge([*front(), v], front(), [v])

        assert movable == []
        (warning,) = warnings(log)
        assert "only 1 selected chain end" in warning

    def test_via_a_selected_chain_passes_stays_fixed(self, log: list[str]) -> None:
        v = kipy_via("v", "SIG", 2, 0)

        movable, _ = judge([*front(), v], front(), [v])

        assert movable == []
        (warning,) = warnings(log)
        assert "track a on F.Cu" in warning

    @pytest.mark.parametrize("selected", [False, True])
    def test_unselected_via_pins_as_before(
        self, log: list[str], selected: bool
    ) -> None:
        v = kipy_via("v", "SIG", 2, 0)

        movable, chains = judge([*front(), *([v] if selected else [])], front(), [v])

        assert movable == []
        (chain,) = chains
        pinned = [p for p, pin in zip(chain.points, chain.pinned, strict=True) if pin]
        assert pinned == [(0, 0), pytest.approx((1.7, 0)), pytest.approx((2.3, 0)), VIA]
        assert len(warnings(log)) == int(selected)


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
