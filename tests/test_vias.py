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
from kipy.proto.board.board_types_pb2 import ViaType
from kipy.wrapper import Wrapper

from kibends.chain import Chain, extract_chains
from kibends.snapshot import BoardSnapshot, read_selection
from kibends.vias import ChainEnd, MovableVia, movable_vias

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
    whole = BoardSnapshot.read(fake, layers, chosen.vias)
    snapshots = {layer: whole.slice(layer, replaced) for layer in layers}
    chains = extract_chains(
        chosen.segments, lambda layer, net: snapshots[layer].same_net_copper(net)
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
        assert "copper that does not move with it touches it: track e on B.Cu" in (
            warning
        )

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
