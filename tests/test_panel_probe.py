from typing import cast

import kipy
import kipy.board_types as kbt
import pytest
from conftest import (
    BOARD_FILE,
    TYPELESS,
    FakeBoard,
    FakeKiCad,
    kipy_arc,
    kipy_copper_zone,
    kipy_footprint,
    kipy_group,
    kipy_rect,
    kipy_track,
    kipy_via,
)
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.common.types.base_types_pb2 import DocumentSpecifier

from kibends.panel.probe import BoardId, Counts, Prober, Totals
from kibends.run import NoBoardError
from kibends.snapshot import GroundNets, GroundSource

TRACKS = 10
VIAS = 4


def ten_tracks_four_vias() -> FakeBoard:
    """TRACKS tracks, the last an arc, and VIAS vias, nothing selected."""
    tracks: list[kbt.Track | kbt.ArcTrack] = [
        kipy_track(f"t{i}", "SIG", (0, i), (5, i)) for i in range(TRACKS - 1)
    ]
    tracks.append(kipy_arc("arc", "SIG", (0, 20), (2.5, 21), (5, 20)))
    vias = [kipy_via(f"v{i}", "GND", 10, i) for i in range(VIAS)]
    return FakeBoard(tracks=tracks, vias=vias)


def prober(board: FakeBoard) -> Prober:
    return Prober(lambda: cast(kipy.KiCad, FakeKiCad(board)))


class TestSelection:
    def test_counts_tracks_arcs_and_group_members_against_the_board(self) -> None:
        board = ten_tracks_four_vias()
        grouped = cast(kbt.Track, board.tracks[2])
        group = kipy_group("g", [grouped])
        board.groups = [group]
        board.selection = [board.tracks[0], board.tracks[1], board.tracks[-1], group]

        probe = prober(board)(None)

        assert probe.selected == Counts(4, 0)
        assert probe.totals is not None
        assert (probe.totals.tracks, probe.totals.vias) == (TRACKS, VIAS)

    def test_nothing_selected_is_none(self) -> None:
        assert prober(ten_tracks_four_vias())(None).selected is None

    def test_a_selection_without_copper_counts_none(self) -> None:
        board = ten_tracks_four_vias()
        board.selection = [kipy_footprint("U1")]

        assert prober(board)(None).selected == Counts(0, 0)

    def test_items_kicad_sends_without_a_type_count_none(self) -> None:
        board = ten_tracks_four_vias()
        board.unreadable = [TYPELESS]

        assert prober(board)(None).selected == Counts(0, 0)


class TestBoard:
    def test_names_the_board(self) -> None:
        board = ten_tracks_four_vias()
        board.document.project.path = "/projects/demo"

        assert prober(board)(None).board == BoardId(BOARD_FILE, "/projects/demo")

    def test_a_known_board_is_not_read_again(self) -> None:
        board = ten_tracks_four_vias()
        probe = prober(board)
        known = probe(None).board
        board.reads.clear()

        again = probe(known)

        assert again.totals is None
        assert again.board == known
        assert board.reads["get_tracks"] == 0
        assert board.reads["get_zones"] == 0

    def test_another_board_is_read(self) -> None:
        board = ten_tracks_four_vias()
        probe = prober(board)
        known = probe(None).board
        board.document = DocumentSpecifier(board_filename="other.kicad_pcb")

        assert probe(known).totals is not None

    def test_detects_the_ground_nets(self) -> None:
        board = ten_tracks_four_vias()
        board.zones = [
            kipy_copper_zone(
                "plane", "GND", kipy_rect(0, 0, 10, 10), [BL.BL_In1_Cu, BL.BL_In2_Cu]
            )
        ]

        totals = prober(board)(None).totals

        assert totals == Totals(
            TRACKS, VIAS, GroundNets(frozenset({"GND"}), GroundSource.INNER_ZONES)
        )


class TestFailures:
    def test_an_unreachable_kicad_is_reconnected_next_time(self) -> None:
        board = ten_tracks_four_vias()
        attempts: list[int] = []

        def connect() -> kipy.KiCad:
            attempts.append(1)
            if len(attempts) == 1:
                raise KiCadConnectionError("no socket")
            return cast(kipy.KiCad, FakeKiCad(board))

        probe = Prober(connect)
        with pytest.raises(KiCadConnectionError):
            probe(None)

        assert probe(None).board.name == BOARD_FILE
        assert len(attempts) == 2

    def test_a_lost_connection_is_dropped(self) -> None:
        connections: list[FakeKiCad] = []

        class Lost(FakeKiCad):
            def get_board(self) -> FakeBoard:
                raise KiCadConnectionError("gone")

        def connect() -> kipy.KiCad:
            connections.append(Lost(FakeBoard()))
            return cast(kipy.KiCad, connections[-1])

        probe = Prober(connect)
        for _ in range(2):
            with pytest.raises(KiCadConnectionError):
                probe(None)

        assert len(connections) == 2

    def test_no_board_open(self) -> None:
        class Empty(FakeKiCad):
            def get_board(self) -> FakeBoard:
                raise ApiError("Expected to be able to retrieve at least one board")

        with pytest.raises(NoBoardError):
            Prober(lambda: cast(kipy.KiCad, Empty(FakeBoard())))(None)
