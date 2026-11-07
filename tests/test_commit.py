from typing import cast

import kipy.board
import kipy.board_types as kbt
import numpy as np
import pytest
from conftest import CommitBoard, kipy_track, nm
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.common.envelope_pb2 import ApiStatusCode

from kibends import commit
from kibends.chain import Chain, Segment, extract_chain
from kibends.energy import FloatArray
from kibends.snapshot import segment_from_track

WIDTH = 0.3
NEW_PATH = np.array([[0.0, 0.0], [2.0, 1.0], [4.0000004, 3.0], [5.0, 5.0]])


def l_tracks() -> list[kbt.Track]:
    return [
        kipy_track("a", "SIG", (0, 0), (5, 0), width=WIDTH),
        kipy_track("b", "SIG", (5, 0), (5, 5), width=WIDTH),
    ]


def setup(
    **board_options: object,
) -> tuple[CommitBoard, Chain, list[Segment]]:
    tracks = l_tracks()
    board = CommitBoard(selection=list(tracks), tracks=list(tracks), **board_options)  # type: ignore[arg-type]
    selected = [segment_from_track(t) for t in tracks]
    return board, extract_chain(selected), selected


def run_commit(
    board: CommitBoard,
    chain: Chain,
    selected: list[Segment],
    polyline: FloatArray = NEW_PATH,
) -> None:
    commit.commit_chain(
        cast(kipy.board.Board, board),
        chain,
        selected,
        polyline,
    )


class TestCommitChain:
    def test_replaces_the_chain_in_one_commit(self) -> None:
        board, chain, selected = setup()

        run_commit(board, chain, selected)

        assert board.calls == [
            "begin_commit",
            "delete",
            "create",
            "push_commit",
        ]
        assert sorted(board.deleted) == ["a", "b"]
        assert board.labels[0].startswith("kibends smooth ")

    def test_creates_straight_tracks_along_the_polyline(self) -> None:
        board, chain, selected = setup()

        run_commit(board, chain, selected)

        tracks = board.created
        assert all(type(t) is kbt.Track for t in tracks)
        created = cast(list[kbt.Track], tracks)
        nm_path = [(nm(x), nm(y)) for x, y in NEW_PATH]
        assert [(t.start.x, t.start.y) for t in created] == nm_path[:-1]
        assert [(t.end.x, t.end.y) for t in created] == nm_path[1:]
        assert {t.width for t in created} == {nm(WIDTH)}
        assert {t.layer for t in created} == {BL.BL_F_Cu}
        assert {t.net.name for t in created} == {"SIG"}

    def test_vertices_on_the_same_nanometre_make_no_track(self) -> None:
        board, chain, selected = setup()
        doubled = np.array([[0.0, 0.0], [2.0, 1.0], [2.0000000001, 1.0], [5.0, 5.0]])

        run_commit(board, chain, selected, doubled)

        assert len(board.created) == 2

    def test_locked_segments_are_left_alone(self) -> None:
        tracks = [*l_tracks(), kipy_track("c", "SIG", (5, 5), (8, 5), locked=True)]
        tracks[2].width = nm(WIDTH)
        board = CommitBoard(selection=list(tracks), tracks=list(tracks))
        selected = [segment_from_track(t) for t in tracks]

        run_commit(board, extract_chain(selected), selected)

        assert sorted(board.deleted) == ["a", "b"]


class TestBoardChanged:
    def test_vanished_track_aborts_before_any_change(self) -> None:
        board, chain, selected = setup()
        board.tracks = board.tracks[:1]

        with pytest.raises(commit.BoardChangedError, match="b"):
            run_commit(board, chain, selected)

        assert board.calls == []

    def test_moved_track_aborts_before_any_change(self) -> None:
        board, chain, selected = setup()
        board.tracks[1] = kipy_track("b", "SIG", (5, 0), (5, 6), width=WIDTH)

        with pytest.raises(commit.BoardChangedError, match="b"):
            run_commit(board, chain, selected)

        assert board.calls == []


class TestRefusals:
    def test_refused_create_drops_the_commit(self) -> None:
        board, chain, selected = setup(refuse_create=True)

        with pytest.raises(commit.CreateItemsError):
            run_commit(board, chain, selected)

        assert board.calls[-1] == "drop_commit"
        assert "push_commit" not in board.calls

    def test_refused_delete_drops_the_commit(self) -> None:
        board, chain, selected = setup(refuse_delete=frozenset({"b"}))

        with pytest.raises(commit.DeleteItemsError, match="b"):
            run_commit(board, chain, selected)

        assert board.calls == ["begin_commit", "delete", "drop_commit"]

    def test_push_refused_by_kicad_drops_the_commit(self) -> None:
        board, chain, selected = setup(
            push_error=ApiError("busy", code=ApiStatusCode.AS_BUSY)
        )

        with pytest.raises(ApiError):
            run_commit(board, chain, selected)

        assert board.calls[-1] == "drop_commit"

    def test_push_timeout_keeps_the_commit_and_raises(self) -> None:
        # The commit may have landed: dropping it then would be wrong.
        board, chain, selected = setup(push_error=KiCadConnectionError("timed out"))

        with pytest.raises(commit.CommitUncertainError, match="may have landed"):
            run_commit(board, chain, selected)

        assert "drop_commit" not in board.calls
