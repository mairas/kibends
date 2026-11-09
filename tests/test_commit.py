from typing import cast

import kipy.board
import kipy.board_types as kbt
import numpy as np
import pytest
from conftest import COMMIT_CALLS, CommitBoard, kipy_track, kipy_via, nm
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.common.envelope_pb2 import ApiStatusCode

from kibends import commit
from kibends.chain import Chain, Segment, extract_chain
from kibends.coords import BOARD_FRAME
from kibends.energy import FloatArray
from kibends.snapshot import segment_from_track, via_from_kipy

WIDTH = 0.3
NEW_PATH = np.array([[0.0, 0.0], [2.0, 1.0], [4.0000004, 3.0], [5.0, 5.0]])


def l_tracks() -> list[kbt.Track]:
    return [
        kipy_track("a", "SIG", (0, 0), (5, 0), width=WIDTH),
        kipy_track("b", "SIG", (5, 0), (5, 5), width=WIDTH),
    ]


def setup(
    *,
    refuse_create: bool = False,
    short_create: bool = False,
    refuse_delete: frozenset[str] = frozenset(),
    unanswered_delete: frozenset[str] = frozenset(),
    push_error: BaseException | None = None,
) -> tuple[CommitBoard, Chain, list[Segment]]:
    tracks = l_tracks()
    board = CommitBoard(
        selection=list(tracks),
        tracks=list(tracks),
        refuse_create=refuse_create,
        short_create=short_create,
        refuse_delete=refuse_delete,
        unanswered_delete=unanswered_delete,
        push_error=push_error,
    )
    selected = [segment_from_track(t) for t in tracks]
    return board, extract_chain(selected, frame=BOARD_FRAME), selected


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

        assert board.calls == COMMIT_CALLS
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

        run_commit(board, extract_chain(selected, frame=BOARD_FRAME), selected)

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

    def test_short_create_reply_drops_the_commit(self) -> None:
        board, chain, selected = setup(short_create=True)

        with pytest.raises(
            commit.CreateItemsError, match=r"^KiCad created 2 of 3 items$"
        ):
            run_commit(board, chain, selected)

        assert board.calls[-1] == "drop_commit"
        assert "push_commit" not in board.calls

    def test_refused_delete_drops_the_commit(self) -> None:
        board, chain, selected = setup(refuse_delete=frozenset({"b"}))

        with pytest.raises(
            commit.DeleteItemsError, match=r"items: b \(IDS_IMMUTABLE\)$"
        ):
            run_commit(board, chain, selected)

        assert board.calls == ["begin_commit", "delete", "drop_commit"]

    def test_unanswered_delete_drops_the_commit(self) -> None:
        board, chain, selected = setup(unanswered_delete=frozenset({"b"}))

        with pytest.raises(commit.DeleteItemsError, match=r"items: b \(no answer\)$"):
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


SECOND_PATH = np.array([[0.0, 10.0], [3.0, 12.0], [5.0, 15.0]])


def two_chains(
    *, refuse_create_nets: frozenset[str] = frozenset()
) -> tuple[CommitBoard, list[tuple[Chain, FloatArray]], list[Segment]]:
    tracks = [
        *l_tracks(),
        kipy_track("c", "PWR", (0, 10), (5, 10), width=WIDTH),
        kipy_track("d", "PWR", (5, 10), (5, 15), width=WIDTH),
    ]
    board = CommitBoard(
        selection=list(tracks),
        tracks=list(tracks),
        refuse_create_nets=refuse_create_nets,
    )
    selected = [segment_from_track(t) for t in tracks]
    chains = [
        (extract_chain(selected[:2], frame=BOARD_FRAME), NEW_PATH),
        (extract_chain(selected[2:], frame=BOARD_FRAME), SECOND_PATH),
    ]
    return board, chains, selected


def run_commit_chains(
    board: CommitBoard,
    chains: list[tuple[Chain, FloatArray]],
    selected: list[Segment],
) -> None:
    commit.commit_chains(cast(kipy.board.Board, board), chains, selected)


class TestCommitChains:
    def test_replaces_every_chain_in_one_request_per_step(self) -> None:
        board, chains, selected = two_chains()

        run_commit_chains(board, chains, selected)

        assert board.reads["get_items_by_id"] == 1
        assert board.calls == COMMIT_CALLS
        assert len(board.labels) == 1
        assert sorted(board.deleted) == ["a", "b", "c", "d"]
        created = cast(list[kbt.Track], board.created)
        assert [t.net.name for t in created] == ["SIG"] * 3 + ["PWR"] * 2
        assert [(t.end.x, t.end.y) for t in created[3:]] == [
            (nm(x), nm(y)) for x, y in SECOND_PATH[1:]
        ]

    def test_one_changed_chain_refuses_the_whole_commit(self) -> None:
        board, chains, selected = two_chains()
        board.tracks[3] = kipy_track("d", "PWR", (5, 10), (5, 16), width=WIDTH)

        with pytest.raises(commit.BoardChangedError, match="d"):
            run_commit_chains(board, chains, selected)

        assert board.calls == []
        assert board.deleted == []

    def test_refused_create_on_the_second_chain_drops_everything(self) -> None:
        board, chains, selected = two_chains(refuse_create_nets=frozenset({"PWR"}))

        with pytest.raises(commit.CreateItemsError, match="created 3 of 5"):
            run_commit_chains(board, chains, selected)

        assert board.calls == ["begin_commit", "delete", "create", "drop_commit"]

    def test_no_chains_and_no_vias_is_an_error(self) -> None:
        board, _, selected = two_chains()

        with pytest.raises(ValueError, match="Nothing to commit"):
            run_commit_chains(board, [], selected)

        assert board.calls == []


class TestSnappedPath:
    def test_drops_vertices_that_do_not_bend_the_path(self) -> None:
        polyline = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]])

        assert commit.snapped_path(polyline) == [(0, 0), (nm(2), 0), (nm(2), nm(1))]

    def test_drops_a_vertex_one_nanometre_off_the_line(self) -> None:
        polyline = np.array([[0.0, 0.0], [1.0, 1e-6], [2.0, 0.0]])

        assert commit.snapped_path(polyline) == [(0, 0), (nm(2), 0)]

    def test_keeps_a_vertex_two_nanometres_off_the_line(self) -> None:
        polyline = np.array([[0.0, 0.0], [1.0, 2e-6], [2.0, 0.0]])

        assert len(commit.snapped_path(polyline)) == 3

    def test_keeps_a_reversal(self) -> None:
        polyline = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]])

        assert len(commit.snapped_path(polyline)) == 3


VIA = (5.0, 5.0)
MOVED_TO = (4.0, 4.5)


def via_route(
    *, refuse_create_vias: bool = False
) -> tuple[
    CommitBoard, list[tuple[Chain, FloatArray]], list[Segment], list[commit.ViaMove]
]:
    """SIG from (0, 0) on F.Cu down the via `v` at VIA and on to (10, 0) on
    B.Cu, with the via moved to MOVED_TO and both chains straight to it."""
    tracks = [
        kipy_track("a", "SIG", (0, 0), VIA, width=WIDTH),
        kipy_track("e", "SIG", VIA, (10, 0), width=WIDTH, layer=BL.BL_B_Cu),
    ]
    via = kipy_via("v", "SIG", *VIA, diameter=0.8, drill=0.4)
    board = CommitBoard(
        selection=[*tracks, via],
        tracks=list(tracks),
        vias=[via],
        refuse_create_vias=refuse_create_vias,
    )
    selected = [segment_from_track(t) for t in tracks]
    chains = [
        (
            extract_chain(selected[:1], frame=BOARD_FRAME),
            np.array([[0.0, 0.0], MOVED_TO]),
        ),
        (
            extract_chain(selected[1:], frame=BOARD_FRAME),
            np.array([MOVED_TO, [10.0, 0.0]]),
        ),
    ]
    return board, chains, selected, [commit.ViaMove(via_from_kipy(via), MOVED_TO)]


def run_commit_vias(
    board: CommitBoard,
    chains: list[tuple[Chain, FloatArray]],
    selected: list[Segment],
    vias: list[commit.ViaMove],
) -> None:
    commit.commit_chains(cast(kipy.board.Board, board), chains, selected, vias)


class TestMovedVias:
    def test_moves_the_via_with_its_chains_in_one_commit(self) -> None:
        board, chains, selected, vias = via_route()
        old = board.vias[0]

        run_commit_vias(board, chains, selected, vias)

        assert board.reads["get_items_by_id"] == 1
        assert board.calls == COMMIT_CALLS
        assert sorted(board.deleted) == ["a", "e", "v"]
        (via,) = [i for i in board.created if isinstance(i, kbt.Via)]
        assert (via.position.x, via.position.y) == (nm(4.0), nm(4.5))
        assert via.id.value in board.assigned_ids
        assert (via.net.name, via.diameter, via.drill_diameter, via.type) == (
            old.net.name,
            old.diameter,
            old.drill_diameter,
            old.type,
        )
        assert via.padstack.drill.start_layer == old.padstack.drill.start_layer
        assert via.padstack.drill.end_layer == old.padstack.drill.end_layer
        tracks = [i for i in board.created if isinstance(i, kbt.Track)]
        moved = (nm(4.0), nm(4.5))
        assert [(t.layer, (t.end.x, t.end.y)) for t in tracks[:1]] == [
            (BL.BL_F_Cu, moved)
        ]
        assert [(t.layer, (t.start.x, t.start.y)) for t in tracks[1:]] == [
            (BL.BL_B_Cu, moved)
        ]

    def test_moves_a_via_without_chains(self) -> None:
        board, _, selected, vias = via_route()

        run_commit_vias(board, [], selected, vias)

        assert board.calls == COMMIT_CALLS
        assert board.deleted == ["v"]
        (via,) = board.created
        assert isinstance(via, kbt.Via)
        assert (via.position.x, via.position.y) == (nm(4.0), nm(4.5))

    @pytest.mark.parametrize(
        ("with_chains", "summary"),
        [(True, "replaced 2 tracks with 2 and moved 1 via"), (False, "moved 1 via")],
    )
    def test_logs_only_what_it_committed(
        self, log: list[str], with_chains: bool, summary: str
    ) -> None:
        board, chains, selected, vias = via_route()

        run_commit_vias(board, chains if with_chains else [], selected, vias)

        (line,) = [line for line in log if "Committed" in line]
        assert line.rstrip().endswith(f"': {summary}")

    def test_refused_via_create_drops_the_whole_commit(self) -> None:
        board, chains, selected, vias = via_route(refuse_create_vias=True)

        with pytest.raises(commit.CreateItemsError, match="created 2 of 3 items"):
            run_commit_vias(board, chains, selected, vias)

        assert board.calls == ["begin_commit", "delete", "create", "drop_commit"]

    @pytest.mark.parametrize(
        "now",
        [
            kipy_via("v", "SIG", 5.0, 5.1, diameter=0.8, drill=0.4),
            kipy_via("v", "SIG", *VIA, diameter=0.8, drill=0.3),
            kipy_via("v", "SIG", *VIA, diameter=0.6, drill=0.4),
            kipy_via("v", "GND", *VIA, diameter=0.8, drill=0.4),
            kipy_via("v", "SIG", *VIA, diameter=0.8, drill=0.4, locked=True),
            kipy_via(
                "v",
                "SIG",
                *VIA,
                diameter=0.8,
                drill=0.4,
                span=(BL.BL_F_Cu, BL.BL_In1_Cu),
            ),
            None,
        ],
        ids=["moved", "drill", "diameter", "net", "locked", "span", "deleted"],
    )
    def test_via_changed_since_reading_refuses_the_commit(
        self, now: kbt.Via | None
    ) -> None:
        board, chains, selected, vias = via_route()
        board.vias = [] if now is None else [now]

        with pytest.raises(commit.BoardChangedError, match="v"):
            run_commit_vias(board, chains, selected, vias)

        assert board.calls == []

    def test_the_moved_via_leaves_its_group(self) -> None:
        board, chains, selected, vias = via_route()
        board.vias[0].proto.parent.value = "group"

        run_commit_vias(board, chains, selected, vias)

        (via,) = [i for i in board.created if isinstance(i, kbt.Via)]
        assert not via.proto.HasField("parent")


class TestSelection:
    def test_selects_the_new_tracks_and_keeps_the_unchanged_ones(self) -> None:
        tracks = [*l_tracks(), kipy_track("c", "SIG", (5, 5), (8, 5), locked=True)]
        tracks[2].width = nm(WIDTH)
        board = CommitBoard(selection=list(tracks), tracks=list(tracks))
        selected = [segment_from_track(t) for t in tracks]

        run_commit(board, extract_chain(selected, frame=BOARD_FRAME), selected)

        assert board.selected_ids() == ["c", *board.assigned_ids]
        assert len(board.assigned_ids) == len(NEW_PATH) - 1

    def test_selects_the_moved_via(self) -> None:
        board, chains, selected, vias = via_route()

        run_commit_vias(board, chains, selected, vias)

        assert board.selected_ids() == board.assigned_ids
        (via,) = [i for i in board.selection if isinstance(i, kbt.Via)]
        assert (via.position.x, via.position.y) == (nm(4.0), nm(4.5))
        assert via.id.value in board.assigned_ids

    @pytest.mark.parametrize(
        "error",
        [
            ApiError("busy", code=ApiStatusCode.AS_BUSY),
            KiCadConnectionError("timed out"),
            KeyboardInterrupt(),
        ],
    )
    def test_a_failure_to_select_after_the_push_is_logged_at_debug(
        self, log: list[str], error: BaseException
    ) -> None:
        board, chain, selected = setup()
        board.select_error = error

        run_commit(board, chain, selected)

        assert board.calls[-2:] == ["push_commit", "select"]
        (line,) = [line for line in log if "Could not select" in line]
        assert line.startswith("DEBUG")
        assert any("Committed" in line for line in log)

    @pytest.mark.parametrize(
        ("error", "raised"),
        [
            (ApiError("refused"), ApiError),
            (KiCadConnectionError("timed out"), commit.CommitUncertainError),
            (KeyboardInterrupt(), commit.CommitUncertainError),
        ],
    )
    def test_a_push_that_fails_or_may_not_have_landed_selects_nothing(
        self, error: BaseException, raised: type[Exception]
    ) -> None:
        board, chain, selected = setup(push_error=error)
        before = list(board.selection)

        with pytest.raises(raised):
            run_commit(board, chain, selected)

        assert "select" not in board.calls
        assert board.selection == before

    def test_a_dropped_commit_leaves_the_selection_alone(self) -> None:
        board, chain, selected = setup(refuse_create=True)
        before = list(board.selection)

        with pytest.raises(commit.CreateItemsError):
            run_commit(board, chain, selected)

        assert board.selection == before
