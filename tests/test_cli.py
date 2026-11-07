import sys
from pathlib import Path
from typing import Any, cast

import kipy.board_types as kbt
import pytest
from conftest import (
    PAD_HALF_SIZE,
    CommitBoard,
    FakeKiCad,
    kipy_arc,
    kipy_pad,
    kipy_track,
    nm,
)
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from shapely.geometry import LineString, box

from kibends import cli
from kibends.energy import EnergyParams
from kibends.obstacles import Obstacles
from kibends.optimize import OptimizeResult, Termination

WIDTH = 0.25
CLEARANCE = 0.2
PAD = (3.2, 0.8)
"""Other-net pad inside the L's inner corner, between the L and its chord."""
# Rounding the created tracks to whole nanometres may cost this much clearance.
NM_ROUNDING = 2e-6


def l_chain() -> list[kbt.Track]:
    return [
        kipy_track("a", "SIG", (0, 0), (4, 0), width=WIDTH),
        kipy_track("b", "SIG", (4, 0), (4, 4), width=WIDTH),
    ]


def l_board(**options: object) -> CommitBoard:
    chain = l_chain()
    return CommitBoard(
        selection=list(chain),
        tracks=list(chain),
        pads=[kipy_pad("pad", "GND", *PAD)],
        **options,  # type: ignore[arg-type]
    )


def straight_board() -> CommitBoard:
    """A straight chain in free space: nothing to optimise."""
    track = kipy_track("a", "SIG", (0, 0), (10, 0), width=WIDTH)
    return CommitBoard(selection=[track], tracks=[track])


@pytest.fixture(autouse=True)
def keep_test_log_sinks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "configure_logging", lambda: None)


@pytest.fixture
def connect(monkeypatch: pytest.MonkeyPatch) -> "Connector":
    return Connector(monkeypatch)


class Connector:
    """Points kipy.KiCad() at a fake board, or makes it fail."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch

    def to(self, board: CommitBoard) -> CommitBoard:
        self.monkeypatch.setattr(cli.kipy, "KiCad", lambda: FakeKiCad(board))
        return board

    def failing(self, error: Exception) -> None:
        class Failing:
            def get_board(self) -> None:
                raise error

        self.monkeypatch.setattr(cli.kipy, "KiCad", Failing)


def run(*args: str) -> None:
    cli.main(["smooth", *args])


def exit_message(log: list[str], *args: str) -> str:
    """The one error a failing run logs."""
    with pytest.raises(SystemExit) as raised:
        run(*args)
    assert raised.value.code == 1
    errors = [line for line in log if line.startswith("ERROR")]
    assert len(errors) == 1
    return errors[0]


class TestSmooth:
    def test_commits_tracks_that_clear_the_pad(self, connect: Connector) -> None:
        board = connect.to(l_board())

        run()

        assert sorted(board.deleted) == ["a", "b"]
        assert "push_commit" in board.calls
        created = cast(list[kbt.Track], board.created)
        assert all(type(t) is kbt.Track for t in created)
        assert {t.width for t in created} == {nm(WIDTH)}
        assert {t.net.name for t in created} == {"SIG"}
        x, y = PAD
        h = PAD_HALF_SIZE
        pad = box(x - h, y - h, x + h, y + h)
        required = CLEARANCE + WIDTH / 2
        for t in created:
            segment = LineString(
                [(t.start.x / 1e6, t.start.y / 1e6), (t.end.x / 1e6, t.end.y / 1e6)]
            )
            assert segment.distance(pad) >= required - NM_ROUNDING
        assert len(created) > 2

    def test_prints_statistics(
        self, connect: Connector, capsys: pytest.CaptureFixture[str]
    ) -> None:
        connect.to(l_board())

        run()

        out = capsys.readouterr().out
        for label in (
            "Length",
            "Tracks",
            "Tightest bend",
            "Min clearance margin",
            "Mean clearance margin",
            "Frozen vertices",
            "converged",
        ):
            assert label in out

    def test_dry_run_prints_statistics_and_commits_nothing(
        self, connect: Connector, capsys: pytest.CaptureFixture[str]
    ) -> None:
        board = connect.to(l_board())

        run("--dry-run")

        assert board.calls == []
        assert "Length" in capsys.readouterr().out

    def test_nothing_to_do_begins_no_commit(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = connect.to(straight_board())

        run()

        assert board.calls == []
        assert any("Nothing to do" in line for line in log)

    def test_options_reach_the_optimizer(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connect.to(straight_board())
        seen: dict[str, object] = {}
        real_build = cli.build_obstacles

        def build_spy(*args: Any, **kwargs: Any) -> Obstacles:
            seen.update(kwargs)
            return real_build(*args, **kwargs)

        def optimize_spy(*args: Any, **kwargs: Any) -> OptimizeResult:
            seen["params"] = args[3]
            raise SystemExit(0)

        monkeypatch.setattr(cli, "build_obstacles", build_spy)
        monkeypatch.setattr(cli, "optimize", optimize_spy)

        with pytest.raises(SystemExit):
            run(
                "--stiffness=0.7",
                "--repulsion=2",
                "--range=0.4",
                "--edge-clearance=0.5",
                "--hole-clearance=0.1",
            )

        params = cast(EnergyParams, seen["params"])
        assert (params.stiffness, params.repulsion, params.falloff_range) == (
            0.7,
            2.0,
            0.4,
        )
        assert (seen["edge_clearance"], seen["hole_clearance"]) == (0.5, 0.1)

    @pytest.mark.parametrize(
        ("args", "expected"), [((), 0.025), (("--chord-tolerance=0.01",), 0.01)]
    )
    def test_chord_tolerance_reaches_simplify(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        args: tuple[str, ...],
        expected: float,
    ) -> None:
        connect.to(l_board())
        seen: dict[str, object] = {}
        real_simplify = cli.simplify

        def simplify_spy(*a: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return real_simplify(*a, **kwargs)

        monkeypatch.setattr(cli, "simplify", simplify_spy)

        run("--dry-run", *args)

        assert seen["tolerance"] == expected

    def test_negative_weight_is_rejected(self) -> None:
        with pytest.raises(SystemExit) as raised:
            run("--stiffness=-1")

        assert raised.value.code == 2

    def test_abnormal_termination_is_refused(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        log: list[str],
    ) -> None:
        board = connect.to(l_board())
        real = cli.optimize

        def abnormal(*args: object, **kwargs: object) -> OptimizeResult:
            result = real(*args, **kwargs)  # type: ignore[arg-type]
            return OptimizeResult(
                **{**result.__dict__, "termination": Termination.ABNORMAL}
            )

        monkeypatch.setattr(cli, "optimize", abnormal)

        assert "abnormally" in exit_message(log)
        assert board.calls == []


class TestPlot:
    def test_writes_the_figure(self, connect: Connector, tmp_path: Path) -> None:
        board = connect.to(l_board())
        figure = tmp_path / "chain.png"

        run("--dry-run", f"--plot={figure}")

        assert figure.stat().st_size > 0
        assert board.calls == []

    def test_without_matplotlib_names_the_extra(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        log: list[str],
        tmp_path: Path,
    ) -> None:
        board = connect.to(l_board())
        for name in [m for m in sys.modules if m.split(".")[0] == "matplotlib"]:
            monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.delitem(sys.modules, "kibends.plot", raising=False)

        message = exit_message(log, f"--plot={tmp_path / 'x.png'}")

        assert "plot" in message and "extra" in message
        assert board.calls == []


class TestErrors:
    def test_busy_kicad(self, connect: Connector, log: list[str]) -> None:
        connect.failing(ApiError("busy", code=ApiStatusCode.AS_BUSY))

        assert "KiCad is busy — close any open tool or dialog" in exit_message(log)

    def test_kicad_not_running(self, connect: Connector, log: list[str]) -> None:
        connect.failing(KiCadConnectionError("Failed to connect"))

        assert "Cannot reach KiCad" in exit_message(log)

    def test_no_board_open(self, connect: Connector, log: list[str]) -> None:
        connect.failing(ApiError("Expected to be able to retrieve at least one board"))

        assert "No board" in exit_message(log)

    def test_empty_selection(self, connect: Connector, log: list[str]) -> None:
        connect.to(CommitBoard())

        assert "no tracks" in exit_message(log)

    def test_arc_in_selection(self, connect: Connector, log: list[str]) -> None:
        connect.to(CommitBoard(selection=[kipy_arc("arc")]))

        assert "arc" in exit_message(log)

    def test_selection_that_is_no_chain(
        self, connect: Connector, log: list[str]
    ) -> None:
        tracks = [
            kipy_track("a", "SIG", (0, 0), (5, 0)),
            kipy_track("b", "SIG", (7, 0), (9, 0)),
        ]
        connect.to(CommitBoard(selection=tracks, tracks=tracks))

        assert "ChainError" not in exit_message(log)

    def test_refused_create_exits_with_the_reason(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = connect.to(l_board(refuse_create=True))

        assert "KiCad created 0 of" in exit_message(log)
        assert board.calls[-1] == "drop_commit"

    def test_vanished_track_exits_before_any_change(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        log: list[str],
    ) -> None:
        board = connect.to(l_board())
        board.get_items_by_id = lambda ids: []  # type: ignore[method-assign]

        assert "changed in KiCad" in exit_message(log)
        assert board.calls == []
