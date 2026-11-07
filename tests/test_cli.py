import json
import stat
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

from kibends import cli, drc
from kibends.energy import EnergyParams
from kibends.obstacles import Obstacles
from kibends.optimize import OptimizeResult, Termination

WIDTH = 0.25
CLEARANCE = 0.2
PAD = (3.2, 0.8)
"""Other-net pad inside the L's inner corner, between the L and its chord."""
LOCKED_ACROSS_CHORD = ((1.5, 2.5), (2.5, 1.5))
"""Other-net track across the chord the L would otherwise be pulled to."""
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


# Answers DRC runs with the reports in reports.json next to it, keyed by the
# board's stem ("baseline" or "candidate"); no file means no violations.
FAKE_KICAD_CLI = """
import json, pathlib, sys
here = pathlib.Path(sys.argv[0])
args = sys.argv[1:]
if args == ["version"]:
    print("10.0.8")
    sys.exit(0)
with here.with_suffix(".log").open("a") as f:
    f.write(json.dumps(args) + "\\n")
pcb = pathlib.Path(args[-1])
reports = here.parent / "reports.json"
report = json.loads(reports.read_text()).get(pcb.stem, {}) if reports.exists() else {}
pathlib.Path(args[args.index("-o") + 1]).write_text(json.dumps(report))
"""


def write_kicad_cli(directory: Path) -> Path:
    directory.mkdir(exist_ok=True)
    path = directory / "kicad-cli"
    path.write_text(f"#!{sys.executable}\n{FAKE_KICAD_CLI}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture(autouse=True)
def kicad_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = write_kicad_cli(tmp_path / "bin")
    monkeypatch.setattr(drc.shutil, "which", lambda name: str(path))
    monkeypatch.setattr(drc, "KNOWN_KICAD_CLI_PATHS", ())
    return path


def drc_runs(kicad_cli: Path) -> int:
    log = kicad_cli.with_suffix(".log")
    return len(log.read_text().splitlines()) if log.exists() else 0


def report_on(kicad_cli: Path, **reports: list[dict[str, Any]]) -> None:
    """Make the fake kicad-cli report `reports[stem]` as violations."""
    (kicad_cli.parent / "reports.json").write_text(
        json.dumps({stem: {"violations": vs} for stem, vs in reports.items()})
    )


PAD_CLEARANCE = {
    "type": "clearance",
    "description": "Clearance violation (rule 'pad_clearance' clearance 0.1700 mm; "
    "actual 0.1670 mm)",
    "severity": "error",
    "items": [
        {"description": "Pad 3 [GND] of J1 on F.Cu", "pos": {"x": 3.2, "y": 0.8}}
    ],
}


@pytest.fixture
def connect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> "Connector":
    return Connector(monkeypatch, tmp_path / "project")


class Connector:
    """Points kipy.KiCad() at a fake board, or makes it fail.

    The board is saved to a project directory unless `saved` is false.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
        self.monkeypatch = monkeypatch
        self.directory = directory
        self.client_options: dict[str, object] = {}

    def to(self, board: CommitBoard, saved: bool = True) -> CommitBoard:
        if saved:
            self.directory.mkdir(exist_ok=True)
            pcb = self.directory / board.name
            pcb.write_text(board.get_as_string())
            pcb.with_suffix(".kicad_pro").write_text("{}")
            board.project_dir = str(self.directory)

        def kicad(**options: object) -> FakeKiCad:
            self.client_options = options
            return FakeKiCad(board)

        self.monkeypatch.setattr(cli.kipy, "KiCad", kicad)
        return board

    def failing(self, error: Exception) -> None:
        class Failing:
            def __init__(self, **options: object) -> None:
                pass

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

    def test_locked_other_net_track_in_the_selection_stays_an_obstacle(
        self, connect: Connector
    ) -> None:
        chain = l_chain()
        locked = kipy_track(
            "locked", "GND", *LOCKED_ACROSS_CHORD, width=WIDTH, locked=True
        )
        board = connect.to(
            CommitBoard(selection=[*chain, locked], tracks=[*chain, locked])
        )

        run()

        assert sorted(board.deleted) == ["a", "b"]
        obstacle = LineString(LOCKED_ACROSS_CHORD)
        required = CLEARANCE + WIDTH
        for t in cast(list[kbt.Track], board.created):
            segment = LineString(
                [(t.start.x / 1e6, t.start.y / 1e6), (t.end.x / 1e6, t.end.y / 1e6)]
            )
            assert segment.distance(obstacle) >= required - NM_ROUNDING

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


class TestDrc:
    def test_clean_result_is_committed_after_both_runs(
        self, connect: Connector, kicad_cli: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        board = connect.to(l_board())

        run()

        assert drc_runs(kicad_cli) == 2
        assert "push_commit" in board.calls
        assert "DRC: no new violations" in capsys.readouterr().out

    def test_introduced_violation_refuses_to_commit_and_lists_it(
        self,
        connect: Connector,
        kicad_cli: Path,
        log: list[str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        board = connect.to(l_board())
        report_on(kicad_cli, candidate=[PAD_CLEARANCE])

        message = exit_message(log)

        assert "1 new violation" in message and "nothing committed" in message
        out = capsys.readouterr().out
        assert "clearance: Clearance violation (rule 'pad_clearance'" in out
        assert "Pad 3 [GND] of J1 on F.Cu @ (3.200, 0.800)" in out
        assert board.calls == []

    def test_violation_already_in_the_baseline_does_not_block(
        self, connect: Connector, kicad_cli: Path
    ) -> None:
        board = connect.to(l_board())
        report_on(kicad_cli, baseline=[PAD_CLEARANCE], candidate=[PAD_CLEARANCE])

        run()

        assert "push_commit" in board.calls

    def test_split_pour_names_the_net(
        self,
        connect: Connector,
        kicad_cli: Path,
        log: list[str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        connect.to(l_board())
        unconnected = {
            "type": "unconnected_items",
            "description": "Missing connection between items",
            "severity": "error",
            "items": [{"description": "Zone [GND] on F.Cu", "pos": {"x": 1, "y": 1}}],
        }
        report_on(kicad_cli, candidate=[unconnected])

        exit_message(log)

        assert "split a pour of GND" in capsys.readouterr().out

    def test_dry_run_prints_the_verdict_and_commits_nothing(
        self,
        connect: Connector,
        kicad_cli: Path,
        log: list[str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        board = connect.to(l_board())
        report_on(kicad_cli, candidate=[PAD_CLEARANCE])

        run("--dry-run")

        out = capsys.readouterr().out
        assert "DRC: 1 new violation" in out
        assert "Pad 3 [GND]" in out
        assert any("would refuse" in line for line in log)
        assert board.calls == []

    def test_no_drc_commits_without_running_it_and_warns(
        self, connect: Connector, kicad_cli: Path, log: list[str]
    ) -> None:
        board = connect.to(l_board(), saved=False)
        report_on(kicad_cli, candidate=[PAD_CLEARANCE])

        run("--no-drc")

        assert drc_runs(kicad_cli) == 0
        assert "push_commit" in board.calls
        assert any(line.startswith("WARNING") and "--no-drc" in line for line in log)

    def test_unsaved_board_is_refused_before_optimizing(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch, log: list[str]
    ) -> None:
        board = connect.to(l_board(), saved=False)
        monkeypatch.setattr(cli, "optimize", lambda *a, **k: pytest.fail("optimized"))

        message = exit_message(log)

        assert "save" in message
        assert board.calls == []

    def test_says_unsaved_rule_edits_are_not_seen(
        self, connect: Connector, log: list[str]
    ) -> None:
        connect.to(l_board())

        run("--dry-run")

        assert any("unsaved" in line and "not seen" in line for line in log)

    def test_kicad_cli_option_picks_the_binary(
        self, connect: Connector, kicad_cli: Path, tmp_path: Path
    ) -> None:
        connect.to(l_board())
        other = write_kicad_cli(tmp_path / "other")

        run("--dry-run", f"--kicad-cli={other}")

        assert drc_runs(other) == 2
        assert drc_runs(kicad_cli) == 0

    def test_missing_kicad_cli_suggests_no_drc(
        self, connect: Connector, tmp_path: Path, log: list[str]
    ) -> None:
        connect.to(l_board())

        message = exit_message(log, f"--kicad-cli={tmp_path / 'nope'}")

        assert "not an executable" in message

    def test_nothing_to_do_runs_no_drc(
        self, connect: Connector, kicad_cli: Path
    ) -> None:
        connect.to(straight_board())

        run()

        assert drc_runs(kicad_cli) == 0

    def test_track_missing_from_the_board_text_fails_before_drc(
        self,
        connect: Connector,
        kicad_cli: Path,
        monkeypatch: pytest.MonkeyPatch,
        log: list[str],
    ) -> None:
        board = connect.to(l_board())
        monkeypatch.setattr(board, "get_as_string", lambda: "(kicad_pcb\n)\n")

        assert "not found in the board text" in exit_message(log)
        assert drc_runs(kicad_cli) == 0
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

        message = exit_message(log)

        assert "Cannot reach KiCad" in message
        assert "not running" in message and "timed out" in message

    def test_kicad_client_waits_longer_than_kipys_default(
        self, connect: Connector
    ) -> None:
        connect.to(l_board())

        run("--dry-run")

        assert connect.client_options == {"timeout_ms": cli.KICAD_TIMEOUT_MS}
        assert cli.KICAD_TIMEOUT_MS > 2000

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

        message = exit_message(log)

        assert "selected tracks form 2 separate pieces" in message
        assert "ChainError" not in message

    def test_refused_create_exits_with_the_reason(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = connect.to(l_board(refuse_create=True))

        assert "KiCad created 0 of" in exit_message(log)
        assert board.calls[-1] == "drop_commit"

    def test_lost_connection_on_push_exits_with_a_warning(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = connect.to(l_board(push_error=KiCadConnectionError("timed out")))

        message = exit_message(log)

        assert "may have landed" in message and "undo history" in message
        assert "drop_commit" not in board.calls

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
