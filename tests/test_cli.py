import json
import re
import stat
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import NoReturn, cast

import kipy.board_types as kbt
import numpy as np
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
from shapely.geometry import LineString, Point, box

from kibends import cli, drc
from kibends import run as pipeline
from kibends.energy import EnergyParams, FloatArray
from kibends.obstacles import Obstacles
from kibends.optimize import (
    OptimizerSettings,
    RunResult,
    Termination,
    optimize_chains,
)

WIDTH = 0.25
CLEARANCE = 0.2
PAD = (3.2, 0.8)
"""Other-net pad inside the L's inner corner, between the L and its chord."""
LOCKED_ACROSS_CHORD = ((1.5, 2.5), (2.5, 1.5))
"""Other-net track across the chord the L would otherwise be pulled to."""
ON_CHORD = (2.2, 1.8)
"""Pad centre inside the L, near enough its chord to block it."""
STUB_TOUCH = (2, 0)
"""Where a same-net stub ends, mid-way along the L's first leg."""
# Rounding the created tracks to whole nanometres may cost this much clearance.
NM_ROUNDING = 2e-6


def l_chain() -> list[kbt.Track]:
    return [
        kipy_track("a", "SIG", (0, 0), (4, 0), width=WIDTH),
        kipy_track("b", "SIG", (4, 0), (4, 4), width=WIDTH),
    ]


def l_board(
    *, refuse_create: bool = False, push_error: Exception | None = None
) -> CommitBoard:
    chain = l_chain()
    return CommitBoard(
        selection=list(chain),
        tracks=list(chain),
        pads=[kipy_pad("pad", "GND", *PAD)],
        refuse_create=refuse_create,
        push_error=push_error,
    )


def straight_board() -> CommitBoard:
    """A straight chain in free space: nothing to optimise."""
    track = kipy_track("a", "SIG", (0, 0), (10, 0), width=WIDTH)
    return CommitBoard(selection=[track], tracks=[track])


@pytest.fixture(autouse=True)
def keep_test_log_sinks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "configure_logging", lambda: None)


COARSE = OptimizerSettings(spacing=0.3, burst_maxiter=30)


@pytest.fixture
def coarse_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real optimizer at coarse settings, about 5x faster.

    For tests of error mapping and commit wiring, which need a real smoothed
    chain but not a finely smoothed one.
    """
    monkeypatch.setattr(
        pipeline, "optimize_chains", partial(optimize_chains, settings=COARSE)
    )


# Answers DRC runs with the reports in reports.json next to it, keyed by the
# board's stem ("baseline" or "candidate"); no file means no violations.
# Keeps a copy of each board it checks next to it.
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
(here.parent / pcb.name).write_text(pcb.read_text())
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

    def which(name: str) -> str:
        return str(path)

    monkeypatch.setattr(drc.shutil, "which", which)
    monkeypatch.setattr(drc, "KNOWN_KICAD_CLI_PATHS", ())
    return path


def drc_runs(kicad_cli: Path) -> int:
    log = kicad_cli.with_suffix(".log")
    return len(log.read_text().splitlines()) if log.exists() else 0


SEGMENT_ENDS = re.compile(r"\(start (\S+) (\S+)\)\s*\(end (\S+) (\S+)\)")


def checked_segments(kicad_cli: Path, stem: str) -> list[tuple[int, int, int, int]]:
    """Segment endpoints (nm) in the board the fake kicad-cli checked last."""
    text = (kicad_cli.parent / f"{stem}.kicad_pcb").read_text()
    return [
        (nm(float(a)), nm(float(b)), nm(float(c)), nm(float(d)))
        for a, b, c, d in SEGMENT_ENDS.findall(text)
    ]


def report_on(kicad_cli: Path, **reports: list[drc.ReportViolation]) -> None:
    """Make the fake kicad-cli report `reports[stem]` as violations."""
    (kicad_cli.parent / "reports.json").write_text(
        json.dumps({stem: {"violations": vs} for stem, vs in reports.items()})
    )


PAD_CLEARANCE: drc.ReportViolation = {
    "type": "clearance",
    "description": "Clearance violation (rule 'pad_clearance' clearance 0.1700 mm; "
    "actual 0.1670 mm)",
    "severity": "error",
    "items": [
        {"description": "Pad 3 [GND] of J1 on F.Cu", "pos": {"x": 3.2, "y": 0.8}}
    ],
}


@pytest.fixture
def connect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Connector:
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

        self.monkeypatch.setattr(pipeline.kipy, "KiCad", kicad)
        return board

    def failing(self, error: Exception) -> None:
        class Failing:
            def __init__(self, **options: object) -> None:
                pass

            def get_board(self) -> None:
                raise error

        self.monkeypatch.setattr(pipeline.kipy, "KiCad", Failing)


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

    def test_same_net_stub_on_the_chain_keeps_its_touch_point(
        self, connect: Connector
    ) -> None:
        chain = l_chain()
        stub = kipy_track("stub", "SIG", (2, -3), STUB_TOUCH, width=WIDTH)
        board = connect.to(
            CommitBoard(
                selection=list(chain),
                tracks=[*chain, stub],
                pads=[kipy_pad("pad", "GND", *PAD)],
            )
        )

        run("--no-drc")

        assert sorted(board.deleted) == ["a", "b"]
        ends = Counter(
            (p.x, p.y)
            for t in cast(list[kbt.Track], board.created)
            for p in (t.start, t.end)
        )
        assert sorted(p for p, n in ends.items() if n == 1) == [
            (0, 0),
            (nm(4), nm(4)),
        ]
        assert ends[nm(STUB_TOUCH[0]), nm(STUB_TOUCH[1])] == 2
        assert (nm(4), 0) not in ends

    @pytest.mark.usefixtures("coarse_optimizer")
    @pytest.mark.parametrize("tip", [-0.25, -0.245, -0.24, -0.1])
    def test_selected_stub_ending_in_the_chains_copper_stays_connected(
        self, connect: Connector, tip: float
    ) -> None:
        chain = l_chain()
        stub = kipy_track("stub", "SIG", (2, -3), (2, tip), width=WIDTH)
        tracks = [*chain, stub]
        board = connect.to(CommitBoard(selection=list(tracks), tracks=list(tracks)))

        run("--no-drc")

        assert "stub" not in board.deleted
        end = Point(2, tip)
        created = cast(list[kbt.Track], board.created)
        gap = min(segment(t).distance(end) for t in created)
        assert gap <= WIDTH + NM_ROUNDING

    @pytest.mark.usefixtures("coarse_optimizer")
    @pytest.mark.parametrize("tip", [-0.25, -0.1])
    def test_selected_stub_touching_the_chain_is_no_clearance_violation(
        self, connect: Connector, tip: float
    ) -> None:
        tracks = [*l_chain(), kipy_track("stub", "SIG", (2, -3), (2, tip), width=WIDTH)]
        connect.to(CommitBoard(selection=list(tracks), tracks=list(tracks)))
        options = cli.smooth_options(cli.parse_args(["smooth", "-n", "--no-drc"]))

        report = pipeline.run_smooth(options)

        assert len(report.chains) == 2
        for c in report.chains:
            assert c.after.min_clearance >= -NM_ROUNDING
            assert c.frozen == ()

    @pytest.mark.usefixtures("coarse_optimizer")
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

    @pytest.mark.usefixtures("coarse_optimizer")
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
        real_build: Callable[..., Obstacles] = pipeline.build_obstacles

        def build_spy(*args: object, **kwargs: object) -> Obstacles:
            seen.update(kwargs)
            return real_build(*args, **kwargs)

        def optimize_spy(*args: object, **kwargs: object) -> NoReturn:
            seen["params"] = args[1]
            raise SystemExit(0)

        monkeypatch.setattr(pipeline, "build_obstacles", build_spy)
        monkeypatch.setattr(pipeline, "optimize_chains", optimize_spy)

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

    @pytest.mark.usefixtures("coarse_optimizer")
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
        real_simplify: Callable[..., list[FloatArray]] = pipeline.simplify_chains

        def simplify_spy(*a: object, **kwargs: object) -> list[FloatArray]:
            seen.update(kwargs)
            return real_simplify(*a, **kwargs)

        monkeypatch.setattr(pipeline, "simplify_chains", simplify_spy)

        run("--dry-run", *args)

        assert seen["tolerance"] == expected

    def test_negative_weight_is_rejected(self) -> None:
        with pytest.raises(SystemExit) as raised:
            run("--stiffness=-1")

        assert raised.value.code == 2

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_abnormal_termination_is_refused(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        log: list[str],
    ) -> None:
        board = connect.to(l_board())
        real: Callable[..., RunResult] = pipeline.optimize_chains

        def abnormal(*args: object, **kwargs: object) -> RunResult:
            return replace(real(*args, **kwargs), termination=Termination.ABNORMAL)

        monkeypatch.setattr(pipeline, "optimize_chains", abnormal)

        assert "abnormally" in exit_message(log)
        assert board.calls == []

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_iteration_cap_warns_and_still_checks_and_commits(
        self,
        connect: Connector,
        kicad_cli: Path,
        monkeypatch: pytest.MonkeyPatch,
        log: list[str],
    ) -> None:
        board = connect.to(l_board())
        real: Callable[..., RunResult] = pipeline.optimize_chains

        def capped(*args: object, **kwargs: object) -> RunResult:
            return replace(real(*args, **kwargs), termination=Termination.ITERATION_CAP)

        monkeypatch.setattr(pipeline, "optimize_chains", capped)

        run()

        assert any(
            line.startswith("WARNING") and "without converging" in line for line in log
        )
        assert drc_runs(kicad_cli) == 2
        assert "push_commit" in board.calls


class TestDrc:
    @pytest.mark.usefixtures("coarse_optimizer")
    def test_clean_result_is_committed_after_both_runs(
        self, connect: Connector, kicad_cli: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        board = connect.to(l_board())

        run()

        assert drc_runs(kicad_cli) == 2
        assert "push_commit" in board.calls
        assert "DRC: no new violations" in capsys.readouterr().out

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_checks_the_tracks_it_commits(
        self, connect: Connector, kicad_cli: Path
    ) -> None:
        board = connect.to(l_board())

        run()

        created = cast(list[kbt.Track], board.created)
        assert sorted(checked_segments(kicad_cli, "candidate")) == sorted(
            (t.start.x, t.start.y, t.end.x, t.end.y) for t in created
        )
        candidate = (kicad_cli.parent / "candidate.kicad_pcb").read_text()
        assert '(uuid "a")' not in candidate and '(uuid "b")' not in candidate

    @pytest.mark.usefixtures("coarse_optimizer")
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

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_violation_already_in_the_baseline_does_not_block(
        self, connect: Connector, kicad_cli: Path
    ) -> None:
        board = connect.to(l_board())
        report_on(kicad_cli, baseline=[PAD_CLEARANCE], candidate=[PAD_CLEARANCE])

        run()

        assert "push_commit" in board.calls

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_split_pour_names_the_net(
        self,
        connect: Connector,
        kicad_cli: Path,
        log: list[str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        connect.to(l_board())
        unconnected: drc.ReportViolation = {
            "type": "unconnected_items",
            "description": "Missing connection between items",
            "severity": "error",
            "items": [{"description": "Zone [GND] on F.Cu", "pos": {"x": 1, "y": 1}}],
        }
        report_on(kicad_cli, candidate=[unconnected])

        exit_message(log)

        assert "split a pour of GND" in capsys.readouterr().out

    @pytest.mark.usefixtures("coarse_optimizer")
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

    @pytest.mark.usefixtures("coarse_optimizer")
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

        def never(*args: object, **kwargs: object) -> NoReturn:
            pytest.fail("optimized")

        monkeypatch.setattr(pipeline, "optimize_chains", never)

        message = exit_message(log)

        assert "save" in message
        assert board.calls == []

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_says_unsaved_rule_edits_are_not_seen(
        self, connect: Connector, log: list[str]
    ) -> None:
        connect.to(l_board())

        run("--dry-run")

        assert any("unsaved" in line and "not seen" in line for line in log)

    @pytest.mark.usefixtures("coarse_optimizer")
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

    @pytest.mark.usefixtures("coarse_optimizer")
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
    @pytest.mark.usefixtures("coarse_optimizer")
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

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_kicad_client_waits_longer_than_kipys_default(
        self, connect: Connector
    ) -> None:
        connect.to(l_board())

        run("--dry-run")

        assert connect.client_options == {"timeout_ms": pipeline.KICAD_TIMEOUT_MS}
        assert pipeline.KICAD_TIMEOUT_MS > 2000

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
            kipy_track("b", "SIG", (5, 0), (9, 0)),
            kipy_track("c", "SIG", (5, 0), (5, 3)),
        ]
        connect.to(CommitBoard(selection=list(tracks), tracks=list(tracks)))

        message = exit_message(log)

        assert "selected tracks branch at" in message
        assert "ChainError" not in message

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_refused_create_exits_with_the_reason(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = connect.to(l_board(refuse_create=True))

        assert "KiCad created 0 of" in exit_message(log)
        assert board.calls[-1] == "drop_commit"

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_push_refused_by_kicad_exits_with_the_reason(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = connect.to(l_board(push_error=ApiError("nope")))

        message = exit_message(log, "--no-drc")

        assert "KiCad refused the request" in message and "nope" in message
        assert board.calls[-1] == "drop_commit"

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_lost_connection_on_push_exits_with_a_warning(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = connect.to(l_board(push_error=KiCadConnectionError("timed out")))

        message = exit_message(log)

        assert "may have landed" in message and "undo history" in message
        assert "drop_commit" not in board.calls

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_vanished_track_exits_before_any_change(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        log: list[str],
    ) -> None:
        board = connect.to(l_board())
        board.get_items_by_id = lambda ids: []

        assert "changed in KiCad" in exit_message(log)
        assert board.calls == []


FAILING_DRC = """
import sys
if sys.argv[1:] == ["version"]:
    print("10.0.8")
    sys.exit(0)
print("DRC crashed", file=sys.stderr)
sys.exit(3)
"""


@pytest.mark.usefixtures("coarse_optimizer")
class TestFailureAfterOptimizing:
    """The statistics and warnings still reach the user before the error."""

    def assert_reported_before(
        self, error: str, log: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = capsys.readouterr().out
        assert "2 tracks of SIG on F.Cu, 0.25 mm wide" in out
        assert re.search(r"^Optimizer: converged after", out, re.MULTILINE)
        (warning,) = [i for i, line in enumerate(log) if "abnormally" in line]
        assert warning < log.index(error)

    def test_drc_failure(
        self,
        connect: Connector,
        kicad_cli: Path,
        monkeypatch: pytest.MonkeyPatch,
        log: list[str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        board = connect.to(l_board())
        kicad_cli.write_text(f"#!{sys.executable}\n{FAILING_DRC}")
        with_abnormal_bursts(monkeypatch, [2])

        error = exit_message(log)

        assert "DRC crashed" in error
        self.assert_reported_before(error, log, capsys)
        assert board.calls == []

    def test_lost_connection_on_push(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        log: list[str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        connect.to(l_board(push_error=KiCadConnectionError("timed out")))
        with_abnormal_bursts(monkeypatch, [2])

        error = exit_message(log, "--no-drc")

        assert "may have landed" in error
        self.assert_reported_before(error, log, capsys)


INNER_OFFSET = 0.6
"""How far the inner L of `two_l_board` lies inside the outer one, mm: clear
of it, but their chords would be closer than clearance plus width."""
OUTER = ("a", "b")
INNER = ("c", "d")


def two_l_board(*, inner: list[kbt.Track] | None = None) -> CommitBoard:
    """Two nested L chains of different nets that must push each other apart."""
    o = INNER_OFFSET
    outer = l_chain()
    if inner is None:
        inner = [
            kipy_track("c", "SIG2", (0, o), (4 - o, o), width=WIDTH),
            kipy_track("d", "SIG2", (4 - o, o), (4 - o, 4), width=WIDTH),
        ]
    return CommitBoard(selection=[*outer, *inner], tracks=[*outer, *inner])


def far_straight() -> list[kbt.Track]:
    """A SIG2 chain in free space far from the L: nothing to optimise."""
    return [kipy_track("s", "SIG2", (0, 10), (10, 10), width=WIDTH)]


def segment(t: kbt.Track) -> LineString:
    return LineString(
        [(t.start.x / 1e6, t.start.y / 1e6), (t.end.x / 1e6, t.end.y / 1e6)]
    )


SIG2_TRACK_CLEARANCE: drc.ReportViolation = {
    "type": "clearance",
    "description": "Clearance violation (netclass 'Default' clearance 0.2000 mm; "
    "actual 0.1500 mm)",
    "severity": "error",
    "items": [
        {"description": "Track [SIG2] on F.Cu, length 1.0 mm", "pos": {"x": 1, "y": 1}}
    ],
}


@pytest.mark.usefixtures("coarse_optimizer")
class TestSeveralChains:
    def test_commits_both_in_one_commit_and_prints_a_block_for_each(
        self, connect: Connector, kicad_cli: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        board = connect.to(two_l_board())

        run()

        assert board.calls == ["begin_commit", "delete", "create", "push_commit"]
        assert sorted(board.deleted) == [*OUTER, *INNER]
        created = cast(list[kbt.Track], board.created)
        by_net = {
            net: [segment(t) for t in created if t.net.name == net]
            for net in ("SIG", "SIG2")
        }
        assert all(by_net.values())
        required = CLEARANCE + WIDTH
        assert all(
            a.distance(b) >= required - NM_ROUNDING
            for a in by_net["SIG"]
            for b in by_net["SIG2"]
        )
        assert drc_runs(kicad_cli) == 2
        out = capsys.readouterr().out
        assert "2 tracks of SIG on F.Cu, 0.25 mm wide" in out
        assert "2 tracks of SIG2 on F.Cu, 0.25 mm wide" in out
        assert out.count("Frozen vertices: 0") == 2
        assert out.count("Optimizer: ") == 1

    def test_dry_run_prints_both_and_commits_nothing(
        self, connect: Connector, capsys: pytest.CaptureFixture[str]
    ) -> None:
        board = connect.to(two_l_board())

        run("--dry-run")

        out = capsys.readouterr().out
        assert "of SIG on" in out and "of SIG2 on" in out
        assert board.calls == []

    def test_unchanged_chain_is_neither_checked_nor_committed(
        self,
        connect: Connector,
        kicad_cli: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        board = connect.to(two_l_board(inner=far_straight()))

        run()

        assert sorted(board.deleted) == list(OUTER)
        assert {t.net.name for t in cast(list[kbt.Track], board.created)} == {"SIG"}
        candidate = (kicad_cli.parent / "candidate.kicad_pcb").read_text()
        assert '(uuid "s")' in candidate
        assert "Unchanged: left out of the DRC check and the commit" in (
            capsys.readouterr().out
        )

    def test_chain_that_only_gains_straight_vertices_stays_unchanged(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        board = connect.to(two_l_board(inner=far_straight()))
        real: Callable[..., list[FloatArray]] = pipeline.simplify_chains

        def with_midpoints(*args: object, **kwargs: object) -> list[FloatArray]:
            outer, straight = real(*args, **kwargs)
            midpoints = (straight[:-1] + straight[1:]) / 2
            split = np.insert(straight, range(1, len(straight)), midpoints, axis=0)
            return [outer, split]

        monkeypatch.setattr(pipeline, "simplify_chains", with_midpoints)

        run("--no-drc")

        assert sorted(board.deleted) == list(OUTER)
        straight = capsys.readouterr().out.split("\n\n")[1]
        assert re.search(r"^Tracks +1 +1$", straight, re.MULTILINE)

    def test_all_unchanged_commits_nothing(
        self, connect: Connector, kicad_cli: Path, log: list[str]
    ) -> None:
        tracks = [
            kipy_track("a", "SIG", (0, 0), (10, 0), width=WIDTH),
            *far_straight(),
        ]
        board = connect.to(CommitBoard(selection=list(tracks), tracks=list(tracks)))

        run()

        assert board.calls == []
        assert drc_runs(kicad_cli) == 0
        assert any("Nothing to do" in line for line in log)

    def test_abnormal_run_commits_nothing(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch, log: list[str]
    ) -> None:
        board = connect.to(two_l_board())
        real: Callable[..., RunResult] = pipeline.optimize_chains

        def abnormal(*args: object, **kwargs: object) -> RunResult:
            return replace(real(*args, **kwargs), termination=Termination.ABNORMAL)

        monkeypatch.setattr(pipeline, "optimize_chains", abnormal)

        assert "abnormally" in exit_message(log)
        assert board.calls == []

    def test_drc_refusal_names_the_violating_chains_net(
        self, connect: Connector, kicad_cli: Path, log: list[str]
    ) -> None:
        board = connect.to(two_l_board())
        report_on(kicad_cli, candidate=[SIG2_TRACK_CLEARANCE])

        message = exit_message(log)

        assert "1 new violation on SIG2; nothing committed" in message
        assert board.calls == []

    def test_frozen_warning_names_the_other_chain(
        self, connect: Connector, log: list[str]
    ) -> None:
        # 0.3 mm from the L's corner, inside clearance plus both half widths.
        across = [kipy_track("x", "SIG2", (4.3, -1), (4.3, 1), width=WIDTH)]
        connect.to(two_l_board(inner=across))

        run("--dry-run", "--no-drc")

        frozen = [line for line in log if "Left the vertex at (4.000, 0.000)" in line]
        assert len(frozen) == 1
        assert frozen[0].startswith("WARNING")
        assert "selected chain SIG2 on F.Cu from (4.300, -1.000)" in frozen[0]

    def test_run_function_reports_each_chain(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        board = connect.to(two_l_board())
        options = cli.smooth_options(cli.parse_args(["smooth", "-n", "--no-drc"]))
        simplified: list[FloatArray] = []
        real: Callable[..., list[FloatArray]] = pipeline.simplify_chains

        def spy(*args: object, **kwargs: object) -> list[FloatArray]:
            simplified.extend(real(*args, **kwargs))
            return simplified

        monkeypatch.setattr(pipeline, "simplify_chains", spy)

        report = pipeline.run_smooth(options)

        assert [(c.net, c.layer, c.width, c.tracks) for c in report.chains] == [
            ("SIG", "F.Cu", WIDTH, 2),
            ("SIG2", "F.Cu", WIDTH, 2),
        ]
        for c in report.chains:
            assert c.changed
            assert c.after.length < c.before.length
            assert c.after.min_clearance >= -NM_ROUNDING
            assert (c.frozen, c.abnormal_bursts) == ((), 0)
        assert report.termination in (Termination.CONVERGED, Termination.ITERATION_CAP)
        assert report.iterations > 0
        assert (report.drc, report.committed, report.refusal) == (None, False, None)
        assert board.calls == []
        # Nothing else is near: only the other chain sets the clearance margin.
        gap = LineString(simplified[0]).distance(LineString(simplified[1]))
        for c in report.chains:
            assert c.after.min_clearance == pytest.approx(
                gap - (CLEARANCE + WIDTH), abs=1e-3
            )

    def test_other_nets_pad_stays_an_obstacle_at_equal_widths(
        self, connect: Connector
    ) -> None:
        sig = kipy_track("s", "SIG", (0, 10), (10, 10), width=WIDTH)
        sig2 = [
            kipy_track("c", "SIG2", (0, 0), (4, 0), width=WIDTH),
            kipy_track("d", "SIG2", (4, 0), (4, 4), width=WIDTH),
        ]
        tracks = [sig, *sig2]
        board = connect.to(
            CommitBoard(
                selection=list(tracks),
                tracks=list(tracks),
                pads=[kipy_pad("pad", "SIG", *ON_CHORD)],
            )
        )

        run("--no-drc")

        assert sorted(board.deleted) == list(INNER)
        x, y = ON_CHORD
        h = PAD_HALF_SIZE
        pad = box(x - h, y - h, x + h, y + h)
        required = CLEARANCE + WIDTH / 2
        for t in cast(list[kbt.Track], board.created):
            assert segment(t).distance(pad) >= required - NM_ROUNDING

    def test_plot_draws_both_chains(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from kibends import plot

        connect.to(two_l_board())
        drawn: list[int] = []
        real = plot.render

        def spy(path: Path, panels: Sequence[plot.Panel]) -> None:
            drawn.append(len(panels))
            real(path, panels)

        monkeypatch.setattr(plot, "render", spy)
        figure = tmp_path / "chains.png"

        run("--dry-run", "--no-drc", f"--plot={figure}")

        assert drawn == [2]
        assert figure.stat().st_size > 0


def with_abnormal_bursts(
    monkeypatch: pytest.MonkeyPatch, counts: Sequence[int]
) -> None:
    """Make the optimizer report `counts` abnormal bursts per chain."""
    real: Callable[..., RunResult] = pipeline.optimize_chains

    def abnormal(*args: object, **kwargs: object) -> RunResult:
        result = real(*args, **kwargs)
        chains = tuple(
            replace(c, abnormal_bursts=n)
            for c, n in zip(result.chains, counts, strict=True)
        )
        return replace(result, chains=chains)

    monkeypatch.setattr(pipeline, "optimize_chains", abnormal)


@pytest.mark.usefixtures("coarse_optimizer")
class TestAbnormalBursts:
    def test_single_chain_shows_them_on_the_optimizer_line(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        connect.to(l_board())
        with_abnormal_bursts(monkeypatch, [3])

        run("--dry-run", "--no-drc")

        assert re.search(
            r"^Optimizer: converged after \d+ iterations, \d+ rejected edge "
            r"moves, 3 abnormal bursts$",
            capsys.readouterr().out,
            re.MULTILINE,
        )

    def test_several_chains_name_the_chain_on_the_optimizer_line(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        log: list[str],
    ) -> None:
        connect.to(two_l_board())
        with_abnormal_bursts(monkeypatch, [0, 1])

        run("--dry-run", "--no-drc")

        (line,) = [
            ln for ln in capsys.readouterr().out.splitlines() if "Optimizer:" in ln
        ]
        assert line.endswith(
            "rejected edge moves, 1 abnormal burst in SIG2 on F.Cu from (0.000, 0.600)"
        )
        assert "SIG on" not in line
        (warning,) = [ln for ln in log if "abnormally" in ln]
        assert warning.startswith("WARNING")
        assert "1 burst of SIG2 on F.Cu from (0.000, 0.600)" in warning


@pytest.mark.usefixtures("coarse_optimizer")
def test_single_chain_report_keeps_its_layout(
    connect: Connector, capsys: pytest.CaptureFixture[str]
) -> None:
    connect.to(l_board())

    run("--dry-run")

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "2 tracks of SIG on F.Cu, 0.25 mm wide"
    assert lines[1].split() == ["before", "after"]
    assert [line.split("  ")[0] for line in lines[2:7]] == [
        "Length (mm)",
        "Tracks",
        "Tightest bend radius (mm)",
        "Min clearance margin (mm)",
        "Mean clearance margin (mm)",
    ]
    assert lines[7] == "Frozen vertices: 0"
    assert re.fullmatch(
        r"Optimizer: converged after \d+ iterations, \d+ rejected edge moves",
        lines[8],
    )
    assert lines[9:] == ["DRC: no new violations"]
