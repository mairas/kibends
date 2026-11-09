import contextlib
import itertools
import math
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import NoReturn, cast

import kipy.board_types as kbt
import numpy as np
import pytest
import scenarios
import shapely
from conftest import (
    PAD_HALF_SIZE,
    BurstClock,
    CommitBoard,
    FakeKiCad,
    Ticking,
    kipy_arc,
    kipy_filled_zone,
    kipy_pad,
    kipy_polygon,
    kipy_rect,
    kipy_track,
    kipy_via,
    kipy_zone,
    netclass,
    nm,
)
from distance_fields import FreeSpace
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.geometry import PolygonWithHoles
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.board.board_types_pb2 import ViaType
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from kipy.util.board_layer import iter_copper_layers
from shapely.geometry import LineString, Point, box
from shapely.ops import linemerge, unary_union

from kibends import cli, plot
from kibends import run as pipeline
from kibends.bursts import (
    DEFAULT_SETTINGS,
    INSIDE_TOLERANCE,
    OptimizerSettings,
)
from kibends.chain import NM_PER_MM
from kibends.coords import BOARD_FRAME, UserFrame
from kibends.energy import EnergyParams, FloatArray
from kibends.neighbours import ChainInput
from kibends.obstacles import ChainCentreline, Obstacles
from kibends.optimize import (
    MAX_ABNORMAL_BURSTS,
    FrozenVertex,
    PolylineStats,
    RunResult,
    Termination,
    optimize_chains,
)
from kibends.pair_blocks import Hold, PairBlock, PairInput
from kibends.pairs import GAP_TOLERANCE, BendCheck, gap_range, offset_pair
from kibends.snapshot import KICAD_MAX_ERROR, LayerSnapshot, ViaZone

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
    *, refuse_create: bool = False, push_error: BaseException | None = None
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


COARSE = OptimizerSettings(spacing=0.3, settled_change=0.003, burst_maxiter=30)


KEPT = 1e-3
"""mm a chain's length may rise by between runs, as repulsion trades it."""


def path_length(points: FloatArray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def at(settings: OptimizerSettings) -> Callable[..., RunResult]:
    """`optimize_chains` at `settings`, but the time limit the run gives it."""
    real: Callable[..., RunResult] = optimize_chains

    def optimize(
        chains: object, params: object, given: OptimizerSettings, **kwargs: object
    ) -> RunResult:
        return real(
            chains, params, replace(settings, time_limit=given.time_limit), **kwargs
        )

    return optimize


@pytest.fixture
def coarse_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real optimizer at coarse settings, about 5x faster.

    For tests of error mapping and commit wiring, which need a real smoothed
    chain but not a finely smoothed one.
    """
    monkeypatch.setattr(pipeline, "optimize_chains", at(COARSE))


GRID_ORIGIN = (1.0, 2.0)
"""A grid origin away from the page origin, in mm, which every printed
position is relative to."""


def off_grid(board: CommitBoard) -> CommitBoard:
    """`board` with its grid origin at GRID_ORIGIN."""
    board.grid_origin = GRID_ORIGIN
    return board


@pytest.fixture
def connect(monkeypatch: pytest.MonkeyPatch) -> Connector:
    return Connector(monkeypatch)


class Connector:
    """Points kipy.KiCad() at a fake board, or makes it fail."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch
        self.client_options: dict[str, object] = {}

    def to(self, board: CommitBoard) -> CommitBoard:
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


@pytest.fixture
def reports(monkeypatch: pytest.MonkeyPatch) -> list[pipeline.RunReport]:
    """The report of each run the command makes, which it does not print."""
    captured: list[pipeline.RunReport] = []
    real = cli.run_smooth

    def spy(
        options: pipeline.SmoothOptions,
        on_optimized: Callable[[pipeline.OptimizerReport], None],
    ) -> pipeline.RunReport:
        report = real(options, on_optimized)
        captured.append(report)
        return report

    monkeypatch.setattr(cli, "run_smooth", spy)
    return captured


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

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_a_tee_moves_its_junction_and_commits_branches_that_meet_there(
        self, connect: Connector
    ) -> None:
        tee = [
            kipy_track("a", "SIG", (0, 0), (5, 0), width=WIDTH),
            kipy_track("b", "SIG", (5, 0), (10, 0), width=WIDTH),
            kipy_track("c", "SIG", (5, 0), (5, 5), width=WIDTH),
        ]
        board = connect.to(CommitBoard(selection=list(tee), tracks=list(tee)))

        run()

        assert sorted(board.deleted) == ["a", "b", "c"]
        degree = Counter(
            (p.x, p.y)
            for t in cast(list[kbt.Track], board.created)
            for p in (t.start, t.end)
        )
        (junction,) = [p for p, n in degree.items() if n == 3]
        ends = {p for p, n in degree.items() if n == 1}
        assert ends == {(nm(0), nm(0)), (nm(10), nm(0)), (nm(5), nm(5))}
        assert set(degree.values()) <= {1, 2, 3}
        # Pulled toward the branches' Fermat point, 5 tan 30 deg below the bar.
        assert junction[0] == pytest.approx(nm(5), abs=nm(0.01))
        assert nm(2.0) < junction[1] < nm(3.2)

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_a_track_ending_mid_track_joins_it_at_a_junction(
        self, connect: Connector
    ) -> None:
        tee = [
            kipy_track("a", "SIG", (0, 0), (10, 0), width=WIDTH),
            kipy_track("c", "SIG", (5, 5), (5, 0), width=WIDTH),
        ]
        board = connect.to(CommitBoard(selection=list(tee), tracks=list(tee)))

        run()

        assert sorted(board.deleted) == ["a", "c"]
        degree = Counter(
            (p.x, p.y)
            for t in cast(list[kbt.Track], board.created)
            for p in (t.start, t.end)
        )
        (junction,) = [p for p, n in degree.items() if n == 3]
        ends = {p for p, n in degree.items() if n == 1}
        assert ends == {(nm(0), nm(0)), (nm(10), nm(0)), (nm(5), nm(5))}
        assert nm(2.0) < junction[1] < nm(3.2)

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_a_neck_down_overlapping_along_its_line_stays_on_it(
        self, connect: Connector
    ) -> None:
        tracks = [
            kipy_track("thin", "SIG", (0, 0), (10, 0), width=0.25),
            kipy_track("w1", "SIG", (9.775, 0), (10, 0), width=0.5),
            kipy_track("w2", "SIG", (10, 0), (20, 0), width=0.5),
        ]
        board = connect.to(CommitBoard(selection=list(tracks), tracks=list(tracks)))

        run()

        created = cast(list[kbt.Track], board.created)
        line = LineString([(0, 0), (20, 0)])
        assert all(line.distance(segment(t)) <= NM_ROUNDING for t in created)
        thin, wide = (
            unary_union([segment(t) for t in created if t.width == nm(w)])
            for w in (0.25, 0.5)
        )
        assert thin.covers(LineString([(0, 0), (9.775, 0)]))
        assert wide.covers(LineString([(10, 0), (20, 0)]))
        assert thin.union(wide).covers(line)

    def test_an_untouched_tee_along_one_track_commits_nothing(
        self, connect: Connector, log: list[str]
    ) -> None:
        tee = [
            kipy_track("a", "SIG", (0, 0), (10, 0), width=WIDTH),
            kipy_track("c", "SIG", (5, 5), (5, 0), width=WIDTH),
        ]
        board = connect.to(
            CommitBoard(
                selection=list(tee),
                tracks=list(tee),
                pads=[kipy_pad("pad", "SIG", 5, 0)],
            )
        )

        run()

        assert board.calls == []
        assert any("Nothing to do" in line for line in log)

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_a_split_track_is_replaced_whole_when_one_part_changes(
        self, connect: Connector, reports: list[pipeline.RunReport]
    ) -> None:
        tee = [
            kipy_track("a", "SIG", (0, 0), (10, 0), width=WIDTH),
            kipy_track("b", "SIG", (10, 0), (10, 4), width=WIDTH),
            kipy_track("c", "SIG", (5, 5), (5, 0), width=WIDTH),
        ]
        board = connect.to(
            CommitBoard(
                selection=list(tee),
                tracks=list(tee),
                pads=[kipy_pad("pad", "SIG", 5, 0)],
            )
        )

        run()

        assert sorted(board.deleted) == ["a", "b"]
        (report,) = reports
        assert sorted(c.changed for c in report.chains) == [False, True, True]
        created = cast(list[kbt.Track], board.created)
        assert unary_union([segment(t) for t in created]).covers(
            LineString([(0, 0), (5, 0)])
        )

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_a_change_spreads_through_two_split_tracks(
        self, connect: Connector, reports: list[pipeline.RunReport]
    ) -> None:
        tracks = [
            kipy_track("a1", "SIG", (0, -3), (0, 0), width=WIDTH),
            kipy_track("a", "SIG", (0, 0), (10, 0), width=WIDTH),
            kipy_track("b", "SIG", (10, 0), (20, 0), width=WIDTH),
            kipy_track("s", "SIG", (5, 5), (5, 0), width=WIDTH),
            kipy_track("t", "SIG", (15, 5), (15, 0), width=WIDTH),
        ]
        board = connect.to(
            CommitBoard(
                selection=list(tracks),
                tracks=list(tracks),
                pads=[kipy_pad("p1", "SIG", 5, 0), kipy_pad("p2", "SIG", 15, 0)],
            )
        )

        run()

        assert sorted(board.deleted) == ["a", "a1", "b"]
        (report,) = reports
        assert sum(c.changed for c in report.chains) == 3
        created = cast(list[kbt.Track], board.created)
        assert unary_union([segment(t) for t in created]).covers(
            LineString([(15, 0), (20, 0)])
        )

    @pytest.mark.usefixtures("coarse_optimizer")
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

        run()

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
    def test_selected_stub_ending_in_the_chains_copper_joins_it_at_a_junction(
        self, connect: Connector, tip: float
    ) -> None:
        chain = l_chain()
        stub = kipy_track("stub", "SIG", (2, -3), (2, tip), width=WIDTH)
        tracks = [*chain, stub]
        board = connect.to(CommitBoard(selection=list(tracks), tracks=list(tracks)))

        run()

        assert sorted(board.deleted) == ["a", "b", "stub"]
        degree = Counter(
            (p.x, p.y)
            for t in cast(list[kbt.Track], board.created)
            for p in (t.start, t.end)
        )
        assert sorted(p for p, n in degree.items() if n == 1) == [
            (0, 0),
            (nm(2), nm(-3)),
            (nm(4), nm(4)),
        ]
        assert list(degree.values()).count(3) == 1

    @pytest.mark.usefixtures("coarse_optimizer")
    @pytest.mark.parametrize("tip", [-0.25, -0.1])
    def test_selected_stub_touching_the_chain_is_no_clearance_violation(
        self, connect: Connector, tip: float
    ) -> None:
        tracks = [*l_chain(), kipy_track("stub", "SIG", (2, -3), (2, tip), width=WIDTH)]
        connect.to(CommitBoard(selection=list(tracks), tracks=list(tracks)))
        options = cli.smooth_options(cli.parse_args(["smooth", "-n"]))

        report = pipeline.run_smooth(options)

        assert len(report.chains) == 3
        for c in report.chains:
            assert c.after.min_clearance >= -NM_ROUNDING
            assert c.frozen == ()

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_dry_run_prints_the_outcome_and_commits_nothing(
        self, connect: Connector, capsys: pytest.CaptureFixture[str]
    ) -> None:
        board = connect.to(l_board())

        run("--dry-run")

        assert board.calls == []
        assert "Optimizer: converged" in capsys.readouterr().out

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
            seen["settings"] = args[2]
            raise SystemExit(0)

        monkeypatch.setattr(pipeline, "build_obstacles", build_spy)
        monkeypatch.setattr(pipeline, "optimize_chains", optimize_spy)

        with pytest.raises(SystemExit):
            run(
                "--repulsion=2",
                "--range=0.4",
                "--edge-clearance=0.5",
                "--hole-clearance=0.1",
                "--time-limit=30",
            )

        params = cast(EnergyParams, seen["params"])
        assert (params.repulsion, params.falloff_range) == (2.0, 0.4)
        assert (seen["edge_clearance"], seen["hole_clearance"]) == (0.5, 0.1)
        assert cast(OptimizerSettings, seen["settings"]).time_limit == 30.0

    @pytest.mark.usefixtures("coarse_optimizer")
    @pytest.mark.parametrize(
        ("args", "expected"), [((), 0.01), (("--chord-tolerance=0.025",), 0.025)]
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
            run("--repulsion=-1")

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
    def test_iteration_cap_warns_and_still_commits(
        self,
        connect: Connector,
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
        assert "push_commit" in board.calls

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_time_limit_commits_the_best_state_after_the_burst_past_it(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        log: list[str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        capped = connect.to(l_board())
        monkeypatch.setattr(
            pipeline, "optimize_chains", at(replace(COARSE, max_iterations=1))
        )
        run()
        monkeypatch.setattr(pipeline, "optimize_chains", at(COARSE))
        monkeypatch.setattr(pipeline, "monotonic", Ticking())
        capsys.readouterr()
        board = connect.to(l_board())

        run("--time-limit=0.5")

        out = capsys.readouterr().out
        assert re.search(
            r"^Optimizer: time limit reached after 1 iterations in \d+\.\d s,",
            out,
            re.MULTILINE,
        )
        assert any(line.startswith("WARNING") and "time limit" in line for line in log)
        assert "push_commit" in board.calls
        # The one burst of a single chain is the first outer iteration.
        assert [segment(t) for t in cast(list[kbt.Track], board.created)] == [
            segment(t) for t in cast(list[kbt.Track], capped.created)
        ]

    def test_reports_the_optimize_stages_wall_time(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connect.to(l_board())
        options = cli.smooth_options(cli.parse_args(["smooth", "-n"]))
        clock = Ticking(step=0.0)
        monkeypatch.setattr(pipeline, "monotonic", clock)

        def optimizing(*args: object, **kwargs: object) -> RunResult:
            clock.now += 10.0
            return at(COARSE)(*args, **kwargs)

        monkeypatch.setattr(pipeline, "optimize_chains", optimizing)
        optimized: list[pipeline.OptimizerReport] = []

        report = pipeline.run_smooth(options, optimized.append)

        assert report.elapsed == optimized[0].elapsed == 10.0

    @pytest.mark.parametrize(
        "option",
        [
            "--time-limit=0",
            "--time-limit=-1",
            "--time-limit=nan",
            "--time-limit=inf",
            "--stall-tolerance=nan",
            "--settled-change=0",
            "--stall-tolerance=-0.001",
            "--stall-window=0",
            "--stall-window=1.5",
            "--max-iterations=0",
            "--burst-iterations=-3",
            "--lbfgs-ftol=0",
            "--lbfgs-gtol=-1e-6",
        ],
    )
    def test_stop_rules_must_be_positive(
        self, option: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as raised:
            run(option)

        assert raised.value.code == 2
        assert f"argument {option.split('=')[0]}:" in capsys.readouterr().err

    def test_stop_rules_reach_the_settings(self) -> None:
        options = cli.smooth_options(
            cli.parse_args(
                [
                    "smooth",
                    "--settled-change=0.005",
                    "--stall-tolerance=0.01",
                    "--stall-window=5",
                    "--max-iterations=50",
                    "--burst-iterations=50",
                    "--lbfgs-ftol=1e-9",
                    "--lbfgs-gtol=1e-6",
                    "--time-limit=60",
                ]
            )
        )

        s = options.settings
        assert s.settled_change == 0.005
        assert (s.stall_tolerance, s.stall_window) == (0.01, 5)
        assert (s.max_iterations, s.burst_maxiter, s.time_limit) == (50, 50, 60.0)
        assert (s.lbfgs_ftol, s.lbfgs_gtol) == (1e-9, 1e-6)

    def test_no_move_bounds_reaches_the_settings(self) -> None:
        options = cli.smooth_options(cli.parse_args(["smooth", "--no-move-bounds"]))

        assert options.settings == replace(DEFAULT_SETTINGS, move_bounds=False)

    def test_stop_rules_default_to_the_optimizers(self) -> None:
        options = cli.smooth_options(cli.parse_args(["smooth"]))

        assert options.settings == DEFAULT_SETTINGS


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

    def test_selection_of_only_locked_tracks(
        self, connect: Connector, log: list[str]
    ) -> None:
        tracks = [
            kipy_track("a", "SIG", (0, 0), (4, 0), locked=True),
            kipy_track("b", "SIG", (4, 0), (4, 4), locked=True),
        ]
        board = connect.to(CommitBoard(selection=list(tracks), tracks=list(tracks)))

        assert "no tracks that can be modified" in exit_message(log)
        assert board.calls == []

    # A selected via that may move makes the run extract the chains once
    # without pins first, to find the copper the via carries.
    @pytest.mark.parametrize("with_via", [False, True], ids=["tracks", "via"])
    def test_selection_that_is_no_chain(
        self, connect: Connector, log: list[str], with_via: bool
    ) -> None:
        tracks = [
            kipy_track("a", "SIG", (0, 0), (9, 0)),
            kipy_track("b", "SIG", (9, 0), (9, 3)),
            kipy_track("c", "SIG", (9, 3), (0, 0)),
        ]
        vias = [kipy_via("v", "SIG", 9, 0)] if with_via else []
        board = CommitBoard(
            selection=[*tracks, *vias],
            tracks=list(tracks),
            vias=vias,
            grid_origin=GRID_ORIGIN,
        )
        connect.to(board)

        message = exit_message(log)

        assert "selected tracks form a closed loop through (-1.000, -2.000) mm" in (
            message
        )
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

        message = exit_message(log)

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
    def test_interrupt_during_the_push_may_have_landed(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = connect.to(l_board(push_error=KeyboardInterrupt()))

        message = exit_message(log)

        assert "Interrupted" in message and "may have landed" in message
        assert "drop_commit" not in board.calls

    def test_interrupt_before_the_push_commits_nothing(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch, log: list[str]
    ) -> None:
        board = connect.to(l_board())

        def interrupted(*args: object, **kwargs: object) -> NoReturn:
            raise KeyboardInterrupt

        monkeypatch.setattr(pipeline, "optimize_chains", interrupted)

        assert exit_message(log) == "ERROR Interrupted: nothing committed\n"
        assert board.calls == []

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


@pytest.mark.usefixtures("coarse_optimizer")
class TestFailureAfterOptimizing:
    """The optimizer's outcome and warnings still reach the user before the
    error."""

    def assert_reported_before(
        self, error: str, log: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = capsys.readouterr().out
        assert re.search(r"^Optimizer: ", out, re.MULTILINE)
        (warning,) = [i for i, line in enumerate(log) if "without converging" in line]
        assert warning < log.index(error)

    def test_lost_connection_on_push(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        log: list[str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        connect.to(l_board(push_error=KiCadConnectionError("timed out")))
        stopped_at_cap(monkeypatch)

        error = exit_message(log)

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


@pytest.mark.usefixtures("coarse_optimizer")
class TestSeveralChains:
    def test_commits_both_in_one_commit_and_reports_each(
        self,
        connect: Connector,
        capsys: pytest.CaptureFixture[str],
        reports: list[pipeline.RunReport],
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
        (report,) = reports
        assert [(c.net, c.layer, c.width, c.tracks) for c in report.chains] == [
            ("SIG", "F.Cu", WIDTH, 2),
            ("SIG2", "F.Cu", WIDTH, 2),
        ]
        assert all(c.frozen == () for c in report.chains)
        assert capsys.readouterr().out.count("Optimizer: ") == 1

    def test_dry_run_reports_both_and_commits_nothing(
        self, connect: Connector, reports: list[pipeline.RunReport]
    ) -> None:
        board = connect.to(two_l_board())

        run("--dry-run")

        (report,) = reports
        assert [c.net for c in report.chains] == ["SIG", "SIG2"]
        assert board.calls == []

    def test_unchanged_chain_is_not_committed(
        self,
        connect: Connector,
        reports: list[pipeline.RunReport],
    ) -> None:
        board = connect.to(two_l_board(inner=far_straight()))

        run()

        assert sorted(board.deleted) == list(OUTER)
        assert {t.net.name for t in cast(list[kbt.Track], board.created)} == {"SIG"}
        (report,) = reports
        assert [c.changed for c in report.chains] == [True, False]

    def test_chain_that_only_gains_straight_vertices_stays_unchanged(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        reports: list[pipeline.RunReport],
    ) -> None:
        board = connect.to(two_l_board(inner=far_straight()))
        real: Callable[..., list[FloatArray]] = pipeline.simplify_chains

        def with_midpoints(*args: object, **kwargs: object) -> list[FloatArray]:
            outer, straight = real(*args, **kwargs)
            midpoints = (straight[:-1] + straight[1:]) / 2
            split = np.insert(straight, range(1, len(straight)), midpoints, axis=0)
            return [outer, split]

        monkeypatch.setattr(pipeline, "simplify_chains", with_midpoints)

        run()

        assert sorted(board.deleted) == list(OUTER)
        (report,) = reports
        straight = report.chains[1]
        assert not straight.changed
        assert (straight.before.segment_count, straight.after.segment_count) == (1, 1)

    def test_all_unchanged_commits_nothing(
        self, connect: Connector, log: list[str]
    ) -> None:
        tracks = [
            kipy_track("a", "SIG", (0, 0), (10, 0), width=WIDTH),
            *far_straight(),
        ]
        board = connect.to(CommitBoard(selection=list(tracks), tracks=list(tracks)))

        run()

        assert board.calls == []
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

    # Resampling vertices held deeper than δ inside each other's clearance
    # let them creep out and kept the run going: 200 iterations, the cap.
    # Measured: 12.
    SETTLED_PRESSED = 20

    def test_frozen_warning_names_the_other_chain(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch, log: list[str]
    ) -> None:
        # 0.3 mm from the L's corner, inside clearance plus both half widths.
        across = [kipy_track("x", "SIG2", (4.3, -1), (4.3, 1), width=WIDTH)]
        connect.to(off_grid(two_l_board(inner=across)))
        results: list[RunResult] = []
        optimize = at(COARSE)

        def kept(*args: object, **kwargs: object) -> RunResult:
            results.append(optimize(*args, **kwargs))
            return results[-1]

        monkeypatch.setattr(pipeline, "optimize_chains", kept)

        run("--dry-run")

        # The chains pressed into each other's clearance settle: what lies
        # deeper than δ inside stays put, resampled or not.
        (result,) = results
        assert result.termination is Termination.CONVERGED
        assert result.iterations <= self.SETTLED_PRESSED
        # (4, 0) and (4.3, -1) relative to the grid origin.
        frozen = [line for line in log if "Left the vertex at (3.000, -2.000)" in line]
        assert len(frozen) == 1
        assert frozen[0].startswith("WARNING")
        assert "selected chain SIG2 on F.Cu from (3.300, -3.000)" in frozen[0]

    def test_run_function_reports_each_chain(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        board = connect.to(two_l_board())
        options = cli.smooth_options(cli.parse_args(["smooth", "-n"]))
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
            assert c.frozen == ()
        assert report.abnormal == ()
        assert report.termination in (Termination.CONVERGED, Termination.ITERATION_CAP)
        assert report.iterations > 0
        assert (report.committed, report.refusal) == (False, None)
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

        run()

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

        run("--dry-run", f"--plot={figure}")

        assert drawn == [2]
        assert figure.stat().st_size > 0


def with_abnormal_bursts(
    monkeypatch: pytest.MonkeyPatch, abnormal: Sequence[tuple[str, int]]
) -> None:
    """Make the optimizer report `abnormal` bursts per problem."""
    real: Callable[..., RunResult] = pipeline.optimize_chains

    def with_them(*args: object, **kwargs: object) -> RunResult:
        return replace(real(*args, **kwargs), abnormal=tuple(abnormal))

    monkeypatch.setattr(pipeline, "optimize_chains", with_them)


def stopped_at_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the optimizer report that it stopped at its iteration cap."""
    real: Callable[..., RunResult] = pipeline.optimize_chains

    def capped(*args: object, **kwargs: object) -> RunResult:
        return replace(real(*args, **kwargs), termination=Termination.ITERATION_CAP)

    monkeypatch.setattr(pipeline, "optimize_chains", capped)


FRONT = "the chains on F.Cu"
"""How the optimizer names the problem of F.Cu's chains."""


@pytest.mark.usefixtures("coarse_optimizer")
def test_abnormal_bursts_are_logged_only_at_debug(
    connect: Connector,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    log: list[str],
) -> None:
    connect.to(l_board())
    with_abnormal_bursts(monkeypatch, [(FRONT, 3)])

    run("--dry-run")

    assert "abnormal" not in capsys.readouterr().out
    (line,) = [ln for ln in log if "abnormally" in ln]
    assert line.startswith("DEBUG") and f"3 bursts of {FRONT}" in line


@pytest.mark.usefixtures("coarse_optimizer")
def test_prints_only_the_outcome(
    connect: Connector, capsys: pytest.CaptureFixture[str]
) -> None:
    connect.to(l_board())

    run("--dry-run")

    (outcome,) = capsys.readouterr().out.splitlines()
    assert re.fullmatch(
        r"Optimizer: converged after \d+ iterations in \d+\.\d s, "
        r"\d+ rejected edge moves",
        outcome,
    )


VIA = (4.0, 4.0)
"""Where `l_chain` ends: the via that takes SIG down to B.Cu."""
BOARD_WIDE_READS = (
    "get_selection",
    "get_footprints",
    "get_tracks",
    "get_pads",
    "get_vias",
    "get_shapes",
    "get_text",
    "get_zones",
    "get_netclass_for_nets",
)


def back_l(layer: BL.ValueType = BL.BL_B_Cu) -> list[kbt.Track]:
    """An L on `layer` leaving the via at `VIA`."""
    return [
        kipy_track("e", "SIG", VIA, (8, 4), width=WIDTH, layer=layer),
        kipy_track("f", "SIG", (8, 4), (8, 8), width=WIDTH, layer=layer),
    ]


def via_route_board(
    pad: tuple[float, float] = PAD,
    pad_layers: Sequence[int] = (BL.BL_F_Cu, BL.BL_B_Cu),
    back_layer: BL.ValueType = BL.BL_B_Cu,
    unselected: Sequence[kbt.Track] = (),
) -> CommitBoard:
    """SIG along `l_chain` on F.Cu, down an unselected via, on along `back_l`
    on `back_layer`, with a GND pad at `pad`, an unselected GND track on B.Cu
    and the `unselected` tracks."""
    route = [*l_chain(), *back_l(back_layer)]
    gnd = kipy_track("g", "GND", (0, 10), (8, 10), width=WIDTH, layer=BL.BL_B_Cu)
    return CommitBoard(
        selection=list(route),
        tracks=[*route, gnd, *unselected],
        vias=[kipy_via("v", "SIG", *VIA)],
        pads=[kipy_pad("pad", "GND", *pad, layers=pad_layers)],
    )


def created_on(board: CommitBoard, layer: BL.ValueType) -> list[kbt.Track]:
    return [t for t in board.created if isinstance(t, kbt.Track) and t.layer == layer]


def ends(tracks: Sequence[kbt.Track]) -> set[tuple[int, int]]:
    return {(p.x, p.y) for t in tracks for p in (t.start, t.end)}


def layout(board: CommitBoard) -> set[tuple[int, int, int, int, int]]:
    return {
        (t.layer, t.start.x, t.start.y, t.end.x, t.end.y)
        for t in cast(list[kbt.Track], board.created)
    }


@pytest.mark.usefixtures("coarse_optimizer")
class TestSeveralLayers:
    def test_route_through_a_via_smooths_both_sides_in_one_commit(
        self, connect: Connector
    ) -> None:
        board = connect.to(via_route_board())

        run()

        assert board.calls == ["begin_commit", "delete", "create", "push_commit"]
        assert sorted(board.deleted) == ["a", "b", "e", "f"]
        front, back = created_on(board, BL.BL_F_Cu), created_on(board, BL.BL_B_Cu)
        assert len(front) + len(back) == len(board.created)
        via = (nm(VIA[0]), nm(VIA[1]))
        assert {(0, 0), via} <= ends(front)
        assert {via, (nm(8), nm(8))} <= ends(back)

    def test_inner_layer_route_is_committed_on_it(self, connect: Connector) -> None:
        board = connect.to(via_route_board(back_layer=BL.BL_In1_Cu))

        run()

        assert sorted(board.deleted) == ["a", "b", "e", "f"]
        front, inner = created_on(board, BL.BL_F_Cu), created_on(board, BL.BL_In1_Cu)
        assert len(front) + len(inner) == len(board.created)
        via = (nm(VIA[0]), nm(VIA[1]))
        assert {via, (nm(8), nm(8))} <= ends(inner)

    def test_each_chain_keeps_touches_of_its_own_layers_copper(
        self, connect: Connector
    ) -> None:
        # Mid-leg on each side, where the smoothed L would otherwise leave.
        front_touch, back_touch = STUB_TOUCH, (8.0, 6.0)
        stubs = [
            kipy_track("fs", "SIG", (2, -3), front_touch, width=WIDTH),
            kipy_track("bs", "SIG", (11, 6), back_touch, width=WIDTH, layer=BL.BL_B_Cu),
        ]
        board = connect.to(via_route_board(unselected=stubs))

        run()

        assert sorted(board.deleted) == ["a", "b", "e", "f"]
        assert (nm(front_touch[0]), nm(front_touch[1])) in ends(
            created_on(board, BL.BL_F_Cu)
        )
        assert (nm(back_touch[0]), nm(back_touch[1])) in ends(
            created_on(board, BL.BL_B_Cu)
        )

    def test_each_layer_avoids_its_own_obstacles(self, connect: Connector) -> None:
        # On F.Cu only, on the chord the front L would otherwise be pulled to.
        on_chord = (2.0, 2.0)
        board = connect.to(via_route_board(on_chord, pad_layers=(BL.BL_F_Cu,)))

        run()

        x, y = on_chord
        h = PAD_HALF_SIZE
        pad = box(x - h, y - h, x + h, y + h)
        front = created_on(board, BL.BL_F_Cu)
        assert front
        assert all(
            segment(t).distance(pad) >= CLEARANCE + WIDTH / 2 - NM_ROUNDING
            for t in front
        )

    def test_other_net_via_on_other_layers_is_no_obstacle(
        self, connect: Connector
    ) -> None:
        back = [
            kipy_track("e", "SIG", (0, 0), (4, 0), width=WIDTH, layer=BL.BL_B_Cu),
            kipy_track("f", "SIG", (4, 0), (4, 4), width=WIDTH, layer=BL.BL_B_Cu),
        ]
        blind = kipy_via(
            "bv",
            "OTHER",
            2,
            0,
            via_type=ViaType.VT_BLIND,
            span=(BL.BL_F_Cu, BL.BL_In1_Cu),
        )
        board = connect.to(
            CommitBoard(selection=list(back), tracks=list(back), vias=[blind])
        )

        run()

        assert layout(board) == {(BL.BL_B_Cu, 0, 0, nm(4), nm(4))}

    def test_selected_fixed_via_leaves_the_run_as_without_it(
        self, connect: Connector, log: list[str]
    ) -> None:
        def locked_via_board() -> CommitBoard:
            board = via_route_board()
            board.vias = [kipy_via("v", "SIG", *VIA, locked=True)]
            return board

        without = connect.to(locked_via_board())
        run()
        board = locked_via_board()
        board.selection.append(board.vias[0])
        connect.to(board)

        run()

        assert layout(board) == layout(without)
        assert board.reads == without.reads
        assert board.deleted == without.deleted
        assert len([line for line in log if "stays fixed" in line]) == 1

    @pytest.mark.parametrize(
        "fixed",
        [
            partial(kipy_via, locked=True),
            partial(kipy_via, via_type=ViaType.VT_BLIND),
        ],
        ids=["locked", "blind"],
    )
    def test_selected_vias_that_cannot_move_cost_no_reads(
        self, connect: Connector, fixed: Callable[[str, str, float, float], kbt.Via]
    ) -> None:
        # A through-hole pad of the via's net lists layers no run reads.
        pth = kipy_pad("pth", "SIG", 0, 8, layers=list(iter_copper_layers()))
        board = via_route_board()
        board.pads.append(pth)
        board.vias = [fixed("v", "SIG", *VIA)]
        board.selection.append(board.vias[0])
        connect.to(board)

        run("--dry-run")

        assert board.reads["get_enabled_layers"] == 0
        assert board.reads["get_pad_shapes_as_polygons"] == 2

    def test_selected_via_that_must_stay_is_named(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = off_grid(via_route_board())
        board.vias = [kipy_via("v", "SIG", *VIA, locked=True)]
        board.selection.append(board.vias[0])
        connect.to(board)

        run()

        assert "push_commit" in board.calls
        (warning,) = [line for line in log if "stays fixed" in line]
        assert warning.startswith("DEBUG Via v at (3.000, 2.000) ")
        assert "locked" in warning

    def test_selected_via_with_an_unselected_track_is_named(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = via_route_board()
        third = kipy_track("g2", "SIG", VIA, (4, 9), width=WIDTH, layer=BL.BL_In1_Cu)
        board.tracks.append(third)
        board.selection.append(board.vias[0])
        connect.to(board)

        run()

        (warning,) = [line for line in log if "stays fixed" in line]
        assert warning.startswith("DEBUG Via v ") and "track g2 on In1.Cu" in warning

    def test_selected_via_held_by_other_copper_leaves_the_run_as_without_it(
        self, connect: Connector, log: list[str]
    ) -> None:
        locked = kipy_track("k", "OTHER", (0, 7), (3, 7), width=WIDTH, locked=True)

        def held_via_board() -> CommitBoard:
            """The via may move by kind, but g2 on In1.Cu holds it."""
            board = via_route_board()
            third = kipy_track(
                "g2", "SIG", VIA, (4, 9), width=WIDTH, layer=BL.BL_In1_Cu
            )
            board.tracks += [third, locked]
            board.selection.append(locked)
            return board

        without = connect.to(held_via_board())
        run()
        board = held_via_board()
        board.selection.append(board.vias[0])
        connect.to(board)
        log.clear()

        run()

        assert layout(board) == layout(without)
        assert board.deleted == without.deleted
        assert len([line for line in log if "stays fixed" in line]) == 1
        assert len([line for line in log if "Leaving locked" in line]) == 1

    def test_reads_only_the_selected_layers_without_a_moving_via(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = via_route_board()
        board.enabled_layers = [BL.BL_F_Cu, BL.BL_In1_Cu, BL.BL_B_Cu]
        connect.to(board)

        run("--dry-run")

        read = [line.split()[3].rstrip(":") for line in log if "Snapshot of" in line]
        assert read == ["B.Cu", "F.Cu"]
        assert board.reads["get_enabled_layers"] == 0

    def test_reads_each_board_wide_kind_once(self, connect: Connector) -> None:
        board = connect.to(via_route_board())

        run()

        assert {kind: board.reads[kind] for kind in BOARD_WIDE_READS} == dict.fromkeys(
            BOARD_WIDE_READS, 1
        )
        assert board.reads["get_pad_shapes_as_polygons"] == 2

    def test_two_front_chains_of_one_net_need_no_clearance_to_each_other(
        self, connect: Connector
    ) -> None:
        # Both ends of the two front chains lie closer than width plus
        # clearance, which only chains of one net may.
        gap = WIDTH + CLEARANCE / 2
        inner_via = (VIA[0] - gap, VIA[1])
        tracks = [
            *l_chain(),
            kipy_track("m", "SIG", VIA, inner_via, width=WIDTH, layer=BL.BL_B_Cu),
            kipy_track("c", "SIG", (0, gap), (4 - gap, gap), width=WIDTH),
            kipy_track("d", "SIG", (4 - gap, gap), inner_via, width=WIDTH),
        ]
        connect.to(
            CommitBoard(
                selection=list(tracks),
                tracks=list(tracks),
                vias=[kipy_via("v", "SIG", *VIA), kipy_via("w", "SIG", *inner_via)],
            )
        )
        options = cli.smooth_options(cli.parse_args(["smooth", "-n"]))

        report = pipeline.run_smooth(options)

        assert [(c.layer, c.tracks, c.changed) for c in report.chains] == [
            ("B.Cu", 1, False),
            ("F.Cu", 2, True),
            ("F.Cu", 2, True),
        ]
        for c in report.chains[1:]:
            assert c.after.min_clearance >= -NM_ROUNDING
            assert c.frozen == ()

    def test_chains_crossing_on_two_layers_smooth_as_if_alone(
        self, connect: Connector
    ) -> None:
        front = l_chain()
        back = [
            kipy_track("e", "SIG2", (2, -2), (2, 2), width=WIDTH, layer=BL.BL_B_Cu),
            kipy_track("f", "SIG2", (2, 2), (6, 2), width=WIDTH, layer=BL.BL_B_Cu),
        ]

        def smoothed(tracks: list[kbt.Track]) -> set[tuple[int, int, int, int, int]]:
            board = connect.to(CommitBoard(selection=list(tracks), tracks=list(tracks)))
            run()
            return layout(board)

        together = smoothed([*front, *back])

        assert together == smoothed(front) | smoothed(back)

    def test_plot_titles_name_each_chains_layer(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from kibends import plot

        connect.to(via_route_board())
        titles: list[str] = []

        def spy(path: Path, panels: Sequence[plot.Panel]) -> None:
            titles.extend(p.title for p in panels)

        monkeypatch.setattr(plot, "render", spy)

        run("--dry-run", f"--plot={tmp_path / 'chains.png'}")

        assert [t.split(":")[0] for t in titles] == ["SIG on B.Cu", "SIG on F.Cu"]


def detour_back() -> list[kbt.Track]:
    """From the via at `VIA` back up to (8, 0) on B.Cu: with `l_chain`, a
    route whose straight line runs 4 mm from the via."""
    return [
        kipy_track("e", "SIG", VIA, (8, 4), width=WIDTH, layer=BL.BL_B_Cu),
        kipy_track("f", "SIG", (8, 4), (8, 0), width=WIDTH, layer=BL.BL_B_Cu),
    ]


def moving_via_board(*, refuse_create_vias: bool = False) -> CommitBoard:
    """`l_chain` down the selected via `v` at `VIA` and on along
    `detour_back`, in free space."""
    route = [*l_chain(), *detour_back()]
    via = kipy_via("v", "SIG", *VIA)
    return CommitBoard(
        selection=[*route, via],
        tracks=list(route),
        vias=[via],
        refuse_create_vias=refuse_create_vias,
    )


def created_vias(board: CommitBoard) -> list[kbt.Via]:
    return [i for i in board.created if isinstance(i, kbt.Via)]


@pytest.mark.usefixtures("coarse_optimizer")
class TestMovingVias:
    def test_moves_the_via_with_its_chains_in_one_commit(
        self, connect: Connector
    ) -> None:
        board = connect.to(moving_via_board())

        run()

        assert board.calls == ["begin_commit", "delete", "create", "push_commit"]
        assert sorted(board.deleted) == ["a", "b", "e", "f", "v"]
        (via,) = created_vias(board)
        moved = (via.position.x, via.position.y)
        assert moved[1] < nm(1)
        assert moved in ends(created_on(board, BL.BL_F_Cu))
        assert moved in ends(created_on(board, BL.BL_B_Cu))

    @pytest.mark.parametrize("unfilled", ["pwr pour", "pour keepout"])
    def test_a_via_in_a_zone_of_its_net_moves_within_its_fill_and_commits(
        self, connect: Connector, log: list[str], unfilled: str
    ) -> None:
        # Without the zone, the via ends below y = 1, in the part of the
        # outline that the PWR pour or the keepout leaves unfilled.
        fill_top = 2.8
        board = moving_via_board()
        unfilled_area = kipy_rect(-2, -3, 10, 2.5)
        if unfilled == "pwr pour":
            other = kipy_filled_zone("pwr", "PWR", {BL.BL_In1_Cu: unfilled_area})
        else:
            other = kipy_zone(unfilled_area, [BL.BL_In1_Cu], keepout_tracks=False)
            other.proto.rule_area_settings.keepout_copper = True
        board.zones = [
            kipy_filled_zone(
                "plane",
                "SIG",
                {BL.BL_In1_Cu: kipy_rect(0, fill_top, 10, 10)},
                outline=kipy_rect(-2, -3, 10, 10),
            ),
            other,
        ]
        connect.to(board)

        run()

        assert "stays fixed" not in "".join(log)
        assert board.calls == ["begin_commit", "delete", "create", "push_commit"]
        (via,) = created_vias(board)
        y = via.position.y / 1e6
        assert fill_top + KICAD_MAX_ERROR - NM_ROUNDING <= y < VIA[1]

    @pytest.mark.parametrize("unfillable", ["pwr pour", "pour keepout"])
    def test_a_stale_fill_does_not_lead_the_via_into_what_the_refill_leaves_out(
        self, connect: Connector, log: list[str], unfillable: str
    ) -> None:
        # The plane's fill predates a higher-priority PWR zone or a pour
        # keepout over y < 2.5, drawn since the last refill: the fill covers
        # the whole outline, and the via would end below y = 1 within it.
        board = moving_via_board()
        area = kipy_rect(-2, -3, 10, 2.5)
        stale = {BL.BL_In1_Cu: kipy_rect(-2, -3, 10, 10)}
        plane = kipy_filled_zone("plane", "SIG", stale)
        plane.clearance = nm(0.2)
        if unfillable == "pwr pour":
            other = kipy_zone(area, [BL.BL_In1_Cu], rule_area=False, net="PWR")
            other.priority = 1
            other.clearance = nm(0.3)
            refill_top = 2.5 + 0.3
        else:
            other = kipy_zone(area, [BL.BL_In1_Cu], keepout_tracks=False)
            other.proto.rule_area_settings.keepout_copper = True
            refill_top = 2.5
        board.zones = [plane, other]
        connect.to(board)

        run()

        assert "stays fixed" not in "".join(log)
        (via,) = created_vias(board)
        y = via.position.y / 1e6
        assert refill_top + KICAD_MAX_ERROR - NM_ROUNDING <= y < VIA[1]

    def test_a_via_at_the_edge_of_a_fill_stays_naming_the_zone(
        self, connect: Connector, log: list[str]
    ) -> None:
        # The edge fill ends 0.1 mm below the via's centre, short of the
        # 0.13 mm that half its minimum width and the arc error ask for; the
        # plane holds it deep inside.
        board = moving_via_board()
        board.zones = [
            kipy_filled_zone("plane", "SIG", {BL.BL_In2_Cu: kipy_rect(0, 0, 10, 10)}),
            kipy_filled_zone(
                "edge",
                "SIG",
                {BL.BL_In1_Cu: kipy_rect(0, 3.9, 10, 10)},
                min_width=0.25,
            ),
        ]
        connect.to(board)

        run()

        assert "push_commit" in board.calls
        assert created_vias(board) == [] and "v" not in board.deleted
        (warning,) = [line for line in log if "stays fixed" in line]
        assert warning.rstrip().endswith(
            "Via v at (4.000, 4.000) stays fixed: "
            "it sits at the edge of zone edge on In1.Cu"
        )
        assert "Left via" not in "".join(log)

    def test_copper_touching_only_a_carried_end_holds_the_via_and_the_end(
        self, connect: Connector, log: list[str]
    ) -> None:
        # The wide end's copper reaches y = 4.5 and the pad's edge 4.4; the
        # via's ring reaches only 4.3.
        wide = [
            kipy_track("a", "SIG", (0, 0), (4, 0), width=1.0),
            kipy_track("b", "SIG", (4, 0), VIA, width=1.0),
        ]
        board = moving_via_board()
        board.tracks = [*wide, *detour_back()]
        board.selection = [*board.tracks, *board.vias]
        board.pads = [kipy_pad("p", "SIG", 4.0, 4.65, layers=(BL.BL_F_Cu,))]
        connect.to(board)

        run()

        assert "push_commit" in board.calls
        assert created_vias(board) == [] and "v" not in board.deleted
        (warning,) = [line for line in log if "stays fixed" in line]
        assert "pad p" in warning
        assert (nm(VIA[0]), nm(VIA[1])) in ends(created_on(board, BL.BL_F_Cu))

    @pytest.mark.parametrize("in_pad", [False, True])
    def test_same_net_pad_on_its_path_stops_the_via_unless_allowed(
        self, connect: Connector, in_pad: bool
    ) -> None:
        # The pad sits where the via ends up without it.
        free = connect.to(moving_via_board())
        run()
        (landed,) = created_vias(free)
        x, y = landed.position.x / 1e6, landed.position.y / 1e6
        board = moving_via_board()
        board.pads = [kipy_pad("p", "SIG", x, y, layers=(BL.BL_F_Cu,))]
        connect.to(board)

        run(*(["--allow-via-in-pad"] if in_pad else []))

        (via,) = created_vias(board)
        h = PAD_HALF_SIZE
        pad = box(x - h, y - h, x + h, y + h)
        centre = Point(via.position.x / 1e6, via.position.y / 1e6)
        if in_pad:
            assert pad.contains(centre)
        else:
            assert pad.distance(centre) >= via.diameter / 2e6 - NM_ROUNDING

    def test_chains_meeting_at_a_moved_via_on_one_layer_need_no_trim_there(
        self, connect: Connector
    ) -> None:
        # Two widths, so that the two sides are two chains of one layer.
        wide = WIDTH + 0.05
        route = [
            *l_chain(),
            kipy_track("e", "SIG", VIA, (8, 4), width=wide),
            kipy_track("f", "SIG", (8, 4), (8, 0), width=wide),
        ]
        via = kipy_via("v", "SIG", *VIA)
        connect.to(CommitBoard(selection=[*route, via], tracks=list(route), vias=[via]))
        options = cli.smooth_options(cli.parse_args(["smooth", "-n"]))

        report = pipeline.run_smooth(options)

        assert report.vias[0].end[1] < 1.0
        for chain in report.chains:
            assert chain.after.segment_count == 1
            assert chain.after.min_clearance >= -NM_ROUNDING

    def test_a_frozen_via_stays_while_its_chains_change(
        self,
        connect: Connector,
        log: list[str],
        reports: list[pipeline.RunReport],
    ) -> None:
        # The pad's edge is 0.1 mm from the via's ring, inside its clearance,
        # and clear of both chains.
        board = off_grid(moving_via_board())
        board.pads = [kipy_pad("p", "GND", 4.0, 4.65, layers=(BL.BL_F_Cu,))]
        connect.to(board)

        run()

        assert "push_commit" in board.calls
        assert sorted(board.deleted) == ["a", "b", "e", "f"]
        assert created_vias(board) == []
        assert "DEBUG Left via v at (3.000, 2.000) in place" in "".join(log)
        (report,) = reports
        (via,) = report.vias
        assert (via.id, via.start, via.frozen, via.moved) == ("v", VIA, True, False)

    def test_reads_each_board_wide_kind_once(self, connect: Connector) -> None:
        board = connect.to(moving_via_board())

        run()

        assert {kind: board.reads[kind] for kind in BOARD_WIDE_READS} == dict.fromkeys(
            BOARD_WIDE_READS, 1
        )
        assert board.reads["get_enabled_layers"] == 1
        assert board.reads["get_items_by_id"] == 1

    def test_reads_every_copper_layer_for_the_via(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = moving_via_board()
        board.enabled_layers = [BL.BL_F_Cu, BL.BL_In1_Cu, BL.BL_B_Cu, BL.BL_F_SilkS]
        connect.to(board)

        run("--dry-run")

        read = [line.split()[3].rstrip(":") for line in log if "Snapshot of" in line]
        assert read == ["B.Cu", "F.Cu", "In1.Cu"]
        assert "INFO Optimizing 2 chains and 1 moving via\n" in log

    def test_reports_the_move(self, connect: Connector) -> None:
        board = connect.to(moving_via_board())
        options = cli.smooth_options(cli.parse_args(["smooth", "-n"]))

        report = pipeline.run_smooth(options)

        (via,) = report.vias
        assert (via.id, via.start, via.moved) == ("v", VIA, True)
        assert via.end[1] < 1.0
        assert board.calls == []

    def test_a_refused_via_create_drops_the_whole_commit(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = connect.to(moving_via_board(refuse_create_vias=True))

        error = exit_message(log)

        assert re.search(r"KiCad created 2 of 3 items", error)
        assert board.calls == ["begin_commit", "delete", "create", "drop_commit"]

    def test_a_via_changed_in_kicad_refuses_the_commit(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch, log: list[str]
    ) -> None:
        board = connect.to(moving_via_board())
        optimize: Callable[..., RunResult] = pipeline.optimize_chains

        def edited_meanwhile(*args: object, **kwargs: object) -> RunResult:
            board.vias = [kipy_via("v", "SIG", 4.0, 4.5)]
            return optimize(*args, **kwargs)

        monkeypatch.setattr(pipeline, "optimize_chains", edited_meanwhile)

        error = exit_message(log)

        assert "changed in KiCad" in error and "v" in error
        assert board.calls == []

    def test_merged_tracks_keep_clear_of_where_a_via_moved_to(
        self, connect: Connector
    ) -> None:
        # The via presses an inner-layer chain of another net down into a
        # bow, which a chord this loose would straighten through the via.
        board = moving_via_board()
        board.enabled_layers = [BL.BL_F_Cu, BL.BL_In1_Cu, BL.BL_B_Cu]
        inner = kipy_track(
            "i", "SIG2", (0, 1.5), (10, 1.5), width=WIDTH, layer=BL.BL_In1_Cu
        )
        board.tracks.append(inner)
        board.selection.append(inner)
        connect.to(board)

        run("--chord-tolerance=2")

        (via,) = created_vias(board)
        centre = Point(via.position.x / 1e6, via.position.y / 1e6)
        bowed = created_on(board, BL.BL_In1_Cu)
        keep = via.diameter / 2e6 + CLEARANCE + WIDTH / 2
        assert all(segment(t).distance(centre) >= keep - NM_ROUNDING for t in bowed)
        assert len(bowed) > 1

    def test_other_chains_may_take_the_place_the_via_left(
        self, connect: Connector
    ) -> None:
        # Straight, the inner chain would run through the via's start.
        board = moving_via_board()
        board.enabled_layers = [BL.BL_F_Cu, BL.BL_In1_Cu, BL.BL_B_Cu]
        inner = [
            kipy_track("i", "SIG2", (2, 2), (2, 6), width=WIDTH, layer=BL.BL_In1_Cu),
            kipy_track("j", "SIG2", (2, 6), (6, 6), width=WIDTH, layer=BL.BL_In1_Cu),
        ]
        board.tracks += inner
        board.selection += inner
        connect.to(board)

        run()

        (via,) = created_vias(board)
        assert via.position.y < nm(1)
        start = Point(VIA)
        keep = via.diameter / 2e6 + CLEARANCE + WIDTH / 2
        inner_tracks = created_on(board, BL.BL_In1_Cu)
        assert min(segment(t).distance(start) for t in inner_tracks) < keep

    def test_plot_marks_where_the_via_was_and_is(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from kibends import plot

        connect.to(moving_via_board())
        drawn: list[plot.Panel] = []
        real = plot.render

        def spy(path: Path, panels: Sequence[plot.Panel]) -> None:
            drawn.extend(panels)
            real(path, panels)

        monkeypatch.setattr(plot, "render", spy)
        figure = tmp_path / "vias.png"

        run("--dry-run", f"--plot={figure}")

        assert len(drawn) == 2
        for panel in drawn:
            ((start, end),) = panel.vias
            assert start == VIA and end[1] < 1.0
        assert figure.stat().st_size > 0

    def test_plot_marks_the_via_only_on_the_chains_it_joins(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from kibends import plot

        board = moving_via_board()
        board.enabled_layers = [BL.BL_F_Cu, BL.BL_In1_Cu, BL.BL_B_Cu]
        inner = kipy_track(
            "i", "SIG2", (0, 8), (10, 8), width=WIDTH, layer=BL.BL_In1_Cu
        )
        board.tracks.append(inner)
        board.selection.append(inner)
        connect.to(board)
        marks: dict[str, int] = {}

        def spy(path: Path, panels: Sequence[plot.Panel]) -> None:
            marks.update((p.title.split(":")[0], len(p.vias)) for p in panels)

        monkeypatch.setattr(plot, "render", spy)

        run("--dry-run", f"--plot={tmp_path / 'vias.png'}")

        assert marks == {"SIG on B.Cu": 1, "SIG on F.Cu": 1, "SIG2 on In1.Cu": 0}


ZONE_BOARD = Path(__file__).parent / "fixtures" / "zone" / "zone.kicad_pcb"
ZONE_FILL = re.compile(
    r'\(filled_polygon\s*\(layer "In2\.Cu"\)\s*\(pts(.*?)\)\s*\)', re.DOTALL
)
XY = re.compile(r"\(xy (\S+) (\S+)\)")


def fixture_id(tail: str) -> str:
    return f"00000000-0000-0000-0000-{tail:0>12}"


def refilled_sig_fill() -> list[PolygonWithHoles]:
    """The SIG zone's fill on In2.Cu as KiCad refilled the zone fixture: the
    first zone the board text lists. KiCad saved it with a Default clearance
    of 0.127 mm, which `zone_fixture_board` mirrors, so after any edit to the
    fixture refill it with that clearance."""
    text = ZONE_BOARD.read_text(encoding="utf-8")
    sig_zone = text.split('(net "OTHER")\n\t\t(layer')[0]
    return [
        kipy_polygon([(float(x), float(y)) for x, y in XY.findall(pts)])
        for pts in ZONE_FILL.findall(sig_zone)
    ]


ZONE_MIN_WIDTH = 0.2
"""The SIG zone's minimum fill width in the zone fixture, mm."""


def zone_fixture_board(sig_fill: Sequence[PolygonWithHoles]) -> CommitBoard:
    """The zone fixture as KiCad would give it, with the SIG zone filled with
    `sig_fill` and the route and the via at (10, 10) selected."""
    route = [
        kipy_track(fixture_id("11"), "SIG", (2, 10), (10, 10)),
        kipy_track(fixture_id("12"), "SIG", (10, 10), (10, 2), layer=BL.BL_B_Cu),
    ]
    via = kipy_via(fixture_id("e1"), "SIG", 10, 10)
    plane = kipy_filled_zone(
        fixture_id("d1"),
        "SIG",
        {BL.BL_In2_Cu: sig_fill},
        outline=kipy_rect(6, 6, 14, 14),
        min_width=ZONE_MIN_WIDTH,
    )
    plane.clearance = nm(0.2)
    other = kipy_zone(
        kipy_rect(6, 6, 14, 8),
        [BL.BL_In2_Cu],
        rule_area=False,
        zone_id=fixture_id("d2"),
        net="OTHER",
    )
    other.priority = 1
    other.clearance = nm(0.2)
    return CommitBoard(
        selection=[*route, via],
        tracks=[
            *route,
            kipy_track(fixture_id("13"), "SIG", (12, 13), (16, 16)),
            kipy_track(
                fixture_id("14"), "OTHER", (10.625, 10), (18, 10), layer=BL.BL_In2_Cu
            ),
        ],
        vias=[via, kipy_via(fixture_id("e2"), "SIG", 12, 13)],
        zones=[plane, other],
        enabled_layers=[BL.BL_F_Cu, BL.BL_In1_Cu, BL.BL_In2_Cu, BL.BL_B_Cu],
        default_clearance=0.127,
    )


@pytest.mark.usefixtures("coarse_optimizer")
class TestZoneFixture:
    """The zone fixture as KiCad fills it: the SIG route bends
    at the via at (10, 10) in the SIG plane on In2.Cu, and pulls it toward
    (6, 6), into the higher-priority OTHER zone over y < 8. An OTHER track
    on In2.Cu ends at the zone's clearance from the via's ring, so the fill
    cuts through the ring there. A second SIG via, at (12, 13), reaches the
    route only through the plane."""

    def smooth(
        self, connect: Connector, sig_fill: Sequence[PolygonWithHoles]
    ) -> CommitBoard:
        board = connect.to(zone_fixture_board(sig_fill))
        run()
        return board

    def test_a_via_the_fill_reaches_short_of_its_ring_moves_and_commits(
        self, connect: Connector, log: list[str]
    ) -> None:
        fill = refilled_sig_fill()
        assert fill
        board = self.smooth(connect, fill)

        assert "stays fixed" not in "".join(log)
        assert board.calls == ["begin_commit", "delete", "create", "push_commit"]
        (via,) = created_vias(board)
        x, y = via.position.x / 1e6, via.position.y / 1e6
        assert y < 9.5
        filled = unary_union(
            [
                shapely.Polygon([(n.point.x / 1e6, n.point.y / 1e6) for n in p.outline])
                for p in fill
            ]
        )
        (column,) = [
            piece
            for piece in shapely.get_parts(
                LineString([(x, 0), (x, 20)]).intersection(filled)
            )
            if piece.distance(Point(x, y)) < NM_ROUNDING
        ]
        _, edge, _, _ = column.bounds
        margin = ViaZone("SIG", {}, ZONE_MIN_WIDTH).margin
        assert y >= edge + margin - NM_ROUNDING

    def test_a_stale_fill_keeps_the_via_out_of_a_newer_zone(
        self, connect: Connector, log: list[str]
    ) -> None:
        # The fill predates the OTHER zone and covers the whole outline.
        board = self.smooth(connect, [kipy_rect(6, 6, 14, 14)])

        assert "stays fixed" not in "".join(log)
        assert board.calls == ["begin_commit", "delete", "create", "push_commit"]
        (via,) = created_vias(board)
        assert via.position.y / 1e6 >= 8 + 0.2 + 0.1 + KICAD_MAX_ERROR - NM_ROUNDING


def test_outcome_counts_the_vias_rejected_edge_moves() -> None:
    stats = PolylineStats(1.0, 1, math.inf, 0.1, 0.1)
    chain = pipeline.ChainReport(
        "SIG", "F.Cu", WIDTH, "SIG", 1, stats, stats, (), 2, True
    )
    via = pipeline.ViaReport("v", VIA, (4.0, 1.0), frozen=False, rejected_edges=3)
    report = pipeline.OptimizerReport(
        (chain,), (via,), (), Termination.CONVERGED, 7, 3.42, (), BOARD_FRAME
    )

    outcome = cli.format_outcome(report)

    assert outcome == (
        "Optimizer: converged after 7 iterations in 3.4 s, 5 rejected edge moves"
    )


def test_outcome_and_warnings_count_rejected_moves_and_frozen() -> None:
    problem = "the chains and vias on F.Cu and B.Cu"
    stats = PolylineStats(1.0, 1, math.inf, 0.1, 0.1)
    chain = pipeline.ChainReport(
        "USB_D+", "F.Cu", WIDTH, "USB_D+", 1, stats, stats, (), 1, True
    )
    frozen = FrozenVertex((1.0, 2.0), -0.05)
    pair = pipeline.PairReport(
        nets=("USB_D+", "USB_D-"),
        layer="F.Cu",
        gap=0.15,
        gap_after=(0.14, 0.16),
        skew_before=0.0,
        skew_after=0.1,
        sections=1,
        held=0,
        frozen=(frozen,),
        rejected_edges=3,
    )
    report = pipeline.OptimizerReport(
        (chain,),
        (),
        (pair,),
        Termination.CONVERGED,
        7,
        0.04,
        ((problem, 2),),
        UserFrame((0.5, 3.0)),
    )

    outcome = cli.format_outcome(report)
    warnings = cli.report_warnings(report)

    assert outcome == (
        "Optimizer: converged after 7 iterations in 0.0 s, 4 rejected edge moves"
    )
    # The frozen vertex at (1, 2) relative to the frame's origin.
    assert any("(0.500, -1.000)" in w and "0.050 mm inside" in w for w in warnings)
    assert not any(problem in w for w in warnings)


def test_says_what_it_optimizes(connect: Connector, log: list[str]) -> None:
    connect.to(straight_board())

    run("--dry-run")

    assert "INFO Optimizing 1 chain\n" in log


PAIR_WIDTH = 0.2
PAIR_GAP = 0.15
PAIR_CLEARANCE = 0.1
"""Below the gap, so that only coupling keeps the tracks at their gap."""
PAIR_HALF_PITCH = (PAIR_GAP + PAIR_WIDTH) / 2
PAIR_FAN = (-1.0, 1.0 + PAIR_HALF_PITCH)
"""Where track P starts, fanned out at 45° from the coupled section; N
starts mirrored in y."""
SECOND_PAIR = 5.0
"""How far, mm, a second pair runs below the first."""
FAR = 100.0
"""mm, beyond any track of a test board."""
REPORTED = 5e-4
"""mm: how near a pair's reported gap or skew must be to the expected value:
the optimizer and simplification leave them within a micrometre."""
COUPLED_FROM = 0.5
"""x beyond which the pair's tracks run coupled, clear of the fan-out."""
TOO_TIGHT_RADIUS = 0.01
OTHER_PAIR_GAP = 0.2
ABANDONED = "a chain whose bursts all ended abnormally"
PAIR_IDS = ("n1", "n2", "n3", "n4", "p1", "p2", "p3", "p4")
PADS_AT = 10.5
"""x of the same-net pads beside the pair's last straight, mm."""
PAD_OVERLAP = 0.05
"""How far, mm, each pad reaches past its track's centreline."""
ROUNDED_OFF = 0.05
"""mm: how far a rounded corner's track passes from where its corner was,
at least."""
GAP_SLACK = GAP_TOLERANCE * PAIR_GAP
"""How far, mm, a pair's gap may stray and still count as coupled."""
STUB_NEAR_CORNER = 0.02
"""mm before a corner of the pair's inner track where a stub leaves it."""
STUBS_APART = 0.05
"""mm between two stubs leaving one track, less than the default spacing."""
NO_OUTLINE = "Edge.Cuts has no closed outline"
"""The warning of every test board, which has no board edge."""


USB = ("USB_D+", "USB_D-")


def unexpected_warnings(log: Sequence[str]) -> list[str]:
    """The errors and warnings in `log` but the test boards' `NO_OUTLINE`."""
    return [
        line
        for line in log
        if line.startswith(("ERROR", "WARNING")) and NO_OUTLINE not in line
    ]


def pair_tracks(
    nets: tuple[str, str] = USB, at: float = 0.0, prefixes: tuple[str, str] = ("p", "n")
) -> list[kbt.Track]:
    """Pair USB_D+/USB_D-, or `nets`, fanned out at its start, coupled along
    the centreline (0, 0) → (5, 0) → (8, 3) → (13, 3) with 45° corners,
    moved down by `at`."""
    h = PAIR_HALF_PITCH
    mitre = math.tan(math.pi / 8) * h
    tracks: list[kbt.Track] = []
    for net, side, prefix in zip(nets, (1.0, -1.0), prefixes, strict=True):
        points = [
            (PAIR_FAN[0], at + side * PAIR_FAN[1]),
            (0.0, at + side * h),
            (5.0 - side * mitre, at + side * h),
            (8.0 - side * mitre, at + 3.0 + side * h),
            (13.0, at + 3.0 + side * h),
        ]
        tracks += [
            kipy_track(f"{prefix}{k + 1}", net, start, end, width=PAIR_WIDTH)
            for k, (start, end) in enumerate(itertools.pairwise(points))
        ]
    return tracks


def pair_board(
    *, diff_pair_gap: float = PAIR_GAP, extra: Sequence[kbt.Track] = ()
) -> CommitBoard:
    tracks = pair_tracks()
    usb = netclass("USB", PAIR_CLEARANCE, diff_pair_gap)
    return CommitBoard(
        selection=list(tracks),
        tracks=[*tracks, *extra],
        netclasses={"USB_D+": usb, "USB_D-": usb},
    )


PAIR_SECOND_TRACK = 5.0
"""How far, mm, track USB_D- moves away to not pair."""
PAIR_WIDER = 0.25


def with_via_at_the_fan_out(board: CommitBoard) -> CommitBoard:
    """`board` with a via where track USB_D+ fans out, joining it to a
    selected track on B.Cu."""
    down = kipy_track(
        "down",
        "USB_D+",
        PAIR_FAN,
        (-4.0, PAIR_FAN[1]),
        width=PAIR_WIDTH,
        layer=BL.BL_B_Cu,
    )
    via = kipy_via("v", "USB_D+", *PAIR_FAN)
    board.selection = [*board.selection, down, via]
    board.tracks = [*board.tracks, down]
    board.vias = [via]
    return board


def by_net(
    board: CommitBoard, nets: tuple[str, str] = USB
) -> dict[str, list[LineString]]:
    created = cast(list[kbt.Track], board.created)
    return {net: [segment(t) for t in created if t.net.name == net] for net in nets}


def coupled_gaps(p: list[LineString], n: list[LineString]) -> tuple[float, float]:
    """The smallest and largest gap between two tracks of a pair made of
    `p` and `n`, beyond the fan-out where they run coupled."""
    beyond = box(COUPLED_FROM, -FAR, FAR, FAR)
    a, b = (np.array(linemerge(lines).intersection(beyond).coords) for lines in (p, n))
    return gap_range(a, b, PAIR_WIDTH)


@pytest.mark.usefixtures("coarse_optimizer")
class TestPairs:
    def test_commits_both_tracks_in_one_commit_at_their_gap(
        self, connect: Connector
    ) -> None:
        board = connect.to(pair_board())

        run()

        assert board.calls == ["begin_commit", "delete", "create", "push_commit"]
        assert sorted(board.deleted) == list(PAIR_IDS)
        gaps = coupled_gaps(*by_net(board).values())
        assert gaps == pytest.approx((PAIR_GAP, PAIR_GAP), abs=NM_ROUNDING)

    def test_two_pairs_commit_each_joined_and_at_its_gap(
        self, connect: Connector, log: list[str]
    ) -> None:
        eth = ("ETH_P", "ETH_N")
        tracks = [
            *pair_tracks(),
            *pair_tracks(eth, at=SECOND_PAIR, prefixes=("q", "m")),
        ]
        pairs = netclass("DP", PAIR_CLEARANCE, PAIR_GAP)
        board = connect.to(
            CommitBoard(
                selection=list(tracks),
                tracks=list(tracks),
                netclasses={net: pairs for net in (*USB, *eth)},
            )
        )

        run()

        assert "INFO Optimizing 4 chains, 2 differential pairs among them\n" in log
        assert board.calls == ["begin_commit", "delete", "create", "push_commit"]
        for nets, at in ((USB, 0.0), (eth, SECOND_PAIR)):
            lines = by_net(board, nets)
            for net, side in zip(nets, (1.0, -1.0), strict=True):
                joined = linemerge(lines[net])
                assert isinstance(joined, LineString)
                assert {joined.coords[0], joined.coords[-1]} == {
                    (PAIR_FAN[0], at + side * PAIR_FAN[1]),
                    (13.0, at + 3.0 + side * PAIR_HALF_PITCH),
                }
            gaps = coupled_gaps(*lines.values())
            assert gaps == pytest.approx((PAIR_GAP, PAIR_GAP), abs=NM_ROUNDING)

    def test_reports_the_pair_its_gap_and_skew(
        self, connect: Connector, reports: list[pipeline.RunReport]
    ) -> None:
        connect.to(pair_board())

        run("--dry-run")

        ((pair,),) = (r.pairs for r in reports)
        assert (pair.nets, pair.layer) == (USB, "F.Cu")
        assert pair.gap == pytest.approx(PAIR_GAP, abs=REPORTED)
        assert pair.gap_after == pytest.approx((PAIR_GAP, PAIR_GAP), abs=REPORTED)
        # The two corners turn opposite ways, so the tracks start equally long.
        assert pair.skew_before == pytest.approx(0.0, abs=REPORTED)

    def test_reports_the_skew_of_the_committed_tracks(
        self, connect: Connector, reports: list[pipeline.RunReport]
    ) -> None:
        p, n = offset_pair(np.array(BENT_PAIR), PAIR_GAP, PAIR_WIDTH)
        lines = {"USB_D+": vertices(p), "USB_D-": vertices(n)}
        tracks = [
            kipy_track(f"{net}{k}", net, start, end, width=PAIR_WIDTH)
            for net, points in lines.items()
            for k, (start, end) in enumerate(itertools.pairwise(points))
        ]
        usb = netclass("USB", PAIR_CLEARANCE, PAIR_GAP)
        board = connect.to(
            CommitBoard(
                selection=list(tracks),
                tracks=list(tracks),
                netclasses={"USB_D+": usb, "USB_D-": usb},
            )
        )

        run()

        ((pair,),) = (r.pairs for r in reports)
        given = (LineString(points).length for points in lines.values())
        skew = abs(next(given) - next(given))
        assert pair.skew_before == pytest.approx(skew, abs=REPORTED)
        p_now, n_now = (unary_union(c).length for c in by_net(board).values())
        assert pair.skew_after == pytest.approx(abs(p_now - n_now), abs=REPORTED)

    def test_netclass_gap_like_the_measured_one_does_not_warn(
        self, connect: Connector, log: list[str]
    ) -> None:
        connect.to(pair_board())

        run("--dry-run")

        assert not any("netclass" in line for line in log)

    def test_netclass_gap_of_either_net_is_checked(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = pair_board()
        board.netclasses = {
            "USB_D+": netclass("USB", PAIR_CLEARANCE),
            "USB_D-": netclass("USB", PAIR_CLEARANCE, OTHER_PAIR_GAP),
        }
        connect.to(board)

        run("--dry-run")

        (warning,) = [w for w in log if w.startswith("WARNING") and "netclass" in w]
        assert f"{OTHER_PAIR_GAP:.3f}" in warning

    def test_section_fields_leave_out_moving_vias(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch, log: list[str]
    ) -> None:
        seen: list[set[str]] = []
        real = pipeline.build_pair_obstacles

        def spy(
            snapshot: LayerSnapshot,
            a: ChainCentreline,
            b: ChainCentreline,
            gap: float,
            *,
            edge_clearance: float,
            hole_clearance: float,
        ) -> Obstacles:
            seen.append({v.id for v in snapshot.vias})
            return real(
                snapshot,
                a,
                b,
                gap,
                edge_clearance=edge_clearance,
                hole_clearance=hole_clearance,
            )

        monkeypatch.setattr(pipeline, "build_pair_obstacles", spy)
        top = kipy_track("top", "SIG", (2.0, -3.0), (8.0, -3.0), width=PAIR_WIDTH)
        down = kipy_track(
            "down",
            "SIG",
            (8.0, -3.0),
            (12.0, -3.0),
            width=PAIR_WIDTH,
            layer=BL.BL_B_Cu,
        )
        moving, still = kipy_via("v", "SIG", 8.0, -3.0), kipy_via("w", "GND", 4.0, -5.0)
        board = pair_board()
        board.selection = [*board.selection, top, down, moving]
        board.tracks = [*board.tracks, top, down]
        board.vias = [moving, still]
        connect.to(board)

        run("--dry-run")

        assert any("1 moving via" in line for line in log)
        assert seen == [{"w"}]

    def test_section_with_copper_between_its_ends_is_split_there(
        self, connect: Connector, reports: list[pipeline.RunReport]
    ) -> None:
        stub = kipy_track(
            "stub", "USB_D+", (2.5, PAIR_HALF_PITCH), (2.5, 2.0), width=PAIR_WIDTH
        )
        connect.to(pair_board(extra=[stub]))

        run("--dry-run")

        ((pair,),) = (r.pairs for r in reports)
        assert (pair.sections, pair.held) == (2, 0)

    def test_pair_past_same_net_pads_commits_smoothed_and_pinned(
        self, connect: Connector
    ) -> None:
        # As on the board of https://github.com/mairas/kibends/issues/42:
        # each track runs over the edge of a pad of its own net beyond the
        # pair's corners, which pins it where it crosses the pad's outline.
        inside = PAD_HALF_SIZE - PAD_OVERLAP
        pads = [
            kipy_pad(f"pad{net}", net, PADS_AT, 3.0 + side * (PAIR_HALF_PITCH + inside))
            for net, side in zip(USB, (1.0, -1.0), strict=True)
        ]
        board = connect.to(pair_board())
        board.pads = pads

        run()

        assert board.calls == ["begin_commit", "delete", "create", "push_commit"]
        lines = by_net(board)
        for net, side in zip(USB, (1.0, -1.0), strict=True):
            joined = linemerge(lines[net])
            assert isinstance(joined, LineString)
            y = 3.0 + side * PAIR_HALF_PITCH
            for x in (PADS_AT - PAD_HALF_SIZE, PADS_AT + PAD_HALF_SIZE):
                assert (x, y) in joined.coords
            # Both corners ahead of the pads are rounded off.
            mitre = math.tan(math.pi / 8) * PAIR_HALF_PITCH
            for corner in ((5.0, y - 3.0), (8.0, y)):
                at = Point(corner[0] - side * mitre, corner[1])
                assert joined.distance(at) >= ROUNDED_OFF
        low, high = coupled_gaps(*lines.values())
        assert PAIR_GAP - GAP_SLACK <= low <= high <= PAIR_GAP + GAP_SLACK

    @pytest.mark.parametrize("before", [0.0, STUB_NEAR_CORNER])
    def test_a_stub_at_or_near_a_corner_holds_the_section_without_a_warning(
        self,
        connect: Connector,
        log: list[str],
        reports: list[pipeline.RunReport],
        before: float,
    ) -> None:
        # Same-net copper at P's inner corner, or just before it, closer
        # than two pitches: the section stays as routed, while its fan-outs
        # are smoothed.
        corner = 5.0 - math.tan(math.pi / 8) * PAIR_HALF_PITCH
        x = nm(corner - before) / NM_PER_MM
        stub = kipy_track(
            "stub", "USB_D+", (x, PAIR_HALF_PITCH), (x, 2.0), width=PAIR_WIDTH
        )
        board = connect.to(pair_board(extra=[stub]))

        run()

        ((pair,),) = (r.pairs for r in reports)
        assert (pair.sections, pair.held) == (1, 1)
        assert not unexpected_warnings(log)
        lines = by_net(board)
        joined = linemerge(lines[USB[0]])
        assert isinstance(joined, LineString)
        assert (x, PAIR_HALF_PITCH) in joined.coords
        low, high = coupled_gaps(*lines.values())
        assert PAIR_GAP - GAP_SLACK <= low <= high <= PAIR_GAP + GAP_SLACK

    def test_a_part_held_between_two_stubs_commits_the_rest(
        self,
        connect: Connector,
        log: list[str],
        reports: list[pipeline.RunReport],
    ) -> None:
        stubs = [
            kipy_track(
                f"stub{k}", "USB_D+", (x, PAIR_HALF_PITCH), (x, 2.0), width=PAIR_WIDTH
            )
            for k, x in enumerate((2.5, 2.5 + STUBS_APART))
        ]
        board = connect.to(pair_board(extra=stubs))

        run()

        ((pair,),) = (r.pairs for r in reports)
        assert (pair.sections, pair.held) == (3, 1)
        assert board.calls == ["begin_commit", "delete", "create", "push_commit"]
        assert not unexpected_warnings(log)

    def test_netclass_gap_unlike_the_measured_one_warns(
        self, connect: Connector, log: list[str]
    ) -> None:
        connect.to(pair_board(diff_pair_gap=0.2))

        run("--dry-run")

        (warning,) = [w for w in log if w.startswith("WARNING") and "netclass" in w]
        assert "0.150" in warning and "0.200" in warning

    def test_via_on_a_pair_track_stays_fixed(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = connect.to(off_grid(with_via_at_the_fan_out(pair_board())))

        run()

        assert created_vias(board) == []
        assert "v" not in board.deleted
        (warning,) = [w for w in log if "Via v" in w]
        x, y = PAIR_FAN[0] - GRID_ORIGIN[0], PAIR_FAN[1] - GRID_ORIGIN[1]
        assert f"Via v at ({x:.3f}, {y:.3f}) stays fixed" in warning
        assert "differential pair USB_D+ and USB_D- on F.Cu ends at it" in warning
        assert all(by_net(board).values())

    @pytest.mark.parametrize("unpaired", ["far", "wide"])
    def test_via_on_a_chain_that_does_not_pair_may_move(
        self, connect: Connector, log: list[str], unpaired: str
    ) -> None:
        board = pair_board()
        apart = PAIR_SECOND_TRACK if unpaired == "far" else 0.0
        width = PAIR_WIDTH if unpaired == "far" else PAIR_WIDER
        board.tracks = [
            t
            if t.net.name == "USB_D+"
            else kipy_track(
                t.id.value,
                "USB_D-",
                (t.start.x / 1e6, t.start.y / 1e6 - apart),
                (t.end.x / 1e6, t.end.y / 1e6 - apart),
                width=width,
            )
            for t in board.tracks
        ]
        board.selection = list(board.tracks)
        connect.to(with_via_at_the_fan_out(board))

        run("--dry-run")

        assert not any("Via v" in w for w in log)
        assert any("1 moving via" in line for line in log)

    @pytest.mark.parametrize(
        ("too_close", "hold"), [(0, Hold.CROWDING), (1, Hold.FOLDING)]
    )
    def test_section_left_bent_too_tightly_is_held_in_a_second_run(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        log: list[str],
        too_close: int,
        hold: Hold,
    ) -> None:
        real: Callable[..., RunResult] = pipeline.optimize_chains
        held: list[list[Hold | None]] = []

        def too_tight(*args: object, **kwargs: object) -> RunResult:
            result = real(*args, **kwargs)
            sections = cast(Sequence[PairInput], kwargs["pairs"])
            held.append([s.hold for s in sections])
            pairs = tuple(
                p
                if p.block.held
                else replace(
                    p,
                    bend=BendCheck(TOO_TIGHT_RADIUS, too_tight=True),
                    too_close=too_close,
                )
                for p in result.pairs
            )
            return replace(result, pairs=pairs)

        monkeypatch.setattr(pipeline, "optimize_chains", too_tight)
        board = connect.to(pair_board())

        run()

        assert held == [[None], [hold]]
        # Whether a net's chain is committed depends on how far repulsion
        # bends its fan-out (some 25 µm, varying with the BLAS) against the
        # chord tolerance; either way the held section, all pins, stays
        # exactly as routed.
        created = {
            (t.start.x, t.start.y, t.end.x, t.end.y)
            for t in cast(list[kbt.Track], board.created)
        }
        for t in pair_tracks():
            if not t.id.value.endswith("1"):
                assert (
                    t.id.value not in board.deleted
                    or (t.start.x, t.start.y, t.end.x, t.end.y) in created
                )
        (line,) = [w for w in log if "too tight" in w]
        assert line.startswith("DEBUG")
        assert hold.value in line and "again" in line

    def test_time_limit_keeps_the_first_runs_work_in_a_second_run(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        real: Callable[..., RunResult] = pipeline.optimize_chains
        runs: list[RunResult] = []

        def too_tight(*args: object, **kwargs: object) -> RunResult:
            result = real(*args, **kwargs)
            runs.append(result)
            pairs = tuple(
                p
                if p.block.held
                else replace(p, bend=BendCheck(TOO_TIGHT_RADIUS, too_tight=True))
                for p in result.pairs
            )
            return replace(result, pairs=pairs)

        monkeypatch.setattr(pipeline, "optimize_chains", too_tight)
        monkeypatch.setattr(pipeline, "monotonic", BurstClock(monkeypatch))
        board = pair_board()
        # Two Ls of other nets beside the pair, which the first run shortens.
        bend = [
            kipy_track(f"{net}{k}", net, start, end, width=WIDTH)
            for net, y in (("SIG", 10), ("SIG2", 20))
            for k, (start, end) in enumerate([((0, y), (4, y)), ((4, y), (4, y + 4))])
        ]
        board.selection = [*board.selection, *bend]
        board.tracks = [*board.tracks, *bend]
        connect.to(board)

        run("--dry-run", "--time-limit=1.5")

        first, second = runs
        assert first.termination is Termination.TIME_LIMIT
        # Past the limit, the second run still completes one outer iteration,
        # then stops at the first burst of the next.
        assert second.iterations <= 2
        # The first run shortened both Ls, and the second kept that.
        shortened = [
            k
            for k, c in enumerate(first.chains)
            if path_length(c.polyline) < c.before.length - KEPT
        ]
        assert len(shortened) == 2
        for before, after in zip(first.chains, second.chains, strict=True):
            assert path_length(after.polyline) <= path_length(before.polyline) + KEPT
        total = first.iterations + second.iterations
        assert f"after {total} iterations" in capsys.readouterr().out

    def test_a_second_run_reports_only_its_own_abandoned_chains(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch, log: list[str]
    ) -> None:
        real: Callable[..., RunResult] = pipeline.optimize_chains

        def gives_up(*args: object, **kwargs: object) -> RunResult:
            result = real(*args, **kwargs)
            pairs = tuple(
                p
                if p.block.held
                else replace(p, bend=BendCheck(TOO_TIGHT_RADIUS, too_tight=True))
                for p in result.pairs
            )
            abandoned = (ABANDONED, MAX_ABNORMAL_BURSTS)
            return replace(result, pairs=pairs, gave_up=(abandoned,))

        monkeypatch.setattr(pipeline, "optimize_chains", gives_up)
        connect.to(pair_board())

        run("--dry-run")

        (line,) = [w for w in log if "Gave up" in w]
        assert line.startswith("DEBUG") and ABANDONED in line

    def test_dry_run_plot_shows_both_tracks_of_the_pair(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        drawn: list[str] = []

        def render(path: Path, panels: Sequence[plot.Panel]) -> None:
            drawn.extend(p.title for p in panels)

        monkeypatch.setattr(plot, "render", render)
        board = connect.to(pair_board())

        run("--dry-run", f"--plot={tmp_path / 'pair.png'}")

        assert len(drawn) == 2
        assert all("pair with" in title for title in drawn)
        assert board.calls == []


FOLDED = [("USB_D+", 2.3), ("USB_D-", 2.7), ("USB_D+", 3.4)]
"""Net and x, mm, of a same-net pad over one track of the pair's first
straight that once folded a section 180° in a trial step of the first burst
at default settings: https://github.com/mairas/kibends/issues/50."""


@pytest.mark.parametrize(("net", "x"), FOLDED)
def test_a_pair_pinned_on_a_straight_keeps_its_sections_finite(
    connect: Connector, monkeypatch: pytest.MonkeyPatch, net: str, x: float
) -> None:
    results: list[RunResult] = []
    # One outer iteration, as the fold came in the first burst.
    first_burst = at(replace(OptimizerSettings(), max_iterations=1))

    def spy(*args: object, **kwargs: object) -> RunResult:
        result = first_burst(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(pipeline, "optimize_chains", spy)
    # Without its fan-out, the pair is coupled from its start.
    tracks = [t for t in pair_tracks() if t.id.value not in ("p1", "n1")]
    side = 1.0 if net == USB[0] else -1.0
    usb = netclass("USB", PAIR_CLEARANCE, PAIR_GAP)
    board = connect.to(
        CommitBoard(
            selection=list(tracks),
            tracks=list(tracks),
            pads=[kipy_pad("pad", net, x + PAD_HALF_SIZE, side * PAIR_HALF_PITCH)],
            netclasses={n: usb for n in USB},
        )
    )

    run()

    assert board.calls == ["begin_commit", "delete", "create", "push_commit"]
    sections = [p for r in results for p in r.pairs]
    assert sections
    for p in sections:
        assert all(np.isfinite(t).all() for t in (p.centreline, p.a, p.b))
        assert p.bend is None or not p.bend.too_tight


ROUND_OFF = 1e-12


PAIR_BOARD_TRACKS = 9
"""Per net on `pair_board`, as measured at default and coarse settings: two
in its fan-out and seven through the S-bend."""


def pair_tracks_at(
    connect: Connector, monkeypatch: pytest.MonkeyPatch, clearance: float
) -> tuple[dict[str, list[LineString]], list[Simplified]]:
    """The tracks committed for `pair_board` with netclass `clearance`, and
    the run's calls of `simplify_chains`."""
    calls = spy_simplify(monkeypatch)
    board = pair_board()
    usb = netclass("USB", clearance, PAIR_GAP)
    board.netclasses = {"USB_D+": usb, "USB_D-": usb}
    connect.to(board)
    run()
    return by_net(board), calls


@pytest.mark.parametrize(
    "settings",
    [
        # Slow: about 2 s at the default spacing, which neither shorter
        # bursts nor a spacing up to 0.2 mm bring under 1 s.
        pytest.param(DEFAULT_SETTINGS, id="default", marks=pytest.mark.slow),
        pytest.param(COARSE, id="coarse"),
    ],
)
def test_pair_keeps_a_netclass_clearance_equal_to_its_gap(
    connect: Connector, monkeypatch: pytest.MonkeyPatch, settings: OptimizerSettings
) -> None:
    monkeypatch.setattr(pipeline, "optimize_chains", at(settings))

    tracks, calls = pair_tracks_at(connect, monkeypatch, PAIR_GAP)

    p, n = tracks.values()
    assert len(p) <= PAIR_BOARD_TRACKS and len(n) <= PAIR_BOARD_TRACKS
    gap = unary_union(p).distance(unary_union(n)) - PAIR_WIDTH
    assert gap >= PAIR_GAP - NM_ROUNDING
    # Offset from one simplified centreline, the tracks keep the optimised
    # tracks' gap.
    (call,) = calls
    for i, result in zip(call.sections, call.run.pairs, strict=True):
        assert gap_range(*call.tracks(i), PAIR_WIDTH) == pytest.approx(
            gap_range(result.a, result.b, PAIR_WIDTH), abs=ROUND_OFF
        )


@pytest.mark.usefixtures("coarse_optimizer")
def test_a_clearance_equal_to_the_gap_costs_a_pair_no_tracks(
    connect: Connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    below, _ = pair_tracks_at(connect, monkeypatch, PAIR_CLEARANCE)
    equal, _ = pair_tracks_at(connect, monkeypatch, PAIR_GAP)

    for at_gap, below_gap in zip(equal.values(), below.values(), strict=True):
        assert len(at_gap) <= len(below_gap)


@dataclass
class Simplified:
    """One call of `simplify_chains`: its run, sections, tolerance and result."""

    run: RunResult
    sections: Mapping[int, PairBlock]
    tolerance: float
    result: list[FloatArray]

    def tracks(self, i: int) -> tuple[FloatArray, FloatArray]:
        """Section i's tracks as simplified."""
        return self.sections[i].offset_kept(self.run.chains[i].polyline, self.result[i])


def spy_simplify(monkeypatch: pytest.MonkeyPatch) -> list[Simplified]:
    calls: list[Simplified] = []
    real = pipeline.simplify_chains

    def spy(
        chains: Sequence[ChainInput],
        run: RunResult,
        params: EnergyParams,
        tolerance: float,
        *,
        sections: Mapping[int, PairBlock],
    ) -> list[FloatArray]:
        result = real(chains, run, params, tolerance, sections=sections)
        calls.append(Simplified(run, sections, tolerance, result))
        return result

    monkeypatch.setattr(pipeline, "simplify_chains", spy)
    return calls


def unit(step: FloatArray) -> FloatArray:
    return step / float(np.linalg.norm(step))


@pytest.mark.usefixtures("coarse_optimizer")
def test_simplified_sections_meet_their_fan_outs_where_and_about_as_they_did(
    connect: Connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = spy_simplify(monkeypatch)
    connect.to(pair_board())

    run("--dry-run")

    (call,) = calls
    ((i, block),) = call.sections.items()
    # A merged edge passes through the direction pin, a spacing in from the
    # end along the end's direction, to within float round-off.
    turn = math.asin(INSIDE_TOLERANCE / COARSE.spacing) + ROUND_OFF
    for given, simplified in zip(
        (block.given.a.points, block.given.b.points), call.tracks(i), strict=True
    ):
        np.testing.assert_array_equal(simplified[[0, -1]], given[[0, -1]])
        for line, before in ((simplified, given), (simplified[::-1], given[::-1])):
            cosine = float(unit(line[1] - line[0]) @ unit(before[1] - before[0]))
            assert math.acos(min(cosine, 1.0)) <= turn


SPLIT_SECTIONS = ((0.0, 4.0), (6.0, 11.0))
"""x ranges of the two straight coupled sections of `split_pair_board`."""
SPLIT_DETOUR = 1.2


def split_pair_board(second_gap: float = PAIR_GAP) -> CommitBoard:
    """Pair USB_D+/USB_D- fanned out at both ends, coupled along y = 0 in
    two straight sections, split by a detour of each track outward; the
    second section at `second_gap`."""
    h, h2 = PAIR_HALF_PITCH, (second_gap + PAIR_WIDTH) / 2
    (start, split), (rejoin, end) = SPLIT_SECTIONS
    tracks: list[kbt.Track] = []
    for net, side, prefix in (("USB_D+", 1.0, "p"), ("USB_D-", -1.0, "n")):
        points = [
            (start - 1.0, side * (1.0 + h)),
            (start, side * h),
            (split, side * h),
            (split + 0.5, side * SPLIT_DETOUR),
            (rejoin - 0.5, side * SPLIT_DETOUR),
            (rejoin, side * h2),
            (end, side * h2),
            (end + 1.0, side * (1.0 + h2)),
        ]
        tracks += [
            kipy_track(f"{prefix}{k + 1}", net, a, b, width=PAIR_WIDTH)
            for k, (a, b) in enumerate(itertools.pairwise(points))
        ]
    usb = netclass("USB", PAIR_CLEARANCE, PAIR_GAP)
    return CommitBoard(
        selection=list(tracks),
        tracks=list(tracks),
        netclasses={"USB_D+": usb, "USB_D-": usb},
    )


@pytest.mark.usefixtures("coarse_optimizer")
def test_straight_coupled_sections_commit_as_one_track_per_net(
    connect: Connector,
) -> None:
    board = connect.to(split_pair_board())

    run()

    for net, lines in by_net(board).items():
        y = PAIR_HALF_PITCH if net == "USB_D+" else -PAIR_HALF_PITCH
        coupled = sorted(
            tuple(round(x, 6) for x, _ in line.coords)
            for line in lines
            if all(abs(ys - y) < NM_ROUNDING for _, ys in line.coords)
        )
        assert coupled == list(SPLIT_SECTIONS)


BENT_PAIR = ((0.0, 0.0), (5.0, 0.0), (10.0, 5.0))
"""Centreline of a pair turning once, so that its tracks differ in length."""


def vertices(points: FloatArray) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in points]


BUS_CENTRE = ((0.0, 0.0), (5.0, 0.0), (8.0, 3.0), (13.0, 3.0))
BUS_NETS = (("USB_D+", "USB_D-"), ("ETH_P", "ETH_N"))
BUS_MARGIN = 1e-4
"""mm beyond clearance between the two pairs of `bus_board`."""


def bus_board(margin: float = BUS_MARGIN) -> CommitBoard:
    """Two pairs side by side along `BUS_CENTRE`, without fan-outs, `margin`
    beyond a netclass clearance equal to their gap."""
    apart = 2 * PAIR_WIDTH + PAIR_GAP + PAIR_GAP + margin
    first = np.array(BUS_CENTRE)
    second, _ = offset_pair(first, 2 * apart, 0.0)
    tracks: list[kbt.Track] = []
    for centre, nets in zip((first, second), BUS_NETS, strict=True):
        pair = offset_pair(centre, PAIR_GAP, PAIR_WIDTH)
        for net, track in zip(nets, pair, strict=True):
            tracks += [
                kipy_track(f"{net}{k}", net, start, end, width=PAIR_WIDTH)
                for k, (start, end) in enumerate(itertools.pairwise(vertices(track)))
            ]
    bus = netclass("BUS", PAIR_GAP, PAIR_GAP)
    return CommitBoard(
        selection=list(tracks),
        tracks=list(tracks),
        netclasses={net: bus for nets in BUS_NETS for net in nets},
    )


@pytest.mark.usefixtures("coarse_optimizer")
@pytest.mark.parametrize("margin", [BUS_MARGIN, 0.02], ids=["at", "beyond"])
def test_two_pairs_at_clearance_both_smooth_and_simplify(
    connect: Connector, log: list[str], margin: float
) -> None:
    board = connect.to(bus_board(margin))

    run()

    lines = {net: by_net(board, nets)[net] for nets in BUS_NETS for net in nets}
    straight = len(BUS_CENTRE) - 1
    assert all(len(tracks) > straight for tracks in lines.values())
    for a, b in BUS_NETS:
        assert len(lines[a]) == len(lines[b])
    assert not any("unsimplified" in line or "inside" in line for line in log)
    usb = unary_union([*lines["USB_D+"], *lines["USB_D-"]])
    eth = unary_union([*lines["ETH_P"], *lines["ETH_N"]])
    assert usb.distance(eth) - PAIR_WIDTH >= PAIR_GAP - NM_ROUNDING


NARROWER_GAP = 0.14
"""Of the second section of a split pair: within the pair's tolerance."""


@pytest.mark.usefixtures("coarse_optimizer")
def test_reports_the_gap_range_over_every_section(
    connect: Connector, reports: list[pipeline.RunReport]
) -> None:
    connect.to(split_pair_board(NARROWER_GAP))

    run("--dry-run")

    ((pair,),) = (r.pairs for r in reports)
    assert pair.gap_after == pytest.approx((NARROWER_GAP, PAIR_GAP), abs=REPORTED)


def routed_too_tightly_board() -> CommitBoard:
    """Pair USB_D+/USB_D- turning 90° in a chamfer so short that its
    centreline bends tighter than the inner track allows."""
    turn = (5.0, 0.0)
    centre = np.array(
        [
            (0.0, 0.0),
            turn,
            (turn[0] + SHORT_CHAMFER, SHORT_CHAMFER),
            (turn[0] + SHORT_CHAMFER, 5.0),
        ]
    )
    tracks = [
        kipy_track(f"{net}{k}", net, start, end, width=PAIR_WIDTH)
        for net, track in zip(
            USB, offset_pair(centre, PAIR_GAP, PAIR_WIDTH), strict=True
        )
        for k, (start, end) in enumerate(itertools.pairwise(vertices(track)))
    ]
    usb = netclass("USB", PAIR_CLEARANCE, PAIR_GAP)
    return CommitBoard(
        selection=list(tracks),
        tracks=list(tracks),
        netclasses={"USB_D+": usb, "USB_D-": usb},
    )


SHORT_CHAMFER = 0.14
"""mm along each axis: a bend radius of about 0.24 mm, below the 0.275 mm
limit, with the inner track's chamfer still of positive length."""


@pytest.mark.usefixtures("coarse_optimizer")
def test_a_section_routed_too_tightly_is_held(
    connect: Connector, log: list[str], reports: list[pipeline.RunReport]
) -> None:
    connect.to(routed_too_tightly_board())

    run("--dry-run")

    ((pair,),) = (r.pairs for r in reports)
    assert pair.held == 1
    (line,) = [w for w in log if Hold.ROUTING.value in w]
    assert line.startswith("DEBUG") and "radius of" in line


@pytest.mark.usefixtures("coarse_optimizer")
def test_a_chain_beside_a_pair_at_clearance_does_not_freeze_it(
    connect: Connector, log: list[str]
) -> None:
    first = np.array(BUS_CENTRE)
    apart = PAIR_WIDTH + PAIR_GAP / 2 + CLEARANCE + PAIR_WIDTH / 2 + BUS_MARGIN
    side, _ = offset_pair(first, 2 * apart, 0.0)
    tracks = [
        kipy_track(f"{net}{k}", net, start, end, width=PAIR_WIDTH)
        for net, points in (
            *zip(USB, offset_pair(first, PAIR_GAP, PAIR_WIDTH), strict=True),
            ("SIG", side),
        )
        for k, (start, end) in enumerate(itertools.pairwise(vertices(points)))
    ]
    usb = netclass("USB", PAIR_GAP, PAIR_GAP)
    board = connect.to(
        CommitBoard(
            selection=list(tracks),
            tracks=list(tracks),
            netclasses={"USB_D+": usb, "USB_D-": usb},
        )
    )

    run()

    assert not any("inside" in line for line in log)
    assert all(len(lines) > len(BUS_CENTRE) - 1 for lines in by_net(board).values())


CROWDED_PAIR = ((0.0, 0.0), (3.0, 0.0), (4.8, 1.8), (7.8, 1.8), (7.8, 4.8))
"""Centreline of a pair with an S-bend and then a sharp corner."""
CROWDED_PAD = (7.175, 2.425)
"""A GND pad inside the sharp corner, at the pair's clearance, which keeps
the pair from rounding the corner that resampling cuts too tightly."""
SIDE_CHAIN = ((0.6, 0.0), (3.0, 0.0), (4.8, 1.8), (7.2, 1.8))
SIDE_OFFSET = 0.62
"""Of the selected SIG chain beside the S-bend, outside the bend."""
CROWDED_FAN = 0.6
FINE = OptimizerSettings(
    spacing=0.15, settled_change=0.0015, burst_maxiter=30, max_iterations=30
)
"""Coarser settings round the corner cleanly."""


@pytest.mark.slow
def test_a_crowded_section_is_held_and_its_neighbours_keep_clear_of_it(
    connect: Connector, monkeypatch: pytest.MonkeyPatch, log: list[str]
) -> None:
    # Slow: only spacings as fine as this cut the corner too tightly, and the
    # run then repeats with the section held, about 2 s in all.
    monkeypatch.setattr(pipeline, "optimize_chains", at(FINE))
    p, n = offset_pair(np.array(CROWDED_PAIR), PAIR_GAP, PAIR_WIDTH)
    x, y = CROWDED_PAIR[0]
    fanned = {
        "USB_D+": [(x - CROWDED_FAN, y + PAIR_HALF_PITCH + CROWDED_FAN), *vertices(p)],
        "USB_D-": [(x - CROWDED_FAN, y - PAIR_HALF_PITCH - CROWDED_FAN), *vertices(n)],
    }
    _, side = offset_pair(np.array(SIDE_CHAIN), 2 * SIDE_OFFSET, 0.0)
    tracks = [
        kipy_track(f"{net}{k}", net, start, end, width=PAIR_WIDTH)
        for net, points in (*fanned.items(), ("SIG", vertices(side)))
        for k, (start, end) in enumerate(itertools.pairwise(points))
    ]
    usb = netclass("USB", PAIR_CLEARANCE, PAIR_GAP)
    board = connect.to(
        CommitBoard(
            selection=list(tracks),
            tracks=list(tracks),
            pads=[kipy_pad("pad", "GND", *CROWDED_PAD)],
            netclasses={"USB_D+": usb, "USB_D-": usb},
        )
    )

    run()

    assert any(Hold.CROWDING.value in line for line in log)
    created = cast(list[kbt.Track], board.created)
    final = {
        net: unary_union([segment(t) for t in created if t.net.name == net])
        or LineString(points)
        for net, points in (*fanned.items(), ("SIG", vertices(side)))
    }
    # The section's tracks stay as routed; at most the fan-outs change.
    for net, points in fanned.items():
        coupled = LineString(points[1:])
        assert coupled.difference(final[net].buffer(NM_ROUNDING)).is_empty
    assert not final["SIG"].equals(LineString(vertices(side)))
    pair = unary_union([final[net] for net in fanned])
    assert final["SIG"].distance(pair) - PAIR_WIDTH >= CLEARANCE - NM_ROUNDING


STUB_GAP = 0.138
"""The gap of a pair's 1 mm end stubs, unlike the 0.150 mm between them."""
STUB = 1.0
STUBBED_LENGTH = 10.0
STUB_JOG = 0.1
"""How long, along the pair, each track takes to step between the gaps."""
TAPER_REACH = 2.0
TAPER_SAMPLES = 50
"""mm from each end within which the gap may still follow the stubs':
the stub, the jog and the taper the optimizer turns the step into."""


def stubbed_pair_board() -> CommitBoard:
    """Pair USB_D+/USB_D- along y = 0, at `STUB_GAP` over `STUB` at each end
    and at `PAIR_GAP` between."""
    end, middle = (STUB_GAP + PAIR_WIDTH) / 2, PAIR_HALF_PITCH
    far = STUBBED_LENGTH
    tracks: list[kbt.Track] = []
    for net, side in zip(USB, (1.0, -1.0), strict=True):
        points = [
            (0.0, side * end),
            (STUB, side * end),
            (STUB + STUB_JOG, side * middle),
            (far - STUB - STUB_JOG, side * middle),
            (far - STUB, side * end),
            (far, side * end),
        ]
        tracks += [
            kipy_track(f"{net}{k}", net, a, b, width=PAIR_WIDTH)
            for k, (a, b) in enumerate(itertools.pairwise(points))
        ]
    usb = netclass("USB", PAIR_GAP, PAIR_GAP)
    return CommitBoard(
        selection=list(tracks),
        tracks=list(tracks),
        netclasses={"USB_D+": usb, "USB_D-": usb},
    )


def test_end_gaps_taper_into_the_pair_s_gap_and_no_further(
    connect: Connector,
) -> None:
    board = connect.to(stubbed_pair_board())

    run()

    p, n = (unary_union(lines) for lines in by_net(board).values())
    middle = box(TAPER_REACH, -FAR, STUBBED_LENGTH - TAPER_REACH, FAR)
    inner = p.intersection(middle)
    assert not inner.is_empty
    gaps = [
        n.distance(inner.interpolate(d)) - PAIR_WIDTH
        for d in np.linspace(0.0, inner.length, TAPER_SAMPLES)
    ]
    assert gaps == pytest.approx([PAIR_GAP] * len(gaps), abs=NM_ROUNDING)


PARALLEL = 1e-6
"""rad: the most a segment of one track may turn from the other's."""


def corner_board(*centre: tuple[float, float]) -> CommitBoard:
    """Pair USB_D+/USB_D- along `centre`, at clearance below its gap."""
    tracks = [
        kipy_track(f"{net}{k}", net, start, end, width=PAIR_WIDTH)
        for net, track in zip(
            USB, offset_pair(np.array(centre), PAIR_GAP, PAIR_WIDTH), strict=True
        )
        for k, (start, end) in enumerate(itertools.pairwise(vertices(track)))
    ]
    usb = netclass("USB", PAIR_CLEARANCE, PAIR_GAP)
    return CommitBoard(
        selection=list(tracks),
        tracks=list(tracks),
        netclasses={"USB_D+": usb, "USB_D-": usb},
    )


PAIRED_BOARDS: dict[str, Callable[[], CommitBoard]] = {
    "bus": bus_board,
    "pair_board": pair_board,
    "90": partial(corner_board, (0.0, 0.0), (5.0, 0.0), (5.0, 5.0)),
    "135": partial(corner_board, (0.0, 0.0), (5.0, 0.0), (1.5, 3.5)),
}


@pytest.mark.parametrize(
    ("layout", "settings"),
    [
        *(pytest.param(name, COARSE, id=f"{name}-coarse") for name in PAIRED_BOARDS),
        # The bus at the default spacing takes about 9 s; the coarse bus
        # and the other layouts at the default spacing cover it. Those take
        # 1 to 2 s each, slow too.
        *(
            pytest.param(
                name, DEFAULT_SETTINGS, id=f"{name}-default", marks=pytest.mark.slow
            )
            for name in PAIRED_BOARDS
            if name != "bus"
        ),
    ],
)
def test_coupled_tracks_commit_segment_for_segment_in_parallel(
    connect: Connector,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    settings: OptimizerSettings,
) -> None:
    # As KiCad's router expects of a pair: otherwise a later edit of it in
    # KiCad no longer treats it as coupled.
    monkeypatch.setattr(pipeline, "optimize_chains", at(settings))
    calls = spy_simplify(monkeypatch)
    board = connect.to(PAIRED_BOARDS[layout]())

    run()

    (call,) = calls
    created = cast(list[kbt.Track], board.created)
    for i, block in call.sections.items():
        a, b = call.tracks(i)
        assert len(a) == len(b)
        optimised = call.run.chains[i].polyline
        gaps = block.gaps.at(optimised)[
            [
                int(np.flatnonzero((optimised == v).all(axis=1))[0])
                for v in call.result[i]
            ]
        ]
        for k in np.flatnonzero(np.diff(gaps) == 0.0):
            turn = unit(a[k + 1] - a[k]) @ unit(b[k + 1] - b[k])
            assert math.acos(min(float(turn), 1.0)) <= PARALLEL
        for track, net in ((a, block.given.a.net), (b, block.given.b.net)):
            on_track = [
                t
                for t in created
                if t.net.name == net
                and all(
                    np.hypot(*(track - point).T).min() <= NM_ROUNDING
                    for point in segment(t).coords
                )
            ]
            assert len(on_track) == len(track) - 1


FOLD_KEEPERS = (
    kipy_track("g1", "GND", (3.5, 0.8), (4.5, 0.8)),
    kipy_track("g2", "GND", (8.5, 2.2), (9.5, 2.2)),
)
"""GND tracks between the pair's route and its straight shortcut, which keep
it bent: along a straight centreline no order of its vertices crosses."""


@pytest.mark.usefixtures("coarse_optimizer")
def test_a_pair_whose_tracks_cross_is_not_committed(
    connect: Connector, monkeypatch: pytest.MonkeyPatch, log: list[str]
) -> None:
    real = pipeline.simplify_chains

    def folded(
        chains: Sequence[ChainInput],
        run: RunResult,
        params: EnergyParams,
        tolerance: float,
        *,
        sections: Mapping[int, PairBlock],
    ) -> list[FloatArray]:
        result = real(chains, run, params, tolerance, sections=sections)
        for i, block in sections.items():
            optimised = run.chains[i].polyline
            last = len(optimised) - 1
            # Some order of a few of its vertices where its tracks cross.
            result[i] = next(
                kept
                for j, k in itertools.permutations(range(1, last), 2)
                if LineString(
                    (
                        tracks := block.offset_kept(
                            optimised, kept := optimised[[0, j, k, last]]
                        )
                    )[0]
                ).intersects(LineString(tracks[1]))
            )
        return result

    monkeypatch.setattr(pipeline, "simplify_chains", folded)
    board = connect.to(pair_board(extra=FOLD_KEEPERS))

    message = exit_message(log)

    assert "USB_D+ and USB_D-" in message and "come within" in message
    assert board.calls == []


V_REPULSION = "0.5"


def v_board(turn: float) -> CommitBoard:
    """Pair USB_D+/USB_D- turning back by `turn` degrees, with a GND pad
    inside the V halfway along its legs."""
    centre, pad = scenarios.turning_back(turn)
    tracks = [
        kipy_track(f"{net}{k}", net, start, end, width=PAIR_WIDTH)
        for net, track in zip(
            USB, offset_pair(centre, PAIR_GAP, PAIR_WIDTH), strict=True
        )
        for k, (start, end) in enumerate(itertools.pairwise(vertices(track)))
    ]
    usb = netclass("USB", PAIR_GAP, PAIR_GAP)
    return CommitBoard(
        selection=list(tracks),
        tracks=list(tracks),
        pads=[kipy_pad("pad", "GND", *pad)],
        netclasses={"USB_D+": usb, "USB_D-": usb},
    )


def test_a_pair_turning_back_around_a_pad_keeps_clear_of_itself(
    connect: Connector,
    monkeypatch: pytest.MonkeyPatch,
    reports: list[pipeline.RunReport],
) -> None:
    # Simplification and the gap check on a pair whose legs face each other,
    # end to end, at coarse settings; test_optimize_pairs covers the
    # optimizer on the same scenario.
    monkeypatch.setattr(pipeline, "optimize_chains", at(COARSE))
    board = connect.to(v_board(150.0))

    run(f"--repulsion={V_REPULSION}")

    ((pair,),) = (r.pairs for r in reports)
    assert pair.gap_after == pytest.approx((PAIR_GAP, PAIR_GAP), abs=REPORTED)
    p, n = (unary_union(lines) for lines in by_net(board).values())
    assert p.distance(n) - PAIR_WIDTH >= PAIR_GAP - NM_ROUNDING


SPIRAL_GAP = 0.1
SPIRAL_CLEARANCE = 0.2
"""Above the pair's gap, as custom rules let a pair route closer than its
nets' clearance."""
SPIRAL_LAP = 2 * PAIR_WIDTH + SPIRAL_GAP + SPIRAL_CLEARANCE
"""mm between the laps' centrelines: one lap's tracks keep exactly their
clearance from the next's."""
SPIRAL_SIDE = 6.0
SPIRAL_ALONG = 5.0
"""mm along the tracks within which they face each other as a pair, not
across laps: well under a lap."""
SPIRAL_SAMPLE = 0.02


def spiral(lap: float) -> FloatArray:
    """Two laps of a square spiral winding inwards, `lap` apart."""
    corners = [(0.0, 0.0)]
    for k in range(2):
        near, far = k * lap, SPIRAL_SIDE - k * lap
        corners += [(far, near), (far, far), (near, far), (near, near + lap)]
    return np.array(corners)


def spiral_board() -> CommitBoard:
    """Pair USB_D+/USB_D- winding two laps of a square spiral inwards."""
    tracks = [
        kipy_track(f"{net}{k}", net, start, end, width=PAIR_WIDTH)
        for net, track in zip(
            USB, offset_pair(spiral(SPIRAL_LAP), SPIRAL_GAP, PAIR_WIDTH), strict=True
        )
        for k, (start, end) in enumerate(itertools.pairwise(vertices(track)))
    ]
    usb = netclass("USB", SPIRAL_CLEARANCE, SPIRAL_GAP)
    return CommitBoard(
        selection=list(tracks),
        tracks=list(tracks),
        netclasses={"USB_D+": usb, "USB_D-": usb},
    )


def across_laps(p: LineString, n: LineString) -> float:
    """The smallest edge-to-edge gap between p and n where they lie more
    than SPIRAL_ALONG apart along themselves, sampled."""
    lines = [shapely.segmentize(line, SPIRAL_SAMPLE) for line in (p, n)]
    a, b = (shapely.get_coordinates(line) for line in lines)
    along_a, along_b = (
        np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(c, axis=0).T))])
        for c in (a, b)
    )
    distance = np.hypot(*(a[:, None] - b[None]).transpose(2, 0, 1))
    beyond = np.abs(along_a[:, None] - along_b[None]) > SPIRAL_ALONG
    return float(distance[beyond].min()) - PAIR_WIDTH


@pytest.mark.parametrize("chord", ["0.025", "0.1"])
@pytest.mark.usefixtures("coarse_optimizer")
def test_a_spiral_pair_keeps_its_clearance_across_laps(
    connect: Connector, log: list[str], chord: str
) -> None:
    board = connect.to(spiral_board())

    with contextlib.suppress(SystemExit):
        run(f"--chord-tolerance={chord}")

    # Refused, or held as routed, it commits nothing; which one depends on
    # the platform's floating point.
    committed = by_net(board)
    if not any(committed.values()):
        assert board.calls == []
        return
    p, n = (linemerge(lines) for lines in committed.values())
    assert isinstance(p, LineString) and isinstance(n, LineString)
    assert across_laps(p, n) >= SPIRAL_CLEARANCE - NM_ROUNDING


PINCH_LEG = 10.0
PINCH_APART = 2 * PAIR_WIDTH + SPIRAL_GAP + SPIRAL_CLEARANCE
"""How far apart the run keeps a section's parts beyond a turn: its outer
extent plus clearance."""
PINCH_MARGIN = 0.001
"""mm by which a result's gap misses its floor, well above round-off."""


def section(
    gap: float, clearance: float, other_clearance: float | None = None
) -> PairInput:
    """A straight section of pair USB_D+/USB_D- routed at `gap`."""
    a, b = offset_pair(np.array([(0.0, 0.0), (PINCH_LEG, 0.0)]), gap, PAIR_WIDTH)
    fixed = np.zeros(len(a), dtype=bool)
    return PairInput(
        ChainCentreline("P", USB[0], PAIR_WIDTH, clearance, a),
        ChainCentreline("N", USB[1], PAIR_WIDTH, other_clearance or clearance, b),
        fixed,
        fixed,
        gap,
        "F.Cu",
        FreeSpace(),
    )


def pinched(
    given: PairInput, *centreline: tuple[float, float], gap: float
) -> str | None:
    tracks = offset_pair(np.array(centreline), gap, PAIR_WIDTH)
    return pipeline._pinched([given], [PINCH_APART], [tracks])  # pyright: ignore[reportPrivateUsage]


STRAIGHT = ((0.0, 0.0), (PINCH_LEG, 0.0))


def test_a_pair_may_come_closer_than_routed_down_to_its_clearance() -> None:
    given = section(PAIR_GAP, PAIR_CLEARANCE)

    assert pinched(given, *STRAIGHT, gap=PAIR_CLEARANCE) is None


def test_a_pair_may_not_come_closer_than_the_larger_of_its_clearances() -> None:
    wide = SPIRAL_CLEARANCE + PAIR_GAP
    given = section(wide, PAIR_CLEARANCE, SPIRAL_CLEARANCE)

    message = pinched(given, *STRAIGHT, gap=SPIRAL_CLEARANCE - PINCH_MARGIN)

    assert message is not None and f"{SPIRAL_CLEARANCE:.4f} mm" in message


def test_a_pair_routed_below_its_clearance_may_not_come_closer_still() -> None:
    given = section(SPIRAL_GAP, SPIRAL_CLEARANCE)

    message = pinched(given, *STRAIGHT, gap=SPIRAL_GAP - PINCH_MARGIN)

    assert message is not None and f"{SPIRAL_GAP:.4f} mm" in message


def test_a_pair_routed_below_its_clearance_keeps_it_across_a_turn() -> None:
    given = section(SPIRAL_GAP, SPIRAL_CLEARANCE)
    # Each leg's inner track lies as far from the other leg's outer track,
    # of the other net, as the legs' centrelines lie apart.
    legs = PAIR_WIDTH + SPIRAL_CLEARANCE - PINCH_MARGIN
    u_turn = ((0.0, 0.0), (PINCH_LEG, 0.0), (PINCH_LEG, legs), (0.0, legs))

    message = pinched(given, *u_turn, gap=SPIRAL_GAP)

    assert message is not None and "across a turn" in message


@pytest.mark.parametrize(
    ("turn", "width", "gap", "clearance"),
    [(150.0, 0.2, 0.1, 0.2), (135.0, 0.1, 0.1, 0.3), (150.0, 0.1, 0.1, 0.3)],
)
def test_a_sharp_pair_held_as_routed_is_not_refused(
    connect: Connector,
    log: list[str],
    reports: list[pipeline.RunReport],
    turn: float,
    width: float,
    gap: float,
    clearance: float,
) -> None:
    centre, _ = scenarios.turning_back(turn)
    p, n = offset_pair(centre, gap, width)
    tracks = [
        kipy_track(f"{net}{k}", net, start, end, width=width)
        for net, track in zip(USB, (p, n), strict=True)
        for k, (start, end) in enumerate(itertools.pairwise(vertices(track)))
    ]
    # Same-net copper on P's apex, whose outline crosses P closer to the
    # turn than two pitches, holds the section as routed.
    held_by = kipy_pad("pad", USB[0], float(p[1, 0]), float(p[1, 1]))
    usb = netclass("USB", clearance, gap)
    board = connect.to(
        CommitBoard(
            selection=list(tracks),
            tracks=list(tracks),
            pads=[held_by],
            netclasses={"USB_D+": usb, "USB_D-": usb},
        )
    )

    run()

    ((pair,),) = (r.pairs for r in reports)
    assert (pair.sections, pair.held) == (1, 1)
    assert not any(line.startswith("ERROR") for line in log)
    assert board.calls == []


def test_a_pair_routed_closer_than_its_clearance_across_a_turn_may_stay_so() -> None:
    legs = PAIR_WIDTH + SPIRAL_CLEARANCE - PINCH_MARGIN
    given, u_turn = u_section(legs, SPIRAL_GAP, SPIRAL_CLEARANCE)
    shift = np.array([PINCH_MARGIN, 0.0])

    assert pinched(given, *(u_turn + shift), gap=SPIRAL_GAP) is None


def u_section(
    legs: float, gap: float, clearance: float
) -> tuple[PairInput, FloatArray]:
    """A section routed as a 10 mm U with its legs `legs` apart, and that U."""
    u_turn = np.array([(0.0, 0.0), (PINCH_LEG, 0.0), (PINCH_LEG, legs), (0.0, legs)])
    a, b = offset_pair(u_turn, gap, PAIR_WIDTH)
    straight = section(gap, clearance)
    return (
        replace(
            straight,
            a=replace(straight.a, points=a),
            b=replace(straight.b, points=b),
        ),
        u_turn,
    )


HOOK_BACK = 0.2
"""mm a hook runs back beside itself before turning away."""
HOOK_SHORT = 0.01
"""mm by which the hook's legs fall short of clearance across."""


def test_a_hook_s_turn_is_side_by_side_within_half_a_circle() -> None:
    # Its two short legs lie farther apart along it than its outer extent
    # plus clearance, but within the half circle of that width.
    given = section(SPIRAL_GAP, SPIRAL_CLEARANCE)
    legs = PAIR_WIDTH + SPIRAL_CLEARANCE - HOOK_SHORT
    back = PINCH_LEG - HOOK_BACK
    hook = ((0.0, 0.0), (PINCH_LEG, 0.0), (PINCH_LEG, legs), (back, legs), (back, 5.0))

    assert pinched(given, *hook, gap=SPIRAL_GAP) is None


def test_a_pair_keeps_its_side_by_side_gap_though_closer_across_a_turn() -> None:
    wide = 3 * SPIRAL_GAP
    clearance = 2 * wide
    # One lap's tracks face the next's closer than each other.
    laps = spiral(2 * PAIR_WIDTH + 2 * wide - SPIRAL_GAP)
    a, b = offset_pair(laps, wide, PAIR_WIDTH)
    straight = section(wide, clearance)
    given = replace(
        straight, a=replace(straight.a, points=a), b=replace(straight.b, points=b)
    )

    message = pinched(given, *laps, gap=wide - PINCH_MARGIN)

    assert message is not None and "across" not in message
