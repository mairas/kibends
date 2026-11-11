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

import conftest
import kipy.board_types as kbt
import numpy as np
import pytest
import scenarios
import shapely
from conftest import (
    COMMIT_CALLS,
    PAD_HALF_SIZE,
    RETURN_SIGNAL_VIA,
    RETURN_VIAS_APART,
    BurstClock,
    CommitBoard,
    FakeKiCad,
    Ticking,
    kipy_arc,
    kipy_copper_zone,
    kipy_group,
    kipy_pad,
    kipy_rect,
    kipy_track,
    kipy_via,
    kipy_zone,
    netclass,
    neutral_distance,
    nm,
    return_via_board,
    vec,
    via_ids,
)
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.geometry import Vector2
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.board.board_types_pb2 import ViaType
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from kipy.util.board_layer import iter_copper_layers
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, unary_union

import kibends.optimize as optimize_module
from kibends import cli, plot
from kibends import run as pipeline
from kibends.bursts import (
    DEFAULT_SETTINGS,
    INSIDE_TOLERANCE,
    Burst,
    Objective,
    OptimizerSettings,
    Revert,
    Stop,
)
from kibends.chain import NM_PER_MM, Point2
from kibends.coords import BOARD_FRAME, UserFrame
from kibends.energy import (
    EnergyParams,
    FloatArray,
)
from kibends.neighbours import ChainInput
from kibends.obstacles import ChainCentreline, Obstacles
from kibends.optimize import MAX_ABNORMAL_BURSTS, optimize_chains
from kibends.pair_blocks import Hold, PairBlock, PairInput
from kibends.pairs import GAP_TOLERANCE, BendCheck, gap_range, offset_pair
from kibends.results import PolylineStats, RunResult, Termination
from kibends.snapshot import CONTACT_MARGIN, KICAD_MAX_ERROR, LayerSnapshot
from kibends.vias import StaticVia, ViaInput

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


WIDE = 0.5
NARROW = 0.25
WIDTH_CHANGE = (12.0, 4.0)
"""Where `width_change` narrows, at its corner."""
# Positions of a straight result within this of the ideal line, mm: COARSE
# leaves the width change bowed by about 2 µm.
STRAIGHT_TOLERANCE = 0.01


def width_change() -> list[kbt.Track]:
    """A WIDE route from the origin narrowing to NARROW at `WIDTH_CHANGE`,
    ending at (20, 4): 20.396 mm straight."""
    return [
        kipy_track("w1", "SIG", (0, 0), (8, 0), width=WIDE),
        kipy_track("w2", "SIG", (8, 0), WIDTH_CHANGE, width=WIDE),
        kipy_track("n", "SIG", WIDTH_CHANGE, (20, 4), width=NARROW),
    ]


def straight_board() -> CommitBoard:
    """A straight chain in free space: nothing to optimise."""
    track = kipy_track("a", "SIG", (0, 0), (10, 0), width=WIDTH)
    return CommitBoard(selection=[track], tracks=[track])


@pytest.fixture(autouse=True)
def keep_test_log_sinks(monkeypatch: pytest.MonkeyPatch) -> None:
    def configure_logging(*, verbose: bool) -> None:
        pass

    monkeypatch.setattr(cli, "configure_logging", configure_logging)


COARSE = OptimizerSettings(spacing=0.3, settled_change=0.003, burst_maxiter=30)


KEPT = 1e-3
"""mm a chain's length may rise by between runs, as repulsion trades it."""
# The runs of `return_via_board` with `BALANCED` here end 0.0005 to
# 0.004 mm from the neutral distance; the rest is platform slack.
RETURN_SETTLED = 0.02
"""How near its neutral distance a SIG via and its return via settle under
COARSE, mm."""
BALANCED = "--attraction=0.25"
"""The argument for a weight whose pull stays below the push beyond the
clearance and standoff of `return_via_board`'s vias, so that they balance
about 1.58 mm apart; the default pulls harder."""
# The default's runs of `return_via_board` here end 0.0065 mm shorter to
# 0.0033 mm longer than without attraction; the rest is platform slack.
RETURN_TRADED = 0.01
"""mm the default's pull may lengthen `return_via_board`'s route by, against
the run without attraction."""


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

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_a_width_change_moves_and_commits_one_straight_line(
        self, connect: Connector
    ) -> None:
        tracks = width_change()
        board = connect.to(CommitBoard(selection=list(tracks), tracks=list(tracks)))

        run()

        assert sorted(board.deleted) == ["n", "w1", "w2"]
        created = cast(list[kbt.Track], board.created)
        (wide,) = [t for t in created if t.width == nm(WIDE)]
        (narrow,) = [t for t in created if t.width == nm(NARROW)]
        assert (wide.start.x, wide.start.y) == (0, 0)
        assert (wide.end.x, wide.end.y) == (narrow.start.x, narrow.start.y)
        assert (narrow.end.x, narrow.end.y) == (nm(20), nm(4))
        line = LineString([(0, 0), (20, 4)])
        assert line.distance(Point(segment(wide).coords[-1])) < STRAIGHT_TOLERANCE

    @pytest.mark.usefixtures("coarse_optimizer")
    @pytest.mark.parametrize("holder", ["pad", "track"])
    def test_copper_at_a_width_change_keeps_both_ends_there(
        self, connect: Connector, holder: str
    ) -> None:
        tracks = width_change()
        board = CommitBoard(selection=list(tracks), tracks=list(tracks))
        if holder == "pad":
            board.pads = [kipy_pad("p", "SIG", *WIDTH_CHANGE)]
        else:
            board.tracks.append(
                kipy_track("u", "SIG", WIDTH_CHANGE, (12, 8), width=NARROW)
            )
        connect.to(board)

        run()

        # The narrow side, held at both ends, is straight and stays as drawn.
        assert sorted(board.deleted) == ["w1", "w2"]
        at = (nm(WIDTH_CHANGE[0]), nm(WIDTH_CHANGE[1]))
        assert at in ends(cast(list[kbt.Track], board.created))

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
    def test_a_stub_crossing_the_route_past_its_copper_is_left_behind(
        self,
        connect: Connector,
        reports: list[pipeline.RunReport],
        log: list[str],
    ) -> None:
        # Copper of one net does not interact, and the stub ends beyond the
        # route's copper, so nothing holds the two together: the route
        # straightens off the stub, which stays as drawn, two pieces of SIG.
        route = [
            kipy_track("r1", "SIG", (0, 0), (4, -1), width=WIDTH),
            kipy_track("r2", "SIG", (4, -1), (6, -1), width=WIDTH),
            kipy_track("r3", "SIG", (6, -1), (10, 0), width=WIDTH),
        ]
        stub = kipy_track("stub", "SIG", (5, -3), (5, -0.6), width=WIDTH)
        tracks = [*route, stub]
        board = connect.to(CommitBoard(selection=list(tracks), tracks=list(tracks)))

        run()

        (report,) = reports
        assert report.committed
        assert sorted(board.deleted) == ["r1", "r2", "r3"]
        assert board.selected_ids() == [
            "stub",
            *board.assigned_ids,
        ]
        created = cast(list[kbt.Track], board.created)
        points = [(p.x, p.y) for t in created for p in (t.start, t.end)]
        assert all(abs(y) < nm(0.01) for _, y in points)
        assert {(0, 0), (nm(10), 0)} <= set(points)
        # The fake board has no outline, which is the only warning.
        assert not any(
            line.startswith(("WARNING", "ERROR")) and "Edge.Cuts" not in line
            for line in log
        )

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_dry_run_prints_the_outcome_and_commits_nothing(
        self, connect: Connector, capsys: pytest.CaptureFixture[str]
    ) -> None:
        board = connect.to(l_board())

        run("--dry-run")

        assert board.calls == []
        assert board.selected_ids() == ["a", "b"]
        assert "Optimizer: converged" in capsys.readouterr().out

    @pytest.mark.usefixtures("coarse_optimizer")
    def test_a_failure_to_select_after_the_push_still_reports_committed(
        self, connect: Connector, reports: list[pipeline.RunReport], log: list[str]
    ) -> None:
        board = connect.to(l_board())
        board.select_error = ApiError("busy", code=ApiStatusCode.AS_BUSY)

        run()

        (report,) = reports
        assert report.committed
        lines = [line for line in log if "Could not select" in line]
        assert [line.split()[0] for line in lines] == ["DEBUG"]

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
                "--attraction=0.4",
                "--attraction-range=2.5",
                "--edge-clearance=0.5",
                "--hole-clearance=0.1",
                "--time-limit=30",
            )

        params = cast(EnergyParams, seen["params"])
        assert (params.repulsion, params.falloff_range) == (2.0, 0.4)
        assert (params.attraction, params.attraction_range) == (0.4, 2.5)
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
    def test_a_run_given_up_after_progress_says_so_and_commits(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        log: list[str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        board = connect.to(l_board())
        real = optimize_module.run_burst

        def failing(
            objective: Objective,
            start: FloatArray,
            revert: Revert,
            settings: OptimizerSettings,
            stop: Stop | None = None,
        ) -> Burst:
            return replace(
                real(objective, start, revert, settings, stop), abnormal=True
            )

        monkeypatch.setattr(optimize_module, "run_burst", failing)

        run()

        (outcome,) = capsys.readouterr().out.splitlines()
        assert outcome.startswith(f"Optimizer: {Termination.GAVE_UP.value} after ")
        assert "converged" not in outcome
        assert not any(
            line.startswith("WARNING") and "abnormal" in line for line in log
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
        monkeypatch.setattr(pipeline, "monotonic", BurstClock(monkeypatch))
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
                    "--settled-change=0.002",
                    "--stall-tolerance=0.01",
                    "--stall-window=5",
                    "--max-iterations=50",
                    "--burst-iterations=40",
                    "--lbfgs-ftol=1e-8",
                    "--lbfgs-gtol=1e-6",
                    "--time-limit=60",
                ]
            )
        )

        s = options.settings
        assert s.settled_change == 0.002
        assert (s.stall_tolerance, s.stall_window) == (0.01, 5)
        assert (s.max_iterations, s.burst_maxiter, s.time_limit) == (50, 40, 60.0)
        assert (s.lbfgs_ftol, s.lbfgs_gtol) == (1e-8, 1e-6)

    def test_stop_rules_default_to_the_optimizers(self) -> None:
        options = cli.smooth_options(cli.parse_args(["smooth"]))

        assert options.settings == DEFAULT_SETTINGS

    def test_settled_change_is_half_the_chord_tolerance(self) -> None:
        args = cli.parse_args(["smooth", "--chord-tolerance=0.04"])

        assert cli.smooth_options(args).settings.settled_change == 0.02

    def test_a_given_settled_change_wins_over_the_chord_tolerance(self) -> None:
        args = ["smooth", "--chord-tolerance=0.04", "--settled-change=0.003"]

        options = cli.smooth_options(cli.parse_args(args))

        assert options.settings.settled_change == 0.003


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

        assert "no tracks or vias" in exit_message(log)

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

    def test_selection_of_only_zero_length_tracks(
        self, connect: Connector, log: list[str]
    ) -> None:
        dots = [kipy_track("a", "SIG", (1, 1), (1, 1))]
        board = connect.to(CommitBoard(selection=list(dots), tracks=list(dots)))

        assert "no tracks of non-zero length" in exit_message(log)
        assert board.calls == []

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


ARC_MID = (5 + 5 * math.sqrt(0.5), 5 - 5 * math.sqrt(0.5))
"""On the 90° arc from (5, 0) to (10, 5) around (5, 5)."""


def quarter_arc(
    net: str = "SIG", *, locked: bool = False, down: float = 0.0
) -> kbt.ArcTrack:
    """The arc of ARC_MID, `down` mm lower."""
    return kipy_arc(
        "arc",
        net,
        start=(5, down),
        mid=(ARC_MID[0], ARC_MID[1] + down),
        end=(10, 5 + down),
        width=WIDTH,
        locked=locked,
    )


def selected(*tracks: kbt.Track | kbt.ArcTrack) -> CommitBoard:
    return CommitBoard(selection=list(tracks), tracks=list(tracks))


def track_arc_track(
    net: str = "SIG", down: float = 0.0
) -> list[kbt.Track | kbt.ArcTrack]:
    """A track, a 90° arc and a track, from (0, 0) to (10, 10), `down` mm
    lower."""
    return [
        kipy_track("in", net, (0, down), (5, down), width=WIDTH),
        quarter_arc(net, down=down),
        kipy_track("out", net, (10, 5 + down), (10, 10 + down), width=WIDTH),
    ]


@pytest.mark.usefixtures("coarse_optimizer")
class TestArcs:
    def test_track_arc_and_track_become_straight_tracks_in_one_commit(
        self, connect: Connector, reports: list[pipeline.RunReport]
    ) -> None:
        board = connect.to(selected(*track_arc_track()))

        run()

        assert board.calls == COMMIT_CALLS
        assert sorted(board.deleted) == ["arc", "in", "out"]
        assert board.created and all(type(t) is kbt.Track for t in board.created)
        (report,) = reports
        assert [c.tracks for c in report.chains] == [3]

    def test_dry_run_with_an_arc_ends_without_an_error(
        self, connect: Connector, reports: list[pipeline.RunReport]
    ) -> None:
        board = connect.to(selected(*track_arc_track()))

        run("--dry-run")

        assert board.calls == []
        (report,) = reports
        assert [c.changed for c in report.chains] == [True]

    def test_arc_alone_becomes_one_straight_track(self, connect: Connector) -> None:
        board = connect.to(selected(quarter_arc()))

        run()

        assert board.deleted == ["arc"]
        (track,) = cast(list[kbt.Track], board.created)
        assert ends([track]) == {(nm(5), nm(0)), (nm(10), nm(5))}

    def test_locked_arc_stays_and_holds_the_chains_at_its_ends(
        self, connect: Connector, reports: list[pipeline.RunReport]
    ) -> None:
        board = connect.to(
            selected(
                kipy_track("a1", "SIG", (0, 3), (0, 0), width=WIDTH),
                kipy_track("a2", "SIG", (0, 0), (5, 0), width=WIDTH),
                quarter_arc(locked=True),
                kipy_track("c1", "SIG", (10, 5), (10, 10), width=WIDTH),
                kipy_track("c2", "SIG", (10, 10), (15, 10), width=WIDTH),
            )
        )

        run()

        assert sorted(board.deleted) == ["a1", "a2", "c1", "c2"]
        created = cast(list[kbt.Track], board.created)
        assert {(nm(5), nm(0)), (nm(10), nm(5))} <= ends(created)
        (report,) = reports
        assert [c.changed for c in report.chains] == [True, True]

    def test_chain_with_an_arc_left_as_it_was_is_not_committed(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        reports: list[pipeline.RunReport],
    ) -> None:
        board = connect.to(selected(*l_chain(), *track_arc_track("SIG2", down=20)))
        real: Callable[..., list[FloatArray]] = pipeline.simplify_chains

        def arc_chain_as_given(
            chains: Sequence[ChainInput], *args: object, **kwargs: object
        ) -> list[FloatArray]:
            return [
                c.centreline.points if c.centreline.net == "SIG2" else s
                for c, s in zip(chains, real(chains, *args, **kwargs), strict=True)
            ]

        monkeypatch.setattr(pipeline, "simplify_chains", arc_chain_as_given)

        run()

        assert sorted(board.deleted) == list(OUTER)
        created = cast(list[kbt.Track], board.created)
        assert {t.net.name for t in created} == {"SIG"}
        (report,) = reports
        assert [c.changed for c in report.chains] == [True, False]

    def test_arc_split_at_a_tee_is_replaced_whole_when_one_side_changes(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 2° past the vertex at the arc's middle, partway along a piece.
        angle = math.radians(-45 + 2)
        on_arc = (5 + 5 * math.cos(angle), 5 + 5 * math.sin(angle))
        tee = kipy_track("tee", "SIG", (12, -2), on_arc, width=WIDTH)
        board = connect.to(selected(*track_arc_track(), tee))
        real: Callable[..., list[FloatArray]] = pipeline.simplify_chains

        def out_side_as_given(
            chains: Sequence[ChainInput], *args: object, **kwargs: object
        ) -> list[FloatArray]:
            given = [c.centreline.points for c in chains]
            return [
                points if (10, 10) in (tuple(points[0]), tuple(points[-1])) else s
                for points, s in zip(given, real(chains, *args, **kwargs), strict=True)
            ]

        monkeypatch.setattr(pipeline, "simplify_chains", out_side_as_given)

        run()

        assert board.calls == COMMIT_CALLS
        assert board.deleted.count("arc") == 1
        assert {"in", "out"} <= set(board.deleted)
        created = cast(list[kbt.Track], board.created)
        assert {(nm(0), nm(0)), (nm(10), nm(5)), (nm(10), nm(10))} <= ends(created)

    def test_moving_via_carries_the_end_of_an_arc(self, connect: Connector) -> None:
        arc = kipy_arc(
            "arc",
            start=(0, 0),
            mid=(4 * math.sqrt(0.5), 4 - 4 * math.sqrt(0.5)),
            end=VIA,
            width=WIDTH,
        )
        route = [arc, *detour_back()]
        via = kipy_via("v", "SIG", *VIA)
        board = connect.to(
            CommitBoard(selection=[*route, via], tracks=list(route), vias=[via])
        )

        run()

        assert sorted(board.deleted) == ["arc", "e", "f", "v"]
        (moved_via,) = created_vias(board)
        moved = (moved_via.position.x, moved_via.position.y)
        assert moved != VIA_NM
        assert moved in ends(created_on(board, BL.BL_F_Cu))
        assert moved in ends(created_on(board, BL.BL_B_Cu))


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

        assert board.calls == COMMIT_CALLS
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
        assert board.selection == tracks
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

    # Chains pressed into each other's clearance settle well before the
    # iteration cap. Measured: 3.
    SETTLED_PRESSED = 20

    def test_chains_starting_inside_each_other_s_clearance_commit_unwarned(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch, log: list[str]
    ) -> None:
        # 0.3 mm from the L's corner, inside clearance plus both half widths.
        across = [kipy_track("x", "SIG2", (4.3, -1), (4.3, 1), width=WIDTH)]
        board = connect.to(off_grid(two_l_board(inner=across)))
        results: list[RunResult] = []
        optimize = at(COARSE)

        def kept(*args: object, **kwargs: object) -> RunResult:
            results.append(optimize(*args, **kwargs))
            return results[-1]

        monkeypatch.setattr(pipeline, "optimize_chains", kept)

        run()

        # The chains pressed into each other's clearance settle.
        (result,) = results
        assert result.termination is Termination.CONVERGED
        assert result.iterations <= self.SETTLED_PRESSED
        # Only the board's missing outline warns.
        assert not [
            line
            for line in log
            if line.startswith("WARNING") and "Edge.Cuts" not in line
        ]
        assert "push_commit" in board.calls

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
        assert report.abnormal == ()
        assert report.termination in (Termination.CONVERGED, Termination.ITERATION_CAP)
        assert report.iterations > 0
        assert (report.committed, report.refusal) == (False, None)
        assert board.calls == []
        gap = LineString(simplified[0]).distance(LineString(simplified[1]))
        assert gap >= CLEARANCE + WIDTH - NM_ROUNDING

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


Z_BUS_TRACKS = 10
Z_BUS_PITCH = 0.6
Z_BUS_WIDTH = 0.2
Z_BUS_CLEARANCE = 0.2


def z_bus_board() -> CommitBoard:
    """The bus of issue 106: Z-shaped tracks at equal pitch, each centred
    between its neighbours."""
    tracks: list[kbt.Track] = []
    for i in range(Z_BUS_TRACKS):
        y = i * Z_BUS_PITCH
        corners = [
            (0.0, y),
            (10.0 - y, y),
            (16.0 - y, y + 6.0),
            (30.0, y + 6.0),
        ]
        tracks += [
            kipy_track(f"S{i}.{k}", f"S{i}", start, end, width=Z_BUS_WIDTH)
            for k, (start, end) in enumerate(itertools.pairwise(corners))
        ]
    bus = netclass("BUS", Z_BUS_CLEARANCE)
    return CommitBoard(
        selection=list(tracks),
        tracks=list(tracks),
        netclasses={f"S{i}": bus for i in range(Z_BUS_TRACKS)},
    )


Z_BUS_SPACING = 5.0
"""Vertex spacing for the bus, mm: coarse, so that the run converges
within a second, and still several vertices to each of its edges, which
the nearest neighbour alone left on the tie: there every problem gave
up."""


def test_an_equally_spaced_bus_converges(
    connect: Connector,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    reports: list[pipeline.RunReport],
) -> None:
    """The energy is differentiable at the tie, so no problem gives up and
    every track of the bus shortens."""
    monkeypatch.setattr(
        pipeline, "optimize_chains", at(replace(COARSE, spacing=Z_BUS_SPACING))
    )
    connect.to(z_bus_board())

    run("--dry-run")

    (outcome,) = capsys.readouterr().out.splitlines()
    assert outcome.startswith(f"Optimizer: {Termination.CONVERGED.value} ")
    (report,) = reports
    assert len(report.chains) == Z_BUS_TRACKS
    for chain in report.chains:
        assert chain.changed
        assert chain.after.length < chain.before.length


def triangle_loop(net: str = "SIG") -> list[kbt.Track]:
    """A closed triangle through (0, 0), (9, 0) and (9, 3)."""
    return [
        kipy_track("a", net, (0, 0), (9, 0), width=WIDTH),
        kipy_track("b", net, (9, 0), (9, 3), width=WIDTH),
        kipy_track("c", net, (9, 3), (0, 0), width=WIDTH),
    ]


IN_TRIANGLE = (7.75, 1.25)
"""Near the centre of the circle inside `triangle_loop`, 1.26 mm across."""


TRIANGLE_PADS = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))
"""Pads of SIG on three corners of a square loop."""


def pad_box(x: float, y: float) -> BaseGeometry:
    h = PAD_HALF_SIZE
    return box(x - h, y - h, x + h, y + h)


def assert_round_the_pad(path: LineString, loop: Sequence[kbt.Track]) -> None:
    """`path` goes round the pad at `IN_TRIANGLE` at its clearance and is
    shorter than `loop`."""
    pad = pad_box(*IN_TRIANGLE)
    assert shapely.Polygon(path.coords).contains(pad)
    assert path.distance(pad) >= CLEARANCE + WIDTH / 2 - NM_ROUNDING
    assert path.length < sum(segment(t).length for t in loop)


def created_path(board: CommitBoard) -> LineString:
    """The created tracks merged into one line, which they must form."""
    tracks = [t for t in board.created if isinstance(t, kbt.Track)]
    merged = linemerge([segment(t) for t in tracks])
    assert isinstance(merged, LineString)
    return merged


@pytest.mark.usefixtures("coarse_optimizer")
class TestLoopsAndDots:
    @pytest.mark.parametrize(
        ("held_by", "at"),
        [
            ([kipy_track("t", "SIG", (-3, 0), (0, 0), width=WIDTH)], (0.0, 0.0)),
            ([kipy_pad("corner", "SIG", 0, 0)], (0.0, 0.0)),
        ],
        ids=["track", "pad"],
    )
    def test_a_loop_is_cut_where_its_net_touches_it_and_smoothed_round_a_pad(
        self,
        connect: Connector,
        held_by: list[kbt.BoardItem],
        at: Point2,
    ) -> None:
        loop = triangle_loop()
        board = connect.to(
            CommitBoard(
                selection=list(loop),
                tracks=[*loop, *(t for t in held_by if isinstance(t, kbt.Track))],
                pads=[
                    kipy_pad("pad", "GND", *IN_TRIANGLE),
                    *(p for p in held_by if isinstance(p, kbt.Pad)),
                ],
            )
        )

        run()

        assert sorted(board.deleted) == ["a", "b", "c"]
        path = created_path(board)
        assert path.is_closed
        assert at in path.coords[:]
        assert_round_the_pad(path, loop)

    def test_a_loop_through_a_selected_via_moves_it_and_is_smoothed_round_a_pad(
        self, connect: Connector, log: list[str]
    ) -> None:
        loop = triangle_loop()
        via = kipy_via("v", "SIG", 9, 0)
        board = connect.to(
            CommitBoard(
                selection=[*loop, via],
                tracks=list(loop),
                vias=[via],
                pads=[kipy_pad("pad", "GND", *IN_TRIANGLE)],
            )
        )

        run()

        assert sorted(board.deleted) == ["a", "b", "c", "v"]
        (moved,) = created_vias(board)
        at = (moved.position.x, moved.position.y)
        assert at != (nm(9), nm(0))
        path = created_path(board)
        assert path.is_closed
        assert (at[0] / NM_PER_MM, at[1] / NM_PER_MM) in path.coords[:]
        assert_round_the_pad(path, loop)
        # The first extraction already cuts the loop at the via.
        assert not any("Leaving out" in line for line in log)

    def test_a_loop_off_a_junction_moves_with_it(self, connect: Connector) -> None:
        tracks = [
            kipy_track("a", "SIG", (0, 0), (5, 0), width=WIDTH),
            kipy_track("b", "SIG", (5, 0), (10, 0), width=WIDTH),
            kipy_track("c", "SIG", (5, 0), (7, 2), width=WIDTH),
            kipy_track("d", "SIG", (7, 2), (5, 4), width=WIDTH),
            kipy_track("e", "SIG", (5, 4), (3, 2), width=WIDTH),
            kipy_track("f", "SIG", (3, 2), (5, 0), width=WIDTH),
        ]
        board = connect.to(
            CommitBoard(
                selection=list(tracks),
                tracks=list(tracks),
                pads=[kipy_pad("pad", "GND", 5, 2)],
            )
        )

        run()

        assert sorted(board.deleted) == ["a", "b", "c", "d", "e", "f"]
        degree = Counter(
            (p.x, p.y)
            for t in cast(list[kbt.Track], board.created)
            for p in (t.start, t.end)
        )
        (junction,) = [p for p, n in degree.items() if n == 4]
        assert {p for p, n in degree.items() if n == 1} == {
            (nm(0), nm(0)),
            (nm(10), nm(0)),
        }
        assert set(degree.values()) == {1, 2, 4}
        # The loop round the pad pulls the junction off the straight route.
        assert junction[1] > nm(0.1)

    def test_a_loop_through_three_pads_of_its_net_settles_into_their_triangle(
        self, connect: Connector
    ) -> None:
        loop = [
            kipy_track("a", "SIG", (0, 0), (10, 0), width=WIDTH),
            kipy_track("b", "SIG", (10, 0), (10, 10), width=WIDTH),
            kipy_track("c", "SIG", (10, 10), (0, 10), width=WIDTH),
            kipy_track("d", "SIG", (0, 10), (0, 0), width=WIDTH),
        ]
        pads = [kipy_pad(f"p{k}", "SIG", *at) for k, at in enumerate(TRIANGLE_PADS)]
        board = connect.to(
            CommitBoard(selection=list(loop), tracks=list(loop), pads=pads)
        )

        run()

        assert sorted(board.deleted) == ["a", "b", "c", "d"]
        corners = {(nm(x), nm(y)) for x, y in TRIANGLE_PADS}
        created = cast(list[kbt.Track], board.created)
        assert {(p.x, p.y) for t in created for p in (t.start, t.end)} == corners
        assert len(created) == len(corners)

    @pytest.mark.parametrize("touched", [False, True], ids=["alone", "touched"])
    def test_a_loop_closed_by_an_arc_is_smoothed_round_a_pad(
        self, connect: Connector, touched: bool
    ) -> None:
        # The arc bulges below (0, 0)-(9, 0), round the pad of GND.
        arc = kipy_arc("arc", "SIG", (0, 0), (4.5, 3), (9, 0), width=WIDTH)
        back = kipy_track("back", "SIG", (9, 0), (0, 0), width=WIDTH)
        touch = kipy_track("t", "SIG", (-3, 0), (0, 0), width=WIDTH)
        board = connect.to(
            CommitBoard(
                selection=[arc, back],
                tracks=[arc, back, *([touch] if touched else [])],
                pads=[kipy_pad("pad", "GND", 4.5, 1.5)],
            )
        )

        run()

        assert sorted(board.deleted) == ["arc", "back"]
        path = created_path(board)
        assert path.is_closed and (0.0, 0.0) in path.coords[:]
        pad = pad_box(4.5, 1.5)
        assert shapely.Polygon(path.coords).contains(pad)
        assert path.distance(pad) >= CLEARANCE + WIDTH / 2 - NM_ROUNDING

    def test_a_zero_length_track_beside_a_chain_stays(
        self, connect: Connector, log: list[str]
    ) -> None:
        dot = kipy_track("dot", "SIG", (8, 8), (8, 8), width=WIDTH)
        board = l_board()
        board.selection.append(dot)
        board.tracks.append(dot)
        connect.to(board)

        run()

        assert sorted(board.deleted) == ["a", "b"]
        (left_out,) = [line for line in log if "Leaving out" in line]
        assert left_out.startswith("DEBUG") and "dot" in left_out

    def test_a_zero_length_track_on_a_chain_goes_with_it(
        self, connect: Connector
    ) -> None:
        dot = kipy_track("dot", "SIG", (4, 0), (4, 0), width=WIDTH)
        board = l_board()
        board.selection.append(dot)
        board.tracks.append(dot)
        connect.to(board)

        run()

        assert sorted(board.deleted) == ["a", "b", "dot"]

    def test_a_dot_of_another_width_left_out_pins_the_chain_it_lies_on(
        self, connect: Connector
    ) -> None:
        dot = kipy_track("dot", "SIG", (2, 0), (2, 0), width=0.4)
        board = l_board()
        board.selection.append(dot)
        board.tracks.append(dot)
        connect.to(board)

        run()

        assert sorted(board.deleted) == ["a", "b"]
        assert (nm(2), nm(0)) in ends(cast(list[kbt.Track], board.created))


VIA = (4.0, 4.0)
VIA_DIAMETER = 0.6
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
        vias=[kipy_via("v", "SIG", *VIA, VIA_DIAMETER)],
        pads=[kipy_pad("pad", "GND", *pad, layers=pad_layers)],
    )


def created_on(board: CommitBoard, layer: BL.ValueType) -> list[kbt.Track]:
    return [t for t in board.created if isinstance(t, kbt.Track) and t.layer == layer]


def ends(tracks: Sequence[kbt.Track]) -> set[tuple[int, int]]:
    return {(p.x, p.y) for t in tracks for p in (t.start, t.end)}


VIA_NM = (nm(VIA[0]), nm(VIA[1]))
"""The via's centre in KiCad's nanometres, where the tracks ending at it
must end exactly, or dragging the via in KiCad leaves them behind."""


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

        assert board.calls == COMMIT_CALLS
        assert sorted(board.deleted) == ["a", "b", "e", "f"]
        front, back = created_on(board, BL.BL_F_Cu), created_on(board, BL.BL_B_Cu)
        assert len(front) + len(back) == len(board.created)
        assert {(0, 0), VIA_NM} <= ends(front)
        assert {VIA_NM, (nm(8), nm(8))} <= ends(back)

    def test_inner_layer_route_is_committed_on_it(self, connect: Connector) -> None:
        board = connect.to(via_route_board(back_layer=BL.BL_In1_Cu))

        run()

        assert sorted(board.deleted) == ["a", "b", "e", "f"]
        front, inner = created_on(board, BL.BL_F_Cu), created_on(board, BL.BL_In1_Cu)
        assert len(front) + len(inner) == len(board.created)
        assert {VIA_NM, (nm(8), nm(8))} <= ends(inner)

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
        assert VIA_NM in ends(created_on(board, BL.BL_F_Cu))
        assert VIA_NM in ends(created_on(board, BL.BL_B_Cu))

    def test_a_selected_locked_via_costs_no_reads(self, connect: Connector) -> None:
        # A through-hole pad of the via's net lists layers no run reads.
        pth = kipy_pad("pth", "SIG", 0, 8, layers=list(iter_copper_layers()))
        board = via_route_board()
        board.pads.append(pth)
        board.vias = [kipy_via("v", "SIG", *VIA, locked=True)]
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
        assert VIA_NM in ends(created_on(board, BL.BL_F_Cu))
        assert VIA_NM in ends(created_on(board, BL.BL_B_Cu))

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


INSIDE_PAD = (4.0, 4.65)
"""An other-net pad whose edge is 0.1 mm from the ring of the via at VIA,
inside its 0.2 mm clearance."""


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


PUSHED_STITCHING = (2.0, 3.0)


def pushed_stitching_board() -> CommitBoard:
    """A selected straight SIG track along y = 0, far from the selected GND
    stitching via `s` at `PUSHED_STITCHING` in a GND plane, and an
    unselected PWR track within range above the via that pushes it down."""
    straight = [kipy_track("a", "SIG", (0, 0), (4, 0), width=WIDTH)]
    pwr = kipy_track("p", "PWR", (0, 3.8), (4, 3.8), width=WIDTH)
    stitching = kipy_via("s", "GND", *PUSHED_STITCHING)
    plane = kipy_rect(-2, -3, 10, 10)
    return CommitBoard(
        selection=[*straight, stitching],
        tracks=[*straight, pwr],
        vias=[stitching],
        zones=[kipy_copper_zone("plane", "GND", plane, [BL.BL_F_Cu, BL.BL_B_Cu])],
    )


def created_vias(board: CommitBoard) -> list[kbt.Via]:
    return [i for i in board.created if isinstance(i, kbt.Via)]


THROUGH = (5.0, 6.0)
"""Where the route of `through_via_board` turns at its via."""


def through_via_board(
    down: kbt.Track | None = None, unselected: kbt.Track | None = None
) -> CommitBoard:
    """SIG on F.Cu from (0, 0), bent, through a selected via at `THROUGH`
    to (10, 0), with `down` selected and `unselected` on the board too."""
    route = [
        kipy_track("a", "SIG", (0, 0), (1, 3), width=WIDTH),
        kipy_track("b", "SIG", (1, 3), THROUGH, width=WIDTH),
        kipy_track("c", "SIG", THROUGH, (10, 0), width=WIDTH),
        *([down] if down else []),
    ]
    via = kipy_via("v", "SIG", *THROUGH)
    return CommitBoard(
        selection=[*route, via],
        tracks=[*route, *([unselected] if unselected else [])],
        vias=[via],
    )


@pytest.mark.usefixtures("coarse_optimizer")
class TestMovingVias:
    def test_moves_the_via_with_its_chains_in_one_commit(
        self, connect: Connector
    ) -> None:
        board = connect.to(moving_via_board())

        run()

        assert board.calls == COMMIT_CALLS
        assert sorted(board.deleted) == ["a", "b", "e", "f", "v"]
        (via,) = created_vias(board)
        moved = (via.position.x, via.position.y)
        assert moved[1] < nm(1)
        assert moved in ends(created_on(board, BL.BL_F_Cu))
        assert moved in ends(created_on(board, BL.BL_B_Cu))

    def test_a_via_a_group_track_ends_at_stays_and_the_run_commits(
        self, connect: Connector, log: list[str]
    ) -> None:
        # Moved, the via would drag the group's track, which the commit's
        # re-read then finds changed.
        route = [*l_chain(), *detour_back()]
        via = kipy_via("v", "SIG", *VIA)
        group = kipy_group("Tuning Pattern", route[:2])
        board = connect.to(
            CommitBoard(
                selection=[group, *route[2:], via],
                tracks=list(route),
                vias=[via],
                groups=[group],
            )
        )

        run("--verbose")

        assert board.calls == COMMIT_CALLS
        assert not created_vias(board)
        assert not {"a", "b", "v"} & {*board.deleted}
        assert any("a track of a selected group ends at it" in line for line in log)

    def test_a_via_at_a_width_change_carries_both_ends(
        self, connect: Connector
    ) -> None:
        tracks = width_change()
        via = kipy_via("v", "SIG", *WIDTH_CHANGE)
        board = connect.to(
            CommitBoard(selection=[*tracks, via], tracks=list(tracks), vias=[via])
        )

        run()

        assert sorted(board.deleted) == ["n", "v", "w1", "w2"]
        (moved,) = created_vias(board)
        at = (moved.position.x, moved.position.y)
        assert math.dist(at, (nm(WIDTH_CHANGE[0]), nm(WIDTH_CHANGE[1]))) > nm(1)
        created = created_on(board, BL.BL_F_Cu)
        for width in (WIDE, NARROW):
            assert at in ends([t for t in created if t.width == nm(width)])

    @pytest.mark.parametrize(
        "via_type", [ViaType.VT_BLIND, ViaType.VT_MICRO], ids=["blind", "micro"]
    )
    def test_a_blind_or_micro_via_moves_keeping_its_type_and_span(
        self, connect: Connector, log: list[str], via_type: ViaType.ValueType
    ) -> None:
        # Across its way on B.Cu, which it does not reach, lies a GND track
        # that would hold a through via 2 mm short of where it goes.
        inner = [
            kipy_track("e", "SIG", VIA, (8, 4), width=WIDTH, layer=BL.BL_In1_Cu),
            kipy_track("f", "SIG", (8, 4), (8, 0), width=WIDTH, layer=BL.BL_In1_Cu),
        ]
        across = kipy_track("g", "GND", (0, 2), (8, 2), width=WIDTH, layer=BL.BL_B_Cu)
        span = (BL.BL_F_Cu, BL.BL_In1_Cu)
        via = kipy_via("v", "SIG", *VIA, via_type=via_type, span=span)
        route = [*l_chain(), *inner]
        board = CommitBoard(
            selection=[*route, via], tracks=[*route, across], vias=[via]
        )
        board.enabled_layers = [BL.BL_F_Cu, BL.BL_In1_Cu, BL.BL_B_Cu]
        connect.to(board)

        run()

        assert "stays fixed" not in "".join(log)
        assert sorted(board.deleted) == ["a", "b", "e", "f", "v"]
        (moved,) = created_vias(board)
        assert moved.type == via_type
        drill = moved.padstack.drill
        assert (drill.start_layer, drill.end_layer) == span
        at = (moved.position.x, moved.position.y)
        assert at[1] < nm(1)
        assert at in ends(created_on(board, BL.BL_F_Cu))
        assert at in ends(created_on(board, BL.BL_In1_Cu))

    def test_a_via_a_chain_runs_through_moves_carrying_both_sides(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = connect.to(through_via_board())

        run()

        assert not any("stays fixed" in line for line in log)
        assert sorted(board.deleted) == ["a", "b", "c", "v"]
        (moved,) = created_vias(board)
        at = (moved.position.x, moved.position.y)
        # Straightened, the route pulls the via down to its far ends' line.
        assert at[1] < nm(1)
        on_front = created_on(board, BL.BL_F_Cu)
        assert [(p.x, p.y) for t in on_front for p in (t.start, t.end)].count(at) == 2

    def test_a_via_a_chain_runs_through_carries_a_track_of_another_layer(
        self, connect: Connector
    ) -> None:
        down = kipy_track("d", "SIG", THROUGH, (5, 12), width=WIDTH, layer=BL.BL_B_Cu)
        board = connect.to(through_via_board(down))

        run()

        assert sorted(board.deleted) == ["a", "b", "c", "d", "v"]
        (moved,) = created_vias(board)
        at = (moved.position.x, moved.position.y)
        assert math.dist(at, (nm(THROUGH[0]), nm(THROUGH[1]))) > nm(1)
        on_front = created_on(board, BL.BL_F_Cu)
        assert [(p.x, p.y) for t in on_front for p in (t.start, t.end)].count(at) == 2
        assert at in ends(created_on(board, BL.BL_B_Cu))

    def test_a_via_a_chain_runs_through_stays_with_another_track_on_it(
        self, connect: Connector, log: list[str]
    ) -> None:
        stub = kipy_track("s", "SIG", THROUGH, (5, 12), width=WIDTH)
        board = connect.to(through_via_board(unselected=stub))

        run()

        assert created_vias(board) == []
        # Not cut at the via that stays, the route is one chain, replaced
        # whole, and keeps a vertex on the via.
        assert sorted(board.deleted) == ["a", "b", "c"]
        assert (nm(THROUGH[0]), nm(THROUGH[1])) in ends(created_on(board, BL.BL_F_Cu))
        (stays,) = [line for line in log if "Via v" in line and "stays fixed" in line]
        assert stays.startswith("DEBUG") and "track s on F.Cu" in stays

    def test_a_stitching_via_the_chain_crowds_moves_in_the_same_commit(
        self, connect: Connector, log: list[str]
    ) -> None:
        # Inside the L, where the chain cutting its corner presses on it.
        start = (3.0, 1.0)
        stitching = kipy_via("s", "GND", *start)
        plane = kipy_rect(-2, -3, 10, 10)
        board = connect.to(
            CommitBoard(
                selection=[*l_chain(), stitching],
                tracks=[*l_chain()],
                vias=[stitching],
                zones=[
                    kipy_copper_zone("plane", "GND", plane, [BL.BL_F_Cu, BL.BL_B_Cu])
                ],
            )
        )

        run()

        assert "stays fixed" not in "".join(log)
        assert board.calls == COMMIT_CALLS
        assert sorted(board.deleted) == ["a", "b", "s"]
        (via,) = created_vias(board)
        assert via.net.name == "GND"
        moved = (via.position.x / 1e6, via.position.y / 1e6)
        assert math.dist(moved, start) > 0.05

    def test_a_stitching_via_moves_though_no_chain_changes(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = connect.to(pushed_stitching_board())

        run()

        assert "stays fixed" not in "".join(log)
        assert "Nothing to do" not in "".join(log)
        assert board.calls == COMMIT_CALLS
        assert board.deleted == ["s"]
        (via,) = board.created
        assert isinstance(via, kbt.Via)
        assert via.position.y / 1e6 < PUSHED_STITCHING[1] - 0.05

    @pytest.mark.parametrize("still", ["unselected", "locked"])
    @pytest.mark.parametrize(
        ("args", "pushed"), [(("--attraction=0",), False), ((), True)]
    )
    def test_a_ground_via_of_its_net_that_stands_still_pushes_it_with_attraction(
        self,
        connect: Connector,
        log: list[str],
        still: str,
        args: tuple[str, ...],
        pushed: bool,
    ) -> None:
        # Within range of the stitching via, beyond their touching distance;
        # nothing else is in range. Without attraction, vias of one net only
        # keep hole-to-hole; with it, ground vias of one net also repel.
        start = (2.0, 3.0)
        straight = [kipy_track("a", "SIG", (0, 0), (4, 0), width=WIDTH)]
        stitching = kipy_via("s", "GND", *start)
        other = kipy_via(
            "u", "GND", start[0] + 1.27, start[1], locked=still == "locked"
        )
        plane = kipy_rect(-2, -3, 10, 10)
        board = connect.to(
            CommitBoard(
                selection=[
                    *straight,
                    stitching,
                    *([other] if still == "locked" else []),
                ],
                tracks=[*straight],
                vias=[stitching, other],
                zones=[
                    kipy_copper_zone("plane", "GND", plane, [BL.BL_F_Cu, BL.BL_B_Cu])
                ],
            )
        )

        run(*args)

        if not pushed:
            assert board.calls == []
            assert "Nothing to do" in "".join(log)
            return
        assert board.calls == COMMIT_CALLS
        (via,) = board.created
        assert isinstance(via, kbt.Via)
        assert via.position.x / 1e6 < start[0] - 0.1

    def test_a_dry_run_moving_only_a_stitching_via_commits_nothing(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = connect.to(pushed_stitching_board())

        run("--dry-run")

        assert board.calls == []
        assert "Dry run: nothing committed" in "".join(log)
        assert "Nothing to do" not in "".join(log)

    def test_a_routing_via_and_a_stitching_via_both_move(
        self, connect: Connector, log: list[str]
    ) -> None:
        # Beside the route, an unselected PWR track within range above the
        # stitching via pushes it down.
        board = moving_via_board()
        stitching = kipy_via("s", "GND", 12.0, 3.0)
        board.vias.append(stitching)
        board.selection.append(stitching)
        board.tracks.append(kipy_track("p", "PWR", (10, 3.8), (14, 3.8), width=WIDTH))
        plane = kipy_rect(-2, -3, 16, 10)
        board.zones = [
            kipy_copper_zone("plane", "GND", plane, [BL.BL_F_Cu, BL.BL_B_Cu])
        ]
        connect.to(board)

        run()

        assert board.calls == COMMIT_CALLS
        assert sorted(board.deleted) == ["a", "b", "e", "f", "s", "v"]
        created = {v.net.name: v for v in created_vias(board)}
        assert sorted(created) == ["GND", "SIG"]
        assert created["SIG"].position.y < nm(1)
        assert "stays fixed" not in "".join(log)

    @pytest.mark.parametrize("unfilled", ["pwr pour", "pour keepout"])
    def test_a_via_in_a_zone_of_its_net_moves_within_its_region_and_commits(
        self, connect: Connector, log: list[str], unfilled: str
    ) -> None:
        # Without the zone, the via ends below y = 1, in the part of the
        # outline that the higher-priority PWR zone or the keepout leaves
        # unfilled.
        board = moving_via_board()
        area = kipy_rect(-2, -3, 10, 2.5)
        plane = kipy_copper_zone(
            "plane", "SIG", kipy_rect(-2, -3, 10, 10), [BL.BL_In1_Cu]
        )
        plane.clearance = nm(0.2)
        if unfilled == "pwr pour":
            other = kipy_copper_zone("pwr", "PWR", area, [BL.BL_In1_Cu])
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
        assert board.calls == COMMIT_CALLS
        (via,) = created_vias(board)
        y = via.position.y / 1e6
        assert refill_top + KICAD_MAX_ERROR - NM_ROUNDING <= y < VIA[1]

    def test_the_via_keeps_the_zone_clearance_from_other_copper_in_it(
        self, connect: Connector, log: list[str]
    ) -> None:
        # The OTHER track across the via's way up crosses the plane, whose
        # refill keeps 0.5 mm from it, more than the netclass clearance.
        board = moving_via_board()
        other = kipy_track("o", "OTHER", (0, 1), (10, 1), layer=BL.BL_In1_Cu)
        board.tracks = [*board.tracks, other]
        plane = kipy_copper_zone(
            "plane", "SIG", kipy_rect(0, 0, 10, 10), [BL.BL_In1_Cu], min_width=0.25
        )
        plane.clearance = nm(0.5)
        board.zones = [plane]
        board.enabled_layers = [BL.BL_F_Cu, BL.BL_In1_Cu, BL.BL_B_Cu]
        connect.to(board)

        run()

        assert "stays fixed" not in "".join(log)
        (via,) = created_vias(board)
        y = via.position.y / 1e6
        kept = 1 + 0.25 / 2 + 0.5 + 0.25 / 2 + KICAD_MAX_ERROR
        assert kept - INSIDE_TOLERANCE - NM_ROUNDING <= y < kept + 0.1

    def test_the_via_keeps_the_zone_clearance_from_a_route_moving_with_it(
        self, connect: Connector, log: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The selected OTHER route on the plane's layer bows toward the via
        # and straightens across its way up; the refill keeps 0.5 mm from
        # it wherever it ends. The shapes then wander by tens of µm a burst,
        # which the settled change lets end.
        monkeypatch.setattr(
            pipeline, "optimize_chains", at(replace(COARSE, settled_change=0.05))
        )
        board = moving_via_board()
        route = [
            kipy_track("o1", "OTHER", (0, 1.5), (4, 2.2), layer=BL.BL_In1_Cu),
            kipy_track("o2", "OTHER", (4, 2.2), (8, 1.5), layer=BL.BL_In1_Cu),
        ]
        board.selection = [*board.selection, *route]
        board.tracks = [*board.tracks, *route]
        plane = kipy_copper_zone(
            "plane", "SIG", kipy_rect(0, 0, 10, 10), [BL.BL_In1_Cu], min_width=0.25
        )
        plane.clearance = nm(0.5)
        board.zones = [plane]
        board.enabled_layers = [BL.BL_F_Cu, BL.BL_In1_Cu, BL.BL_B_Cu]
        connect.to(board)

        run("--repulsion", "0")

        assert "stays fixed" not in "".join(log)
        (via,) = created_vias(board)
        centre = Point(via.position.x / 1e6, via.position.y / 1e6)
        (other,) = {t.net.name for t in created_on(board, BL.BL_In1_Cu)}
        assert other == "OTHER"
        gap = unary_union(
            [segment(t) for t in created_on(board, BL.BL_In1_Cu)]
        ).distance(centre)
        kept = 0.25 / 2 + 0.5 + 0.25 / 2 + KICAD_MAX_ERROR
        assert kept > 0.25 / 2 + 0.2 + 0.3  # the netclass spacing
        assert kept - INSIDE_TOLERANCE - NM_ROUNDING <= gap < kept + 0.1

    def test_a_via_starting_inside_the_zone_clearance_moves_out_of_it(
        self, connect: Connector
    ) -> None:
        # The selected OTHER route bends 0.625 mm from the via, at the
        # netclass spacing and inside the plane's 0.5 mm zone clearance plus
        # margin.
        board = moving_via_board()
        bend = (VIA[0], VIA[1] + 0.3 + 0.2 + 0.25 / 2)
        route = [
            kipy_track("o1", "OTHER", (1, bend[1]), bend, layer=BL.BL_In1_Cu),
            kipy_track("o2", "OTHER", bend, (7, 5.5), layer=BL.BL_In1_Cu),
        ]
        board.selection = [*board.selection, *route]
        board.tracks = [*board.tracks, *route]
        plane = kipy_copper_zone(
            "plane", "SIG", kipy_rect(0, 0, 10, 10), [BL.BL_In1_Cu], min_width=0.25
        )
        plane.clearance = nm(0.5)
        board.zones = [plane]
        board.enabled_layers = [BL.BL_F_Cu, BL.BL_In1_Cu, BL.BL_B_Cu]
        connect.to(board)

        run()

        assert board.calls == COMMIT_CALLS
        assert {"v", "o1", "o2"} <= set(board.deleted)
        (via,) = created_vias(board)
        centre = Point(via.position.x / 1e6, via.position.y / 1e6)
        gap = unary_union(
            [segment(t) for t in created_on(board, BL.BL_In1_Cu)]
        ).distance(centre)
        kept = 0.25 / 2 + 0.5 + 0.25 / 2 + KICAD_MAX_ERROR
        assert gap >= kept - INSIDE_TOLERANCE - NM_ROUNDING

    def test_a_via_at_the_edge_of_a_zone_stays_naming_the_zone(
        self, connect: Connector, log: list[str]
    ) -> None:
        # The edge zone ends 0.1 mm below the via's centre, short of the
        # 0.13 mm that half its minimum width and the arc error ask for: the
        # via may be connected now, and moving could lose that. The plane
        # holds it deep inside.
        board = moving_via_board()
        board.zones = [
            kipy_copper_zone("plane", "SIG", kipy_rect(0, 0, 10, 10), [BL.BL_In2_Cu]),
            kipy_copper_zone(
                "edge",
                "SIG",
                kipy_rect(0, 3.9, 10, 10),
                [BL.BL_In1_Cu],
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
        assert VIA_NM in ends(created_on(board, BL.BL_F_Cu))

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

    def test_a_via_starting_inside_its_clearance_moves_out_of_it(
        self,
        connect: Connector,
        log: list[str],
        reports: list[pipeline.RunReport],
    ) -> None:
        # The pad's edge is 0.1 mm from the via's ring, inside its clearance,
        # and clear of both chains.
        board = off_grid(moving_via_board())
        board.pads = [kipy_pad("p", "GND", *INSIDE_PAD, layers=(BL.BL_F_Cu,))]
        connect.to(board)

        run()

        assert "push_commit" in board.calls
        assert sorted(board.deleted) == ["a", "b", "e", "f", "v"]
        assert len(created_vias(board)) == 1
        (report,) = reports
        (via,) = report.vias
        assert (via.id, via.start, via.moved) == ("v", VIA, True)
        x, y, h = *INSIDE_PAD, PAD_HALF_SIZE
        pad = box(x - h, y - h, x + h, y + h)
        ring = pad.distance(Point(via.end)) - VIA_DIAMETER / 2
        assert ring >= CLEARANCE - INSIDE_TOLERANCE - NM_ROUNDING

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


ZONE_MIN_WIDTH = 0.2
"""The SIG plane's minimum fill width in `plane_board`, mm."""
ZONE_MARGIN = ZONE_MIN_WIDTH / 2 + KICAD_MAX_ERROR
"""How far inside its region the SIG zone keeps a via's centre, mm."""
ZONE_REGION_TOP = 8 + 0.2
"""The top of the SIG zone's region: its outline less the higher-priority
OTHER zone over y < 8, grown by the zones' 0.2 mm clearance."""


INNER_PLANES = [BL.BL_F_Cu, BL.BL_In1_Cu, BL.BL_In2_Cu, BL.BL_B_Cu]


def ground_lines(log: Sequence[str]) -> list[str]:
    return [line.rstrip() for line in log if "round net" in line]


class TestGroundNets:
    def test_names_those_with_zones_on_two_inner_layers(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = moving_via_board()
        board.enabled_layers = INNER_PLANES
        planes = [BL.BL_In1_Cu, BL.BL_In2_Cu]
        board.zones = [kipy_copper_zone("p", "RET", kipy_rect(20, 20, 30, 30), planes)]
        connect.to(board)

        run("--dry-run")

        assert ground_lines(log) == [
            "INFO Ground net RET: zones on at least two inner layers"
        ]

    def test_falls_back_on_gnd_with_copper(
        self, connect: Connector, log: list[str]
    ) -> None:
        connect.to(pushed_stitching_board())

        run("--dry-run")

        assert ground_lines(log) == [
            "INFO Ground net GND: no net has zones on at least two inner layers"
        ]

    def test_with_none_says_so_and_the_run_goes_on(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = connect.to(moving_via_board())

        run()

        assert ground_lines(log) == [
            "INFO No ground nets: no net has zones on at least two inner layers and "
            "GND has no zones, vias or pads"
        ]
        assert board.calls == COMMIT_CALLS

    def test_given_ones_override_and_one_without_copper_warns(
        self, connect: Connector, log: list[str]
    ) -> None:
        connect.to(pushed_stitching_board())

        run("--dry-run", "--ground-nets", "PGND", "GND")

        assert ground_lines(log) == [
            "WARNING Ground net PGND has no zones, vias or pads on the board; "
            "check --ground-nets",
            "INFO Ground nets GND and PGND, as given",
        ]

    def test_are_not_named_without_a_via_that_may_move(
        self, connect: Connector, log: list[str]
    ) -> None:
        connect.to(straight_board())

        run("--dry-run", "--ground-nets", "PGND")

        assert ground_lines(log) == []

    @pytest.mark.parametrize(
        ("args", "expected"), [((), ()), (("--ground-nets", "A", "B"), ("A", "B"))]
    )
    def test_the_option_reaches_the_run(
        self, args: tuple[str, ...], expected: tuple[str, ...]
    ) -> None:
        options = cli.smooth_options(cli.parse_args(["smooth", *args]))

        assert options.ground_nets == expected


@dataclass
class Optimized:
    """What `run_smooth` gave `optimize_chains` on its last call."""

    vias: Sequence[ViaInput] = ()
    static_vias: Sequence[StaticVia] = ()


@pytest.fixture
def optimized(monkeypatch: pytest.MonkeyPatch) -> Optimized:
    seen = Optimized()
    real = at(COARSE)

    def spy(*args: object, **kwargs: object) -> RunResult:
        seen.vias = cast(Sequence[ViaInput], kwargs["vias"])
        seen.static_vias = cast(Sequence[StaticVia], kwargs["static_vias"])
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "optimize_chains", spy)
    return seen


def classes(vias: Sequence[ViaInput | StaticVia]) -> dict[str, bool]:
    return {v.via.id: v.ground for v in vias}


def on_net(tracks: Sequence[kbt.Track], net: str) -> list[kbt.Track]:
    for t in tracks:
        t.net = kbt.Net(name=net)
    return list(tracks)


class TestViaClasses:
    def test_board_vias_that_stand_still_are_static_with_their_class(
        self, connect: Connector, optimized: Optimized
    ) -> None:
        board = moving_via_board()
        board.vias += [kipy_via("g", "GND", 6, 6), kipy_via("w", "PWR", 1, 6)]
        connect.to(board)

        run("--dry-run")

        assert classes(optimized.vias) == {"v": False}
        assert classes(optimized.static_vias) == {"g": True, "w": False}
        (given,) = optimized.vias
        assert "g" not in via_ids(given.field)
        assert "w" in via_ids(given.field)
        assert via_ids(given.static_vias) == {"g"}

    def test_a_selected_via_starting_inside_its_clearance_is_no_static_via(
        self, connect: Connector, optimized: Optimized
    ) -> None:
        # As in `test_a_via_starting_inside_its_clearance_moves_out_of_it`,
        # with the route and its via of GND.
        route = on_net([*l_chain(), *detour_back()], "GND")
        via = kipy_via("v", "GND", *VIA)
        board = CommitBoard(
            selection=[*route, via],
            tracks=list(route),
            vias=[via, kipy_via("u", "GND", 6, 6)],
            pads=[kipy_pad("p", "P", *INSIDE_PAD, layers=(BL.BL_F_Cu,))],
        )
        connect.to(board)

        run("--dry-run")

        assert classes(optimized.vias) == {"v": True}
        assert classes(optimized.static_vias) == {"u": True}

    def test_given_ground_nets_set_the_classes(
        self, connect: Connector, optimized: Optimized
    ) -> None:
        board = moving_via_board()
        board.vias += [kipy_via("g", "GND", 6, 6), kipy_via("r", "RET", 1, 6)]
        connect.to(board)

        run("--dry-run", "--ground-nets", "SIG", "RET")

        assert classes(optimized.vias) == {"v": True}
        assert classes(optimized.static_vias) == {"g": False, "r": True}

    def test_without_ground_nets_nothing_is_static_or_split(
        self, connect: Connector, optimized: Optimized
    ) -> None:
        board = moving_via_board()
        board.vias.append(kipy_via("w", "PWR", 1, 6))
        connect.to(board)

        run("--dry-run")

        assert optimized.static_vias == ()
        (given,) = optimized.vias
        assert (given.ground, given.static_vias) == (False, None)

    def test_without_attraction_ground_nets_are_neither_found_nor_used(
        self, connect: Connector, optimized: Optimized, log: list[str]
    ) -> None:
        board = moving_via_board()
        board.vias += [kipy_via("g", "GND", 6, 6), kipy_via("w", "PWR", 1, 6)]
        connect.to(board)

        run("--dry-run", "--attraction=0", "--ground-nets", "SIG")

        assert ground_lines(log) == []
        assert optimized.static_vias == ()
        (given,) = optimized.vias
        assert (given.ground, given.static_vias) == (False, None)

    def test_attraction_is_on_by_default(self) -> None:
        params = cli.smooth_options(cli.parse_args(["smooth"])).params

        assert params.attraction > 0.0
        assert params.attraction_range > 0.0

    @pytest.mark.parametrize(
        ("ground", "stitching"),
        [((5.0, 1.5), False), ((6.06, 1.06), False), ((5.0, 2.5), True)],
        ids=["across", "diagonal", "stitching"],
    )
    def test_a_return_via_settles_at_the_neutral_distance_and_nearer_by_default(
        self,
        connect: Connector,
        coarse_optimizer: None,
        reports: list[pipeline.RunReport],
        ground: tuple[float, float],
        stitching: bool,
    ) -> None:
        # An unselected GND via 1.5 mm across the route from the via or
        # diagonally beside it, or a stitching via 2.5 mm across it, which
        # moves too. The default pulls the vias nearer than the balance,
        # trading some chain length for it.
        lengths: list[float] = []
        distances: list[float] = []
        for args in ((BALANCED,), ("--attraction=0",), ()):
            connect.to(return_via_board(ground, stitching=stitching))
            run("--dry-run", *args)
            lengths.append(sum(c.after.length for c in reports[-1].chains))
            ends = {v.id: v.end for v in reports[-1].vias}
            distances.append(math.dist(ends["v"], ends.get("g", ground)))

        pulled, alone, defaulted = lengths
        assert pulled <= alone + KEPT
        assert defaulted <= alone + RETURN_TRADED
        params = cli.smooth_options(cli.parse_args(["smooth", BALANCED])).params
        neutral = neutral_distance(params, RETURN_VIAS_APART)
        balanced, _, by_default = distances
        assert balanced == pytest.approx(neutral, abs=RETURN_SETTLED)
        assert RETURN_VIAS_APART - NM_ROUNDING <= by_default < neutral - RETURN_SETTLED

    @pytest.mark.parametrize("locked", [False, True], ids=["alone", "locked tracks"])
    def test_a_stitching_via_selected_without_tracks_moves(
        self,
        connect: Connector,
        coarse_optimizer: None,
        reports: list[pipeline.RunReport],
        locked: bool,
    ) -> None:
        # The stitching via 2.5 mm across the route from the SIG via, which
        # stays with its unselected tracks; with them selected but locked,
        # they hold it as well.
        board = return_via_board((5.0, 2.5), stitching=True)
        route = [t for t in board.tracks if t.net.name == "SIG"]
        for track in route:
            track.locked = locked
        stitching = [v for v in board.vias if v.id.value == "g"]
        board.selection = [*(route if locked else []), *stitching]
        connect.to(board)

        run("--dry-run")

        (via,) = reports[-1].vias
        assert via.moved
        assert math.dist(via.end, RETURN_SIGNAL_VIA) < math.dist(
            via.start, RETURN_SIGNAL_VIA
        )
        assert reports[-1].chains == ()

    def test_a_selection_without_tracks_draws_no_plot(
        self,
        connect: Connector,
        coarse_optimizer: None,
        reports: list[pipeline.RunReport],
        tmp_path: Path,
    ) -> None:
        board = return_via_board((5.0, 2.5), stitching=True)
        board.selection = [v for v in board.vias if v.id.value == "g"]
        connect.to(board)
        figure = tmp_path / "vias.png"

        run("--dry-run", f"--plot={figure}")

        (via,) = reports[-1].vias
        assert via.moved
        assert not figure.exists()

    @pytest.mark.parametrize(
        "args", [("--attraction=-0.1",), ("--attraction-range=0",)]
    )
    def test_a_negative_weight_or_an_empty_range_is_refused(
        self, args: tuple[str, ...]
    ) -> None:
        with pytest.raises(SystemExit):
            cli.parse_args(["smooth", *args])


def plane_board() -> CommitBoard:
    """A SIG route bending at the via at (10, 10) in the SIG plane on
    In2.Cu, which pulls the via toward (6, 6), into the higher-priority
    OTHER zone over y < 8. An OTHER track on In2.Cu ends at the zone's
    clearance from the via's ring. A second SIG via, at (12, 13), reaches
    the route only through the plane. The route and the via at (10, 10)
    are selected."""
    route = [
        kipy_track("11", "SIG", (2, 10), (10, 10)),
        kipy_track("12", "SIG", (10, 10), (10, 2), layer=BL.BL_B_Cu),
    ]
    via = kipy_via("e1", "SIG", 10, 10)
    plane = kipy_copper_zone(
        "d1",
        "SIG",
        kipy_rect(6, 6, 14, 14),
        [BL.BL_In2_Cu],
        min_width=ZONE_MIN_WIDTH,
    )
    plane.clearance = nm(0.2)
    other = kipy_zone(
        kipy_rect(6, 6, 14, 8),
        [BL.BL_In2_Cu],
        rule_area=False,
        zone_id="d2",
        net="OTHER",
    )
    other.priority = 1
    other.clearance = nm(0.2)
    return CommitBoard(
        selection=[*route, via],
        tracks=[
            *route,
            kipy_track("13", "SIG", (12, 13), (16, 16)),
            kipy_track("14", "OTHER", (10.625, 10), (18, 10), layer=BL.BL_In2_Cu),
        ],
        vias=[via, kipy_via("e2", "SIG", 12, 13)],
        zones=[plane, other],
        enabled_layers=[BL.BL_F_Cu, BL.BL_In1_Cu, BL.BL_In2_Cu, BL.BL_B_Cu],
        default_clearance=0.127,
    )


PLANE_BURSTS = 25
"""Outer iterations for `plane_board`: the via reaches its region's top in
about 16, then creeps along it to the iteration cap
(https://github.com/mairas/kibends/issues/21)."""


def test_a_via_in_a_plane_moves_within_its_region_and_commits(
    connect: Connector, log: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pipeline,
        "optimize_chains",
        at(replace(COARSE, max_iterations=PLANE_BURSTS)),
    )
    board = connect.to(plane_board())

    run()

    assert "stays fixed" not in "".join(log)
    assert board.calls == COMMIT_CALLS
    (via,) = created_vias(board)
    y = via.position.y / 1e6
    assert ZONE_REGION_TOP + ZONE_MARGIN - NM_ROUNDING <= y < 9.5


def test_outcome_counts_the_vias_rejected_edge_moves() -> None:
    stats = PolylineStats(1.0, 1, math.inf)
    chain = pipeline.ChainReport("SIG", "F.Cu", WIDTH, "SIG", 1, stats, stats, 2, True)
    via = pipeline.ViaReport("v", VIA, (4.0, 1.0), rejected_edges=3)
    report = pipeline.OptimizerReport(
        (chain,), (via,), (), Termination.CONVERGED, 7, 3.42, (), BOARD_FRAME
    )

    outcome = cli.format_outcome(report)

    assert outcome == (
        "Optimizer: converged after 7 iterations in 3.4 s, 5 rejected edge moves"
    )


def test_outcome_counts_a_pair_s_rejected_moves_and_warns_of_nothing() -> None:
    problem = "the chains and vias on F.Cu and B.Cu"
    stats = PolylineStats(1.0, 1, math.inf)
    chain = pipeline.ChainReport(
        "USB_D+", "F.Cu", WIDTH, "USB_D+", 1, stats, stats, 1, True
    )
    pair = pipeline.PairReport(
        nets=("USB_D+", "USB_D-"),
        layer="F.Cu",
        gap=0.15,
        gap_after=(0.14, 0.16),
        skew_before=0.0,
        skew_after=0.1,
        sections=1,
        held=0,
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

    assert outcome == (
        "Optimizer: converged after 7 iterations in 0.0 s, 4 rejected edge moves"
    )
    # Abnormal bursts are logged at debug only.
    assert cli.report_warnings(report) == []


def test_says_what_it_optimizes(connect: Connector, log: list[str]) -> None:
    connect.to(straight_board())

    run("--dry-run")

    assert "INFO Optimizing 1 chain\n" in log


def test_says_when_setup_and_optimization_start(
    connect: Connector, log: list[str]
) -> None:
    connect.to(straight_board())

    run("--dry-run")

    infos = [line for line in log if line.startswith("INFO ")]
    setting_up = infos.index("INFO Setting up the optimizer\n")
    starting = next(
        k for k, line in enumerate(infos) if line.startswith("INFO Optimizer set up")
    )
    assert setting_up < starting
    assert re.search(
        r"set up in \d+\.\d s: 1 problem, length \d+\.\d{3} mm, energy \d+\.\d{3}",
        infos[starting],
    )


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
JOG_NM = 100
"""A sideways jog in a track, nm."""
JOG_AT_NM = 200_000
"""nm along a pair's coupled start where its tracks jog."""
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


def pair_board(*, extra: Sequence[kbt.Track] = ()) -> CommitBoard:
    tracks = pair_tracks()
    usb = netclass("USB", PAIR_CLEARANCE)
    return CommitBoard(
        selection=list(tracks),
        tracks=[*tracks, *extra],
        netclasses={"USB_D+": usb, "USB_D-": usb},
    )


STUB_AT = 2.5
"""x where a stub leaves a pair's first straight, mm."""
N_STUB_AT = 2.0
"""x where `cut_partner_board`'s stub leaves USB_D-, mm."""
MEANDER_DEPTH = 0.8
"""How far, mm, `cut_partner_board`'s meander bulges away from USB_D+."""
MEANDER = [
    (9.5, 3.0 - PAIR_HALF_PITCH),
    (9.5, 3.0 - PAIR_HALF_PITCH - MEANDER_DEPTH),
    (11.0, 3.0 - PAIR_HALF_PITCH - MEANDER_DEPTH),
    (11.0, 3.0 - PAIR_HALF_PITCH),
]
"""An unselected meander on USB_D-'s last straight, copper the selection
leaves out."""


def cut_partner_board() -> CommitBoard:
    """`pair_board` with USB_D- cut by a selected stub on its first straight
    and by `MEANDER` on its last, USB_D+ running past both."""
    h = PAIR_HALF_PITCH
    mitre = math.tan(math.pi / 8) * h
    p = [t for t in pair_tracks() if t.net.name == USB[0]]
    n_points = [
        (PAIR_FAN[0], -PAIR_FAN[1]),
        (0.0, -h),
        (N_STUB_AT, -h),
        (5.0 + mitre, -h),
        (8.0 + mitre, 3.0 - h),
        MEANDER[0],
    ]
    n = [
        kipy_track(f"n{k}", USB[1], start, end, width=PAIR_WIDTH)
        for k, (start, end) in enumerate(itertools.pairwise(n_points))
    ]
    end = (13.0, 3.0 - h)
    n.append(kipy_track("n_end", USB[1], MEANDER[-1], end, width=PAIR_WIDTH))
    stub = kipy_track(
        "n_stub", USB[1], (N_STUB_AT, -h), (N_STUB_AT, -2.0), width=PAIR_WIDTH
    )
    meander = [
        kipy_track(f"m{k}", USB[1], start, end, width=PAIR_WIDTH)
        for k, (start, end) in enumerate(itertools.pairwise(MEANDER))
    ]
    usb = netclass("USB", PAIR_CLEARANCE)
    selected = [*p, *n, stub]
    return CommitBoard(
        selection=list(selected),
        tracks=[*selected, *meander],
        netclasses={net: usb for net in USB},
    )


PATTERN_END = 17.0
"""x where `pattern_board`'s diff pair ends, mm, long enough for the
pattern's straights to run coupled."""
PATTERN_X = (8.3, 10.5, 12.0, 14.2)
"""x of `pattern_board`'s pattern's start, its bump's two sides, and its
end, mm."""


def pattern_board(
    nesting: int, pattern_x: tuple[float, float, float, float] = PATTERN_X
) -> CommitBoard:
    """`pair_board` run on to x `PATTERN_END`, USB_D- through a selected
    length-tuning pattern group, inside `nesting` more groups: two coupled
    straights and a bump between them, away from USB_D+, of two tracks and
    an arc, at the x of `pattern_x` as `PATTERN_X` gives them."""
    h = PAIR_HALF_PITCH
    y = 3.0 - h
    deep = y - MEANDER_DEPTH
    tracks = pair_tracks()
    for t in tracks:
        if t.end.x == nm(13.0):
            t.end = vec(PATTERN_END, t.end.y / 1e6)
    n_end = next(t for t in tracks if t.id.value == "n4")
    n_end.end = vec(PATTERN_X[0], y)
    start, left, right, end = ((x, y) for x in pattern_x)
    pattern_items: list[kbt.BoardItem] = [
        kipy_track("pt_left", USB[1], start, left, width=PAIR_WIDTH),
        kipy_track("pt_down", USB[1], left, (left[0], deep), width=PAIR_WIDTH),
        kipy_arc(
            "pt_bump",
            USB[1],
            (left[0], deep),
            ((left[0] + right[0]) / 2, deep - MEANDER_DEPTH / 2),
            (right[0], deep),
            width=PAIR_WIDTH,
        ),
        kipy_track("pt_up", USB[1], (right[0], deep), right, width=PAIR_WIDTH),
        kipy_track("pt_right", USB[1], right, end, width=PAIR_WIDTH),
    ]
    after = kipy_track("n5", USB[1], end, (PATTERN_END, y), width=PAIR_WIDTH)
    groups = [kipy_group("Tuning Pattern", pattern_items)]
    for k in range(nesting):
        groups.append(kipy_group(f"outer{k}", [groups[-1]]))
    usb = netclass("USB", PAIR_CLEARANCE)
    return CommitBoard(
        selection=[*tracks, after, groups[-1]],
        tracks=[
            *tracks,
            after,
            *cast(list[kbt.Track | kbt.ArcTrack], pattern_items),
        ],
        groups=groups,
        netclasses={net: usb for net in USB},
    )


PATTERN_IDS = {"pt_left", "pt_down", "pt_bump", "pt_up", "pt_right"}
SHORT_PATTERN_X = (8.3, 9.0, 10.5, 11.2)
"""`PATTERN_X` with straights of two pitches."""


def corner_stub_board(at: float) -> CommitBoard:
    """`pair_board` with a selected USB_D- stub leaving the outer track at x
    `at`, near its first 45° corner at x 5 + mitre."""
    h = PAIR_HALF_PITCH
    corner = 5.0 + math.tan(math.pi / 8) * h
    foot = (at, -h + max(at - corner, 0.0))
    stub = kipy_track("n_stub", USB[1], foot, (at, -2.0), width=PAIR_WIDTH)
    board = pair_board(extra=[stub])
    board.selection = [*board.selection, stub]
    return board


PAIR_SECOND_TRACK = 5.0
"""How far, mm, track USB_D- moves away to not pair."""
PAIR_WIDER = 0.25


PAIR_FAN_N = (PAIR_FAN[0], -PAIR_FAN[1])
"""Where track N starts."""
PAIR_END = (13.0, 3.0 + PAIR_HALF_PITCH)
"""Where track P ends, at the end of the pair's coupled section."""
ONWARD = 3.0
"""How far, mm, the track on B.Cu at a via runs on along x."""


def with_via(
    board: CommitBoard,
    net: str = USB[0],
    at: Point2 = PAIR_FAN,
    onward: float = -ONWARD,
    via_id: str = "v",
) -> CommitBoard:
    """`board` with a via of `net` at `at`, joining the pair's track there to
    a selected track on B.Cu that runs on by `onward` along x."""
    down = kipy_track(
        f"down_{via_id}",
        net,
        at,
        (at[0] + onward, at[1]),
        width=PAIR_WIDTH,
        layer=BL.BL_B_Cu,
    )
    via = kipy_via(via_id, net, *at)
    board.selection = [*board.selection, down, via]
    board.tracks = [*board.tracks, down]
    board.vias = [*board.vias, via]
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


def assert_stubs_kept(board: CommitBoard, feet: Sequence[float]) -> None:
    """USB_D+ of `pair_board`, as committed or else as routed, still passes
    through the foot of each stub at x of `feet` on its first straight."""
    routed = [segment(t) for t in pair_tracks() if t.net.name == USB[0]]
    p = linemerge(by_net(board)[USB[0]] or routed)
    for x in feet:
        assert p.distance(Point(x, PAIR_HALF_PITCH)) <= NM_ROUNDING


TVS_WIDTH = 0.261
TVS_GAP = 0.203
"""The track width and gap, mm, of the USB pair on the board that
`tvs_board` follows."""
TVS_CENTRELINE = ((0.0, 3.1), (0.0, 0.0), (0.7, -0.7), (1.5, -0.7))
"""`tvs_board`'s centreline: up from the connector's pins, then right by
90° through a 45° chamfer."""
TVS_PADS_Y = 0.6
"""y of the TVS part's pads, each passed through by a track of its net."""
TVS_PAD_SHIFT = 0.2
"""How far, mm, each pad's centre lies outward of its track, so that the
two nets' pads keep apart."""
TVS_VIA_DIAMETER = 0.4
TVS_VIA_DRILL = 0.2
TVS_N_RUNS_ON = 0.5
"""How far, mm, USB_D- runs on past the pair's end to its via, so that
the two vias keep apart."""


def tvs_tracks() -> tuple[FloatArray, FloatArray]:
    """`tvs_board`'s tracks USB_D+ and USB_D-, mitred about its centreline."""
    n, p = offset_pair(np.array(TVS_CENTRELINE), TVS_GAP, TVS_WIDTH)
    n = np.concatenate([n, [n[-1] + (TVS_N_RUNS_ON, 0.0)]])
    return p, n


def tvs_board() -> CommitBoard:
    """A pair leaving a connector's pins straight up through the pads of a
    TVS part, one pad of its own net on each track, then turning 90° to two
    vias through a 45° chamfer."""
    tracks: list[kbt.Track] = []
    pads: list[kbt.Pad] = []
    vias: list[kbt.Via] = []
    for net, track, side in zip(USB, tvs_tracks(), (-1.0, 1.0), strict=True):
        route = [(float(x), float(y)) for x, y in track]
        tracks += [
            kipy_track(f"{net}{k}", net, start, end, width=TVS_WIDTH)
            for k, (start, end) in enumerate(itertools.pairwise(route))
        ]
        (x, y), end = route[0], route[-1]
        pads += [
            kipy_pad(f"pin{net}", net, x + side * TVS_PAD_SHIFT, y),
            kipy_pad(f"tvs{net}", net, x + side * TVS_PAD_SHIFT, TVS_PADS_Y),
        ]
        vias.append(kipy_via(f"via{net}", net, *end, TVS_VIA_DIAMETER, TVS_VIA_DRILL))
    usb = netclass("USB", PAIR_CLEARANCE)
    return CommitBoard(
        selection=list(tracks),
        tracks=list(tracks),
        pads=pads,
        vias=vias,
        netclasses={net: usb for net in USB},
    )


@pytest.mark.usefixtures("coarse_optimizer")
class TestPairs:
    def test_commits_both_tracks_in_one_commit_at_their_gap(
        self, connect: Connector
    ) -> None:
        board = connect.to(pair_board())

        run()

        assert board.calls == COMMIT_CALLS
        assert sorted(board.deleted) == list(PAIR_IDS)
        gaps = coupled_gaps(*by_net(board).values())
        assert gaps == pytest.approx((PAIR_GAP, PAIR_GAP), abs=NM_ROUNDING)

    def test_a_nanometre_track_at_a_corner_is_smoothed_away(
        self, connect: Connector, reports: list[pipeline.RunReport]
    ) -> None:
        board = pair_board()
        corner = next(t for t in board.tracks if t.id.value == "p3")
        # As KiCad leaves one at a 45° corner: across it, 1 nm each way.
        dot = kipy_track("dot", USB[0], (0, 0), (0, 0), width=PAIR_WIDTH)
        dot.start = Vector2.from_xy(corner.start.x, corner.start.y)
        dot.end = Vector2.from_xy(corner.start.x + 1, corner.start.y - 1)
        corner.start = dot.end
        board.selection = [*board.selection, dot]
        board.tracks = [*board.tracks, dot]
        connect.to(board)

        run()

        ((pair,),) = (r.pairs for r in reports)
        assert pair.held == 0
        assert sorted(board.deleted) == sorted([*PAIR_IDS, "dot"])

    def test_a_sideways_jog_in_both_tracks_is_smoothed_away(
        self, connect: Connector, reports: list[pipeline.RunReport]
    ) -> None:
        board = pair_board()
        jogs: list[kbt.Track] = []
        for prefix, net in zip(("p", "n"), USB, strict=True):
            (straight,) = (t for t in board.tracks if t.id.value == f"{prefix}2")
            beyond_ids = {f"{prefix}3", f"{prefix}4"}
            later = [t for t in board.tracks if t.id.value in beyond_ids]
            # As KiCad leaves one where a track was nudged: the rest of it
            # moved aside, toward N, so near the section's start that the
            # centreline's offset turns there within 0.03 mm of it.
            at = Vector2.from_xy(straight.start.x + JOG_AT_NM, straight.start.y)
            aside = Vector2.from_xy(at.x, at.y - JOG_NM)
            step = kipy_track(f"{prefix}jog", net, (0, 0), (0, 0), width=PAIR_WIDTH)
            step.start, step.end = at, aside
            beyond = kipy_track(f"{prefix}on", net, (0, 0), (0, 0), width=PAIR_WIDTH)
            beyond.start = aside
            beyond.end = Vector2.from_xy(straight.end.x, straight.end.y - JOG_NM)
            straight.end = at
            for track in later:
                track.start = Vector2.from_xy(track.start.x, track.start.y - JOG_NM)
                track.end = Vector2.from_xy(track.end.x, track.end.y - JOG_NM)
            jogs += [step, beyond]
        board.selection = [*board.selection, *jogs]
        board.tracks = [*board.tracks, *jogs]
        connect.to(board)

        run()

        ((pair,),) = (r.pairs for r in reports)
        assert pair.held == 0
        assert sorted(board.deleted) == sorted(
            [*PAIR_IDS, "pjog", "pon", "njog", "non"]
        )
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
        pairs = netclass("DP", PAIR_CLEARANCE)
        board = connect.to(
            CommitBoard(
                selection=list(tracks),
                tracks=list(tracks),
                netclasses={net: pairs for net in (*USB, *eth)},
            )
        )

        run()

        assert "INFO Optimizing 4 chains, 2 differential pairs among them\n" in log
        assert board.calls == COMMIT_CALLS
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
        usb = netclass("USB", PAIR_CLEARANCE)
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

    def test_a_section_with_copper_between_its_ends_moves_whole(
        self, connect: Connector, reports: list[pipeline.RunReport]
    ) -> None:
        stub = kipy_track(
            "stub", "USB_D+", (2.5, PAIR_HALF_PITCH), (2.5, 2.0), width=PAIR_WIDTH
        )
        connect.to(pair_board(extra=[stub]))

        run("--dry-run")

        ((pair,),) = (r.pairs for r in reports)
        assert (pair.sections, pair.held) == (1, 0)

    def test_pins_inside_a_section_hold_it_there_and_let_its_turn_round(
        self, connect: Connector, log: list[str], reports: list[pipeline.RunReport]
    ) -> None:
        board = connect.to(tvs_board())

        run()

        assert board.calls == COMMIT_CALLS
        assert not unexpected_warnings(log)
        ((pair,),) = (r.pairs for r in reports)
        assert (pair.sections, pair.held) == (1, 0)
        lines = by_net(board)
        coupled = box(-FAR, -FAR, TVS_CENTRELINE[-1][0], TVS_CENTRELINE[0][1])
        tracks: list[FloatArray] = []
        for net, routed in zip(USB, tvs_tracks(), strict=True):
            joined = linemerge(lines[net])
            assert isinstance(joined, LineString)
            assert joined.distance(Point(routed[0][0], TVS_PADS_Y)) <= NM_ROUNDING
            for corner in routed[1:3]:
                assert joined.distance(Point(corner)) >= ROUNDED_OFF
            tracks.append(np.array(joined.intersection(coupled).coords))
        low, high = gap_range(tracks[0], tracks[1], TVS_WIDTH)
        slack = GAP_TOLERANCE * TVS_GAP
        assert TVS_GAP - slack <= low <= high <= TVS_GAP + slack

    def test_pair_past_same_net_pads_commits_smoothed_and_pinned(
        self, connect: Connector
    ) -> None:
        # As on the board of https://github.com/mairas/kibends/issues/42:
        # each track runs over the edge of a pad of its own net beyond the
        # pair's corners. A pair keeps its contacts where they are, so each
        # track stays pinned at the middle of its stretch over the pad.
        inside = PAD_HALF_SIZE - PAD_OVERLAP
        pads = [
            kipy_pad(f"pad{net}", net, PADS_AT, 3.0 + side * (PAIR_HALF_PITCH + inside))
            for net, side in zip(USB, (1.0, -1.0), strict=True)
        ]
        board = connect.to(pair_board())
        board.pads = pads

        run()

        assert board.calls == COMMIT_CALLS
        lines = by_net(board)
        for net, side in zip(USB, (1.0, -1.0), strict=True):
            joined = linemerge(lines[net])
            assert isinstance(joined, LineString)
            y = 3.0 + side * PAIR_HALF_PITCH
            assert (PADS_AT, y) in joined.coords
            # Both corners ahead of the pads are rounded off.
            mitre = math.tan(math.pi / 8) * PAIR_HALF_PITCH
            for corner in ((5.0, y - 3.0), (8.0, y)):
                at = Point(corner[0] - side * mitre, corner[1])
                assert joined.distance(at) >= ROUNDED_OFF
        low, high = coupled_gaps(*lines.values())
        assert PAIR_GAP - GAP_SLACK <= low <= high <= PAIR_GAP + GAP_SLACK

    @pytest.mark.parametrize(
        ("before", "held"),
        # The stub holds the routing beside it. At the corner, direction
        # pins a coarse spacing along both edges keep the corner within the
        # bend-radius limit; just before it, the corner is the direction
        # pin, too close to bend within it, so the section is held.
        [(0.0, 0), (STUB_NEAR_CORNER, 1)],
    )
    def test_a_stub_at_or_near_a_corner_stays_on_its_track_without_a_warning(
        self,
        connect: Connector,
        log: list[str],
        reports: list[pipeline.RunReport],
        before: float,
        held: int,
    ) -> None:
        corner = 5.0 - math.tan(math.pi / 8) * PAIR_HALF_PITCH
        x = nm(corner - before) / NM_PER_MM
        stub = kipy_track(
            "stub", "USB_D+", (x, PAIR_HALF_PITCH), (x, 2.0), width=PAIR_WIDTH
        )
        board = connect.to(pair_board(extra=[stub]))

        run()

        ((pair,),) = (r.pairs for r in reports)
        assert (pair.sections, pair.held) == (1, held)
        assert not unexpected_warnings(log)
        assert_stubs_kept(board, [x])

    def test_two_stubs_close_together_let_their_section_move(
        self,
        connect: Connector,
        log: list[str],
        reports: list[pipeline.RunReport],
    ) -> None:
        feet = (2.5, 2.5 + STUBS_APART)
        stubs = [
            kipy_track(
                f"stub{k}", "USB_D+", (x, PAIR_HALF_PITCH), (x, 2.0), width=PAIR_WIDTH
            )
            for k, x in enumerate(feet)
        ]
        board = connect.to(pair_board(extra=stubs))

        run()

        ((pair,),) = (r.pairs for r in reports)
        assert (pair.sections, pair.held) == (1, 0)
        assert board.calls == COMMIT_CALLS
        assert not unexpected_warnings(log)
        assert_stubs_kept(board, feet)

    def test_a_selected_stub_on_one_track_keeps_both_sides_coupled(
        self,
        connect: Connector,
        log: list[str],
        reports: list[pipeline.RunReport],
    ) -> None:
        # The stub cuts USB_D+ into two chains at its foot; USB_D- is cut
        # opposite it, and each part pairs with the one beside it.
        foot = (STUB_AT, PAIR_HALF_PITCH)
        stub = kipy_track("stub", USB[0], foot, (STUB_AT, 2.0), width=PAIR_WIDTH)
        board = pair_board(extra=[stub])
        board.selection = [*board.selection, stub]
        connect.to(board)

        run()

        assert board.calls == COMMIT_CALLS
        (pairs,) = (r.pairs for r in reports)
        assert [(p.sections, p.held) for p in pairs] == [(1, 0), (1, 0)]
        for pair in pairs:
            assert pair.gap_after == pytest.approx((PAIR_GAP, PAIR_GAP), abs=REPORTED)
        assert not [w for w in log if "on its own" in w or "do not run coupled" in w]
        lines = by_net(board)
        p = [line for line in lines[USB[0]] if {x for x, _ in line.coords} != {STUB_AT}]
        gaps = coupled_gaps(p, lines[USB[1]])
        assert gaps == pytest.approx((PAIR_GAP, PAIR_GAP), abs=NM_ROUNDING)
        n_tracks = [t for t in created_on(board, BL.BL_F_Cu) if t.net.name == USB[1]]
        assert (nm(STUB_AT), nm(-PAIR_HALF_PITCH)) in ends(n_tracks)

    def test_a_track_cut_by_a_stub_and_a_meander_pairs_part_by_part(
        self,
        connect: Connector,
        log: list[str],
        reports: list[pipeline.RunReport],
    ) -> None:
        # USB_D- is cut by a selected stub and by an unselected meander;
        # USB_D+ runs past both.
        board = connect.to(cut_partner_board())

        run()

        assert board.calls == COMMIT_CALLS
        (pairs,) = (r.pairs for r in reports)
        assert [(p.sections, p.held) for p in pairs] == [(1, 0)] * 3
        for pair in pairs:
            assert pair.gap_after == pytest.approx((PAIR_GAP, PAIR_GAP), abs=REPORTED)
        assert not unexpected_warnings(log)
        assert not {f"m{k}" for k in range(len(MEANDER) - 1)} & set(board.deleted)
        # USB_D+ is cut opposite the stub and both ends of the meander.
        p = [t for t in created_on(board, BL.BL_F_Cu) if t.net.name == USB[0]]
        h = PAIR_HALF_PITCH
        cuts = [(N_STUB_AT, h), (MEANDER[0][0], 3.0 + h), (MEANDER[-1][0], 3.0 + h)]
        assert {(nm(x), nm(y)) for x, y in cuts} <= ends(p)

    @pytest.mark.parametrize("at", [4.9, 5.0])
    def test_a_stub_near_a_corner_keeps_the_gap(
        self, connect: Connector, log: list[str], at: float
    ) -> None:
        # Within a pitch before the outer track's corner, a section ends
        # short of the stub's foot on one track or the other.
        board = connect.to(corner_stub_board(at))

        run()

        assert board.calls == COMMIT_CALLS
        assert not unexpected_warnings(log)
        lines = by_net(board)
        n = [line for line in lines[USB[1]] if {x for x, _ in line.coords} != {at}]
        gaps = coupled_gaps(lines[USB[0]], n)
        assert gaps == pytest.approx((PAIR_GAP, PAIR_GAP), abs=REPORTED)

    def test_a_stub_on_a_corner_holds_the_section(
        self,
        connect: Connector,
        log: list[str],
        reports: list[pipeline.RunReport],
    ) -> None:
        # A stub on the corner's diagonal pins the outer track where it does
        # not run straight, so the section stays as drawn.
        board = connect.to(corner_stub_board(5.1))

        run()

        assert board.calls == []
        assert not unexpected_warnings(log)
        (pairs,) = (r.pairs for r in reports)
        assert [(p.sections, p.held) for p in pairs] == [(1, 1)]

    @pytest.mark.parametrize("nesting", [0, 2])
    def test_a_selected_tuning_pattern_stays_and_pairs(
        self,
        connect: Connector,
        log: list[str],
        reports: list[pipeline.RunReport],
        nesting: int,
    ) -> None:
        board = connect.to(pattern_board(nesting))

        run()

        assert board.calls == COMMIT_CALLS
        assert not unexpected_warnings(log)
        assert not [line for line in log if "do not run coupled" in line]
        assert not PATTERN_IDS & set(board.deleted)
        # The pattern's straights pair with USB_D+, which stays at its
        # routed gap beside them.
        (pairs,) = (r.pairs for r in reports)
        assert (2, 2) in [(p.sections, p.held) for p in pairs]
        p = linemerge(by_net(board)[USB[0]])
        beside = p.intersection(box(PATTERN_X[0], -FAR, PATTERN_X[-1], FAR))
        y = 3.0 + PAIR_HALF_PITCH
        assert [y for _, y in beside.coords] == pytest.approx(
            [y] * len(beside.coords), abs=NM_ROUNDING
        )

    def test_a_pattern_s_short_straights_pair_against_a_push(
        self,
        connect: Connector,
        log: list[str],
        reports: list[pipeline.RunReport],
    ) -> None:
        # A via beside USB_D+, opposite the pattern's first straight, would
        # push it toward the pattern if it smoothed on its own.
        board = pattern_board(0, SHORT_PATTERN_X)
        y = 3.0 + PAIR_HALF_PITCH
        x = (SHORT_PATTERN_X[0] + SHORT_PATTERN_X[1]) / 2
        board.vias = [kipy_via("gnd", "GND", x, y + PAIR_HALF_PITCH + 0.3)]
        connect.to(board)

        run()

        assert not unexpected_warnings(log)
        (pairs,) = (r.pairs for r in reports)
        assert (2, 2) in [(p.sections, p.held) for p in pairs]
        p = linemerge(by_net(board)[USB[0]])
        for start, end in itertools.batched(SHORT_PATTERN_X, 2, strict=True):
            beside = p.intersection(box(start, -FAR, end, FAR))
            assert [y for _, y in beside.coords] == pytest.approx(
                [y] * len(beside.coords), abs=NM_ROUNDING
            )

    def test_a_diff_pair_cut_by_a_stub_counts_once(
        self, connect: Connector, log: list[str]
    ) -> None:
        foot = (STUB_AT, PAIR_HALF_PITCH)
        stub = kipy_track("stub", USB[0], foot, (STUB_AT, 2.0), width=PAIR_WIDTH)
        board = pair_board(extra=[stub])
        board.selection = [*board.selection, stub]
        connect.to(board)

        run("--dry-run")

        assert "INFO Optimizing 4 chains, 1 differential pair among them\n" in log

    def test_via_where_a_coupled_section_ends_stays_fixed(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = connect.to(off_grid(with_via(pair_board(), at=PAIR_END, onward=ONWARD)))

        run()

        assert created_vias(board) == []
        assert "v" not in board.deleted
        (line,) = [w for w in log if "Via v" in w]
        x, y = PAIR_END[0] - GRID_ORIGIN[0], PAIR_END[1] - GRID_ORIGIN[1]
        assert f"Via v at ({x:.3f}, {y:.3f}) stays fixed" in line
        assert "it is on USB_D+, a net of a diff pair" in line
        assert all(by_net(board).values())

    def test_vias_at_the_fan_outs_stay_and_hold_their_ends(
        self, connect: Connector, log: list[str]
    ) -> None:
        board = with_via(pair_board())
        connect.to(with_via(board, USB[1], PAIR_FAN_N, via_id="w"))

        run()

        assert created_vias(board) == []
        assert not {"v", "w"} & set(board.deleted)
        for via in ("v", "w"):
            (line,) = [w for w in log if f"Via {via} " in w]
            assert "a net of a diff pair" in line
        front = {
            net: [t for t in created_on(board, BL.BL_F_Cu) if t.net.name == net]
            for net in USB
        }
        for net, at in zip(USB, (PAIR_FAN, PAIR_FAN_N), strict=True):
            assert (nm(at[0]), nm(at[1])) in ends(front[net])
        gaps = coupled_gaps(*([segment(t) for t in front[net]] for net in USB))
        assert gaps == pytest.approx((PAIR_GAP, PAIR_GAP), abs=NM_ROUNDING)

    def test_a_via_a_diff_pair_track_runs_through_stays_on_it(
        self, connect: Connector, log: list[str]
    ) -> None:
        h = PAIR_HALF_PITCH
        m = math.tan(math.pi / 8) * h
        mid = (PAIR_FAN[0] / 2, (PAIR_FAN[1] + h) / 2)
        tracks: list[kbt.Track] = []
        for net, side, prefix in zip(USB, (1.0, -1.0), ("p", "n"), strict=True):
            points = [
                (PAIR_FAN[0], side * PAIR_FAN[1]),
                *([mid] if side > 0 else []),
                (0.0, side * h),
                (5.0 - side * m, side * h),
                (8.0 - side * m, 3.0 + side * h),
                (13.0, 3.0 + side * h),
            ]
            tracks += [
                kipy_track(f"{prefix}{k}", net, a, b, width=PAIR_WIDTH)
                for k, (a, b) in enumerate(itertools.pairwise(points))
            ]
        via = kipy_via("v", USB[0], *mid, diameter=0.3, drill=0.15)
        usb = netclass("USB", PAIR_CLEARANCE)
        board = connect.to(
            CommitBoard(
                selection=[*tracks, via],
                tracks=list(tracks),
                vias=[via],
                netclasses={USB[0]: usb, USB[1]: usb},
            )
        )

        run()

        assert created_vias(board) == []
        assert "v" not in board.deleted
        (line,) = [w for w in log if "Via v" in w]
        assert "a net of a diff pair" in line
        plus = [t for t in created_on(board, BL.BL_F_Cu) if t.net.name == USB[0]]
        assert (nm(mid[0]), nm(mid[1])) in ends(plus)

    @pytest.mark.parametrize("unpaired", ["far", "wide"])
    def test_via_on_a_diff_pair_chain_that_does_not_pair_stays(
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
        connect.to(with_via(board))

        run("--dry-run")

        (line,) = [w for w in log if "Via v" in w]
        assert "a net of a diff pair" in line
        assert not any("moving via" in line for line in log)

    def test_section_left_bent_too_tightly_is_held_in_a_second_run(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        log: list[str],
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
                )
                for p in result.pairs
            )
            return replace(result, pairs=pairs)

        monkeypatch.setattr(pipeline, "optimize_chains", too_tight)
        board = connect.to(pair_board())

        run()

        assert held == [[None], [Hold.CROWDING]]
        assert any(Hold.CROWDING.value in line for line in log)
        # The repeated run says it repeats, rather than seeming to start over.
        assert log.count("INFO Setting up the optimizer\n") == 1
        assert "INFO Setting up the optimizer again, with held sections\n" in log
        assert any(line.startswith("INFO Optimizer set up again") for line in log)
        # Whether a net's chain is committed depends on how far repulsion
        # bends its fan-out (some 25 µm, varying with the BLAS) against the
        # chord tolerance; either way the held section, all pins, stays as
        # routed, however its tracks are split.
        copper = {
            net: unary_union(lines).buffer(NM_ROUNDING)
            for net, lines in by_net(board).items()
        }
        for t in pair_tracks():
            if not t.id.value.endswith("1"):
                assert t.id.value not in board.deleted or copper[t.net.name].covers(
                    segment(t)
                )
        (line,) = [w for w in log if "too tight" in w]
        assert line.startswith("DEBUG")
        assert Hold.CROWDING.value in line and "again" in line

    def test_time_limit_covers_a_second_run_that_starts_over(
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
        connect.to(pair_board())

        run("--dry-run", "--time-limit=1.5")

        first, second = runs
        assert first.termination is Termination.TIME_LIMIT
        # Past the limit, the second run still completes one outer iteration,
        # then stops at the first burst of the next.
        assert second.iterations <= 2
        total = first.iterations + second.iterations
        assert f"after {total} iterations" in capsys.readouterr().out

    def first_run_gives_up(
        self, monkeypatch: pytest.MonkeyPatch, last: Termination | None = None
    ) -> list[RunResult]:
        """Make the first run give up and leave its section too tight, so
        that the stage repeats it, and the repeat end `last` if given; the
        runs it makes, as reported."""
        real: Callable[..., RunResult] = pipeline.optimize_chains
        runs: list[RunResult] = []

        def first_gives_up(*args: object, **kwargs: object) -> RunResult:
            result = real(*args, **kwargs)
            if not kwargs["again"]:
                pairs = tuple(
                    replace(p, bend=BendCheck(TOO_TIGHT_RADIUS, too_tight=True))
                    for p in result.pairs
                )
                abandoned = (ABANDONED, MAX_ABNORMAL_BURSTS)
                result = replace(
                    result,
                    pairs=pairs,
                    gave_up=(abandoned,),
                    termination=Termination.GAVE_UP,
                )
            elif last is not None:
                result = replace(result, termination=last)
            runs.append(result)
            return result

        monkeypatch.setattr(pipeline, "optimize_chains", first_gives_up)
        return runs

    def test_a_first_run_s_give_up_outlasts_a_second_that_converges(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        log: list[str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        runs = self.first_run_gives_up(monkeypatch)
        # Another net's L, so that the second run, which starts over with the
        # section held, has work to do.
        bend = [
            kipy_track("l1", "SIG", (0, 10), (4, 10), width=WIDTH),
            kipy_track("l2", "SIG", (4, 10), (4, 14), width=WIDTH),
        ]
        board = pair_board(extra=bend)
        board.selection = [*board.selection, *bend]
        connect.to(board)

        run("--dry-run")

        assert [r.termination for r in runs] == [
            Termination.GAVE_UP,
            Termination.CONVERGED,
        ]
        assert not runs[1].gave_up
        (outcome,) = capsys.readouterr().out.splitlines()
        assert outcome.startswith(f"Optimizer: {Termination.GAVE_UP.value} after ")
        (line,) = [w for w in log if "Gave up" in w]
        assert ABANDONED in line

    @pytest.mark.parametrize(
        "last", [Termination.TIME_LIMIT, Termination.ITERATION_CAP]
    )
    def test_a_second_run_that_stops_early_reports_its_own_ending(
        self,
        connect: Connector,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        last: Termination,
    ) -> None:
        runs = self.first_run_gives_up(monkeypatch, last)
        connect.to(pair_board())

        run("--dry-run")

        assert len(runs) == 2
        (outcome,) = capsys.readouterr().out.splitlines()
        assert outcome.startswith(f"Optimizer: {last.value} after ")

    def test_a_problem_abandoned_in_both_runs_is_logged_once(
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
straight that once folded a section 180° in a trial step of the first burst:
https://github.com/mairas/kibends/issues/50. The fold needs the settings it
was found at, `FOLDING`, not the defaults."""

FOLDING = OptimizerSettings(burst_maxiter=100, lbfgs_ftol=1e-15, max_iterations=1)
"""The defaults when the fold was found, for one outer iteration, as the fold
came in the first burst."""


@pytest.mark.parametrize(("net", "x"), FOLDED)
def test_a_pair_pinned_on_a_straight_keeps_its_sections_finite(
    connect: Connector, monkeypatch: pytest.MonkeyPatch, net: str, x: float
) -> None:
    results: list[RunResult] = []
    first_burst = at(FOLDING)

    def spy(*args: object, **kwargs: object) -> RunResult:
        result = first_burst(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(pipeline, "optimize_chains", spy)
    # Without its fan-out, the pair is coupled from its start.
    tracks = [t for t in pair_tracks() if t.id.value not in ("p1", "n1")]
    side = 1.0 if net == USB[0] else -1.0
    usb = netclass("USB", PAIR_CLEARANCE)
    board = connect.to(
        CommitBoard(
            selection=list(tracks),
            tracks=list(tracks),
            pads=[kipy_pad("pad", net, x + PAD_HALF_SIZE, side * PAIR_HALF_PITCH)],
            netclasses={n: usb for n in USB},
        )
    )

    run()

    assert board.calls == COMMIT_CALLS
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
    usb = netclass("USB", clearance)
    board.netclasses = {"USB_D+": usb, "USB_D-": usb}
    connect.to(board)
    run()
    return by_net(board), calls


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param(DEFAULT_SETTINGS, id="default"),
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
        return self.sections[i].offset(self.result[i])


def spy_simplify(monkeypatch: pytest.MonkeyPatch) -> list[Simplified]:
    calls: list[Simplified] = []
    real = pipeline.simplify_chains

    def spy(
        chains: Sequence[ChainInput],
        run: RunResult,
        tolerance: float,
        *,
        sections: Mapping[int, PairBlock],
    ) -> list[FloatArray]:
        result = real(chains, run, tolerance, sections=sections)
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
    two straight sections, split by a detour of each track outward around a
    pad of another net; the second section at `second_gap`."""
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
    usb = netclass("USB", PAIR_CLEARANCE)
    return CommitBoard(
        selection=list(tracks),
        tracks=list(tracks),
        pads=[kipy_pad("pad", "GND", (split + rejoin) / 2, 0.0)],
        netclasses={"USB_D+": usb, "USB_D-": usb},
    )


@pytest.mark.usefixtures("coarse_optimizer")
def test_straight_coupled_sections_commit_straight_along_their_length(
    connect: Connector,
) -> None:
    board = connect.to(split_pair_board())

    run()

    for net, lines in by_net(board).items():
        y = PAIR_HALF_PITCH if net == "USB_D+" else -PAIR_HALF_PITCH
        pieces = sorted(
            tuple(round(x, 6) for x, _ in line.coords)
            for line in lines
            if all(abs(ys - y) < NM_ROUNDING for _, ys in line.coords)
        )
        coupled: list[tuple[float, float]] = []
        for start, end in pieces:
            if coupled and coupled[-1][1] == start:
                coupled[-1] = (coupled[-1][0], end)
            else:
                coupled.append((start, end))
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
    bus = netclass("BUS", PAIR_GAP)
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
    usb = netclass("USB", PAIR_CLEARANCE)
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
def test_a_chain_beside_a_pair_at_clearance_lets_both_smooth(
    connect: Connector,
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
    usb = netclass("USB", PAIR_GAP)
    board = connect.to(
        CommitBoard(
            selection=list(tracks),
            tracks=list(tracks),
            netclasses={"USB_D+": usb, "USB_D-": usb},
        )
    )

    run()

    assert all(len(lines) > len(BUS_CENTRE) - 1 for lines in by_net(board).values())


STUB_GAP = 0.138
"""The gap of a pair's 1 mm end stubs, unlike the 0.150 mm between them."""
STUB = 1.0
STUBBED_LENGTH = 10.0
STUB_JOG = 0.1
"""How long, along the pair, each track takes to step between the gaps."""
END_REACH = 2.0
END_SAMPLES = 50
"""mm from each end within which the gap may still follow the stubs':
the stub, the jog and the end edge that takes up the step."""


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
    usb = netclass("USB", PAIR_GAP)
    return CommitBoard(
        selection=list(tracks),
        tracks=list(tracks),
        netclasses={"USB_D+": usb, "USB_D-": usb},
    )


def test_end_gaps_give_way_to_the_pair_s_gap_at_the_ends(
    connect: Connector,
) -> None:
    board = connect.to(stubbed_pair_board())

    run()

    p, n = (unary_union(lines) for lines in by_net(board).values())
    middle = box(END_REACH, -FAR, STUBBED_LENGTH - END_REACH, FAR)
    inner = p.intersection(middle)
    assert not inner.is_empty
    gaps = [
        n.distance(inner.interpolate(d)) - PAIR_WIDTH
        for d in np.linspace(0.0, inner.length, END_SAMPLES)
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
    usb = netclass("USB", PAIR_CLEARANCE)
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
        # The bus at the default settings takes about 12 s; the coarse bus
        # and the other layouts at the default settings cover it. Those take
        # 0.6 to 3 s each, slow too.
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
        # The end edges take up any difference of the ends' own gaps.
        for k in range(1, len(a) - 2):
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
def test_a_pair_whose_tracks_cross_is_committed_for_the_drc_to_report(
    connect: Connector, monkeypatch: pytest.MonkeyPatch, log: list[str]
) -> None:
    real = pipeline.simplify_chains

    def folded(
        chains: Sequence[ChainInput],
        run: RunResult,
        tolerance: float,
        *,
        sections: Mapping[int, PairBlock],
    ) -> list[FloatArray]:
        result = real(chains, run, tolerance, sections=sections)
        for i, block in sections.items():
            optimised = run.chains[i].polyline
            last = len(optimised) - 1
            # Some order of a few of its vertices where its tracks cross.
            result[i] = next(
                kept
                for j, k in itertools.permutations(range(1, last), 2)
                if LineString(
                    (tracks := block.offset(kept := optimised[[0, j, k, last]]))[0]
                ).intersects(LineString(tracks[1]))
            )
        return result

    monkeypatch.setattr(pipeline, "simplify_chains", folded)
    board = connect.to(pair_board(extra=FOLD_KEEPERS))

    run()

    assert board.calls == COMMIT_CALLS
    assert not any(line.startswith("ERROR") for line in log)


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
    usb = netclass("USB", PAIR_GAP)
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
    # Simplification on a pair whose legs face each other, end to end, at
    # coarse settings; test_optimize_pairs covers the optimizer on the same
    # scenario.
    monkeypatch.setattr(pipeline, "optimize_chains", at(COARSE))
    board = connect.to(v_board(150.0))

    run(f"--repulsion={V_REPULSION}")

    ((pair,),) = (r.pairs for r in reports)
    assert pair.gap_after == pytest.approx((PAIR_GAP, PAIR_GAP), abs=REPORTED)
    p, n = (unary_union(lines) for lines in by_net(board).values())
    assert p.distance(n) - PAIR_WIDTH >= PAIR_GAP - NM_ROUNDING


@pytest.mark.parametrize(
    ("turn", "width", "gap", "clearance"),
    [(150.0, 0.2, 0.1, 0.2), (135.0, 0.1, 0.1, 0.3), (150.0, 0.1, 0.1, 0.3)],
)
def test_a_sharp_pair_held_as_routed_commits_nothing(
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
    usb = netclass("USB", clearance)
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


def pad_inside(at: tuple[float, float]) -> BaseGeometry:
    """Where a vertex keeping the fake board's pad at `at` may lie."""
    x, y = at
    h = PAD_HALF_SIZE - CONTACT_MARGIN
    return box(x - h, y - h, x + h, y + h)


def committed_route(board: CommitBoard) -> LineString:
    """The created tracks, joined into one route."""
    joined = linemerge([segment(t) for t in cast(list[kbt.Track], board.created)])
    assert isinstance(joined, LineString)
    return joined


@pytest.mark.usefixtures("coarse_optimizer")
class TestContacts:
    def test_a_tap_ending_in_a_pad_on_the_trunk_stays_on_the_trunk(
        self, connect: Connector
    ) -> None:
        # The tap's end lies in a pad and its copper on the trunk's: the pad
        # keeps it from joining the trunk, so it must not slide off it. The
        # ends lie off their pads' anchors, so that they may slide.
        route = [
            kipy_track("trunk", "SIG", (0, 0), (10, 0), width=WIDTH),
            kipy_track("tap", "SIG", (5, 0.2), (5, 3), width=WIDTH),
        ]
        pads = [
            kipy_pad(f"p{k}", "SIG", *at)
            for k, at in enumerate([(0.1, 0), (9.9, 0), (5, 0.4), (5, 2.9)])
        ]
        board = connect.to(
            CommitBoard(selection=list(route), tracks=list(route), pads=pads)
        )

        run()

        merged = linemerge([segment(t) for t in cast(list[kbt.Track], board.created)])
        assert isinstance(merged, MultiLineString)
        tap_end = Point(5, 0.2)
        # The tap still ends where it did, and its copper reaches the trunk's.
        ending = [
            tap_end.equals(Point(line.coords[0]))
            or tap_end.equals(Point(line.coords[-1]))
            for line in merged.geoms
        ]
        assert sorted(ending) == [False, True]
        trunk = merged.geoms[ending.index(False)]
        assert trunk.distance(tap_end) <= WIDTH

    def test_a_route_from_a_resistor_pad_slides_within_it_and_stays_connected(
        self, connect: Connector
    ) -> None:
        # One track lies inside pad 1, the corner where the second leaves it
        # on the pad's edge: pinned at both, the route could not move. The
        # route starts off the pad's anchor, so that its end may slide.
        pad = (-0.05, 0.0)
        edge = pad[0] + PAD_HALF_SIZE
        route = [
            kipy_track("in", "SIG", (0, 0), (edge, 0), width=WIDTH),
            kipy_track("out", "SIG", (edge, 0), (3, 3), width=WIDTH),
        ]
        board = connect.to(
            CommitBoard(
                selection=list(route),
                tracks=list(route),
                pads=[kipy_pad("r1", "SIG", *pad)],
            )
        )

        run()

        assert board.calls == COMMIT_CALLS
        assert sorted(board.deleted) == ["in", "out"]
        joined = committed_route(board)
        start, end = joined.coords[0], joined.coords[-1]
        assert end == (3, 3) or start == (3, 3)
        near = start if end == (3, 3) else end
        assert pad_inside(pad).covers(Point(near))
        assert near != (0, 0)

    def test_a_route_passing_over_a_pad_keeps_a_vertex_inside_it(
        self, connect: Connector
    ) -> None:
        # The corner lies in the pad, off its anchor.
        route = [
            kipy_track("a", "SIG", (0, 0), (5, 1), width=WIDTH),
            kipy_track("b", "SIG", (5, 1), (10, 0), width=WIDTH),
        ]
        board = connect.to(
            CommitBoard(
                selection=list(route),
                tracks=list(route),
                pads=[kipy_pad("p", "SIG", 5, 0.9)],
            )
        )

        run()

        assert board.calls == COMMIT_CALLS
        joined = committed_route(board)
        assert {joined.coords[0], joined.coords[-1]} == {(0, 0), (10, 0)}
        inside = pad_inside((5, 0.9))
        kept = [c for c in joined.coords[1:-1] if inside.covers(Point(c))]
        assert kept
        assert joined.length < 2 * math.hypot(5, 1)

    @pytest.mark.parametrize("kind", ["pad", "via"])
    def test_an_end_on_an_anchor_commits_exactly_there(
        self, connect: Connector, kind: str
    ) -> None:
        # KiCad drags only the track ends at a pad's anchor or a via's
        # centre along with it; this one lies off the millimetre grid.
        anchor = (4.000_123_457, 3.999_876_543)
        route = [
            kipy_track("a", "SIG", (0, 0), (4, 0), width=WIDTH),
            kipy_track("b", "SIG", (4, 0), anchor, width=WIDTH),
        ]
        board = connect.to(anchored_board(route, kind, anchor))

        run()

        assert board.calls == COMMIT_CALLS
        assert (nm(anchor[0]), nm(anchor[1])) in ends(created_on(board, BL.BL_F_Cu))

    @pytest.mark.parametrize("kind", ["pad", "via"])
    def test_a_route_through_an_anchor_keeps_its_vertex_there(
        self, connect: Connector, kind: str
    ) -> None:
        # The corner before the anchor is smoothed; the anchor stays.
        anchor = (5.000_123_457, 1.000_987_654)
        route = [
            kipy_track("a", "SIG", (0, 0), (3, 0), width=WIDTH),
            kipy_track("b", "SIG", (3, 0), anchor, width=WIDTH),
            kipy_track("c", "SIG", anchor, (10, 0), width=WIDTH),
        ]
        board = connect.to(anchored_board(route, kind, anchor))

        run()

        assert board.calls == COMMIT_CALLS
        assert (nm(anchor[0]), nm(anchor[1])) in ends(created_on(board, BL.BL_F_Cu))
        drawn = 3 + math.dist((3, 0), anchor) + math.dist(anchor, (10, 0))
        assert committed_route(board).length < drawn

    def test_a_junction_at_a_via_s_centre_keeps_every_branch_there(
        self, connect: Connector
    ) -> None:
        centre = (5.000_123_457, 5.000_987_654)
        route = [
            kipy_track("a1", "SIG", (0, 0), (0, 5), width=WIDTH),
            kipy_track("a2", "SIG", (0, 5), centre, width=WIDTH),
            kipy_track("b1", "SIG", (10, 0), (10, 5), width=WIDTH),
            kipy_track("b2", "SIG", (10, 5), centre, width=WIDTH),
            kipy_track("c1", "SIG", (2, 10), (5, 10), width=WIDTH),
            kipy_track("c2", "SIG", (5, 10), centre, width=WIDTH),
        ]
        board = connect.to(anchored_board(route, "via", centre))

        run()

        assert board.calls == COMMIT_CALLS
        at = (nm(centre[0]), nm(centre[1]))
        reaching = [
            t
            for t in created_on(board, BL.BL_F_Cu)
            if at in {(t.start.x, t.start.y), (t.end.x, t.end.y)}
        ]
        assert len(reaching) == 3

    def test_a_collinear_vertex_on_an_anchor_survives_simplification(
        self, connect: Connector
    ) -> None:
        # Merging the straight run would drop the vertex on the pad's anchor.
        anchor = (5.000_123_457, 0.0)
        route = [
            kipy_track("a", "SIG", (0, 0), anchor, width=WIDTH),
            kipy_track("b", "SIG", anchor, (10, 0), width=WIDTH),
            kipy_track("c", "SIG", (10, 0), (10, 5), width=WIDTH),
        ]
        board = connect.to(anchored_board(route, "pad", anchor))

        run()

        assert board.calls == COMMIT_CALLS
        assert (nm(anchor[0]), nm(anchor[1])) in ends(created_on(board, BL.BL_F_Cu))

    def test_a_pair_keeps_its_vertex_on_a_pad_anchor_in_a_straight(
        self, connect: Connector, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A pad no wider than the track, so that it keeps clear of the other.
        monkeypatch.setattr(conftest, "PAD_HALF_SIZE", PAIR_WIDTH / 2)
        y = 3.0 + PAIR_HALF_PITCH
        anchor = (PADS_AT, y)
        board = pair_board()
        (last,) = [t for t in board.tracks if t.id.value == "p4"]
        start = (last.start.x / NM_PER_MM, last.start.y / NM_PER_MM)
        split = [
            kipy_track("p4", "USB_D+", start, anchor, width=PAIR_WIDTH),
            kipy_track("p5", "USB_D+", anchor, (13.0, y), width=PAIR_WIDTH),
        ]
        board.tracks = [*(t for t in board.tracks if t is not last), *split]
        board.selection = [*(t for t in board.selection if t is not last), *split]
        board.pads = [kipy_pad("pad", "USB_D+", *anchor)]
        connect.to(board)

        run()

        assert board.calls == COMMIT_CALLS
        plus = [t for t in created_on(board, BL.BL_F_Cu) if t.net.name == "USB_D+"]
        assert (nm(anchor[0]), nm(anchor[1])) in ends(plus)
        low, high = coupled_gaps(*by_net(board).values())
        assert PAIR_GAP - GAP_SLACK <= low <= high <= PAIR_GAP + GAP_SLACK


def anchored_board(
    route: Sequence[kbt.Track], kind: str, at: tuple[float, float]
) -> CommitBoard:
    """`route`, selected, with a SIG pad or via anchored `at`."""
    pads = [kipy_pad("p", "SIG", *at)] if kind == "pad" else []
    vias = [kipy_via("v", "SIG", *at)] if kind == "via" else []
    return CommitBoard(selection=list(route), tracks=list(route), pads=pads, vias=vias)
