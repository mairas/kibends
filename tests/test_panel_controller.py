from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import cast

import kipy.board_types as kbt
import pytest
from conftest import (
    BOARD_FILE,
    COMMIT_CALLS,
    CommitBoard,
    FakeBoard,
    FakeKiCad,
    Ticking,
    kipy_group,
)
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from kipy.proto.common.types.base_types_pb2 import DocumentSpecifier
from test_cli import COARSE, at, l_board
from test_panel_probe import TRACKS, VIAS, prober, ten_tracks_four_vias
from test_present import chain, report

from kibends import present
from kibends import run as pipeline
from kibends.commit import BoardChangedError
from kibends.optimize import Progress
from kibends.panel.controller import Controller, RunSmooth, Spawn, State
from kibends.panel.probe import BoardId, Probe
from kibends.present import BUSY_MESSAGE, PanelFields
from kibends.results import Termination
from kibends.run import (
    ABNORMAL_MESSAGE,
    OptimizerReport,
    RunReport,
    ScopeOption,
    SmoothOptions,
    Stage,
)
from kibends.snapshot import SelectionError

STAGES = (
    Stage.READING,
    Stage.OBSTACLES,
    Stage.OPTIMIZING,
    Stage.SIMPLIFYING,
    Stage.COMMITTING,
)
PROGRESS = (
    Progress(0.5, 1, 9.0, 0, 1),
    Progress(1.0, 2, 4.0, 0, 1),
    Progress(1.5, 3, 3.5, 0, 1),
)
"""A run at one spacing."""
LEVELLED = (
    Progress(0.5, 1, 9.0, 0, 3),
    Progress(1.0, 2, 4.0, 0, 3),
    Progress(1.5, 3, 6.0, 1, 3),
    Progress(2.0, 4, 5.0, 1, 3),
    Progress(2.5, 5, 5.5, 2, 3),
    Progress(3.0, 6, 5.2, 2, 3),
)
"""A run through three levels, its energy jumping as it enters each finer
one."""
CHANGED = report((chain(10.0, 7.5),))
STOPPED = replace(CHANGED, termination=Termination.STOPPED)
UNCHANGED_STOP = replace(report(), termination=Termination.STOPPED)
EDITED = PanelFields(ground_nets="GND3", time_limit="5", repulsion="0.7")


class Panic(BaseException):
    """Like the `PanicException` a Rust panic raises: no `Exception`."""


@dataclass
class Scripted:
    """A run that reports `stages` and, while optimizing, `progress`, calls
    `during` after each stage and ends with `outcome`."""

    outcome: RunReport | BaseException = CHANGED
    stages: Sequence[Stage] = STAGES
    progress: Sequence[Progress] = PROGRESS
    during: Callable[[Stage], None] = lambda _stage: None
    options: list[SmoothOptions] = field(default_factory=list[SmoothOptions])
    stop: Callable[[], bool] | None = None

    def __call__(
        self,
        options: SmoothOptions,
        on_optimized: Callable[[OptimizerReport], None] = lambda _report: None,
        *,
        stop: Callable[[], bool] | None = None,
        on_stage: Callable[[Stage], None] = lambda _stage: None,
        on_progress: Callable[[Progress], None] | None = None,
    ) -> RunReport:
        self.options.append(options)
        self.stop = stop
        for stage in self.stages:
            on_stage(stage)
            self.during(stage)
            if stage is Stage.OPTIMIZING and on_progress is not None:
                for p in self.progress:
                    on_progress(p)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


@dataclass
class Manual:
    """A clock that reads what the test sets."""

    now: float = 0.0

    def __call__(self) -> float:
        return self.now


class Spawned:
    """Holds the work handed to worker threads until the test runs it."""

    def __init__(self) -> None:
        self.waiting: list[Callable[[], None]] = []
        self.daemons: list[bool] = []

    def __call__(self, work: Callable[[], None], *, daemon: bool) -> None:
        self.waiting.append(work)
        self.daemons.append(daemon)

    def finish(self) -> None:
        while self.waiting:
            self.waiting.pop(0)()


@dataclass
class Panel:
    """A controller whose workers run at once on the test's thread, unless
    `spawn` holds them, and whose every change is recorded in `seen` and
    told to `watchers`. Unless given a run, it runs `scripted`."""

    controller: Controller
    scripted: Scripted
    seen: list[tuple[State, str]] = field(default_factory=list[tuple[State, str]])
    watchers: list[Callable[[], None]] = field(default_factory=list[Callable[[], None]])
    closed: list[bool] = field(default_factory=list[bool])


def panel(
    board: FakeBoard | None = None,
    run: RunSmooth | None = None,
    *,
    spawn: Spawn = lambda work, daemon: work(),
    probe: Callable[[BoardId | None], Probe] | None = None,
    clock: Callable[[], float] | None = None,
) -> Panel:
    panels: list[Panel] = []

    def notify() -> None:
        p = panels[0]
        c = p.controller
        if not p.seen or p.seen[-1] != (c.state, c.status):
            p.seen.append((c.state, c.status))
        for watcher in p.watchers:
            watcher()

    scripted = Scripted()
    controller = Controller(
        post=lambda call: call(),
        spawn=spawn,
        probe=probe or prober(board or ten_tracks_four_vias()),
        run=run or scripted,
        clock=clock or Ticking(),
        notify=notify,
        closable=lambda: panels[0].closed.append(True),
    )
    panels.append(Panel(controller, scripted))
    return panels[0]


def opened(board: FakeBoard | None = None, run: RunSmooth | None = None) -> Panel:
    p = panel(board, run)
    p.controller.tick()
    p.seen.clear()
    return p


class TestOpening:
    def test_reads_the_board_before_it_runs(self) -> None:
        p = panel(spawn=Spawned())

        c = p.controller
        assert c.state is State.READING_BOARD
        assert c.status == "Reading board…"
        assert c.scope_line == "—"
        assert not c.button_enabled
        assert c.title == "kibends"

    def test_then_shows_the_board(self) -> None:
        c = opened().controller

        assert c.state is State.IDLE
        assert c.title == f"kibends — {BOARD_FILE}"
        assert c.scope_line == f"Whole board: {TRACKS} tracks, {VIAS} vias"
        assert (c.button_label, c.button_enabled) == ("Run", True)
        assert c.status == ""
        assert c.editable

    def test_fields_start_at_the_defaults(self) -> None:
        assert opened().controller.fields == PanelFields()


class TestScopeLine:
    def test_counts_the_selection_against_the_board(self) -> None:
        board = ten_tracks_four_vias()
        group = kipy_group("g", [cast(kbt.Track, board.tracks[2])])
        board.groups = [group]
        board.selection = [board.tracks[0], board.tracks[1], board.tracks[-1], group]

        c = opened(board).controller

        assert c.scope_line == f"4 of {TRACKS} tracks, 0 of {VIAS} vias"

    def test_follows_the_selection(self) -> None:
        board = ten_tracks_four_vias()
        p = panel(board, clock=Ticking(step=1.0))
        p.controller.tick()
        board.selection = [board.vias[0]]

        p.controller.tick()

        assert p.controller.scope_line == f"0 of {TRACKS} tracks, 1 of {VIAS} vias"

    def test_counts_one_in_the_singular(self) -> None:
        board = ten_tracks_four_vias()
        board.tracks = board.tracks[:1]
        board.vias = board.vias[:1]

        assert opened(board).controller.scope_line == "Whole board: 1 track, 1 via"

    def test_stays_while_kicad_is_busy(self) -> None:
        busy = ApiError("busy", code=ApiStatusCode.AS_BUSY)
        probes = Answers([ten_tracks_four_vias(), busy])
        p = panel(probe=probes, clock=Ticking())
        p.controller.tick()
        line = p.controller.scope_line

        p.controller.tick()

        assert p.controller.scope_line == line
        assert p.controller.status == ""

    def test_kicad_unreachable_shows_a_dash_and_says_so(self) -> None:
        lost = KiCadConnectionError("no socket")
        p = panel(probe=Answers([ten_tracks_four_vias(), lost]), clock=Ticking())
        p.controller.tick()

        p.controller.tick()

        c = p.controller
        assert c.scope_line == "—"
        assert c.status == present.error_message(lost, panel=True)
        assert c.button_enabled

    def test_kicad_unreachable_at_open_lets_run_retry(self) -> None:
        lost = KiCadConnectionError("no socket")
        p = panel(probe=Answers([lost]))

        p.controller.tick()

        c = p.controller
        assert c.state is State.IDLE
        assert c.scope_line == "—"
        assert c.status.startswith("Cannot reach KiCad")
        assert c.button_enabled

    def test_kicad_busy_at_open_keeps_reading(self) -> None:
        busy = ApiError("busy", code=ApiStatusCode.AS_BUSY)
        p = panel(probe=Answers([busy]))

        p.controller.tick()

        assert p.controller.state is State.READING_BOARD


class Answers:
    """A probe answering from boards, or raising errors, in turn; the last
    answers from then on."""

    def __init__(self, answers: Sequence[FakeBoard | BaseException]) -> None:
        self.answers = list(answers)
        self.calls = 0
        self.known: list[BoardId | None] = []

    def __call__(self, known: BoardId | None) -> Probe:
        self.known.append(known)
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        if isinstance(answer, BaseException):
            raise answer
        return prober(answer)(known)


class TestProbing:
    def test_probes_about_once_a_second(self) -> None:
        probes = Answers([ten_tracks_four_vias()])
        p = panel(probe=probes, clock=Ticking(step=0.25))

        for _ in range(8):
            p.controller.tick()

        assert probes.calls == 2

    def test_a_probe_in_flight_is_not_queued_behind(self) -> None:
        spawned = Spawned()
        probes = Answers([ten_tracks_four_vias()])
        p = panel(probe=probes, spawn=spawned, clock=Ticking(step=2.0))

        for _ in range(3):
            p.controller.tick()
        spawned.finish()

        assert probes.calls == 1

    def test_no_probes_while_running(self) -> None:
        probes = Answers([ten_tracks_four_vias()])
        p = panel(probe=probes, clock=Ticking(step=2.0))
        p.controller.tick()
        calls: list[int] = []

        def during(_stage: Stage) -> None:
            p.controller.tick()
            calls.append(probes.calls)

        p.scripted.during = during
        p.controller.run()

        assert calls == [1] * len(STAGES)

    def test_a_probe_answering_during_a_run_is_ignored(self) -> None:
        spawned = Spawned()
        board = ten_tracks_four_vias()
        p = panel(board, Scripted(stages=(Stage.READING,)), spawn=spawned)
        p.controller.tick()
        spawned.finish()
        p.controller.tick()  # in flight
        board.selection = [board.vias[0]]
        p.controller.run()
        line = p.controller.scope_line

        spawned.waiting[0]()  # the probe answers, mid-run

        assert p.controller.scope_line == line
        assert p.controller.state is State.RUNNING

    def test_a_reread_sent_before_a_run_and_answering_after_it_is_ignored(
        self,
    ) -> None:
        spawned = Spawned()
        probes = Answers([ten_tracks_four_vias()])
        p = panel(probe=probes, spawn=spawned, clock=Ticking(step=2.0))
        p.controller.tick()
        spawned.finish()
        p.controller.run()
        spawned.finish()
        p.controller.tick()  # the reread after the run, in flight
        p.controller.run()
        stale, run = spawned.waiting
        spawned.waiting.clear()
        run()

        stale()  # answers with the totals from before the commit
        p.controller.tick()
        spawned.finish()

        assert len(probes.known) == 3
        assert probes.known[-1] is None  # read again after the second run

    def test_probes_run_on_daemon_threads_and_runs_do_not(self) -> None:
        # A probe may be out when the window closes, and must not keep the
        # process alive; a run must end before the window closes.
        spawned = Spawned()
        p = panel(spawn=spawned)
        p.controller.tick()
        spawned.finish()

        p.controller.run()

        assert spawned.daemons == [True, False]

    def test_rereads_the_totals_after_a_run(self) -> None:
        probes = Answers([ten_tracks_four_vias()])
        p = panel(probe=probes, clock=Ticking(step=2.0))
        p.controller.tick()
        p.controller.tick()

        p.controller.run()
        p.controller.tick()

        assert probes.known[1] is not None
        assert probes.known[2] is None


class TestBoardSwitch:
    def switched(self, fields: PanelFields) -> Panel:
        board = ten_tracks_four_vias()
        p = panel(board, clock=Ticking(step=2.0))
        p.controller.tick()
        p.controller.edit(fields)
        board.document = DocumentSpecifier(board_filename="other.kicad_pcb")
        board.tracks = board.tracks[:3]
        p.controller.tick()
        return p

    def test_updates_title_and_totals(self) -> None:
        c = self.switched(PanelFields()).controller

        assert c.title == "kibends — other.kicad_pcb"
        assert c.scope_line == f"Whole board: 3 tracks, {VIAS} vias"

    def test_clears_the_ground_nets_and_keeps_the_rest(self) -> None:
        c = self.switched(EDITED).controller

        assert c.fields == replace(EDITED, ground_nets="")

    def test_a_run_in_progress_is_unaffected(self) -> None:
        board = ten_tracks_four_vias()
        p = panel(board, clock=Ticking(step=2.0))
        p.controller.tick()

        def during(stage: Stage) -> None:
            if stage is Stage.OPTIMIZING:
                board.document = DocumentSpecifier(board_filename="other.kicad_pcb")
                p.controller.tick()

        p.scripted.during = during
        p.controller.run()

        assert p.controller.title == f"kibends — {BOARD_FILE}"


class TestGroundNets:
    def test_placeholder_names_the_detected_nets(self) -> None:
        board = ten_tracks_four_vias()
        board.vias = [*board.vias]  # GND has copper: the fallback

        assert opened(board).controller.ground_placeholder == "GND (detected)"

    def test_placeholder_says_when_none_are_found(self) -> None:
        board = ten_tracks_four_vias()
        board.vias = []

        assert opened(board).controller.ground_placeholder == "none detected"

    def test_an_empty_field_leaves_detection_to_the_run(self) -> None:
        run = Scripted()
        opened(run=run).controller.run()

        assert run.options[0].ground_nets == ()

    def test_typed_names_are_used(self) -> None:
        run = Scripted()
        c = opened(run=run).controller
        c.edit(PanelFields(ground_nets="GND, AGND"))

        c.run()

        assert run.options[0].ground_nets == ("GND", "AGND")


class TestValidation:
    def test_an_invalid_field_disables_run_and_says_why(self) -> None:
        c = opened().controller

        c.edit(PanelFields(repulsion="-1"))

        assert not c.button_enabled
        assert c.status == "Repulsion: must not be negative"
        assert c.invalid == {present.Field.REPULSION}

    def test_every_invalid_field_is_named(self) -> None:
        c = opened().controller

        c.edit(PanelFields(repulsion="", time_limit="0"))

        assert c.status.splitlines() == [
            "Time limit: must be positive",
            "Repulsion: needs a value",
        ]

    def test_run_does_nothing_while_a_field_is_invalid(self) -> None:
        run = Scripted()
        c = opened(run=run).controller
        c.edit(PanelFields(falloff_range="x"))

        c.run()

        assert run.options == []
        assert c.state is State.IDLE

    def test_fixing_the_field_enables_run(self) -> None:
        c = opened().controller
        c.edit(PanelFields(repulsion="-1"))

        c.edit(PanelFields(repulsion="1"))

        assert c.button_enabled
        assert c.status == ""


class TestRun:
    def test_goes_through_its_stages_to_done(self) -> None:
        p = opened()

        p.controller.run()

        assert p.seen == [
            (State.RUNNING, "Reading selection…"),
            (State.RUNNING, "Building obstacles…"),
            (State.RUNNING, "Optimizing"),
            (State.RUNNING, "Simplifying…"),
            (State.COMMITTING, "Committing"),
            (State.DONE, present.summary(CHANGED)),
        ]

    def test_runs_the_selection_else_the_board_with_the_fields(self) -> None:
        run = Scripted()
        c = opened(run=run).controller
        c.edit(EDITED)

        c.run()

        options = run.options[0]
        assert options.scope is ScopeOption.SELECTION_ELSE_BOARD
        assert options.settings.time_limit == 5.0
        assert options.params.repulsion == 0.7

    def test_shows_progress_and_the_energy_curve(self) -> None:
        clock = Ticking(step=0.5)
        p = panel(run=Scripted(), clock=clock)
        p.controller.tick()

        p.controller.run()

        c = p.controller
        assert c.energies == tuple(e.energy for e in PROGRESS)
        assert c.progress_line.endswith(" s · iteration 3")
        assert c.separators == ()

    def test_marks_each_level_change_and_names_the_level(self) -> None:
        p = panel(run=Scripted(progress=LEVELLED), clock=Ticking(step=0.5))
        p.controller.tick()

        p.controller.run()

        c = p.controller
        assert c.energies == tuple(e.energy for e in LEVELLED)
        # Before the third and the fifth value, the first at their level.
        assert c.separators == (2, 4)
        assert c.progress_line.endswith(" s · level 3 of 3 · iteration 6")

    def test_names_the_level_as_it_runs(self) -> None:
        lines: list[str] = []
        p = opened()
        p.scripted.progress = LEVELLED[:3]
        p.scripted.during = lambda _stage: lines.append(p.controller.progress_line)

        p.controller.run()

        # After the stage that optimized.
        assert lines[3].endswith(" s · level 2 of 3 · iteration 3")

    def test_a_new_run_starts_without_separators(self) -> None:
        p = opened()
        p.scripted.progress = LEVELLED
        p.controller.run()
        p.scripted.progress = PROGRESS

        p.controller.run()

        assert p.controller.separators == ()

    def test_progress_counts_from_the_runs_start(self) -> None:
        lines: list[str] = []
        clock = Manual()
        p = panel(clock=clock)
        p.controller.tick()

        def during(stage: Stage) -> None:
            if stage is Stage.READING:
                clock.now += 1.5
                lines.append(p.controller.progress_line)
            if stage is Stage.SIMPLIFYING:
                clock.now += 1.0

        p.scripted.during = during
        clock.now = 10.0
        p.controller.run()
        clock.now += 5.0

        assert lines == ["1.5 s"]
        assert p.controller.progress_line == "2.5 s · iteration 3"

    def test_fields_are_read_only_while_running(self) -> None:
        editable: list[bool] = []
        p = opened()

        def during(_stage: Stage) -> None:
            editable.append(p.controller.editable)
            p.controller.edit(EDITED)

        p.scripted.during = during
        p.controller.run()

        assert editable == [False] * len(STAGES)
        assert p.controller.fields == PanelFields()
        assert p.controller.editable

    def test_the_button_stops_while_running(self) -> None:
        buttons: list[tuple[str, bool]] = []
        p = opened()

        def during(_stage: Stage) -> None:
            buttons.append((p.controller.button_label, p.controller.button_enabled))

        p.scripted.during = during
        p.controller.run()

        # Disabled from simplifying on, when nothing is left to stop.
        assert buttons == [("Stop", True)] * 3 + [("Stop", False)] * 2
        assert (p.controller.button_label, p.controller.button_enabled) == (
            "Run",
            True,
        )

    def test_the_scope_line_holds_what_the_run_reads(self) -> None:
        board = ten_tracks_four_vias()
        lines: list[str] = []
        p = panel(board, clock=Ticking(step=2.0))
        p.controller.tick()

        def during(_stage: Stage) -> None:
            board.selection = [board.vias[0]]
            p.controller.tick()
            lines.append(p.controller.scope_line)

        p.scripted.during = during
        p.controller.run()

        assert set(lines) == {f"Whole board: {TRACKS} tracks, {VIAS} vias"}

    def test_nothing_to_do(self) -> None:
        p = opened(run=Scripted(outcome=report(), stages=STAGES[:-1]))

        p.controller.run()

        assert p.seen[-1] == (State.DONE, "Nothing to do")
        assert State.COMMITTING not in [s for s, _ in p.seen]

    def test_runs_again_from_done(self) -> None:
        run = Scripted()
        c = opened(run=run).controller
        c.run()

        c.run()

        assert len(run.options) == 2
        assert c.state is State.DONE


class TestStop:
    def test_stops_and_commits_the_best_so_far(self) -> None:
        p = opened()

        def during(stage: Stage) -> None:
            if stage is Stage.OPTIMIZING:
                p.controller.stop()

        run = p.scripted
        run.outcome = STOPPED
        run.during = during
        p.controller.run()

        assert run.stop is not None and run.stop()
        assert p.seen[-4:] == [
            (State.RUNNING, "Optimizing"),
            (State.STOPPING, "Stopping…"),
            (State.COMMITTING, "Committing"),
            (State.DONE, present.summary(STOPPED)),
        ]
        assert p.controller.status.startswith("Stopped after")

    def test_the_button_says_stopping(self) -> None:
        buttons: list[tuple[str, bool]] = []
        p = opened()

        def during(stage: Stage) -> None:
            if stage is Stage.OPTIMIZING:
                p.controller.stop()
            buttons.append((p.controller.button_label, p.controller.button_enabled))

        p.scripted.outcome = STOPPED
        p.scripted.stages = STAGES[:-1]
        p.scripted.during = during
        p.controller.run()

        assert buttons[-2:] == [("Stopping…", False)] * 2

    def test_progress_freezes_while_stopping(self) -> None:
        p = opened()

        def during(stage: Stage) -> None:
            if stage is Stage.OBSTACLES:
                p.controller.stop()

        p.scripted.outcome = STOPPED
        p.scripted.during = during
        p.scripted.progress = LEVELLED
        p.controller.run()

        assert p.controller.energies == ()
        assert p.controller.separators == ()

    def test_before_optimizing_nothing_changes(self) -> None:
        p = opened()

        def during(stage: Stage) -> None:
            if stage is Stage.OBSTACLES:
                p.controller.stop()

        p.scripted.outcome = UNCHANGED_STOP
        p.scripted.stages = STAGES[:2]
        p.scripted.during = during
        p.controller.run()

        assert p.seen[-1] == (State.DONE, "Stopped; nothing changed")
        assert State.COMMITTING not in [s for s, _ in p.seen]

    def test_stop_while_simplifying_does_nothing(self) -> None:
        p = opened()
        states: list[State] = []

        def during(stage: Stage) -> None:
            if stage is Stage.SIMPLIFYING:
                p.controller.stop()
                states.append(p.controller.state)

        run = p.scripted
        run.during = during
        p.controller.run()

        assert states == [State.RUNNING]
        assert run.stop is not None and not run.stop()
        assert p.controller.status.startswith("Settled after")

    def test_stop_outside_a_run_does_nothing(self) -> None:
        c = opened().controller

        c.stop()

        assert c.state is State.IDLE


class TestErrors:
    @pytest.mark.parametrize(
        "error",
        [
            ApiError("busy", code=ApiStatusCode.AS_BUSY),
            BoardChangedError("the board changed under the run; nothing committed"),
            SelectionError("selection has no tracks or vias"),
            KiCadConnectionError("no socket"),
        ],
    )
    def test_a_failure_says_why_and_keeps_the_fields(self, error: Exception) -> None:
        c = opened(run=Scripted(outcome=error)).controller
        c.edit(EDITED)

        c.run()

        assert c.state is State.FAILED
        assert c.status == present.error_message(error, panel=True)
        assert c.fields == EDITED
        assert (c.button_label, c.button_enabled) == ("Run", True)
        assert c.editable

    def test_busy_says_to_close_the_tool(self) -> None:
        busy = ApiError("busy", code=ApiStatusCode.AS_BUSY)
        c = opened(run=Scripted(outcome=busy)).controller

        c.run()

        assert c.status == BUSY_MESSAGE

    def test_an_unexpected_error_is_named_and_logged(self, log: list[str]) -> None:
        c = opened(run=Scripted(outcome=RuntimeError("boom"))).controller

        c.run()

        assert c.status == "Unexpected error: RuntimeError: boom"
        assert any("Traceback" in line for line in log)

    def test_a_panic_fails_the_run_and_is_logged(self, log: list[str]) -> None:
        c = opened(run=Scripted(outcome=Panic("boom"))).controller

        c.run()

        assert c.state is State.FAILED
        assert c.status == "Unexpected error: Panic: boom"
        assert any("Traceback" in line for line in log)
        assert c.close()

    def test_a_panic_in_a_probe_lets_the_next_probe_run(self, log: list[str]) -> None:
        probes = Answers([Panic("boom"), ten_tracks_four_vias()])
        p = panel(probe=probes, clock=Ticking(step=2.0))

        p.controller.tick()
        assert p.controller.state is State.IDLE
        assert p.controller.status == "Unexpected error: Panic: boom"
        assert any("Traceback" in line for line in log)
        p.controller.tick()

        assert probes.calls == 2
        assert p.controller.status == ""

    def test_an_abnormal_stop_names_the_advanced_fields(self) -> None:
        refused = replace(CHANGED, committed=False, refusal=ABNORMAL_MESSAGE)
        c = opened(run=Scripted(outcome=refused)).controller

        c.run()

        assert c.state is State.FAILED
        assert c.status == present.PANEL_ABNORMAL_MESSAGE

    def test_a_runs_failure_replaces_an_earlier_probe_problem(self) -> None:
        lost = KiCadConnectionError("no socket")
        p = panel(probe=Answers([lost]), run=Scripted(outcome=SelectionError("bad")))
        p.controller.tick()

        p.controller.run()

        assert p.controller.status == "bad"

    def test_a_probe_that_reaches_kicad_again_clears_its_problem(self) -> None:
        lost = KiCadConnectionError("no socket")
        board = ten_tracks_four_vias()
        p = panel(probe=Answers([board, lost, board]), clock=Ticking(step=2.0))
        for _ in range(3):
            p.controller.tick()

        assert p.controller.status == ""
        assert p.controller.scope_line == f"Whole board: {TRACKS} tracks, {VIAS} vias"


class TestClose:
    def test_closes_at_once_when_idle(self) -> None:
        p = opened()

        assert p.controller.close()

    def test_while_running_stops_and_closes_after_the_commit(self) -> None:
        p = opened()
        answers: list[bool] = []

        def during(stage: Stage) -> None:
            if stage is Stage.OPTIMIZING:
                answers.append(p.controller.close())

        run = p.scripted
        run.outcome = STOPPED
        run.during = during
        p.controller.run()

        assert answers == [False]
        assert run.stop is not None and run.stop()
        assert p.closed == [True]
        assert p.controller.close()

    def test_while_committing_waits_for_the_push(self) -> None:
        p = opened()
        answers: list[bool] = []
        stops: list[bool] = []

        def during(stage: Stage) -> None:
            if stage is Stage.COMMITTING:
                answers.append(p.controller.close())
                assert run.stop is not None
                stops.append(run.stop())

        run = p.scripted
        run.during = during
        p.controller.run()

        assert answers == [False]
        assert stops == [False]
        assert p.closed == [True]

    def test_after_a_failure_mid_close_it_closes(self) -> None:
        p = opened()

        def during(stage: Stage) -> None:
            if stage is Stage.OPTIMIZING:
                p.controller.close()

        p.scripted.outcome = KiCadConnectionError("gone")
        p.scripted.during = during
        p.controller.run()

        assert p.closed == [True]

    def test_no_probes_after_closing(self) -> None:
        probes = Answers([ten_tracks_four_vias()])
        p = panel(probe=probes, clock=Ticking(step=2.0))
        p.controller.tick()
        p.controller.close()

        p.controller.tick()

        assert probes.calls == 1


@pytest.fixture
def connected(monkeypatch: pytest.MonkeyPatch) -> Callable[[CommitBoard], Panel]:
    """A panel running the real pipeline, at coarse settings, on a board."""
    monkeypatch.setattr(pipeline, "optimize_chains", at(COARSE))

    def connect(board: CommitBoard) -> Panel:
        def kicad(**_options: object) -> FakeKiCad:
            return FakeKiCad(board)

        monkeypatch.setattr(pipeline.kipy, "KiCad", kicad)
        p = panel(
            board,
            pipeline.run_smooth,
            probe=prober(board),
        )
        p.controller.tick()
        p.seen.clear()
        return p

    return connect


class TestPipeline:
    def test_a_run_smooths_and_commits_the_selection(
        self, connected: Callable[[CommitBoard], Panel]
    ) -> None:
        board = l_board()
        p = connected(board)

        p.controller.run()

        assert board.calls == COMMIT_CALLS
        assert [s for s, _ in p.seen] == [State.RUNNING] * 4 + [
            State.COMMITTING,
            State.DONE,
        ]
        assert p.controller.status.startswith("Settled after")
        assert p.controller.status.endswith("selection of 2 tracks, 0 vias")
        assert p.controller.energies

    def test_a_stop_while_optimizing_commits_the_best_so_far(
        self, connected: Callable[[CommitBoard], Panel]
    ) -> None:
        board = l_board()
        p = connected(board)
        stopped: list[bool] = []

        def notify() -> None:
            c = p.controller
            if c.state is State.RUNNING and c.energies and not stopped:
                stopped.append(True)
                c.stop()

        p.watchers.append(notify)
        p.controller.run()

        assert board.calls == COMMIT_CALLS
        assert p.controller.status.startswith("Stopped after")

    def test_a_stop_before_optimizing_commits_nothing(
        self, connected: Callable[[CommitBoard], Panel]
    ) -> None:
        board = l_board()
        p = connected(board)

        def notify() -> None:
            c = p.controller
            if c.status == "Building obstacles…":
                c.stop()

        p.watchers.append(notify)
        p.controller.run()

        assert board.calls == []
        assert p.controller.status == "Stopped; nothing changed"

    def test_a_connection_lost_in_the_push_may_have_landed(
        self, connected: Callable[[CommitBoard], Panel]
    ) -> None:
        board = l_board(push_error=KiCadConnectionError("gone"))
        p = connected(board)

        p.controller.run()

        assert p.controller.state is State.FAILED
        assert "may have landed" in p.controller.status

    def test_an_empty_selection_runs_the_whole_board(
        self, connected: Callable[[CommitBoard], Panel]
    ) -> None:
        board = l_board()
        board.selection = []
        p = connected(board)

        p.controller.run()

        assert p.controller.status.endswith("whole board of 2 tracks, 0 vias")
