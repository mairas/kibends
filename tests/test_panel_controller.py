import threading
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
    kipy_copper_zone,
    kipy_footprint,
    kipy_group,
    kipy_rect,
)
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from kipy.proto.common.types.base_types_pb2 import DocumentSpecifier
from loguru import logger
from test_cli import COARSE, at, l_board
from test_panel_probe import TRACKS, VIAS, prober, ten_tracks_four_vias
from test_present import chain, report, via

from kibends import run as pipeline
from kibends.chain import ChainError
from kibends.commit import (
    BoardChangedError,
    CommitUncertainError,
    CreateItemsError,
)
from kibends.coords import UserFrame
from kibends.inside import InsideClearanceError, Violation
from kibends.optimize import Progress
from kibends.panel import controller as panel_controller
from kibends.panel.controller import (
    COMMITTING_HINT,
    IDLE_HINT,
    MINUS,
    NO_BOARD,
    NO_OPTIMIZATION,
    NO_RUN_YET,
    NOTHING_TO_OPTIMIZE,
    STOP_HINT,
    STOPPING_HINT,
    TOO_LATE_HINT,
    UNREACHABLE,
    Controller,
    Icon,
    Line,
    Outcome,
    RunSmooth,
    Spawn,
    State,
    duration,
    signed,
)
from kibends.panel.probe import BoardId, Probe
from kibends.present import Field, PanelFields
from kibends.results import Termination
from kibends.run import (
    ABNORMAL_MESSAGE,
    NoBoardError,
    OptimizerReport,
    RunReport,
    Scope,
    ScopeKind,
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
    Progress(0.0, 0, 20.0, 10.0, 0, 1),
    Progress(0.5, 1, 18.0, 9.0, 0, 1),
    Progress(1.0, 2, 15.0, 8.0, 0, 1),
    Progress(1.5, 3, 14.0, 7.5, 0, 1),
)
"""A run at one spacing, from its start."""
LEVELLED = (
    Progress(0.0, 0, 20.0, 10.0, 0, 3),
    Progress(0.5, 1, 18.0, 9.0, 0, 3),
    Progress(1.0, 2, 16.0, 8.0, 0, 3),
    Progress(1.5, 3, 16.5, 8.25, 1, 3),
    Progress(2.0, 4, 15.8, 7.9, 1, 3),
    Progress(2.5, 5, 15.9, 7.95, 2, 3),
    Progress(3.0, 6, 15.6, 7.8, 2, 3),
)
"""A run through the default schedule's three levels, its length jumping as
it enters each finer one."""
CHANGED = report((chain(10.0, 7.5),))
STOPPED = replace(CHANGED, termination=Termination.STOPPED)
UNCHANGED_STOP = replace(report(), termination=Termination.STOPPED, iterations=0)
EDITED = PanelFields(time_limit="5", repulsion="0.7")


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
            if stage is Stage.OPTIMIZING and on_progress is not None:
                for p in self.progress:
                    on_progress(p)
            self.during(stage)
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


def text(line: Line) -> str:
    return line.bold + line.rest


@dataclass
class Panel:
    """A controller whose workers run at once on the test's thread, unless
    `spawn` holds them, and whose every change of state and progress line
    is recorded in `seen`, each change told to `watchers`. Unless given a
    run, it runs `scripted`."""

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
        now = (c.state, text(c.progress_line))
        if not p.seen or p.seen[-1] != now:
            p.seen.append(now)
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


def outcome(c: Controller) -> Outcome:
    report = c.report
    assert isinstance(report, Outcome)
    return report


def rows(c: Controller) -> dict[str, str]:
    return dict(outcome(c).rows)


WHOLE_BOARD = Line("Whole board", f": {TRACKS} tracks, {VIAS} vias (nothing selected)")


class TestOpening:
    def test_reads_the_board_before_it_runs(self) -> None:
        p = panel(spawn=Spawned())

        c = p.controller
        assert c.state is State.READING_BOARD
        assert c.scope_line == Line("", "Connecting to KiCad…")
        assert c.progress_line == Line("", "Waiting for the board")
        assert c.report == IDLE_HINT
        assert not c.run_enabled and not c.stop_enabled
        assert c.title == "kibends"
        assert c.elapsed == ""

    def test_then_shows_the_board(self) -> None:
        c = opened().controller

        assert c.state is State.IDLE
        assert c.title == f"kibends — {BOARD_FILE}"
        assert c.scope_line == WHOLE_BOARD
        assert c.progress_line == Line("", "Ready")
        assert c.run_enabled and not c.stop_enabled
        assert c.run_tip == "Run on the scope shown at the top (Return)"
        assert c.report == IDLE_HINT
        assert c.graph.points == () and c.graph.placeholder == NO_RUN_YET
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

        assert c.scope_line == Line(
            "Selection", f": 4 of {TRACKS} tracks, 0 of {VIAS} vias"
        )

    def test_follows_the_selection(self) -> None:
        board = ten_tracks_four_vias()
        p = panel(board, clock=Ticking(step=1.0))
        p.controller.tick()
        board.selection = [board.vias[0]]

        p.controller.tick()

        assert p.controller.scope_line == Line(
            "Selection", f": 0 of {TRACKS} tracks, 1 of {VIAS} vias"
        )

    def test_a_selection_without_copper_says_what_to_select(self) -> None:
        board = ten_tracks_four_vias()
        board.selection = [kipy_footprint("U1")]

        c = opened(board).controller

        assert c.scope_line == Line(
            "Selection",
            f": 0 of {TRACKS} tracks, 0 of {VIAS} vias — select tracks or vias, "
            "or nothing for the whole board",
        )
        assert c.run_enabled

    def test_counts_one_in_the_singular(self) -> None:
        board = ten_tracks_four_vias()
        board.tracks = board.tracks[:1]
        board.vias = board.vias[:1]

        assert opened(board).controller.scope_line == Line(
            "Whole board", ": 1 track, 1 via (nothing selected)"
        )

    def test_stays_while_kicad_is_busy_and_says_so(self) -> None:
        busy = ApiError("busy", code=ApiStatusCode.AS_BUSY)
        p = panel(probe=Answers([ten_tracks_four_vias(), busy]), clock=Ticking())
        p.controller.tick()

        p.controller.tick()

        c = p.controller
        assert c.scope_line == replace(
            WHOLE_BOARD, rest=f"{WHOLE_BOARD.rest} (KiCad is busy)"
        )
        assert c.run_enabled

    def test_kicad_unreachable_says_so_and_disables_run(self) -> None:
        lost = KiCadConnectionError("no socket")
        p = panel(probe=Answers([ten_tracks_four_vias(), lost]), clock=Ticking())
        p.controller.tick()

        p.controller.tick()

        c = p.controller
        assert c.scope_line == Line("", UNREACHABLE)
        assert not c.run_enabled
        assert c.state is State.IDLE

    def test_kicad_unreachable_at_open_keeps_waiting(self) -> None:
        p = panel(probe=Answers([KiCadConnectionError("no socket")]))

        p.controller.tick()

        c = p.controller
        assert c.state is State.READING_BOARD
        assert c.scope_line == Line("", UNREACHABLE)
        assert not c.run_enabled

    def test_no_board_open_says_so_and_disables_run(self) -> None:
        none = NoBoardError("No board is open in KiCad's PCB editor (none)")
        p = panel(probe=Answers([ten_tracks_four_vias(), none]), clock=Ticking())
        p.controller.tick()

        p.controller.tick()

        c = p.controller
        assert c.scope_line == Line("", NO_BOARD)
        assert c.title == "kibends"
        assert not c.run_enabled

    def test_kicad_busy_at_open_keeps_reading(self) -> None:
        busy = ApiError("busy", code=ApiStatusCode.AS_BUSY)
        p = panel(probe=Answers([busy]))

        p.controller.tick()

        c = p.controller
        assert c.state is State.READING_BOARD
        assert c.scope_line == Line("", "Connecting to KiCad… (KiCad is busy)")

    def test_a_probe_that_reaches_kicad_again_clears_its_problem(self) -> None:
        lost = KiCadConnectionError("no socket")
        board = ten_tracks_four_vias()
        p = panel(probe=Answers([board, lost, board]), clock=Ticking(step=2.0))
        for _ in range(3):
            p.controller.tick()

        assert p.controller.scope_line == WHOLE_BOARD
        assert p.controller.run_enabled


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

    def test_a_probe_from_before_a_run_cannot_switch_boards_under_it(
        self,
    ) -> None:
        spawned = Spawned()
        board = ten_tracks_four_vias()
        p = panel(board, spawn=spawned)
        p.controller.tick()
        spawned.finish()
        p.controller.tick()  # in flight
        seen: list[tuple[int, str, str]] = []
        p.controller.run()
        probe, run = spawned.waiting
        spawned.waiting.clear()

        def during(stage: Stage) -> None:
            if stage is Stage.OPTIMIZING:
                board.document = DocumentSpecifier(board_filename="other.kicad_pcb")
                probe()  # answers from the other board, mid-run
                c = p.controller
                seen.append((len(c.graph.points), c.title, c.elapsed))

        p.scripted.during = during
        run()

        assert seen[0][:2] == (len(PROGRESS), f"kibends — {BOARD_FILE}")
        assert seen[0][2] != ""
        assert len(p.controller.graph.points) == len(PROGRESS)

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

    def test_rereads_the_totals_after_a_failed_probe(self) -> None:
        lost = KiCadConnectionError("no socket")
        board = ten_tracks_four_vias()
        probes = Answers([board, lost, board])
        p = panel(probe=probes, clock=Ticking(step=2.0))

        for _ in range(3):
            p.controller.tick()

        assert probes.known[1] is not None
        assert probes.known[2] is None

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
    def switched(self, typed: str | None = None, *, ran: bool = False) -> Panel:
        board = ten_tracks_four_vias()
        p = panel(board, clock=Ticking(step=2.0))
        p.controller.tick()
        p.controller.edit(EDITED)
        if typed is not None:
            p.controller.type_ground_nets(typed)
        if ran:
            p.controller.run()
        board.document = DocumentSpecifier(board_filename="other.kicad_pcb")
        board.tracks = board.tracks[:3]
        p.controller.tick()
        return p

    def test_updates_title_and_totals(self) -> None:
        c = self.switched().controller

        assert c.title == "kibends — other.kicad_pcb"
        assert c.scope_line == Line(
            "Whole board", f": 3 tracks, {VIAS} vias (nothing selected)"
        )

    def test_drops_a_typed_list_says_so_and_keeps_the_rest(self) -> None:
        c = self.switched("GND3").controller

        assert c.fields == EDITED
        assert c.ground.text == "GND"
        assert c.ground.caption == (
            "Detected for this board; your list was for the previous board"
        )
        assert not c.ground.use_detected

    def test_says_so_until_the_next_edit(self) -> None:
        c = self.switched("GND3").controller

        c.type_ground_nets("AGND")

        assert c.ground.caption == "Your list, not detection."

    def test_detection_carries_on_silently(self) -> None:
        c = self.switched().controller

        assert c.ground.caption.startswith("Detected: ")

    def test_clears_the_last_runs_graph_and_report(self) -> None:
        c = self.switched(ran=True).controller

        assert c.state is State.IDLE
        assert c.graph.points == () and c.graph.placeholder == NO_RUN_YET
        assert c.report == IDLE_HINT
        assert c.progress_line == Line("", "Ready")
        assert c.elapsed == ""

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


def inner_zones() -> FakeBoard:
    board = ten_tracks_four_vias()
    board.zones = [
        kipy_copper_zone(
            f"plane {net}", net, kipy_rect(0, 0, 10, 10), [BL.BL_In1_Cu, BL.BL_In2_Cu]
        )
        for net in ("GND", "AGND")
    ]
    return board


class TestGroundNets:
    def test_before_the_board_is_read_the_field_waits(self) -> None:
        ground = panel(spawn=Spawned()).controller.ground

        assert (ground.text, ground.placeholder, ground.enabled) == (
            "",
            "reading the board…",
            False,
        )

    def test_shows_the_nets_detected_on_inner_zones(self) -> None:
        ground = opened(inner_zones()).controller.ground

        assert (ground.text, ground.caption, ground.enabled, ground.use_detected) == (
            "AGND, GND",
            "Detected: zones on two or more inner layers",
            True,
            False,
        )

    def test_shows_the_fallback(self) -> None:
        ground = opened().controller.ground

        assert (ground.text, ground.caption) == (
            "GND",
            "Detected: GND (no net has zones on two inner layers)",
        )

    def test_says_when_none_are_found(self) -> None:
        board = ten_tracks_four_vias()
        board.vias = []

        ground = opened(board).controller.ground

        assert (ground.text, ground.placeholder, ground.caption) == (
            "",
            "none found — type net names",
            "No ground net found: vias are not drawn to return vias",
        )

    def test_detection_is_left_to_the_run(self) -> None:
        run = Scripted()
        c = opened(run=run).controller

        c.run()

        assert c.fields.ground_nets == ""
        assert run.options[0].ground_nets == ()

    def test_typing_makes_the_list_the_users(self) -> None:
        run = Scripted()
        c = opened(run=run).controller

        c.type_ground_nets("GND, AGND")
        c.run()

        assert (c.ground.text, c.ground.caption, c.ground.use_detected) == (
            "GND, AGND",
            "Your list, not detection.",
            True,
        )
        assert run.options[0].ground_nets == ("GND", "AGND")

    def test_use_detected_goes_back_to_detection(self) -> None:
        c = opened().controller
        c.type_ground_nets("AGND")

        c.use_detected()

        assert c.fields.ground_nets == ""
        assert c.ground.text == "GND"
        assert c.ground.caption.startswith("Detected: ")

    def test_leaving_it_empty_goes_back_to_detection(self) -> None:
        c = opened().controller
        c.type_ground_nets(" , ")

        c.leave(Field.GROUND_NETS)

        assert (c.ground.text, c.ground.use_detected) == ("GND", False)

    def test_leaving_it_lists_the_names_with_commas(self) -> None:
        c = opened().controller
        c.type_ground_nets("GND3  AGND,")

        c.leave(Field.GROUND_NETS)

        assert c.ground.text == "GND3, AGND"

    @pytest.mark.parametrize("attraction", ["0", "0.0"])
    def test_is_ignored_while_attraction_is_0(self, attraction: str) -> None:
        c = opened().controller

        c.edit(PanelFields(attraction=attraction))

        assert (c.ground.enabled, c.ground.caption) == (
            False,
            "Ignored while Attraction is 0",
        )

    def test_an_invalid_attraction_leaves_it_enabled(self) -> None:
        c = opened().controller

        c.edit(PanelFields(attraction="x"))

        assert c.ground.enabled

    def test_cannot_be_typed_in_while_running(self) -> None:
        p = opened()
        p.scripted.during = lambda _stage: p.controller.type_ground_nets("AGND")

        p.controller.run()

        assert p.controller.fields.ground_nets == ""


class TestValidation:
    def test_an_invalid_field_disables_run_and_names_itself_in_its_tip(
        self,
    ) -> None:
        c = opened().controller

        c.edit(PanelFields(repulsion="-1"))

        assert not c.run_enabled
        assert c.invalid == {Field.REPULSION: "Repulsion: must not be negative"}
        assert c.run_tip == "Repulsion: must not be negative"
        assert c.progress_line == Line("", "Ready — fix the marked field to run")

    def test_its_message_shows_once_the_field_is_left(self) -> None:
        c = opened().controller
        c.edit(PanelFields(repulsion="-1"))
        assert c.shown_invalid == {}

        c.leave(Field.REPULSION)

        assert c.shown_invalid == {Field.REPULSION: "Repulsion: must not be negative"}

    def test_trying_to_run_shows_every_message_and_runs_nothing(self) -> None:
        run = Scripted()
        c = opened(run=run).controller
        c.edit(PanelFields(time_limit="0", falloff_range="x"))

        assert not c.run()

        assert set(c.shown_invalid) == {Field.TIME_LIMIT, Field.RANGE}
        assert c.first_invalid is Field.TIME_LIMIT
        assert run.options == []
        assert c.state is State.IDLE

    def test_a_fixed_field_drops_its_message_until_left_again(self) -> None:
        c = opened().controller
        c.edit(PanelFields(repulsion="-1"))
        c.leave(Field.REPULSION)

        c.edit(PanelFields(repulsion="1"))
        assert c.run_enabled and c.shown_invalid == {}
        c.edit(PanelFields(repulsion="-"))

        assert c.shown_invalid == {}

    def test_counts_the_fields_behind_advanced_that_need_a_fix(self) -> None:
        c = opened().controller

        c.edit(PanelFields(time_limit="0", repulsion="-1", spacing="0.4 0.5"))

        assert c.advanced_fixes == 2

    @pytest.mark.parametrize(
        ("typed", "shown"), [("0.4 0.1", "0.4, 0.1"), ("0..5 x,", "0..5, x")]
    )
    def test_leaving_spacing_lists_it_with_commas(self, typed: str, shown: str) -> None:
        c = opened().controller
        c.edit(PanelFields(spacing=typed))

        c.leave(Field.SPACING)

        assert c.fields.spacing == shown

    def test_restore_defaults_resets_advanced_only(self) -> None:
        c = opened().controller
        c.type_ground_nets("AGND")
        c.edit(
            replace(
                c.fields,
                time_limit="5",
                repulsion="0.7",
                falloff_range="x",
                attraction="0",
                attraction_range="2",
                spacing="0.1",
                allow_via_in_pad=True,
            )
        )

        c.restore_defaults()

        assert c.fields == PanelFields(ground_nets="AGND", time_limit="5")
        assert c.run_enabled


class TestRun:
    def test_goes_through_its_stages_to_done(self) -> None:
        p = opened()

        p.controller.run()

        assert p.seen == [
            (State.RUNNING, "Reading the board"),
            (State.RUNNING, "Building obstacles"),
            (State.RUNNING, "Optimizing · setting up"),
            (State.RUNNING, "Optimizing · iteration 0"),
            (State.RUNNING, "Optimizing · iteration 1"),
            (State.RUNNING, "Optimizing · iteration 2"),
            (State.RUNNING, "Optimizing · iteration 3"),
            (State.RUNNING, "Simplifying the result"),
            (State.COMMITTING, "Committing to KiCad"),
            (State.DONE, "Finished · 38 iterations"),
        ]

    def test_names_a_selection_it_reads(self) -> None:
        board = ten_tracks_four_vias()
        board.selection = [board.tracks[0]]
        lines: list[Line] = []
        p = opened(board)
        p.scripted.stages = ()
        p.watchers.append(lambda: lines.append(p.controller.progress_line))

        p.controller.run()

        assert lines[0] == Line("Reading the selection")

    def test_runs_the_selection_else_the_board_with_the_fields(self) -> None:
        run = Scripted()
        c = opened(run=run).controller
        c.edit(EDITED)

        c.run()

        options = run.options[0]
        assert options.scope is ScopeOption.SELECTION_ELSE_BOARD
        assert options.settings.time_limit == 5.0
        assert options.params.repulsion == 0.7

    def test_says_it_is_setting_up_before_the_first_point(self) -> None:
        lines: list[Line] = []
        p = opened()
        p.scripted.progress = ()
        p.scripted.during = lambda _stage: lines.append(p.controller.progress_line)

        p.controller.run()

        assert lines[2] == Line("Optimizing", " · setting up")

    def test_graphs_the_length_change_from_the_start(self) -> None:
        c = opened().controller

        c.run()

        graph = c.graph
        assert graph.start == 10.0
        assert graph.points == ((0, 0.0), (1, -1.0), (2, -2.0), (3, -2.5))
        assert graph.levels == ()

    def test_marks_where_each_level_starts_and_names_its_spacing(self) -> None:
        lines: list[Line] = []
        p = opened()
        p.scripted.progress = LEVELLED
        p.scripted.during = lambda _stage: lines.append(p.controller.progress_line)

        p.controller.run()

        c = p.controller
        assert c.graph.levels == ((0, 1.6), (3, 0.4), (5, 0.1))
        assert lines[2] == Line("Optimizing", " · level 3 of 3, 0.1 mm · iteration 6")

    def test_names_the_level_as_it_runs(self) -> None:
        lines: list[Line] = []
        p = opened()
        p.scripted.progress = LEVELLED[:5]
        p.scripted.during = lambda _stage: lines.append(p.controller.progress_line)

        p.controller.run()

        assert lines[2] == Line("Optimizing", " · level 2 of 3, 0.4 mm · iteration 4")

    def test_the_figure_is_now_while_optimizing_then_before_simplifying(
        self,
    ) -> None:
        headers: list[tuple[str, str]] = []
        p = opened()

        def during(_stage: Stage) -> None:
            c = p.controller
            headers.append((c.graph_title, c.graph_figure))

        p.scripted.during = during
        p.controller.run()

        live = "Length change, mm, by iteration"
        later = "Length change before simplifying, mm"
        figure = f"{MINUS}2.50 mm ({MINUS}25.0 %)"
        assert headers == [
            (live, ""),
            (live, ""),
            (live, f"now {figure}"),
            (later, figure),
            (later, figure),
        ]
        c = p.controller
        assert (c.graph_title, c.graph_figure) == (later, figure)

    def test_a_run_of_vias_alone_has_no_length_change(self) -> None:
        p = opened()
        p.scripted.progress = (
            Progress(0.0, 0, 3.0, 0.0, 0, 1),
            Progress(0.5, 1, 2.0, 0.0, 0, 1),
        )

        p.controller.run()

        c = p.controller
        assert c.graph_figure == "vias only: no track length change"
        assert c.graph.points == ((0, 0.0), (1, 0.0))

    def test_describes_the_graph_in_words(self) -> None:
        p = opened()
        p.scripted.progress = LEVELLED

        p.controller.run()

        assert p.controller.graph_description == (
            f"Length change {MINUS}2.20 mm ({MINUS}22.0 %) after 6 iterations; "
            "levels 1.6 mm from iteration 0, 0.4 mm from 3, 0.1 mm from 5."
        )

    def test_says_what_stop_does_until_it_is_too_late(self) -> None:
        reports: list[Outcome | str] = []
        p = opened()
        p.scripted.during = lambda _stage: reports.append(p.controller.report)

        p.controller.run()

        assert reports == [STOP_HINT] * 3 + [TOO_LATE_HINT, COMMITTING_HINT]

    def test_stop_is_enabled_only_while_it_can_stop(self) -> None:
        buttons: list[tuple[bool, bool]] = []
        p = opened()

        def during(_stage: Stage) -> None:
            c = p.controller
            buttons.append((c.stop_enabled, c.run_enabled))

        p.scripted.during = during
        p.controller.run()

        assert buttons == [(True, False)] * 3 + [(False, False)] * 2
        c = p.controller
        assert (c.state, c.stop_enabled, c.run_enabled) == (State.DONE, False, True)

    def test_elapsed_counts_from_run_and_stops_at_the_end(self) -> None:
        times: list[str] = []
        clock = Manual()
        p = panel(clock=clock)
        p.controller.tick()

        def during(stage: Stage) -> None:
            if stage is Stage.READING:
                clock.now += 1.5
                times.append(p.controller.elapsed)
            if stage is Stage.SIMPLIFYING:
                clock.now += 70.0

        p.scripted.during = during
        clock.now = 10.0
        p.controller.run()
        clock.now += 5.0

        assert times == ["1.5 s"]
        assert p.controller.elapsed == "1:11"

    def test_fields_are_read_only_while_running(self) -> None:
        editable: list[bool] = []
        p = opened()

        def during(_stage: Stage) -> None:
            editable.append(p.controller.editable)
            p.controller.edit(EDITED)
            p.controller.restore_defaults()

        p.scripted.during = during
        p.controller.run()

        assert editable == [False] * len(STAGES)
        assert p.controller.fields == PanelFields()
        assert p.controller.editable

    def test_the_scope_line_holds_what_the_run_reads(self) -> None:
        board = ten_tracks_four_vias()
        lines: list[Line] = []
        p = panel(board, clock=Ticking(step=2.0))
        p.controller.tick()

        def during(_stage: Stage) -> None:
            board.selection = [board.vias[0]]
            p.controller.tick()
            lines.append(p.controller.scope_line)

        p.scripted.during = during
        p.controller.run()

        assert set(lines) == {WHOLE_BOARD}

    def test_reports_what_it_committed(self) -> None:
        clock = Manual()
        p = panel(clock=clock)
        p.controller.tick()
        p.scripted.during = lambda _stage: setattr(clock, "now", clock.now + 2.0)

        p.controller.run()

        o = outcome(p.controller)
        assert (o.icon, o.title, o.rest) == (
            Icon.SUCCESS,
            "Settled",
            " — committed as one undo step in KiCad",
        )
        assert o.rows == (
            ("Length", f"{MINUS}2.50 mm"),
            ("Changed", "1 chain, 0 vias"),
            ("Optimized", "38 iterations in 12.4 s (10.0 s in all)"),
            ("Rejected", "1 edge move"),
            ("Scope", "Selection of 12 tracks, 3 vias"),
        )

    def test_reports_a_whole_board_run_and_no_rejected_moves(self) -> None:
        whole = Scope(ScopeKind.BOARD, 1567, 210)
        moved = replace(report((), (via(moved=True),), scope=whole), iterations=1)
        c = opened(run=Scripted(outcome=moved)).controller

        c.run()

        assert rows(c)["Scope"] == "Whole board of 1567 tracks, 210 vias"
        assert rows(c)["Changed"] == "0 chains, 1 via"
        assert rows(c)["Rejected"] == "2 edge moves"
        assert rows(c)["Optimized"].startswith("1 iteration in ")

    @pytest.mark.parametrize(
        ("termination", "icon", "title", "rest"),
        [
            (
                Termination.STOPPED,
                Icon.INFO,
                "Stopped",
                " — best result so far committed as one undo step",
            ),
            (
                Termination.TIME_LIMIT,
                Icon.INFO,
                "Stopped at the time limit",
                " — best result committed as one undo step",
            ),
            (
                Termination.ITERATION_CAP,
                Icon.INFO,
                "Stopped at the iteration cap",
                " — best result committed as one undo step",
            ),
            (
                Termination.GAVE_UP,
                Icon.WARNING,
                "Stopped on abnormal bursts",
                " — best result committed; it may fall short of the smoothest",
            ),
        ],
    )
    def test_the_headline_says_how_it_ended(
        self, termination: Termination, icon: Icon, title: str, rest: str
    ) -> None:
        ended = replace(CHANGED, termination=termination)
        c = opened(run=Scripted(outcome=ended)).controller

        c.run()

        o = outcome(c)
        assert (o.icon, o.title, o.rest) == (icon, title, rest)

    def test_notes_a_stop_before_the_final_spacing(self) -> None:
        early = replace(STOPPED, before_final_spacing=True)
        c = opened(run=Scripted(outcome=early)).controller

        c.run()

        keys = [key for key, _ in outcome(c).rows]
        assert keys == ["Length", "Changed", "Optimized", "Note", "Rejected", "Scope"]
        assert rows(c)["Note"].startswith("Stopped before reaching the final spacing")

    def test_nothing_to_do(self) -> None:
        idle = replace(report(), termination=Termination.NOTHING_TO_DO, iterations=0)
        p = opened(run=Scripted(outcome=idle, stages=STAGES[:-1], progress=()))

        p.controller.run()

        c = p.controller
        o = outcome(c)
        assert (o.icon, o.title, o.rest) == (
            Icon.INFO,
            "Nothing to do",
            " — nothing committed",
        )
        assert o.body == (
            "The tracks are already as short and smooth as these settings allow."
        )
        assert o.rows == (("Scope", "Selection of 12 tracks, 3 vias"),)
        assert c.graph.placeholder == NOTHING_TO_OPTIMIZE
        assert c.progress_line == Line("Finished")
        assert State.COMMITTING not in [s for s, _ in p.seen]

    @pytest.mark.parametrize(
        ("termination", "title"),
        [
            (Termination.CONVERGED, "Nothing to do"),
            (Termination.STOPPED, "Stopped; nothing changed"),
        ],
    )
    def test_nothing_changed_after_iterations_says_how_long_it_optimized(
        self, termination: Termination, title: str
    ) -> None:
        unchanged = report(termination=termination)
        c = opened(run=Scripted(outcome=unchanged)).controller

        c.run()

        o = outcome(c)
        assert o.title == title
        assert [key for key, _ in o.rows] == ["Optimized", "Scope"]
        assert rows(c)["Optimized"].startswith("38 iterations in 12.4 s (")

    def test_collects_the_warnings_the_run_logs_on_its_thread(self) -> None:
        p = opened()

        def during(stage: Stage) -> None:
            if stage is Stage.READING:
                logger.warning("No copper on net GND3")
                elsewhere = threading.Thread(target=lambda: logger.warning("other"))
                elsewhere.start()
                elsewhere.join()
                logger.info("not a warning")

        p.scripted.during = during
        p.controller.run()

        assert outcome(p.controller).warnings == ("No copper on net GND3",)

    def test_a_new_run_starts_with_an_empty_graph_and_report(self) -> None:
        seen: list[tuple[int, Outcome | str]] = []
        p = opened()
        p.scripted.progress = LEVELLED
        p.controller.run()
        p.scripted.during = lambda _stage: seen.append(
            (len(p.controller.graph.levels), p.controller.report)
        )
        p.scripted.progress = PROGRESS

        p.controller.run()

        assert seen[0] == (0, STOP_HINT)
        assert p.controller.graph.levels == ()

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
        assert [s for s, _ in p.seen][-4:] == [
            State.STOPPING,
            State.RUNNING,
            State.COMMITTING,
            State.DONE,
        ]
        assert outcome(p.controller).title == "Stopped"

    def test_stopping_says_so_and_disables_stop(self) -> None:
        seen: list[tuple[str, bool, Line, Outcome | str]] = []
        p = opened()

        def during(stage: Stage) -> None:
            c = p.controller
            if stage is Stage.OPTIMIZING:
                c.stop()
            seen.append((c.stop_label, c.stop_enabled, c.progress_line, c.report))

        p.scripted.outcome = STOPPED
        p.scripted.during = during
        p.controller.run()

        assert seen[2:] == [
            (
                "Stopping…",
                False,
                Line("Stopping", " — ending the current step"),
                STOPPING_HINT,
            ),
            ("Stop", False, Line("Simplifying", " the result"), STOPPING_HINT),
            ("Stop", False, Line("Committing to KiCad"), COMMITTING_HINT),
        ]

    def test_the_graph_takes_the_points_until_the_last(self) -> None:
        p = opened()

        def during(stage: Stage) -> None:
            if stage is Stage.OBSTACLES:
                p.controller.stop()

        p.scripted.outcome = STOPPED
        p.scripted.during = during
        p.controller.run()

        assert len(p.controller.graph.points) == len(PROGRESS)

    def test_before_optimizing_nothing_changes(self) -> None:
        p = opened()

        def during(stage: Stage) -> None:
            if stage is Stage.OBSTACLES:
                p.controller.stop()

        p.scripted.outcome = UNCHANGED_STOP
        p.scripted.stages = STAGES[:2]
        p.scripted.during = during
        p.controller.run()

        c = p.controller
        o = outcome(c)
        assert (o.title, o.rest) == ("Stopped; nothing changed", " — nothing committed")
        assert o.rows == (("Scope", "Selection of 12 tracks, 3 vias"),)
        assert c.graph.placeholder == NO_OPTIMIZATION
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
        assert outcome(p.controller).title == "Settled"

    def test_stop_outside_a_run_does_nothing(self) -> None:
        c = opened().controller

        c.stop()

        assert c.state is State.IDLE


VIOLATIONS = (
    Violation("Chain /USB_D+ on F.Cu", "pad J3.4", (12.3, 40.115), 0.008, 0.15),
    Violation("Via /SDA", "track /SCL", (30.02, 18.5), 0.21, 0.2),
)


class TestErrors:
    @pytest.mark.parametrize(
        ("error", "icon", "title", "body", "detail"),
        [
            (
                ApiError("busy", code=ApiStatusCode.AS_BUSY),
                Icon.WARNING,
                "KiCad is busy",
                "Close any open tool or dialog in KiCad's PCB editor, then Run again.",
                "",
            ),
            (
                KiCadConnectionError("no socket"),
                Icon.CRITICAL,
                "Cannot reach KiCad",
                "Start KiCad 10 or later with the API server on (Preferences > "
                "Plugins), or retry: a request timed out after 30 s.",
                "no socket",
            ),
            (
                NoBoardError("No board is open in KiCad's PCB editor (x)"),
                Icon.CRITICAL,
                "No board open",
                "Open a board in KiCad's PCB editor, then Run again.",
                "",
            ),
            (
                SelectionError("selection has no tracks or vias"),
                Icon.INFO,
                "Nothing to smooth",
                "Selection has no tracks or vias. Select unlocked tracks or vias, "
                "or nothing for the whole board.",
                "",
            ),
            (
                ChainError("selection has no tracks that can be modified"),
                Icon.INFO,
                "Nothing to smooth",
                "Selection has no tracks that can be modified. Select unlocked "
                "tracks or vias, or nothing for the whole board.",
                "",
            ),
            (
                InsideClearanceError(VIOLATIONS, UserFrame()),
                Icon.CRITICAL,
                "Starts inside a clearance",
                "Move the copper apart or relax the rules, then Run again. 2 places:",
                "\n".join(v.describe(UserFrame()) for v in VIOLATIONS),
            ),
            (
                BoardChangedError(("t1", "v2")),
                Icon.WARNING,
                "The board changed during the run",
                "Tracks or vias the run read were edited in KiCad meanwhile. Select "
                "them again and Run.",
                "t1, v2",
            ),
            (
                CreateItemsError("KiCad created 0 of 2 tracks"),
                Icon.CRITICAL,
                "KiCad refused the change",
                "KiCad rejected the request. Try again; if it persists, the detail "
                "below says why.",
                "KiCad created 0 of 2 tracks",
            ),
            (
                ApiError("nope"),
                Icon.CRITICAL,
                "KiCad refused the change",
                "KiCad rejected the request. Try again; if it persists, the detail "
                "below says why.",
                "nope",
            ),
        ],
        ids=lambda v: type(v).__name__ if isinstance(v, BaseException) else "",
    )
    def test_a_failure_says_what_failed_and_what_to_do(
        self, error: Exception, icon: Icon, title: str, body: str, detail: str
    ) -> None:
        c = opened(run=Scripted(outcome=error)).controller
        c.edit(EDITED)

        c.run()

        assert c.state is State.FAILED
        assert outcome(c) == Outcome(
            icon, title, " — nothing committed", body, (), detail
        )
        assert c.fields == EDITED
        assert c.run_enabled and not c.stop_enabled
        assert c.editable

    def test_a_commit_that_may_have_landed_does_not_say_nothing_committed(
        self,
    ) -> None:
        lost = CommitUncertainError("Lost the connection; it may have landed")
        c = opened(run=Scripted(outcome=lost)).controller

        c.run()

        o = outcome(c)
        assert (o.icon, o.title, o.rest, o.detail) == (
            Icon.WARNING,
            "The commit may have landed",
            "",
            str(lost),
        )
        assert c.progress_line == Line("Failed", " while committing")

    @pytest.mark.parametrize(
        ("stages", "doing"),
        [
            ((), "connecting to KiCad"),
            (STAGES[:1], "reading the board"),
            (STAGES[:2], "building obstacles"),
            (STAGES[:3], "optimizing"),
        ],
    )
    def test_the_progress_line_names_the_stage_it_failed_in(
        self, stages: tuple[Stage, ...], doing: str
    ) -> None:
        p = opened(run=Scripted(outcome=SelectionError("bad"), stages=stages))

        p.controller.run()

        assert p.controller.progress_line == Line("Failed", f" while {doing}")

    def test_a_failure_before_optimizing_leaves_the_graph_saying_so(self) -> None:
        p = opened(run=Scripted(outcome=SelectionError("bad"), stages=STAGES[:2]))

        p.controller.run()

        assert p.controller.graph.placeholder == NO_OPTIMIZATION

    def test_a_failure_while_optimizing_keeps_the_curve_and_its_last_figure(
        self,
    ) -> None:
        p = opened(run=Scripted(outcome=SelectionError("bad"), stages=STAGES[:3]))

        p.controller.run()

        c = p.controller
        assert len(c.graph.points) == len(PROGRESS)
        assert c.graph_figure.startswith("last ")

    def test_an_unexpected_error_is_named_logged_and_detailed(
        self, log: list[str]
    ) -> None:
        c = opened(run=Scripted(outcome=RuntimeError("boom"))).controller

        c.run()

        o = outcome(c)
        assert (o.icon, o.title) == (Icon.CRITICAL, "Unexpected error")
        assert o.detail.startswith("RuntimeError: boom\n\nTraceback")
        assert any("Traceback" in line for line in log)
        assert o.warnings == ()

    def test_a_panic_fails_the_run_and_is_logged(self, log: list[str]) -> None:
        c = opened(run=Scripted(outcome=Panic("boom"))).controller

        c.run()

        assert c.state is State.FAILED
        assert outcome(c).detail.startswith("Panic: boom")
        assert any("Traceback" in line for line in log)
        assert c.close()

    @pytest.mark.parametrize("error", [RuntimeError("boom"), Panic("boom")])
    def test_an_unexpected_probe_failure_is_named_and_the_next_probe_runs(
        self, error: BaseException, log: list[str]
    ) -> None:
        probes = Answers([error, ten_tracks_four_vias()])
        p = panel(probe=probes, clock=Ticking(step=2.0))

        p.controller.tick()
        assert p.controller.scope_line == Line(
            "", f"Unexpected error: {type(error).__name__}: boom"
        )
        assert any("Traceback" in line for line in log)
        p.controller.tick()

        assert probes.calls == 2
        assert p.controller.scope_line == WHOLE_BOARD

    def test_an_abnormal_end_says_the_optimizer_gave_up(self) -> None:
        refused = replace(CHANGED, committed=False, refusal=ABNORMAL_MESSAGE)
        c = opened(run=Scripted(outcome=refused, stages=STAGES[:4])).controller

        c.run()

        o = outcome(c)
        assert c.state is State.FAILED
        assert (o.icon, o.title, o.rest) == (
            Icon.WARNING,
            "The optimizer gave up",
            " — nothing committed",
        )
        assert "Repulsion range values under Advanced" in o.body
        assert c.progress_line == Line("Failed", " while optimizing")

    @pytest.mark.parametrize(
        "ending",
        [
            InsideClearanceError(VIOLATIONS, UserFrame()),
            replace(CHANGED, committed=False, refusal=ABNORMAL_MESSAGE),
        ],
        ids=["failure", "refusal"],
    )
    def test_keeps_the_warnings_the_run_logged(
        self, ending: RunReport | BaseException
    ) -> None:
        def during(stage: Stage) -> None:
            if stage is Stage.READING:
                logger.warning("Cannot read the project file")

        run = Scripted(outcome=ending, stages=STAGES[:3], during=during)
        c = opened(run=run).controller

        c.run()

        assert c.state is State.FAILED
        assert outcome(c).warnings == ("Cannot read the project file",)


class TestClose:
    def test_closes_at_once_when_idle(self) -> None:
        p = opened()

        assert not p.controller.close_asks
        assert p.controller.close()

    def test_asks_first_while_the_run_can_stop(self) -> None:
        asks: list[bool] = []
        p = opened()
        p.scripted.during = lambda _stage: asks.append(p.controller.close_asks)

        p.controller.run()

        assert asks == [True] * 3 + [False] * 2

    def test_while_running_stops_and_closes_after_the_commit(self) -> None:
        p = opened()
        answers: list[bool] = []
        lines: list[Line] = []

        def during(stage: Stage) -> None:
            if stage is Stage.OPTIMIZING:
                answers.append(p.controller.close())
                lines.append(p.controller.progress_line)

        run = p.scripted
        run.outcome = STOPPED
        run.during = during
        p.controller.run()

        assert answers == [False]
        assert lines == [Line("Closing", " after the commit…")]
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
                answers.append(p.controller.close())
                assert run.stop is not None
                stops.append(run.stop())

        run = p.scripted
        run.during = during
        p.controller.run()

        assert answers == [False, False]
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


class TestFormats:
    @pytest.mark.parametrize(
        ("value", "decimals", "shown"),
        [
            (-2.306, 2, f"{MINUS}2.31"),
            (0.5, 2, "+0.50"),
            (-0.004, 2, "0.00"),
            (0.0, 1, "0.0"),
        ],
    )
    def test_signed(self, value: float, decimals: int, shown: str) -> None:
        assert signed(value, decimals) == shown

    @pytest.mark.parametrize(
        ("seconds", "shown"), [(12.44, "12.4 s"), (59.94, "59.9 s"), (65.2, "1:05")]
    )
    def test_duration(self, seconds: float, shown: str) -> None:
        assert duration(seconds) == shown

    def test_the_module_names_the_fields_behind_advanced(self) -> None:
        assert Field.TIME_LIMIT not in panel_controller.ADVANCED
        assert Field.SPACING in panel_controller.ADVANCED


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

        c = p.controller
        assert board.calls == COMMIT_CALLS
        assert [s for s, _ in p.seen][-2:] == [State.COMMITTING, State.DONE]
        assert outcome(c).title == "Settled"
        assert rows(c)["Scope"] == "Selection of 2 tracks, 0 vias"
        graph = c.graph
        assert graph.points[0] == (0, 0.0)
        assert graph.points[-1][1] < 0

    def test_a_stop_while_optimizing_commits_the_best_so_far(
        self, connected: Callable[[CommitBoard], Panel]
    ) -> None:
        board = l_board()
        p = connected(board)
        stopped: list[bool] = []

        def notify() -> None:
            c = p.controller
            # After the start and the first iteration.
            if c.state is State.RUNNING and len(c.graph.points) > 1 and not stopped:
                stopped.append(True)
                c.stop()

        p.watchers.append(notify)
        p.controller.run()

        assert board.calls == COMMIT_CALLS
        assert outcome(p.controller).title == "Stopped"

    def test_a_stop_before_optimizing_commits_nothing(
        self, connected: Callable[[CommitBoard], Panel]
    ) -> None:
        board = l_board()
        p = connected(board)

        def notify() -> None:
            c = p.controller
            if c.progress_line == Line("Building obstacles"):
                c.stop()

        p.watchers.append(notify)
        p.controller.run()

        assert board.calls == []
        assert outcome(p.controller).title == "Stopped; nothing changed"

    def test_a_connection_lost_in_the_push_may_have_landed(
        self, connected: Callable[[CommitBoard], Panel]
    ) -> None:
        board = l_board(push_error=KiCadConnectionError("gone"))
        p = connected(board)

        p.controller.run()

        assert p.controller.state is State.FAILED
        assert outcome(p.controller).title == "The commit may have landed"

    def test_an_empty_selection_runs_the_whole_board(
        self, connected: Callable[[CommitBoard], Panel]
    ) -> None:
        board = l_board()
        board.selection = []
        p = connected(board)

        p.controller.run()

        assert rows(p.controller)["Scope"] == "Whole board of 2 tracks, 0 vias"
