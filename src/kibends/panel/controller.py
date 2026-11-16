"""The panel's state and everything its window shows, free of Qt.

The states (`State`) run Reading board → Idle → Running → (Stopping) →
Committing → Done or Failed; Run starts Running again from Idle, Done or
Failed, which behave alike for editing and running. An invalid field is no
state of its own: it disables Run in those three, and its message shows
once the user has left the field or tried to run (`leave`, `run`). Stop
works only while the run can still stop (`run.STOPPABLE_STAGES`); from
simplifying on it is disabled.

All methods run on one thread, the controller's. Workers report back
through the injected `post`, which runs a callable on that thread; the
work itself goes to `spawn`, which runs it on a new worker thread, a
daemon one for a probe. A run worker calls `run`, `run.run_smooth` in the
app, with the run's stop flag; it reports stages, progress and the
outcome, and the warnings the run logged on its thread. Between runs,
`tick`, called several times a second, sends `probe` off about once a
second, never two at once: a probe still out when the next is due skips
it. The probe reads the board it does not know, on opening, on a board
switch and after every run, whose commit changed its totals. While a run
is in progress no probe starts, and one sent before the run started is
ignored whenever it answers, so the scope line keeps what the run read
and the board's totals are read again after the commit.

Ground nets follow detection until the user types in the field; a typed
list holds until the user empties the field, asks for detection again
(`use_detected`) or KiCad shows another board, which also clears the last
run's graph and report.

The graph plots the summed track length's change from the run's start,
by iteration, with each spacing level's start marked (`Graph`). The
report area holds a hint while there is no outcome, then the run's
outcome (`Outcome`): its ending and figures (`present.figures`), or what
failed and what to do.
"""

import re
import threading
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
from time import monotonic
from typing import Protocol, assert_never

from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from loguru import logger

from kibends.chain import ChainError
from kibends.commit import (
    BoardChangedError,
    CommitUncertainError,
    CreateItemsError,
    DeleteItemsError,
)
from kibends.inside import InsideClearanceError
from kibends.optimize import Progress
from kibends.panel.probe import BoardId, Counts, Probe, Totals
from kibends.present import (
    MAPPED_ERRORS,
    Field,
    Invalid,
    PanelFields,
    SmoothValues,
    counted,
    error_message,
    figures,
    normalised,
    read_panel,
    smooth_options,
)
from kibends.results import Termination
from kibends.run import (
    KICAD_TIMEOUT_MS,
    STOPPABLE_STAGES,
    NoBoardError,
    OptimizerReport,
    RunReport,
    ScopeKind,
    SmoothOptions,
    Stage,
    run_smooth,
)
from kibends.snapshot import GroundSource, SelectionError

PROBE_INTERVAL = 1.0
"""Seconds between the starts of two probes."""
TITLE = "kibends"
MINUS = "\N{MINUS SIGN}"
NOTHING_COMMITTED = " — nothing committed"
ADVANCED = frozenset(
    {
        Field.REPULSION,
        Field.RANGE,
        Field.ATTRACTION,
        Field.ATTRACTION_RANGE,
        Field.SPACING,
    }
)
"""The fields behind the Advanced disclosure."""

IDLE_HINT = "Select tracks in KiCad, or nothing for the whole board, then Run."
STOP_HINT = "Stop (Esc) ends optimizing and commits the best result so far."
TOO_LATE_HINT = "Too late to stop: the result is being simplified and committed."
STOPPING_HINT = "Will commit the best result so far."
COMMITTING_HINT = "Committing as one undo step…"
RUN_TIP = "Run on the scope shown at the top (Return)"

NO_RUN_YET = "Track length change appears here during a run"
NO_OPTIMIZATION = "No optimization ran"
NOTHING_TO_OPTIMIZE = "Nothing to optimize"

UNREACHABLE = "Cannot reach KiCad — is it running with the API server on? " + (
    "Retrying every second"
)
NO_BOARD = "No board open in KiCad's PCB editor"
BUSY = " (KiCad is busy)"


class State(Enum):
    READING_BOARD = "reading board"
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    COMMITTING = "committing"
    DONE = "done"
    FAILED = "failed"


_SETTLED = frozenset({State.IDLE, State.DONE, State.FAILED})
"""The states that edit fields and start runs."""
_IN_RUN = frozenset({State.RUNNING, State.STOPPING, State.COMMITTING})


@dataclass(frozen=True)
class Line:
    """A line of text, its first part in bold."""

    bold: str
    rest: str = ""


class Icon(Enum):
    SUCCESS = "success"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Outcome:
    """How a run ended, for the report area: an icon and a headline, its
    `title` in bold, then a `body`, rows of figures, a `detail` too long
    or many-lined for the body, and the warnings the run logged."""

    icon: Icon
    title: str
    rest: str = ""
    body: str = ""
    rows: tuple[tuple[str, str], ...] = ()
    detail: str = ""
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundNets:
    """What the Ground nets field shows."""

    text: str
    placeholder: str
    caption: str
    enabled: bool
    use_detected: bool
    """Whether the field holds a typed list detection can replace."""


@dataclass(frozen=True)
class Graph:
    """The run's track length by iteration, for the graph."""

    start: float
    """The summed track length at iteration 0, mm."""
    points: tuple[tuple[int, float], ...]
    """Each iteration and the length's change from `start` there, mm."""
    levels: tuple[tuple[int, float], ...]
    """Where each spacing level starts, the first at the first point, and
    its spacing, mm; empty for a schedule of one level."""
    placeholder: str
    """What the graph says while it has no points."""


class RunSmooth(Protocol):
    def __call__(
        self,
        options: SmoothOptions,
        on_optimized: Callable[[OptimizerReport], None] = ...,
        *,
        stop: Callable[[], bool] | None = ...,
        on_stage: Callable[[Stage], None] = ...,
        on_progress: Callable[[Progress], None] | None = ...,
    ) -> RunReport: ...


type Post = Callable[[Callable[[], None]], None]
"""Runs a callable on the controller's thread."""


class Spawn(Protocol):
    """Runs a callable on a new worker thread, a daemon one if `daemon`."""

    def __call__(self, work: Callable[[], None], *, daemon: bool) -> None: ...


def signed(value: float, decimals: int) -> str:
    """`value` to `decimals`, with a plus sign or a minus sign (U+2212)
    unless it rounds to zero."""
    text = f"{abs(value):.{decimals}f}"
    if float(text) == 0:
        return text
    return ("+" if value > 0 else MINUS) + text


def duration(seconds: float) -> str:
    """`12.4 s` under a minute, `1:05` from then on."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    whole = int(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


class Controller:
    """The panel's state; `notify` hears each change, and `closable` hears
    when a close the window asked for mid-run can go ahead."""

    def __init__(
        self,
        *,
        post: Post,
        spawn: Spawn,
        probe: Callable[[BoardId | None], Probe],
        run: RunSmooth = run_smooth,
        clock: Callable[[], float] = monotonic,
        notify: Callable[[], None] = lambda: None,
        closable: Callable[[], None] = lambda: None,
    ) -> None:
        self._post = post
        self._spawn = spawn
        self._probe = probe
        self._run = run
        self._clock = clock
        self._notify = notify
        self._closable = closable

        self._state = State.READING_BOARD
        self._fields = PanelFields()
        self._read: SmoothValues | Invalid = read_panel(self._fields)
        self._revealed: set[Field] = set()
        """The invalid fields whose message shows."""
        self._typed = False
        """Whether Ground nets holds the user's list, not detection."""
        self._dropped = False
        """Whether a board switch dropped the typed list, until the next
        edit."""
        self._board: BoardId | None = None
        self._totals: Totals | None = None
        self._selected: Counts | None = None
        self._known = False
        """Whether `_totals` still hold for `_board`."""
        self._problem: str | None = None
        """Why the last probe failed, while no later one has answered."""
        self._blocked = False
        """Whether that failure leaves no board to run on."""
        self._busy = False
        """Whether the last probe found KiCad busy."""
        self._probing = False
        self._last_probe = -float("inf")
        self._runs = 0
        """How many runs have started; a probe sent before the latest
        answers with what that run may have changed, and is ignored."""
        self._stage: Stage | None = None
        self._stop = threading.Event()
        self._stop_asked = False
        self._run_scope = Line("")
        self._run_board = False
        """Whether the run reads the whole board."""
        self._schedule: tuple[float, ...] = ()
        self._started: float | None = None
        """When the last run started."""
        self._ended: float | None = None
        self._last: Progress | None = None
        """The last run's latest progress."""
        self._lengths: list[tuple[int, float]] = []
        self._levels: list[tuple[int, float]] = []
        self._graph_description = ""
        self._outcome: Outcome | None = None
        self._ending = Line("")
        """The progress line once the run has ended."""
        self._no_points = NO_RUN_YET
        self._failed_optimizing = False
        self._closing = False
        self._closed = False

    # What the window shows.

    @property
    def state(self) -> State:
        return self._state

    @property
    def title(self) -> str:
        if self._board is None or (self._blocked and self._state not in _IN_RUN):
            return TITLE
        return f"{TITLE} — {self._board.name}"

    @property
    def fields(self) -> PanelFields:
        return self._fields

    @property
    def editable(self) -> bool:
        return self._state not in _IN_RUN

    @property
    def invalid(self) -> Mapping[Field, str]:
        """Each invalid field's message."""
        read = self._read
        return read.messages if isinstance(read, Invalid) else {}

    @property
    def shown_invalid(self) -> Mapping[Field, str]:
        """The messages of the invalid fields the user has left or tried
        to run with."""
        return {f: m for f, m in self.invalid.items() if f in self._revealed}

    @property
    def first_invalid(self) -> Field | None:
        """The first invalid field in the window's order."""
        return next((f for f in Field if f in self.invalid), None)

    @property
    def advanced_fixes(self) -> int:
        """How many fields behind Advanced are invalid."""
        return len(ADVANCED & self.invalid.keys())

    @property
    def scope_line(self) -> Line:
        if self._state in _IN_RUN:
            return self._run_scope
        if self._problem is not None:
            return Line("", self._problem)
        totals = self._totals
        if totals is None:
            text = (
                "Connecting to KiCad…"
                if self._board is None
                else ("Reading the board…")
            )
            return Line("", text + (BUSY if self._busy else ""))
        busy = BUSY if self._busy else ""
        tracks = counted(totals.tracks, "track")
        vias = counted(totals.vias, "via")
        selected = self._selected
        if selected is None:
            return Line("Whole board", f": {tracks}, {vias} (nothing selected){busy}")
        empty = (
            " — select tracks or vias, or nothing for the whole board"
            if selected == Counts(0, 0)
            else ""
        )
        return Line(
            "Selection",
            f": {selected.tracks} of {tracks}, {selected.vias} of {vias}{empty}{busy}",
        )

    @property
    def ground(self) -> GroundNets:
        totals = self._totals
        if totals is None:
            return GroundNets("", "reading the board…", "", False, False)
        detected = totals.ground
        off = self._attraction_off
        if self._typed:
            caption = (
                "Ignored while Attraction is 0"
                if off
                else ("Your list, not detection.")
            )
            return GroundNets(self._fields.ground_nets, "", caption, not off, not off)
        text = ", ".join(sorted(detected.nets))
        placeholder = ""
        match detected.source:
            case GroundSource.INNER_ZONES | GroundSource.GIVEN:
                caption = "Detected: zones on two or more inner layers"
            case GroundSource.FALLBACK:
                caption = "Detected: GND (no net has zones on two inner layers)"
            case GroundSource.NONE:
                placeholder = "none found — type net names"
                caption = "No ground net found: vias are not drawn to return vias"
            case _:
                assert_never(detected.source)
        if self._dropped:
            caption = "Detected for this board; your list was for the previous board"
        if off:
            caption = "Ignored while Attraction is 0"
        return GroundNets(text, placeholder, caption, not off, False)

    @property
    def _attraction_off(self) -> bool:
        try:
            return float(self._fields.attraction) == 0.0
        except ValueError:
            return False

    @property
    def run_enabled(self) -> bool:
        return (
            self._state in _SETTLED
            and not self.invalid
            and not (self._blocked or self._totals is None)
        )

    @property
    def run_tip(self) -> str:
        return "\n".join(self.invalid.values()) or RUN_TIP

    @property
    def stop_enabled(self) -> bool:
        return self._state is State.RUNNING and self._stoppable

    @property
    def stop_label(self) -> str:
        return "Stopping…" if self._state is State.STOPPING else "Stop"

    @property
    def close_asks(self) -> bool:
        """Whether closing now should first ask to stop the run."""
        return self.stop_enabled and not self._closing

    @property
    def progress_line(self) -> Line:
        state = self._state
        if self._closing and state in _IN_RUN:
            return Line("Closing", " after the commit…")
        match state:
            case State.READING_BOARD:
                return Line("", "Waiting for the board")
            case State.RUNNING:
                return self._stage_line
            case State.STOPPING:
                return Line("Stopping", " — ending the current step")
            case State.COMMITTING:
                return Line("Committing to KiCad")
            case State.IDLE | State.DONE | State.FAILED:
                if self.invalid:
                    return Line("", "Ready — fix the marked field to run")
                return Line("", "Ready") if state is State.IDLE else self._ending

    @property
    def _stage_line(self) -> Line:
        match self._stage:
            case None | Stage.READING:
                return Line(
                    "Reading the board" if self._run_board else "Reading the selection"
                )
            case Stage.OBSTACLES:
                return Line("Building obstacles")
            case Stage.OPTIMIZING:
                last = self._last
                if last is None:
                    return Line("Optimizing", " · setting up")
                if last.levels == 1:
                    return Line("Optimizing", f" · iteration {last.iteration}")
                return Line(
                    "Optimizing",
                    f" · level {last.level + 1} of {last.levels}, "
                    f"{self._spacing(last.level):g} mm · iteration {last.iteration}",
                )
            case Stage.SIMPLIFYING:
                return Line("Simplifying", " the result")
            case Stage.COMMITTING:
                return Line("Committing to KiCad")

    @property
    def elapsed(self) -> str:
        """The time since Run, or the run's total once it has ended."""
        if self._started is None:
            return ""
        end = self._clock() if self._ended is None else self._ended
        return duration(end - self._started)

    @property
    def graph(self) -> Graph:
        start = self._lengths[0][1] if self._lengths else 0.0
        return Graph(
            start,
            tuple((i, length - start) for i, length in self._lengths),
            tuple(self._levels),
            self._no_points,
        )

    @property
    def _optimized(self) -> bool:
        """Whether the run has its optimizer's result, which simplification
        changes before the commit."""
        if self._state in _SETTLED:
            return bool(self._lengths) and not self._failed_optimizing
        return self._stage in (Stage.SIMPLIFYING, Stage.COMMITTING)

    @property
    def graph_title(self) -> str:
        """The graph's header: what it plots, and that its figure comes
        before simplification once the run has gone past optimizing, so
        that it is not read as the committed change."""
        if self._optimized:
            return "Length change before simplifying, mm"
        return "Length change, mm, by iteration"

    @property
    def graph_figure(self) -> str:
        """The graph's latest length change: `now` while optimizing, `last`
        once a run failed while optimizing."""
        if not self._lengths:
            return ""
        start = self._lengths[0][1]
        if start == 0:
            return "vias only: no track length change"
        change = self._lengths[-1][1] - start
        figure = f"{signed(change, 2)} mm ({signed(100 * change / start, 1)} %)"
        if self._optimized:
            return figure
        return f"{'last' if self._failed_optimizing else 'now'} {figure}"

    @property
    def graph_description(self) -> str:
        """The graph in words, as of the last level change or the run's
        end."""
        return self._graph_description

    @property
    def report(self) -> Outcome | str:
        """The run's outcome, or a hint in its place."""
        match self._state:
            case State.READING_BOARD | State.IDLE | State.DONE | State.FAILED:
                return self._outcome or IDLE_HINT
            case State.RUNNING:
                if self._stoppable:
                    return STOP_HINT
                return STOPPING_HINT if self._stop_asked else TOO_LATE_HINT
            case State.STOPPING:
                return STOPPING_HINT
            case State.COMMITTING:
                return COMMITTING_HINT

    # What the window asks.

    def tick(self) -> None:
        """Send a probe off when one is due."""
        now = self._clock()
        if (
            self._closed
            or self._probing
            or self._state in _IN_RUN
            or now - self._last_probe < PROBE_INTERVAL
        ):
            return
        self._probing = True
        self._last_probe = now
        known = self._board if self._known else None
        runs = self._runs

        def work() -> None:
            try:
                probe = self._probe(known)
            # A Rust panic is a BaseException; any failure must reach the
            # controller, or it waits for this worker forever.
            except BaseException as e:
                if not isinstance(e, MAPPED_ERRORS):
                    logger.opt(exception=e).error("The panel's probe failed")
                self._post(partial(self._probe_failed, runs, e))
                return
            self._post(partial(self._probed, runs, probe))

        # A probe out when the window closes must not keep the process
        # alive for the IPC timeout.
        self._spawn(work, daemon=True)

    def edit(self, fields: PanelFields) -> None:
        """Take the fields as typed; Ground nets only through
        `type_ground_nets`."""
        if not self.editable:
            return
        self._set(replace(fields, ground_nets=self._fields.ground_nets))

    def type_ground_nets(self, text: str) -> None:
        if not self.editable:
            return
        self._typed = True
        self._dropped = False
        self._set(replace(self._fields, ground_nets=text))

    def use_detected(self) -> None:
        if not self.editable:
            return
        self._typed = False
        self._dropped = False
        self._set(replace(self._fields, ground_nets=""))

    def leave(self, field: Field) -> None:
        """The user left `field`: show its message, if any, and list its
        items as the field shows them; an emptied Ground nets goes back to
        detection."""
        if not self.editable:
            return
        self._revealed.add(field)
        if field is Field.SPACING:
            self._set(replace(self._fields, spacing=normalised(self._fields.spacing)))
        elif field is Field.GROUND_NETS and self._typed:
            nets = normalised(self._fields.ground_nets)
            if nets:
                self._set(replace(self._fields, ground_nets=nets))
            else:
                self.use_detected()
        else:
            self._notify()

    def restore_defaults(self) -> None:
        """Every field behind Advanced back at its default."""
        if not self.editable:
            return
        f = self._fields
        self._set(
            replace(PanelFields(), ground_nets=f.ground_nets, time_limit=f.time_limit)
        )

    def _set(self, fields: PanelFields) -> None:
        self._fields = fields
        self._read = read_panel(fields)
        self._revealed &= self.invalid.keys()
        self._notify()

    def run(self) -> bool:
        """Start a run, if Run is enabled; with an invalid field, show
        every field's message instead. Whether it started."""
        values = self._read
        if isinstance(values, Invalid):
            if self._state in _SETTLED:
                self._revealed |= values.messages.keys()
                self._notify()
            return False
        if not self.run_enabled:
            return False
        options = smooth_options(values)
        self._run_scope = self.scope_line
        self._run_board = self._selected is None
        self._schedule = values.spacing
        self._runs += 1
        self._state = State.RUNNING
        self._stage = None
        self._stop = threading.Event()
        self._stop_asked = False
        self._started = self._clock()
        self._ended = None
        self._last = None
        self._lengths = []
        self._levels = []
        self._graph_description = ""
        self._outcome = None
        self._no_points = NO_RUN_YET
        self._failed_optimizing = False
        self._notify()
        stop = self._stop
        run = self._run
        post = self._post

        def work() -> None:
            thread = threading.get_ident()
            warnings: list[str] = []
            sink = logger.add(
                lambda m: warnings.append(m.record["message"]),
                level="WARNING",
                filter=lambda record: record["thread"].id == thread,
                format="{message}",
            )
            try:
                report = run(
                    options,
                    stop=stop.is_set,
                    on_stage=lambda stage: post(partial(self._staged, stage)),
                    on_progress=lambda p: post(partial(self._progressed, p)),
                )
            except BaseException as e:  # as the probe's
                if not isinstance(e, MAPPED_ERRORS):
                    logger.opt(exception=e).error("The panel's run failed")
                post(partial(self._failed, e))
                return
            finally:
                logger.remove(sink)
            post(partial(self._finished, report, tuple(warnings)))

        self._spawn(work, daemon=False)
        return True

    @property
    def _stoppable(self) -> bool:
        """Whether the run can still stop: before its first stage, or in
        one of `STOPPABLE_STAGES`."""
        return self._stage is None or self._stage in STOPPABLE_STAGES

    def stop(self) -> None:
        if not self.stop_enabled:
            return
        self._stop.set()
        self._stop_asked = True
        self._state = State.STOPPING
        self._notify()

    def close(self) -> bool:
        """Whether the window may close now. Mid-run it may not: a run
        that can still stop is stopped, and `closable` hears when the
        run has ended."""
        if self._state in _IN_RUN:
            if not self._closing:
                self._closing = True
                self.stop()
                self._notify()
            return False
        self._closed = True
        return True

    # What the workers report.

    def _probed(self, runs: int, probe: Probe) -> None:
        self._probing = False
        if runs != self._runs:
            return
        if probe.totals is not None:
            if self._board is not None and probe.board != self._board:
                self._switched()
            self._board = probe.board
            self._totals = probe.totals
            self._known = True
        self._selected = probe.selected
        self._problem = None
        self._blocked = False
        self._busy = False
        if self._state is State.READING_BOARD:
            self._state = State.IDLE
        self._notify()

    def _switched(self) -> None:
        """Forget what belonged to the board KiCad no longer shows: a typed
        ground net list, and the last run's graph and report."""
        if self._typed:
            self._typed = False
            self._dropped = True
            self._set(replace(self._fields, ground_nets=""))
        self._lengths = []
        self._levels = []
        self._graph_description = ""
        self._outcome = None
        self._no_points = NO_RUN_YET
        self._started = None
        if self._state in _SETTLED:
            self._state = State.IDLE

    def _probe_failed(self, runs: int, error: BaseException) -> None:
        self._probing = False
        if runs != self._runs:
            return
        if isinstance(error, ApiError) and error.code == ApiStatusCode.AS_BUSY:
            self._busy = True
            self._notify()
            return
        match error:
            case KiCadConnectionError():
                self._problem, self._blocked = UNREACHABLE, True
            case NoBoardError():
                self._problem, self._blocked = NO_BOARD, True
            case _:
                self._problem, self._blocked = error_message(error), False
        self._busy = False
        self._known = False
        self._notify()

    def _staged(self, stage: Stage) -> None:
        if stage is Stage.COMMITTING or self._state is State.COMMITTING:
            self._state = State.COMMITTING
        elif stage not in STOPPABLE_STAGES:
            self._state = State.RUNNING
        self._stage = stage
        self._notify()

    def _progressed(self, progress: Progress) -> None:
        if self._state not in (State.RUNNING, State.STOPPING):
            return
        last = self._last
        self._last = progress
        self._lengths.append((progress.iteration, progress.length))
        entered = last is None or progress.level > last.level
        if entered and progress.levels > 1:
            self._levels.append((progress.iteration, self._spacing(progress.level)))
        if entered:
            self._describe()
        self._notify()

    def _spacing(self, level: int) -> float:
        schedule = self._schedule
        return schedule[min(level, len(schedule) - 1)]

    def _describe(self) -> None:
        if not self._lengths:
            self._graph_description = ""
            return
        start = self._lengths[0][1]
        iteration, length = self._lengths[-1]
        change = length - start
        percent = f" ({signed(100 * change / start, 1)} %)" if start else ""
        text = (
            f"Length change {signed(change, 2)} mm{percent} after "
            f"{counted(iteration, 'iteration')}"
        )
        if self._levels:
            parts = [
                f"{spacing:g} mm from {'iteration ' if k == 0 else ''}{i}"
                for k, (i, spacing) in enumerate(self._levels)
            ]
            text += f"; levels {', '.join(parts)}"
        self._graph_description = text + "."

    def _finished(self, report: RunReport, warnings: tuple[str, ...]) -> None:
        if report.refusal is not None:
            self._failed_optimizing = False
            self._fail(GAVE_UP, Stage.OPTIMIZING)
            return
        if not self._lengths:
            self._no_points = (
                NOTHING_TO_OPTIMIZE
                if report.termination is Termination.NOTHING_TO_DO
                else NO_OPTIMIZATION
            )
        iterations = report.iterations
        self._ending = Line(
            "Finished", f" · {counted(iterations, 'iteration')}" if iterations else ""
        )
        self._end(State.DONE, self._outcome_of(report, warnings))

    def _outcome_of(self, report: RunReport, warnings: tuple[str, ...]) -> Outcome:
        f = figures(report)
        total = self._clock() - (self._started or 0.0)
        optimized = (
            "Optimized",
            f"{counted(f.iterations, 'iteration')} in {duration(f.elapsed)} "
            f"({duration(total)} in all)",
        )
        kind = "Selection" if f.scope.kind is ScopeKind.SELECTION else "Whole board"
        scope = (
            "Scope",
            f"{kind} of {counted(f.scope.tracks, 'track')}, "
            f"{counted(f.scope.vias, 'via')}",
        )
        ran = (optimized,) if f.iterations else ()
        if not report.changes:
            if report.termination is Termination.STOPPED:
                return Outcome(
                    Icon.INFO,
                    "Stopped; nothing changed",
                    NOTHING_COMMITTED,
                    rows=(*ran, scope),
                    warnings=warnings,
                )
            return Outcome(
                Icon.INFO,
                "Nothing to do",
                NOTHING_COMMITTED,
                "The tracks are already as short and smooth as these settings allow.",
                rows=(*ran, scope),
                warnings=warnings,
            )
        icon, rest = _ENDINGS[report.termination]
        note = (
            (
                (
                    "Note",
                    "Stopped before reaching the final spacing; the result is "
                    "laid out at it but less refined.",
                ),
            )
            if report.before_final_spacing
            else ()
        )
        return Outcome(
            icon,
            f.ending,
            rest,
            rows=(
                ("Length", f"{signed(f.length_change, 2)} mm"),
                (
                    "Changed",
                    f"{counted(f.chains, 'chain')}, {counted(f.vias, 'via')}",
                ),
                optimized,
                *note,
                (
                    "Rejected",
                    counted(f.rejected, "edge move") if f.rejected else "none",
                ),
                scope,
            ),
            warnings=warnings,
        )

    def _failed(self, error: BaseException) -> None:
        self._failed_optimizing = self._stage is Stage.OPTIMIZING
        self._fail(failure(error), self._stage)

    def _fail(self, outcome: Outcome, stage: Stage | None) -> None:
        if not self._lengths:
            self._no_points = NO_OPTIMIZATION
        self._ending = Line("Failed", f" while {self._doing(stage)}")
        self._end(State.FAILED, outcome)

    def _doing(self, stage: Stage | None) -> str:
        match stage:
            case None:
                return "connecting to KiCad"
            case Stage.READING:
                return (
                    "reading the board"
                    if self._run_board
                    else ("reading the selection")
                )
            case Stage.OBSTACLES:
                return "building obstacles"
            case Stage.OPTIMIZING:
                return "optimizing"
            case Stage.SIMPLIFYING:
                return "simplifying"
            case Stage.COMMITTING:
                return "committing"

    def _end(self, state: State, outcome: Outcome) -> None:
        self._state = state
        self._outcome = outcome
        self._ended = self._clock()
        self._describe()
        self._known = False
        self._last_probe = -float("inf")
        self._notify()
        if self._closing:
            self._closed = True
            self._closable()


_ENDINGS: Mapping[Termination, tuple[Icon, str]] = {
    Termination.CONVERGED: (Icon.SUCCESS, " — committed as one undo step in KiCad"),
    Termination.NOTHING_TO_DO: (
        Icon.SUCCESS,
        " — committed as one undo step in KiCad",
    ),
    Termination.STOPPED: (
        Icon.INFO,
        " — best result so far committed as one undo step",
    ),
    Termination.TIME_LIMIT: (Icon.INFO, " — best result committed as one undo step"),
    Termination.ITERATION_CAP: (
        Icon.INFO,
        " — best result committed as one undo step",
    ),
    Termination.GAVE_UP: (
        Icon.WARNING,
        " — best result committed; it may fall short of the smoothest",
    ),
    Termination.ABNORMAL: (Icon.WARNING, NOTHING_COMMITTED),
}
"""Each ending's icon and the headline after its name, for a run that
changed something."""

GAVE_UP = Outcome(
    Icon.WARNING,
    "The optimizer gave up",
    NOTHING_COMMITTED,
    "L-BFGS ended abnormally without improving the tracks. Try other "
    "Repulsion or Repulsion range values under Advanced.",
)

_CHANGED_ITEMS = re.compile(r": (.*); select them again")


def failure(error: BaseException) -> Outcome:
    """What the report area says of a run that failed with `error`."""
    match error:
        case ApiError() if error.code == ApiStatusCode.AS_BUSY:
            return Outcome(
                Icon.WARNING,
                "KiCad is busy",
                NOTHING_COMMITTED,
                "Close any open tool or dialog in KiCad's PCB editor, then Run again.",
            )
        case KiCadConnectionError():
            return Outcome(
                Icon.CRITICAL,
                "Cannot reach KiCad",
                NOTHING_COMMITTED,
                "Start KiCad 10 or later with the API server on (Preferences > "
                "Plugins), or retry: a request timed out after "
                f"{KICAD_TIMEOUT_MS / 1000:g} s.",
                detail=str(error),
            )
        case NoBoardError():
            return Outcome(
                Icon.CRITICAL,
                "No board open",
                NOTHING_COMMITTED,
                "Open a board in KiCad's PCB editor, then Run again.",
            )
        case SelectionError() | ChainError():
            message = str(error).rstrip(".")
            return Outcome(
                Icon.INFO,
                "Nothing to smooth",
                NOTHING_COMMITTED,
                f"{message[:1].upper()}{message[1:]}. Select unlocked tracks or "
                "vias, or nothing for the whole board.",
            )
        case InsideClearanceError():
            places = error.violations
            return Outcome(
                Icon.CRITICAL,
                "Starts inside a clearance",
                NOTHING_COMMITTED,
                "Move the copper apart or relax the rules, then Run again. "
                f"{counted(len(places), 'place')}:",
                detail="\n".join(v.describe(error.frame) for v in places),
            )
        case BoardChangedError():
            items = _CHANGED_ITEMS.search(str(error))
            return Outcome(
                Icon.WARNING,
                "The board changed during the run",
                NOTHING_COMMITTED,
                "Tracks or vias the run read were edited in KiCad meanwhile. "
                "Select them again and Run.",
                detail=str(error) if items is None else items[1],
            )
        case CommitUncertainError():
            return Outcome(
                Icon.WARNING,
                "The commit may have landed",
                "",
                "The connection to KiCad was lost while committing. Check KiCad's "
                "undo history (Edit > Undo) before running again.",
                detail=str(error),
            )
        case CreateItemsError() | DeleteItemsError() | ApiError():
            return Outcome(
                Icon.CRITICAL,
                "KiCad refused the change",
                NOTHING_COMMITTED,
                "KiCad rejected the request. Try again; if it persists, the detail "
                "below says why.",
                detail=str(error),
            )
        case _:
            return Outcome(
                Icon.CRITICAL,
                "Unexpected error",
                NOTHING_COMMITTED,
                "kibends hit an error it does not expect. Please report it with "
                "the details below.",
                detail=f"{type(error).__name__}: {error}\n\n"
                + "".join(traceback.format_exception(error)),
            )
