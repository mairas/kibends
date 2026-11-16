"""The panel's state and everything its window shows, free of Qt.

The states (`State`) run Reading board → Idle → Running → (Stopping) →
Committing → Done or Failed; Run starts Running again from Idle, Done or
Failed, which behave alike for editing and running. An invalid field is no
state of its own: it disables Run in those three.

All methods run on one thread, the controller's. Workers report back
through the injected `post`, which runs a callable on that thread; the
work itself goes to `spawn`, which runs it on a new worker thread. A run
worker calls `run`, `run.run_smooth` in the app, with the run's stop flag;
it reports stages, progress and the outcome. Between runs, `tick`, called
several times a second, sends `probe` off about once a second, never two
at once: a probe still out when the next is due skips it. The probe reads
the board it does not know, on opening, on a board switch and after every
run, whose commit changed its totals. While a run is in progress no probe
starts, and one that answers is ignored, so the scope line keeps what the
run reads.
"""

import threading
from collections.abc import Callable
from dataclasses import replace
from enum import Enum
from functools import partial
from time import monotonic
from typing import Protocol

from kipy.errors import ApiError
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from loguru import logger

from kibends.optimize import Progress
from kibends.panel.probe import BoardId, Counts, Probe, Totals
from kibends.present import (
    MAPPED_ERRORS,
    Field,
    Invalid,
    PanelFields,
    RunRefusedError,
    SmoothValues,
    error_message,
    read_panel,
    smooth_options,
    summary,
)
from kibends.run import OptimizerReport, RunReport, SmoothOptions, Stage, run_smooth
from kibends.snapshot import GroundSource

PROBE_INTERVAL = 1.0
"""Seconds between the starts of two probes."""
NO_SCOPE = "—"
TITLE = "kibends"


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
_STAGE_STATUS = {
    Stage.READING: "Reading selection…",
    Stage.OBSTACLES: "Building obstacles…",
    Stage.OPTIMIZING: "Optimizing",
    Stage.SIMPLIFYING: "Simplifying…",
}


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
type Spawn = Callable[[Callable[[], None]], None]
"""Runs a callable on a new worker thread."""


def _counted(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


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
        self._board: BoardId | None = None
        self._totals: Totals | None = None
        self._selected: Counts | None = None
        self._known = False
        """Whether `_totals` still hold for `_board`."""
        self._probe_problem: str | None = None
        """Why the last probe failed, while no later one has answered."""
        self._probing = False
        self._last_probe = -float("inf")
        self._message = ""
        """The status between runs: the last run's ending or failure, or a
        later probe's problem, whichever came last."""
        self._stage: Stage | None = None
        self._stop = threading.Event()
        self._run_scope = NO_SCOPE
        self._started: float | None = None
        """When the last run started."""
        self._ended: float | None = None
        self._last: Progress | None = None
        """The last run's latest progress."""
        self._energies: list[float] = []
        self._separators: list[int] = []
        self._closing = False
        self._closed = False

    # What the window shows.

    @property
    def state(self) -> State:
        return self._state

    @property
    def title(self) -> str:
        return TITLE if self._board is None else f"{TITLE} — {self._board.name}"

    @property
    def fields(self) -> PanelFields:
        return self._fields

    @property
    def editable(self) -> bool:
        return self._state not in _IN_RUN

    @property
    def invalid(self) -> frozenset[Field]:
        read = self._read
        return frozenset(read.messages) if isinstance(read, Invalid) else frozenset()

    @property
    def ground_placeholder(self) -> str:
        if self._totals is None:
            return ""
        ground = self._totals.ground
        if ground.source is GroundSource.NONE:
            return "none detected"
        return f"{', '.join(sorted(ground.nets))} (detected)"

    @property
    def scope_line(self) -> str:
        if self._state in _IN_RUN:
            return self._run_scope
        totals = self._totals
        if totals is None or self._probe_problem is not None:
            return NO_SCOPE
        selected = self._selected
        if selected is None:
            return (
                f"Whole board: {_counted(totals.tracks, 'track')}, "
                f"{_counted(totals.vias, 'via')}"
            )
        return (
            f"{selected.tracks} of {_counted(totals.tracks, 'track')}, "
            f"{selected.vias} of {_counted(totals.vias, 'via')}"
        )

    @property
    def button_label(self) -> str:
        match self._state:
            case State.RUNNING | State.COMMITTING:
                return "Stop"
            case State.STOPPING:
                return "Stopping…"
            case _:
                return "Run"

    @property
    def button_enabled(self) -> bool:
        match self._state:
            case State.RUNNING:
                return True
            case State.IDLE | State.DONE | State.FAILED:
                return not isinstance(self._read, Invalid)
            case _:
                return False

    @property
    def progress_line(self) -> str:
        if self._started is None:
            return ""
        end = self._clock() if self._ended is None else self._ended
        elapsed = f"{end - self._started:.1f} s"
        last = self._last
        if last is None:
            return elapsed
        if last.levels == 1:
            return f"{elapsed} · iteration {last.iteration}"
        return (
            f"{elapsed} · level {last.level + 1} of {last.levels} · "
            f"iteration {last.iteration}"
        )

    @property
    def energies(self) -> tuple[float, ...]:
        return tuple(self._energies)

    @property
    def separators(self) -> tuple[int, ...]:
        """Where in `energies` the run entered another level: each the
        index of the first value at it."""
        return tuple(self._separators)

    @property
    def status(self) -> str:
        match self._state:
            case State.READING_BOARD:
                return "Reading board…"
            case State.RUNNING:
                return _STAGE_STATUS[self._stage or Stage.READING]
            case State.STOPPING:
                return "Stopping…"
            case State.COMMITTING:
                return "Committing"
            case State.IDLE | State.DONE | State.FAILED:
                if isinstance(self._read, Invalid):
                    return "\n".join(self._read.messages.values())
                return self._message

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

        def work() -> None:
            try:
                probe = self._probe(known)
            except Exception as e:
                if not isinstance(e, MAPPED_ERRORS):
                    logger.opt(exception=e).error("The panel's probe failed")
                self._post(partial(self._probe_failed, e))
                return
            self._post(partial(self._probed, probe))

        self._spawn(work)

    def edit(self, fields: PanelFields) -> None:
        if not self.editable:
            return
        self._fields = fields
        self._read = read_panel(fields)
        self._notify()

    def run(self) -> None:
        values = self._read
        if self._state not in _SETTLED or isinstance(values, Invalid):
            return
        options = smooth_options(values)
        self._run_scope = self.scope_line
        self._state = State.RUNNING
        self._stage = None
        self._stop = threading.Event()
        self._started = self._clock()
        self._ended = None
        self._last = None
        self._energies = []
        self._separators = []
        self._message = ""
        self._notify()
        stop = self._stop
        run = self._run
        post = self._post

        def work() -> None:
            try:
                report = run(
                    options,
                    stop=stop.is_set,
                    on_stage=lambda stage: post(partial(self._staged, stage)),
                    on_progress=lambda p: post(partial(self._progressed, p)),
                )
            except Exception as e:
                if not isinstance(e, MAPPED_ERRORS):
                    logger.opt(exception=e).error("The panel's run failed")
                post(partial(self._failed, e))
                return
            post(partial(self._finished, report))

        self._spawn(work)

    def stop(self) -> None:
        if self._state is not State.RUNNING:
            return
        self._stop.set()
        self._state = State.STOPPING
        self._notify()

    def close(self) -> bool:
        """Whether the window may close now. Mid-run it may not: a run
        that can still stop is stopped, and `closable` hears when the
        run has ended."""
        if self._state in _IN_RUN:
            self._closing = True
            self.stop()
            return False
        self._closed = True
        return True

    # What the workers report.

    def _probed(self, probe: Probe) -> None:
        self._probing = False
        if self._state in _IN_RUN:
            return
        if probe.totals is not None:
            if self._board is not None and probe.board != self._board:
                self._fields = replace(self._fields, ground_nets="")
                self._read = read_panel(self._fields)
            self._board = probe.board
            self._totals = probe.totals
            self._known = True
        self._selected = probe.selected
        if self._probe_problem is not None and self._message == self._probe_problem:
            self._message = ""
        self._probe_problem = None
        if self._state is State.READING_BOARD:
            self._state = State.IDLE
        self._notify()

    def _probe_failed(self, error: Exception) -> None:
        self._probing = False
        if self._state in _IN_RUN:
            return
        if isinstance(error, ApiError) and error.code == ApiStatusCode.AS_BUSY:
            return
        self._probe_problem = error_message(error, panel=True)
        self._message = self._probe_problem
        self._known = False
        if self._state is State.READING_BOARD:
            self._state = State.IDLE
        self._notify()

    def _staged(self, stage: Stage) -> None:
        if stage is Stage.COMMITTING:
            self._state = State.COMMITTING
        self._stage = stage
        self._notify()

    def _progressed(self, progress: Progress) -> None:
        if self._state is not State.RUNNING:
            return
        if self._last is not None and progress.level != self._last.level:
            self._separators.append(len(self._energies))
        self._last = progress
        self._energies.append(progress.energy)
        self._notify()

    def _finished(self, report: RunReport) -> None:
        if report.refusal is not None:
            self._end(
                State.FAILED,
                error_message(RunRefusedError(report.refusal), panel=True),
            )
        else:
            self._end(State.DONE, summary(report))

    def _failed(self, error: Exception) -> None:
        self._end(State.FAILED, error_message(error, panel=True))

    def _end(self, state: State, message: str) -> None:
        self._state = state
        self._message = message
        self._ended = self._clock()
        self._known = False
        self._last_probe = -float("inf")
        self._notify()
        if self._closing:
            self._closed = True
            self._closable()
