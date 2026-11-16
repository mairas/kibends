"""What the command line and the panel both present: the values a run takes
and the `SmoothOptions` they give (`SmoothValues`, `smooth_options`), the
checks on the fields the panel edits (`PanelFields`, `read_panel`) and on
the command line's numbers (`finite_problem` and its kin,
`schedule_problem`), a finished run's summary (`summary`) and the message
for a failure (`error_message`).

The failure messages follow kistitch's cli.py error mapping (MIT,
Copyright (c) 2025 Hat Labs Oy).
"""

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import pairwise
from pathlib import Path
from typing import assert_never

from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.proto.common.envelope_pb2 import ApiStatusCode

from kibends.bursts import (
    DEFAULT_CHORD_TOLERANCE,
    DEFAULT_SETTINGS,
    MAX_COARSE_LEVELS,
)
from kibends.chain import ChainError
from kibends.commit import (
    BoardChangedError,
    CommitUncertainError,
    CreateItemsError,
    DeleteItemsError,
)
from kibends.energy import DEFAULT_ATTRACTION_RANGE, EnergyParams
from kibends.inside import InsideClearanceError
from kibends.results import Termination
from kibends.run import (
    ABNORMAL_MESSAGE,
    KICAD_TIMEOUT_MS,
    NoBoardError,
    PlotUnavailableError,
    RunReport,
    ScopeKind,
    ScopeOption,
    SmoothOptions,
)
from kibends.snapshot import SelectionError

DEFAULT_REPULSION = 0.5
DEFAULT_RANGE = 1.0
# Chosen from runs on real boards. A via's tracks hold it well beyond its
# clearance: a signal via on a straight route with a return via 1.5 mm
# across it ends 1.37 mm from it, 0.57 mm beyond the clearance, the route
# 0.0033 mm longer than without attraction. With nothing else near, a pair
# of 0.4 mm or 0.6 mm vias settles about 0.02 mm beyond its clearance,
# since the pull beats the repulsion's push at every distance past it. Stitching
# grids that routes cross settled in 11 to 51 iterations, against 16 to 32
# at 0.25, as scripts/benchmark.py measured.
DEFAULT_ATTRACTION = 1.15
# Not user-facing: the clearance hinge only shapes the approach to the
# clearance boundary, which the burst's edge check keeps edges from newly
# crossing.
CLEARANCE_WEIGHT = 50.0
# Above the worst-case sag between quadrature points at the final spacing,
# at most MAX_SPACING; the coarse levels' longer edges sag more, and the final
# level runs at the final spacing.
STANDOFF = 0.02
POINTS_PER_EDGE = 3
MAX_SPACING = 0.1
"""The coarsest final spacing a schedule takes, mm: STANDOFF and the
obstacle fields' MAX_EDGE_LENGTH are tuned to it."""
DEFAULT_SCHEDULE = (*DEFAULT_SETTINGS.coarse_spacings, DEFAULT_SETTINGS.spacing)

BUSY_MESSAGE = (
    "KiCad is busy — close any open tool or dialog in the PCB editor and retry"
)
PANEL_ABNORMAL_MESSAGE = (
    "The optimizer terminated abnormally without improving the track; "
    "nothing committed. "
    "Try other Repulsion or Range values under Advanced"
)


class RunRefusedError(RuntimeError):
    """The run refused the result; nothing committed."""


# Failures that need no traceback; all but CommitUncertainError leave the
# board untouched.
EXPECTED_ERRORS = (
    SelectionError,
    ChainError,
    BoardChangedError,
    CommitUncertainError,
    CreateItemsError,
    DeleteItemsError,
    InsideClearanceError,
    NoBoardError,
    PlotUnavailableError,
    RunRefusedError,
)
MAPPED_ERRORS = (KeyboardInterrupt, *EXPECTED_ERRORS, ApiError, KiCadConnectionError)
"""The failures `error_message` words; any other is unexpected."""


@dataclass(frozen=True)
class SmoothValues:
    """The values a run takes, as the user gives them; the defaults are the
    command line's."""

    repulsion: float = DEFAULT_REPULSION
    falloff_range: float = DEFAULT_RANGE
    attraction: float = DEFAULT_ATTRACTION
    attraction_range: float = DEFAULT_ATTRACTION_RANGE
    edge_clearance: float | None = None
    hole_clearance: float | None = None
    hole_to_hole: float | None = None
    override_names: Mapping[str, str] = field(default_factory=dict[str, str])
    chord_tolerance: float = DEFAULT_CHORD_TOLERANCE
    spacing: tuple[float, ...] = DEFAULT_SCHEDULE
    """Coarsest first; the last is the final spacing."""
    settled_change: float | None = None
    """None: half `chord_tolerance`."""
    stall_tolerance: float = DEFAULT_SETTINGS.stall_tolerance
    stall_window: int = DEFAULT_SETTINGS.stall_window
    max_iterations: int = DEFAULT_SETTINGS.max_iterations
    burst_iterations: int = DEFAULT_SETTINGS.burst_maxiter
    lbfgs_ftol: float = DEFAULT_SETTINGS.lbfgs_ftol
    lbfgs_gtol: float = DEFAULT_SETTINGS.lbfgs_gtol
    time_limit: float | None = None
    scope: ScopeOption = ScopeOption.SELECTION
    dry_run: bool = False
    plot: Path | None = None
    allow_via_in_pad: bool = False
    ground_nets: tuple[str, ...] = ()


def smooth_options(values: SmoothValues) -> SmoothOptions:
    """The run's options for `values`. A problem settles at half the chord
    tolerance unless `values` gives its own settled change, so that
    simplification does not undo what the last bursts moved."""
    return SmoothOptions(
        params=EnergyParams(
            repulsion=values.repulsion,
            falloff_range=values.falloff_range,
            clearance_weight=CLEARANCE_WEIGHT,
            standoff=STANDOFF,
            points_per_edge=POINTS_PER_EDGE,
            attraction=values.attraction,
            attraction_range=values.attraction_range,
        ),
        edge_clearance=values.edge_clearance,
        hole_clearance=values.hole_clearance,
        hole_to_hole=values.hole_to_hole,
        override_names=values.override_names,
        chord_tolerance=values.chord_tolerance,
        dry_run=values.dry_run,
        plot=values.plot,
        allow_via_in_pad=values.allow_via_in_pad,
        ground_nets=values.ground_nets,
        scope=values.scope,
        settings=replace(
            DEFAULT_SETTINGS,
            spacing=values.spacing[-1],
            coarse_spacings=values.spacing[:-1],
            settled_change=(
                values.chord_tolerance / 2
                if values.settled_change is None
                else values.settled_change
            ),
            stall_tolerance=values.stall_tolerance,
            stall_window=values.stall_window,
            max_iterations=values.max_iterations,
            burst_maxiter=values.burst_iterations,
            lbfgs_ftol=values.lbfgs_ftol,
            lbfgs_gtol=values.lbfgs_gtol,
            time_limit=values.time_limit,
        ),
    )


def finite_problem(value: float) -> str | None:
    return None if math.isfinite(value) else "must be a finite number"


def non_negative_problem(value: float) -> str | None:
    return finite_problem(value) or (None if value >= 0 else "must not be negative")


def positive_problem(value: float) -> str | None:
    return finite_problem(value) or (None if value > 0 else "must be positive")


def schedule_problem(spacings: Sequence[float]) -> str | None:
    """What is wrong with a spacing schedule whose values are each positive
    and finite: they must fall strictly to a final spacing of at most
    MAX_SPACING and number at most one more than MAX_COARSE_LEVELS."""
    if any(finer >= coarser for coarser, finer in pairwise(spacings)):
        return f"must fall strictly, coarsest first, got {listed(spacings)}"
    if spacings[-1] > MAX_SPACING:
        return (
            f"the final spacing, the last value, must be at most {MAX_SPACING:g}, "
            f"got {spacings[-1]:g}"
        )
    if len(spacings) > MAX_COARSE_LEVELS + 1:
        return f"takes at most {MAX_COARSE_LEVELS + 1} values, got {len(spacings)}"
    return None


def listed(spacings: Sequence[float]) -> str:
    return " ".join(f"{s:g}" for s in spacings)


class Field(Enum):
    """A field the panel edits, by its label."""

    GROUND_NETS = "Ground nets"
    TIME_LIMIT = "Time limit"
    REPULSION = "Repulsion"
    RANGE = "Range"
    ATTRACTION = "Attraction"
    ATTRACTION_RANGE = "Attraction range"
    SPACING = "Spacing"


@dataclass(frozen=True)
class PanelFields:
    """The panel's fields as the user typed them, starting at the defaults.
    Empty ground nets means their detection, an empty time limit none; every
    other text field needs a value."""

    ground_nets: str = ""
    """Net names separated by commas or spaces."""
    time_limit: str = ""
    repulsion: str = f"{DEFAULT_REPULSION:g}"
    falloff_range: str = f"{DEFAULT_RANGE:g}"
    attraction: str = f"{DEFAULT_ATTRACTION:g}"
    attraction_range: str = f"{DEFAULT_ATTRACTION_RANGE:g}"
    spacing: str = listed(DEFAULT_SCHEDULE)
    """Values separated by commas or spaces, coarsest first."""
    allow_via_in_pad: bool = False


@dataclass(frozen=True)
class Invalid:
    """The fields that hold no valid value, each with its message, which
    names the field."""

    messages: Mapping[Field, str]


class _FieldError(ValueError):
    pass


_SEPARATORS = re.compile(r"[,\s]+")


def _words(text: str) -> list[str]:
    return [w for w in _SEPARATORS.split(text) if w]


def _number(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise _FieldError(f"must be a number, got {text}") from None


def _checked(text: str, problem: Callable[[float], str | None]) -> float:
    value = _number(text.strip())
    found = problem(value)
    if found is not None:
        raise _FieldError(found)
    return value


def _required(text: str, problem: Callable[[float], str | None]) -> float:
    if not text.strip():
        raise _FieldError("needs a value")
    return _checked(text, problem)


def _time_limit(text: str) -> float | None:
    return _checked(text, positive_problem) if text.strip() else None


def _schedule(text: str) -> tuple[float, ...]:
    words = _words(text)
    if not words:
        raise _FieldError("needs a value")
    spacings = tuple(_checked(w, positive_problem) for w in words)
    found = schedule_problem(spacings)
    if found is not None:
        raise _FieldError(found)
    return spacings


def read_panel(fields: PanelFields) -> SmoothValues | Invalid:
    """The values the panel's fields give, for a run that reads the
    selection, else the whole board; or, when some field holds no valid
    value, a message for each such field."""
    messages: dict[Field, str] = {}

    def read[T](which: Field, parse: Callable[[], T], fallback: T) -> T:
        try:
            return parse()
        except _FieldError as e:
            messages[which] = f"{which.value}: {e}"
            return fallback

    values = SmoothValues(
        ground_nets=tuple(_words(fields.ground_nets)),
        time_limit=read(Field.TIME_LIMIT, lambda: _time_limit(fields.time_limit), None),
        repulsion=read(
            Field.REPULSION,
            lambda: _required(fields.repulsion, non_negative_problem),
            DEFAULT_REPULSION,
        ),
        falloff_range=read(
            Field.RANGE,
            lambda: _required(fields.falloff_range, positive_problem),
            DEFAULT_RANGE,
        ),
        attraction=read(
            Field.ATTRACTION,
            lambda: _required(fields.attraction, non_negative_problem),
            DEFAULT_ATTRACTION,
        ),
        attraction_range=read(
            Field.ATTRACTION_RANGE,
            lambda: _required(fields.attraction_range, positive_problem),
            DEFAULT_ATTRACTION_RANGE,
        ),
        spacing=read(
            Field.SPACING, lambda: _schedule(fields.spacing), DEFAULT_SCHEDULE
        ),
        allow_via_in_pad=fields.allow_via_in_pad,
        scope=ScopeOption.SELECTION_ELSE_BOARD,
    )
    return Invalid(messages) if messages else values


def _ending(termination: Termination) -> str:
    match termination:
        case Termination.CONVERGED | Termination.NOTHING_TO_DO:
            return "Settled"
        case Termination.STOPPED:
            return "Stopped"
        case Termination.TIME_LIMIT:
            return "Stopped at the time limit"
        case Termination.ITERATION_CAP:
            return "Stopped at the iteration cap"
        case Termination.GAVE_UP:
            return "Stopped on abnormal bursts"
        case Termination.ABNORMAL:
            return "Terminated abnormally"
        case _:
            assert_never(termination)


def counted(n: int, noun: str) -> str:
    """`n` and `noun`, plural unless `n` is one."""
    return f"{n} {noun}{'' if n == 1 else 's'}"


def summary(report: RunReport) -> str:
    """How a run without a refusal ended, in one line: its ending,
    iterations and time, the chains and vias it changed, their summed
    length change, the edge moves it rejected and its scope; "Nothing to
    do" or, for a stop, "Stopped; nothing changed" when nothing changed."""
    if not report.changes:
        return (
            "Stopped; nothing changed"
            if report.termination is Termination.STOPPED
            else "Nothing to do"
        )
    changed = [c for c in report.chains if c.changed]
    # Adding 0.0 turns a -0.0 from rounding into 0.0.
    change = round(sum(c.after.length - c.before.length for c in changed), 2) + 0.0
    rejected = (
        sum(c.rejected_edges for c in report.chains)
        + sum(v.rejected_edges for v in report.vias)
        + sum(p.rejected_edges for p in report.pairs)
    )
    scope = report.scope
    kind = "selection" if scope.kind is ScopeKind.SELECTION else "whole board"
    return (
        f"{_ending(report.termination)} after "
        f"{counted(report.iterations, 'iteration')} in {report.elapsed:.1f} s; "
        f"{counted(len(changed), 'chain')}, "
        f"{counted(sum(v.moved for v in report.vias), 'via')} changed; "
        f"length {change:{'+' if change else ''}.2f} mm; "
        f"{counted(rejected, 'rejected edge move')}; "
        f"{kind} of {counted(scope.tracks, 'track')}, {counted(scope.vias, 'via')}"
    )


def error_message(error: BaseException, *, panel: bool = False) -> str:
    """The one line that tells the user of `error`: the command line's, or
    with `panel` the panel's, which names its own fields. An error outside
    `MAPPED_ERRORS` is unexpected, named by its type; the caller logs its
    traceback."""
    match error:
        case KeyboardInterrupt():
            # The push maps its own interrupt to CommitUncertainError.
            return "Interrupted: nothing committed"
        case RunRefusedError() if panel and str(error) == ABNORMAL_MESSAGE:
            return PANEL_ABNORMAL_MESSAGE
        case _ if isinstance(error, EXPECTED_ERRORS):
            return str(error)
        case ApiError() if error.code == ApiStatusCode.AS_BUSY:
            return BUSY_MESSAGE
        case ApiError():
            return f"KiCad refused the request: {error}"
        case KiCadConnectionError():
            return (
                f"Cannot reach KiCad ({error}): either it is not running — start "
                "KiCad 10 or later with the API server enabled (Preferences > "
                "Plugins) — or a request timed out after "
                f"{KICAD_TIMEOUT_MS / 1000:g} s"
            )
        case _:
            return f"Unexpected error: {type(error).__name__}: {error}"
