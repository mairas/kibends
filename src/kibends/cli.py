"""The `kibends` command line.

`smooth` prints the optimizer's outcome line, logs its warnings (an
iteration cap or time limit) and, at debug, the abnormal bursts, all from
`run_smooth`'s `on_optimized` callback, and raises the returned report's
refusal. The per-chain, per-pair and per-via figures stay in the reports
and are not printed. The first Ctrl-C while reading, building the
obstacles or optimizing is the run's `stop`, so the run commits the best
result so far; a second, or one from simplifying on, raises
`KeyboardInterrupt`, which ends the run with nothing committed, or, during
the push, as one that may have landed. Every IPC failure ends the
run with one log line (`_exit_on_failure`), except a copper-text render
KiCad refuses for another reason, which only warns, and the selection
after the push (`commit._select`), logged at debug since the commit has
landed.

`kibends-panel` (`panel_main`) opens the panel (`kibends.panel`), imported
only then, since PySide6 comes with the `gui` extra; without it the command
says how to install it. `kibends` never imports it.

`_exit_on`-style error mapping and `configure_logging` follow kistitch's
cli.py (MIT, Copyright (c) 2025 Hat Labs Oy).
"""

import argparse
import os
import signal
import sys
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import NoReturn, assert_never, cast

from loguru import logger

from kibends import present
from kibends.bursts import DEFAULT_CHORD_TOLERANCE, DEFAULT_SETTINGS
from kibends.energy import DEFAULT_ATTRACTION_RANGE
from kibends.obstacles import EDGE_CLEARANCE, HOLE_CLEARANCE, HOLE_TO_HOLE
from kibends.present import (
    DEFAULT_ATTRACTION,
    DEFAULT_RANGE,
    DEFAULT_REPULSION,
    DEFAULT_SCHEDULE,
    MAPPED_ERRORS,
    MAX_SPACING,
    RunRefusedError,
    SmoothValues,
    error_message,
    listed,
    non_negative_problem,
    positive_problem,
    schedule_problem,
)
from kibends.results import Termination
from kibends.run import (
    STOPPABLE_STAGES,
    OptimizerReport,
    ScopeOption,
    SmoothOptions,
    Stage,
    run_smooth,
)

CLEARANCE_FLAGS = {
    "edge": "--edge-clearance",
    "hole": "--hole-clearance",
    "hole_to_hole": "--hole-to-hole",
}
"""The options that override the project's clearance rules, by
`rules.Clearances` field."""
GUI_EXTRA_MESSAGE = (
    "The panel needs PySide6, which comes with the `gui` extra: "
    "uv tool install 'kibends[gui] @ git+https://github.com/mairas/kibends'"
)
STOPPING_MESSAGE = "Stopping; press Ctrl-C again to abort"

DEFAULT_LOG_LEVEL = "INFO"
VERBOSE_LOG_LEVEL = "DEBUG"
LOG_LEVEL_VARIABLE = "LOGURU_LEVEL"


def _fail(message: str) -> NoReturn:
    logger.error(message)
    raise SystemExit(1)


@contextmanager
def _exit_on_failure() -> Generator[None]:
    """Report an expected failure as one log line, worded by
    `present.error_message`, and exit 1."""
    try:
        yield
    except MAPPED_ERRORS as e:
        _fail(error_message(e))


@contextmanager
def _first_interrupt_stops() -> Generator[
    tuple[Callable[[], bool], Callable[[Stage], None]]
]:
    """A run's `stop` and `on_stage`, with a SIGINT handler under which the
    first Ctrl-C makes `stop` answer true and a second raises
    `KeyboardInterrupt`. From the first stage outside `STOPPABLE_STAGES`
    on, and always on leaving, the previous handler is back.

    The handler sets a flag and writes its notice straight to stderr's file
    descriptor, so the notice shows at once, while `stop` may not be asked
    for seconds. It runs wherever the main thread is, the burst's signal
    check included, and logging from there could re-enter the logger.
    """
    previous = signal.getsignal(signal.SIGINT)
    interrupted = False

    def handle(_signum: int, _frame: FrameType | None) -> None:
        nonlocal interrupted
        if interrupted:
            raise KeyboardInterrupt
        interrupted = True
        os.write(sys.stderr.fileno(), f"{STOPPING_MESSAGE}\n".encode())

    def stop() -> bool:
        return interrupted

    def on_stage(stage: Stage) -> None:
        if stage not in STOPPABLE_STAGES:
            signal.signal(signal.SIGINT, previous)

    signal.signal(signal.SIGINT, handle)
    try:
        yield stop, on_stage
    finally:
        signal.signal(signal.SIGINT, previous)


def smooth(options: SmoothOptions) -> None:
    """Smooth the chains selected in KiCad, print the optimizer's outcome
    and, unless dry-running, commit them. The first Ctrl-C while reading,
    building the obstacles or optimizing stops the run, which commits the
    best result so far (`_first_interrupt_stops`)."""

    def show(optimized: OptimizerReport) -> None:
        print(format_outcome(optimized))
        for warning in report_warnings(optimized):
            logger.warning(warning)
        for description, n in optimized.abnormal:
            logger.debug(
                f"L-BFGS ended {n} burst{'' if n == 1 else 's'} of "
                f"{description} abnormally; the run judged each like any other"
            )

    with _first_interrupt_stops() as (stop, on_stage):
        report = run_smooth(options, show, stop=stop, on_stage=on_stage)
    if report.refusal is not None:
        raise RunRefusedError(report.refusal)
    if not report.changes:
        logger.info(
            "Stopped before anything changed: nothing committed"
            if report.termination is Termination.STOPPED
            else "Nothing to do: the optimizer leaves the selection as it is"
        )
    elif options.dry_run:
        logger.info("Dry run: nothing committed")


def format_outcome(report: OptimizerReport) -> str:
    """How the optimizer ended, after how many iterations and seconds, with
    the rejected edge moves it counted."""
    rejected = (
        sum(c.rejected_edges for c in report.chains)
        + sum(v.rejected_edges for v in report.vias)
        + sum(p.rejected_edges for p in report.pairs)
    )
    return (
        f"Optimizer: {report.termination.value} after {report.iterations} "
        f"iterations in {report.elapsed:.1f} s, {rejected} rejected edge moves"
    )


def report_warnings(report: OptimizerReport) -> list[str]:
    """An iteration cap, saying whether the run stopped before its final
    spacing, or a time limit, which runs out only at the final spacing, one
    line each; a stop, only when it came before the final spacing, since
    the user asked for it."""
    unfinished = (
        "before reaching the final spacing"
        if report.before_final_spacing
        else "without converging"
    )
    warnings: list[str] = []
    if report.termination is Termination.ITERATION_CAP:
        warnings.append(
            f"The optimizer stopped at its cap of {report.iterations} "
            f"outer iterations {unfinished}; using the best result found"
        )
    if report.termination is Termination.TIME_LIMIT:
        warnings.append(
            "The optimizer stopped at its time limit without converging; "
            "using the best result found"
        )
    if report.termination is Termination.STOPPED and report.before_final_spacing:
        warnings.append(
            "Stopped before reaching the final spacing; using the best result found"
        )
    return warnings


def _checked(text: str, problem: Callable[[float], str | None]) -> float:
    value = float(text)
    found = problem(value)
    if found is not None:
        raise argparse.ArgumentTypeError(f"{found}, got {text}")
    return value


def _non_negative(text: str) -> float:
    return _checked(text, non_negative_problem)


def _positive(text: str) -> float:
    return _checked(text, positive_problem)


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {text}")
    return value


class _Schedule(argparse.Action):
    """Takes --spacing's values, each already positive and finite, as a
    tuple once `schedule_problem` finds nothing wrong with them."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[object] | None,
        option_string: str | None = None,
    ) -> None:
        spacings = tuple(cast(Sequence[float], values))
        problem = schedule_problem(spacings)
        if problem is not None:
            raise argparse.ArgumentError(self, problem)
        setattr(namespace, self.dest, spacings)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kibends", description="Smooth KiCad tracks through the IPC API."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    smooth_parser = subparsers.add_parser(
        "smooth",
        help="Smooth the track chains selected in the PCB editor",
        description="Replace each selected chain of tracks with a short, smooth "
        "path of straight tracks that keeps clearance to other copper and to "
        "the other selected chains. The result minimises length + "
        "repulsion from nearby copper.",
    )
    smooth_parser.add_argument(
        "--repulsion",
        type=_non_negative,
        default=DEFAULT_REPULSION,
        metavar="WEIGHT",
        help="Weight of keeping away from other copper, relative to length; "
        f"0 disables it (default: {DEFAULT_REPULSION:g})",
    )
    smooth_parser.add_argument(
        "--range",
        type=_positive,
        default=DEFAULT_RANGE,
        metavar="MM",
        help="Distance beyond the required clearance at which repulsion fades "
        f"to zero (default: {DEFAULT_RANGE:g})",
    )
    smooth_parser.add_argument(
        "--attraction",
        type=_non_negative,
        default=DEFAULT_ATTRACTION,
        metavar="WEIGHT",
        help="Weight of the pull between ground vias and other vias, so that "
        "signal vias keep their return vias near. While it is above 0, ground "
        "vias of one net also repel each other, with --repulsion and --range. "
        f"0 turns off both (default: {DEFAULT_ATTRACTION:g})",
    )
    smooth_parser.add_argument(
        "--attraction-range",
        type=_positive,
        default=DEFAULT_ATTRACTION_RANGE,
        metavar="MM",
        help="Centre distance at which that pull fades to zero "
        f"(default: {DEFAULT_ATTRACTION_RANGE:g})",
    )
    smooth_parser.add_argument(
        CLEARANCE_FLAGS["edge"],
        type=_non_negative,
        metavar="MM",
        help="Clearance to the board edge (default: the minimum copper to edge "
        "clearance in the project's Board Setup, as last saved, else "
        f"{EDGE_CLEARANCE:g})",
    )
    smooth_parser.add_argument(
        CLEARANCE_FLAGS["hole"],
        type=_non_negative,
        metavar="MM",
        help="Clearance to drilled holes (default: the minimum hole clearance "
        "in the project's Board Setup, as last saved, else "
        f"{HOLE_CLEARANCE:g})",
    )
    smooth_parser.add_argument(
        CLEARANCE_FLAGS["hole_to_hole"],
        type=_non_negative,
        metavar="MM",
        help="Clearance between a moving via's hole and other holes (default: "
        "the minimum hole to hole in the project's Board Setup, as last saved, "
        f"else {HOLE_TO_HOLE:g})",
    )
    smooth_parser.add_argument(
        "--chord-tolerance",
        type=_positive,
        default=DEFAULT_CHORD_TOLERANCE,
        metavar="MM",
        help="How far the committed tracks may depart from the optimised curve; "
        "larger values give fewer, longer tracks "
        f"(default: {DEFAULT_CHORD_TOLERANCE:g})",
    )
    smooth_parser.add_argument(
        "--spacing",
        nargs="+",
        type=_positive,
        action=_Schedule,
        default=DEFAULT_SCHEDULE,
        metavar="MM",
        help="Distances between a chain's vertices that each problem runs at, "
        "coarsest first, each handing over to the next as the problem "
        "settles or --time-limit runs out; the last, at most "
        f"{MAX_SPACING:g}, is the final spacing, and one value runs at it alone "
        f"(default: {listed(DEFAULT_SCHEDULE)})",
    )
    smooth_parser.add_argument(
        "--settled-change",
        type=_positive,
        metavar="MM",
        help="A problem, the chains of one layer or of the layers moving vias "
        "tie together, settles when its burst moves no chain and no via by "
        "this much (default: half --chord-tolerance)",
    )
    smooth_parser.add_argument(
        "--stall-tolerance",
        type=_positive,
        default=DEFAULT_SETTINGS.stall_tolerance,
        metavar="MM",
        help="A problem also settles when, over --stall-window of its bursts, "
        "its energy, in mm of track length, improves by less than this for "
        "each of its moving chains and vias, and its tracks' summed length by "
        "less than this "
        f"(default: {DEFAULT_SETTINGS.stall_tolerance:g})",
    )
    smooth_parser.add_argument(
        "--stall-window",
        type=_positive_int,
        default=DEFAULT_SETTINGS.stall_window,
        metavar="N",
        help=f"Bursts of a problem the stall check looks back over "
        f"(default: {DEFAULT_SETTINGS.stall_window})",
    )
    smooth_parser.add_argument(
        "--max-iterations",
        type=_positive_int,
        default=DEFAULT_SETTINGS.max_iterations,
        metavar="N",
        help="Outer iterations, each one burst of every problem not yet "
        "settled, after which the optimizer stops with the best result found "
        f"(default: {DEFAULT_SETTINGS.max_iterations})",
    )
    smooth_parser.add_argument(
        "--burst-iterations",
        type=_positive_int,
        default=DEFAULT_SETTINGS.burst_maxiter,
        metavar="N",
        help="L-BFGS iterations per burst, which moves every chain and via "
        "of a problem at once; fewer are faster but may leave a different "
        f"result (default: {DEFAULT_SETTINGS.burst_maxiter})",
    )
    smooth_parser.add_argument(
        "--lbfgs-ftol",
        type=_positive,
        default=DEFAULT_SETTINGS.lbfgs_ftol,
        metavar="TOL",
        help="L-BFGS ends a burst early when the energy's relative reduction "
        f"falls below this (default: {DEFAULT_SETTINGS.lbfgs_ftol:g})",
    )
    smooth_parser.add_argument(
        "--lbfgs-gtol",
        type=_positive,
        default=DEFAULT_SETTINGS.lbfgs_gtol,
        metavar="TOL",
        help="L-BFGS ends a burst early when the largest gradient component "
        f"falls below this (default: {DEFAULT_SETTINGS.lbfgs_gtol:g})",
    )
    smooth_parser.add_argument(
        "--time-limit",
        type=_positive,
        metavar="SECONDS",
        help="How long each spacing may optimize, past which a problem's first "
        "burst at a spacing it was handed to still takes one L-BFGS step: "
        "when the time is up, "
        "the problems at a coarse spacing go on at the next and the time starts "
        "again, and once every problem still running is at the final spacing, "
        "optimizing stops and goes on with the best result found so far; so "
        "it takes the time at most once per spacing, besides its overrun "
        "(default: no limit)",
    )
    smooth_parser.add_argument(
        "--all",
        action="store_true",
        help="Smooth every track and via on the board, whatever is selected, "
        "shown or filtered in KiCad; group members stay as drawn",
    )
    smooth_parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Optimize without changing the board",
    )
    smooth_parser.add_argument(
        "--plot",
        type=Path,
        metavar="FILE",
        help="Draw each original and smoothed chain over the area it must "
        "avoid, one subplot per chain; the format follows the extension "
        "(needs the `plot` extra)",
    )
    smooth_parser.add_argument(
        "--allow-via-in-pad",
        action="store_true",
        help="Let a moving via settle inside a pad of its own net. KiCad's DRC "
        "does not check via-in-pad; without this option a moving via never "
        "overlaps a pad",
    )
    smooth_parser.add_argument(
        "--ground-nets",
        nargs="+",
        default=(),
        metavar="NET",
        help="The ground nets, whose vias are return vias for the other nets' "
        "vias; by default the nets with copper zones on at least two inner "
        "layers, else GND if it has copper. Ignored with --attraction 0",
    )
    _add_verbose(smooth_parser)
    return parser.parse_args(argv)


def _add_verbose(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help=f"Log at {VERBOSE_LOG_LEVEL}, overriding the level "
        f"{LOG_LEVEL_VARIABLE} sets",
    )


def parse_panel_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kibends-panel",
        description="Open a small window that runs kibends smooth on the PCB "
        "editor's selection, or on the whole board when nothing is selected, "
        "shows its progress and can stop it, keeping the result so far. Needs "
        "the gui extra.",
    )
    _add_verbose(parser)
    return parser.parse_args(argv)


def smooth_options(args: argparse.Namespace) -> SmoothOptions:
    return present.smooth_options(
        SmoothValues(
            repulsion=args.repulsion,
            falloff_range=args.range,
            attraction=args.attraction,
            attraction_range=args.attraction_range,
            edge_clearance=args.edge_clearance,
            hole_clearance=args.hole_clearance,
            hole_to_hole=args.hole_to_hole,
            override_names=CLEARANCE_FLAGS,
            chord_tolerance=args.chord_tolerance,
            spacing=args.spacing,
            settled_change=args.settled_change,
            stall_tolerance=args.stall_tolerance,
            stall_window=args.stall_window,
            max_iterations=args.max_iterations,
            burst_iterations=args.burst_iterations,
            lbfgs_ftol=args.lbfgs_ftol,
            lbfgs_gtol=args.lbfgs_gtol,
            time_limit=args.time_limit,
            scope=ScopeOption.BOARD if args.all else ScopeOption.SELECTION,
            dry_run=args.dry_run,
            plot=args.plot,
            allow_via_in_pad=args.allow_via_in_pad,
            ground_nets=tuple(args.ground_nets),
        )
    )


def execute_smooth(args: argparse.Namespace) -> None:
    options = smooth_options(args)
    with _exit_on_failure():
        smooth(options)


def panel_main(argv: Sequence[str] | None = None) -> None:
    """The `kibends-panel` command."""
    args = parse_panel_args(argv)
    configure_logging(verbose=args.verbose)
    try:
        from kibends.panel.window import main as open_panel
    except ImportError as e:
        _fail(f"{GUI_EXTRA_MESSAGE} ({e})")
    open_panel()


def log_level(*, verbose: bool) -> str:
    """DEBUG when verbose, else the level in LOGURU_LEVEL, else INFO."""
    if verbose:
        return VERBOSE_LOG_LEVEL
    return os.environ.get(LOG_LEVEL_VARIABLE, DEFAULT_LOG_LEVEL)


def configure_logging(*, verbose: bool) -> None:
    """Log to stderr at `log_level`.

    Loguru's own default sink logs at DEBUG.
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level(verbose=verbose))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    match args.command:
        case "smooth":
            execute_smooth(args)
        case _:
            assert_never(args.command)
