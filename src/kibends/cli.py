"""The `kibends` command line.

`_exit_on`-style error mapping and `configure_logging` follow kistitch's
cli.py (MIT, Copyright (c) 2025 Hat Labs Oy).
"""

import argparse
import math
import os
import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import NoReturn, assert_never

from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from loguru import logger

from kibends.bursts import DEFAULT_SETTINGS
from kibends.chain import ChainError
from kibends.commit import (
    BoardChangedError,
    CommitUncertainError,
    CreateItemsError,
    DeleteItemsError,
)
from kibends.coords import UserFrame
from kibends.energy import EnergyParams
from kibends.obstacles import EDGE_CLEARANCE, HOLE_CLEARANCE
from kibends.optimize import FrozenVertex, Termination
from kibends.run import (
    KICAD_TIMEOUT_MS,
    NoBoardError,
    OptimizerReport,
    PlotUnavailableError,
    SmoothOptions,
    run_smooth,
)
from kibends.simplify import DEFAULT_CHORD_TOLERANCE
from kibends.snapshot import SelectionError

DEFAULT_REPULSION = 0.5
DEFAULT_RANGE = 1.0
# Not user-facing: the clearance hinge only shapes the approach to the
# clearance boundary, which the bubble bounds already keep vertices outside.
CLEARANCE_WEIGHT = 50.0
# Above the worst-case sag between quadrature points at the default spacing.
STANDOFF = 0.02
POINTS_PER_EDGE = 3

BUSY_MESSAGE = (
    "KiCad is busy — close any open tool or dialog in the PCB editor and retry"
)

DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL_VARIABLE = "LOGURU_LEVEL"


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
    NoBoardError,
    PlotUnavailableError,
    RunRefusedError,
)


def _fail(message: str) -> NoReturn:
    logger.error(message)
    raise SystemExit(1)


@contextmanager
def _exit_on_failure() -> Generator[None]:
    """Report an expected failure as one log line and exit 1."""
    try:
        yield
    except KeyboardInterrupt:
        # The push maps its own interrupt to CommitUncertainError.
        _fail("Interrupted: nothing committed")
    except EXPECTED_ERRORS as e:
        _fail(str(e))
    except ApiError as e:
        _fail(
            BUSY_MESSAGE
            if e.code == ApiStatusCode.AS_BUSY
            else f"KiCad refused the request: {e}"
        )
    except KiCadConnectionError as e:
        _fail(
            f"Cannot reach KiCad ({e}): either it is not running — start KiCad "
            "10 or later with the API server enabled (Preferences > Plugins) — "
            f"or a request timed out after {KICAD_TIMEOUT_MS / 1000:g} s"
        )


def smooth(options: SmoothOptions) -> None:
    """Smooth the chains selected in KiCad, print the optimizer's outcome
    and, unless dry-running, commit them."""

    def show(optimized: OptimizerReport) -> None:
        print(format_outcome(optimized))
        for warning in report_warnings(optimized):
            logger.warning(warning)
        for v in optimized.vias:
            if v.frozen:
                logger.debug(
                    f"Left via {v.id} at {optimized.frame.point(v.start)} in "
                    "place: it starts inside another item's clearance or a rule "
                    "area keeping out vias"
                )
        for description, n in optimized.abnormal:
            logger.debug(
                f"L-BFGS-B ended {n} burst{'' if n == 1 else 's'} of "
                f"{description} abnormally; the run judged each like any other"
            )

    report = run_smooth(options, show)
    if report.refusal is not None:
        raise RunRefusedError(report.refusal)
    if not any(c.changed for c in report.chains):
        logger.info("Nothing to do: the optimizer leaves the selection as it is")
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
    """Frozen vertices and an iteration cap or time limit, one line each,
    positions in the report's frame."""
    frame = report.frame
    warnings = [_frozen_warning(v, frame) for c in report.chains for v in c.frozen]
    warnings += [_frozen_warning(v, frame) for p in report.pairs for v in p.frozen]
    if report.termination is Termination.ITERATION_CAP:
        warnings.append(
            f"The optimizer stopped at its cap of {report.iterations} "
            "outer iterations without converging; using the best result found"
        )
    if report.termination is Termination.TIME_LIMIT:
        warnings.append(
            "The optimizer stopped at its time limit without converging; "
            "using the best result found"
        )
    return warnings


def _frozen_warning(v: FrozenVertex, frame: UserFrame) -> str:
    message = (
        f"Left the vertex at {frame.point(v.position)} in "
        f"place: it starts {-v.distance:.3f} mm inside another item's clearance"
    )
    if v.neighbours:
        message += (
            f", that of the selected chain {' and '.join(v.neighbours)}; "
            "a re-run may free it"
        )
    return message


def _finite(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be a finite number, got {text}")
    return value


def _non_negative(text: str) -> float:
    value = _finite(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must not be negative, got {text}")
    return value


def _positive(text: str) -> float:
    value = _finite(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {text}")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {text}")
    return value


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
        "--edge-clearance",
        type=_non_negative,
        default=EDGE_CLEARANCE,
        metavar="MM",
        help="Clearance to the board edge; KiCad's API does not expose the "
        f"board's own (default: {EDGE_CLEARANCE:g})",
    )
    smooth_parser.add_argument(
        "--hole-clearance",
        type=_non_negative,
        default=HOLE_CLEARANCE,
        metavar="MM",
        help="Clearance to drilled holes; KiCad's API does not expose the "
        f"board's own (default: {HOLE_CLEARANCE:g})",
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
        "--settled-change",
        type=_positive,
        default=DEFAULT_SETTINGS.settled_change,
        metavar="MM",
        help="A problem, the chains of one layer or of the layers moving vias "
        "tie together, settles when its burst moves no chain and no via by "
        f"this much (default: {DEFAULT_SETTINGS.settled_change:g})",
    )
    smooth_parser.add_argument(
        "--stall-tolerance",
        type=_positive,
        default=DEFAULT_SETTINGS.stall_tolerance,
        metavar="MM",
        help="A problem also settles when its energy, in mm of track length, "
        "improves by less than this over --stall-window of its bursts "
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
        help="L-BFGS-B iterations per burst, which moves every chain and via "
        "of a problem at once; fewer are faster but may leave a different "
        f"result (default: {DEFAULT_SETTINGS.burst_maxiter})",
    )
    smooth_parser.add_argument(
        "--lbfgs-ftol",
        type=_positive,
        default=DEFAULT_SETTINGS.lbfgs_ftol,
        metavar="TOL",
        help="L-BFGS-B ends a burst early when the energy's relative reduction "
        f"falls below this (default: {DEFAULT_SETTINGS.lbfgs_ftol:g})",
    )
    smooth_parser.add_argument(
        "--lbfgs-gtol",
        type=_positive,
        default=DEFAULT_SETTINGS.lbfgs_gtol,
        metavar="TOL",
        help="L-BFGS-B ends a burst early when the projected gradient falls "
        f"below this (default: {DEFAULT_SETTINGS.lbfgs_gtol:g})",
    )
    smooth_parser.add_argument(
        "--no-move-bounds",
        action="store_true",
        help="Let every vertex and via move without bounds in a burst. Tracks "
        "and vias may then pass through each other, which the optimizer does "
        "not always catch, so inspect the result and run KiCad's DRC",
    )
    smooth_parser.add_argument(
        "--time-limit",
        type=_positive,
        metavar="SECONDS",
        help="Stop optimizing after this long and go on with the best result "
        "found so far (default: no limit)",
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
    return parser.parse_args(argv)


def smooth_options(args: argparse.Namespace) -> SmoothOptions:
    return SmoothOptions(
        params=EnergyParams(
            repulsion=args.repulsion,
            falloff_range=args.range,
            clearance_weight=CLEARANCE_WEIGHT,
            standoff=STANDOFF,
            points_per_edge=POINTS_PER_EDGE,
        ),
        edge_clearance=args.edge_clearance,
        hole_clearance=args.hole_clearance,
        chord_tolerance=args.chord_tolerance,
        dry_run=args.dry_run,
        plot=args.plot,
        allow_via_in_pad=args.allow_via_in_pad,
        settings=replace(
            DEFAULT_SETTINGS,
            settled_change=args.settled_change,
            stall_tolerance=args.stall_tolerance,
            stall_window=args.stall_window,
            max_iterations=args.max_iterations,
            burst_maxiter=args.burst_iterations,
            lbfgs_ftol=args.lbfgs_ftol,
            lbfgs_gtol=args.lbfgs_gtol,
            move_bounds=not args.no_move_bounds,
            time_limit=args.time_limit,
        ),
    )


def execute_smooth(args: argparse.Namespace) -> None:
    options = smooth_options(args)
    with _exit_on_failure():
        smooth(options)


def configure_logging() -> None:
    """Log to stderr at INFO, or at the level in LOGURU_LEVEL.

    Loguru's own default sink logs at DEBUG.
    """
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get(LOG_LEVEL_VARIABLE, DEFAULT_LOG_LEVEL))


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    match args.command:
        case "smooth":
            execute_smooth(args)
        case _:
            assert_never(args.command)
