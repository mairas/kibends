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
from pathlib import Path
from typing import NoReturn, assert_never

import numpy as np
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from loguru import logger

from kibends import drc
from kibends.bursts import DEFAULT_SETTINGS
from kibends.chain import ChainError, Point2
from kibends.commit import (
    BoardChangedError,
    CommitUncertainError,
    CreateItemsError,
    DeleteItemsError,
)
from kibends.energy import EnergyParams
from kibends.obstacles import EDGE_CLEARANCE, HOLE_CLEARANCE
from kibends.optimize import FrozenVertex, Termination
from kibends.run import (
    KICAD_TIMEOUT_MS,
    ChainReport,
    NoBoardError,
    OptimizerReport,
    PairReport,
    PlotUnavailableError,
    SmoothOptions,
    ViaReport,
    new_violations,
    run_smooth,
)
from kibends.simplify import DEFAULT_CHORD_TOLERANCE
from kibends.snapshot import SelectionError

DEFAULT_STIFFNESS = 0.3
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
    """The optimizer or the DRC check refused the result; nothing committed."""


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
    drc.KiCadCliNotFound,
    drc.ProjectFilesError,
    drc.BoardTextError,
    drc.DrcError,
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
    """Smooth the chains selected in KiCad, print the report and, unless
    dry-running, commit them."""

    def show(optimized: OptimizerReport) -> None:
        print(format_statistics(optimized))
        for warning in report_warnings(optimized):
            logger.warning(warning)

    report = run_smooth(options, show)
    if report.drc is not None:
        print(_verdict(report.drc))
    if report.drc and options.dry_run:
        logger.warning("A real run would refuse to commit this result")
    if report.refusal is not None:
        raise RunRefusedError(report.refusal)
    if not any(c.changed for c in report.chains):
        logger.info("Nothing to do: the optimizer leaves the selection as it is")
    elif options.dry_run:
        logger.info("Dry run: nothing committed")


def format_statistics(report: OptimizerReport) -> str:
    """One statistics block per chain and per differential pair, a line per
    movable via and the optimizer's outcome."""
    some_changed = any(c.changed for c in report.chains)
    blocks = [_chain_block(c, some_changed) for c in report.chains]
    blocks += [_pair_block(p) for p in report.pairs]
    if report.vias:
        blocks.append("\n".join(_via_line(v) for v in report.vias))
    several = len(report.chains) > 1
    rejected = (
        sum(c.rejected_edges for c in report.chains)
        + sum(v.rejected_edges for v in report.vias)
        + sum(p.rejected_edges for p in report.pairs)
    )
    # A run can converge while a chain gave up after abnormal bursts.
    abnormal = [
        f"{n} abnormal burst{'' if n == 1 else 's'}"
        + (f" in {description}" if several else "")
        for description, n in _abnormal(report)
    ]
    outcome = ", ".join(
        [
            f"Optimizer: {report.termination.value} after {report.iterations} "
            f"iterations",
            f"{rejected} rejected edge moves",
            *abnormal,
        ]
    )
    return ("\n\n" if several else "\n").join([*blocks, outcome])


def _via_line(v: ViaReport) -> str:
    if not v.moved:
        return f"Via {v.id} stays at {_point(v.start)}"
    distance = math.dist(v.start, v.end)
    return (
        f"Via {v.id} moved {distance:.3f} mm, from {_point(v.start)} to {_point(v.end)}"
    )


def _point(p: Point2) -> str:
    return f"({p[0]:.3f}, {p[1]:.3f})"


def report_warnings(report: OptimizerReport) -> list[str]:
    """Frozen vertices and vias, abnormal bursts and an iteration cap, one
    line each."""
    warnings = [_frozen_warning(v) for c in report.chains for v in c.frozen]
    warnings += [
        f"Left via {v.id} at {_point(v.start)} in place: it starts inside "
        "another item's clearance"
        for v in report.vias
        if v.frozen
    ]
    warnings += [_frozen_warning(v) for p in report.pairs for v in p.frozen]
    warnings += [
        f"L-BFGS-B ended {n} burst{'' if n == 1 else 's'} of {description} "
        "abnormally; the run judged each like any other burst"
        for description, n in _abnormal(report)
    ]
    if report.termination is Termination.ITERATION_CAP:
        warnings.append(
            f"The optimizer stopped at its cap of {DEFAULT_SETTINGS.max_iterations} "
            "outer iterations without converging; using the best result found"
        )
    return warnings


def _abnormal(report: OptimizerReport) -> list[tuple[str, int]]:
    """What had abnormal bursts, and how many."""
    return [
        (description, n)
        for description, n in (
            *((c.description, c.abnormal_bursts) for c in report.chains),
            *(
                (f"the pair {' and '.join(p.nets)}", p.abnormal_bursts)
                for p in report.pairs
            ),
        )
        if n > 0
    ]


def _frozen_warning(v: FrozenVertex) -> str:
    message = (
        f"Left the vertex at ({v.position[0]:.3f}, {v.position[1]:.3f}) in "
        f"place: it starts {-v.distance:.3f} mm inside another item's clearance"
    )
    if v.neighbours:
        message += (
            f", that of the selected chain {' and '.join(v.neighbours)}; "
            "a re-run may free it"
        )
    return message


def _chain_block(chain: ChainReport, some_changed: bool) -> str:
    before, after = chain.before, chain.after
    rows = [
        ("Length (mm)", f"{before.length:.3f}", f"{after.length:.3f}"),
        ("Tracks", str(before.segment_count), str(after.segment_count)),
        (
            "Tightest bend radius (mm)",
            _radius(before.tightest_bend_radius),
            _radius(after.tightest_bend_radius),
        ),
        (
            "Min clearance margin (mm)",
            f"{before.min_clearance:.4f}",
            f"{after.min_clearance:.4f}",
        ),
        (
            "Mean clearance margin (mm)",
            f"{before.mean_clearance:.4f}",
            f"{after.mean_clearance:.4f}",
        ),
    ]
    lines = [
        f"{chain.tracks} tracks of {chain.net} on {chain.layer}, "
        f"{chain.width:g} mm wide",
        *_table(rows),
        f"Frozen vertices: {len(chain.frozen)}",
    ]
    if some_changed and not chain.changed:
        lines.append("Unchanged: left out of the DRC check and the commit")
    return "\n".join(lines)


def _pair_block(pair: PairReport) -> str:
    low, high = pair.gap_after
    rows = [
        ("Gap (mm)", f"{pair.gap:.3f}", f"{low:.3f}-{high:.3f}"),
        ("Skew (mm)", f"{pair.skew_before:.3f}", f"{pair.skew_after:.3f}"),
    ]
    sections = f"{pair.sections} coupled section{'' if pair.sections == 1 else 's'}"
    lines = [
        f"Differential pair {' and '.join(pair.nets)} on {pair.layer}, {sections}",
        *_table(rows),
    ]
    if pair.held:
        one = pair.held == 1
        lines.append(
            f"{pair.held} coupled section{'' if one else 's'} held as routed, "
            "for same-net copper between its ends, routing that bends too "
            "tightly, or copper or its own parts beyond a turn that kept it "
            "from rounding a corner"
        )
    return "\n".join(lines)


def _table(rows: Sequence[tuple[str, str, str]]) -> list[str]:
    """Labelled before and after columns."""
    label_width = max(len(r[0]) for r in rows)
    return [
        f"{'':{label_width}}  {'before':>10}  {'after':>10}",
        *(f"{name:{label_width}}  {b:>10}  {a:>10}" for name, b, a in rows),
    ]


def _verdict(found: drc.Introduced) -> str:
    if not found:
        return "DRC: no new violations"
    return "\n".join(
        [
            f"DRC: {new_violations(len(found.violations))}",
            *(f"  {v}" for v in found.violations),
            *(
                f"  More connectivity violations on {net}: the new path may "
                f"split a pour of {net}"
                for net in sorted(found.split_nets)
            ),
        ]
    )


def _radius(r: float) -> str:
    return "straight" if np.isinf(r) else f"{r:.3f}"


def _non_negative(text: str) -> float:
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must not be negative, got {text}")
    return value


def _positive(text: str) -> float:
    value = float(text)
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
        "the other selected chains. The result minimises length + bending "
        "weighted by stiffness squared + repulsion from nearby copper.",
    )
    smooth_parser.add_argument(
        "--stiffness",
        type=_non_negative,
        default=DEFAULT_STIFFNESS,
        metavar="MM",
        help="Weight of bending, as a length: larger values give wider, rounder "
        f"bends (default: {DEFAULT_STIFFNESS:g})",
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
        "--dry-run",
        "-n",
        action="store_true",
        help="Print the statistics without changing the board",
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
        "--no-drc",
        action="store_true",
        help="Commit without the kicad-cli DRC check against the board's rules",
    )
    smooth_parser.add_argument(
        "--kicad-cli",
        metavar="PATH",
        help="kicad-cli to run the DRC check with (default: the one the running "
        "KiCad reports, then PATH, then the usual install locations)",
    )
    return parser.parse_args(argv)


def smooth_options(args: argparse.Namespace) -> SmoothOptions:
    return SmoothOptions(
        params=EnergyParams(
            stiffness=args.stiffness,
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
        drc=not args.no_drc,
        kicad_cli=args.kicad_cli,
        allow_via_in_pad=args.allow_via_in_pad,
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
