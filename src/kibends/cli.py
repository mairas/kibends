"""The `kibends` command line.

`_exit_on`-style error mapping and `configure_logging` follow kistitch's
cli.py (MIT, Copyright (c) 2025 Hat Labs Oy).
"""

import argparse
import os
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, Protocol

import kipy
import kipy.board
import numpy as np
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from loguru import logger

from kibends import drc
from kibends.chain import Chain, ChainError, Segment, extract_chain
from kibends.commit import (
    BoardChangedError,
    CommitUncertainError,
    CreateItemsError,
    DeleteItemsError,
    commit_chain,
    snapped_points,
)
from kibends.energy import DistanceField, EnergyParams, FloatArray
from kibends.obstacles import EDGE_CLEARANCE, HOLE_CLEARANCE, build_obstacles
from kibends.optimize import (
    DEFAULT_SETTINGS,
    FrozenVertex,
    OptimizeResult,
    PolylineStats,
    Termination,
    measure,
    optimize,
)
from kibends.simplify import DEFAULT_CHORD_TOLERANCE, simplify
from kibends.snapshot import LayerSnapshot, SelectionError, read_selection

DEFAULT_STIFFNESS = 0.3
DEFAULT_REPULSION = 0.5
DEFAULT_RANGE = 1.0
# Not user-facing: the clearance hinge only shapes the approach to the
# clearance boundary, which the bubble bounds already keep vertices outside.
CLEARANCE_WEIGHT = 50.0
# Above the worst-case sag between quadrature points at the default spacing.
STANDOFF = 0.02
POINTS_PER_EDGE = 3

# kipy's 2 s default is too short for reading a large board as text and for
# deleting or creating items on one.
KICAD_TIMEOUT_MS = 30_000

BUSY_MESSAGE = (
    "KiCad is busy — close any open tool or dialog in the PCB editor and retry"
)
PLOT_EXTRA_MESSAGE = (
    "--plot needs matplotlib, which comes with the `plot` extra: "
    "uv tool install 'kibends[plot] @ git+https://github.com/mairas/kibends'"
)

DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL_VARIABLE = "LOGURU_LEVEL"


class NoBoardError(RuntimeError):
    pass


class PlotUnavailableError(RuntimeError):
    pass


class OptimizerFailedError(RuntimeError):
    pass


class DrcViolationsError(RuntimeError):
    pass


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
    OptimizerFailedError,
    DrcViolationsError,
    drc.KiCadCliNotFound,
    drc.ProjectFilesError,
    drc.BoardTextError,
    drc.DrcError,
)


class Renderer(Protocol):
    def __call__(
        self,
        path: Path,
        field: DistanceField,
        *,
        original: FloatArray,
        optimised: FloatArray,
        simplified: FloatArray,
        frozen: Sequence[FrozenVertex],
        title: str,
    ) -> None: ...


@dataclass(frozen=True)
class SmoothOptions:
    params: EnergyParams
    edge_clearance: float
    hole_clearance: float
    chord_tolerance: float
    dry_run: bool
    plot: Path | None
    drc: bool
    kicad_cli: str | None


@dataclass(frozen=True)
class DrcSetup:
    kicad_cli: Path
    project: drc.ProjectFiles


def _fail(message: str) -> NoReturn:
    logger.error(message)
    raise SystemExit(1)


@contextmanager
def _exit_on_failure() -> Iterator[None]:
    """Report an expected failure as one log line and exit 1."""
    try:
        yield
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
    """Smooth the chain selected in KiCad and, unless dry-running, commit it."""
    render = _plot_renderer() if options.plot is not None else None
    kicad = kipy.KiCad(timeout_ms=KICAD_TIMEOUT_MS)
    board = _open_board(kicad)
    gate = _prepare_drc(kicad, board, options.kicad_cli) if options.drc else None
    if gate is None:
        logger.warning(
            "Skipping the DRC check (--no-drc): custom rules, board minimums "
            "and edge and hole clearances are not checked"
        )

    selected = read_selection(board)
    layer, net = _layer_and_net(selected)
    # Locked tracks stay on the board, so they are obstacles or pins like any
    # unselected track.
    replaced = {s.id for s in selected if not s.locked}
    snapshot = LayerSnapshot.from_board(board, layer, replaced)
    chain = extract_chain(selected, snapshot.same_net_copper(net))
    obstacles = build_obstacles(
        snapshot,
        chain.net,
        chain.width,
        edge_clearance=options.edge_clearance,
        hole_clearance=options.hole_clearance,
    )

    original = np.array(chain.points)
    result = optimize(original, np.array(chain.pinned), obstacles, options.params)
    nothing_to_do = result.termination is Termination.NOTHING_TO_DO
    simplified = (
        result.polyline
        if nothing_to_do
        else simplify(
            result.polyline,
            result.fixed,
            obstacles,
            tolerance=options.chord_tolerance,
        )
    )
    after = measure(simplified, obstacles, DEFAULT_SETTINGS.spacing)
    print(
        _report(
            f"{len(chain.segment_ids)} tracks of {chain.net} on {chain.layer}, "
            f"{chain.width:g} mm wide",
            result,
            after,
        )
    )
    for v in result.frozen:
        logger.warning(
            f"Left the vertex at ({v.position[0]:.3f}, {v.position[1]:.3f}) in "
            f"place: it starts {-v.distance:.3f} mm inside another item's clearance"
        )

    if render is not None and options.plot is not None:
        render(
            options.plot,
            obstacles,
            original=original,
            optimised=result.polyline,
            simplified=simplified,
            frozen=result.frozen,
            title=f"{chain.net} on {chain.layer}: {result.termination.value}, "
            f"length {result.before.length:.3f} → {after.length:.3f} mm, "
            f"min clearance margin {after.min_clearance:.4f} mm",
        )
        logger.info(f"Wrote {options.plot}")

    if result.termination is Termination.ABNORMAL:
        raise OptimizerFailedError(
            "The optimizer terminated abnormally; nothing committed. "
            "Try other --stiffness, --repulsion or --range values"
        )
    if nothing_to_do:
        logger.info("Nothing to do: the optimizer leaves the chain as it is")
        return
    if result.termination is Termination.ITERATION_CAP:
        logger.warning(
            f"The optimizer stopped at its cap of {DEFAULT_SETTINGS.max_iterations} "
            "iterations without converging; using the best result found"
        )
    if gate is not None:
        _check_drc(gate, board, chain, simplified, options.dry_run)
    if options.dry_run:
        logger.info("Dry run: nothing committed")
        return
    commit_chain(board, chain, selected, simplified)


def _prepare_drc(
    kicad: kipy.KiCad, board: kipy.board.Board, explicit_cli: str | None
) -> DrcSetup:
    kicad_cli = drc.locate_kicad_cli(explicit_cli, kicad)
    logger.info(f"Using {kicad_cli} ({drc.kicad_cli_version(kicad_cli)})")
    project = drc.project_files(board)
    logger.info(
        f"DRC uses the project files on disk: {project.pro}"
        + (f", {project.dru}" if project.dru else " (no .kicad_dru)")
        + "; unsaved rule or netclass edits in KiCad are not seen"
    )
    return DrcSetup(kicad_cli, project)


def _check_drc(
    gate: DrcSetup,
    board: kipy.board.Board,
    chain: Chain,
    polyline: FloatArray,
    dry_run: bool,
) -> None:
    """Print the DRC verdict; raise if the result adds violations."""
    tracks = drc.NewTracks(
        points=tuple(snapped_points(polyline)),
        width=chain.width,
        layer=chain.layer,
        net=chain.net,
    )
    with tempfile.TemporaryDirectory(prefix="kibends-drc-") as tmp:
        found = drc.check(
            gate.kicad_cli,
            gate.project,
            board.get_as_string(),
            chain.segment_ids,
            tracks,
            Path(tmp),
        )
    print(_verdict(found))
    if not found:
        return
    if dry_run:
        logger.warning("A real run would refuse to commit this result")
        return
    raise DrcViolationsError(
        f"DRC: the result adds {_new_violations(len(found.violations))}; "
        "nothing committed"
    )


def _new_violations(n: int) -> str:
    return f"{n} new violation{'' if n == 1 else 's'}"


def _verdict(found: drc.Introduced) -> str:
    if not found:
        return "DRC: no new violations"
    return "\n".join(
        [
            f"DRC: {_new_violations(len(found.violations))}",
            *(f"  {v}" for v in found.violations),
            *(
                f"  More connectivity violations on {net}: the new path may "
                f"split a pour of {net}"
                for net in sorted(found.split_nets)
            ),
        ]
    )


def _plot_renderer() -> Renderer:
    try:
        from kibends.plot import render
    except ImportError as e:
        raise PlotUnavailableError(f"{PLOT_EXTRA_MESSAGE} ({e})") from None
    return render


def _open_board(kicad: kipy.KiCad) -> kipy.board.Board:
    try:
        return kicad.get_board()
    except ApiError as e:
        if e.code == ApiStatusCode.AS_BUSY:
            raise
        raise NoBoardError(f"No board is open in KiCad's PCB editor ({e})") from None


def _layer_and_net(selected: Sequence[Segment]) -> tuple[str, str]:
    """The layer and net of the chain, read before the chain is built.

    Mixed layers or nets are left for `extract_chain` to report.
    """
    first = next((s for s in selected if not s.locked), selected[0])
    return first.layer, first.net


def _report(heading: str, result: OptimizeResult, after: PolylineStats) -> str:
    before = result.before
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
    label_width = max(len(r[0]) for r in rows)
    lines = [
        heading,
        f"{'':{label_width}}  {'before':>10}  {'after':>10}",
        *(f"{name:{label_width}}  {b:>10}  {a:>10}" for name, b, a in rows),
        f"Frozen vertices: {len(result.frozen)}",
        (
            f"Optimizer: {result.termination.value} after {result.iterations} "
            f"iterations, {result.rejected_edges} rejected edge moves"
        ),
    ]
    return "\n".join(lines)


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
        help="Smooth the track chain selected in the PCB editor",
        description="Replace the selected chain of tracks with a short, smooth "
        "path of straight tracks that keeps clearance to other copper. The "
        "result minimises length + stiffness² × bending + repulsion from "
        "nearby copper.",
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
        f"larger values give fewer, longer tracks (default: {DEFAULT_CHORD_TOLERANCE:g})",
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
        help="Draw the original and smoothed chain over the area it must avoid; "
        "the format follows the extension (needs the `plot` extra)",
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


def execute_smooth(args: argparse.Namespace) -> None:
    options = SmoothOptions(
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
    )
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
