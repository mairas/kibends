"""One smoothing run, from the KiCad selection to the commit.

The command line and the GUI both call `run_smooth` and present the reports
it hands over; it prints nothing.
"""

import importlib
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import kipy
import kipy.board
import numpy as np
from kipy.errors import ApiError
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from loguru import logger

from kibends import drc
from kibends.chain import Chain, extract_chains
from kibends.commit import commit_chains, snapped_path, snapped_points
from kibends.energy import EnergyParams, FloatArray
from kibends.neighbours import (
    ChainInput,
    chain_field,
    neighbour_fields,
    obstacle_reach,
    same_layer_neighbours,
)
from kibends.obstacles import ChainCentreline, Obstacles, build_obstacles
from kibends.optimize import (
    DEFAULT_SETTINGS,
    FrozenVertex,
    PolylineStats,
    RunResult,
    Termination,
    measure,
    optimize_chains,
)
from kibends.simplify import simplify_chains
from kibends.snapshot import BoardSnapshot, LayerSnapshot, read_selection

# kipy's 2 s default is too short for reading a large board as text and for
# deleting or creating items on one.
KICAD_TIMEOUT_MS = 30_000

PLOT_EXTRA_MESSAGE = (
    "--plot needs matplotlib, which comes with the `plot` extra: "
    "uv tool install 'kibends[plot] @ git+https://github.com/mairas/kibends'"
)
ABNORMAL_MESSAGE = (
    "The optimizer terminated abnormally without improving the track; "
    "nothing committed. "
    "Try other --stiffness, --repulsion or --range values"
)


class NoBoardError(RuntimeError):
    pass


class PlotUnavailableError(RuntimeError):
    pass


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
class ChainReport:
    net: str
    layer: str
    width: float
    """mm."""
    description: str
    """How warnings and other chains' frozen vertices name this chain."""
    tracks: int
    """Selected tracks it replaces."""
    before: PolylineStats
    after: PolylineStats
    """Of the simplified result, or of the selection when unchanged, with the
    other chains at their simplified results."""
    frozen: tuple[FrozenVertex, ...]
    rejected_edges: int
    abnormal_bursts: int
    changed: bool
    """False when the result equals the selected tracks: then it is neither
    checked nor committed."""


@dataclass(frozen=True)
class OptimizerReport:
    chains: tuple[ChainReport, ...]
    """In the order `extract_chains` gives."""
    termination: Termination
    iterations: int
    """Outer iterations, each giving every chain one burst."""


@dataclass(frozen=True)
class RunReport(OptimizerReport):
    drc: drc.Introduced | None
    """None when the DRC check did not run."""
    committed: bool
    refusal: str | None
    """Why nothing was committed, when that is a failure of the run."""


@dataclass(frozen=True)
class _DrcSetup:
    kicad_cli: Path
    project: drc.ProjectFiles


def run_smooth(
    options: SmoothOptions,
    on_optimized: Callable[[OptimizerReport], None] = lambda _report: None,
) -> RunReport:
    """Smooth the chains selected in KiCad and, unless dry-running, commit them.

    `on_optimized` gets the optimizer's report before the DRC check and the
    commit, which may raise. Raises on failures that leave nothing to report
    (no board, a bad selection, a refused commit); an abnormal optimizer stop
    or new DRC violations come back as the report's `refusal`, with nothing
    committed.
    """
    if options.plot is not None:
        _require_plot()
    kicad = kipy.KiCad(timeout_ms=KICAD_TIMEOUT_MS)
    board = _open_board(kicad)
    gate = _prepare_drc(kicad, board, options.kicad_cli) if options.drc else None
    if gate is None:
        logger.warning(
            "Skipping the DRC check (--no-drc): custom rules, board minimums "
            "and edge and hole clearances are not checked"
        )

    selected = read_selection(board)
    # Locked tracks stay on the board, so they are obstacles or pins like any
    # unselected track.
    replaced = {s.id for s in selected if not s.locked}
    layers = {s.layer for s in selected if not s.locked}
    whole = BoardSnapshot.read(board, layers)
    snapshots = {layer: whole.slice(layer, replaced) for layer in sorted(layers)}
    chains = extract_chains(
        selected, lambda layer, net: snapshots[layer].same_net_copper(net)
    )
    inputs = _chain_inputs(chains, snapshots, options)

    run = optimize_chains(inputs, options.params)
    nothing_to_do = run.termination is Termination.NOTHING_TO_DO
    simplified = (
        [c.polyline for c in run.chains]
        if nothing_to_do
        else simplify_chains(
            inputs, run, options.params, tolerance=options.chord_tolerance
        )
    )
    reports = _chain_reports(chains, inputs, run, simplified, options.params)

    if options.plot is not None:
        _draw(options.plot, inputs, run, simplified, reports)
    on_optimized(OptimizerReport(reports, run.termination, run.iterations))

    def report(
        found: drc.Introduced | None = None,
        *,
        committed: bool = False,
        refusal: str | None = None,
    ) -> RunReport:
        return RunReport(
            reports, run.termination, run.iterations, found, committed, refusal
        )

    if run.termination is Termination.ABNORMAL:
        return report(refusal=ABNORMAL_MESSAGE)
    changed = [
        (chain, polyline)
        for chain, polyline, r in zip(chains, simplified, reports, strict=True)
        if r.changed
    ]
    if not changed:
        return report()
    found = None
    if gate is not None:
        found = _check_drc(gate, board, changed)
        if found and not options.dry_run:
            return report(found, refusal=_drc_refusal(found))
    if options.dry_run:
        return report(found)
    commit_chains(board, changed, selected)
    return report(found, committed=True)


def _chain_inputs(
    chains: Sequence[Chain],
    snapshots: Mapping[str, LayerSnapshot],
    options: SmoothOptions,
) -> list[ChainInput]:
    """One optimizer input per chain, over its layer's snapshot; chains of one
    layer, net and width share a field."""
    fields: dict[tuple[str, str, float], Obstacles] = {}
    inputs: list[ChainInput] = []
    for chain in chains:
        snapshot = snapshots[chain.layer]
        key = (chain.layer, chain.net, chain.width)
        if key not in fields:
            fields[key] = build_obstacles(
                snapshot,
                chain.net,
                chain.width,
                edge_clearance=options.edge_clearance,
                hole_clearance=options.hole_clearance,
            )
        x, y = chain.points[0]
        centreline = ChainCentreline(
            f"{chain.net} on {chain.layer} from ({x:.3f}, {y:.3f})",
            chain.net,
            chain.width,
            snapshot.clearance(chain.net),
            np.array(chain.points),
        )
        inputs.append(
            ChainInput(centreline, chain.layer, np.array(chain.pinned), fields[key])
        )
    return inputs


def _chain_reports(
    chains: Sequence[Chain],
    inputs: Sequence[ChainInput],
    run: RunResult,
    simplified: Sequence[FloatArray],
    params: EnergyParams,
) -> tuple[ChainReport, ...]:
    neighbours = same_layer_neighbours(inputs)
    reach = obstacle_reach(params)
    reports: list[ChainReport] = []
    for i, (chain, given, result) in enumerate(
        zip(chains, inputs, run.chains, strict=True)
    ):
        field = chain_field(
            given.field, neighbour_fields(inputs, i, neighbours[i], simplified, reach)
        )
        given_points = given.centreline.points
        changed = snapped_path(simplified[i]) != snapped_path(given_points)
        # An unchanged chain stays on the board as it is.
        after = simplified[i] if changed else given_points
        reports.append(
            ChainReport(
                net=chain.net,
                layer=chain.layer,
                width=chain.width,
                description=given.centreline.description,
                tracks=len(chain.segment_ids),
                before=result.before,
                after=measure(after, field, DEFAULT_SETTINGS.spacing),
                frozen=result.frozen,
                rejected_edges=result.rejected_edges,
                abnormal_bursts=result.abnormal_bursts,
                changed=changed,
            )
        )
    return tuple(reports)


def _require_plot() -> None:
    try:
        importlib.import_module("kibends.plot")
    except ImportError as e:
        raise PlotUnavailableError(f"{PLOT_EXTRA_MESSAGE} ({e})") from None


def _draw(
    path: Path,
    inputs: Sequence[ChainInput],
    run: RunResult,
    simplified: Sequence[FloatArray],
    reports: Sequence[ChainReport],
) -> None:
    """One panel per chain, over its own static field."""
    from kibends.plot import Panel, render

    render(
        path,
        [
            Panel(
                given.field,
                original=given.centreline.points,
                optimised=result.polyline,
                simplified=polyline,
                frozen=result.frozen,
                title=f"{r.net} on {r.layer}: {run.termination.value}, "
                f"length {r.before.length:.3f} → {r.after.length:.3f} mm, "
                f"min clearance margin {r.after.min_clearance:.4f} mm",
            )
            for given, result, polyline, r in zip(
                inputs, run.chains, simplified, reports, strict=True
            )
        ],
    )
    logger.info(f"Wrote {path}")


def _open_board(kicad: kipy.KiCad) -> kipy.board.Board:
    try:
        return kicad.get_board()
    except ApiError as e:
        if e.code == ApiStatusCode.AS_BUSY:
            raise
        raise NoBoardError(f"No board is open in KiCad's PCB editor ({e})") from None


def _prepare_drc(
    kicad: kipy.KiCad, board: kipy.board.Board, explicit_cli: str | None
) -> _DrcSetup:
    kicad_cli = drc.locate_kicad_cli(explicit_cli, kicad)
    logger.info(f"Using {kicad_cli} ({drc.kicad_cli_version(kicad_cli)})")
    project = drc.project_files(board)
    logger.info(
        f"DRC uses the project files on disk: {project.pro}"
        + (f", {project.dru}" if project.dru else " (no .kicad_dru)")
        + "; unsaved rule or netclass edits in KiCad are not seen"
    )
    return _DrcSetup(kicad_cli, project)


def _check_drc(
    gate: _DrcSetup,
    board: kipy.board.Board,
    changed: Sequence[tuple[Chain, FloatArray]],
) -> drc.Introduced:
    """The violations replacing the changed chains would add, in one check."""
    tracks = [
        drc.NewTracks(
            points=tuple(snapped_points(polyline)),
            width=chain.width,
            layer=chain.layer,
            net=chain.net,
        )
        for chain, polyline in changed
    ]
    with tempfile.TemporaryDirectory(prefix="kibends-drc-") as tmp:
        return drc.check(
            gate.kicad_cli,
            gate.project,
            board.get_as_string(),
            [chain.segment_ids for chain, _ in changed],
            tracks,
            Path(tmp),
        )


def new_violations(n: int) -> str:
    return f"{n} new violation{'' if n == 1 else 's'}"


def _drc_refusal(found: drc.Introduced) -> str:
    nets = sorted(
        {n for v in found.violations for n in drc.violation_nets(v)} - {drc.UNNAMED_NET}
    )
    on = f" on {', '.join(nets)}" if nets else ""
    return (
        f"DRC: the result adds {new_violations(len(found.violations))}{on}; "
        "nothing committed"
    )
