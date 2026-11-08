"""One smoothing run, from the KiCad selection to the commit.

The command line and the GUI both call `run_smooth` and present the reports
it hands over; it prints nothing.
"""

import importlib
import tempfile
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import kipy
import kipy.board
import numpy as np
from kipy.errors import ApiError
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from loguru import logger

from kibends import drc
from kibends.bursts import DEFAULT_SETTINGS
from kibends.chain import (
    Chain,
    Point2,
    SameNetCopper,
    Segment,
    split_chains,
    split_locked,
)
from kibends.commit import ViaMove, commit_chains, snapped_path, snapped_points
from kibends.energy import EnergyParams, FloatArray, obstacle_reach
from kibends.neighbours import (
    ChainInput,
    chain_field,
    neighbour_fields,
    same_layer_neighbours,
    with_moved_ends,
)
from kibends.obstacles import (
    HOLE_TO_HOLE,
    ChainCentreline,
    Obstacles,
    build_obstacles,
    build_via_obstacles,
)
from kibends.optimize import (
    FrozenVertex,
    PolylineStats,
    RunResult,
    Termination,
    measure,
    optimize_chains,
)
from kibends.simplify import simplify_chains
from kibends.snapshot import BoardSnapshot, LayerSnapshot, read_selection
from kibends.via_moves import MovingVias
from kibends.vias import MovableVia, ViaInput, carried_copper, may_move, movable_vias

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
class ViaReport:
    """A selected via that may move, and where the run leaves it."""

    id: str
    start: Point2
    end: Point2
    """On KiCad's 1 nm grid, mm."""
    frozen: bool
    """It started inside its clearance and was left in place."""
    rejected_edges: int

    @property
    def moved(self) -> bool:
        return self.end != self.start


@dataclass(frozen=True)
class OptimizerReport:
    chains: tuple[ChainReport, ...]
    """In the order `extract_chains` gives."""
    vias: tuple[ViaReport, ...]
    """The movable vias, in selection order."""
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

    selection = read_selection(board)
    selected = selection.segments
    # Locked tracks stay on the board, so they are obstacles or pins like any
    # unselected track.
    replaced = {s.id for s in selected if not s.locked}
    may = [v for v in selection.vias if may_move(v)]
    candidates = {v.id for v in may}
    kept, locked = split_locked(selected)
    # Pins never change a chain's order or ends, which is all the copper a
    # via carries and `movable_vias` look at.
    carried = (
        carried_copper(may, split_chains(kept, locked, lambda _l, _n: ()))
        if may
        else {}
    )
    # A moving via meets the copper of every layer it spans, so a selection
    # holding one reads them all.
    whole = BoardSnapshot.read(
        board,
        {s.layer for s in selected if not s.locked},
        may,
        carried=carried,
        every_copper_layer=bool(may),
    )
    snapshots = {layer: whole.slice(layer, replaced) for layer in sorted(whole.layers)}
    # A moving via's copper must not pin the chains it carries where they
    # leave it.
    chains = _extract_chains(kept, locked, snapshots, candidates)
    movable = movable_vias(selection.vias, whole.via_contacts, chains)
    moving = {m.via.id for m in movable}
    if moving != candidates:
        chains = _extract_chains(kept, locked, snapshots, moving)
    inputs = _chain_inputs(chains, snapshots, options, moving)

    via_inputs = _via_inputs(movable, snapshots, options)
    run = optimize_chains(
        inputs,
        options.params,
        vias=via_inputs,
        hole_clearance=options.hole_clearance,
    )
    vias = _via_reports(movable, run)
    # Simplification and the statistics see the moved vias where they end up.
    inputs = _with_vias(inputs, via_inputs, vias, options)
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
        _draw(options.plot, inputs, run, simplified, reports, movable, vias)
    on_optimized(OptimizerReport(reports, vias, run.termination, run.iterations))

    def report(
        found: drc.Introduced | None = None,
        *,
        committed: bool = False,
        refusal: str | None = None,
    ) -> RunReport:
        return RunReport(
            reports, vias, run.termination, run.iterations, found, committed, refusal
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
    moves = [
        ViaMove(m.via, r.end) for m, r in zip(movable, vias, strict=True) if r.moved
    ]
    found = None
    if gate is not None:
        found = _check_drc(gate, board, changed, moves)
        if found and not options.dry_run:
            return report(found, refusal=_drc_refusal(found))
    if options.dry_run:
        return report(found)
    commit_chains(board, changed, selected, moves)
    return report(found, committed=True)


def _extract_chains(
    kept: Sequence[Segment],
    locked: Sequence[Segment],
    snapshots: Mapping[str, LayerSnapshot],
    unpinning: Collection[str],
) -> list[Chain]:
    """The chains of the selection split into `kept` and `locked`, pinned by
    the same-net copper of their layers but the vias `unpinning`."""

    def copper(layer: str, net: str) -> list[SameNetCopper]:
        snapshot = snapshots[layer]
        if unpinning:
            pinning = tuple(v for v in snapshot.vias if v.id not in unpinning)
            snapshot = replace(snapshot, vias=pinning)
        return snapshot.same_net_copper(net)

    return split_chains(kept, locked, copper)


def _chain_inputs(
    chains: Sequence[Chain],
    snapshots: Mapping[str, LayerSnapshot],
    options: SmoothOptions,
    moving: Collection[str],
) -> list[ChainInput]:
    """One optimizer input per chain, over its layer's snapshot without the
    `moving` vias, which the optimizer adds where they are; chains of one
    layer, net and width share a field."""
    fields: dict[tuple[str, str, float], Obstacles] = {}
    inputs: list[ChainInput] = []
    for chain in chains:
        snapshot = snapshots[chain.layer]
        key = (chain.layer, chain.net, chain.width)
        if key not in fields:
            still = tuple(v for v in snapshot.vias if v.id not in moving)
            fields[key] = build_obstacles(
                replace(snapshot, vias=still),
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


def _via_inputs(
    movable: Sequence[MovableVia],
    snapshots: Mapping[str, LayerSnapshot],
    options: SmoothOptions,
) -> list[ViaInput]:
    """One optimizer input per movable via, its field over every layer read,
    without the other movable vias, which the optimizer adds where they are."""
    moving = {m.via.id for m in movable}
    layers = list(snapshots.values())
    return [
        ViaInput(
            m.via,
            m.ends,
            build_via_obstacles(
                layers,
                m.via,
                moving=moving,
                edge_clearance=options.edge_clearance,
                hole_clearance=options.hole_clearance,
            ),
        )
        for m in movable
    ]


def _via_reports(
    movable: Sequence[MovableVia], run: RunResult
) -> tuple[ViaReport, ...]:
    reports: list[ViaReport] = []
    for m, result in zip(movable, run.vias, strict=True):
        (end,) = snapped_points(np.array([result.position]))
        reports.append(
            ViaReport(
                m.via.id, m.via.position, end, result.frozen, result.rejected_edges
            )
        )
    return tuple(reports)


def _with_vias(
    inputs: Sequence[ChainInput],
    via_inputs: Sequence[ViaInput],
    vias: Sequence[ViaReport],
    options: SmoothOptions,
) -> list[ChainInput]:
    """`inputs` with the movable vias where they end up added to the fields
    of the chains near them, as the optimizer adds them."""
    if not via_inputs:
        return list(inputs)
    moving = MovingVias(
        inputs, via_inputs, options.params, options.hole_clearance, HOLE_TO_HOLE
    )
    positions = [np.array(r.end, dtype=float) for r in vias]
    out: list[ChainInput] = []
    for i, given in enumerate(inputs):
        part = moving.via_obstacles_for_chain(i, positions)
        out.append(
            given
            if part is None
            else replace(given, field=chain_field(given.field, [part]))
        )
    return out


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
        given_points = given.centreline.points
        own = with_moved_ends(given_points, result.polyline)
        field = chain_field(
            given.field,
            neighbour_fields(inputs, i, neighbours[i], simplified, reach, own),
        )
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
    movable: Sequence[MovableVia],
    vias: Sequence[ViaReport],
) -> None:
    """One panel per chain, over its own static field, with the start and end
    of each via it ends at."""
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
                vias=[
                    (v.start, v.end)
                    for m, v in zip(movable, vias, strict=True)
                    if any(end.chain == i for end in m.ends)
                ],
                title=f"{r.net} on {r.layer}: {run.termination.value}, "
                f"length {r.before.length:.3f} → {r.after.length:.3f} mm, "
                f"min clearance margin {r.after.min_clearance:.4f} mm",
            )
            for i, (given, result, polyline, r) in enumerate(
                zip(inputs, run.chains, simplified, reports, strict=True)
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
    moves: Sequence[ViaMove],
) -> drc.Introduced:
    """The violations replacing the changed chains and moving the vias would
    add, in one check."""
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
            [drc.MovedVia(m.via.id, m.position) for m in moves],
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
