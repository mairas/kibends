"""One smoothing run, from the KiCad selection to the commit.

The command line and the GUI both call `run_smooth` and present the reports
it hands over; it prints nothing.
"""

import importlib
import math
import tempfile
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic

import kipy
import kipy.board
import numpy as np
from kipy.errors import ApiError
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from loguru import logger

from kibends import drc
from kibends.bursts import DEFAULT_SETTINGS, INSIDE_TOLERANCE, OptimizerSettings
from kibends.chain import (
    Chain,
    Point2,
    SameNetCopper,
    Segment,
    split_chains,
    split_locked,
)
from kibends.commit import ViaMove, commit_chains, snapped_path, snapped_points
from kibends.coords import UserFrame
from kibends.energy import (
    EnergyParams,
    FloatArray,
    ObstacleField,
    obstacle_reach,
)
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
    build_pair_obstacles,
    build_via_obstacles,
)
from kibends.optimize import (
    ChainResult,
    FrozenVertex,
    PairResult,
    PolylineStats,
    RunResult,
    Termination,
    measure,
    optimize_chains,
    pinned_at_junctions,
)
from kibends.pair_blocks import Hold, PairInput
from kibends.pair_layout import (
    ChainPair,
    Layout,
    find_pairs,
    lay_out,
    pairs_among,
)
from kibends.pairs import GAP_TOLERANCE, gap_range, gaps_across
from kibends.simplify import simplify_chains
from kibends.snapshot import (
    BoardSnapshot,
    LayerSnapshot,
    read_selection,
    read_user_frame,
)
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
    allow_via_in_pad: bool = False
    """Let a moving via settle inside a pad of its net, which KiCad's DRC
    cannot flag."""
    settings: OptimizerSettings = DEFAULT_SETTINGS
    """Its time limit covers the whole optimize stage, re-runs included."""


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
class PairReport:
    """A differential pair among the selected chains."""

    nets: tuple[str, str]
    layer: str
    gap: float
    """The measured gap its coupled sections keep, their median, mm."""
    gap_after: tuple[float, float]
    """Smallest and largest gap along the result's coupled sections, mm."""
    skew_before: float
    skew_after: float
    """Length difference of the two chains, mm."""
    sections: int
    held: int
    """Coupled sections left as they are: same-net copper touches them
    between their ends, their routing bends too tightly, or the copper
    around them kept them from rounding a corner (`Hold`)."""
    frozen: tuple[FrozenVertex, ...]
    """Of the coupled sections' centrelines."""
    rejected_edges: int


@dataclass(frozen=True)
class OptimizerReport:
    chains: tuple[ChainReport, ...]
    """In the order `extract_chains` gives."""
    vias: tuple[ViaReport, ...]
    """The movable vias, in selection order."""
    pairs: tuple[PairReport, ...]
    """The differential pairs among the chains."""
    termination: Termination
    iterations: int
    """Outer iterations, each giving every problem that has not settled
    one burst, summed over the runs."""
    elapsed: float
    """Wall time of the optimize stage, re-runs included, s."""
    abnormal: tuple[tuple[str, int], ...]
    """The problems, each the chains of a layer or of the layers moving
    vias tie together, whose bursts L-BFGS-B ended abnormally, by
    description, and how many there were, summed over the runs."""
    frame: UserFrame
    """The frame to print positions in. The report's positions stay in the
    board frame; its descriptions are text already formatted in this
    frame."""


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


@dataclass(frozen=True)
class _Read:
    """The selection as the board holds it, and its chains and movable vias."""

    selected: list[Segment]
    chains: list[Chain]
    snapshots: dict[str, LayerSnapshot]
    movable: list[MovableVia]
    frame: UserFrame
    """The frame warnings and descriptions give positions in."""


@dataclass(frozen=True)
class _Optimized:
    whole: list[ChainInput]
    """One per selected chain."""
    pairs: list[ChainPair]
    layout: Layout
    via_inputs: list[ViaInput]
    """With the selected chains' ends."""
    run: RunResult
    elapsed: float
    """Wall time of the optimize stage, s."""


@dataclass(frozen=True)
class _Tracks:
    """A coupled section's two tracks as the run leaves them, b running the
    same way as a."""

    a: FloatArray
    b: FloatArray


@dataclass(frozen=True)
class _WholeResult:
    """A selected chain as the optimizer left it, joined from its parts."""

    polyline: FloatArray
    before: PolylineStats
    frozen: tuple[FrozenVertex, ...]
    rejected_edges: int


@dataclass(frozen=True)
class _Outcome:
    """The simplified result and its reports, per selected chain."""

    final_inputs: list[ChainInput]
    """The chains' inputs with the moved vias where they end up."""
    unsimplified: list[FloatArray]
    simplified: list[FloatArray]
    reports: tuple[ChainReport, ...]
    via_reports: tuple[ViaReport, ...]
    pair_reports: tuple[PairReport, ...]
    pinched: str | None
    """Why a coupled section's tracks may not be committed, if one's may
    not."""


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
    read = _read(board, options.chord_tolerance, read_user_frame(board))
    optimized = _optimize(read, options)
    outcome = _simplify_and_report(read, optimized, options)
    run = optimized.run
    if options.plot is not None:
        _draw(options.plot, read, optimized, outcome)
    on_optimized(
        OptimizerReport(
            outcome.reports,
            outcome.via_reports,
            outcome.pair_reports,
            run.termination,
            run.iterations,
            optimized.elapsed,
            run.abnormal,
            read.frame,
        )
    )
    return _check_and_commit(board, gate, read, optimized, outcome, options)


def _read(board: kipy.board.Board, chord_tolerance: float, frame: UserFrame) -> _Read:
    """Read the selection and the copper it needs, split it into chains and
    judge which selected vias move, giving positions in `frame`."""
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
        carried_copper(
            may,
            split_chains(kept, locked, lambda _l, _n: (), frame=frame, branches=True),
        )
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
    chains = _extract_chains(kept, locked, snapshots, candidates, frame)
    movable = movable_vias(selection.vias, whole.via_contacts, chains, frame=frame)
    if movable:
        # Pairing depends on the chains' shapes, not their pins, so the
        # chains extracted again below pair alike.
        movable = _off_pairs(
            movable, chains, pairs_among(chains, chord_tolerance), frame
        )
    moving = {m.via.id for m in movable}
    if moving != candidates:
        chains = _extract_chains(kept, locked, snapshots, moving, frame)
    return _Read(selected, chains, snapshots, movable, frame)


def _optimize(read: _Read, options: SmoothOptions) -> _Optimized:
    moving = {m.via.id for m in read.movable}
    whole = _chain_inputs(read.chains, read.snapshots, options, moving, read.frame)
    pairs = find_pairs(read.chains, options.chord_tolerance, frame=read.frame)

    def section_field(
        layer: str, a: ChainCentreline, b: ChainCentreline, gap: float
    ) -> ObstacleField:
        snapshot = read.snapshots[layer]
        still = tuple(v for v in snapshot.vias if v.id not in moving)
        return build_pair_obstacles(
            replace(snapshot, vias=still),
            a,
            b,
            gap,
            edge_clearance=options.edge_clearance,
            hole_clearance=options.hole_clearance,
        )

    layout = lay_out(whole, pairs, section_field)
    logger.info(_optimizing(len(read.chains), len(pairs), len(read.movable)))
    via_inputs = _via_inputs(read.movable, read.snapshots, options)
    started = monotonic()
    previous: RunResult | None = None
    iterations = 0
    abnormal: dict[str, int] = {}
    while True:
        run = optimize_chains(
            layout.inputs,
            options.params,
            options.settings,
            vias=_laid_out_vias(layout, via_inputs),
            hole_clearance=options.hole_clearance,
            pairs=layout.sections,
            clock=monotonic,
            started=started,
            start=previous,
        )
        iterations += run.iterations
        for description, count in run.abnormal:
            abnormal[description] = abnormal.get(description, 0) + count
        held = _bent_too_tightly(layout.sections, run.pairs)
        if not held:
            # Only the last run's: a first one was repeated.
            _warn_gave_up(run)
            run = replace(run, iterations=iterations, abnormal=tuple(abnormal.items()))
            elapsed = monotonic() - started
            return _Optimized(whole, pairs, layout, via_inputs, run, elapsed)
        # The other chains were smoothed around the sections' results, so
        # the sections cannot simply keep their own tracks; the run goes on
        # from where this one left them, with these sections held.
        previous = run
        sections = list(layout.sections)
        for k, hold in held.items():
            sections[k] = replace(sections[k], hold=hold)
        layout = replace(layout, sections=sections)


def _warn_gave_up(run: RunResult) -> None:
    for description, streak in run.gave_up:
        logger.warning(
            f"Gave up on {description} after {streak} abnormal optimizer bursts "
            "in a row; they keep their best shapes"
        )


def _bent_too_tightly(
    sections: Sequence[PairInput], results: Sequence[PairResult]
) -> dict[int, Hold]:
    """The sections the run moved but could not keep within the bend-radius
    limit, by index, and why: its own parts beyond the turn, if they came
    too close to each other, else the copper around it, kept it from
    rounding a corner that resampling cut too tightly. Each with a
    warning."""
    held: dict[int, Hold] = {}
    for k, (given, result) in enumerate(zip(sections, results, strict=True)):
        bend = result.bend
        if result.block.held or bend is None or not bend.too_tight:
            continue
        hold = Hold.FOLDING if result.too_close else Hold.CROWDING
        logger.warning(
            f"Kept {given.a.description} and {given.b.description} as they are: "
            f"{hold.value} (a radius of {bend.radius:.3f} mm); "
            "optimizing again with them held"
        )
        held[k] = hold
    return held


def _simplify_and_report(
    read: _Read, optimized: _Optimized, options: SmoothOptions
) -> _Outcome:
    """Simplify the optimizer's chains and the coupled sections' tracks
    together, join them into the selected chains and report on those."""
    run, layout = optimized.run, optimized.layout
    via_reports = _via_reports(read.movable, run)
    tracks = _section_tracks(layout.sections, run.pairs)
    chains, final_tracks = _simplify(optimized, tracks, via_reports, options)
    unsimplified = layout.whole_polylines(
        [c.polyline for c in run.chains], [(t.a, t.b) for t in tracks]
    )
    simplified = layout.whole_polylines(chains, final_tracks)
    # The statistics see the moved vias where they end up, and the branches
    # of a junction leave each other out there as at any touch.
    final_inputs = pinned_at_junctions(
        _with_vias(optimized.whole, optimized.via_inputs, via_reports, options)
    )
    reports = _chain_reports(
        read.chains,
        final_inputs,
        _whole_results(optimized, final_inputs, unsimplified, options.params),
        simplified,
        options.params,
    )
    pair_reports = _pair_reports(
        read, optimized.pairs, layout, run.pairs, final_tracks, reports
    )
    return _Outcome(
        final_inputs,
        unsimplified,
        simplified,
        reports,
        via_reports,
        pair_reports,
        _pinched(layout.sections, [p.block.apart for p in run.pairs], final_tracks),
    )


def _pinched(
    sections: Sequence[PairInput],
    aparts: Sequence[float],
    tracks: Sequence[tuple[FloatArray, FloatArray]],
) -> str | None:
    """Why the first section whose tracks come closer to each other than
    both its nets' clearance and its routing did, where they run side by
    side or more than a turn apart along it (`PairBlock.apart`), may not be
    committed."""
    for given, apart, (a, b) in zip(sections, aparts, tracks, strict=True):
        if np.array_equal(a, given.a.points) and np.array_equal(b, given.b.points):
            continue
        width = given.a.width
        clearance = max(given.a.clearance, given.b.clearance)
        turn = math.pi / 2 * apart
        reach = clearance + width
        routed = gaps_across(given.a.points, given.b.points, width, turn, reach)
        found = gaps_across(a, b, width, turn, reach)
        # The run keeps each section clear of itself; should its tracks still
        # come too close, committing them would short or crowd the pair.
        for smallest, was, where in zip(
            found, routed, ("", " across a turn"), strict=True
        ):
            floor = min(clearance, was)
            if smallest < floor - INSIDE_TOLERANCE:
                return (
                    f"The tracks of the differential pair {given.a.net} and "
                    f"{given.b.net} come within {smallest:.4f} mm of each "
                    f"other{where}, below the {floor:.4f} mm they keep; "
                    "nothing committed"
                )
    return None


def _check_and_commit(
    board: kipy.board.Board,
    gate: _DrcSetup | None,
    read: _Read,
    optimized: _Optimized,
    outcome: _Outcome,
    options: SmoothOptions,
) -> RunReport:
    """Check the changed chains and moved vias against the DRC and, unless
    dry-running or refused, commit them."""
    run = optimized.run

    def report(
        found: drc.Introduced | None = None,
        *,
        committed: bool = False,
        refusal: str | None = None,
    ) -> RunReport:
        return RunReport(
            outcome.reports,
            outcome.via_reports,
            outcome.pair_reports,
            run.termination,
            run.iterations,
            optimized.elapsed,
            run.abnormal,
            read.frame,
            found,
            committed,
            refusal,
        )

    if run.termination is Termination.ABNORMAL:
        return report(refusal=ABNORMAL_MESSAGE)
    if outcome.pinched is not None:
        return report(refusal=outcome.pinched)
    changed = [
        (chain, polyline)
        for chain, polyline, r in zip(
            read.chains, outcome.simplified, outcome.reports, strict=True
        )
        if r.changed
    ]
    if not changed:
        return report()
    moves = [
        ViaMove(m.via, r.end)
        for m, r in zip(read.movable, outcome.via_reports, strict=True)
        if r.moved
    ]
    found = None
    if gate is not None:
        found = _check_drc(gate, board, changed, moves)
        if found and not options.dry_run:
            return report(found, refusal=_drc_refusal(found))
    if options.dry_run:
        return report(found)
    commit_chains(board, changed, read.selected, moves)
    return report(found, committed=True)


def _optimizing(chains: int, pairs: int, vias: int) -> str:
    message = f"Optimizing {chains} chain{'' if chains == 1 else 's'}"
    if pairs:
        message += f", {pairs} differential pair{'' if pairs == 1 else 's'} among them"
    if not vias:
        return message
    return (
        f"{message} and {vias} moving via{'' if vias == 1 else 's'}; "
        "moving vias may take a while"
    )


def _off_pairs(
    movable: Sequence[MovableVia],
    chains: Sequence[Chain],
    pairs: Sequence[ChainPair],
    frame: UserFrame,
) -> list[MovableVia]:
    """`movable` but the vias that join a chain of one of `pairs`, which stay
    fixed, with a warning each."""
    nets = {k: (chains[p.a].net, chains[p.b].net) for p in pairs for k in (p.a, p.b)}
    kept: list[MovableVia] = []
    for m in movable:
        joined = next((nets[e.chain] for e in m.ends if e.chain in nets), None)
        if joined is None:
            kept.append(m)
            continue
        logger.warning(
            f"Via {m.via.id} at {frame.point(m.via.position)} stays fixed: it joins a "
            f"track of the differential pair {joined[0]} and {joined[1]}, and "
            "pairs do not move vias"
        )
    return kept


def _extract_chains(
    kept: Sequence[Segment],
    locked: Sequence[Segment],
    snapshots: Mapping[str, LayerSnapshot],
    unpinning: Collection[str],
    frame: UserFrame,
) -> list[Chain]:
    """The chains of the selection split into `kept` and `locked`, pinned by
    the same-net copper of their layers but the vias `unpinning`."""

    def copper(layer: str, net: str) -> list[SameNetCopper]:
        snapshot = snapshots[layer]
        if unpinning:
            pinning = tuple(v for v in snapshot.vias if v.id not in unpinning)
            snapshot = replace(snapshot, vias=pinning)
        return snapshot.same_net_copper(net)

    return split_chains(kept, locked, copper, frame=frame, branches=True)


def _chain_inputs(
    chains: Sequence[Chain],
    snapshots: Mapping[str, LayerSnapshot],
    options: SmoothOptions,
    moving: Collection[str],
    frame: UserFrame,
) -> list[ChainInput]:
    """One optimizer input per chain, over its layer's snapshot without the
    `moving` vias, which the optimizer adds where they are; chains of one
    layer, net and width share a field. Descriptions give positions in
    `frame`."""
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
        centreline = ChainCentreline(
            f"{chain.net} on {chain.layer} from {frame.point(chain.points[0])}",
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
                allow_via_in_pad=options.allow_via_in_pad,
            ),
        )
        for m in movable
    ]


def _laid_out_vias(layout: Layout, via_inputs: Sequence[ViaInput]) -> list[ViaInput]:
    """`via_inputs` with their ends among the optimizer's chains."""
    return [replace(v, ends=layout.ends(v.ends)) for v in via_inputs]


def _section_tracks(
    sections: Sequence[PairInput], results: Sequence[PairResult]
) -> list[_Tracks]:
    """Each coupled section's tracks: its own where it is held or the run did
    nothing, else as the run offset them."""
    tracks: list[_Tracks] = []
    for given, result in zip(sections, results, strict=True):
        bend = result.bend
        if result.block.hold is Hold.ROUTING and bend is not None:
            logger.warning(
                f"Kept {given.a.description} and {given.b.description} as they "
                f"are: {Hold.ROUTING.value} (a radius of {bend.radius:.3f} mm)"
            )
        if result.block.held or bend is None:
            tracks.append(_Tracks(given.a.points, given.b.points))
        else:
            tracks.append(_Tracks(result.a, result.b))
    return tracks


def _simplify(
    optimized: _Optimized,
    tracks: Sequence[_Tracks],
    via_reports: Sequence[ViaReport],
    options: SmoothOptions,
) -> tuple[list[FloatArray], list[tuple[FloatArray, FloatArray]]]:
    """The optimizer's chains and each section's centreline simplified
    together, each against the others and the moved vias where they end
    up, and each section's two tracks offset from its simplified centreline
    as the run offset them; `tracks` where the run did nothing."""
    run, layout = optimized.run, optimized.layout
    count = len(run.chains)
    if run.termination is Termination.NOTHING_TO_DO:
        return [c.polyline for c in run.chains], [(t.a, t.b) for t in tracks]
    blocks = {count + k: result.block for k, result in enumerate(run.pairs)}
    inputs = [
        *layout.inputs,
        *(block.facing for block in blocks.values()),
    ]
    results = [
        *run.chains,
        *(ChainResult(r.centreline, r.fixed, 0, (), r.before) for r in run.pairs),
    ]
    simplified = simplify_chains(
        pinned_at_junctions(
            _with_vias(
                inputs,
                _laid_out_vias(layout, optimized.via_inputs),
                via_reports,
                options,
            )
        ),
        replace(run, chains=tuple(results)),
        options.params,
        tolerance=options.chord_tolerance,
        sections=blocks,
    )
    return simplified[:count], [
        block.offset_kept(results[i].polyline, simplified[i])
        for i, block in blocks.items()
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
    moving = MovingVias(inputs, via_inputs, options.hole_clearance, HOLE_TO_HOLE)
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


def _whole_results(
    optimized: _Optimized,
    inputs: Sequence[ChainInput],
    unsimplified: Sequence[FloatArray],
    params: EnergyParams,
) -> list[_WholeResult]:
    """Each selected chain as the optimizer left it: a chain of its own as
    its result, a chain of a pair from its pieces and its sections' tracks,
    measured before the run among the other chains as they started."""
    run, layout = optimized.run, optimized.layout
    originals = [given.centreline.points for given in inputs]
    neighbours = same_layer_neighbours(inputs)
    reach = obstacle_reach(params)
    results: list[_WholeResult] = []
    for k, slot in enumerate(layout.slots):
        if slot is not None:
            r = run.chains[slot]
            results.append(
                _WholeResult(r.polyline, r.before, r.frozen, r.rejected_edges)
            )
            continue
        pieces = [
            run.chains[s.piece] for s in layout.stretches[k] if s.piece is not None
        ]
        field = chain_field(
            inputs[k].field,
            neighbour_fields(inputs, k, neighbours[k], originals, reach, originals[k]),
        )
        results.append(
            _WholeResult(
                unsimplified[k],
                measure(originals[k], field, DEFAULT_SETTINGS.spacing),
                tuple(v for p in pieces for v in p.frozen),
                sum(p.rejected_edges for p in pieces),
            )
        )
    return results


def _pair_reports(
    read: _Read,
    pairs: Sequence[ChainPair],
    layout: Layout,
    results: Sequence[PairResult],
    tracks: Sequence[tuple[FloatArray, FloatArray]],
    reports: Sequence[ChainReport],
) -> tuple[PairReport, ...]:
    """A report per pair, warning where its gap is not its netclass's."""
    pair_reports: list[PairReport] = []
    for pair, sections in zip(pairs, layout.pair_sections, strict=True):
        a, b = read.chains[pair.a], read.chains[pair.b]
        _check_netclass_gap(read.snapshots[a.layer], a, b, pair.pair.gap)
        gaps = [gap_range(*tracks[k], a.width) for k in sections]
        ours = [results[k] for k in sections]
        before = (reports[pair.a].before, reports[pair.b].before)
        after = (reports[pair.a].after, reports[pair.b].after)
        pair_reports.append(
            PairReport(
                nets=(a.net, b.net),
                layer=a.layer,
                gap=pair.pair.gap,
                gap_after=(min(g[0] for g in gaps), max(g[1] for g in gaps)),
                skew_before=abs(before[0].length - before[1].length),
                skew_after=abs(after[0].length - after[1].length),
                sections=len(sections),
                held=sum(r.block.held for r in ours),
                frozen=tuple(v for r in ours for v in r.frozen),
                rejected_edges=sum(r.rejected_edges for r in ours),
            )
        )
    return tuple(pair_reports)


def _check_netclass_gap(
    snapshot: LayerSnapshot, a: Chain, b: Chain, gap: float
) -> None:
    expected = snapshot.diff_pair_gap(a.net) or snapshot.diff_pair_gap(b.net)
    if expected is not None and abs(gap - expected) > GAP_TOLERANCE * expected:
        logger.warning(
            f"Differential pair {a.net} and {b.net} keeps its measured gap of "
            f"{gap:.3f} mm, not its netclass's {expected:.3f} mm"
        )


def _chain_reports(
    chains: Sequence[Chain],
    inputs: Sequence[ChainInput],
    results: Sequence[_WholeResult],
    simplified: Sequence[FloatArray],
    params: EnergyParams,
) -> tuple[ChainReport, ...]:
    neighbours = same_layer_neighbours(inputs)
    reach = obstacle_reach(params)
    reports: list[ChainReport] = []
    for i, (chain, given, result) in enumerate(
        zip(chains, inputs, results, strict=True)
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
                changed=changed,
            )
        )
    return tuple(reports)


def _require_plot() -> None:
    try:
        importlib.import_module("kibends.plot")
    except ImportError as e:
        raise PlotUnavailableError(f"{PLOT_EXTRA_MESSAGE} ({e})") from None


def _draw(path: Path, read: _Read, optimized: _Optimized, outcome: _Outcome) -> None:
    """One panel per chain, over its own static field, with the start and end
    of each via it ends at."""
    from kibends.plot import Panel, render

    run = optimized.run
    partner = {
        k: read.chains[other].net
        for p in optimized.pairs
        for k, other in ((p.a, p.b), (p.b, p.a))
    }
    render(
        path,
        [
            Panel(
                given.field,
                original=given.centreline.points,
                optimised=unsimplified,
                simplified=polyline,
                frozen=r.frozen,
                vias=[
                    (v.start, v.end)
                    for m, v in zip(read.movable, outcome.via_reports, strict=True)
                    if any(end.chain == i for end in m.ends)
                ],
                title=f"{r.net} on {r.layer}"
                + (f", pair with {partner[i]}" if i in partner else "")
                + f": {run.termination.value}, "
                f"length {r.before.length:.3f} → {r.after.length:.3f} mm, "
                f"min clearance margin {r.after.min_clearance:.4f} mm",
            )
            for i, (given, unsimplified, polyline, r) in enumerate(
                zip(
                    outcome.final_inputs,
                    outcome.unsimplified,
                    outcome.simplified,
                    outcome.reports,
                    strict=True,
                )
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
