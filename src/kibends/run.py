"""One smoothing run, from the KiCad selection to the commit.

The command line and the GUI both call `run_smooth` and present the reports
it hands over; it prints nothing.

`run_smooth` reads the board once, extracts the chains, cut at every
selected via that may move, and, if any of them stays fixed, extracts them
again with the chains through that via whole and its copper pinning. A diff
pair's vias never move (`vias.may_move`), so no moving via carries a paired
chain. Selected tracks that form no chain, stray zero-length tracks, stay on
the board as unselected copper, and the chains are extracted again with them
pinning. When a selected via may move and attraction is on, it finds the
ground nets, logs them and classes every via by them; with attraction off
nothing is classed, and the optimizer's energy is the one without ground
vias. It then builds the fields, optimizes, simplifies and commits. A
coupled section the run would leave bent too tightly is held as drawn, and
the optimizer goes on in the same run (`optimize.optimize_chains`). A chain
whose result equals its selected tracks, and that shares no split track with
a changed chain, is left out of the commit; nothing is committed when every
chain is unchanged and no via moved. A selected arc thus stays an arc only
in a chain left exactly as drawn; simplification merges its pieces and the
optimizer shortens it, so it is nearly always replaced.

The run checks no clearance of its own after optimizing: what the
optimizer and simplification leave is committed, and the user's KiCad DRC
reports any violation. It refuses only an abnormal optimizer stop.

Before the commit, which may raise, `run_smooth` hands an `OptimizerReport`
(per-chain, per-pair and per-via figures, termination, abnormal bursts,
the optimize stage's wall time) to its `on_optimized` callback; it returns
a `RunReport`, which adds whether it committed and any refusal. Positions
in reports stay in the board frame; the `coords.UserFrame` they carry is
what messages print positions in.
"""

import importlib
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import NamedTuple

import kipy
import kipy.board
import numpy as np
from kipy.errors import ApiError
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from loguru import logger

from kibends.bursts import DEFAULT_SETTINGS, OptimizerSettings
from kibends.chain import (
    Chain,
    Point2,
    SameNetCopper,
    Segment,
    shared_track_groups,
    split_chains,
    split_locked,
)
from kibends.commit import ViaMove, commit_chains, snapped_path, snapped_points
from kibends.contacts import no_contacts
from kibends.coords import UserFrame
from kibends.energy import (
    EnergyParams,
    FloatArray,
    ObstacleField,
    arc_lengths,
)
from kibends.neighbours import (
    ChainInput,
    chain_field,
)
from kibends.obstacles import (
    HOLE_TO_HOLE,
    ChainCentreline,
    Obstacles,
    build_obstacles,
    build_pair_obstacles,
    own_net_stops,
    split_via_obstacles,
)
from kibends.optimize import optimize_chains
from kibends.pair_blocks import Hold, PairInput
from kibends.pair_layout import (
    ChainPair,
    ChainPart,
    Layout,
    cut_parts,
    find_pairs,
    lay_out,
)
from kibends.polylines import measure
from kibends.results import (
    ChainResult,
    PairResult,
    PolylineStats,
    RunResult,
    Termination,
)
from kibends.simplify import simplify_chains
from kibends.snapshot import (
    FALLBACK_GROUND,
    Arc,
    BoardSnapshot,
    GroundSource,
    LayerSnapshot,
    Via,
    read_selection,
    read_user_frame,
)
from kibends.via_moves import MovingVias
from kibends.vias import (
    MovableVia,
    StaticVia,
    ViaInput,
    carried_copper,
    carried_ends,
    chain_cuts,
    may_move,
    movable_vias,
    pairs_with,
)

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
    "Try other --repulsion or --range values"
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
    allow_via_in_pad: bool = False
    """Let a moving via settle inside a pad of its net, which KiCad's DRC
    cannot flag."""
    settings: OptimizerSettings = DEFAULT_SETTINGS
    """Its time limit covers the whole optimize stage, holds included.
    Its settle threshold should be half `chord_tolerance` at most, as
    `cli.smooth_options` sets it; the default suits the default tolerance."""
    ground_nets: tuple[str, ...] = ()
    """The ground nets, overriding their detection when not empty
    (`BoardSnapshot.ground_nets`)."""


@dataclass(frozen=True)
class ChainReport:
    net: str
    layer: str
    width: float
    """mm."""
    description: str
    """The chain as the optimizer's and simplification's log lines name it
    (`ChainCentreline.description`)."""
    tracks: int
    """Selected tracks it replaces, counting a track split at a junction in
    each chain that has a part of it."""
    before: PolylineStats
    after: PolylineStats
    """Of the simplified result, or of the selection when unchanged."""
    rejected_edges: int
    changed: bool
    """False when the result equals the selected tracks and the chain shares
    no track with a changed chain, since a track split at a junction is
    replaced whole: then it is neither checked nor committed."""


@dataclass(frozen=True)
class ViaReport:
    """A selected via that may move, and where the run leaves it."""

    id: str
    start: Point2
    end: Point2
    """On KiCad's 1 nm grid, mm."""
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
    skew_before: float
    skew_after: float
    """Length difference of its two tracks, mm."""
    sections: int
    held: int
    """Coupled sections left as they are: their routing, or what same-net
    copper touching them keeps of it, bends too tightly, they are too short
    to move, or the copper around them kept them from rounding a corner
    (`Hold`)."""
    rejected_edges: int


@dataclass(frozen=True)
class OptimizerReport:
    chains: tuple[ChainReport, ...]
    """In the order `chain.split_chains` gives."""
    vias: tuple[ViaReport, ...]
    """The movable vias, in selection order."""
    pairs: tuple[PairReport, ...]
    """The differential pairs among the chains."""
    termination: Termination
    """How the run ended, `Termination.GAVE_UP` also where it converged
    after giving up on a problem that a hold then built again."""
    iterations: int
    """Outer iterations, each giving every problem that has not settled
    one burst, holds included."""
    elapsed: float
    """Wall time of the optimize stage, holds included, s."""
    abnormal: tuple[tuple[str, int], ...]
    """The problems, each the chains of a layer or of the layers moving
    vias tie together, whose bursts L-BFGS-B ended abnormally, by
    description, and how many there were, before and after any hold."""
    frame: UserFrame
    """The frame to print positions in. The report's positions stay in the
    board frame; its descriptions are text already formatted in this
    frame."""


@dataclass(frozen=True)
class RunReport(OptimizerReport):
    committed: bool
    refusal: str | None
    """Why nothing was committed, when the optimizer stopped abnormally
    (`ABNORMAL_MESSAGE`). The run checks no clearance after optimizing;
    the user's KiCad DRC does."""

    @property
    def changes(self) -> bool:
        """Whether some chain changed or some via moved: whether there is
        anything to commit."""
        return any(c.changed for c in self.chains) or any(v.moved for v in self.vias)


@dataclass(frozen=True)
class _Read:
    """The selection as the board holds it, and its chains and movable vias."""

    selected: list[Segment]
    chains: list[Chain]
    snapshots: dict[str, LayerSnapshot]
    movable: list[MovableVia]
    frame: UserFrame
    """The frame warnings and descriptions give positions in."""
    ground_nets: frozenset[str]
    """Empty when no selected via may move or vias are not paired, as they
    are then not looked for."""
    arcs: tuple[Arc, ...]
    """The selected arcs, whose segments are among `selected`, as read."""


@dataclass(frozen=True)
class _Optimized:
    whole: list[ChainInput]
    """One per selected chain."""
    parts: list[ChainPart]
    """The selected chains' parts, which `pairs` index."""
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
    rejected_edges: int


@dataclass(frozen=True)
class _Outcome:
    """The simplified result and its reports, per selected chain."""

    unsimplified: list[FloatArray]
    simplified: list[FloatArray]
    reports: tuple[ChainReport, ...]
    via_reports: tuple[ViaReport, ...]
    pair_reports: tuple[PairReport, ...]


def run_smooth(
    options: SmoothOptions,
    on_optimized: Callable[[OptimizerReport], None] = lambda _report: None,
) -> RunReport:
    """Smooth the chains selected in KiCad and, unless dry-running, commit them.

    `on_optimized` gets the optimizer's report before the commit, which may
    raise. Raises on failures that leave nothing to report (no board, a bad
    selection, a refused commit); an abnormal optimizer stop comes back as
    the report's `refusal`, with nothing committed.
    """
    if options.plot is not None:
        _require_plot()
    kicad = kipy.KiCad(timeout_ms=KICAD_TIMEOUT_MS)
    board = _open_board(kicad)
    read = _read(
        board,
        read_user_frame(board),
        ground_nets=options.ground_nets,
        paired=options.params.attraction > 0.0,
    )
    optimized = _optimize(read, options)
    outcome = _simplify_and_report(read, optimized, options)
    run = optimized.run
    if options.plot is not None:
        if read.chains:
            _draw(options.plot, read, optimized, outcome, options)
        else:
            logger.info("No plot: the selection has no chains to draw")
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
    return _commit(board, read, optimized, outcome, options)


def _read(
    board: kipy.board.Board,
    frame: UserFrame,
    *,
    ground_nets: Collection[str] = (),
    paired: bool = True,
) -> _Read:
    """Read the selection and the copper it needs, split it into chains and
    judge which selected vias move, giving positions in `frame`. When a
    selected via may move and vias are `paired`, find the ground nets,
    `ground_nets` if given, and name them."""
    selection = read_selection(board)
    selected = selection.segments
    # Locked tracks stay on the board, so they are obstacles or pins like any
    # unselected track.
    replaced = {s.id for s in selected if not s.locked}
    may = [v for v in selection.vias if may_move(v, selection.nets)]
    kept, locked = split_locked(selected)
    # Selected vias without tracks that can be modified leave no chains: only
    # the stitching vias among them move.
    chained = bool(kept) or not selection.vias
    # Pins never change a chain's order or ends, which is all the copper a
    # via carries and `movable_vias` look at.
    carried = carried_copper(
        may,
        split_chains(
            kept,
            locked,
            lambda _l, _n: (),
            frame=frame,
            left_out=lambda _line: None,
            cuts=chain_cuts(may),
        )
        if may and chained
        else [],
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
    snapshots = _slices(whole, replaced)
    ground = _ground_nets(whole, ground_nets) if may and paired else frozenset[str]()
    # A moving via's copper must not hold the chains it carries where they
    # leave it, and the chains through it end there. Only the last extraction
    # before the copper is final says what it leaves out.
    notes: list[str] = []
    chains = (
        _extract_chains(kept, locked, snapshots, may, frame, notes.append)
        if chained
        else []
    )
    movable = movable_vias(
        selection.vias,
        whole.via_contacts,
        chains,
        frame=frame,
        zones=whole.via_zones,
        nets=selection.nets,
    )
    moving = [m.via for m in movable]
    if chained and {v.id for v in moving} != {v.id for v in may}:
        notes = []
        chains = _extract_chains(kept, locked, snapshots, moving, frame, notes.append)
    chains, snapshots = _leave_out(
        _Extracted(kept, locked, snapshots, chains, notes),
        whole,
        replaced,
        moving,
        frame,
    )
    # Extracting again may order the chains differently.
    movable = [replace(m, ends=carried_ends(m.via, chains)) for m in movable]
    return _Read(selected, chains, snapshots, movable, frame, ground, selection.arcs)


def _slices(
    whole: BoardSnapshot, replaced: Collection[str]
) -> dict[str, LayerSnapshot]:
    """Each layer of `whole` without the tracks `replaced`."""
    return {layer: whole.slice(layer, replaced) for layer in sorted(whole.layers)}


class _Extracted(NamedTuple):
    """An extraction: what it took and gave, and its lines on what it left
    out."""

    kept: Sequence[Segment]
    locked: Sequence[Segment]
    snapshots: dict[str, LayerSnapshot]
    chains: list[Chain]
    notes: Sequence[str]


def _leave_out(
    extracted: _Extracted,
    whole: BoardSnapshot,
    replaced: Collection[str],
    moving: Sequence[Via],
    frame: UserFrame,
) -> tuple[list[Chain], dict[str, LayerSnapshot]]:
    """The final chains and the layers they are extracted over. The selected
    tracks no chain of `extracted` took are taken out of the `replaced`
    ones and stay on the board as unselected copper, which pins and blocks
    the chains as any other does, and the chains are extracted again; the
    extraction's notes on them are logged."""
    kept, locked, snapshots, chains, notes = extracted
    left_out = {s.id for s in kept} - {i for c in chains for i in c.segment_ids}
    if not left_out:
        return chains, snapshots
    for note in notes:
        logger.debug(note)
    still = [s for s in kept if s.id not in left_out]
    snapshots = _slices(whole, {*replaced} - left_out)
    if not still:
        return [], snapshots
    return (
        _extract_chains(still, locked, snapshots, moving, frame, logger.debug),
        snapshots,
    )


def _ground_nets(whole: BoardSnapshot, given: Collection[str]) -> frozenset[str]:
    """The ground nets `whole.ground_nets` finds, logged with where they come
    from, warning of each of `given` without copper."""
    for net in sorted(set(given) - whole.copper_nets):
        logger.warning(
            f"Ground net {net} has no zones, vias or pads on the board; "
            "check --ground-nets"
        )
    found = whole.ground_nets(given)
    names = " and ".join(sorted(found.nets))
    noun = "Ground net" if len(found.nets) == 1 else "Ground nets"
    match found.source:
        case GroundSource.GIVEN:
            logger.info(f"{noun} {names}, as given")
        case GroundSource.INNER_ZONES:
            logger.info(f"{noun} {names}: zones on at least two inner layers")
        case GroundSource.FALLBACK:
            logger.info(
                f"{noun} {names}: no net has zones on at least two inner layers"
            )
        case GroundSource.NONE:
            logger.info(
                "No ground nets: no net has zones on at least two inner layers and "
                f"{FALLBACK_GROUND} has no zones, vias or pads"
            )
    return found.nets


def _optimize(read: _Read, options: SmoothOptions) -> _Optimized:
    """Optimize the chains of `read`."""
    moving = {m.via.id for m in read.movable}
    whole = _chain_inputs(read.chains, read.snapshots, options, moving, read.frame)
    # A diff pair's tracks keep their contacts where they are.
    parts = cut_parts(
        [chain.with_slides_pinned() for chain in read.chains], options.chord_tolerance
    )
    pairs = find_pairs([part.chain for part in parts], options.chord_tolerance)

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

    layout = lay_out(whole, parts, pairs, section_field)
    logger.info(
        _optimizing(len(read.chains), len(layout.diff_pairs), len(read.movable))
    )
    # One classing of the vias standing still both splits the moving vias'
    # fields and gives the pairwise term its vias.
    static = _static_vias(read.snapshots, moving, read.ground_nets)
    via_inputs = _via_inputs(
        read.movable, read.snapshots, options, read.ground_nets, static
    )
    started = monotonic()
    run = optimize_chains(
        layout.inputs,
        options.params,
        options.settings,
        vias=_laid_out_vias(layout, via_inputs),
        hole_clearance=options.hole_clearance,
        pairs=layout.sections,
        clock=monotonic,
        started=started,
        static_vias=tuple(static.values()),
        to_hold=_bent_too_tightly,
    )
    _log_gave_up(run.gave_up)
    elapsed = monotonic() - started
    return _Optimized(whole, parts, pairs, layout, via_inputs, run, elapsed)


def _log_gave_up(gave_up: Sequence[tuple[str, int]]) -> None:
    for description, streak in gave_up:
        logger.debug(
            f"Gave up on {description} after {streak} abnormal optimizer bursts "
            "in a row; they keep their best shapes"
        )


def _bent_too_tightly(results: Sequence[PairResult]) -> list[int]:
    """The sections the run moved but could not keep within the bend-radius
    limit, by index: the copper around them kept them from rounding a
    corner that resampling cut too tightly. Each logged at debug."""
    held: list[int] = []
    for k, result in enumerate(results):
        bend = result.bend
        if result.block.held or bend is None or not bend.too_tight:
            continue
        given = result.block.given
        logger.debug(
            f"Kept {given.a.description} and {given.b.description} as they are: "
            f"{Hold.CROWDING.value} (a radius of {bend.radius:.3f} mm); "
            "holding them and going on"
        )
        held.append(k)
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
    reports = _chain_reports(
        read.chains,
        optimized.whole,
        _whole_results(optimized, unsimplified),
        simplified,
    )
    pair_reports = _pair_reports(optimized, chains, final_tracks)
    return _Outcome(
        unsimplified,
        simplified,
        reports,
        via_reports,
        pair_reports,
    )


def _commit(
    board: kipy.board.Board,
    read: _Read,
    optimized: _Optimized,
    outcome: _Outcome,
    options: SmoothOptions,
) -> RunReport:
    """Commit the changed chains and moved vias unless dry-running or
    refused."""
    run = optimized.run

    def report(
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
            committed,
            refusal,
        )

    if run.termination is Termination.ABNORMAL:
        return report(refusal=ABNORMAL_MESSAGE)
    if options.dry_run or not report().changes:
        return report()
    changed = [
        (chain, polyline)
        for chain, polyline, r in zip(
            read.chains, outcome.simplified, outcome.reports, strict=True
        )
        if r.changed
    ]
    moves = [
        ViaMove(m.via, r.end)
        for m, r in zip(read.movable, outcome.via_reports, strict=True)
        if r.moved
    ]
    commit_chains(board, changed, read.selected, read.arcs, moves)
    return report(committed=True)


def _optimizing(chains: int, pairs: int, vias: int) -> str:
    message = f"Optimizing {chains} chain{'' if chains == 1 else 's'}"
    if pairs:
        message += f", {pairs} differential pair{'' if pairs == 1 else 's'} among them"
    if not vias:
        return message
    return f"{message} and {vias} moving via{'' if vias == 1 else 's'}"


def _extract_chains(
    kept: Sequence[Segment],
    locked: Sequence[Segment],
    snapshots: Mapping[str, LayerSnapshot],
    moving: Sequence[Via],
    frame: UserFrame,
    left_out: Callable[[str], None],
) -> list[Chain]:
    """The chains of the selection split into `kept` and `locked`, held by
    the same-net copper of their layers but the vias `moving`: pinned where
    it touches them, or, for pads and vias, kept by a vertex inside. The
    vias `moving` cut the chains through their centres instead. Each piece
    that forms no chain is said to `left_out`."""
    ids = {v.id for v in moving}

    def copper(layer: str, net: str) -> list[SameNetCopper]:
        snapshot = snapshots[layer]
        if ids:
            staying = tuple(v for v in snapshot.vias if v.id not in ids)
            snapshot = replace(snapshot, vias=staying)
        return snapshot.same_net_copper(net)

    return split_chains(
        kept,
        locked,
        copper,
        frame=frame,
        left_out=left_out,
        cuts=chain_cuts(moving),
    )


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
            fields[key] = _static_field(snapshot, chain, options, moving)
        centreline = ChainCentreline(
            f"{chain.net} on {chain.layer} from {frame.point(chain.points[0])}",
            chain.net,
            chain.width,
            snapshot.clearance(chain.net),
            np.array(chain.points),
        )
        inputs.append(
            ChainInput(
                centreline,
                chain.layer,
                np.array(chain.pinned),
                fields[key],
                tuple((i, Obstacles((), region)) for i, region in chain.slides),
                held=chain.held,
            )
        )
    return inputs


def _static_field(
    snapshot: LayerSnapshot,
    chain: Chain,
    options: SmoothOptions,
    moving: Collection[str],
) -> Obstacles:
    """The obstacles of `chain`'s net and width on `snapshot`'s layer, but
    the `moving` vias."""
    still = tuple(v for v in snapshot.vias if v.id not in moving)
    return build_obstacles(
        replace(snapshot, vias=still),
        chain.net,
        chain.width,
        edge_clearance=options.edge_clearance,
        hole_clearance=options.hole_clearance,
    )


def _via_inputs(
    movable: Sequence[MovableVia],
    snapshots: Mapping[str, LayerSnapshot],
    options: SmoothOptions,
    ground: Collection[str],
    static: Mapping[str, StaticVia],
) -> list[ViaInput]:
    """One optimizer input per movable via, its field over every layer read,
    without the other movable vias, which the optimizer adds where they are,
    and without the vias of `static`, by id, that it pairs with among the
    `ground` nets (`vias.pairs_with`), which are its `static_vias`."""
    moving = {m.via.id for m in movable}
    layers = list(snapshots.values())
    inputs: list[ViaInput] = []
    for m in movable:
        grounded = m.via.net in ground

        def split(other: Via, via: Via = m.via, grounded: bool = grounded) -> bool:
            standing = static.get(other.id)
            return standing is not None and pairs_with(via, grounded, standing)

        field, static_vias = split_via_obstacles(
            layers,
            m.via,
            split,
            moving=moving,
            edge_clearance=options.edge_clearance,
            hole_clearance=options.hole_clearance,
            zones=m.zones,
        )
        inputs.append(
            ViaInput(
                m.via,
                m.ends,
                field,
                layers[0].clearance(m.via.net),
                m.zones,
                own_net_stops(
                    layers,
                    m.via,
                    moving=moving,
                    hole_clearance=options.hole_clearance,
                    allow_via_in_pad=options.allow_via_in_pad,
                ),
                ground=grounded,
                static_vias=static_vias,
            )
        )
    return inputs


def _static_vias(
    snapshots: Mapping[str, LayerSnapshot],
    moving: Collection[str],
    ground: Collection[str],
) -> dict[str, StaticVia]:
    """By id, the board's vias the snapshots hold but those `moving`, with
    their classes; none without `ground` nets."""
    if not ground:
        return {}
    vias = {
        v.id: (v, s.clearance(v.net))
        for s in snapshots.values()
        for v in s.vias
        if v.id not in moving
    }
    return {i: StaticVia(v, v.net in ground, c) for i, (v, c) in vias.items()}


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
            logger.debug(
                f"Kept {given.a.description} and {given.b.description} as they "
                f"are: {Hold.ROUTING.value} (a radius of {bend.radius:.3f} mm)"
            )
        tracks.append(_Tracks(*result.kept()))
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
        *(
            ChainResult(r.centreline, r.fixed, 0, r.before, no_contacts(r.centreline))
            for r in run.pairs
        ),
    ]
    simplified = simplify_chains(
        _with_vias(
            inputs,
            _laid_out_vias(layout, optimized.via_inputs),
            via_reports,
            options,
        ),
        replace(run, chains=tuple(results)),
        tolerance=options.chord_tolerance,
        sections=blocks,
    )
    return simplified[:count], [
        block.offset(simplified[i]) for i, block in blocks.items()
    ]


def _via_reports(
    movable: Sequence[MovableVia], run: RunResult
) -> tuple[ViaReport, ...]:
    reports: list[ViaReport] = []
    for m, result in zip(movable, run.vias, strict=True):
        (end,) = snapped_points(np.array([result.position]))
        reports.append(ViaReport(m.via.id, m.via.position, end, result.rejected_edges))
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
            else replace(given, field=chain_field(given.field, part))
        )
    return out


def _whole_results(
    optimized: _Optimized, unsimplified: Sequence[FloatArray]
) -> list[_WholeResult]:
    """Each selected chain as the optimizer left it: a chain of its own as
    its result, a chain of a pair from its pieces and its sections' tracks,
    measured before the run."""
    run, layout = optimized.run, optimized.layout
    results: list[_WholeResult] = []
    for k, slot in enumerate(layout.slots):
        if slot is not None:
            r = run.chains[slot]
            results.append(_WholeResult(r.polyline, r.before, r.rejected_edges))
            continue
        stretches = layout.chain_stretches(k)
        pieces = [run.chains[s.piece] for s in stretches if s.piece is not None]
        results.append(
            _WholeResult(
                unsimplified[k],
                measure(optimized.whole[k].centreline.points, DEFAULT_SETTINGS.spacing),
                sum(p.rejected_edges for p in pieces),
            )
        )
    return results


def _pair_reports(
    optimized: _Optimized,
    chains: Sequence[FloatArray],
    tracks: Sequence[tuple[FloatArray, FloatArray]],
) -> tuple[PairReport, ...]:
    """A report per pair of parts, from `chains` and `tracks` as simplified."""
    layout, results = optimized.layout, optimized.run.pairs
    pair_reports: list[PairReport] = []
    for pair, sections in zip(optimized.pairs, layout.pair_sections, strict=True):
        a, b = (optimized.parts[k].chain for k in (pair.a, pair.b))
        ours = [results[k] for k in sections]
        before = (_length(np.array(a.points)), _length(np.array(b.points)))
        after = (
            _length(layout.part_polyline(k, chains, tracks)) for k in (pair.a, pair.b)
        )
        pair_reports.append(
            PairReport(
                nets=(a.net, b.net),
                layer=a.layer,
                gap=pair.pair.gap,
                skew_before=abs(before[0] - before[1]),
                skew_after=abs(next(after) - next(after)),
                sections=len(sections),
                held=sum(r.block.held for r in ours),
                rejected_edges=sum(r.rejected_edges for r in ours),
            )
        )
    return tuple(pair_reports)


def _length(points: FloatArray) -> float:
    return float(arc_lengths(points)[-1])


def _chain_reports(
    chains: Sequence[Chain],
    inputs: Sequence[ChainInput],
    results: Sequence[_WholeResult],
    simplified: Sequence[FloatArray],
) -> tuple[ChainReport, ...]:
    reports: list[ChainReport] = []
    changes = _changed_with_shared_tracks(
        chains,
        [
            snapped_path(s) != snapped_path(given.centreline.points)
            for s, given in zip(simplified, inputs, strict=True)
        ],
    )
    for chain, given, result, polyline, changed in zip(
        chains, inputs, results, simplified, changes, strict=True
    ):
        # An unchanged chain stays on the board as it is.
        after = polyline if changed else given.centreline.points
        reports.append(
            ChainReport(
                net=chain.net,
                layer=chain.layer,
                width=chain.width,
                description=given.centreline.description,
                tracks=len(chain.segment_ids),
                before=result.before,
                after=measure(after, DEFAULT_SETTINGS.spacing),
                rejected_edges=result.rejected_edges,
                changed=changed,
            )
        )
    return tuple(reports)


def _changed_with_shared_tracks(
    chains: Sequence[Chain], changed: Sequence[bool]
) -> list[bool]:
    """`changed`, spread to every chain that shares a track with a changed
    one, since a track split at a junction is replaced whole."""
    groups = shared_track_groups([c.segment_ids for c in chains])
    replaced = {g for g, ch in zip(groups, changed, strict=True) if ch}
    return [g in replaced for g in groups]


def _require_plot() -> None:
    try:
        importlib.import_module("kibends.plot")
    except ImportError as e:
        raise PlotUnavailableError(f"{PLOT_EXTRA_MESSAGE} ({e})") from None


def _draw(
    path: Path,
    read: _Read,
    optimized: _Optimized,
    outcome: _Outcome,
    options: SmoothOptions,
) -> None:
    """One panel per chain, over its own static field with the moved vias
    where they end up, with the start and end of each via it ends at."""
    from kibends.plot import Panel, render

    run = optimized.run
    parts = optimized.parts
    partner = {
        parts[k].selected: parts[other].chain.net
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
                vias=[
                    (v.start, v.end)
                    for m, v in zip(read.movable, outcome.via_reports, strict=True)
                    if any(end.chain == i for end in m.ends)
                ],
                title=f"{r.net} on {r.layer}"
                + (f", pair with {partner[i]}" if i in partner else "")
                + f": {run.termination.value}, "
                f"length {r.before.length:.3f} → {r.after.length:.3f} mm",
            )
            for i, (given, unsimplified, polyline, r) in enumerate(
                zip(
                    _with_vias(
                        optimized.whole,
                        optimized.via_inputs,
                        outcome.via_reports,
                        options,
                    ),
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
