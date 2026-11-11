"""One smoothing run, from the KiCad selection to the commit.

The command line and the GUI both call `run_smooth` and present the reports
it hands over; it prints nothing.

`run_smooth` reads the board once, extracts the chains, cut at every
selected via that may move, and, if any of them stays fixed, extracts them
again with the chains through that via whole and its copper pinning. A via
at which a coupled section of a differential pair ends stays fixed, as does
one whose cut takes a piece's partner: the pair's other track pairs with one
piece of the cut track and runs coupled with the other, which is left
unpaired (`_off_pairs`). The final chains, paired as the optimizer pairs
them, decide that, and `_settle` extracts the chains again until no moving
via is left so. Selected tracks that form no chain, a loop nothing of its
net touches or a stray zero-length track, stay on the board as unselected
copper, and the chains are extracted again with them pinning. When a
selected via may move and attraction is on, it finds the ground nets, logs
them and classes every via by them; with attraction off nothing is classed,
and the optimizer's energy is the one without ground vias. It then builds
the fields, optimizes, simplifies and commits. A coupled section the run
leaves bent too tightly is held, and the optimizer runs again from the
previous run's best state. A loop that surrounds nothing would shrink to no
area, to a point or a line doubled back: one that nothing holds open is held
as drawn from the start (`_collapsing`), with the junction it hangs off if
any, and one that shrinks all the same (`_shrunk`) is held as drawn in a run
that resumes from the previous one's best state, on the same clock and time
limit. A chain whose result equals its selected tracks, and that shares no
split track with a changed chain, is left out of the commit; nothing is
committed when every chain is unchanged and no via moved. A selected arc
thus stays an arc only in a chain left exactly as drawn; simplification
merges its pieces and the optimizer shortens it, so it is nearly always
replaced.

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
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from time import monotonic
from typing import NamedTuple

import kipy
import kipy.board
import numpy as np
import shapely
from kipy.errors import ApiError
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from loguru import logger
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from kibends.bursts import DEFAULT_SETTINGS, OptimizerSettings
from kibends.chain import (
    PIN_MERGE_TOLERANCE_MM,
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
    obstacle_reach,
)
from kibends.neighbours import (
    ChainInput,
    chain_field,
    neighbour_fields,
    same_layer_neighbours,
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
    Layout,
    find_pairs,
    lay_out,
    pairs_and_partners_taken,
)
from kibends.pairs import GAP_TOLERANCE, gap_range
from kibends.polylines import enclosed_area, measure
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
    ChainEnd,
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
    """Its time limit covers the whole optimize stage, re-runs included.
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
    """Of the simplified result, or of the selection when unchanged, with the
    other chains at their simplified results."""
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
    """How the last run ended, but `Termination.GAVE_UP` where it converged
    after an earlier run gave up on some problem."""
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
    pairs: list[ChainPair]
    layout: Layout
    via_inputs: list[ViaInput]
    """With the selected chains' ends."""
    run: RunResult
    started: float
    """When the optimize stage started, on `monotonic`'s clock."""
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

    final_inputs: list[ChainInput]
    """The chains' inputs with the moved vias where they end up."""
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
        options.chord_tolerance,
        read_user_frame(board),
        ground_nets=options.ground_nets,
        paired=options.params.attraction > 0.0,
    )
    read, optimized, outcome = _smooth(read, options)
    run = optimized.run
    if options.plot is not None:
        if read.chains:
            _draw(options.plot, read, optimized, outcome)
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
    chord_tolerance: float,
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
    may = [v for v in selection.vias if may_move(v)]
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
    )
    chains, snapshots, movable = _settle(
        _Extracted(kept, locked, snapshots, chains, notes),
        may,
        movable,
        whole,
        replaced,
        frame,
        chord_tolerance,
    )
    return _Read(selected, chains, snapshots, movable, frame, ground, selection.arcs)


def _settle(
    first: _Extracted,
    cut_at: Sequence[Via],
    movable: Sequence[MovableVia],
    whole: BoardSnapshot,
    replaced: Collection[str],
    frame: UserFrame,
    chord_tolerance: float,
) -> tuple[list[Chain], dict[str, LayerSnapshot], list[MovableVia]]:
    """The final chains, the layers they are extracted over, and the vias
    that move, from the `first` extraction, cut at the vias `cut_at`, and
    the vias `movable` judged on it.

    Each pass extracts the chains again with the vias that still move, if
    they differ from the ones the chains were last cut at, leaves out what
    forms no chain (`_leave_out`), and pairs the result as the optimizer
    pairs it; the vias that pairing holds (`_off_pairs`) stay fixed from
    then on. A pass that holds none is the last, so each pass removes a via
    or stops. Only the last pass's lines on what it left out are logged."""
    kept, locked, snapshots, chains, notes = first
    extracted_with = {v.id for v in cut_at}
    movable = list(movable)
    while True:
        said: list[str] = []
        moving = [m.via for m in movable]
        if kept and {v.id for v in moving} != extracted_with:
            fresh: list[str] = []
            chains = _extract_chains(
                kept, locked, snapshots, moving, frame, fresh.append
            )
            notes = fresh
            extracted_with = {v.id for v in moving}
        final, final_snapshots = _leave_out(
            _Extracted(kept, locked, snapshots, chains, notes),
            whole,
            replaced,
            moving,
            frame,
            said.append,
        )
        # Extracting again may order the chains differently.
        movable = [replace(m, ends=carried_ends(m.via, final)) for m in movable]
        pairs, taken = (
            pairs_and_partners_taken(_as_paired(final), chord_tolerance)
            if movable
            else ([], [])
        )
        staying = _off_pairs(movable, final, pairs, taken, frame)
        if len(staying) == len(movable):
            break
        movable = staying
    for line in said:
        logger.debug(line)
    return final, final_snapshots, movable


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
    say: Callable[[str], None],
) -> tuple[list[Chain], dict[str, LayerSnapshot]]:
    """The final chains and the layers they are extracted over. The selected
    tracks no chain of `extracted` took are taken out of the `replaced`
    ones and stay on the board as unselected copper, which pins and blocks
    the chains as any other does, and the chains are extracted again; the
    extraction's notes on them, and the new one's, go to `say`."""
    kept, locked, snapshots, chains, notes = extracted
    left_out = {s.id for s in kept} - {i for c in chains for i in c.segment_ids}
    if not left_out:
        return chains, snapshots
    for note in notes:
        say(note)
    still = [s for s in kept if s.id not in left_out]
    snapshots = _slices(whole, {*replaced} - left_out)
    if not still:
        return [], snapshots
    return (
        _extract_chains(still, locked, snapshots, moving, frame, say),
        snapshots,
    )


def _shrunk(read: _Read, simplified: Sequence[FloatArray]) -> list[int]:
    """The loops among the chains whose results enclose nothing, by index,
    each logged at debug.

    A loop round any copper keeps its centreline at least half its width
    from it, so it encloses at least a circle of that radius. One that
    encloses less surrounds nothing: it is on its way to a point, or, held
    at two points, to a line doubled back on itself."""
    shrunk: list[int] = []
    for k, (chain, polyline) in enumerate(zip(read.chains, simplified, strict=True)):
        if chain.points[0] != chain.points[-1] or all(chain.pinned):
            continue
        if enclosed_area(polyline) >= math.pi * (chain.width / 2) ** 2:
            continue
        logger.debug(
            f"Leaving the closed loop {', '.join(chain.segment_ids)} through "
            f"{read.frame.point(chain.points[0])} as drawn: it surrounds "
            "nothing and shrinks to no area; optimizing again with it held"
        )
        shrunk.append(k)
    return shrunk


def _hold(read: _Read, held: Collection[int]) -> tuple[_Read, list[int]]:
    """`read` with the loops `held` pinned whole, and the moving vias that
    carry an end of one, by index, which stay fixed now and hold every end
    they carried, each logged at debug.

    The other branches of a junction a loop hangs off are pinned there with
    it, as `lay_out` pins every loose end that meets a fixed one."""
    released = [
        v for v, m in enumerate(read.movable) if any(e.chain in held for e in m.ends)
    ]
    ends = {
        (end.chain, -1 if end.last else 0)
        for v in released
        for end in read.movable[v].ends
    }
    for v in released:
        via = read.movable[v].via
        loops = "; ".join(
            ", ".join(read.chains[k].segment_ids)
            for k in dict.fromkeys(e.chain for e in read.movable[v].ends)
            if k in held
        )
        logger.debug(
            f"Via {via.id} at {read.frame.point(via.position)} stays fixed: the "
            f"closed loop {loops} it ends is held as drawn"
        )

    def pinned(k: int, chain: Chain) -> Chain:
        if k in held:
            return replace(chain, pinned=(True,) * len(chain.points), slides=())
        last = len(chain.points) - 1
        pins = tuple(
            pin or (i in (0, last) and (k, -1 if i == last else 0) in ends)
            for i, pin in enumerate(chain.pinned)
        )
        return replace(chain, pinned=pins)

    chains = [pinned(k, chain) for k, chain in enumerate(read.chains)]
    movable = [m for v, m in enumerate(read.movable) if v not in released]
    return replace(read, chains=chains, movable=movable), released


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


def _smooth(read: _Read, options: SmoothOptions) -> tuple[_Read, _Optimized, _Outcome]:
    """Optimize, simplify and report, each loop that would collapse held as
    drawn: one that nothing holds open (`_collapsing`) from the start, and
    one whose result shrank to nothing all the same (`_shrunk`) in a run
    that resumes from the last, on its clock and time limit and with its
    counts."""
    fields = _StaticFields(options)
    collapsing = _collapsing(read, fields)
    if collapsing:
        read, _ = _hold(read, collapsing)
    optimized = _optimize(read, options, fields)
    outcome = _simplify_and_report(read, optimized, options)
    while shrunk := _shrunk(read, outcome.simplified):
        read, released = _hold(read, shrunk)
        optimized = _optimize(
            read,
            options,
            fields,
            earlier=optimized,
            held_loops=shrunk,
            released=released,
        )
        outcome = _simplify_and_report(read, optimized, options)
    return read, optimized, outcome


class _StaticFields:
    """The obstacles of each layer, net and width over the vias that stand
    still, built once for each set of moving vias."""

    def __init__(self, options: SmoothOptions) -> None:
        self._options = options
        self._built: dict[tuple[str, str, float, frozenset[str]], Obstacles] = {}

    def __call__(
        self, snapshot: LayerSnapshot, chain: Chain, moving: Collection[str]
    ) -> Obstacles:
        key = (chain.layer, chain.net, chain.width, frozenset(moving))
        if key not in self._built:
            self._built[key] = _static_field(snapshot, chain, self._options, moving)
        return self._built[key]


def _collapsing(read: _Read, fields: _StaticFields) -> list[int]:
    """The loops among the chains that would shrink to no area, by index,
    each logged at debug: what they are held by cannot keep them open
    (`_held_open`), and they surround nothing they keep clear of."""
    moving = {m.via.id for m in read.movable}
    collapsing: list[int] = []
    for k, chain in enumerate(read.chains):
        if (
            chain.points[0] != chain.points[-1]
            or all(chain.pinned)
            or _held_open(chain)
            or _surrounds_something(chain, read, fields, moving)
        ):
            continue
        logger.debug(
            f"Leaving the closed loop {', '.join(chain.segment_ids)} through "
            f"{read.frame.point(chain.points[0])} as drawn: it surrounds "
            "nothing, and nothing holds it open"
        )
        collapsing.append(k)
    return collapsing


def _held_open(chain: Chain) -> bool:
    """Whether what holds a loop may keep it from shrinking to no area.

    The shortest loop through its pins, in order, is the polygon they make,
    which encloses the area `_shrunk` measures. Its contacts let it pick a
    point in each region instead: it can fold flat when one straight line
    passes through every pin and region. A loose junction goes where the
    other branches let it, so with it, only a loop held at one other point
    or region surely shrinks to nothing."""
    pins = list(
        dict.fromkeys(
            p for p, pin in zip(chain.points, chain.pinned, strict=True) if pin
        )
    )
    regions = [region for _, region in chain.slides]
    loose = not chain.pinned[0]
    if len(pins) + len(regions) + loose <= 2:
        return False
    if loose:
        return True
    if regions:
        return not _on_one_line(pins, regions)
    loop = [p for p, pin in zip(chain.points, chain.pinned, strict=True) if pin]
    return enclosed_area(np.array(loop)) >= math.pi * (chain.width / 2) ** 2


def _on_one_line(points: Sequence[Point2], regions: Sequence[BaseGeometry]) -> bool:
    """Whether one straight line passes through all `points` and `regions`.

    If one does, one passes through two of their corners too, so those
    lines are all there is to try."""
    corners = [
        *points,
        *(
            (float(x), float(y))
            for region in regions
            for x, y in shapely.get_coordinates(region)
        ),
    ]
    x0, y0, x1, y1 = shapely.MultiPoint(corners).bounds
    reach = math.hypot(x1 - x0, y1 - y0)
    for a, b in combinations(corners, 2):
        length = math.dist(a, b)
        if length == 0.0:
            continue
        dx, dy = (b[0] - a[0]) / length * reach, (b[1] - a[1]) / length * reach
        line = LineString([(a[0] - dx, a[1] - dy), (a[0] + dx, a[1] + dy)])
        if all(
            line.distance(Point(p)) <= PIN_MERGE_TOLERANCE_MM for p in points
        ) and all(line.intersects(region) for region in regions):
            return True
    return False


def _surrounds_something(
    chain: Chain, read: _Read, fields: _StaticFields, moving: Collection[str]
) -> bool:
    """Whether the area a loop encloses as drawn reaches anything it keeps
    clear of: the static copper, holes and keepouts of its field, the
    board's edge, or another net's selected chain or moving via."""
    interior = shapely.make_valid(shapely.Polygon(chain.points))
    field = fields(read.snapshots[chain.layer], chain, moving)
    if field.board_outline is not None and not field.board_outline.covers(interior):
        return True
    others = [
        *(o.geometry for o in field.items),
        *(
            LineString(c.points)
            for c in read.chains
            if c.layer == chain.layer and c.net != chain.net
        ),
        *(
            Point(m.via.position)
            for m in read.movable
            if m.via.net != chain.net and m.via.reaches(chain.layer)
        ),
    ]
    return any(interior.intersects(other) for other in others)


def _optimize(
    read: _Read,
    options: SmoothOptions,
    fields: _StaticFields,
    *,
    earlier: _Optimized | None = None,
    held_loops: Collection[int] = (),
    released: Collection[int] = (),
) -> _Optimized:
    """Optimize the chains of `read`, again after `earlier` if given, with
    the loops `held_loops` held since and the vias `released` fixed since,
    by their index in that run: the run resumes from its result
    (`_resumed_holding`), its start stays the stage's start, which the time
    limit counts from, and its iterations, abnormal bursts and give-ups
    count in."""
    moving = {m.via.id for m in read.movable}
    whole = _chain_inputs(read.chains, read.snapshots, fields, moving, read.frame)
    pairs = find_pairs(
        _as_paired(read.chains), options.chord_tolerance, frame=read.frame
    )

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
    if earlier is None:
        logger.info(_optimizing(len(read.chains), len(pairs), len(read.movable)))
    # One classing of the vias standing still both splits the moving vias'
    # fields and gives the pairwise term its vias.
    static = _static_vias(read.snapshots, moving, read.ground_nets)
    via_inputs = _via_inputs(
        read.movable, read.snapshots, options, read.ground_nets, static
    )
    started = monotonic() if earlier is None else earlier.started
    previous = (
        None
        if earlier is None
        else _resumed_holding(earlier.run, layout, held_loops, released)
    )
    held_label = "loops"
    iterations = 0 if earlier is None else earlier.run.iterations
    abnormal = {} if earlier is None else dict(earlier.run.abnormal)
    # The earlier run logged its own.
    logged = {} if earlier is None else dict(earlier.run.gave_up)
    gave_up = dict(logged)
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
            held_label=held_label,
            static_vias=tuple(static.values()),
        )
        iterations += run.iterations
        for description, count in run.abnormal:
            abnormal[description] = abnormal.get(description, 0) + count
        for description, streak in run.gave_up:
            gave_up.setdefault(description, streak)
        held = _bent_too_tightly(layout.sections, run.pairs)
        if not held:
            # A problem given up on stopped on abnormal bursts; a later run
            # converging from its best state does not make that stop a
            # convergence.
            termination = (
                Termination.GAVE_UP
                if gave_up and run.termination is Termination.CONVERGED
                else run.termination
            )
            run = replace(
                run,
                termination=termination,
                iterations=iterations,
                abnormal=tuple(abnormal.items()),
                gave_up=tuple(gave_up.items()),
            )
            _log_gave_up([(d, streak) for d, streak in run.gave_up if d not in logged])
            elapsed = monotonic() - started
            return _Optimized(whole, pairs, layout, via_inputs, run, started, elapsed)
        # The other chains were smoothed around the sections' results, so
        # the sections cannot simply keep their own tracks; the run goes on
        # from where this one left them, with these sections held.
        previous = run
        held_label = "sections"
        sections = list(layout.sections)
        for k, hold in held.items():
            sections[k] = replace(sections[k], hold=hold)
        layout = replace(layout, sections=sections)


def _resumed_holding(
    start: RunResult,
    layout: Layout,
    held: Collection[int],
    released: Collection[int],
) -> RunResult | None:
    """`start`, the result of a run on `layout`'s inputs, with each chain
    `held` since back as drawn and pinned, every chain end pinned since,
    at a junction one hangs off or a via `released` since, back where it
    was drawn, and those vias left out; None when a held chain is paired,
    and the run starts over."""
    slots = [layout.slots[k] for k in held]
    reset = {slot for slot in slots if slot is not None}
    if len(reset) < len(slots):
        return None
    chains: list[ChainResult] = []
    for i, (given, result) in enumerate(zip(layout.inputs, start.chains, strict=True)):
        points = given.centreline.points
        if i in reset:
            chains.append(
                replace(
                    result,
                    polyline=points,
                    fixed=given.fixed,
                    contact_of=no_contacts(points),
                )
            )
            continue
        polyline, fixed = result.polyline.copy(), result.fixed.copy()
        # A pinned end is where it was drawn, unless something carried it.
        for end in (0, -1):
            if given.fixed[end]:
                polyline[end], fixed[end] = points[end], True
        chains.append(replace(result, polyline=polyline, fixed=fixed))
    vias = tuple(via for v, via in enumerate(start.vias) if v not in released)
    return replace(start, chains=tuple(chains), vias=vias)


def _log_gave_up(gave_up: Sequence[tuple[str, int]]) -> None:
    for description, streak in gave_up:
        logger.debug(
            f"Gave up on {description} after {streak} abnormal optimizer bursts "
            "in a row; they keep their best shapes"
        )


def _bent_too_tightly(
    sections: Sequence[PairInput], results: Sequence[PairResult]
) -> dict[int, Hold]:
    """The sections the run moved but could not keep within the bend-radius
    limit, by index, and why: its own parts beyond the turn, if they came
    too close to each other, else the copper around it, kept it from
    rounding a corner that resampling cut too tightly. Each logged at
    debug."""
    held: dict[int, Hold] = {}
    for k, (given, result) in enumerate(zip(sections, results, strict=True)):
        bend = result.bend
        if result.block.held or bend is None or not bend.too_tight:
            continue
        hold = Hold.FOLDING if result.too_close else Hold.CROWDING
        logger.debug(
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
    # The statistics see the moved vias where they end up.
    final_inputs = _with_vias(
        optimized.whole, optimized.via_inputs, via_reports, options
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


def _as_paired(chains: Sequence[Chain]) -> list[Chain]:
    """`chains` as the run pairs them: a pair's tracks keep their contacts
    where they are."""
    return [chain.with_slides_pinned() for chain in chains]


def _off_pairs(
    movable: Sequence[MovableVia],
    chains: Sequence[Chain],
    pairs: Sequence[ChainPair],
    partners_taken: Sequence[tuple[int, int]],
    frame: UserFrame,
) -> list[MovableVia]:
    """`movable` but the vias that stay fixed for a pair, with a debug line
    each: one at which a coupled section of one of `pairs` itself ends, as
    a section's ends do not move, and one that ends a chain of
    `partners_taken`, (alone, partner) chain indices, as cut there a pair's
    track pairs only in part and must be whole through it. A via at an
    uncoupled piece of a pair's chain moves with that piece."""
    sections = {
        end: f"{chains[p.a].net} and {chains[p.b].net} on {chains[end.chain].layer}"
        for p in pairs
        for end in p.coupled_ends()
    }
    partner_of = dict(partners_taken)

    def section_end(end: ChainEnd) -> str | None:
        if end not in sections:
            return None
        return f"a coupled section of differential pair {sections[end]} ends at it"

    def partner_taken(end: ChainEnd) -> str | None:
        if end.chain not in partner_of:
            return None
        a, b = chains[end.chain], chains[partner_of[end.chain]]
        return (
            f"cut there, {a.net} on {a.layer} runs coupled with {b.net} beside "
            "the piece that pairs with it"
        )

    kept: list[MovableVia] = []
    for m in movable:
        reason = next(
            (
                r
                for rule in (section_end, partner_taken)
                for end in m.ends
                if (r := rule(end)) is not None
            ),
            None,
        )
        if reason is None:
            kept.append(m)
            continue
        logger.debug(
            f"Via {m.via.id} at {frame.point(m.via.position)} stays fixed: {reason}"
        )
    return kept


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
    fields: _StaticFields,
    moving: Collection[str],
    frame: UserFrame,
) -> list[ChainInput]:
    """One optimizer input per chain, over its layer's snapshot without the
    `moving` vias, which the optimizer adds where they are; chains of one
    layer, net and width share a field. Descriptions give positions in
    `frame`."""
    inputs: list[ChainInput] = []
    for chain in chains:
        snapshot = snapshots[chain.layer]
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
                fields(snapshot, chain, moving),
                tuple((i, Obstacles((), region)) for i, region in chain.slides),
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
            results.append(_WholeResult(r.polyline, r.before, r.rejected_edges))
            continue
        pieces = [
            run.chains[s.piece] for s in layout.stretches[k] if s.piece is not None
        ]
        field = chain_field(
            inputs[k].field,
            neighbour_fields(inputs, k, neighbours[k], originals, reach),
        )
        results.append(
            _WholeResult(
                unsimplified[k],
                measure(originals[k], field, DEFAULT_SETTINGS.spacing),
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
    changes = _changed_with_shared_tracks(
        chains,
        [
            snapped_path(s) != snapped_path(given.centreline.points)
            for s, given in zip(simplified, inputs, strict=True)
        ],
    )
    for i, (chain, given, result, changed) in enumerate(
        zip(chains, inputs, results, changes, strict=True)
    ):
        given_points = given.centreline.points
        field = chain_field(
            given.field, neighbour_fields(inputs, i, neighbours[i], simplified, reach)
        )
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
