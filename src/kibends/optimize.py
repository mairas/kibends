"""Outer optimisation loop: resample, bubble bounds, L-BFGS-B burst, edge check.

Free vertices move only inside their bubble, a box inscribed in the free disc
around them, so they never jump over an obstacle; an exact edge check after
each burst rejects edges that newly cross an obstacle. Together they keep the
chain on the same side of every obstacle (topology preservation).

Several chains are optimised by alternating bursts: each outer iteration gives
every chain one burst in turn while the other chains on its layer stand still
as obstacles, so the same-side guarantee holds between chains too. The
coupled sections of differential pairs take their turns after the chains,
each as its centreline (`kibends.pair_blocks`), and moving vias follow as
blocks of their own (`kibends.via_moves`).
"""

import math
import time
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
from itertools import pairwise
from typing import NamedTuple

import numpy as np
from loguru import logger
from scipy.spatial import cKDTree

from kibends.bursts import (
    DEFAULT_SETTINGS,
    INSIDE_TOLERANCE,
    Burst,
    OptimizerSettings,
    bounded_burst,
    half_side,
    revert_until_clear,
)
from kibends.chain import Point2
from kibends.energy import (
    MIN_EDGE_LENGTH,
    BoolArray,
    DistanceField,
    EnergyParams,
    FloatArray,
    IntArray,
    ObstacleField,
    arc_lengths,
    free_vertex_objective,
    obstacle_reach,
    total_energy,
)
from kibends.neighbours import (
    ChainInput,
    chain_field,
    neighbour_field,
    same_layer_neighbours,
)
from kibends.obstacles import HOLE_CLEARANCE, HOLE_TO_HOLE, ChainCentreline
from kibends.pair_blocks import PairBlock, PairInput, pair_block
from kibends.pairs import BendCheck, bend_radii, check_bend_radius, too_tight
from kibends.via_moves import MovingVias, ViaBurst, via_burst
from kibends.vias import ViaInput

# A span only slightly longer than a multiple of the spacing gets no extra
# vertex, so float noise in re-measured lengths does not change the count.
RESAMPLE_SLACK = 1e-6

# Below this turning angle a vertex counts as straight (infinite radius).
STRAIGHT_TURN = 1e-9

MEASURE_SAMPLES_PER_SPACING = 4

# After this many abnormal L-BFGS-B ends in a row a chain counts as settled
# and keeps what its bursts reached; a neighbour's move can still wake it.
MAX_ABNORMAL_BURSTS = 3

# A via's burst resamples a chain that has an edge longer than this many
# spacings, as stretching it while the chain rested may leave it.
STRETCHED = 2.0

# Shape change is measured from each vertex to the other polyline's edges
# within this many edges of the same index.
SHAPE_WINDOW = 3

# A run whose steps change no shape by more than this, KiCad's 1 nm grid,
# mm, has nothing to do, however its stop rules are set.
NOTHING_MOVED = 1e-6

# Offsets from edge k of the vertices its bend radius depends on: the turns
# at both its ends take the vertices on either side.
BEND_REACH = np.arange(-1, 3)

# A run logs its progress at INFO at most this often, seconds, and not at all
# if it ends sooner; DEBUG gets every outer iteration.
PROGRESS_INTERVAL = 5.0


class Termination(Enum):
    CONVERGED = "converged"
    NOTHING_TO_DO = "nothing to do"
    ITERATION_CAP = "iteration cap reached"
    TIME_LIMIT = "time limit reached"
    ABNORMAL = "L-BFGS-B terminated abnormally"
    """Some chain stopped after MAX_ABNORMAL_BURSTS abnormal bursts in a row
    and no burst of the run lowered the energy: nothing usable was found."""


@dataclass(frozen=True)
class FrozenVertex:
    """A vertex left in place because it starts inside an obstacle."""

    position: Point2
    distance: float
    """Signed distance to the clearance boundary, negative, mm."""
    neighbours: tuple[str, ...] = ()
    """Descriptions of the other selected chains whose clearance it starts in."""


@dataclass(frozen=True)
class PolylineStats:
    length: float
    segment_count: int
    tightest_bend_radius: float
    """Smallest discrete radius of curvature after resampling at the spacing;
    infinite for a straight polyline, mm."""
    min_clearance: float
    """Smallest signed distance to the clearance boundary along the edges, mm."""
    mean_clearance: float
    """Length-weighted mean of that distance, mm."""


@dataclass(frozen=True)
class OptimizeResult:
    polyline: FloatArray
    fixed: BoolArray
    """Pinned and frozen vertices of `polyline`; simplification keeps them."""
    termination: Termination
    iterations: int
    rejected_edges: int
    """Edges whose burst moves were undone because they crossed an obstacle."""
    abnormal_bursts: int
    """Bursts L-BFGS-B ended abnormally; each was taken like any other."""
    frozen: tuple[FrozenVertex, ...]
    before: PolylineStats
    after: PolylineStats


@dataclass(frozen=True)
class ChainResult:
    polyline: FloatArray
    fixed: BoolArray
    """Pinned and frozen vertices of `polyline`; simplification keeps them."""
    rejected_edges: int
    """Edges whose burst moves were undone because they crossed an obstacle."""
    abnormal_bursts: int
    """Bursts L-BFGS-B ended abnormally; each was taken like any other."""
    frozen: tuple[FrozenVertex, ...]
    before: PolylineStats


@dataclass(frozen=True)
class ViaResult:
    position: Point2
    frozen: bool
    """It started more than δ inside its clearance and was left in place."""
    rejected_edges: int
    """End edges of its chains whose crossing an obstacle undid its move."""


@dataclass(frozen=True)
class PairResult:
    block: PairBlock
    """The section as the run took it."""
    centreline: FloatArray
    fixed: BoolArray
    """The vertices of `centreline` that simplification keeps: those the run
    pinned or froze, but the direction pins of a section it moved, as a
    merged edge stays within the chord tolerance of them."""
    a: FloatArray
    b: FloatArray
    """The two tracks along `centreline`, b running the same way as a; the
    section's own tracks when the run did nothing."""
    rejected_edges: int
    too_close: int
    """Of `rejected_edges`, the vertex pairs that came closer than the
    section keeps its parts beyond a turn (`close_to_itself`)."""
    abnormal_bursts: int
    frozen: tuple[FrozenVertex, ...]
    before: PolylineStats
    """Of the starting centreline."""
    bend: BendCheck | None
    """The centreline's bend radius against the limit of the section's
    widest gap: its routing's for a section held because that bends too
    tightly, its result's for one the run moved; else None."""


@dataclass(frozen=True)
class RunResult:
    chains: tuple[ChainResult, ...]
    """In the order the chains were given."""
    termination: Termination
    iterations: int
    """Outer iterations, each giving every chain, pair and via one burst."""
    vias: tuple[ViaResult, ...] = ()
    """In the order the vias were given."""
    pairs: tuple[PairResult, ...] = ()
    """In the order the coupled sections were given."""
    gave_up: tuple[tuple[str, int], ...] = ()
    """The chains given up on after abnormal bursts in a row, by description,
    and how many there were."""


def optimize(
    polyline: FloatArray,
    fixed: BoolArray,
    field: ObstacleField,
    params: EnergyParams,
    settings: OptimizerSettings = DEFAULT_SETTINGS,
) -> OptimizeResult:
    """Minimise the chain's energy with its fixed vertices held in place."""
    alone = ChainInput(
        ChainCentreline("chain", "", 0.0, 0.0, polyline), "", fixed, field
    )
    run = optimize_chains([alone], params, settings)
    (chain,) = run.chains
    after = (
        chain.before
        if run.termination is Termination.NOTHING_TO_DO
        else measure(chain.polyline, field, settings.spacing)
    )
    return OptimizeResult(
        polyline=chain.polyline,
        fixed=chain.fixed,
        termination=run.termination,
        iterations=run.iterations,
        rejected_edges=chain.rejected_edges,
        abnormal_bursts=chain.abnormal_bursts,
        frozen=chain.frozen,
        before=chain.before,
        after=after,
    )


def optimize_chains(
    chains: Sequence[ChainInput],
    params: EnergyParams,
    settings: OptimizerSettings = DEFAULT_SETTINGS,
    vias: Sequence[ViaInput] = (),
    *,
    hole_clearance: float = HOLE_CLEARANCE,
    hole_to_hole: float = HOLE_TO_HOLE,
    pairs: Sequence[PairInput] = (),
    clock: Callable[[], float] = time.monotonic,
    started: float | None = None,
    start: RunResult | None = None,
) -> RunResult:
    """Minimise the chains' summed energy by alternating bursts, in the given order.

    Each chain's energy is taken in its own field plus the other chains on its
    layer, so every pair counts from both sides. The run keeps one best state,
    all polylines together, replaced only after a burst that does not raise
    the summed energy and leaves no chain with more violations than it started
    with; a cap or abnormal stop returns that state. A burst L-BFGS-B ends
    abnormally (status 2, a failed line search) still holds the best point it
    found, so the same guard decides whether it enters the best state.

    A chain rests once it has converged or stalled on its own energy, until a
    neighbour's move changes that energy by more than the stall tolerance.
    The run converges when every chain rests.

    Each of `vias` gets a burst after the chains in every outer iteration:
    its position and the free vertices of the chains it joins move together,
    each inside its bubble, the chains' ends keeping their offsets from its
    centre, and so do the chains pressing on it. Its field holds the chains
    near it and the other moving vias at their current shapes, kept at
    `hole_clearance` and `hole_to_hole` where holes are involved; a chain of
    its net that does not end at it and the via may touch but not overlap,
    so that the via makes no new connection. The run energy adds each via's
    obstacle terms. A move is undone if it raises the run energy, adds a
    violation to any chain, or puts the via newly inside its clearance. The
    best state holds the vias' positions too, and a via rests like a chain.

    Each of `pairs`, a coupled section of a differential pair, takes its
    burst after the chains as a chain of its own, its centreline, with the
    bend-radius hinge added to its energy (`kibends.pair_blocks`). The
    other chains see its two tracks, the vias its outer extent widened to
    cover the outer track's mitred corners. It counts in the run energy,
    the violations guard, the best state and the rest like any chain. A
    burst moves back the vertices of edges it newly bends below the limit,
    as it does those of edges that newly cross an obstacle. Resampling can
    cut a corner into a bend below the limit, which the copper around the
    section may keep it from rounding, so a centreline may end too tight:
    its result's `bend` says so. A via's burst never carries one.

    With a time limit in `settings`, the run stops after the first burst,
    of a chain or a via, that ends when `clock` reads the limit or more past
    `started` (by default when the run starts), and returns its best state.

    A run given the result of an earlier one on the same inputs as `start`
    starts from its best state, the chains, the coupled sections not held
    in this run and the vias where it left them, unless that leaves some
    chain with more violations than its input or a via newly inside its
    clearance; then it starts from the inputs. Either way it completes its
    first outer iteration before it looks at the time limit, so that every
    chain has a burst against what changed since.

    Each outer iteration logs its progress at DEBUG, and at INFO once
    `PROGRESS_INTERVAL` has passed since `started` or the last INFO line.
    """
    for chain in chains:
        if not (chain.fixed[0] and chain.fixed[-1]):
            raise ValueError("both endpoints of the chain must be fixed")
    blocks = [pair_block(pair, settings.spacing) for pair in pairs]
    everything = [*chains, *(block.chain for block in blocks)]
    run = _Run(
        everything,
        params,
        settings,
        MovingVias(everything, vias, params, hole_clearance, hole_to_hole),
        blocks,
        start,
    )
    if not run.moving and not run.free_vias:
        return run.result(Termination.NOTHING_TO_DO, 0)
    begun = clock()
    origin = begun if started is None else started
    deadline = None if settings.time_limit is None else origin + settings.time_limit
    last_shown = begun
    termination = Termination.ITERATION_CAP
    iteration = 0
    for iteration in range(1, settings.max_iterations + 1):
        timed = deadline is not None and (start is None or iteration > 1)
        out_of_time = False
        for step in run.steps():
            step()
            if timed and deadline is not None and clock() >= deadline:
                out_of_time = True
                break
        now = clock()
        message = _progress(run, iteration, now - origin)
        if now - last_shown >= PROGRESS_INTERVAL:
            logger.info(message)
            last_shown = now
        else:
            logger.debug(message)
        if all(run.resting(i) for i in run.moving) and all(
            run.via_resting(v) for v in run.free_vias
        ):
            if iteration == 1 and not run.moved:
                return run.result(Termination.NOTHING_TO_DO, 1)
            termination = Termination.CONVERGED
            break
        if out_of_time:
            termination = Termination.TIME_LIMIT
            break
    gave_up = tuple(
        (everything[i].centreline.description, streak)
        for i, streak in enumerate(run.abnormal_streak)
        if streak >= MAX_ABNORMAL_BURSTS
    )
    if gave_up and run.best_energy >= run.start_energy:
        termination = Termination.ABNORMAL
    return replace(run.result(termination, iteration, best=True), gave_up=gave_up)


def _progress(run: _Run, iteration: int, elapsed: float) -> str:
    """One line on where the run stands after an outer iteration."""
    chains, vias = run.awake
    moving = f"{chains} of {len(run.moving)} chains"
    if run.free_vias:
        moving += f" and {vias} of {len(run.free_vias)} vias"
    return (
        f"Iteration {iteration}, {elapsed:.1f} s: length {run.length():.3f} mm, "
        f"energy {run.run_energy():.3f} (best {run.best_energy:.3f}), "
        f"{moving} moving"
    )


@dataclass(frozen=True)
class _Shape:
    """A chain as the run holds it between steps."""

    points: FloatArray
    """The polyline as the last step left it: what the energy, the guard,
    the best state and the other chains see."""
    anchors: BoolArray
    resampled: FloatArray
    """Where the chain's next burst starts: `points` resampled."""
    resampled_anchors: BoolArray


@dataclass
class _Rest:
    """Whether a chain or a via rests: it has converged or stalled, and its
    energy has not changed by more than the stall tolerance since."""

    energy: float
    """The energy it settled at, or its starting energy."""
    lowest: list[float]
    """The lowest energy after each of its steps since it last woke."""
    settled: bool = False

    def resting(self, energy: float, settings: OptimizerSettings) -> bool:
        return self.settled and abs(energy - self.energy) <= settings.stall_tolerance

    def wake(self, energy: float, *, forget: bool = False) -> None:
        """Start a step: one woken by a neighbour, or told to `forget`,
        forgets the energies it recorded."""
        if self.settled or forget:
            self.lowest = [energy]

    def record(self, energy: float) -> None:
        self.lowest.append(min(self.lowest[-1], energy))

    def stalled(self, settings: OptimizerSettings) -> bool:
        return stalled(self.lowest, settings)

    def settle(self, settled: bool, energy: float) -> None:
        self.settled = settled
        self.energy = energy


class _Run:
    """One run of `optimize_chains`: every chain's and via's current shape,
    field, energy and rest, and the best state so far.

    The coupled sections of pairs are the last chains, as their centrelines.
    A chain's burst and a via's burst are its steps. Either leaves the run
    in a state that `accept_if_best` takes as the best one if it lowers the
    run energy without adding violations.
    """

    def __init__(
        self,
        chains: Sequence[ChainInput],
        params: EnergyParams,
        settings: OptimizerSettings,
        vias: MovingVias,
        pairs: Sequence[PairBlock] = (),
        start: RunResult | None = None,
    ) -> None:
        self.chains = chains
        self.params = params
        self.settings = settings
        self.vias = vias
        self.count = len(chains)
        self.first_pair = self.count - len(pairs)
        self.pairs = {self.first_pair + k: block for k, block in enumerate(pairs)}
        self.nets = [
            block.nets
            if (block := self.pairs.get(i)) is not None
            else frozenset({chain.centreline.net})
            for i, chain in enumerate(chains)
        ]
        self.bend_limits = [
            block.limit if (block := self.pairs.get(i)) is not None else None
            for i in range(self.count)
        ]
        self.aparts = [
            block.apart if (block := self.pairs.get(i)) is not None else None
            for i in range(self.count)
        ]
        self.neighbours = same_layer_neighbours(chains)
        self.reach = obstacle_reach(params)
        self.originals = [chain.centreline.points for chain in chains]
        self.positions = vias.starting_positions()
        self.starting_positions = list(self.positions)
        self.statics: list[ObstacleField] = [
            _Remembered(chain.field) if self.neighbours[i] or vias.vias else chain.field
            for i, chain in enumerate(chains)
        ]
        self.dynamic: list[dict[int, ObstacleField | None]] = [
            {
                j: self.neighbour_field(
                    i, self.shown(j, self.originals[j]), self.originals[i]
                )
                for j in self.neighbours[i]
            }
            for i in range(self.count)
        ]
        self.via_parts: list[ObstacleField | None] = [
            vias.via_obstacles_for_chain(i, self.positions) for i in range(self.count)
        ]
        self.fields: list[ObstacleField] = [self.field_of(i) for i in range(self.count)]
        self.anchors: list[BoolArray] = []
        self.frozen: list[tuple[FrozenVertex, ...]] = []
        for i, chain in enumerate(chains):
            mask, records = frozen_vertices(
                chain, self.fields[i], self.dynamic[i], chains, settings
            )
            self.anchors.append(chain.fixed | mask)
            self.frozen.append(records)
        self.before = [
            measure(points, field, settings.spacing)
            for points, field in zip(self.originals, self.fields, strict=True)
        ]
        self.rejected_edges = [0] * self.count
        self.too_close = [0] * self.count
        self.abnormal_bursts = [0] * self.count
        self.abnormal_streak = [0] * self.count
        self.via_count = len(vias.vias)
        self.via_frozen = [
            vias.distance(v, self.positions, self.originals)
            < -settings.bubble_tolerance
            for v in range(self.via_count)
        ]
        self.free_vias = [v for v in range(self.via_count) if not self.via_frozen[v]]
        self.via_rejected = [0] * self.via_count

        # A burst starts from its chain's polyline resampled; the polyline as
        # the burst left it is what the energy, the guard, the best state and
        # the other chains see.
        self.states: list[_Shape] = []
        for i, (original, anchor) in enumerate(
            zip(self.originals, self.anchors, strict=True)
        ):
            # Resampling would put free vertices between a held pair's pins.
            if (block := self.pairs.get(i)) is not None and block.held:
                points, mask = original, anchor
            else:
                points, mask = resample(original, anchor, settings.spacing)
            moves = not mask.all()
            self.states.append(
                _Shape(
                    points if moves else original,
                    mask if moves else anchor,
                    points,
                    mask,
                )
            )
        self.moving = [
            i for i in range(self.count) if not self.states[i].resampled_anchors.all()
        ]
        self.moving_chains = [i for i in self.moving if i < self.first_pair]
        self.shapes = list(self.originals)
        # Whether any step changed a shape by more than KiCad's grid.
        self.moved = False
        self.awake = (0, 0)

        # Resampling can lay a chord across an obstacle corner or put a vertex
        # inside one, and a burst only rejects crossings its own moves create,
        # so the result is limited to polylines with no more violations than
        # the resampled input. Counting rather than locating them lets the
        # input's own violations move as the chain tightens.
        self.allowed = [
            violations(s.points, s.anchors, self.fields[i])
            for i, s in enumerate(self.states)
        ]
        self.counts: list[int | None] = list(self.allowed)
        if start is not None and self.resume(start):
            self.counts = [None] * self.count
        self.energies = [self.chain_energy(i) for i in range(self.count)]
        self.via_energies = [
            vias.energy(v, self.positions, self.current) for v in range(self.via_count)
        ]
        self.best, self.best_masks = self.current, self.masks
        self.best_positions = list(self.positions)
        self.start_energy = self.best_energy = self.run_energy()

        # The stall check follows every burst, not just the ones the guard
        # accepts, or a briefly cut corner would end the run early.
        self.rests = [_Rest(energy, [energy]) for energy in self.energies]
        self.carried = [set(vias.incident[v]) for v in range(self.via_count)]
        self.via_rests = [
            _Rest(energy, [energy])
            for energy in (self.via_block_energy(v) for v in range(self.via_count))
        ]

    def resume(self, start: RunResult) -> bool:
        """Take `start`'s best state as the current one, but the sections
        held in this run, unless that leaves some chain with more violations
        than `allowed` or a via newly inside its clearance. Returns whether
        it did."""
        settings = self.settings
        inside = [
            self.vias.distance(v, self.positions, self.current)
            for v in range(self.via_count)
        ]
        saved = (self.states, self.positions, self.dynamic, self.via_parts, self.fields)
        states = list(self.states)
        for i in self.moving:
            block = self.pairs.get(i)
            if block is None:
                result = start.chains[i]
                points, mask = result.polyline, result.fixed
            elif block.held:
                continue
            else:
                section = start.pairs[i - self.first_pair]
                points = section.centreline
                mask = section.fixed | block.direction_pins(points)
            states[i] = _Shape(points, mask, *resample(points, mask, settings.spacing))
        self.states = states
        self.positions = [np.array(v.position, dtype=float) for v in start.vias]
        current = self.current
        self.dynamic = [
            {
                j: self.neighbour_field(i, self.shown(j, current[j]), current[i])
                for j in self.neighbours[i]
            }
            for i in range(self.count)
        ]
        self.via_parts = [
            self.vias.via_obstacles_for_chain(i, self.positions)
            for i in range(self.count)
        ]
        self.fields = [self.field_of(i) for i in range(self.count)]
        if all(
            violations(s.points, s.anchors, self.fields[i]) <= self.allowed[i]
            for i, s in enumerate(self.states)
        ) and all(
            self.vias.distance(v, self.positions, current)
            >= min(inside[v], -INSIDE_TOLERANCE)
            for v in range(self.via_count)
        ):
            self.shapes = list(current)
            self.moved = True
            return True
        self.states, self.positions, self.dynamic, self.via_parts, self.fields = saved
        return False

    @property
    def current(self) -> list[FloatArray]:
        return [s.points for s in self.states]

    @property
    def masks(self) -> list[BoolArray]:
        return [s.anchors for s in self.states]

    def result(
        self, termination: Termination, iterations: int, *, best: bool = False
    ) -> RunResult:
        """The best state, with `best`, else the run's input."""
        polylines = self.best if best else self.originals
        masks = self.best_masks if best else self.anchors
        positions = self.best_positions if best else self.starting_positions
        return RunResult(
            tuple(
                ChainResult(
                    polyline=polylines[i],
                    fixed=masks[i],
                    rejected_edges=self.rejected_edges[i],
                    abnormal_bursts=self.abnormal_bursts[i],
                    frozen=self.frozen[i],
                    before=self.before[i],
                )
                for i in range(self.first_pair)
            ),
            termination,
            iterations,
            tuple(
                ViaResult(
                    (float(positions[v][0]), float(positions[v][1])),
                    self.via_frozen[v],
                    self.via_rejected[v],
                )
                for v in range(self.via_count)
            ),
            tuple(
                self.pair_result(i, block, polylines[i], masks[i], best=best)
                for i, block in self.pairs.items()
            ),
        )

    def pair_result(
        self,
        i: int,
        block: PairBlock,
        points: FloatArray,
        mask: BoolArray,
        *,
        best: bool,
    ) -> PairResult:
        given = block.given
        moved = best and not block.held
        a, b = block.offset(points) if best else (given.a.points, given.b.points)
        return PairResult(
            block=block,
            centreline=points,
            fixed=mask & ~block.direction_pins(points) if moved else mask,
            a=a,
            b=b,
            rejected_edges=self.rejected_edges[i],
            too_close=self.too_close[i],
            abnormal_bursts=self.abnormal_bursts[i],
            frozen=self.frozen[i],
            before=self.before[i],
            bend=(
                check_bend_radius(points, block.gaps.widest, given.a.width)
                if moved
                else block.tight_start
            ),
        )

    def shown(self, j: int, points: FloatArray) -> list[tuple[ChainInput, FloatArray]]:
        """What the others see of chain j at `points`: a pair's two tracks."""
        block = self.pairs.get(j)
        if block is None:
            return [(self.chains[j], points)]
        return list(zip(block.tracks, block.offset(points), strict=True))

    def neighbour_field(
        self,
        i: int,
        shown: Sequence[tuple[ChainInput, FloatArray]],
        own_points: FloatArray,
    ) -> ObstacleField | None:
        """`shown` as obstacles to chain i at `own_points`."""
        block = self.pairs.get(i)
        if block is None:
            return neighbour_field(self.chains[i], shown, self.reach, own_points)
        return neighbour_field(
            block.facing, shown, self.reach, own_points, as_net=block.as_net
        )

    def field_of(
        self,
        i: int,
        without_chains: Collection[int] = (),
        without_via: int | None = None,
    ) -> ObstacleField:
        """Chain i's field, but the chains `without_chains` and the via
        `without_via`."""
        via_part = (
            self.via_parts[i]
            if without_via is None
            else self.vias.via_obstacles_for_chain(i, self.positions, without_via)
        )
        return chain_field(
            self.statics[i],
            [
                *(f for j, f in self.dynamic[i].items() if j not in without_chains),
                via_part,
            ],
        )

    def chain_energy(self, i: int) -> float:
        shape = self.states[i]
        return total_energy(
            shape.points,
            shape.anchors,
            self.fields[i],
            self.params,
            self.settings.short_edge,
            self.bend_limits[i],
        )[0]

    def run_energy(self) -> float:
        return sum(self.energies) + sum(self.via_energies)

    def counted(self, i: int) -> int:
        number = self.counts[i]
        if number is None:
            shape = self.states[i]
            number = self.counts[i] = violations(
                shape.points, shape.anchors, self.fields[i]
            )
        return number

    def accept_if_best(self) -> None:
        """Take the current state as the best if it does not raise the run
        energy over the best and leaves no chain with more violations than
        it started with."""
        energy = self.run_energy()
        if energy <= self.best_energy and all(
            self.counted(j) <= self.allowed[j] for j in range(self.count)
        ):
            self.best, self.best_masks = self.current, self.masks
            self.best_energy = energy
            self.best_positions = list(self.positions)

    def resting(self, i: int) -> bool:
        return self.rests[i].resting(self.energies[i], self.settings)

    def steps(self) -> Iterator[Callable[[], None]]:
        """One outer iteration's steps: each chain's burst, then each via's,
        of those not resting when their turn comes. `awake` counts the
        chains and the vias given one."""
        self.awake = (0, 0)
        for i in self.moving:
            if not self.resting(i):
                self.awake = (self.awake[0] + 1, self.awake[1])
                yield partial(self.chain_step, i)
        for v in self.free_vias:
            if not self.via_resting(v):
                self.awake = (self.awake[0], self.awake[1] + 1)
                yield partial(self.via_step, v)

    def length(self) -> float:
        """The summed length of the selected tracks at their current shapes,
        a pair's section as its two tracks, mm."""
        return sum(
            float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            for j, polyline in enumerate(self.current)
            for _, points in self.shown(j, polyline)
        )

    def via_block_energy(self, v: int) -> float:
        """Via v's energy and that of the chains its last burst moved."""
        return self.via_energies[v] + sum(self.energies[i] for i in self.carried[v])

    def via_resting(self, v: int) -> bool:
        return self.via_rests[v].resting(self.via_block_energy(v), self.settings)

    def reshaped(self, i: int) -> None:
        """Show chain i at its current shape to the chains and vias that see
        it, and measure their energies again."""
        current = self.current
        shown = self.shown(i, current[i])
        for j in self.neighbours[i]:
            self.dynamic[j][i] = self.neighbour_field(j, shown, current[j])
            self.fields[j] = self.field_of(j)
            if self.nets[j] & self.nets[i]:
                # How much of j is left out at a junction depends on the
                # angle i now leaves it at.
                self.dynamic[i][j] = self.neighbour_field(
                    i, self.shown(j, current[j]), current[i]
                )
                self.fields[i] = self.field_of(i)
        for j in (i, *self.neighbours[i]):
            self.energies[j] = self.chain_energy(j)
            self.counts[j] = None
        for v in self.vias.vias_near(i):
            self.via_energies[v] = self.vias.energy(v, self.positions, current)

    def chain_step(self, i: int) -> None:
        """Chain i's burst."""
        settings = self.settings
        rest = self.rests[i]
        rest.wake(self.energies[i])
        shape = self.states[i]
        burst = _burst(
            shape.resampled,
            shape.resampled_anchors,
            self.fields[i],
            self.params,
            settings,
            self.bend_limits[i],
            self.aparts[i],
        )
        if burst.abnormal:
            self.abnormal_bursts[i] += 1
            self.abnormal_streak[i] += 1
            logger.debug(
                f"L-BFGS-B ended a burst of {self.chains[i].centreline.description} "
                f"abnormally, {self.abnormal_streak[i]} in a row"
            )
        else:
            self.abnormal_streak[i] = 0
        self.rejected_edges[i] += burst.rejected_edges
        self.too_close[i] += burst.too_close
        change = shape_change(self.shapes[i], burst.points)
        self.moved |= change > NOTHING_MOVED
        self.shapes[i] = burst.points
        self.states[i] = _Shape(
            burst.points,
            shape.resampled_anchors,
            *resample(burst.points, shape.resampled_anchors, settings.spacing),
        )
        self.reshaped(i)
        self.accept_if_best()
        rest.record(self.energies[i])
        # An abnormal burst may have stopped anywhere, so only the streak
        # settles it, not its shape change or the stall check.
        rest.settle(
            self.abnormal_streak[i] >= MAX_ABNORMAL_BURSTS
            or (
                not burst.abnormal
                # A section pressed onto its own other parts has moves to
                # reject in every burst, which no energy term pushes back.
                and burst.rejected_edges == burst.too_close
                and (change < settings.settled_change or rest.stalled(settings))
            ),
            self.energies[i],
        )

    def via_step(self, v: int) -> None:
        """Via v's burst, with the chains it carries and those pressing on it."""
        vias = self.vias
        rest = self.via_rests[v]
        contacts = vias.contacts(
            v, self.positions, self.current, self.moving_chains, self.settings
        )
        carried = {*vias.incident[v], *contacts}
        # The block's energy sums the chains it carries: with a chain joining
        # or leaving, the energies recorded before no longer compare, and a
        # via still moving would stall on them.
        forget = carried != self.carried[v]
        self.carried[v] = carried
        rest.wake(self.via_block_energy(v), forget=forget)
        # The burst starts from the chains as their own bursts left them:
        # from them resampled, it would win back the energy resampling cost,
        # which the chain's next burst gives up again, and the two blocks
        # would wake each other forever. A chain the via has stretched an
        # edge of, while it rested, starts resampled.
        starts, anchors = self.current, self.masks
        for i in (*vias.incident[v], *contacts):
            if i in self.moving and self.rests[i].settled:
                shape = self.states[i]
                starts[i], anchors[i] = shape.resampled, shape.resampled_anchors
        # A chain pressing on the via counts the two against each other in
        # the burst itself.
        fields = {
            **{
                i: self.field_of(i, without_chains=vias.incident[v])
                for i in vias.incident[v]
            },
            **{i: self.field_of(i, without_via=v) for i in contacts},
        }
        burst = via_burst(
            self.positions[v],
            vias.carried(v, contacts, starts, anchors, fields),
            vias.field(v, self.positions, self.current, contacts),
            vias.vias[v].via.diameter,
            self.params,
            self.settings,
        )
        self.via_rejected[v] += burst.rejected_edges
        change = self.take_via_move(v, burst)
        rest.record(self.via_block_energy(v))
        # A move cut short by crossings may go on; one undone whole waits
        # until something near it changes.
        rest.settle(
            (burst.rejected_edges == 0 or change == 0.0)
            and (change < self.settings.settled_change or rest.stalled(self.settings)),
            self.via_block_energy(v),
        )

    def take_via_move(self, v: int, burst: ViaBurst) -> float:
        """Move via v and its chains as `burst` left them, unless that raises
        the run energy, adds a violation to any chain, or puts the via newly
        inside its clearance. Returns the move's largest shape change, the
        via's step included, or 0 if nothing moved.

        Each chain the move changed and the via still carries or presses on
        settles at its new energy: the burst moved all its free vertices with
        the via, and a burst of its own would only repeat that work, or start
        by resampling it, which near an obstacle costs more energy than the
        burst wins back, and wake the via again. A chain the via has left
        behind is free of it, so it keeps its own bursts. A run that moved
        anything by the tolerance is never "nothing to do", however soon
        everything rests.
        """
        vias = self.vias
        settings = self.settings
        old = self.positions[v]
        changes = {
            i: shape_change(self.states[i].points, points)
            for i, points in burst.polylines.items()
        }
        change = max(
            float(np.linalg.norm(burst.position - old)), *changes.values(), 0.0
        )
        if change == 0.0:
            return 0.0
        saved = {i: self.states[i] for i in burst.polylines}
        affected = {j for i in burst.polylines for j in (i, *self.neighbours[i])} | set(
            vias.chains_near(v)
        )
        counts_before = {j: self.counted(j) for j in affected}
        energy_before = self.run_energy()
        inside_before = vias.distance(v, self.positions, self.current)
        self.positions[v] = burst.position
        for i, points in burst.polylines.items():
            shape = self.states[i]
            mask = burst.anchors[i]
            self.states[i] = (
                _Shape(points, mask, *resample(points, mask, settings.spacing))
                if i in self.moving
                else replace(shape, points=points, anchors=mask)
            )
        self.via_moved(v, burst.polylines)
        inside_after = vias.distance(v, self.positions, self.current)
        if (
            self.run_energy() > energy_before
            or any(self.counted(j) > counts_before[j] for j in affected)
            or (inside_after < -INSIDE_TOLERANCE and inside_after < inside_before)
        ):
            self.positions[v] = old
            for i, shape in saved.items():
                self.states[i] = shape
            self.via_moved(v, burst.polylines)
            return 0.0
        self.moved |= change > NOTHING_MOVED
        held = {
            *vias.incident[v],
            *vias.contacts(
                v, self.positions, self.current, self.moving_chains, settings
            ),
        }
        for i in held.intersection(changes):
            self.rests[i].settle(True, self.energies[i])
        self.accept_if_best()
        return change

    def via_moved(self, v: int, chains: Collection[int]) -> None:
        """Show via v and `chains` where they are now to everything that sees
        them, and measure the energies again."""
        for i in sorted(chains):
            self.reshaped(i)
        for j in self.vias.chains_near(v):
            self.via_parts[j] = self.vias.via_obstacles_for_chain(j, self.positions)
            self.fields[j] = self.field_of(j)
            self.energies[j] = self.chain_energy(j)
            self.counts[j] = None
        current = self.current
        for u in range(self.via_count):
            self.via_energies[u] = self.vias.energy(u, self.positions, current)


class _Remembered:
    """A field that answers a query repeating the previous one from memory.

    A chain standing still is measured again with the same points after each
    neighbour's burst, when only the neighbours' part of its field changed.
    """

    def __init__(self, field: ObstacleField) -> None:
        self.field = field
        self._distances: dict[
            float, tuple[FloatArray, tuple[FloatArray, FloatArray]]
        ] = {}
        self._clear: dict[float, tuple[FloatArray, FloatArray, BoolArray]] = {}

    def signed_distance(
        self, points: FloatArray, reach: float = math.inf
    ) -> tuple[FloatArray, FloatArray]:
        last = self._distances.get(reach)
        if last is not None and np.array_equal(last[0], points):
            return last[1]
        answer = self.field.signed_distance(points, reach)
        self._distances[reach] = (points.copy(), answer)
        return answer

    def segments_clear(
        self, starts: FloatArray, ends: FloatArray, tolerance: float = 0.0
    ) -> BoolArray:
        last = self._clear.get(tolerance)
        if (
            last is not None
            and np.array_equal(last[0], starts)
            and np.array_equal(last[1], ends)
        ):
            return last[2]
        answer = self.field.segments_clear(starts, ends, tolerance)
        self._clear[tolerance] = (starts.copy(), ends.copy(), answer)
        return answer


def frozen_vertices(
    chain: ChainInput,
    field: ObstacleField,
    dynamic: dict[int, ObstacleField | None],
    chains: Sequence[ChainInput],
    settings: OptimizerSettings,
) -> tuple[BoolArray, tuple[FrozenVertex, ...]]:
    """Free vertices starting more than δ inside the clearance, and their records."""
    points = chain.centreline.points
    distance, _ = field.signed_distance(points)
    mask = ~chain.fixed & (distance < -settings.bubble_tolerance)
    if not mask.any():
        return mask, ()
    stuck = points[mask]
    inside = [
        (
            chains[j].centreline.description,
            other.signed_distance(stuck)[0] < -settings.bubble_tolerance,
        )
        for j, other in dynamic.items()
        if other is not None
    ]
    return mask, tuple(
        FrozenVertex(
            (float(x), float(y)),
            float(d),
            tuple(name for name, near in inside if near[k]),
        )
        for k, ((x, y), d) in enumerate(zip(stuck, distance[mask], strict=True))
    )


def violations(points: FloatArray, anchors: BoolArray, field: ObstacleField) -> int:
    """Places where the polyline enters the clearance away from its anchors.

    Each corner cut, an edge with both ends clear that passes through an
    obstacle, counts once, and so does each run of consecutive vertices
    inside the clearance that holds no anchor. A run that holds a frozen
    vertex or a pin in the clearance cannot move away and is not counted.
    A run is consecutive in the chain, so inside vertices in two touching
    or overlapping obstacles form one. The count says nothing about how deep
    a violation goes, and an edge from any run to a clear vertex adds nothing
    to it whatever obstacles the edge passes through. Vertices and edges
    within INSIDE_TOLERANCE of the clearance count as clear, so a track
    routed at exactly the clearance starts with no violations.
    """
    distance, _ = field.signed_distance(points)
    inside = distance < -INSIDE_TOLERANCE
    ends_clear = ~inside[:-1] & ~inside[1:]
    cuts = np.count_nonzero(
        ~field.segments_clear(
            points[:-1][ends_clear], points[1:][ends_clear], INSIDE_TOLERANCE
        )
    )
    run_starts = inside & ~np.concatenate(([False], inside[:-1]))
    run = np.cumsum(run_starts)
    loose_runs = np.count_nonzero(run_starts & ~np.isin(run, run[inside & anchors]))
    return int(cuts + loose_runs)


def stalled(lowest_energies: list[float], settings: OptimizerSettings) -> bool:
    if len(lowest_energies) <= settings.stall_window:
        return False
    earlier = lowest_energies[-1 - settings.stall_window]
    return earlier - lowest_energies[-1] <= settings.stall_tolerance


def resample(
    points: FloatArray, fixed: BoolArray, spacing: float
) -> tuple[FloatArray, BoolArray]:
    """Resample each span between fixed vertices to uniform arc-length spacing.

    Fixed vertices are kept exactly; returns the new points and fixed mask.
    """
    anchor_indices = np.flatnonzero(fixed)
    pieces = [points[:1]]
    masks = [np.ones(1, dtype=bool)]
    for start, end in pairwise(anchor_indices):
        span = points[start : end + 1]
        arc = arc_lengths(span)
        total = float(arc[-1])
        count = max(1, int(np.ceil(total / spacing - RESAMPLE_SLACK)))
        targets = np.linspace(0.0, total, count + 1)[1:-1]
        interior = np.stack(
            [np.interp(targets, arc, span[:, 0]), np.interp(targets, arc, span[:, 1])],
            axis=1,
        )
        pieces += [interior, points[end : end + 1]]
        masks += [np.zeros(count - 1, dtype=bool), np.ones(1, dtype=bool)]
    return np.concatenate(pieces), np.concatenate(masks)


def measure(
    polyline: FloatArray, field: DistanceField, spacing: float
) -> PolylineStats:
    """Length, bend and clearance statistics of a polyline.

    The bend radius is the discrete curvature radius l/(4·tan(φ/2)) of the
    polyline resampled at `spacing` with every vertex kept, so a sharp corner
    reports a radius below the spacing however long its edges are.
    """
    edges = np.diff(polyline, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    return PolylineStats(
        length=float(lengths.sum()),
        segment_count=len(polyline) - 1,
        tightest_bend_radius=_tightest_bend_radius(
            resample(polyline, np.ones(len(polyline), dtype=bool), spacing)[0]
        ),
        **_clearance(polyline, lengths, field, spacing / MEASURE_SAMPLES_PER_SPACING),
    )


def _tightest_bend_radius(points: FloatArray) -> float:
    edges = np.diff(points, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    incoming, outgoing = edges[:-1], edges[1:]
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    turn = np.abs(np.arctan2(cross, np.einsum("ij,ij->i", incoming, outgoing)))
    bending = turn > STRAIGHT_TURN
    if not bending.any():
        return float("inf")
    span = (lengths[:-1] + lengths[1:])[bending]
    return float(np.min(span / (4 * np.tan(turn[bending] / 2))))


def _clearance(
    polyline: FloatArray, lengths: FloatArray, field: DistanceField, step: float
) -> dict[str, float]:
    counts = np.maximum(1, np.ceil(lengths / step).astype(int))
    distances: list[FloatArray] = []
    weights: list[FloatArray] = []
    for start, end, count, length in zip(
        polyline[:-1], polyline[1:], counts, lengths, strict=True
    ):
        t = np.linspace(0.0, 1.0, count + 1)
        distances.append(field.signed_distance(start + t[:, None] * (end - start))[0])
        edge_weights = np.full(count + 1, length / count)
        edge_weights[[0, -1]] /= 2
        weights.append(edge_weights)
    distance = np.concatenate(distances)
    weight = np.concatenate(weights)
    total = weight.sum()
    mean = float(distance @ weight / total) if total > 0.0 else float(distance.mean())
    return {"min_clearance": float(distance.min()), "mean_clearance": mean}


@dataclass(frozen=True)
class _ChainBurst(Burst):
    too_close: int
    """How many of `rejected_edges` were vertex pairs that came too close to
    each other (`close_to_itself`), out of a normally ended burst."""


def _burst(
    points: FloatArray,
    fixed: BoolArray,
    field: ObstacleField,
    params: EnergyParams,
    settings: OptimizerSettings,
    bend_limit: float | None = None,
    apart: float | None = None,
) -> _ChainBurst:
    """One bounded L-BFGS-B run, retried with halved bounds where edges
    crossed or, with a `bend_limit`, bent below it, or, with `apart`, where
    the polyline came that close to itself."""
    free = ~fixed
    distance, _ = field.signed_distance(points)
    clear_before = field.segments_clear(points[:-1], points[1:])
    too_close: list[int] = []

    def revert(x: FloatArray) -> tuple[BoolArray, int]:
        candidate = points.copy()
        candidate[free] = x.reshape(-1, 2)
        reverted, rejected = _revert_new_crossings(
            candidate, points, free, clear_before, field, bend_limit, apart
        )
        too_close.append(rejected.too_close)
        x[:] = candidate[free].ravel()
        return np.repeat(reverted[free], 2), rejected.total

    bounded = bounded_burst(
        free_vertex_objective(
            points, fixed, field, params, settings.short_edge, bend_limit
        ),
        points[free].ravel(),
        np.repeat(half_side(distance[free], settings), 2),
        revert,
        settings,
    )
    result = points.copy()
    result[free] = bounded.points.reshape(-1, 2)
    # An abnormally ended burst stops at its last check without counting it.
    counted = too_close[:-1] if bounded.abnormal else too_close
    return _ChainBurst(result, bounded.rejected_edges, bounded.abnormal, sum(counted))


def shape_change(before: FloatArray, after: FloatArray) -> float:
    """Local symmetric distance between two polylines, mm.

    Vertices sliding along an unchanged shape do not count: resampling to
    uniform spacing and the bending term's spreading of vertices at bends
    would otherwise keep moving vertices forever without changing the track.
    """
    return max(_local_distance(after, before), _local_distance(before, after))


def _local_distance(points: FloatArray, polyline: FloatArray) -> float:
    """Largest distance from each point to the polyline's edges near it.

    "Near" is within SHAPE_WINDOW edges of the edge at the same arc-length
    fraction, so the far leg of a hairpin never counts as close.
    """
    last_edge = len(polyline) - 2
    starts, ends = polyline[:-1], polyline[1:]
    edge_starts = _arc_fractions(polyline)[:-1]
    indices = np.searchsorted(edge_starts, _arc_fractions(points), side="right") - 1
    nearest = np.full(len(points), np.inf)
    for offset in range(-SHAPE_WINDOW, SHAPE_WINDOW + 1):
        edge = np.clip(indices + offset, 0, last_edge)
        start, delta = starts[edge], ends[edge] - starts[edge]
        squared = np.einsum("ij,ij->i", delta, delta)
        t = np.einsum("ij,ij->i", points - start, delta) / np.maximum(
            squared, MIN_EDGE_LENGTH**2
        )
        closest: FloatArray = start + np.clip(t, 0.0, 1.0)[:, None] * delta
        nearest = np.minimum(nearest, np.linalg.norm(points - closest, axis=1))
    return float(nearest.max())


def _arc_fractions(points: FloatArray) -> FloatArray:
    arc = arc_lengths(points)
    return arc / max(float(arc[-1]), MIN_EDGE_LENGTH)


def close_to_itself(points: FloatArray, apart: float) -> tuple[IntArray, FloatArray]:
    """The pairs of vertices closer than `apart` that lie more than half a
    circle `apart` across from each other along the polyline, and their
    distances: there it comes back onto itself rather than turning, as a
    turn tighter than that circle would."""
    pairs = cKDTree(points).query_pairs(apart, output_type="ndarray")
    if not len(pairs):
        return pairs, np.empty(0)
    arc = arc_lengths(points)
    i, j = pairs[:, 0], pairs[:, 1]
    beyond = np.abs(arc[i] - arc[j]) > math.pi / 2 * apart
    pairs = pairs[beyond]
    i, j = pairs[:, 0], pairs[:, 1]
    return pairs, np.hypot(*(points[i] - points[j]).T)


class _Rejected(NamedTuple):
    total: int
    """Edges that newly crossed or bent too tightly, and vertex pairs that
    came too close."""
    too_close: int


def _revert_new_crossings(
    candidate: FloatArray,
    original: FloatArray,
    free: BoolArray,
    clear_before: BoolArray,
    field: ObstacleField,
    bend_limit: float | None = None,
    apart: float | None = None,
) -> tuple[BoolArray, _Rejected]:
    """Move the free ends of newly crossing edges back, in place, until none
    cross; with a `bend_limit` the vertices that set the radius of edges
    newly bent below it, until none is; and with `apart` the vertices that
    came closer than that to the polyline's own parts beyond a turn
    (`close_to_itself`), or than either started from them where it started
    closer, but never than the two started: both vertices of such a pair.

    Returns the mask of reverted vertices and what the candidate as given
    did.
    """
    tight_before = (
        None if bend_limit is None else too_tight(bend_radii(original), bend_limit)
    )
    nearest = np.full(len(original), apart or 0.0)
    if apart is not None:
        pairs, distance = close_to_itself(original, apart)
        np.minimum.at(nearest, pairs[:, 0], distance)
        np.minimum.at(nearest, pairs[:, 1], distance)
    first: list[_Rejected] = []

    def crossings(candidate: FloatArray) -> tuple[int, BoolArray]:
        crossing = clear_before & ~field.segments_clear(candidate[:-1], candidate[1:])
        ends = np.zeros(len(candidate), dtype=bool)
        ends[:-1] |= crossing
        ends[1:] |= crossing
        count = int(crossing.sum())
        close = 0
        if bend_limit is not None and tight_before is not None:
            tight = too_tight(bend_radii(candidate), bend_limit) & ~tight_before
            around = np.flatnonzero(tight)[:, None] + BEND_REACH
            ends[np.clip(around, 0, len(candidate) - 1).ravel()] = True
            count += int(tight.sum())
        if apart is not None:
            pairs, distance = close_to_itself(candidate, apart)
            i, j = pairs[:, 0], pairs[:, 1]
            # At most the pair's own starting distance, so a pair moved back
            # whole is never too close, even one that lies beyond a turn only
            # along the candidate.
            started = np.hypot(*(original[i] - original[j]).T)
            floor = np.minimum(np.maximum(nearest[i], nearest[j]), started)
            closer = distance < floor - INSIDE_TOLERANCE
            ends[i[closer]] = ends[j[closer]] = True
            close = int(closer.sum())
            count += close
        if not first:
            first.append(_Rejected(count, close))
        return count, ends & free

    reverted, _ = revert_until_clear(candidate, original, crossings)
    return reverted, first[0]
