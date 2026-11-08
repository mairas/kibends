"""Joint optimisation: every chain of a layer as one L-BFGS-B problem.

The energy is the alternating loop's run energy, the sum of each chain's
energy in its static field plus the other chains on its layer, taken as one
function of every free vertex of the layer: where a quadrature point of one
chain is nearest another chain's edge, the obstacle term's gradient also
reaches that edge's two vertices. Without moving vias the layers do not
interact, so each layer is a problem of its own, and an outer iteration
gives every problem that has not settled one burst: resample, bubbles,
bounded L-BFGS-B, edge check, as in `kibends.optimize`.

Not yet handled: coupled pair sections, moving vias, and two chains of one
net on a layer, which are refused.

Why the chains keep their topology
----------------------------------
Static copper is kept as the loop keeps it: each free vertex moves inside a
box inscribed in the disc of half its free distance, so it never jumps an
obstacle, and every edge that newly crosses one after a run moves its free
ends back, with halved bounds for the retry (`bursts.bounded_burst`).

The other chains move in the same burst. A vertex p at free distance D from
them, its distance to their centrelines less the inflation I, has a bubble
of CROSS_SHARE times D: it moves at most D/4 + δ/2 for a share of 1/2 and the
bubble tolerance δ, and not at all below D = -2δ. For p to pass through
another chain's edge and come out clear beyond it, p must move at least
D + 2I relative to that edge. The edge's ends start within D + I + l of p,
l its length, so they are at most D + l from p's chain and move at most
(D + l)/4 + δ/2 each. Together that is at most D/2 + l/4 + δ, less than
D + 2I for every D ≥ -2δ whenever l < 8(I - δ); resampled edges are at
most one spacing long, far below that for any real track. So no vertex
passes through another chain's edge within a burst, nor an edge across
another chain's vertex: the bubbles alone rule out a swap, which the check
at the end could not see, as both chains end clear.

What the bubbles allow is coming too close: ending inside another chain's
clearance, or an edge ending across one. The check after each run catches
that. An edge clear of the other chains at the burst's start that is not
clear of their new shapes moves its free ends back, and the check repeats,
as moving one chain's vertex back can put another chain's moved edge
across it; moved back whole, every chain is at its start, clear. An edge
that starts inside another chain's clearance is not checked, as in the
loop: its ends lie within δ (deeper ones are frozen), so they barely move,
and the violation guard on the best state counts it.
"""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from loguru import logger

from kibends.bursts import (
    DEFAULT_SETTINGS,
    OptimizerSettings,
    bounded_burst,
    half_side,
    revert_until_clear,
)
from kibends.energy import (
    BoolArray,
    EnergyParams,
    FloatArray,
    IntArray,
    ObstacleField,
    edge_integral,
    obstacle_integrand,
    obstacle_reach,
    quadrature_points,
    shape_energy,
)
from kibends.neighbours import ChainInput, chain_field, neighbour_field
from kibends.obstacles import ChainField
from kibends.optimize import (
    MAX_ABNORMAL_BURSTS,
    NOTHING_MOVED,
    PROGRESS_INTERVAL,
    ChainResult,
    FrozenVertex,
    PolylineStats,
    RunResult,
    Termination,
    frozen_vertices,
    measure,
    resample,
    shape_change,
    stalled,
    violations,
)

# The share of its free distance to the other chains, which move in the same
# burst, that a vertex's bubble takes: at 1/2 two vertices closing on each
# other cover at most half the distance between them.
CROSS_SHARE = 0.5


def optimize_jointly(
    chains: Sequence[ChainInput],
    params: EnergyParams,
    settings: OptimizerSettings = DEFAULT_SETTINGS,
    *,
    clock: Callable[[], float] = time.monotonic,
    started: float | None = None,
) -> RunResult:
    """Minimise the chains' summed energy, each layer's chains jointly.

    Takes and returns what `optimize.optimize_chains` does for plain chains.
    Refuses two chains of one net on a layer. Each problem keeps its best
    state, replaced only after a burst that does not raise its energy and
    leaves no chain with more violations than it started with.

    A problem settles when its burst moved no shape by `settled_change` and
    no edge moved back, or when its best energy improved by less than
    `stall_tolerance` over `stall_window` bursts, or after
    MAX_ABNORMAL_BURSTS abnormal bursts in a row; the run converges when
    every problem has. With a time limit in `settings`, the run stops after
    the first burst that ends when `clock` reads the limit or more past
    `started`, by default when the run starts.

    Each outer iteration logs its progress at DEBUG, and at INFO once
    `PROGRESS_INTERVAL` has passed since `started` or the last INFO line.
    """
    for chain in chains:
        if not (chain.fixed[0] and chain.fixed[-1]):
            raise ValueError("both endpoints of the chain must be fixed")
    _refuse_shared_nets(chains)
    layers = list(dict.fromkeys(chain.layer for chain in chains))
    indices = [
        [i for i, c in enumerate(chains) if c.layer == layer] for layer in layers
    ]
    problems = [
        _Problem(layer, [chains[i] for i in members], params, settings)
        for layer, members in zip(layers, indices, strict=True)
    ]
    moving = [p for p in problems if p.moving]

    def result(termination: Termination, iterations: int, *, best: bool) -> RunResult:
        by_index = {
            i: problem.result(k, best=best)
            for problem, members in zip(problems, indices, strict=True)
            for k, i in enumerate(members)
        }
        return RunResult(
            tuple(by_index[i] for i in range(len(chains))),
            termination,
            iterations,
            gave_up=tuple(
                (p.description, p.abnormal_streak)
                for p in problems
                if p.abnormal_streak >= MAX_ABNORMAL_BURSTS
            ),
            abnormal=tuple(
                (p.description, p.abnormal_bursts)
                for p in problems
                if p.abnormal_bursts
            ),
        )

    if not moving:
        return result(Termination.NOTHING_TO_DO, 0, best=False)
    begun = clock()
    origin = begun if started is None else started
    deadline = None if settings.time_limit is None else origin + settings.time_limit
    last_shown = begun
    termination = Termination.ITERATION_CAP
    iteration = 0
    for iteration in range(1, settings.max_iterations + 1):
        out_of_time = False
        change = 0.0
        for problem in moving:
            if problem.settled:
                continue
            change = max(change, problem.step())
            if deadline is not None and clock() >= deadline:
                out_of_time = True
                break
        now = clock()
        message = _progress(moving, iteration, now - origin, change)
        if now - last_shown >= PROGRESS_INTERVAL:
            logger.info(message)
            last_shown = now
        else:
            logger.debug(message)
        if all(p.settled for p in moving):
            if iteration == 1 and not any(p.moved for p in moving):
                return result(Termination.NOTHING_TO_DO, 1, best=False)
            termination = Termination.CONVERGED
            break
        if out_of_time:
            termination = Termination.TIME_LIMIT
            break
    final = result(termination, iteration, best=True)
    if final.gave_up and sum(p.best_energy for p in moving) >= sum(
        p.start_energy for p in moving
    ):
        return RunResult(
            final.chains,
            Termination.ABNORMAL,
            final.iterations,
            gave_up=final.gave_up,
            abnormal=final.abnormal,
        )
    return final


def _refuse_shared_nets(chains: Sequence[ChainInput]) -> None:
    seen: dict[tuple[str, str], ChainInput] = {}
    for chain in chains:
        key = (chain.layer, chain.centreline.net)
        first = seen.setdefault(key, chain)
        if first is not chain:
            raise ValueError(
                "joint optimization does not handle two chains of one net on a "
                f"layer yet: {first.centreline.description} and "
                f"{chain.centreline.description} are both net "
                f"{chain.centreline.net} on {chain.layer}"
            )


def _progress(
    problems: Sequence[_Problem], iteration: int, elapsed: float, change: float
) -> str:
    """One line on where the run stands after an outer iteration."""
    return (
        f"Iteration {iteration}, {elapsed:.1f} s: "
        f"length {sum(p.length() for p in problems):.3f} mm, "
        f"energy {sum(p.energy for p in problems):.3f} "
        f"(best {sum(p.best_energy for p in problems):.3f}), "
        f"largest change {change:.2e} mm"
    )


def layer_energy(
    chains: Sequence[ChainInput],
    polylines: Sequence[FloatArray],
    masks: Sequence[BoolArray],
    params: EnergyParams,
    short_edge: float,
) -> tuple[float, list[FloatArray]]:
    """The summed energy of chains of one layer, each in its static field
    plus the others at `polylines`, and its gradient at every vertex of
    every chain, with the rows of `masks` zeroed; raises ValueError if it
    is not finite."""
    reach = obstacle_reach(params)
    integrand = obstacle_integrand(params)
    k = params.points_per_edge
    first = np.cumsum([0, *(len(p) for p in polylines)])
    gradient = np.zeros((int(first[-1]), 2))
    value = 0.0
    for i, (chain, points) in enumerate(zip(chains, polylines, strict=True)):
        queries = quadrature_points(points, k)
        static, static_direction = chain.field.signed_distance(queries, reach)
        others = _others(chains, polylines, first, i)
        near = others.field.nearest(queries, reach)
        nearer = near.distance < static
        distance = np.where(nearer, near.distance, static)
        direction = np.where(nearer[:, None], near.gradient, static_direction)
        obstacle, obstacle_gradient, slopes = edge_integral(
            points, distance, direction, integrand, k
        )
        shape, shape_gradient = shape_energy(points, masks[i], params, short_edge)
        value += shape + obstacle
        gradient[first[i] : first[i + 1]] += shape_gradient + obstacle_gradient
        pulled = np.flatnonzero(nearer & (slopes != 0.0))
        if len(pulled):
            edge = near.edge[pulled]
            start = others.vertex[edge]
            t = _parameters(queries[pulled], others.starts[edge], others.ends[edge])
            push = slopes[pulled, None] * near.gradient[pulled]
            np.add.at(gradient, start, -push * (1 - t)[:, None])
            np.add.at(gradient, start + 1, -push * t[:, None])
    gradients = [
        gradient[first[i] : first[i + 1]].copy() for i in range(len(polylines))
    ]
    for g, mask in zip(gradients, masks, strict=True):
        g[mask] = 0.0
    if not (math.isfinite(value) and np.all(np.isfinite(gradient))):
        raise ValueError(f"energy is not finite: {value}")
    return value, gradients


@dataclass(frozen=True)
class _Others:
    """The other chains of a layer as obstacles to one of them."""

    field: ChainField
    starts: FloatArray
    ends: FloatArray
    vertex: IntArray
    """Each edge's first vertex, numbered across every chain of the layer."""


def _others(
    chains: Sequence[ChainInput],
    polylines: Sequence[FloatArray],
    first: IntArray,
    i: int,
) -> _Others:
    """Every chain but i at `polylines`, each edge inflated as
    `obstacles.chain_obstacles` inflates another net's chain."""
    me = chains[i].centreline
    starts: list[FloatArray] = [np.empty((0, 2))]
    ends: list[FloatArray] = [np.empty((0, 2))]
    inflations: list[FloatArray] = [np.empty(0)]
    vertices: list[IntArray] = [np.empty(0, dtype=np.intp)]
    for j, (chain, points) in enumerate(zip(chains, polylines, strict=True)):
        if j == i:
            continue
        them = chain.centreline
        inflation = (me.width + them.width) / 2 + max(me.clearance, them.clearance)
        starts.append(points[:-1])
        ends.append(points[1:])
        inflations.append(np.full(len(points) - 1, inflation))
        vertices.append(first[j] + np.arange(len(points) - 1, dtype=np.intp))
    a, b = np.concatenate(starts), np.concatenate(ends)
    return _Others(
        ChainField(a, b, np.concatenate(inflations)), a, b, np.concatenate(vertices)
    )


def _parameters(points: FloatArray, starts: FloatArray, ends: FloatArray) -> FloatArray:
    """Where along each segment from starts[i] to ends[i] its point nearest
    points[i] lies, from 0 to 1."""
    along = ends - starts
    squared = np.einsum("ij,ij->i", along, along)
    projection = np.einsum("ij,ij->i", points - starts, along)
    return np.clip(projection / np.where(squared > 0.0, squared, 1.0), 0.0, 1.0)


@dataclass(frozen=True)
class _JointBurst:
    polylines: list[FloatArray]
    rejected_edges: list[int]
    """Per chain, its edges whose moves were undone, over every retry."""
    abnormal: bool


class _Problem:
    """One layer's chains, optimised together: their current shapes, the
    best state so far, and whether the problem has settled."""

    def __init__(
        self,
        layer: str,
        chains: Sequence[ChainInput],
        params: EnergyParams,
        settings: OptimizerSettings,
    ) -> None:
        self.description = f"the chains on {layer}"
        self.chains = chains
        self.params = params
        self.settings = settings
        self.reach = obstacle_reach(params)
        self.originals = [chain.centreline.points for chain in chains]
        start_fields = self.fields(self.originals)
        self.anchors: list[BoolArray] = []
        self.frozen: list[tuple[FrozenVertex, ...]] = []
        for i, chain in enumerate(chains):
            others: dict[int, ObstacleField | None] = {
                j: neighbour_field(chain, [(chains[j], self.originals[j])], self.reach)
                for j in range(len(chains))
                if j != i
            }
            mask, records = frozen_vertices(
                chain, start_fields[i], others, chains, settings
            )
            self.anchors.append(chain.fixed | mask)
            self.frozen.append(records)
        self.before: list[PolylineStats] = [
            measure(points, field, settings.spacing)
            for points, field in zip(self.originals, start_fields, strict=True)
        ]
        # A burst starts from the chains resampled; the shapes as the burst
        # left them are what the energy, the guard and the best state see.
        self.points: list[FloatArray] = []
        self.masks: list[BoolArray] = []
        for original, anchors in zip(self.originals, self.anchors, strict=True):
            points, mask = resample(original, anchors, settings.spacing)
            moves = not mask.all()
            self.points.append(points if moves else original)
            self.masks.append(mask if moves else anchors)
        self.free = [i for i, mask in enumerate(self.masks) if not mask.all()]
        self.moving = bool(self.free)
        # Resampling can lay a chord across an obstacle corner, and a burst
        # only rejects crossings its own moves create, so the result is
        # limited to no more violations per chain than the resampled start.
        self.allowed = self.violations(self.points)
        self.energy = self.energy_of(self.points)
        self.start_energy = self.best_energy = self.energy
        self.best = list(self.points)
        self.best_masks = list(self.masks)
        self.best_energies = [self.best_energy]
        self.rejected_edges = [0] * len(chains)
        self.abnormal_bursts = 0
        self.abnormal_streak = 0
        self.settled = not self.moving
        self.moved = False

    def fields(self, polylines: Sequence[FloatArray]) -> list[ObstacleField]:
        """Each chain's field with the others at `polylines`."""
        return [
            chain_field(
                chain.field,
                [
                    neighbour_field(
                        chain,
                        [
                            (other, polylines[j])
                            for j, other in enumerate(self.chains)
                            if j != i
                        ],
                        self.reach,
                        polylines[i],
                    )
                ],
            )
            for i, chain in enumerate(self.chains)
        ]

    def violations(self, polylines: Sequence[FloatArray]) -> list[int]:
        return [
            violations(points, mask, field)
            for points, mask, field in zip(
                polylines, self.masks, self.fields(polylines), strict=True
            )
        ]

    def energy_of(self, polylines: Sequence[FloatArray]) -> float:
        return layer_energy(
            self.chains, polylines, self.masks, self.params, self.settings.short_edge
        )[0]

    def length(self) -> float:
        return sum(
            float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            for points in self.points
        )

    def result(self, k: int, *, best: bool) -> ChainResult:
        """Chain k's best state, with `best`, else its input."""
        return ChainResult(
            polyline=self.best[k] if best else self.originals[k],
            fixed=self.best_masks[k] if best else self.anchors[k],
            rejected_edges=self.rejected_edges[k],
            abnormal_bursts=0,
            frozen=self.frozen[k],
            before=self.before[k],
        )

    def step(self) -> float:
        """One burst of every chain together; returns its largest shape change."""
        settings = self.settings
        starts: list[FloatArray] = []
        masks: list[BoolArray] = []
        for i, (points, mask) in enumerate(zip(self.points, self.masks, strict=True)):
            if i in self.free:
                points, mask = resample(points, mask, settings.spacing)
            starts.append(points)
            masks.append(mask)
        self.masks = masks
        burst = self.burst(starts)
        change = max(
            shape_change(self.points[i], burst.polylines[i]) for i in self.free
        )
        self.moved |= change > NOTHING_MOVED
        self.points = burst.polylines
        for i, count in enumerate(burst.rejected_edges):
            self.rejected_edges[i] += count
        self.energy = self.energy_of(self.points)
        if self.energy <= self.best_energy and all(
            count <= allowed
            for count, allowed in zip(
                self.violations(self.points), self.allowed, strict=True
            )
        ):
            self.best, self.best_masks = list(self.points), list(self.masks)
            self.best_energy = self.energy
        self.best_energies.append(self.best_energy)
        if burst.abnormal:
            self.abnormal_bursts += 1
            self.abnormal_streak += 1
            logger.debug(
                f"L-BFGS-B ended a burst of {self.description} abnormally, "
                f"{self.abnormal_streak} in a row"
            )
        else:
            self.abnormal_streak = 0
        # An abnormal burst may have stopped anywhere, so only the streak
        # settles it. A burst cut short by moves back changed little because
        # of them, not because the shapes settled.
        self.settled = self.abnormal_streak >= MAX_ABNORMAL_BURSTS or (
            not burst.abnormal
            and (
                (not any(burst.rejected_edges) and change < settings.settled_change)
                or stalled(self.best_energies, settings)
            )
        )
        return change

    def burst(self, starts: list[FloatArray]) -> _JointBurst:
        """One bounded L-BFGS-B run over every free vertex of the layer,
        retried with halved bounds where edges newly crossed static copper
        or another chain's new shape."""
        masks = self.masks
        free = [~mask for mask in masks]
        sizes = np.cumsum([0, *(2 * int(f.sum()) for f in free)])
        first = np.cumsum([0, *(len(s) for s in starts)])

        def unflatten(x: FloatArray) -> list[FloatArray]:
            polylines: list[FloatArray] = []
            for i, start in enumerate(starts):
                points = start.copy()
                points[free[i]] = x[sizes[i] : sizes[i + 1]].reshape(-1, 2)
                polylines.append(points)
            return polylines

        def objective(x: FloatArray) -> tuple[float, FloatArray]:
            value, gradients = layer_energy(
                self.chains, unflatten(x), masks, self.params, self.settings.short_edge
            )
            return value, np.concatenate(
                [g[f].ravel() for g, f in zip(gradients, free, strict=True)]
            )

        x0 = np.concatenate([s[f].ravel() for s, f in zip(starts, free, strict=True)])
        static_before = [
            chain.field.segments_clear(s[:-1], s[1:])
            for chain, s in zip(self.chains, starts, strict=True)
        ]
        others_before = [
            _others(self.chains, starts, first, i).field.segments_clear(s[:-1], s[1:])
            for i, s in enumerate(starts)
        ]

        def crossings(x: FloatArray) -> tuple[list[int], BoolArray]:
            """Per chain, its edges newly crossing static copper or another
            chain's shape at `x`, and the coordinates of their free ends."""
            polylines = unflatten(x)
            back = np.zeros(len(x), dtype=bool)
            counts: list[int] = []
            for i, points in enumerate(polylines):
                a, b = points[:-1], points[1:]
                others = _others(self.chains, polylines, first, i).field
                crossing = (
                    static_before[i] & ~self.chains[i].field.segments_clear(a, b)
                ) | (others_before[i] & ~others.segments_clear(a, b))
                counts.append(int(crossing.sum()))
                ends = np.zeros(len(points), dtype=bool)
                ends[:-1] |= crossing
                ends[1:] |= crossing
                back[sizes[i] : sizes[i + 1]] = np.repeat(ends[free[i]], 2)
            return counts, back

        rounds: list[list[int]] = []

        def revert(x: FloatArray) -> tuple[BoolArray, int]:
            found: list[list[int]] = []

            def check(candidate: FloatArray) -> tuple[int, BoolArray]:
                counts, back = crossings(candidate)
                if not found:
                    found.append(counts)
                return sum(counts), back

            reverted, count = revert_until_clear(x, x0, check)
            rounds.append(found[0])
            return reverted, count

        sides = np.repeat(self.half_sides(starts, free, first), 2)
        result = bounded_burst(objective, x0, sides, revert, self.settings)
        # An abnormally ended burst stops at its last check without counting it.
        counted = rounds[:-1] if result.abnormal else rounds
        rejected = [sum(r[i] for r in counted) for i in range(len(starts))]
        return _JointBurst(unflatten(result.points), rejected, result.abnormal)

    def half_sides(
        self,
        starts: Sequence[FloatArray],
        free: Sequence[BoolArray],
        first: IntArray,
    ) -> FloatArray:
        """Per free vertex, the half side of its box: that of the smaller of
        its free distance to static copper and CROSS_SHARE of its free
        distance to the other chains."""
        sides: list[FloatArray] = []
        for i, (chain, points) in enumerate(zip(self.chains, starts, strict=True)):
            at = points[free[i]]
            static, _ = chain.field.signed_distance(at)
            cross, _ = _others(self.chains, starts, first, i).field.signed_distance(at)
            sides.append(
                half_side(np.minimum(static, CROSS_SHARE * cross), self.settings)
            )
        return np.concatenate(sides)
