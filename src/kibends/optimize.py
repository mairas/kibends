"""Outer optimisation loop: resample, bubble bounds, L-BFGS-B burst, edge check.

Free vertices move only inside their bubble, a box inscribed in the free disc
around them, so they never jump over an obstacle; an exact edge check after
each burst rejects edges that newly cross an obstacle. Together they keep the
chain on the same side of every obstacle (topology preservation).

Several chains are optimised by alternating bursts: each outer iteration gives
every chain one burst in turn while the other chains on its layer stand still
as obstacles, so the same-side guarantee holds between chains too.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from itertools import pairwise
from typing import Protocol

import numpy as np
from scipy.optimize import minimize

from kibends.chain import Point2
from kibends.energy import (
    MIN_EDGE_LENGTH,
    BoolArray,
    DistanceField,
    EnergyParams,
    FloatArray,
    free_vertex_objective,
    total_energy,
)
from kibends.obstacles import ChainCentreline, CompositeField, chain_obstacles

# A span only slightly longer than a multiple of the spacing gets no extra
# vertex, so float noise in re-measured lengths does not change the count.
RESAMPLE_SLACK = 1e-6

# Below this turning angle a vertex counts as straight (infinite radius).
STRAIGHT_TURN = 1e-9

MEASURE_SAMPLES_PER_SPACING = 4

# Depth inside the clearance, mm, that the violation guard ignores as float
# noise: a track routed at exactly the clearance resamples to vertices about
# 1e-15 mm inside. Far below KiCad's 1 nm coordinate step.
INSIDE_TOLERANCE = 1e-7

# Halving the bounds of rejected vertices this many times within one outer
# iteration shrinks their step to 1/16; beyond that they wait for the next one.
MAX_BOUND_HALVINGS = 4

LBFGSB_ABNORMAL = 2

# Edges shorter than this fraction of the spacing meet the short-edge barrier.
# Resampling leaves every edge at a free vertex at least half the spacing.
SHORT_EDGE_FRACTION = 0.1

# The chain's length term is ill-conditioned (tension spreads one vertex per
# iteration), so L-BFGS-B's default stopping tolerances end a burst long before
# the shape settles; with these, burst_maxiter bounds the burst instead.
LBFGSB_FTOL = 1e-15
LBFGSB_GTOL = 1e-10

# Shape change is measured from each vertex to the other polyline's edges
# within this many edges of the same index.
SHAPE_WINDOW = 3


class ObstacleField(DistanceField, Protocol):
    """Distance field that can also check segments exactly."""

    def segments_clear(
        self, starts: FloatArray, ends: FloatArray, tolerance: float = 0.0
    ) -> BoolArray:
        """Return per segment whether it stays out of every inflated obstacle.

        starts and ends are (n, 2) arrays in mm. A segment is clear when it
        does not enter the region where the signed distance is below
        -tolerance; touching the boundary is clear.
        """
        ...


class Termination(Enum):
    CONVERGED = "converged"
    NOTHING_TO_DO = "nothing to do"
    ITERATION_CAP = "iteration cap reached"
    ABNORMAL = "L-BFGS-B terminated abnormally"


@dataclass(frozen=True)
class OptimizerSettings:
    spacing: float = 0.1
    """Target distance between resampled vertices, mm."""
    bubble_tolerance: float = 0.01
    """δ, mm: bubbles are measured to obstacles inflated by δ less than the
    clearance, and vertices deeper than δ inside an obstacle are frozen."""
    displacement_tolerance: float = 1e-2
    """Convergence threshold on a burst's largest shape change, as a
    fraction of the spacing."""
    max_iterations: int = 200
    stall_tolerance: float = 1e-3
    """Converged when the best energy improved by less than this over
    `stall_window` iterations. The energy is in mm of track length, so this
    is 1 µm of length: in flat stretches the shape keeps wandering by tens
    of µm while the energy creeps down by far less."""
    stall_window: int = 3
    burst_maxiter: int = 300
    """L-BFGS-B iterations per burst."""

    def __post_init__(self) -> None:
        if self.spacing <= 0.0:
            raise ValueError(f"spacing must be positive, got {self.spacing}")

    @property
    def short_edge(self) -> float:
        """Length below which an edge meets the energy's short-edge barrier, mm."""
        return SHORT_EDGE_FRACTION * self.spacing


DEFAULT_SETTINGS = OptimizerSettings()


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
    frozen: tuple[FrozenVertex, ...]
    before: PolylineStats
    after: PolylineStats


@dataclass(frozen=True, eq=False)
class ChainInput:
    """One chain of a run, with the field of everything but the selected chains."""

    centreline: ChainCentreline
    """Its description, net, width, clearance and starting polyline."""
    layer: str
    """Only chains on the same layer see each other."""
    fixed: BoolArray
    field: ObstacleField
    """Obstacles for its net and width, without any selected chain."""


@dataclass(frozen=True)
class ChainResult:
    polyline: FloatArray
    fixed: BoolArray
    """Pinned and frozen vertices of `polyline`; simplification keeps them."""
    rejected_edges: int
    """Edges whose burst moves were undone because they crossed an obstacle."""
    frozen: tuple[FrozenVertex, ...]
    before: PolylineStats
    after: PolylineStats
    """Measured with the other chains at their returned shapes."""


@dataclass(frozen=True)
class RunResult:
    chains: tuple[ChainResult, ...]
    """In the order the chains were given."""
    termination: Termination
    iterations: int
    """Outer iterations, each giving every chain one burst."""


@dataclass(frozen=True)
class _Burst:
    points: FloatArray
    rejected_edges: int
    abnormal: bool


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
    return OptimizeResult(
        polyline=chain.polyline,
        fixed=chain.fixed,
        termination=run.termination,
        iterations=run.iterations,
        rejected_edges=chain.rejected_edges,
        frozen=chain.frozen,
        before=chain.before,
        after=chain.after,
    )


def optimize_chains(
    chains: Sequence[ChainInput],
    params: EnergyParams,
    settings: OptimizerSettings = DEFAULT_SETTINGS,
) -> RunResult:
    """Minimise the chains' summed energy by alternating bursts, in the given order.

    Each chain's energy is taken in its own field plus the other chains on its
    layer, so every pair counts from both sides. The run keeps one best state,
    all polylines together, replaced only after a burst that does not raise
    the summed energy and leaves no chain with more violations than it started
    with; a cap or abnormal stop returns that state.

    A chain rests once it has converged or stalled on its own energy, until a
    neighbour's move changes that energy by more than the stall tolerance.
    The run converges when every chain rests.
    """
    for chain in chains:
        if not (chain.fixed[0] and chain.fixed[-1]):
            raise ValueError("both endpoints of the chain must be fixed")
    count = len(chains)
    neighbours = same_layer_neighbours(chains)
    reach = obstacle_reach(params)
    originals = [chain.centreline.points for chain in chains]
    statics: list[ObstacleField] = [
        _Remembered(chain.field) if neighbours[i] else chain.field
        for i, chain in enumerate(chains)
    ]
    dynamic = [
        {
            j: _neighbour_field(chains[i], chains[j], originals[j], reach)
            for j in neighbours[i]
        }
        for i in range(count)
    ]
    fields = [_compose(statics[i], dynamic[i]) for i in range(count)]
    anchors: list[BoolArray] = []
    frozen: list[tuple[FrozenVertex, ...]] = []
    for i, chain in enumerate(chains):
        mask, records = _frozen(chain, fields[i], dynamic[i], chains, settings)
        anchors.append(chain.fixed | mask)
        frozen.append(records)
    before = [
        measure(points, field, settings.spacing)
        for points, field in zip(originals, fields, strict=True)
    ]
    rejected_edges = [0] * count

    def finish(
        polylines: list[FloatArray],
        masks: list[BoolArray],
        termination: Termination,
        iterations: int,
    ) -> RunResult:
        unchanged = termination is Termination.NOTHING_TO_DO

        def after(i: int) -> PolylineStats:
            field = _compose(
                statics[i],
                {
                    j: _neighbour_field(chains[i], chains[j], polylines[j], reach)
                    for j in neighbours[i]
                },
            )
            return measure(polylines[i], field, settings.spacing)

        return RunResult(
            tuple(
                ChainResult(
                    polyline=polylines[i],
                    fixed=masks[i],
                    rejected_edges=rejected_edges[i],
                    frozen=frozen[i],
                    before=before[i],
                    after=before[i] if unchanged else after(i),
                )
                for i in range(count)
            ),
            termination,
            iterations,
        )

    # A burst starts from its chain's polyline resampled; the polyline as the
    # burst left it is what the energy, the guard, the best state and the
    # other chains see.
    inputs: list[FloatArray] = []
    input_masks: list[BoolArray] = []
    for original, anchor in zip(originals, anchors, strict=True):
        points, mask = resample(original, anchor, settings.spacing)
        inputs.append(points)
        input_masks.append(mask)
    moving = [i for i in range(count) if not input_masks[i].all()]
    if not moving:
        return finish(originals, anchors, Termination.NOTHING_TO_DO, 0)
    current = [inputs[i] if i in moving else originals[i] for i in range(count)]
    masks = [input_masks[i] if i in moving else anchors[i] for i in range(count)]

    # Resampling can lay a chord across an obstacle corner or put a vertex
    # inside one, and a burst only rejects crossings its own moves create, so
    # the result is limited to polylines with no more violations than the
    # resampled input. Counting rather than locating them lets the input's own
    # violations move as the chain tightens.
    allowed = [violations(current[i], masks[i], fields[i]) for i in range(count)]
    counts: list[int | None] = list(allowed)
    energies = [
        total_energy(current[i], masks[i], fields[i], params, settings.short_edge)[0]
        for i in range(count)
    ]
    best, best_masks = list(current), list(masks)
    best_energy = sum(energies)

    def within_allowed(i: int) -> bool:
        counted = counts[i]
        if counted is None:
            counted = counts[i] = violations(current[i], masks[i], fields[i])
        return counted <= allowed[i]

    # The stall check follows every burst, not just the ones the guard
    # accepts, or a briefly cut corner would end the run early.
    lowest_energies = [[energy] for energy in energies]
    settled = [False] * count
    settled_once = [False] * count
    settled_energy = list(energies)

    def resting(i: int) -> bool:
        return (
            settled[i]
            and abs(energies[i] - settled_energy[i]) <= settings.stall_tolerance
        )

    shapes = list(originals)
    termination = Termination.ITERATION_CAP
    iteration = 0
    for iteration in range(1, settings.max_iterations + 1):
        abnormal = False
        for i in moving:
            if resting(i):
                continue
            burst = _burst(inputs[i], input_masks[i], fields[i], params, settings)
            if burst.abnormal and settled_once[i]:
                # It had already converged; its neighbours' moves leave it
                # where it was.
                settled[i] = True
                settled_energy[i] = energies[i]
                continue
            rejected_edges[i] += burst.rejected_edges
            shape_change = _shape_change(shapes[i], burst.points)
            shapes[i] = current[i] = burst.points
            masks[i] = input_masks[i]
            for j in neighbours[i]:
                dynamic[j][i] = _neighbour_field(
                    chains[j], chains[i], current[i], reach
                )
                fields[j] = _compose(statics[j], dynamic[j])
            for j in (i, *neighbours[i]):
                energies[j] = total_energy(
                    current[j], masks[j], fields[j], params, settings.short_edge
                )[0]
                counts[j] = None
            energy = sum(energies)
            if energy <= best_energy and all(within_allowed(j) for j in range(count)):
                best, best_masks, best_energy = list(current), list(masks), energy
            lowest_energies[i].append(min(lowest_energies[i][-1], energies[i]))
            settled[i] = burst.rejected_edges == 0 and (
                shape_change < settings.displacement_tolerance * settings.spacing
                or stalled(lowest_energies[i], settings)
            )
            settled_once[i] |= settled[i]
            settled_energy[i] = energies[i]
            if burst.abnormal:
                abnormal = True
                break
            inputs[i], input_masks[i] = resample(
                burst.points, masks[i], settings.spacing
            )
        if abnormal:
            termination = Termination.ABNORMAL
            break
        if all(resting(i) for i in moving):
            if iteration == 1:
                return finish(originals, anchors, Termination.NOTHING_TO_DO, 1)
            termination = Termination.CONVERGED
            break
    return finish(best, best_masks, termination, iteration)


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


def _compose(
    static: ObstacleField, dynamic: dict[int, ObstacleField | None]
) -> ObstacleField:
    others = [field for field in dynamic.values() if field is not None]
    return CompositeField([static, *others]) if others else static


def obstacle_reach(params: EnergyParams) -> float:
    """Distance from an obstacle beyond which both obstacle terms of the
    energy are zero, mm."""
    return max(params.standoff, params.falloff_range if params.repulsion else 0.0)


def same_layer_neighbours(chains: Sequence[ChainInput]) -> list[list[int]]:
    """For each chain, the indices of the other chains on its layer."""
    return [
        [j for j, other in enumerate(chains) if j != i and other.layer == chain.layer]
        for i, chain in enumerate(chains)
    ]


def _neighbour_field(
    own: ChainInput, other: ChainInput, points: FloatArray, reach: float
) -> ObstacleField | None:
    """`other` at `points` as an obstacle to `own`, or None if nothing is left."""
    centreline = neighbour_centreline(own, other, points, reach)
    if centreline is None:
        return None
    me = own.centreline
    return chain_obstacles(me.net, me.width, me.clearance, [centreline])


def neighbour_centreline(
    own: ChainInput, other: ChainInput, points: FloatArray, reach: float
) -> ChainCentreline | None:
    """The part of `other` at `points` that is an obstacle to `own`, or None
    if nothing is left.

    A chain of the same net that ends where this one ends continues it, as at
    a neck-down: near that junction both must leave the shared point, so the
    other is left out within its inflation plus `reach` of it.
    """
    me, them = own.centreline, other.centreline
    if them.net == me.net:
        cut = (me.width + them.width) / 2 + reach
        ends = me.points[[0, -1]]

        def shared(point: FloatArray) -> bool:
            return bool(np.any(np.all(ends == point, axis=1)))

        total = float(_arc_lengths(points)[-1])
        start = cut if shared(them.points[0]) else 0.0
        end = total - cut if shared(them.points[-1]) else total
        if start >= end:
            return None
        points = _cut(points, start, end)
    return replace(them, points=points)


def _cut(points: FloatArray, start: float, end: float) -> FloatArray:
    """The part of a polyline between two arc lengths from its first vertex."""
    arc = _arc_lengths(points)
    inside = (arc > start) & (arc < end)
    ends = np.stack(
        [
            np.interp([start, end], arc, points[:, 0]),
            np.interp([start, end], arc, points[:, 1]),
        ],
        axis=1,
    )
    return np.concatenate([ends[:1], points[inside], ends[1:]])


def _frozen(
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
        arc = _arc_lengths(span)
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


def _burst(
    points: FloatArray,
    fixed: BoolArray,
    field: ObstacleField,
    params: EnergyParams,
    settings: OptimizerSettings,
) -> _Burst:
    """One bounded L-BFGS-B run, retried with halved bounds where edges crossed."""
    free = ~fixed
    distance, _ = field.signed_distance(points)
    bubble = np.maximum(distance[free] + settings.bubble_tolerance, 0.0) / 2
    half_side = bubble / np.sqrt(2)
    clear_before = field.segments_clear(points[:-1], points[1:])
    start = points[free]
    scale = np.ones(free.sum())
    current = points.copy()
    rejected_edges = 0
    abnormal = False
    for _ in range(MAX_BOUND_HALVINGS + 1):
        reach = (half_side * scale)[:, None]
        bounds = list(
            zip(
                (start - reach).ravel(),
                (start + reach).ravel(),
                strict=True,
            )
        )
        result = minimize(
            free_vertex_objective(current, fixed, field, params, settings.short_edge),
            current[free].ravel(),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": settings.burst_maxiter,
                "ftol": LBFGSB_FTOL,
                "gtol": LBFGSB_GTOL,
            },
        )
        abnormal = result.status == LBFGSB_ABNORMAL
        candidate = current.copy()
        candidate[free] = result.x.reshape(-1, 2)
        reverted, crossings = _revert_new_crossings(
            candidate, points, free, clear_before, field
        )
        current = candidate
        if crossings == 0 or abnormal:
            break
        rejected_edges += crossings
        scale[reverted[free]] /= 2
    return _Burst(current, rejected_edges, abnormal)


def _shape_change(before: FloatArray, after: FloatArray) -> float:
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
    arc = _arc_lengths(points)
    return arc / max(float(arc[-1]), MIN_EDGE_LENGTH)


def _arc_lengths(points: FloatArray) -> FloatArray:
    """Cumulative arc length at each vertex, starting at 0."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate((np.zeros(1), np.cumsum(steps)))


def _revert_new_crossings(
    candidate: FloatArray,
    original: FloatArray,
    free: BoolArray,
    clear_before: BoolArray,
    field: ObstacleField,
) -> tuple[BoolArray, int]:
    """Move the free ends of newly crossing edges back, in place, until none cross.

    Returns the mask of reverted vertices and the number of edges that crossed
    in the candidate as given. Reverting can make a neighbouring
    edge cross, hence the loop; it ends because fully reverted edges are the
    original, clear ones.
    """
    reverted = np.zeros(len(candidate), dtype=bool)
    initial_crossings: int | None = None
    while True:
        crossing = clear_before & ~field.segments_clear(candidate[:-1], candidate[1:])
        if initial_crossings is None:
            initial_crossings = int(crossing.sum())
        if not crossing.any():
            return reverted, initial_crossings
        ends = np.zeros(len(candidate), dtype=bool)
        ends[:-1] |= crossing
        ends[1:] |= crossing
        ends &= free & ~reverted
        candidate[ends] = original[ends]
        reverted |= ends
