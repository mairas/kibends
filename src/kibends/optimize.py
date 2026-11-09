"""The optimizer: every chain, coupled section and moving via of a problem
as one L-BFGS-B problem.

The energy of a problem is the sum of each chain's energy in its static
field plus the other chains on its layer (`energy.total_energy` with those
chains in its field), taken as one function of every free vertex: where a
quadrature point of one chain is nearest another chain's edge, the obstacle
term's gradient also reaches that edge's two vertices. Without moving vias
the layers do not interact, so each layer is a problem of its own, and an
outer iteration gives every problem that has not settled one burst:
resample each chain to the spacing between its fixed vertices and those
the others hold still (`_Problem.pressed`), bound each free vertex to its
box, run bounded L-BFGS-B (`bursts.bounded_burst`), and move back what the
edge check rejects.

Chains of one net on a layer see each other as `neighbours.neighbour_field`
shows them: inflated by their half widths alone, as they may touch, and
left out near where their copper touches at fixed points
(`neighbours.kept_spans`). A selection that branches is split at each
junction into chains that end there, loose (`junctions`): those ends are
one shared variable, and the branches leave each other out near it as at
any touch.

A coupled section of a differential pair (`kibends.pair_blocks`) is a
member of its layer's problem as its centreline: its free vertices are
variables, its energy has the bend-radius hinge, and it keeps clear of the
others as the pair's outer extent, under either of its nets near its own
fan-outs (`PairBlock.as_net`). The others see its two tracks, offset from
the centreline at each evaluation (`shown`), so where the obstacle term
pulls on a track's edge, the gradient goes back through the offset to the
centreline's vertices (`PairBlock.offset_pullback`). Besides the edge
check, a burst moves back what `optimize.SelfCheck` finds: a centreline
newly bent below its limit, or brought closer to its own parts beyond a
turn than it may. A held section is fully pinned and not resampled; a run
given an earlier one's result resumes from it (`optimize_chains`), which
is how `run` repeats a run with the sections that ended too tight held.

A moving via (`kibends.via_moves`) is two more variables, its centre. The
ends of the chains it joins are not variables of their own: each is the
via's centre plus its offset at the start, at every evaluation, so they keep
their offsets exactly and their gradient goes to the via. Its own obstacle
terms are its ring's, standing for a track as long as it is wide, against
its static field over every layer it spans, the members near it on those
layers and the other vias, with their gradient also at the edge or via that
answers it; the members see it as a disc on each of its layers, so that
where the obstacle term pulls on it, the gradient reaches its centre too. A
section and a via see each other as the section's centreline as wide as
`PairBlock.chain`. A moving via ties the layers it spans into one problem,
every member on them included, since moving vias are through vias and share
every layer. A via that starts more than δ inside its clearance stands
still, as copper the members see; its own terms then could only push them a
second time, so they are left out, and it ties no layers together.

How much of a same-net chain is left out near a touch depends on the
angles both chains leave it at. A burst fixes it, as pieces of the other
chain's edges, from the shapes the burst starts with, which keeps the
energy smooth within the burst; the next burst measures it again. That
errs safe: an angle that closes during the burst meets the other chain's
copper sooner than the start's leave-out expects, which repels and, at
worst, moves edges back; one that opens leaves out more than needed of
copper that is the chain's own net and may touch. The energy that judges
the best state measures the leave-out at the shapes it judges.

Why the chains keep their topology
----------------------------------
Static copper is kept as for a chain alone: each free vertex moves inside a
box inscribed in the disc of half its free distance, so it never jumps an
obstacle, and every edge that newly crosses one after a run moves its free
ends back, with halved bounds for the retry (`bursts.bounded_burst`). A
junction moves inside the smallest of its branches' boxes, and moves back
when any of their edges at it does.

The other chains move in the same burst. A vertex p at free distance D from
them, its distance to their centrelines less the inflation I, has a bubble
of CROSS_SHARE times D, or the step floor c where that is larger: it moves
at most max(D/4 + δ/2, c) for a share of 1/2 and the bubble tolerance δ. c
is STEP_FLOOR, a half, of the smallest inflation I_min between any two
members or vias of the problem, less δ (`_Problem.step_floor`), so
2c ≤ I - δ for any two of them; only a vertex whose edges all start clear
of the others takes it, and one deeper than δ inside does not move at all.
Every point of an edge moves in a straight line, no farther than the
farther of the edge's ends. For a point a of one chain to pass through
another chain, it must meet a point b of it on the way, so the two together
moved at least R, their distance at the burst's start. Where either of
their edges starts clear of the other chain, R ≥ I; where neither does,
neither takes the floor, and the last paragraph covers them. Two floors
move at most 2c < R. A vertex on its bubble lies within R + l of
the other point, l its edge's length, so its free distance is at most
R + l - I and it moves at most (R + l - I)/4 + δ/2: with a floor, less than
R whenever l < 2I, and with another bubble whenever l < 2I - 2δ. Resampled
edges are at most one spacing long, below that for any real track. So no
point of a chain passes through another within a burst: the boxes alone
rule out a swap, which the check at the end could not see, as both chains
end clear. Between chains of one net I is their half widths, still above
the spacing for real tracks; the parts left out near a touch are not seen
at all, as there the chains connect and may overlap. A junction's branches
take neither bubble nor floor against each other, nor count towards I_min
(`_Problem.siblings`): one net, they connect whichever side of each other
they end up on, so only the edge check keeps them from overlapping.

A section moves as its centreline and is seen as its two tracks, which lie
within H of it, H its half pitch widened at mitred corners, and every point
of the centreline lies within H of each track. A track's vertex can move
farther than the centreline's vertex it is offset from, as the mitre turns
with the edges either side, so the boxes do not bound the tracks' moves;
the argument needs no such bound, as it is about the centreline. Another
member that ends on the far side of a section without crossing its
centreline lies within H of it, inside the extent I_s the section keeps
clear of it, which is more than H, and the section's own edge check sees it.
So it is enough that no point of another member meets the centreline. A
centreline edge that starts clear of a chain lies at least I_s from it,
and a chain's edge that starts clear of the tracks, which it keeps at I_t,
at least I_t - H from the centreline; I_min is at most I_t. Two floors
again move less than R. A chain's vertex on its bubble has a free distance
to the tracks of at most R + l + H - I_t, a centreline vertex on its bubble
one to the chain of at most R + l - I_s. With I_s = I_t + H, as for tracks
of one width and clearance, a bubble and a floor together move less than R
whenever l < 2(I_t - H), which is the chain's width plus twice the
clearance less the pair's gap, and two bubbles whenever l < 2I_t - 2δ.
Between two sections each measures the other's tracks, and the same holds
with H on both sides.

A moving via is a member whose one vertex is its centre, inflated by
I = `via_chain_inflation` against another member and seen by it so, and
against another via by `via_via_inflation`. The argument above holds with
l = 0 for the via's own edge of no length: a chain edge's ends see the
via's centre at the same inflation as it sees them, so the via passes
through no edge and no vertex passes through the via, and two vias do not
pass each other. Its box is the smallest of its own, from its free
distance to static copper and the larger of CROSS_SHARE of that to the
members near it and the other vias and the floor, which it takes only when
it starts clear of its clearance, and the boxes of the ends it carries,
which move with it: so each carried end moves inside its own box too, and
the chain ends it carries keep their topology as any vertex does. The
chains a via carries are siblings, as a junction's branches are.

What the boxes allow is coming too close: ending inside another chain's
clearance, or an edge ending across one. The check after each run catches
that. An edge clear of the other chains at the burst's start that is not
clear of their new shapes moves its free ends back, and the check repeats,
as moving one chain's vertex back can put another chain's moved edge
across it; moved back whole, every chain is at its start, clear. An edge
that starts inside another chain's clearance is not checked: its ends lie
within δ (deeper ones are frozen) and take no floor, so they barely move,
and the violation guard on the best state counts it. A via clear at the
burst's start that ends inside its clearance, of static copper, a member or
another via, moves back with the ends it carries; the guard on the best
state keeps every via no deeper inside than it started, as it keeps the
members' violations.
"""

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from itertools import pairwise

import numpy as np
from loguru import logger
from scipy.spatial import cKDTree

from kibends.bursts import (
    DEFAULT_SETTINGS,
    INSIDE_TOLERANCE,
    RETRY_SHARE,
    OptimizerSettings,
    bounded_burst,
    half_side,
    point_distance,
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
    bend_radius_energy,
    edge_integral,
    obstacle_integrand,
    obstacle_reach,
    quadrature_points,
    shape_energy,
)
from kibends.neighbours import (
    ChainInput,
    chain_field,
    kept_spans,
    neighbour_field,
)
from kibends.obstacles import (
    HOLE_CLEARANCE,
    HOLE_TO_HOLE,
    ChainCentreline,
    ChainField,
    via_via_inflation,
)
from kibends.pair_blocks import PairBlock, PairInput, pair_block
from kibends.pairs import BendCheck, bend_radii, too_tight
from kibends.via_moves import MovingVias
from kibends.vias import ViaInput

# A span only slightly longer than a multiple of the spacing gets no extra
# vertex, so float noise in re-measured lengths does not change the count.
RESAMPLE_SLACK = 1e-6

# Below this turning angle a vertex counts as straight (infinite radius).
STRAIGHT_TURN = 1e-9

MEASURE_SAMPLES_PER_SPACING = 4

# After this many abnormal L-BFGS-B ends in a row a problem counts as
# settled and keeps what its bursts reached.
MAX_ABNORMAL_BURSTS = 3

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
    """Some problem stopped after MAX_ABNORMAL_BURSTS abnormal bursts in a
    row and no burst of the run lowered the energy: nothing usable was
    found."""


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
    """Bursts L-BFGS-B ended abnormally; each was judged like any other."""
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
    """Outer iterations, each giving every problem that has not settled one
    burst."""
    vias: tuple[ViaResult, ...] = ()
    """In the order the vias were given."""
    pairs: tuple[PairResult, ...] = ()
    """In the order the coupled sections were given."""
    gave_up: tuple[tuple[str, int], ...] = ()
    """The problems given up on after abnormal bursts in a row, by
    description, and how many there were."""
    abnormal: tuple[tuple[str, int], ...] = ()
    """The problems whose bursts L-BFGS-B ended abnormally, by description,
    and how many there were. A burst moves every chain, section and via of
    its problem, so it is counted once, for the problem."""


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
        abnormal_bursts=sum(n for _, n in run.abnormal),
        frozen=chain.frozen,
        before=chain.before,
        after=after,
    )


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


def shape_change(before: FloatArray, after: FloatArray) -> float:
    """Local symmetric distance between two polylines, mm.

    Vertices sliding along an unchanged shape do not count: resampling to
    uniform spacing and the energy's indifference to where along the track
    a vertex sits would otherwise keep moving vertices forever without
    changing the track.
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


@dataclass(frozen=True)
class SelfCheck:
    """What a burst may not do to a coupled section's centreline, measured
    against `original`, where the burst started: with a `bend_limit`, bend
    an edge newly below it; with `apart`, bring vertices closer than that
    to its parts beyond a turn (`close_to_itself`), or than either started
    from them where it started closer, but never than the two started."""

    original: FloatArray
    bend_limit: float | None
    apart: float | None
    tight_before: BoolArray | None
    nearest: FloatArray
    """Per vertex, how close it started to its parts beyond a turn, at most
    `apart`."""

    @staticmethod
    def of(
        original: FloatArray,
        bend_limit: float | None = None,
        apart: float | None = None,
    ) -> SelfCheck:
        tight_before = (
            None if bend_limit is None else too_tight(bend_radii(original), bend_limit)
        )
        nearest = np.full(len(original), apart or 0.0)
        if apart is not None:
            pairs, distance = close_to_itself(original, apart)
            np.minimum.at(nearest, pairs[:, 0], distance)
            np.minimum.at(nearest, pairs[:, 1], distance)
        return SelfCheck(original, bend_limit, apart, tight_before, nearest)

    def __call__(self, candidate: FloatArray) -> tuple[BoolArray, int, int]:
        """The vertices to move back, how many edges newly bent too tightly
        and vertex pairs came too close, and how many of those were pairs."""
        back = np.zeros(len(candidate), dtype=bool)
        count = close = 0
        if self.bend_limit is not None and self.tight_before is not None:
            tight = too_tight(bend_radii(candidate), self.bend_limit)
            tight &= ~self.tight_before
            around = np.flatnonzero(tight)[:, None] + BEND_REACH
            back[np.clip(around, 0, len(candidate) - 1).ravel()] = True
            count += int(tight.sum())
        if self.apart is not None:
            pairs, distance = close_to_itself(candidate, self.apart)
            i, j = pairs[:, 0], pairs[:, 1]
            # At most the pair's own starting distance, so a pair moved back
            # whole is never too close, even one that lies beyond a turn only
            # along the candidate.
            started = np.hypot(*(self.original[i] - self.original[j]).T)
            floor = np.minimum(np.maximum(self.nearest[i], self.nearest[j]), started)
            closer = distance < floor - INSIDE_TOLERANCE
            back[i[closer]] = back[j[closer]] = True
            close = int(closer.sum())
            count += close
        return back, count, close


# How far, mm, the points of a member and the pieces it sees may move before
# it finds its neighbours again (`NeighbourList`): a wider skin lists more
# pairs at every evaluation, a narrower one searches more often.
LIST_SKIN = 0.05

# The share of its free distance to the other chains, which move in the same
# burst, that a vertex's bubble takes: at 1/2 two vertices closing on each
# other cover at most half the distance between them.
CROSS_SHARE = 0.5

# The share of the smallest inflation between two members or vias, less δ,
# that every variable starting clear of them may move however close they
# are: at 1/2 two such points closing on each other never meet. Measured
# against 1/4 and 0.9 on a DIN-like bundle in
# https://github.com/mairas/kibends/issues/28.
STEP_FLOOR = 0.5

End = tuple[int, int]
"""A chain end: the chain's index and 0 or -1."""

Member = ChainInput | PairBlock
"""What a problem optimises: a chain, or a coupled section as its
centreline."""

Resumed = tuple[FloatArray, BoolArray]
"""Where a member resumes: its polyline and the vertices that stay."""


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
    """Minimise the chains', sections' and vias' summed energy, each
    problem's jointly: the layers that moving vias tie together are one
    problem, and every other layer is one of its own.

    Takes chains whose ends are fixed or loose where they meet at junctions
    (`junctions`), coupled sections of pairs (`pairs`) and moving `vias`,
    and returns each chain's, section's and via's best state; a junction's
    ends come back at one point, loose. A via keeps `hole_clearance` and
    `hole_to_hole` where holes are involved, and one that starts more than
    δ inside its clearance stays where it is, `frozen` in its result, with
    the chain ends it carries pinned. Each
    problem keeps its best state, replaced only after a burst that does not
    raise its energy, leaves no chain or section with more violations than
    it started with, and leaves no via deeper inside its clearance than it
    started.

    A problem settles when its burst moved no shape or via by
    `settled_change` and moved nothing back but for a section's parts that
    came too close to each other, or when the lowest energy its bursts
    reached fell by less than `stall_tolerance` over `stall_window` bursts,
    whether or not the guard took them as its best state, or after
    MAX_ABNORMAL_BURSTS abnormal bursts in a row; the run converges when
    every problem has. With a time limit in `settings`, the run stops after
    the first burst that ends when `clock` reads the limit or more past
    `started`, by default when the run starts.

    A run given the result of an earlier one on the same inputs as `start`
    starts each problem from that run's best state, the sections held in
    this run excepted, unless that leaves some chain or section of the
    problem with more violations or more edges across the others than its
    input, or a via newly inside its clearance; then that problem starts
    from its inputs. Either way the run
    completes its first outer iteration before it looks at the time limit.

    Each outer iteration logs its progress at DEBUG, and at INFO once
    `PROGRESS_INTERVAL` has passed since `started` or the last INFO line.
    """
    junctions(chains)
    blocks = [pair_block(pair, settings.spacing) for pair in pairs]
    everything = [*chains, *(block.chain for block in blocks)]
    moving_vias = MovingVias(everything, vias, hole_clearance, hole_to_hole)
    placed = _problems(chains, blocks, moving_vias, params, settings, start)
    problems = [p.problem for p in placed]
    moving = [p for p in problems if p.moving]

    def result(termination: Termination, iterations: int, *, best: bool) -> RunResult:
        by_index = {
            i: p.problem.result(k, best=best)
            for p in placed
            for k, i in enumerate(p.chains)
        }
        by_section = {
            i: p.problem.pair_result(len(p.chains) + k, best=best)
            for p in placed
            for k, i in enumerate(p.sections)
        }
        by_via = {
            v: p.problem.via_result(k, best=best)
            for p in placed
            for k, v in enumerate(p.vias)
            if not p.problem.vias[k].frozen
        }
        return RunResult(
            tuple(by_index[i] for i in range(len(chains))),
            termination,
            iterations,
            vias=tuple(
                by_via.get(v)
                or ViaResult(via.via.position, frozen=True, rejected_edges=0)
                for v, via in enumerate(vias)
            ),
            pairs=tuple(by_section[k] for k in range(len(blocks))),
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
        # A resumed run gives every problem a burst against what changed
        # since, as the sections held now.
        timed = deadline is not None and (start is None or iteration > 1)
        out_of_time = False
        change = 0.0
        for problem in moving:
            if problem.settled:
                continue
            change = max(change, problem.step())
            if timed and deadline is not None and clock() >= deadline:
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
        return replace(final, termination=Termination.ABNORMAL)
    return final


@dataclass(frozen=True)
class _Placed:
    """A problem and the indices of its chains, sections and vias among
    those of the run."""

    problem: _Problem
    chains: list[int]
    sections: list[int]
    vias: list[int]


def _problems(
    chains: Sequence[ChainInput],
    blocks: Sequence[PairBlock],
    vias: MovingVias,
    params: EnergyParams,
    settings: OptimizerSettings,
    start: RunResult | None,
) -> list[_Placed]:
    """The run's problems: the layers that the vias that move tie together,
    with every via, and each other layer alone, with the vias that stand
    still on it. `vias` holds the chains, then the sections' centrelines."""
    originals = [chain.centreline.points for chain in vias.chains]
    positions = vias.starting_positions()
    frozen = [
        vias.distance(v, positions, originals) < -settings.bubble_tolerance
        for v in range(len(vias.vias))
    ]
    # The ends a via that stays would carry are pinned, as at an unselected one.
    chains = _pinned(
        chains,
        [
            [(end.chain, end.index) for end in vias.ends[v]]
            for v, still in enumerate(frozen)
            if still
        ],
    )
    layers = list(dict.fromkeys(chain.layer for chain in vias.chains))
    tied = [
        layer
        for layer in layers
        if any(
            not still and via.via.reaches(layer)
            for via, still in zip(vias.vias, frozen, strict=True)
        )
    ]
    groups = [tied] if tied else []
    groups += [[layer] for layer in layers if layer not in tied]
    placed: list[_Placed] = []
    for group in groups:
        on = [i for i, c in enumerate(chains) if c.layer in group]
        sections = [k for k, b in enumerate(blocks) if b.given.layer in group]
        members = [*on, *(len(chains) + k for k in sections)]
        local = {i: n for n, i in enumerate(members)}
        joined = [
            v
            for v, via in enumerate(vias.vias)
            if group is tied or any(via.via.reaches(layer) for layer in group)
        ]
        resume = (
            None
            if start is None
            else _Resume(
                _resumed(start, on, sections, blocks),
                [np.array(start.vias[v].position, dtype=float) for v in joined],
            )
        )
        problem = _Problem(
            group,
            [chains[i] for i in on],
            [blocks[k] for k in sections],
            [_problem_via(vias, v, frozen[v], local, joined) for v in joined],
            params,
            settings,
            resume,
        )
        placed.append(_Placed(problem, on, sections, joined))
    return placed


def _problem_via(
    vias: MovingVias,
    v: int,
    frozen: bool,
    local: Mapping[int, int],
    joined: Sequence[int],
) -> _Via:
    """Via v of `vias` in a problem whose members `local` numbers and whose
    vias are `joined`."""
    via = vias.vias[v].via
    apart: dict[int, float] = {}
    for k, u in enumerate(joined):
        if u == v:
            continue
        inflation = via_via_inflation(
            via,
            vias.vias[u].via,
            vias.clearance_of,
            hole_clearance=vias.hole_clearance,
            hole_to_hole=vias.hole_to_hole,
        )
        if inflation is not None:
            apart[k] = inflation
    carried = (
        ()
        if frozen
        else tuple((local[e.chain], e.index, e.offset) for e in vias.ends[v])
    )
    return _Via(
        vias.vias[v],
        frozen,
        carried,
        {local[i]: vias.ring(v, i) for i in vias.chains_near(v) if i in local},
        apart,
    )


@dataclass(frozen=True)
class _Via:
    """A via of a problem, with the problem's members and vias by their
    index in it."""

    given: ViaInput
    frozen: bool
    """It started more than δ inside its clearance and stays."""
    carried: tuple[tuple[int, int, FloatArray], ...]
    """The chain ends it carries, if it moves: the member, 0 or -1 for its
    first or last point, and the end's offset from the via's centre."""
    rings: Mapping[int, float]
    """The members it sees and that see it, and how far their centrelines
    keep from its centre."""
    apart: Mapping[int, float]
    """The other vias it keeps clear of, and how far apart their centres
    keep."""


@dataclass(frozen=True)
class _Resume:
    members: list[Resumed | None]
    positions: list[FloatArray]
    """Of the problem's vias."""


def _resumed(
    start: RunResult,
    chains: Sequence[int],
    sections: Sequence[int],
    blocks: Sequence[PairBlock],
) -> list[Resumed | None]:
    """Where the chains and sections of a problem, by index, resume from
    `start`: its best state, but None for a section held in this run. A
    section the earlier run moved comes back without its direction pins
    among its fixed vertices, as simplification may merge across them;
    here they are pinned again."""
    resumed: list[Resumed | None] = [
        (start.chains[i].polyline, start.chains[i].fixed) for i in chains
    ]
    for k in sections:
        block = blocks[k]
        if block.held:
            resumed.append(None)
            continue
        points = start.pairs[k].centreline
        resumed.append((points, start.pairs[k].fixed | block.direction_pins(points)))
    return resumed


def junctions(chains: Sequence[ChainInput]) -> list[list[End]]:
    """The chain ends that are not fixed, grouped by the point, layer and net
    they share; raises ValueError for one that meets no other."""
    groups: dict[tuple[str, str, float, float], list[End]] = {}
    for i, chain in enumerate(chains):
        for end in (0, -1):
            if not chain.fixed[end]:
                x, y = chain.centreline.points[end]
                key = (chain.layer, chain.centreline.net, float(x), float(y))
                groups.setdefault(key, []).append((i, end))
    for (layer, _, x, y), ends in groups.items():
        if len(ends) < 2:
            raise ValueError(
                "endpoints of a chain must be fixed or shared with another "
                f"chain: {chains[ends[0][0]].centreline.description} ends "
                f"loose at ({x:.3f}, {y:.3f}) on {layer}"
            )
    return list(groups.values())


def pinned_at_junctions(chains: Sequence[ChainInput]) -> list[ChainInput]:
    """The chains with their ends at junctions fixed, as neighbour fields
    need them: chains of one net leave each other out near where they touch
    at fixed points."""
    return _pinned(chains, junctions(chains))


def _pinned(
    chains: Sequence[ChainInput], groups: Sequence[Sequence[End]]
) -> list[ChainInput]:
    pinned = list(chains)
    for group in groups:
        for i, end in group:
            fixed = pinned[i].fixed.copy()
            fixed[end] = True
            pinned[i] = replace(pinned[i], fixed=fixed)
    return pinned


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
    chains: Sequence[Member],
    polylines: Sequence[FloatArray],
    masks: Sequence[BoolArray],
    params: EnergyParams,
    short_edge: float,
    seen: Sequence[SeenParts] | None = None,
) -> tuple[float, list[FloatArray]]:
    """The summed energy of chains and sections of one layer, each in its
    static field plus what it sees of the others at `polylines`, by default
    as `seen_parts` finds it there, a section with its bend-radius hinge,
    and its gradient at every vertex of every chain and centreline, with the
    rows of `masks` zeroed; raises ValueError if it is not finite.

    A section's centreline is what it keeps clear as, and its two tracks,
    offset from it, are what the others keep clear of: their part of the
    gradient is taken back through the offset (`PairBlock.offset_pullback`).
    Chains whose loose ends meet must come with those ends fixed
    (`pinned_at_junctions`), so that they leave each other out there.
    """
    if seen is None:
        seen = seen_parts(chains, polylines, obstacle_reach(params))
    value, gradients, _ = members_energy(
        chains, polylines, masks, params, short_edge, seen, ()
    )
    return value, gradients


def members_energy(
    members: Sequence[Member],
    polylines: Sequence[FloatArray],
    masks: Sequence[BoolArray],
    params: EnergyParams,
    short_edge: float,
    seen: Sequence[SeenParts],
    positions: Sequence[FloatArray],
    listed: NeighbourList | None = None,
) -> tuple[float, list[FloatArray], FloatArray]:
    """`layer_energy` of members on any layers, each seeing the others on its
    own as `seen` shows them, with vias at `positions` among them; and the
    gradient at each via's centre. With `listed`, made for `seen`, each
    member looks only at the pieces it lists."""
    reach = obstacle_reach(params)
    integrand = obstacle_integrand(params)
    k = params.points_per_edge
    lines = [
        shown(member, points) for member, points in zip(members, polylines, strict=True)
    ]
    flat = _flat(lines, positions)
    at_shown = np.zeros_like(flat)
    gradients: list[FloatArray] = []
    value = 0.0
    for i, (member, points) in enumerate(zip(members, polylines, strict=True)):
        queries = quadrature_points(points, k)
        static, static_direction = _body(member).field.signed_distance(queries, reach)
        others = _others(seen[i], flat)
        near = (
            others.field.nearest(queries, reach)
            if listed is None
            else others.field.nearest_among(
                queries, reach, listed.pairs(i, queries, flat, others.field)
            )
        )
        nearer = near.distance < static
        distance = np.where(nearer, near.distance, static)
        direction = np.where(nearer[:, None], near.gradient, static_direction)
        obstacle, obstacle_gradient, slopes = edge_integral(
            points, distance, direction, integrand, k
        )
        shape, shape_gradient = shape_energy(points, masks[i], short_edge)
        value += shape + obstacle
        gradient = shape_gradient + obstacle_gradient
        if isinstance(member, PairBlock):
            bend, bend_gradient = bend_radius_energy(
                points, masks[i], member.limit, params
            )
            value += bend
            gradient += bend_gradient
        gradients.append(gradient)
        pulled = np.flatnonzero(nearer & (slopes != 0.0))
        if len(pulled):
            edge = near.edge[pulled]
            start = others.vertex[edge]
            along = _parameters(queries[pulled], others.starts[edge], others.ends[edge])
            t = others.start[edge] + along * (others.end[edge] - others.start[edge])
            push = slopes[pulled, None] * near.gradient[pulled]
            np.add.at(at_shown, start, -push * (1 - t)[:, None])
            np.add.at(at_shown, start + 1, -push * t[:, None])
    first = 0
    for member, points, gradient, parts in zip(
        members, polylines, gradients, lines, strict=True
    ):
        pieces: list[FloatArray] = []
        for _, part in parts:
            pieces.append(at_shown[first : first + len(part)])
            first += len(part)
        if isinstance(member, PairBlock):
            gradient += member.offset_pullback(points, *pieces)
        else:
            gradient += pieces[0]
    via_gradients = at_shown[first:].reshape(-1, 2, 2).sum(axis=1)
    if not (math.isfinite(value) and all(np.all(np.isfinite(g)) for g in gradients)):
        raise ValueError(f"energy is not finite: {value}")
    for g, mask in zip(gradients, masks, strict=True):
        g[mask] = 0.0
    return value, gradients, via_gradients


def shown(member: Member, points: FloatArray) -> list[tuple[ChainInput, FloatArray]]:
    """What the others see of a member at `points`: a chain itself, a
    section its two tracks."""
    if isinstance(member, PairBlock):
        return list(zip(member.tracks, member.offset(points), strict=True))
    return [(member, points)]


def _body(member: Member) -> ChainInput:
    """The chain a member's own energy and pins are those of."""
    return member.chain if isinstance(member, PairBlock) else member


def _viewer(member: Member) -> tuple[ChainInput, Mapping[str, ChainInput] | None]:
    """The chain a member keeps clear of the others as, and, for a section,
    that chain under each of its nets, as `neighbours.neighbour_field`
    takes them."""
    if isinstance(member, PairBlock):
        return member.facing, member.as_net
    return member, None


def _shown_flat(
    members: Sequence[Member],
    polylines: Sequence[FloatArray],
    positions: Sequence[FloatArray] = (),
) -> FloatArray:
    """What the others see of every member, polylines one after another,
    then each via's centre twice, an edge of no length."""
    return _flat(
        [shown(member, line) for member, line in zip(members, polylines, strict=True)],
        positions,
    )


def _flat(
    lines: Sequence[Sequence[tuple[ChainInput, FloatArray]]],
    positions: Sequence[FloatArray],
) -> FloatArray:
    """`_shown_flat` of the members `shown` as `lines`."""
    return np.concatenate(
        [
            *(points for parts in lines for _, points in parts),
            *(np.array([at, at]) for at in positions),
        ]
    )


@dataclass(frozen=True)
class SeenParts:
    """What one chain of a layer sees of the others: pieces of their edges,
    each inflated by its own amount."""

    vertex: IntArray
    """Each piece's edge's first vertex, numbered across what the others
    see of every chain and section (`shown`), one after another, then two
    for each via (`_shown_flat`)."""
    start: FloatArray
    """Where along its edge each piece starts, from 0 to 1."""
    end: FloatArray
    inflation: FloatArray


def seen_parts(
    chains: Sequence[Member],
    polylines: Sequence[FloatArray],
    reach: float,
    rings: Sequence[Mapping[int, float]] = (),
) -> list[SeenParts]:
    """For each chain or section at its polyline, the others on its layer at
    theirs as `neighbours.neighbour_field` shows them to it: another net's
    chain whole, inflated by both half widths and the larger clearance, and
    a chain of its net inflated by the half widths alone and left out near
    where they touch. A section sees as its centreline of the pair's outer
    extent, under either of its nets (`PairBlock.as_net`), and is seen as
    its two tracks. Each via of `rings` is seen by the members it names,
    inflated by the distance it gives."""
    lines = [
        shown(member, points) for member, points in zip(chains, polylines, strict=True)
    ]
    first = np.cumsum([0, *(len(points) for parts in lines for _, points in parts)])
    seen: list[SeenParts] = []
    for i, member in enumerate(chains):
        own, as_net = _viewer(member)
        me = own.centreline
        nets = {me.net: own} if as_net is None else as_net
        vertices: list[IntArray] = [np.empty(0, dtype=np.intp)]
        starts: list[FloatArray] = [np.empty(0)]
        ends: list[FloatArray] = [np.empty(0)]
        inflations: list[FloatArray] = [np.empty(0)]
        line = 0
        for j, parts in enumerate(lines):
            for other, points in parts:
                line += 1
                if j == i or other.layer != own.layer:
                    continue
                them = other.centreline
                margin = (
                    0.0 if them.net == me.net else max(me.clearance, them.clearance)
                )
                mine = nets.get(them.net)
                spans = (
                    None
                    if mine is None
                    else kept_spans(mine, other, points, reach, polylines[i], margin)
                )
                edge, start, end = _pieces(points, spans)
                vertices.append(first[line - 1] + edge)
                starts.append(start)
                ends.append(end)
                inflations.append(
                    np.full(len(edge), (me.width + them.width) / 2 + margin)
                )
        seeing = [(k, ring[i]) for k, ring in enumerate(rings) if i in ring]
        vertices.append(np.array([first[-1] + 2 * k for k, _ in seeing], dtype=np.intp))
        starts.append(np.zeros(len(seeing)))
        ends.append(np.ones(len(seeing)))
        inflations.append(np.array([inflation for _, inflation in seeing]))
        seen.append(
            SeenParts(
                np.concatenate(vertices),
                np.concatenate(starts),
                np.concatenate(ends),
                np.concatenate(inflations),
            )
        )
    return seen


class NeighbourList:
    """A Verlet list: per member, the pairs of its quadrature points and the
    pieces it sees of the others (`SeenParts`) that can answer each other
    while every one of them stays within `skin` of where it was when the
    pairs were found. A member whose points or pieces have moved farther
    finds its pairs again where they are. Either way the energy is the one a
    search over every piece gives; the list only spares the search."""

    def __init__(self, seen: Sequence[SeenParts], reach: float, skin: float) -> None:
        self.seen = seen
        self.reach = reach
        self.skin = skin
        self._found: list[tuple[FloatArray, FloatArray] | None] = [None] * len(seen)
        self._pairs: list[tuple[IntArray, IntArray]] = [
            (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))
        ] * len(seen)

    def pairs(
        self, i: int, queries: FloatArray, flat: FloatArray, field: ChainField
    ) -> tuple[IntArray, IntArray]:
        """Member i's pairs with its quadrature points at `queries` and what
        it sees of the others at `flat`, as `field`."""
        vertex = self.seen[i].vertex
        ends = flat[np.concatenate([vertex, vertex + 1])]
        found = self._found[i]
        if (
            found is None
            or _moved(queries, found[0]) > self.skin
            or _moved(ends, found[1]) > self.skin
        ):
            # A point's distance to a piece changes by at most its own move
            # plus the larger move of the piece's ends.
            self._pairs[i] = field.candidates(queries, self.reach, 2 * self.skin)
            self._found[i] = (queries, ends)
        return self._pairs[i]


def _moved(points: FloatArray, start: FloatArray) -> float:
    """The largest distance from start[i] to points[i], mm."""
    if not len(points):
        return 0.0
    delta = points - start
    return math.sqrt(float(np.max(np.einsum("ij,ij->i", delta, delta))))


def _subset(seen: SeenParts, keep: BoolArray) -> SeenParts:
    return SeenParts(
        seen.vertex[keep], seen.start[keep], seen.end[keep], seen.inflation[keep]
    )


def _pieces(
    points: FloatArray, spans: Sequence[tuple[float, float]] | None
) -> tuple[IntArray, FloatArray, FloatArray]:
    """The edges of `points` that the arc-length `spans` cover, by default
    all of them, and where along each the cover starts and ends."""
    edges = np.arange(len(points) - 1, dtype=np.intp)
    if spans is None:
        return edges, np.zeros(len(edges)), np.ones(len(edges))
    arc = arc_lengths(points)
    lengths = np.diff(arc)
    found: list[IntArray] = [np.empty(0, dtype=np.intp)]
    starts: list[FloatArray] = [np.empty(0)]
    ends: list[FloatArray] = [np.empty(0)]
    for left, right in spans:
        k = np.flatnonzero((arc[:-1] < right) & (arc[1:] > left))
        length = np.where(lengths[k] > 0.0, lengths[k], 1.0)
        found.append(k)
        starts.append(np.clip((left - arc[k]) / length, 0.0, 1.0))
        ends.append(np.clip((right - arc[k]) / length, 0.0, 1.0))
    return np.concatenate(found), np.concatenate(starts), np.concatenate(ends)


@dataclass(frozen=True)
class _Others:
    """What one chain of a layer sees of the others, at given shapes."""

    field: ChainField
    starts: FloatArray
    ends: FloatArray
    vertex: IntArray
    """Each piece's edge's first vertex, numbered across every chain of the
    layer."""
    start: FloatArray
    """Where along its edge each piece starts, from 0 to 1."""
    end: FloatArray


def _others(seen: SeenParts, flat: FloatArray) -> _Others:
    """The pieces of `seen` with the layer's chains at `flat`, their
    polylines one after another."""
    a, b = flat[seen.vertex], flat[seen.vertex + 1]
    along = b - a
    starts = a + seen.start[:, None] * along
    ends = a + seen.end[:, None] * along
    return _Others(
        ChainField(starts, ends, seen.inflation),
        starts,
        ends,
        seen.vertex,
        seen.start,
        seen.end,
    )


def _parameters(points: FloatArray, starts: FloatArray, ends: FloatArray) -> FloatArray:
    """Where along each segment from starts[i] to ends[i] its point nearest
    points[i] lies, from 0 to 1."""
    along = ends - starts
    squared = np.einsum("ij,ij->i", along, along)
    projection = np.einsum("ij,ij->i", points - starts, along)
    return np.clip(projection / np.where(squared > 0.0, squared, 1.0), 0.0, 1.0)


def _resample(
    points: FloatArray,
    mask: BoolArray,
    loose: Sequence[int],
    spacing: float,
    kept: BoolArray | None = None,
) -> tuple[FloatArray, BoolArray]:
    """`optimize.resample`, keeping the `loose` ends and the `kept` vertices
    where they are, as free vertices."""
    anchors = mask.copy()
    anchors[list(loose)] = True
    if kept is not None:
        anchors |= kept
    points, fixed = resample(points, anchors, spacing)
    # Resampling keeps the anchors, in order.
    fixed[np.flatnonzero(fixed)[~mask[anchors]]] = False
    return points, fixed


def _variables(
    masks: Sequence[BoolArray], groups: Sequence[Sequence[End]]
) -> tuple[list[IntArray], int]:
    """Per chain, each vertex's variable, -1 where it is fixed, with the
    ends of each group sharing one; and how many variables there are."""
    index = [np.full(len(mask), -1, dtype=np.intp) for mask in masks]
    count = 0
    for group in groups:
        for i, end in group:
            index[i][end] = count
        count += 1
    for i, mask in enumerate(masks):
        free = ~mask & (index[i] < 0)
        index[i][free] = count + np.arange(int(free.sum()), dtype=np.intp)
        count += int(free.sum())
    return index, count


@dataclass(frozen=True)
class _Layout:
    """How a burst's variables make the members' polylines and the vias'
    centres: each moving via's centre is a variable, and an end it carries
    is that variable plus the end's offset."""

    starts: list[FloatArray]
    positions: list[FloatArray]
    index: list[IntArray]
    """Per member, each vertex's variable, -1 where it stays."""
    offsets: list[FloatArray]
    """Per member, each vertex's offset from its variable."""
    via_index: IntArray
    """Each via's variable, -1 for one that stays."""
    count: int

    @property
    def x0(self) -> FloatArray:
        x = np.zeros((self.count, 2))
        for start, i, offset in zip(self.starts, self.index, self.offsets, strict=True):
            free = i >= 0
            x[i[free]] = start[free] - offset[free]
        for v, at in zip(self.via_index, self.positions, strict=True):
            if v >= 0:
                x[v] = at
        return x.ravel()

    def unflatten(self, x: FloatArray) -> tuple[list[FloatArray], list[FloatArray]]:
        """The members' polylines and the vias' centres at `x`."""
        at = x.reshape(-1, 2)
        polylines: list[FloatArray] = []
        for start, i, offset in zip(self.starts, self.index, self.offsets, strict=True):
            points = start.copy()
            free = i >= 0
            points[free] = at[i[free]] + offset[free]
            polylines.append(points)
        positions = [
            at[v].copy() if v >= 0 else p
            for v, p in zip(self.via_index, self.positions, strict=True)
        ]
        return polylines, positions

    def gradient(
        self, gradients: Sequence[FloatArray], via_gradients: FloatArray
    ) -> FloatArray:
        """The gradient by the variables, from that at every vertex and at
        every via's centre."""
        gradient = np.zeros((self.count, 2))
        for g, i in zip(gradients, self.index, strict=True):
            free = i >= 0
            np.add.at(gradient, i[free], g[free])
        moving = self.via_index >= 0
        np.add.at(gradient, self.via_index[moving], via_gradients[moving])
        return gradient.ravel()


@dataclass(frozen=True)
class _ViaView:
    """What a via sees of the members near it and the other vias, as edges,
    and whose each edge is."""

    field: ChainField
    starts: FloatArray
    ends: FloatArray
    member: IntArray
    """Each edge's member, -1 for a via."""
    vertex: IntArray
    """Each edge's first vertex in its member, or the via's index."""


@dataclass(frozen=True)
class _Round:
    """What one check of a burst found: per member, its edges that newly
    crossed and, of those, a section's vertex pairs too close; per via, what
    moved it back."""

    members: list[tuple[int, int]]
    vias: list[int]


@dataclass(frozen=True)
class _JointBurst:
    polylines: list[FloatArray]
    positions: list[FloatArray]
    rejected_edges: list[int]
    """Per member, its edges whose moves were undone, and a section's
    vertex pairs, over every retry."""
    too_close: list[int]
    """Per member, of `rejected_edges`, a section's vertex pairs that came
    too close to each other (`optimize.close_to_itself`)."""
    via_rejected: list[int]
    """Per via, the edges at the ends it carries that newly crossed, and
    the times it ended newly inside its clearance, over every retry."""
    abnormal: bool


class _Problem:
    """The chains, coupled sections and vias of the layers that moving vias
    tie together, or of one layer, optimised together: their current shapes,
    the best state so far, and whether the problem has settled. The chains
    come first, then the sections, as their centrelines."""

    def __init__(
        self,
        layers: Sequence[str],
        chains: Sequence[ChainInput],
        sections: Sequence[PairBlock],
        vias: Sequence[_Via],
        params: EnergyParams,
        settings: OptimizerSettings,
        resume: _Resume | None = None,
    ) -> None:
        self.vias = list(vias)
        self.free_vias = [k for k, via in enumerate(vias) if not via.frozen]
        self.description = _description(layers, bool(self.free_vias))
        self.groups = junctions(chains)
        count = len(chains) + len(sections)
        self.loose: list[list[int]] = [[] for _ in range(count)]
        # Branches of one junction leave each other out near it, so what
        # each sees of the others there begins about their half widths plus
        # the reach away: as bubbles, or as the step floor their half widths
        # set, that would hold the vertices near the junction, and with them
        # the junction, to a fraction of the reach per burst. They are one
        # net, so a branch passing through another changes no connection,
        # and the edge check still keeps their edges from newly crossing. So
        # with the chains a via carries.
        self.siblings: list[list[int]] = [[] for _ in range(count)]
        for group in self.groups:
            for i, end in group:
                self.loose[i].append(end)
                self.siblings[i] += [k for k, _ in group if k != i]
        self.carried: list[list[tuple[int, int]]] = [[] for _ in range(count)]
        for k in self.free_vias:
            for i, end, _ in vias[k].carried:
                self.carried[i].append((k, end))
                self.siblings[i] += [j for j, _, _ in vias[k].carried if j != i]
        self.rings = [via.rings for via in vias]
        # The chains as their neighbours see them; their own masks below
        # leave the junction ends loose.
        self.members: list[Member] = [*_pinned(chains, self.groups), *sections]
        self.bodies = [_body(member) for member in self.members]
        self.neighbours = [
            [
                j
                for j, other in enumerate(self.bodies)
                if j != i and other.layer == b.layer
            ]
            for i, b in enumerate(self.bodies)
        ]
        self.params = params
        self.settings = settings
        self.reach = obstacle_reach(params)
        self.integrand = obstacle_integrand(params)
        self.originals = [body.centreline.points for body in self.bodies]
        self.start_positions = [
            np.array(via.given.via.position, dtype=float) for via in vias
        ]
        start_fields = self.fields(self.originals, self.start_positions)
        self.anchors: list[BoolArray] = []
        self.frozen: list[tuple[FrozenVertex, ...]] = []
        for i, (given, body) in enumerate(
            zip([*chains, *(s.chain for s in sections)], self.bodies, strict=True)
        ):
            others: dict[int, ObstacleField | None] = {
                j: self.neighbour(i, [j], self.originals) for j in self.neighbours[i]
            }
            mask, records = frozen_vertices(
                body, start_fields[i], others, self.bodies, settings
            )
            self.anchors.append(given.fixed | mask)
            self.frozen.append(records)
        self.before: list[PolylineStats] = [
            measure(points, field, settings.spacing)
            for points, field in zip(self.originals, start_fields, strict=True)
        ]
        # A burst starts from the members resampled; the shapes as the burst
        # left them are what the energy, the guard and the best state see.
        # Resampling would put free vertices between a held section's pins.
        self.points: list[FloatArray] = []
        self.masks: list[BoolArray] = []
        for member, original, anchors, loose in zip(
            self.members, self.originals, self.anchors, self.loose, strict=True
        ):
            if isinstance(member, PairBlock) and member.held:
                points, mask = original, anchors
            else:
                points, mask = _resample(original, anchors, loose, settings.spacing)
            moves = not mask.all()
            self.points.append(points if moves else original)
            self.masks.append(mask if moves else anchors)
        self.positions = list(self.start_positions)
        self.free = [
            i for i, mask in enumerate(self.masks) if not mask.all() or self.carried[i]
        ]
        self.moving = bool(self.free) or bool(self.free_vias)
        # Resampling can lay a chord across an obstacle corner, and a burst
        # only rejects crossings its own moves create, so the result is
        # limited to no more violations per member than the resampled start,
        # and to no via deeper inside its clearance than it started.
        self.allowed = self.violations(self.points, self.masks, self.positions)
        self.inside = self.via_distances(self.points, self.positions)
        self.moved = False
        # The first burst starts from the shapes `allowed` was measured on:
        # resampled again, they would cut corners deeper still.
        self.resampled = [True] * count
        if resume is not None:
            self.resume(resume)
        self.energy = self.energy_of(self.points, self.positions)
        self.start_energy = self.best_energy = self.energy
        self.best = list(self.points)
        self.best_masks = list(self.masks)
        self.best_positions = list(self.positions)
        self.lowest = [self.energy]
        self.rejected_edges = [0] * count
        self.too_close = [0] * count
        self.via_rejected = [0] * len(vias)
        self.abnormal_bursts = 0
        self.abnormal_streak = 0
        self.settled = not self.moving

    def resume(self, state: _Resume) -> None:
        """Take `state` as the current shapes of the members that move and
        have one, and the vias' centres, unless that leaves some member with
        more violations than `allowed`, or with more edges across the others
        than at its inputs, or some via deeper inside its clearance than it
        started and inside it. Edges are counted too because a held section
        is all pins, which `violations` cannot count, and a burst leaves an
        edge that starts across another member as it is."""
        points, masks = list(self.points), list(self.masks)
        for i in self.free:
            given = state.members[i]
            if given is not None:
                points[i], masks[i] = given
        positions = [
            at if k in self.free_vias else self.positions[k]
            for k, at in enumerate(state.positions)
        ]
        across = zip(
            self.edges_across(points, positions),
            self.edges_across(self.points, self.positions),
            strict=True,
        )
        if self.guard(points, masks, positions) and all(
            count <= before for count, before in across
        ):
            self.points, self.masks, self.positions = points, masks, positions
            self.resampled = [
                done and given is None
                for done, given in zip(self.resampled, state.members, strict=True)
            ]
            self.moved = True

    def edges_across(
        self, polylines: Sequence[FloatArray], positions: Sequence[FloatArray]
    ) -> list[int]:
        """Per member at `polylines`, its edges not clear of the other
        members and the vias at `positions` as it sees them."""
        seen = seen_parts(self.members, polylines, self.reach, self.rings)
        flat = _shown_flat(self.members, polylines, positions)
        return [
            int(
                np.count_nonzero(
                    ~_others(part, flat).field.segments_clear(p[:-1], p[1:])
                )
            )
            for p, part in zip(polylines, seen, strict=True)
        ]

    def guard(
        self,
        polylines: Sequence[FloatArray],
        masks: Sequence[BoolArray],
        positions: Sequence[FloatArray],
    ) -> bool:
        """Whether no member has more violations than `allowed` and no via
        is deeper inside its clearance than it started, and inside it."""
        return all(
            count <= allowed
            for count, allowed in zip(
                self.violations(polylines, masks, positions), self.allowed, strict=True
            )
        ) and all(
            distance >= min(start, -INSIDE_TOLERANCE)
            for distance, start in zip(
                self.via_distances(polylines, positions), self.inside, strict=True
            )
        )

    def neighbour(
        self, i: int, others: Sequence[int], polylines: Sequence[FloatArray]
    ) -> ObstacleField | None:
        """The members `others` at `polylines` as obstacles to member i at
        its own."""
        own, as_net = _viewer(self.members[i])
        return neighbour_field(
            own,
            [part for j in others for part in shown(self.members[j], polylines[j])],
            self.reach,
            polylines[i],
            as_net=as_net,
        )

    def via_obstacles(
        self, i: int, positions: Sequence[FloatArray]
    ) -> ChainField | None:
        """The vias that member i sees, at `positions`, as obstacles to it."""
        seen = [(k, ring[i]) for k, ring in enumerate(self.rings) if i in ring]
        if not seen:
            return None
        at = np.array([positions[k] for k, _ in seen])
        return ChainField(at, at, np.array([inflation for _, inflation in seen]))

    def fields(
        self, polylines: Sequence[FloatArray], positions: Sequence[FloatArray]
    ) -> list[ObstacleField]:
        """Each member's field with the others at `polylines` and the vias
        at `positions`."""
        return [
            chain_field(
                body.field,
                [
                    self.neighbour(i, self.neighbours[i], polylines),
                    self.via_obstacles(i, positions),
                ],
            )
            for i, body in enumerate(self.bodies)
        ]

    def violations(
        self,
        polylines: Sequence[FloatArray],
        masks: Sequence[BoolArray],
        positions: Sequence[FloatArray],
    ) -> list[int]:
        return [
            violations(points, mask, field)
            for points, mask, field in zip(
                polylines, masks, self.fields(polylines, positions), strict=True
            )
        ]

    def via_view(
        self,
        k: int,
        polylines: Sequence[FloatArray],
        positions: Sequence[FloatArray],
    ) -> _ViaView:
        """What via k sees of the members near it at `polylines` and of the
        other vias at `positions`."""
        via = self.vias[k]
        starts: list[FloatArray] = [np.empty((0, 2))]
        ends: list[FloatArray] = [np.empty((0, 2))]
        inflations: list[FloatArray] = [np.empty(0)]
        member: list[IntArray] = [np.empty(0, dtype=np.intp)]
        vertex: list[IntArray] = [np.empty(0, dtype=np.intp)]
        for j, ring in via.rings.items():
            points = polylines[j]
            starts.append(points[:-1])
            ends.append(points[1:])
            inflations.append(np.full(len(points) - 1, ring))
            member.append(np.full(len(points) - 1, j, dtype=np.intp))
            vertex.append(np.arange(len(points) - 1, dtype=np.intp))
        for u, apart in via.apart.items():
            starts.append(positions[u][None])
            ends.append(positions[u][None])
            inflations.append(np.array([apart]))
            member.append(np.array([-1], dtype=np.intp))
            vertex.append(np.array([u], dtype=np.intp))
        a, b = np.concatenate(starts), np.concatenate(ends)
        return _ViaView(
            ChainField(a, b, np.concatenate(inflations)),
            a,
            b,
            np.concatenate(member),
            np.concatenate(vertex),
        )

    def via_distance(
        self, k: int, polylines: Sequence[FloatArray], positions: Sequence[FloatArray]
    ) -> float:
        """Via k's free distance: to its static copper, the members near it
        and the other vias."""
        at = positions[k]
        return min(
            point_distance(self.vias[k].given.field, at),
            point_distance(self.via_view(k, polylines, positions).field, at),
        )

    def via_distances(
        self, polylines: Sequence[FloatArray], positions: Sequence[FloatArray]
    ) -> list[float]:
        """`via_distance` of every via that moves, in order."""
        return [self.via_distance(k, polylines, positions) for k in self.free_vias]

    def via_energy(
        self, polylines: Sequence[FloatArray], positions: Sequence[FloatArray]
    ) -> tuple[float, list[FloatArray], FloatArray]:
        """The moving vias' own obstacle terms, each via's ring standing for
        a track as long as it is wide, and their gradient at every vertex and
        at every via's centre. Where a member's edge or another via answers a
        via, the gradient also pulls on it."""
        gradients = [np.zeros_like(points) for points in polylines]
        via_gradients = np.zeros((len(self.vias), 2))
        value = 0.0
        for k in self.free_vias:
            at = positions[k][None]
            diameter = self.vias[k].given.via.diameter
            static, static_direction = self.vias[k].given.field.signed_distance(
                at, self.reach
            )
            view = self.via_view(k, polylines, positions)
            near = view.field.nearest(at, self.reach)
            nearer = bool(near.distance[0] < static[0])
            distance = near.distance if nearer else static
            direction = near.gradient[0] if nearer else static_direction[0]
            term, slope = self.integrand(distance)
            value += diameter * float(term[0])
            push = diameter * slope[0] * direction
            via_gradients[k] += push
            if not nearer:
                continue
            e = int(near.edge[0])
            j, first = int(view.member[e]), int(view.vertex[e])
            if j < 0:
                via_gradients[first] -= push
                continue
            t = _parameters(at, view.starts[e : e + 1], view.ends[e : e + 1])[0]
            gradients[j][first] -= (1 - t) * push
            gradients[j][first + 1] -= t * push
        return value, gradients, via_gradients

    def energy_masks(self, masks: Sequence[BoolArray]) -> list[BoolArray]:
        """`masks` with the ends the moving vias carry loose."""
        loose = [mask.copy() for mask in masks]
        for mask, carried in zip(loose, self.carried, strict=True):
            for _, end in carried:
                mask[end] = False
        return loose

    def total(
        self,
        polylines: Sequence[FloatArray],
        positions: Sequence[FloatArray],
        masks: Sequence[BoolArray],
        seen: Sequence[SeenParts] | None = None,
        listed: NeighbourList | None = None,
    ) -> tuple[float, list[FloatArray], FloatArray]:
        """The problem's energy, by default with what each member sees of
        the others as `seen_parts` finds it at `polylines`, and its gradient
        at every vertex and every via's centre; `listed` as for
        `members_energy`."""
        if seen is None:
            seen = seen_parts(self.members, polylines, self.reach, self.rings)
        value, gradients, via_gradients = members_energy(
            self.members,
            polylines,
            self.energy_masks(masks),
            self.params,
            self.settings.short_edge,
            seen,
            positions,
            listed,
        )
        own, pulls, own_gradients = self.via_energy(polylines, positions)
        for g, pull in zip(gradients, pulls, strict=True):
            g += pull
        return value + own, gradients, via_gradients + own_gradients

    def energy_of(
        self, polylines: Sequence[FloatArray], positions: Sequence[FloatArray]
    ) -> float:
        return self.total(polylines, positions, self.masks)[0]

    def length(self) -> float:
        """The summed length of the selected tracks at their current shapes,
        a section's as its two tracks, mm."""
        return sum(
            float(np.linalg.norm(np.diff(track, axis=0), axis=1).sum())
            for member, points in zip(self.members, self.points, strict=True)
            for _, track in shown(member, points)
        )

    def result(self, k: int, *, best: bool) -> ChainResult:
        """Chain k's best state, with `best`, else its input."""
        return ChainResult(
            polyline=self.best[k] if best else self.originals[k],
            fixed=self.best_masks[k] if best else self.anchors[k],
            rejected_edges=self.rejected_edges[k],
            frozen=self.frozen[k],
            before=self.before[k],
        )

    def pair_result(self, k: int, *, best: bool) -> PairResult:
        """Member k's, a section's, best state, with `best`, else its input."""
        block = self.members[k]
        assert isinstance(block, PairBlock)
        given = block.given
        moved = best and not block.held
        points = self.best[k] if best else self.originals[k]
        mask = self.best_masks[k] if best else self.anchors[k]
        a, b = block.offset(points) if best else (given.a.points, given.b.points)
        return PairResult(
            block=block,
            centreline=points,
            fixed=mask & ~block.direction_pins(points) if moved else mask,
            a=a,
            b=b,
            rejected_edges=self.rejected_edges[k],
            too_close=self.too_close[k],
            frozen=self.frozen[k],
            before=self.before[k],
            bend=block.bend(points) if moved else block.tight_start,
        )

    def via_result(self, k: int, *, best: bool) -> ViaResult:
        """Via k's best centre, with `best`, else its input."""
        x, y = self.best_positions[k] if best else self.start_positions[k]
        return ViaResult(
            (float(x), float(y)), self.vias[k].frozen, self.via_rejected[k]
        )

    def step(self) -> float:
        """One burst of every member and via together; returns its largest
        change of a shape or a via's centre."""
        settings = self.settings
        # The change counts from the shapes as given, so that resampling the
        # inputs does not count as moving them.
        before = [
            original if resampled else points
            for original, points, resampled in zip(
                self.originals, self.points, self.resampled, strict=True
            )
        ]
        starts: list[FloatArray] = []
        masks: list[BoolArray] = []
        pressed = self.pressed(self.points, self.positions)
        for i, (points, mask) in enumerate(zip(self.points, self.masks, strict=True)):
            if i in self.free and not self.resampled[i]:
                points, mask = _resample(
                    points, mask, self.loose[i], settings.spacing, pressed[i]
                )
            starts.append(points)
            masks.append(mask)
        self.masks = masks
        self.resampled = [False] * len(starts)
        burst = self.burst(starts)
        change = max(
            [
                *(shape_change(before[i], burst.polylines[i]) for i in self.free),
                *(
                    float(np.linalg.norm(burst.positions[k] - self.positions[k]))
                    for k in self.free_vias
                ),
            ]
        )
        self.moved |= change > NOTHING_MOVED
        self.points, self.positions = burst.polylines, burst.positions
        for i, (count, close) in enumerate(
            zip(burst.rejected_edges, burst.too_close, strict=True)
        ):
            self.rejected_edges[i] += count
            self.too_close[i] += close
        for k, count in enumerate(burst.via_rejected):
            self.via_rejected[k] += count
        self.energy = self.energy_of(self.points, self.positions)
        if self.energy <= self.best_energy and self.guard(
            self.points, self.masks, self.positions
        ):
            self.best, self.best_masks = list(self.points), list(self.masks)
            self.best_positions = list(self.positions)
            self.best_energy = self.energy
        self.lowest.append(min(self.lowest[-1], self.energy))
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
        # of them, not because the shapes settled; but a section pressed
        # onto its own parts beyond a turn has moves to undo in every burst,
        # which no energy term pushes back.
        cut_short = any(burst.via_rejected) or any(
            count > close
            for count, close in zip(burst.rejected_edges, burst.too_close, strict=True)
        )
        self.settled = self.abnormal_streak >= MAX_ABNORMAL_BURSTS or (
            not burst.abnormal
            and (
                (not cut_short and change < settings.settled_change)
                or stalled(self.lowest, settings)
            )
        )
        return change

    def pressed(
        self, polylines: Sequence[FloatArray], positions: Sequence[FloatArray]
    ) -> list[BoolArray]:
        """Per member, its vertices at `polylines` that the other members and
        vias at `positions`, but its siblings, hold still: so deep inside
        their clearance that the vertex's box is empty (`half_sides`).
        Resampling keeps them too: spread along a shape that the free
        vertices beside them bend, they would creep out of the clearance a
        little every burst, and the run with them, never settling. Vertices
        deep inside static copper are left to resampling, which lays them
        there where a chord cuts a corner and moves them out again."""
        seen = seen_parts(self.members, polylines, self.reach, self.rings)
        flat = _shown_flat(self.members, polylines, positions)
        return [
            CROSS_SHARE
            * _others(_subset(part, keep), flat).field.signed_distance(points)[0]
            < -self.settings.bubble_tolerance
            for points, part, keep in zip(
                polylines, seen, self.apart(seen, polylines), strict=True
            )
        ]

    def layout(self, starts: Sequence[FloatArray]) -> _Layout:
        """The variables of a burst from `starts` and the current centres."""
        index, count = _variables(self.masks, self.groups)
        offsets = [np.zeros_like(points) for points in starts]
        via_index = np.full(len(self.vias), -1, dtype=np.intp)
        for k in self.free_vias:
            via_index[k] = count
            for i, end, offset in self.vias[k].carried:
                index[i][end] = count
                offsets[i][end] = offset
            count += 1
        return _Layout(
            list(starts), list(self.positions), index, offsets, via_index, count
        )

    def objective(
        self,
        layout: _Layout,
        seen: Sequence[SeenParts] | None = None,
        listed: bool = True,
    ) -> Callable[[FloatArray], tuple[float, FloatArray]]:
        """The problem's energy as a function of `layout`'s variables, with
        what each member sees of the others measured at the layout's start
        unless given, through a `NeighbourList` unless not `listed`."""
        if seen is None:
            seen = seen_parts(self.members, layout.starts, self.reach, self.rings)
        masks = self.masks
        neighbours = NeighbourList(seen, self.reach, LIST_SKIN) if listed else None

        def objective(x: FloatArray) -> tuple[float, FloatArray]:
            polylines, positions = layout.unflatten(x)
            value, gradients, via_gradients = self.total(
                polylines, positions, masks, seen, neighbours
            )
            return value, layout.gradient(gradients, via_gradients)

        return objective

    def burst(self, starts: list[FloatArray]) -> _JointBurst:
        """One bounded L-BFGS-B run over every free vertex and moving via
        of the problem, retried with halved bounds where edges newly crossed
        static copper or another member's new shape, a section newly bent
        below its limit or brought its parts beyond a turn too close
        (`optimize.SelfCheck`) or a via ended newly inside its clearance."""
        members = self.members
        positions = self.positions
        layout = self.layout(starts)
        index, count = layout.index, layout.count
        free = [i >= 0 for i in index]
        seen = seen_parts(members, starts, self.reach, self.rings)
        objective = self.objective(layout, seen)
        x0 = layout.x0
        flat_start = _shown_flat(members, starts, positions)
        static_before = [
            body.field.segments_clear(s[:-1], s[1:])
            for body, s in zip(self.bodies, starts, strict=True)
        ]
        others_before = [
            _others(seen[i], flat_start).field.segments_clear(s[:-1], s[1:])
            for i, s in enumerate(starts)
        ]
        checks = [
            SelfCheck.of(s, member.limit, member.apart)
            if isinstance(member, PairBlock)
            else None
            for member, s in zip(members, starts, strict=True)
        ]
        clear_before = {
            k: distance >= -INSIDE_TOLERANCE
            for k, distance in zip(
                self.free_vias, self.via_distances(starts, positions), strict=True
            )
        }

        def crossings(x: FloatArray) -> tuple[_Round, BoolArray]:
            """What `_Round` counts at `x`, and the coordinates of the
            variables to move back."""
            polylines, centres = layout.unflatten(x)
            flat = _shown_flat(members, polylines, centres)
            back = np.zeros(count, dtype=bool)
            counts: list[tuple[int, int]] = []
            via_counts = [0] * len(self.vias)
            for i, points in enumerate(polylines):
                a, b = points[:-1], points[1:]
                others = _others(seen[i], flat).field
                crossing = (
                    static_before[i] & ~self.bodies[i].field.segments_clear(a, b)
                ) | (others_before[i] & ~others.segments_clear(a, b))
                ends = np.zeros(len(points), dtype=bool)
                ends[:-1] |= crossing
                ends[1:] |= crossing
                for k, end in self.carried[i]:
                    via_counts[k] += int(ends[end])
                rejected, close = int(crossing.sum()), 0
                check = checks[i]
                if check is not None:
                    own, more, close = check(points)
                    ends |= own
                    rejected += more
                counts.append((rejected, close))
                back[index[i][ends & free[i]]] = True
            for k in self.free_vias:
                if (
                    clear_before[k]
                    and self.via_distance(k, polylines, centres) < -INSIDE_TOLERANCE
                ):
                    back[layout.via_index[k]] = True
                    via_counts[k] += 1
            return _Round(counts, via_counts), np.repeat(back, 2)

        rounds: list[_Round] = []

        def revert(x: FloatArray) -> tuple[BoolArray, int]:
            found: list[_Round] = []

            def check(candidate: FloatArray) -> tuple[int, BoolArray]:
                found_now, back = crossings(candidate)
                if not found:
                    found.append(found_now)
                return sum(r for r, _ in found_now.members) + sum(found_now.vias), back

            reverted, total = revert_until_clear(x, x0, check)
            rounds.append(found[0])
            return reverted, total

        sides, bubbles = self.half_sides(
            layout, seen, others_before, list(clear_before.values())
        )
        result = bounded_burst(
            objective,
            x0,
            np.repeat(sides, 2),
            revert,
            self.settings,
            np.repeat(bubbles / RETRY_SHARE, 2),
        )
        # An abnormally ended burst stops at its last check without counting it.
        counted = rounds[:-1] if result.abnormal else rounds
        polylines, centres = layout.unflatten(result.points)
        return _JointBurst(
            polylines,
            centres,
            [sum(r.members[i][0] for r in counted) for i in range(len(starts))],
            [sum(r.members[i][1] for r in counted) for i in range(len(starts))],
            [sum(r.vias[k] for r in counted) for k in range(len(self.vias))],
            result.abnormal,
        )

    def half_sides(
        self,
        layout: _Layout,
        seen: Sequence[SeenParts],
        edges_clear: Sequence[BoolArray],
        vias_clear: Sequence[bool],
    ) -> tuple[FloatArray, FloatArray]:
        """Per variable, the half side of its box: that of the smallest, over
        the vertices and via it moves, of their free distance to static
        copper and of the larger of CROSS_SHARE of their free distance to
        the other members and vias as each sees them, less the other
        branches of its chain's junctions and the chains its vias carry
        (`_Problem.siblings`), and the step floor (`step_floor`). Only a
        variable whose vertices' edges (`edges_clear`, per member) and via
        (`vias_clear`, per moving via) all start clear of the others takes
        the floor. Also the half side of its bubble alone, without the
        floor, by which a burst's retries shrink its box no further than
        they always did. Both are infinite without `move_bounds`."""
        if not self.settings.move_bounds:
            unbounded = np.full(layout.count, np.inf)
            return unbounded, unbounded
        starts, positions = layout.starts, layout.positions
        flat = _shown_flat(self.members, starts, positions)
        apart = self.apart(seen, starts)
        static = np.full(layout.count, np.inf)
        cross = np.full(layout.count, np.inf)
        clear = np.ones(layout.count, dtype=bool)
        for i, (body, points) in enumerate(zip(self.bodies, starts, strict=True)):
            free = layout.index[i] >= 0
            index = layout.index[i][free]
            np.minimum.at(static, index, body.field.signed_distance(points[free])[0])
            others = _others(_subset(seen[i], apart[i]), flat).field
            np.minimum.at(cross, index, others.signed_distance(points[free])[0])
            ends = np.ones(len(points), dtype=bool)
            ends[:-1] &= edges_clear[i]
            ends[1:] &= edges_clear[i]
            np.logical_and.at(clear, index, ends[free])
        for k, via_clear in zip(self.free_vias, vias_clear, strict=True):
            at = positions[k]
            v = layout.via_index[k]
            static[v] = min(static[v], point_distance(self.vias[k].given.field, at))
            view = self.via_view(k, starts, positions)
            cross[v] = min(cross[v], point_distance(view.field, at))
            clear[v] &= via_clear
        floor = np.where(clear, self.step_floor(seen, apart) / math.sqrt(2), 0.0)
        static_side = half_side(static, self.settings)
        cross_side = half_side(CROSS_SHARE * cross, self.settings)
        return (
            np.minimum(static_side, np.maximum(cross_side, floor)),
            np.minimum(static_side, cross_side),
        )

    def apart(
        self, seen: Sequence[SeenParts], polylines: Sequence[FloatArray]
    ) -> list[BoolArray]:
        """Per member, which pieces of `seen`, made at `polylines`, are not
        its siblings'."""
        owner = np.repeat(
            np.arange(len(self.members) + len(self.vias)),
            [
                *(
                    sum(len(points) for _, points in shown(member, line))
                    for member, line in zip(self.members, polylines, strict=True)
                ),
                *(2 for _ in self.vias),
            ],
        )
        return [
            ~np.isin(owner[part.vertex], siblings)
            for part, siblings in zip(seen, self.siblings, strict=True)
        ]

    def step_floor(
        self, seen: Sequence[SeenParts], apart: Sequence[BoolArray]
    ) -> float:
        """How far, mm, every variable may move in a burst however close the
        other members and vias are: STEP_FLOOR of the smallest inflation
        between any two of them but siblings, less δ."""
        inflations = [
            *(
                float(part.inflation[keep].min(initial=math.inf))
                for part, keep in zip(seen, apart, strict=True)
            ),
            *(gap for via in self.vias for gap in via.apart.values()),
        ]
        smallest = min(inflations, default=math.inf)
        if not math.isfinite(smallest):
            return 0.0
        return max(STEP_FLOOR * (smallest - self.settings.bubble_tolerance), 0.0)


def _description(layers: Sequence[str], vias: bool) -> str:
    """How log lines and results name a problem of `layers`, with moving
    `vias` or without."""
    named = (
        layers[0] if len(layers) == 1 else f"{', '.join(layers[:-1])} and {layers[-1]}"
    )
    return f"the chains and vias on {named}" if vias else f"the chains on {named}"
